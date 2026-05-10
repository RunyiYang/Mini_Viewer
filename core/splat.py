"""Gaussian splat data loading.

Supported scene inputs:
- Inria/3DGS style .ply files.
- NumPy folders with coord/quat/scale/opacity/color arrays.
- Optional aligned feature .npy/.pt/.pth files aligned to splats.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

try:
    from plyfile import PlyData
except Exception:  # pragma: no cover - optional until PLY loading is used.
    PlyData = None  # type: ignore[assignment]

from utils.color_shs import SH2RGB

SKIP_FEATURE_NAMES = {"", "none", "null", "language_feature_dummy", "dummy"}
_EPS = 1e-9
_CHUNK_SIZE = 256
_SH_C0 = 0.28209479177387814
_UINT11_MASK = (1 << 11) - 1
_UINT10_MASK = (1 << 10) - 1
_UINT8_MASK = (1 << 8) - 1


SCENE_ARRAY_ALIASES: dict[str, tuple[str, ...]] = {
    "means": ("coord", "coords", "mean", "means", "xyz", "points", "positions"),
    "normals": ("normal", "normals", "norm", "norms"),
    "quats": ("quat", "quats", "rotation", "rotations", "rot"),
    "scales": ("scale", "scales", "scaling"),
    "opacities": ("opacity", "opacities", "alpha", "alphas"),
    "colors": ("color", "colors", "rgb", "rgba", "features_dc", "f_dc"),
}

FEATURE_KEYS = (
    "language_feature",
    "language_features",
    "features",
    "feature",
    "embeddings",
    "embedding",
    "clip",
    "clip_features",
    "siglip",
    "siglip_features",
    "dino",
    "dinov2",
    "dino_feature",
    "dino_features",
    "dinov2_feature",
    "dinov2_features",
    "image_feature",
    "image_features",
    "visual_feature",
    "visual_features",
)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _extract_tensor_from_loaded(value: Any) -> np.ndarray:
    """Pull an (N, D) tensor/array out of common .pt/.pth structures."""
    if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
        return _to_numpy(value)
    if isinstance(value, dict):
        for key in FEATURE_KEYS:
            if key in value:
                return _extract_tensor_from_loaded(value[key])
        # Fallback: first tensor-like value with at least 2 dimensions.
        for item in value.values():
            try:
                arr = _extract_tensor_from_loaded(item)
            except Exception:
                continue
            if arr.ndim >= 2:
                return arr
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                arr = _extract_tensor_from_loaded(item)
            except Exception:
                continue
            if arr.ndim >= 2 or arr.shape[-1] in {3, 512, 768, 1024, 1152}:
                return arr
    raise ValueError("Could not find a feature tensor/array in the loaded object.")


def load_feature_array(path: str | Path) -> np.ndarray:
    """Load a .npy/.npz/.pt/.pth feature array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        return _extract_tensor_from_loaded(arr)
    if suffix == ".npz":
        obj = np.load(path, allow_pickle=True)
        for key in FEATURE_KEYS:
            if key in obj:
                return _extract_tensor_from_loaded(obj[key])
        if obj.files:
            return _extract_tensor_from_loaded(obj[obj.files[0]])
        raise ValueError(f"Empty npz file: {path}")
    if suffix in {".pt", ".pth"}:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        return _extract_tensor_from_loaded(obj)
    raise ValueError(f"Unsupported feature extension '{path.suffix}' for {path}")


def resolve_feature_path(scene_path: Path | None, language_feature: Path | None) -> Path | None:
    if language_feature is None:
        return None
    raw = str(language_feature)
    if raw.strip().lower() in SKIP_FEATURE_NAMES:
        return None
    path = Path(language_feature)
    if path.exists():
        return path
    if scene_path is not None and scene_path.is_dir():
        candidate = scene_path / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Feature file not found: {language_feature}")


def _resolve_index_sidecar(feature_path: Path) -> Path | None:
    candidate = feature_path.with_name(f"{feature_path.stem}_index.npy")
    return candidate if candidate.exists() else None


def _load_index_sidecar(feature_path: Path) -> np.ndarray | None:
    index_path = _resolve_index_sidecar(feature_path)
    if index_path is None:
        return None
    index = np.load(index_path)
    if index.ndim != 1:
        raise ValueError(f"Expected 1D feature index sidecar, got shape {index.shape} from {index_path}.")
    if not np.issubdtype(index.dtype, np.integer):
        raise TypeError(f"Expected integer feature indices in {index_path}, got {index.dtype}.")
    index = index.astype(np.int64, copy=False)
    if index.size == 0:
        raise ValueError(f"Feature index sidecar is empty: {index_path}.")
    if np.unique(index).size != index.size:
        raise ValueError(f"Feature index sidecar contains duplicate source rows: {index_path}.")
    return index


def _find_array_file(folder: Path, names: Iterable[str]) -> Path | None:
    for stem in names:
        for suffix in (".npy", ".npz", ".pt", ".pth"):
            candidate = folder / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    # Last resort: exact case-insensitive stem match.
    stems = {name.lower() for name in names}
    for candidate in folder.iterdir():
        if candidate.is_file() and candidate.stem.lower() in stems:
            return candidate
    return None


def _load_array_file(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".pt", ".pth"}:
        return load_feature_array(path)
    if path.suffix.lower() == ".npz":
        obj = np.load(path, allow_pickle=True)
        if obj.files:
            return _to_numpy(obj[obj.files[0]])
        raise ValueError(f"Empty npz file: {path}")
    return _to_numpy(np.load(path, allow_pickle=True))


def _sigmoid_if_logits(values: torch.Tensor) -> torch.Tensor:
    values = values.float().flatten()
    if torch.any(values < 0.0) or torch.any(values > 1.0):
        return torch.sigmoid(values)
    return values.clamp(0.0, 1.0)


def _normalize_quats(quats: torch.Tensor) -> torch.Tensor:
    quats = quats.float()
    if quats.ndim == 1:
        quats = quats[None]
    if quats.shape[-1] != 4:
        raise ValueError(f"Expected quaternions with shape [N,4], got {tuple(quats.shape)}")
    quats = F.normalize(quats, dim=-1, eps=1e-6)
    # Avoid zero/NaN quaternions after corrupted inputs.
    bad = ~torch.isfinite(quats).all(dim=-1)
    if torch.any(bad):
        quats[bad] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=quats.device)
    return quats


def _normalize_colors(colors: torch.Tensor) -> torch.Tensor:
    colors = colors.float()
    if colors.ndim == 3 and colors.shape[1] == 1:
        colors = colors[:, 0]
    if colors.ndim == 3 and colors.shape[-1] == 1 and colors.shape[1] == 3:
        colors = colors[..., 0]
    if colors.shape[-1] > 3:
        colors = colors[..., :3]
    if colors.max() > 1.5:
        colors = colors / 255.0
    return colors.clamp(0.0, 1.0)


def _set_pca_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_pca_device(requested: str, active_device: torch.device) -> torch.device:
    requested = str(requested or "auto").strip().lower()
    if requested == "auto":
        if active_device.type == "cuda" and torch.cuda.is_available():
            return active_device
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--pca-device cuda was requested, but torch.cuda.is_available() is false.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported PCA device '{requested}'. Use auto, cpu, or cuda.")
    return torch.device(requested)


def _pca_normalize_torch(values: torch.Tensor, brightness: float) -> torch.Tensor:
    min_val = values.amin(dim=0, keepdim=True)
    max_val = values.amax(dim=0, keepdim=True)
    color = (values - min_val) / torch.clamp(max_val - min_val, min=1e-6)
    return (color * float(brightness)).clamp_(0.0, 1.0)


def _feature_preview_torch(
    features: torch.Tensor,
    *,
    device: torch.device,
    brightness: float,
    seed: int,
) -> torch.Tensor:
    _set_pca_seed(seed)
    q = min(6, features.shape[0], features.shape[1])
    if q < 3:
        raise ValueError(f"Need at least 3 PCA components, got q={q} for shape={tuple(features.shape)}")
    feat = features.detach().float().to(device)
    with torch.no_grad():
        _, _, v = torch.pca_lowrank(feat, center=True, q=q, niter=5)
        projection = feat @ v
        if projection.shape[1] >= 6:
            preview = projection[:, :3] * 0.7 + projection[:, 3:6] * 0.3
        else:
            preview = projection[:, :3]
        return _pca_normalize_torch(preview, brightness).cpu()


def _feature_preview_sklearn(
    features: torch.Tensor,
    *,
    brightness: float,
    seed: int,
    batch_size: int = 500_000,
) -> torch.Tensor:
    from sklearn.decomposition import IncrementalPCA, PCA
    from sklearn.preprocessing import StandardScaler

    _set_pca_seed(seed)
    feat = features.detach().float().cpu().numpy()
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat)
    if feat_scaled.shape[0] > 100_000:
        pca = IncrementalPCA(n_components=3, batch_size=batch_size)
        pca.fit(feat_scaled)
        preview = pca.transform(feat_scaled).astype(np.float32)
    else:
        pca = PCA(n_components=3, random_state=seed)
        preview = pca.fit_transform(feat_scaled).astype(np.float32)
    preview_t = torch.from_numpy(preview).float()
    return _pca_normalize_torch(preview_t, brightness).cpu()


def _feature_preview(
    features: torch.Tensor,
    *,
    method: str,
    pca_device: torch.device,
    brightness: float,
    seed: int,
) -> torch.Tensor:
    """Create a 3-channel preview using Chorus-style PCA colorization."""
    features = features.detach().float()
    if features.ndim == 1:
        features = features[:, None]
    if features.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.float32)
    method = str(method or "torch").strip().lower()
    if method == "torch":
        try:
            preview = _feature_preview_torch(
                features,
                device=pca_device,
                brightness=brightness,
                seed=seed,
            )
            print(f"[feature] PCA preview: method=torch, device={pca_device}, seed={seed}.")
            return preview
        except Exception as exc:
            if pca_device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError(
                f"Torch PCA preview failed on {pca_device}: {exc}. "
                "Retry with --pca-method sklearn or --pca-device cpu."
            ) from exc
    if method == "sklearn":
        preview = _feature_preview_sklearn(features, brightness=brightness, seed=seed)
        print(f"[feature] PCA preview: method=sklearn, device=cpu, seed={seed}.")
        return preview
    raise ValueError(f"Unsupported PCA method '{method}'. Use torch or sklearn.")


def _structured_columns(vertex: Any, names: list[str]) -> np.ndarray:
    return np.stack([np.asarray(vertex[name]) for name in names], axis=-1)


def _has_props(vertex: Any, names: list[str]) -> bool:
    return all(name in vertex.data.dtype.names for name in names)


def _normalize_quat_np(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float32, copy=False)
    quat = quat / (np.linalg.norm(quat, axis=1, keepdims=True) + _EPS)
    sign = np.sign(quat[:, 0])
    sign[sign == 0] = 1
    return quat * sign[:, None]


def _read_compressed_gaussian_ply(ply: Any) -> dict[str, torch.Tensor]:
    chunk = ply["chunk"].data
    vertex = ply["vertex"].data
    num = vertex.shape[0]

    chunk_indices = np.arange(num, dtype=np.int64) // _CHUNK_SIZE
    chunk_indices = np.minimum(chunk_indices, len(chunk) - 1)

    min_x = chunk["min_x"][chunk_indices]
    min_y = chunk["min_y"][chunk_indices]
    min_z = chunk["min_z"][chunk_indices]
    max_x = chunk["max_x"][chunk_indices]
    max_y = chunk["max_y"][chunk_indices]
    max_z = chunk["max_z"][chunk_indices]

    min_scale_x = chunk["min_scale_x"][chunk_indices]
    min_scale_y = chunk["min_scale_y"][chunk_indices]
    min_scale_z = chunk["min_scale_z"][chunk_indices]
    max_scale_x = chunk["max_scale_x"][chunk_indices]
    max_scale_y = chunk["max_scale_y"][chunk_indices]
    max_scale_z = chunk["max_scale_z"][chunk_indices]

    min_r = chunk["min_r"][chunk_indices]
    min_g = chunk["min_g"][chunk_indices]
    min_b = chunk["min_b"][chunk_indices]
    max_r = chunk["max_r"][chunk_indices]
    max_g = chunk["max_g"][chunk_indices]
    max_b = chunk["max_b"][chunk_indices]

    packed_position = vertex["packed_position"].astype(np.uint32)
    packed_scale = vertex["packed_scale"].astype(np.uint32)
    packed_rotation = vertex["packed_rotation"].astype(np.uint32)
    packed_color = vertex["packed_color"].astype(np.uint32)

    px = ((packed_position >> 21) & _UINT11_MASK).astype(np.float32) / _UINT11_MASK
    py = ((packed_position >> 11) & _UINT10_MASK).astype(np.float32) / _UINT10_MASK
    pz = (packed_position & _UINT11_MASK).astype(np.float32) / _UINT11_MASK

    coord = np.empty((num, 3), dtype=np.float32)
    coord[:, 0] = min_x * (1.0 - px) + max_x * px
    coord[:, 1] = min_y * (1.0 - py) + max_y * py
    coord[:, 2] = min_z * (1.0 - pz) + max_z * pz

    sx = ((packed_scale >> 21) & _UINT11_MASK).astype(np.float32) / _UINT11_MASK
    sy = ((packed_scale >> 11) & _UINT10_MASK).astype(np.float32) / _UINT10_MASK
    sz = (packed_scale & _UINT11_MASK).astype(np.float32) / _UINT11_MASK

    scale_log = np.empty((num, 3), dtype=np.float32)
    scale_log[:, 0] = min_scale_x * (1.0 - sx) + max_scale_x * sx
    scale_log[:, 1] = min_scale_y * (1.0 - sy) + max_scale_y * sy
    scale_log[:, 2] = min_scale_z * (1.0 - sz) + max_scale_z * sz
    scale = np.exp(scale_log)

    norm = np.float32(1.0 / (np.sqrt(2.0) * 0.5))
    a = ((packed_rotation >> 20) & _UINT10_MASK).astype(np.float32) / _UINT10_MASK
    b = ((packed_rotation >> 10) & _UINT10_MASK).astype(np.float32) / _UINT10_MASK
    c = (packed_rotation & _UINT10_MASK).astype(np.float32) / _UINT10_MASK
    a = (a - 0.5) * norm
    b = (b - 0.5) * norm
    c = (c - 0.5) * norm
    m = np.sqrt(np.maximum(0.0, 1.0 - (a * a + b * b + c * c)))
    which = packed_rotation >> 30

    quat = np.empty((num, 4), dtype=np.float32)
    mask = which == 0
    quat[mask, 0] = m[mask]
    quat[mask, 1] = a[mask]
    quat[mask, 2] = b[mask]
    quat[mask, 3] = c[mask]
    mask = which == 1
    quat[mask, 0] = a[mask]
    quat[mask, 1] = m[mask]
    quat[mask, 2] = b[mask]
    quat[mask, 3] = c[mask]
    mask = which == 2
    quat[mask, 0] = a[mask]
    quat[mask, 1] = b[mask]
    quat[mask, 2] = m[mask]
    quat[mask, 3] = c[mask]
    mask = which == 3
    quat[mask, 0] = a[mask]
    quat[mask, 1] = b[mask]
    quat[mask, 2] = c[mask]
    quat[mask, 3] = m[mask]
    quat = _normalize_quat_np(quat)

    cr = ((packed_color >> 24) & _UINT8_MASK).astype(np.float32) / _UINT8_MASK
    cg = ((packed_color >> 16) & _UINT8_MASK).astype(np.float32) / _UINT8_MASK
    cb = ((packed_color >> 8) & _UINT8_MASK).astype(np.float32) / _UINT8_MASK
    cw = (packed_color & _UINT8_MASK).astype(np.float32) / _UINT8_MASK

    r = min_r * (1.0 - cr) + max_r * cr
    g = min_g * (1.0 - cg) + max_g * cg
    b_val = min_b * (1.0 - cb) + max_b * cb
    fdc = np.stack(
        [(r - 0.5) / _SH_C0, (g - 0.5) / _SH_C0, (b_val - 0.5) / _SH_C0],
        axis=-1,
    )
    colors = np.clip(fdc * _SH_C0 + 0.5, 0.0, 1.0).astype(np.float32)
    opacities = np.clip(cw, 0.0, 1.0).astype(np.float32)

    return {
        "means": torch.from_numpy(coord).float(),
        "normals": torch.zeros((num, 3), dtype=torch.float32),
        "quats": _normalize_quats(torch.from_numpy(quat).float()),
        "scales": torch.from_numpy(scale).float().clamp_min(1e-8),
        "opacities": torch.from_numpy(opacities).float(),
        "colors": _normalize_colors(torch.from_numpy(colors).float()),
    }


def _align_features_by_source_index(
    *,
    raw: np.ndarray,
    feature_index: np.ndarray,
    final_ids: np.ndarray,
    feature_path: Path,
) -> np.ndarray:
    if raw.shape[0] != feature_index.shape[0]:
        raise ValueError(
            f"Feature rows ({raw.shape[0]}) do not match sidecar rows "
            f"({feature_index.shape[0]}) for {feature_path}."
        )

    order = np.argsort(feature_index)
    sorted_index = feature_index[order]
    positions = np.searchsorted(sorted_index, final_ids)
    in_bounds = positions < sorted_index.shape[0]
    matched = np.zeros(final_ids.shape[0], dtype=bool)
    matched[in_bounds] = sorted_index[positions[in_bounds]] == final_ids[in_bounds]
    if not np.all(matched):
        missing = final_ids[~matched][:10].tolist()
        raise ValueError(
            f"Feature index sidecar for {feature_path} does not cover all loaded splats. "
            f"First missing source rows: {missing}"
        )
    return raw[order[positions]]


class SplatData:
    """Container around loaded splat tensors and language feature tensors."""

    def __init__(self, args: Any, device: str = "cuda") -> None:
        self.args = args
        self.device = torch.device(device)
        self.scene_path: Path | None = getattr(args, "ply", None) or getattr(args, "folder_npy", None)
        self._data: dict[str, torch.Tensor] = {}
        self._data_cpu: dict[str, torch.Tensor] = {}
        self._language_feature_large: torch.Tensor | None = None
        self._language_feature_large_cpu: torch.Tensor | None = None
        self._feature_path: Path | None = None
        self._feature_index: np.ndarray | None = None
        self._original_count = 0
        self._final_original_indices: np.ndarray | None = None
        self._load()

    def __len__(self) -> int:
        return int(self._data["means"].shape[0])

    def _load(self) -> None:
        if getattr(self.args, "ply", None) is not None:
            tensors, original_indices = self._load_ply(Path(self.args.ply))
        elif getattr(self.args, "folder_npy", None) is not None:
            tensors, original_indices = self._load_npy_folder(Path(self.args.folder_npy))
        else:
            raise ValueError("Either --ply or --folder-npy must be provided.")

        self._original_count = int(len(original_indices))
        original_n_before_masks = int(tensors["means"].shape[0])

        self._feature_path = resolve_feature_path(
            Path(self.scene_path) if self.scene_path else None,
            getattr(self.args, "language_feature", None),
        )
        if self._feature_path is not None:
            self._feature_index = _load_index_sidecar(self._feature_path)

        valid_mask = None
        if self._feature_index is not None:
            feature_index = self._feature_index
            if np.any(feature_index < 0) or np.any(feature_index >= original_n_before_masks):
                raise ValueError(
                    f"Feature index sidecar for {self._feature_path} contains source rows outside "
                    f"[0, {original_n_before_masks})."
                )
            tensors = {k: v[feature_index] for k, v in tensors.items()}
            original_indices = original_indices[feature_index]
            print(f"[feature] Applied feature index sidecar with {feature_index.shape[0]:,} splats.")
        else:
            valid_mask = self._load_valid_feature_mask()
        if valid_mask is not None:
            if len(valid_mask) != original_n_before_masks:
                print(
                    f"[splat] Ignoring valid_feat_mask.npy: length {len(valid_mask)} != splat count {original_n_before_masks}."
                )
            else:
                tensors = {k: v[valid_mask] for k, v in tensors.items()}
                original_indices = original_indices[valid_mask]

        max_splats = getattr(self.args, "max_splats", None)
        if max_splats is not None and max_splats > 0 and len(original_indices) > max_splats:
            keep = np.linspace(0, len(original_indices) - 1, max_splats).astype(np.int64)
            tensors = {k: v[keep] for k, v in tensors.items()}
            original_indices = original_indices[keep]
            print(f"[splat] Downsampled to {max_splats:,} splats for interactivity.")

        self._final_original_indices = original_indices
        # Keep a CPU master copy so CUDA rerender failures can really fall back
        # to CPU without first touching CUDA tensors.
        self._data_cpu = {key: value.detach().cpu().contiguous() for key, value in tensors.items()}
        self._data = {key: value.to(self.device) for key, value in self._data_cpu.items()}
        self._load_aligned_features(original_n_before_masks, valid_mask)

    def _load_valid_feature_mask(self) -> np.ndarray | None:
        folder = getattr(self.args, "folder_npy", None)
        if folder is None:
            return None
        path = Path(folder) / "valid_feat_mask.npy"
        if not path.exists():
            return None
        return np.load(path).astype(bool)

    def _load_ply(self, path: Path) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        if PlyData is None:
            raise ImportError("plyfile is required for --ply. Install plyfile>=1.0.")
        if not path.exists():
            raise FileNotFoundError(path)
        ply = PlyData.read(path)
        vertex = ply["vertex"]
        names = set(vertex.data.dtype.names)
        n = len(vertex)

        if "packed_position" in names and "chunk" in {element.name for element in ply.elements}:
            tensors = _read_compressed_gaussian_ply(ply)
            print(f"[splat] Loaded compressed PLY {path} ({n:,} splats).")
            return tensors, np.arange(n, dtype=np.int64)

        required = ["x", "y", "z"]
        if not all(name in names for name in required):
            raise ValueError(f"PLY {path} is missing one of {required}.")
        means = torch.from_numpy(_structured_columns(vertex, required)).float()

        if _has_props(vertex, ["nx", "ny", "nz"]):
            normals = torch.from_numpy(_structured_columns(vertex, ["nx", "ny", "nz"])).float()
        else:
            normals = torch.zeros((n, 3), dtype=torch.float32)

        if _has_props(vertex, ["rot_0", "rot_1", "rot_2", "rot_3"]):
            quats = torch.from_numpy(_structured_columns(vertex, ["rot_0", "rot_1", "rot_2", "rot_3"])).float()
        elif _has_props(vertex, ["qw", "qx", "qy", "qz"]):
            quats = torch.from_numpy(_structured_columns(vertex, ["qw", "qx", "qy", "qz"])).float()
        else:
            quats = torch.zeros((n, 4), dtype=torch.float32)
            quats[:, 0] = 1.0
        quats = _normalize_quats(quats)

        if _has_props(vertex, ["scale_0", "scale_1", "scale_2"]):
            scales = torch.from_numpy(_structured_columns(vertex, ["scale_0", "scale_1", "scale_2"])).float().exp()
        elif _has_props(vertex, ["sx", "sy", "sz"]):
            scales = torch.from_numpy(_structured_columns(vertex, ["sx", "sy", "sz"])).float()
        else:
            scales = torch.full((n, 3), 0.01, dtype=torch.float32)
        scales = scales.clamp_min(1e-8)

        if "opacity" in names:
            opacities = _sigmoid_if_logits(torch.from_numpy(np.asarray(vertex["opacity"])).float())
        elif "alpha" in names:
            opacities = _sigmoid_if_logits(torch.from_numpy(np.asarray(vertex["alpha"])).float())
        else:
            opacities = torch.ones(n, dtype=torch.float32)

        if _has_props(vertex, ["f_dc_0", "f_dc_1", "f_dc_2"]):
            sh0 = torch.from_numpy(_structured_columns(vertex, ["f_dc_0", "f_dc_1", "f_dc_2"])).float()
            colors = SH2RGB(sh0)
        elif _has_props(vertex, ["red", "green", "blue"]):
            colors = torch.from_numpy(_structured_columns(vertex, ["red", "green", "blue"])).float() / 255.0
        elif _has_props(vertex, ["r", "g", "b"]):
            colors = torch.from_numpy(_structured_columns(vertex, ["r", "g", "b"])).float()
            colors = _normalize_colors(colors)
        else:
            colors = torch.full((n, 3), 0.6, dtype=torch.float32)

        tensors = {
            "means": means,
            "normals": F.normalize(normals, dim=-1, eps=1e-6),
            "quats": quats,
            "scales": scales,
            "opacities": opacities,
            "colors": _normalize_colors(colors),
        }
        print(f"[splat] Loaded PLY {path} ({n:,} splats).")
        return tensors, np.arange(n, dtype=np.int64)

    def _load_npy_folder(self, folder: Path) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        if not folder.exists():
            raise FileNotFoundError(folder)
        loaded: dict[str, np.ndarray] = {}
        for key, aliases in SCENE_ARRAY_ALIASES.items():
            path = _find_array_file(folder, aliases)
            if path is not None:
                loaded[key] = _load_array_file(path)

        missing = [key for key in ("means", "quats", "scales", "opacities", "colors") if key not in loaded]
        if missing:
            raise FileNotFoundError(f"Missing required arrays in {folder}: {', '.join(missing)}")

        means = torch.from_numpy(np.asarray(loaded["means"])).float()
        if means.ndim != 2 or means.shape[-1] != 3:
            raise ValueError(f"coord/means array must have shape [N,3], got {tuple(means.shape)}")
        n = means.shape[0]

        normals = torch.from_numpy(np.asarray(loaded.get("normals", np.zeros((n, 3), dtype=np.float32)))).float()
        quats = _normalize_quats(torch.from_numpy(np.asarray(loaded["quats"])).float())
        scales = torch.from_numpy(np.asarray(loaded["scales"])).float()
        if getattr(self.args, "npy_scale_log", False):
            scales = scales.exp()
        scales = scales.clamp_min(1e-8)
        opacities = _sigmoid_if_logits(torch.from_numpy(np.asarray(loaded["opacities"])).float())
        colors = _normalize_colors(torch.from_numpy(np.asarray(loaded["colors"])).float())

        for name, tensor in {
            "quats": quats,
            "scales": scales,
            "opacities": opacities,
            "colors": colors,
        }.items():
            if tensor.shape[0] != n:
                raise ValueError(f"{name} first dimension {tensor.shape[0]} != means count {n}")

        tensors = {
            "means": means,
            "normals": F.normalize(normals, dim=-1, eps=1e-6),
            "quats": quats,
            "scales": scales,
            "opacities": opacities,
            "colors": colors,
        }
        print(f"[splat] Loaded NumPy folder {folder} ({n:,} splats).")
        return tensors, np.arange(n, dtype=np.int64)

    def _load_aligned_features(self, original_n_before_masks: int, valid_mask: np.ndarray | None) -> None:
        feature_path = self._feature_path
        if feature_path is None:
            empty_cpu = torch.empty((len(self), 0), dtype=torch.float32)
            self._data_cpu["language_feature"] = empty_cpu
            self._data["language_feature"] = empty_cpu.to(self.device)
            return

        raw = np.asarray(load_feature_array(feature_path))
        if raw.ndim > 2:
            raw = raw.reshape(raw.shape[0], -1)
        if raw.ndim == 1:
            raw = raw[:, None]
        raw = raw.astype(np.float32)

        final_ids = self._final_original_indices
        assert final_ids is not None
        features: np.ndarray | None = None
        if self._feature_index is not None:
            features = _align_features_by_source_index(
                raw=raw,
                feature_index=self._feature_index,
                final_ids=final_ids,
                feature_path=feature_path,
            )
        elif raw.shape[0] == original_n_before_masks:
            features = raw[final_ids]
        elif valid_mask is not None and raw.shape[0] == int(valid_mask.sum()):
            valid_ids = np.nonzero(valid_mask)[0]
            rank = np.full(original_n_before_masks, -1, dtype=np.int64)
            rank[valid_ids] = np.arange(len(valid_ids), dtype=np.int64)
            feature_ids = rank[final_ids]
            if np.all(feature_ids >= 0):
                features = raw[feature_ids]
        elif raw.shape[0] == len(final_ids):
            features = raw

        if features is None:
            print(
                f"[feature] Ignoring {feature_path}: feature count {raw.shape[0]} does not align with splat count {len(final_ids)}."
            )
            empty_cpu = torch.empty((len(self), 0), dtype=torch.float32)
            self._data_cpu["language_feature"] = empty_cpu
            self._data["language_feature"] = empty_cpu.to(self.device)
            return

        raw_full = torch.from_numpy(features).float()
        pca_device = _resolve_pca_device(getattr(self.args, "pca_device", "auto"), self.device)
        preview_cpu = _feature_preview(
            raw_full,
            method=getattr(self.args, "pca_method", "torch"),
            pca_device=pca_device,
            brightness=float(getattr(self.args, "pca_brightness", 1.25)),
            seed=int(getattr(self.args, "pca_seed", 42)),
        ).cpu().contiguous()
        # Normalize for stable cosine queries but keep the PCA preview independent.
        full = F.normalize(raw_full, dim=-1, eps=1e-6)
        self._language_feature_large_cpu = full.cpu().contiguous()
        self._language_feature_large = self._language_feature_large_cpu.to(self.device)
        self._data_cpu["language_feature"] = preview_cpu
        self._data["language_feature"] = preview_cpu.to(self.device)
        print(f"[feature] Loaded aligned feature {feature_path} with shape {tuple(full.shape)}.")

    def get_data(self, device: str | torch.device | None = None) -> dict[str, torch.Tensor]:
        if device is not None and str(device).startswith("cpu"):
            return self._data_cpu
        return self._data

    def get_large(self, device: str | torch.device | None = None) -> torch.Tensor | None:
        if device is not None and str(device).startswith("cpu"):
            return self._language_feature_large_cpu
        return self._language_feature_large

    def as_tuple(self, device: str | torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self.get_data(device)
        return (
            data["means"],
            data["quats"],
            data["scales"],
            data["opacities"],
            data["colors"],
        )
