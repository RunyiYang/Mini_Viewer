"""Gaussian splat data loading.

Supported scene inputs:
- Inria/3DGS style .ply files.
- NumPy folders with coord/quat/scale/opacity/color arrays.
- Optional aligned feature .npy/.pt/.pth files aligned to splats.
"""

from __future__ import annotations

import math
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


def _feature_preview(features: torch.Tensor) -> torch.Tensor:
    """Create a 3-channel preview from arbitrary-dimensional language features."""
    features_cpu = features.detach().float().cpu()
    if features_cpu.ndim == 1:
        features_cpu = features_cpu[:, None]
    if features_cpu.shape[-1] == 3:
        preview = features_cpu
    else:
        try:
            from sklearn.decomposition import PCA

            sample = features_cpu.numpy()
            preview = torch.from_numpy(PCA(n_components=3, random_state=0).fit_transform(sample)).float()
        except Exception:
            if features_cpu.shape[-1] >= 3:
                preview = features_cpu[:, :3]
            else:
                preview = F.pad(features_cpu, (0, 3 - features_cpu.shape[-1]))
    minv = preview.amin(dim=0, keepdim=True)
    maxv = preview.amax(dim=0, keepdim=True)
    return ((preview - minv) / (maxv - minv + 1e-6)).clamp(0.0, 1.0)


def _structured_columns(vertex: Any, names: list[str]) -> np.ndarray:
    return np.stack([np.asarray(vertex[name]) for name in names], axis=-1)


def _has_props(vertex: Any, names: list[str]) -> bool:
    return all(name in vertex.data.dtype.names for name in names)


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
        scene_path = getattr(self.args, "folder_npy", None) or getattr(self.args, "ply", None)
        feature_path = resolve_feature_path(Path(scene_path) if scene_path else None, getattr(self.args, "language_feature", None))
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
        if raw.shape[0] == original_n_before_masks:
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

        full = torch.from_numpy(features).float()
        # Normalize for stable cosine queries but keep the preview independent.
        full = F.normalize(full, dim=-1, eps=1e-6)
        preview_cpu = _feature_preview(full).cpu().contiguous()
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
