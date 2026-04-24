"""Feature visualization, query scoring, pruning, and query bbox export.

Despite the historical filename, this action now supports both language and
visual feature spaces:

- SigLIP2 / SigLIP / CLIP text queries against aligned language features.
- DINOv2 image-vector queries against aligned DINO features.
- Precomputed query vectors for any feature type.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from core.splat import SplatData, load_feature_array
from models.clip_query import (
    CosineClassifier,
    DINOv2Network,
    DINOv2NetworkConfig,
    OpenCLIPNetwork,
    OpenCLIPNetworkConfig,
    SigLIPNetwork,
    SigLIPNetworkConfig,
)

DINO_TYPES = {"dino", "dinov2", "dino2"}


class LanguageFeature:
    """GUI action for generic aligned-feature querying."""

    def __init__(
        self,
        viewer: Any,
        splatdata: SplatData,
        *,
        feature_type: str = "siglip2",
        query_feature_path: str | Path | None = None,
        query_image_path: str | Path | None = None,
    ) -> None:
        self.viewer = viewer
        self.server = viewer.server
        self.splatdata = splatdata
        self.device = torch.device(getattr(viewer.splat_args, "device", "cuda"))
        self.feature_type = self._normalize_feature_type(feature_type)
        self.query_image_path = Path(query_image_path).expanduser() if query_image_path else None

        self.language_feature_preview = splatdata.get_data().get("language_feature")
        self.language_feature_large = splatdata.get_large()
        self.query_feature: torch.Tensor | None = None
        self.network: Any | None = None
        self.gs_scores: torch.Tensor | None = None
        self.current_mask: torch.Tensor | None = None
        self.feature_map_enabled = False
        self.normal_map_enabled = False
        self.show_query_bbox = False
        self.last_prompt = ""
        self.prune_rate = 0.5
        self._bbox_handles: list[Any] = []
        self._last_bbox: dict[str, Any] | None = None

        if query_feature_path is not None:
            self.query_feature = self._load_query_feature(query_feature_path)

        if self.language_feature_large is None or self.language_feature_large.numel() == 0:
            print("[feature] No aligned feature tensor loaded; query controls are disabled.")
        else:
            print(
                f"[feature] Ready: type={self.feature_type}, "
                f"features={tuple(self.language_feature_large.shape)}."
            )

        self._setup_gui()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _normalize_feature_type(feature_type: str) -> str:
        value = str(feature_type or "siglip2").strip().lower()
        if value in {"dino2", "dino"}:
            return "dinov2"
        if value == "siglip":
            return "siglip2"
        return value

    @property
    def is_dino(self) -> bool:
        return self.feature_type in DINO_TYPES

    def _allow_model_on_cpu(self) -> bool:
        args = self.viewer.splat_args
        return bool(
            getattr(args, "enable_feature_model_on_cpu", False)
            or getattr(args, "enable_language_on_cpu", False)
        )

    def _load_query_feature(self, path: str | Path) -> torch.Tensor:
        arr = np.asarray(load_feature_array(path), dtype=np.float32)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.ndim == 1:
            arr = arr[None]
        query = torch.from_numpy(arr).float().to(self.device)
        query = F.normalize(query, dim=-1, eps=1e-6)
        print(f"[feature] Loaded query feature {path} with shape {tuple(query.shape)}.")
        return query

    def _get_or_create_network(self) -> Any | None:
        if self.network is not None:
            return self.network
        if self.device.type != "cuda" and not self._allow_model_on_cpu():
            print(
                "[feature] CPU mode: model-backed query encoders are disabled. "
                "Use --query-feature, CUDA, or --enable-feature-model-on-cpu."
            )
            return None

        args = self.viewer.splat_args
        cache_dir = getattr(args, "hf_cache_dir", None)
        try:
            if self.feature_type == "clip":
                self.network = OpenCLIPNetwork(OpenCLIPNetworkConfig(), device=str(self.device))
            elif self.is_dino:
                model_name = getattr(args, "dino_model", None) or DINOv2NetworkConfig.model_name
                self.network = DINOv2Network(
                    DINOv2NetworkConfig(
                        model_name=str(model_name),
                        cache_dir=str(cache_dir) if cache_dir is not None else None,
                    ),
                    device=str(self.device),
                )
            else:
                model_name = getattr(args, "siglip_model", None) or SigLIPNetworkConfig.model_name
                self.network = SigLIPNetwork(
                    SigLIPNetworkConfig(
                        model_name=str(model_name),
                        cache_dir=str(cache_dir) if cache_dir is not None else None,
                    ),
                    device=str(self.device),
                )
        except Exception as exc:
            print(f"[feature] Could not initialize {self.feature_type} query encoder: {exc}")
            self.network = None
        return self.network

    def _setup_gui(self) -> None:
        try:
            gui = self.server.gui
            initial_prompt = str(self.query_image_path) if self.query_image_path is not None else ""
            with gui.add_folder("Feature Query"):
                feature_button = gui.add_button("Feature Map")
                reset_button = gui.add_button("Reset RGB")
                normal_button = gui.add_button("Normal Map")
                prompt = gui.add_text("Text prompt / DINO image path", initial_value=initial_prompt)
                threshold = gui.add_slider(
                    "Threshold",
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    initial_value=self.prune_rate,
                )
                query_button = gui.add_button("Query text / image / feature")
                prune_button = gui.add_button("Prune by Query")
                show_bbox = gui.add_checkbox("Show query bbox", initial_value=self.show_query_bbox)
                export_bbox = gui.add_button("Export query bbox")

            @feature_button.on_click
            def _feature(_: Any) -> None:
                self.feature_map_enabled = not self.feature_map_enabled
                self.normal_map_enabled = False
                self.update_splat_renderer()

            @reset_button.on_click
            def _reset(_: Any) -> None:
                self.feature_map_enabled = False
                self.normal_map_enabled = False
                self.current_mask = None
                self.gs_scores = None
                self._remove_bbox()
                self.viewer.update_splat_renderer(self.splatdata, render_mode="rgb")

            @normal_button.on_click
            def _normal(_: Any) -> None:
                self.normal_map_enabled = not self.normal_map_enabled
                self.feature_map_enabled = False
                self.update_splat_renderer()

            @threshold.on_update
            def _threshold(event: Any) -> None:
                self.prune_rate = float(event.target.value)
                self._update_query_bbox()
                self.update_splat_renderer()

            @query_button.on_click
            def _query(_: Any) -> None:
                self.last_prompt = str(prompt.value).strip()
                self.update_language_feature(self.last_prompt)

            @prompt.on_update
            def _prompt(event: Any) -> None:
                self.last_prompt = str(event.target.value).strip()

            @prune_button.on_click
            def _prune(_: Any) -> None:
                if self.gs_scores is None:
                    self.update_language_feature(str(prompt.value).strip())
                if self.gs_scores is not None:
                    self.current_mask = self.gs_scores >= self.prune_rate
                    print(f"[feature] Pruned view to {int(self.current_mask.sum())}/{len(self.current_mask)} splats.")
                    self.update_splat_renderer()
                    self._update_query_bbox()

            @show_bbox.on_update
            def _bbox(event: Any) -> None:
                self.show_query_bbox = bool(event.target.value)
                self._update_query_bbox()

            @export_bbox.on_click
            def _export(_: Any) -> None:
                path = self.export_query_bbox()
                if path is not None:
                    print(f"[feature] Exported query bbox to {path}")
        except Exception as exc:
            print(f"[feature] GUI setup warning: {exc}")

    # ------------------------------------------------------------------ scoring
    def _resolve_image_query_path(self, prompt: str) -> Path | None:
        candidate_text = prompt.strip()
        if not candidate_text and self.query_image_path is not None:
            return self.query_image_path
        if not candidate_text:
            return None
        candidate = Path(candidate_text).expanduser()
        if candidate.exists():
            return candidate
        return None

    def _encode_query(self, prompt: str) -> torch.Tensor | None:
        if self.query_feature is not None:
            return self.query_feature
        if self.is_dino:
            image_path = self._resolve_image_query_path(prompt)
            if image_path is None:
                print("[feature] DINO/DINOv2 queries require an image path or --query-feature vector.")
                return None
            network = self._get_or_create_network()
            if network is None:
                return None
            return network.encode_image(image_path).to(self.device)

        if not prompt:
            print("[feature] Empty query. Enter text or pass --query-feature.")
            return None
        network = self._get_or_create_network()
        if network is None:
            print("[feature] No text encoder available. Use CUDA, --query-feature, or --enable-feature-model-on-cpu.")
            return None
        return network.encode_text(prompt).to(self.device)

    def update_language_feature(self, prompt: str) -> None:
        if self.language_feature_large is None or self.language_feature_large.numel() == 0:
            print("[feature] No loaded aligned features to query.")
            return
        query = self._encode_query(prompt)
        if query is None:
            return
        features = self.language_feature_large.to(self.device)
        if query.shape[-1] != features.shape[-1]:
            print(f"[feature] Query dim {query.shape[-1]} != feature dim {features.shape[-1]}; cannot score.")
            return
        raw_scores = CosineClassifier.score(features, query)
        minv, maxv = raw_scores.min(), raw_scores.max()
        self.gs_scores = ((raw_scores - minv) / (maxv - minv + 1e-6)).clamp(0.0, 1.0)
        self.last_prompt = prompt
        self._update_query_bbox()
        self.update_splat_renderer()

    # ------------------------------------------------------------------ rendering
    def _colored_data(self) -> dict[str, torch.Tensor]:
        data = self.splatdata.get_data()
        out = {key: value for key, value in data.items() if key != "language_feature"}
        colors = data["colors"].clone()

        if (
            self.feature_map_enabled
            and self.language_feature_preview is not None
            and self.language_feature_preview.numel() > 0
        ):
            colors = self.language_feature_preview.to(self.device).clone()
        elif self.normal_map_enabled:
            colors = ((data["normals"] + 1.0) * 0.5).clamp(0.0, 1.0)

        if self.gs_scores is not None and not self.feature_map_enabled and not self.normal_map_enabled:
            selected = self.gs_scores >= self.prune_rate
            if torch.any(selected):
                red = torch.tensor([1.0, 0.0, 0.0], device=colors.device, dtype=colors.dtype)
                colors[selected] = red

        out["colors"] = colors
        if self.current_mask is not None:
            mask = self.current_mask.to(self.device)
            out = {key: value[mask] for key, value in out.items()}
        return out

    def update_splat_renderer(self) -> None:
        data = self._colored_data()
        self.viewer.update_splat_renderer(data, render_mode="rgb")

    # ------------------------------------------------------------------ bbox
    def _selected_mask_for_bbox(self) -> torch.Tensor | None:
        if self.gs_scores is None:
            return None
        mask = self.gs_scores >= self.prune_rate
        if self.current_mask is not None:
            mask = mask & self.current_mask.to(mask.device)
        return mask

    def _remove_bbox(self) -> None:
        for handle in self._bbox_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._bbox_handles = []
        self._last_bbox = None

    def _update_query_bbox(self) -> None:
        self._remove_bbox()
        if not self.show_query_bbox:
            return
        mask = self._selected_mask_for_bbox()
        if mask is None or not torch.any(mask):
            return
        means = self.splatdata.get_data()["means"][mask].detach().cpu()
        lo = means.amin(dim=0).numpy()
        hi = means.amax(dim=0).numpy()
        size = hi - lo
        pad = np.maximum(size * 0.02, 1e-3)
        lo = lo - pad
        hi = hi + pad
        center = (lo + hi) * 0.5
        size = hi - lo
        self._last_bbox = {
            "feature_type": self.feature_type,
            "query": self.last_prompt,
            "threshold": float(self.prune_rate),
            "count": int(mask.sum().item()),
            "min": lo.tolist(),
            "max": hi.tolist(),
            "center": center.tolist(),
            "size": size.tolist(),
        }
        self._add_bbox_lines(lo, hi)

    def _add_bbox_lines(self, lo: np.ndarray, hi: np.ndarray) -> None:
        x0, y0, z0 = lo.tolist()
        x1, y1, z1 = hi.tolist()
        corners = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ],
            dtype=np.float32,
        )
        edges = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ],
            dtype=np.int64,
        )
        segments = corners[edges]
        try:
            handle = self.server.scene.add_line_segments(
                "/query_bbox/box",
                points=segments,
                colors=np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(segments), 1)),
                line_width=3.0,
            )
            self._bbox_handles.append(handle)
        except Exception as exc:
            print(f"[feature] Could not draw query bbox line segments: {exc}")
            return
        try:
            label = self.server.scene.add_label(
                "/query_bbox/label",
                text=f"query bbox: {self._last_bbox['count']} splats" if self._last_bbox else "query bbox",
                position=((lo + hi) * 0.5).astype(np.float32),
            )
            self._bbox_handles.append(label)
        except Exception:
            pass

    def export_query_bbox(self, output_path: str | Path = "outputs/query_bbox.json") -> Path | None:
        self._update_query_bbox()
        if self._last_bbox is None:
            print("[feature] No bbox to export. Run a query and enable/show query bbox first.")
            return None
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._last_bbox, indent=2), encoding="utf8")
        return path
