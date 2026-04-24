"""Language feature visualization, query recoloring, pruning, and query bbox."""

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
    OpenCLIPNetwork,
    OpenCLIPNetworkConfig,
    SigLIPNetwork,
    SigLIPNetworkConfig,
)


class LanguageFeature:
    def __init__(
        self,
        viewer: Any,
        splatdata: SplatData,
        *,
        feature_type: str = "siglip",
        query_feature_path: str | Path | None = None,
    ) -> None:
        self.viewer = viewer
        self.server = viewer.server
        self.splatdata = splatdata
        self.device = torch.device(getattr(viewer.splat_args, "device", "cuda"))
        self.feature_type = feature_type.lower()
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

        if self.language_feature_large is not None and self.language_feature_large.numel() > 0:
            self.network = self._maybe_create_network()
        else:
            print("[language] No language feature tensor loaded; query controls are disabled.")

        self._setup_gui()

    # ------------------------------------------------------------------ setup
    def _load_query_feature(self, path: str | Path) -> torch.Tensor:
        arr = np.asarray(load_feature_array(path), dtype=np.float32)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.ndim == 1:
            arr = arr[None]
        query = torch.from_numpy(arr).float().to(self.device)
        query = F.normalize(query, dim=-1, eps=1e-6)
        print(f"[language] Loaded query feature {path} with shape {tuple(query.shape)}.")
        return query

    def _maybe_create_network(self) -> Any | None:
        args = self.viewer.splat_args
        if self.device.type != "cuda" and not getattr(args, "enable_language_on_cpu", False):
            print("[language] CPU mode: text encoder is disabled. Use --query-feature or --enable-language-on-cpu.")
            return None
        try:
            if self.feature_type == "clip":
                return OpenCLIPNetwork(OpenCLIPNetworkConfig(), device=str(self.device))

            model_name = getattr(args, "siglip_model", None) or SigLIPNetworkConfig.model_name
            cache_dir = getattr(args, "hf_cache_dir", None)
            return SigLIPNetwork(
                SigLIPNetworkConfig(
                    model_name=str(model_name),
                    cache_dir=str(cache_dir) if cache_dir is not None else None,
                ),
                device=str(self.device),
            )
        except Exception as exc:
            print(f"[language] Could not initialize {self.feature_type} text encoder: {exc}")
            return None

    def _setup_gui(self) -> None:
        try:
            gui = self.server.gui
            with gui.add_folder("Language Feature"):
                feature_button = gui.add_button("Feature Map")
                reset_button = gui.add_button("Reset RGB")
                normal_button = gui.add_button("Normal Map")
                prompt = gui.add_text("Text Prompt", initial_value="")
                threshold = gui.add_slider("Threshold", min=0.0, max=1.0, step=0.01, initial_value=self.prune_rate)
                query_button = gui.add_button("Query Text / Feature")
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
                    print(f"[language] Pruned view to {int(self.current_mask.sum())}/{len(self.current_mask)} splats.")
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
                    print(f"[language] Exported query bbox to {path}")
        except Exception as exc:
            print(f"[language] GUI setup warning: {exc}")

    # ------------------------------------------------------------------ scoring
    def _encode_query(self, text: str) -> torch.Tensor | None:
        if self.query_feature is not None:
            return self.query_feature
        if not text:
            print("[language] Empty query. Enter text or pass --query-feature.")
            return None
        if self.network is None:
            print("[language] No text encoder available. Install language extras, use CUDA, or pass --query-feature.")
            return None
        return self.network.encode_text(text).to(self.device)

    def update_language_feature(self, text: str) -> None:
        if self.language_feature_large is None or self.language_feature_large.numel() == 0:
            print("[language] No loaded language features to query.")
            return
        query = self._encode_query(text)
        if query is None:
            return
        features = self.language_feature_large.to(self.device)
        if query.shape[-1] != features.shape[-1]:
            print(f"[language] Query dim {query.shape[-1]} != feature dim {features.shape[-1]}; cannot score.")
            return
        raw_scores = CosineClassifier.score(features, query)
        minv, maxv = raw_scores.min(), raw_scores.max()
        self.gs_scores = ((raw_scores - minv) / (maxv - minv + 1e-6)).clamp(0.0, 1.0)
        self.last_prompt = text
        self._update_query_bbox()
        self.update_splat_renderer()

    # ------------------------------------------------------------------ rendering
    def _colored_data(self) -> dict[str, torch.Tensor]:
        data = self.splatdata.get_data()
        out = {key: value for key, value in data.items() if key != "language_feature"}
        colors = data["colors"].clone()

        if self.feature_map_enabled and self.language_feature_preview is not None and self.language_feature_preview.numel() > 0:
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
        # Keep zero-volume selections visible.
        pad = np.maximum(size * 0.02, 1e-3)
        lo = lo - pad
        hi = hi + pad
        center = (lo + hi) * 0.5
        size = hi - lo
        self._last_bbox = {
            "prompt": self.last_prompt,
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
            [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]],
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
            print(f"[language] Could not draw query bbox line segments: {exc}")
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
            print("[language] No bbox to export. Run a query and enable/show query bbox first.")
            return None
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._last_bbox, indent=2), encoding="utf8")
        return path
