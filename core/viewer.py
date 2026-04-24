"""Viewer wrapper with camera persistence and camera-path video rendering."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import imageio.v2 as imageio
import nerfview
import numpy as np
import torch
from PIL import Image

from core.renderer import image_to_uint8_numpy, viewer_render_fn


@dataclass
class RenderTask:
    camera_state: Any
    img_wh: tuple[int, int]
    client_id: int | str | None = None


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""
    m = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q


def _quaternion_wxyz_to_matrix(q: Iterable[float]) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(list(q), dtype=np.float64)
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) + 1e-12
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _fov_to_radians(fov: float) -> float:
    fov = float(fov)
    return math.radians(fov) if fov > math.pi else fov


def _fov_to_degrees(fov: float) -> float:
    fov = float(fov)
    return fov if fov > math.pi else math.degrees(fov)


class ViewerEditor(nerfview.Viewer):
    """Small extension around nerfview.Viewer.

    Adds FOV control, camera save/restore, safe screenshots, and camera-path
    rendering/export support.
    """

    def __init__(self, splat_args: Any, splat_data: Any, *args: Any, **kwargs: Any) -> None:
        self.splat_args = splat_args
        self.splat_data = splat_data
        self.device = getattr(splat_args, "device", "cuda")
        self._saved_camera_path = Path(".tmp/camera_state.json")
        self._snapshot_index = 0
        self._last_camera_state: dict[str, Any] | None = None
        self._render_mode = "rgb"
        self.force_cpu_render = bool(getattr(splat_args, "force_cpu_render", False))
        self.cpu_render_fallback = bool(getattr(splat_args, "cpu_render_fallback", True))
        self.cpu_fallback_splats = int(getattr(splat_args, "cpu_fallback_splats", 80_000) or 0)
        self._active_splats: Any = splat_data
        self._active_sh_degree = int(getattr(splat_args, "sh_degree", 3))
        self._active_backend = getattr(splat_args, "backend", "auto")
        super().__init__(*args, **kwargs)
        # Use the consolidated renderer function for all renderer rebuilds. The
        # render_fn passed to nerfview at construction is only the initial state.
        self._base_render_fn = viewer_render_fn
        self.adjust_viewer()

    # ------------------------------------------------------------------ UI setup
    def adjust_viewer(self) -> None:
        server = self.server
        saved = self._load_saved_camera_state()
        if saved is not None:
            self._last_camera_state = saved

        @server.on_client_connect
        def _on_client_connect(client: Any) -> None:
            try:
                center = self.find_splat_central_point().detach().cpu().numpy()
            except Exception:
                center = np.zeros(3, dtype=np.float32)
            if saved is not None:
                try:
                    c2w = np.asarray(saved["c2w"], dtype=np.float64).reshape(4, 4)
                    client.camera.position = c2w[:3, 3]
                    client.camera.wxyz = _matrix_to_quaternion_wxyz(c2w[:3, :3])
                    client.camera.fov = _fov_to_radians(float(saved.get("fov", math.radians(45.0))))
                    client.camera.update_timestamp = float(saved.get("update_timestamp", 0.0))
                except Exception:
                    pass
            else:
                try:
                    client.camera.position = center + np.array([0.0, 0.0, -6.0])
                    client.camera.look_at = center
                    client.camera.fov = math.radians(45.0)
                except Exception:
                    pass

        try:
            gui = server.gui
            with gui.add_folder("Viewer"):
                fov_slider = gui.add_slider("FOV", min=15.0, max=100.0, step=1.0, initial_value=45.0)
                max_res_slider = gui.add_slider("Max image res", min=256, max=4096, step=64, initial_value=1920)
                cpu_fallback_checkbox = gui.add_checkbox(
                    "CPU fallback on CUDA render error",
                    initial_value=self.cpu_render_fallback,
                )
                force_cpu_checkbox = gui.add_checkbox(
                    "Force CPU renderer",
                    initial_value=self.force_cpu_render,
                )
                fallback_splats_slider = gui.add_slider(
                    "CPU fallback splats",
                    min=10_000,
                    max=500_000,
                    step=10_000,
                    initial_value=max(10_000, min(500_000, self.cpu_fallback_splats or 80_000)),
                )
                snapshot_button = gui.add_button("Snapshot")
                save_camera_button = gui.add_button("Save current camera")

            @fov_slider.on_update
            def _on_fov_update(event: Any) -> None:
                self.rerender_K(math.radians(float(event.target.value)))

            @max_res_slider.on_update
            def _on_res_update(event: Any) -> None:
                try:
                    self.max_img_res = int(event.target.value)
                except Exception:
                    pass
                self._request_rerender()

            @cpu_fallback_checkbox.on_update
            def _on_cpu_fallback_update(event: Any) -> None:
                self.cpu_render_fallback = bool(event.target.value)
                setattr(self.splat_args, "cpu_render_fallback", self.cpu_render_fallback)
                print(f"[viewer] CPU render fallback: {self.cpu_render_fallback}")
                self._rebuild_active_renderer()

            @force_cpu_checkbox.on_update
            def _on_force_cpu_update(event: Any) -> None:
                self.force_cpu_render = bool(event.target.value)
                setattr(self.splat_args, "force_cpu_render", self.force_cpu_render)
                mode = "CPU torch" if self.force_cpu_render else f"{self.device}/{self._active_backend}"
                print(f"[viewer] Active renderer: {mode}")
                self._rebuild_active_renderer()

            @fallback_splats_slider.on_update
            def _on_fallback_splats_update(event: Any) -> None:
                self.cpu_fallback_splats = int(event.target.value)
                setattr(self.splat_args, "cpu_fallback_splats", self.cpu_fallback_splats)
                self._rebuild_active_renderer()

            @snapshot_button.on_click
            def _on_snapshot(_: Any) -> None:
                self._snapshot_index = self.snapshot(self._snapshot_index)

            @save_camera_button.on_click
            def _on_save_camera(_: Any) -> None:
                out = self.save_camera_state(self._saved_camera_path)
                print(f"[camera] Saved {out}")
        except Exception as exc:
            print(f"[viewer] GUI setup warning: {exc}")

    # ------------------------------------------------------------- data swapping
    def update_splat_renderer(
        self,
        splats: Any,
        *,
        sh_degree: int | None = None,
        backend: str | None = None,
        render_mode: str = "rgb",
        update_inplace_memory: bool = False,
    ) -> None:
        del update_inplace_memory  # kept for compatibility with older call sites.
        cpu_data: dict[str, torch.Tensor] | None = None
        if hasattr(splats, "get_data"):
            data = splats.get_data()
            try:
                cpu_data = splats.get_data("cpu")
            except TypeError:
                cpu_data = None
        else:
            data = splats

        sh_degree = int(sh_degree if sh_degree is not None else getattr(self.splat_args, "sh_degree", 3))
        backend = backend or getattr(self.splat_args, "backend", "auto")
        render_device = "cpu" if self.force_cpu_render else self.device
        render_backend = "torch" if self.force_cpu_render else backend

        if cpu_data is None and (self.force_cpu_render or self.cpu_render_fallback):
            try:
                cpu_data = {
                    key: value.detach().cpu().contiguous()
                    for key, value in data.items()
                    if isinstance(value, torch.Tensor)
                }
            except Exception as exc:
                cpu_data = None
                print(f"[viewer] CPU render mirror warning: {exc}")

        render_data = cpu_data if self.force_cpu_render and cpu_data is not None else data

        self._active_splats = splats
        self._active_sh_degree = sh_degree
        self._active_backend = backend
        self._render_mode = render_mode

        self.render_fn = partial(
            self._base_render_fn,
            means=render_data["means"],
            quats=render_data["quats"],
            scales=render_data["scales"],
            opacities=render_data["opacities"],
            colors=render_data["colors"],
            sh_degree=sh_degree,
            device=render_device,
            backend=render_backend,
            render_mode=render_mode,
            max_cpu_splats=getattr(self.splat_args, "max_cpu_splats", 180_000),
            fallback_to_cpu=self.cpu_render_fallback,
            cpu_fallback_splats=self.cpu_fallback_splats,
            cpu_means=cpu_data.get("means") if cpu_data is not None else None,
            cpu_scales=cpu_data.get("scales") if cpu_data is not None else None,
            cpu_opacities=cpu_data.get("opacities") if cpu_data is not None else None,
            cpu_colors=cpu_data.get("colors") if cpu_data is not None else None,
        )
        self._request_rerender()

    def _rebuild_active_renderer(self) -> None:
        """Recreate render_fn after toggling CPU/gsplat runtime options."""
        self.update_splat_renderer(
            self._active_splats,
            sh_degree=self._active_sh_degree,
            backend=self._active_backend,
            render_mode=self._render_mode,
        )

    def _request_rerender(self) -> None:
        """Request nerfview to refresh across API versions.

        Some nerfview releases expose ``rerender(self, event)`` while others
        expose ``rerender(self)``. Call both forms defensively so GUI callbacks
        and programmatic renderer swaps work on both versions.
        """
        try:
            self.rerender(None)
        except TypeError:
            try:
                self.rerender()
            except Exception as exc:
                print(f"[viewer] rerender warning: {exc}")
        except Exception as exc:
            print(f"[viewer] rerender warning: {exc}")

    # ------------------------------------------------------------- camera/state
    def find_splat_central_point(self) -> torch.Tensor:
        data = self.splat_data.get_data()
        if len(data["means"]) == 0:
            return torch.zeros(3, device=data["means"].device)
        return data["means"].mean(dim=0)

    def get_first_client(self) -> Any | None:
        try:
            clients = self.server.get_clients()
            if isinstance(clients, dict):
                return next(iter(clients.values()), None)
            if isinstance(clients, (list, tuple)):
                return clients[0] if clients else None
        except Exception:
            return None
        return None

    def current_camera_state(self, client: Any | None = None) -> Any:
        client = client or self.get_first_client()
        if client is not None:
            position = np.asarray(getattr(client.camera, "position", [0.0, 0.0, 0.0]), dtype=np.float64)
            wxyz = np.asarray(getattr(client.camera, "wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = _quaternion_wxyz_to_matrix(wxyz)
            c2w[:3, 3] = position
            fov = float(getattr(client.camera, "fov", math.radians(45.0)))
            aspect = float(getattr(client.camera, "aspect", 16.0 / 9.0))
            return nerfview.CameraState(c2w=c2w, fov=fov, aspect=aspect)
        if self._last_camera_state is not None:
            c2w = np.asarray(self._last_camera_state["c2w"], dtype=np.float64).reshape(4, 4)
            fov = _fov_to_radians(float(self._last_camera_state.get("fov", math.radians(45.0))))
            aspect = float(self._last_camera_state.get("aspect", 16.0 / 9.0))
            return nerfview.CameraState(c2w=c2w, fov=fov, aspect=aspect)
        return nerfview.CameraState(c2w=np.eye(4, dtype=np.float64), fov=math.radians(45.0), aspect=16.0 / 9.0)

    def serialize_camera_state(self, client: Any | None = None) -> dict[str, Any]:
        state = self.current_camera_state(client)
        return {
            "c2w": np.asarray(state.c2w, dtype=float).reshape(4, 4).tolist(),
            "fov": float(getattr(state, "fov", math.radians(45.0))),
            "fov_degrees": _fov_to_degrees(float(getattr(state, "fov", math.radians(45.0)))),
            "aspect": float(getattr(state, "aspect", 16.0 / 9.0)),
        }

    def save_camera_state(self, path: str | Path = ".tmp/camera_state.json") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.serialize_camera_state()
        path.write_text(json.dumps(data, indent=2), encoding="utf8")
        self._last_camera_state = data
        return path

    def _load_saved_camera_state(self) -> dict[str, Any] | None:
        path = self._saved_camera_path
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf8"))
        except Exception:
            return None

    # ------------------------------------------------------------- rendering ops
    def call_renderer(self, camera_state: Any | None = None, img_wh: tuple[int, int] | None = None) -> np.ndarray:
        camera_state = camera_state or self.current_camera_state()
        if img_wh is None:
            img_wh = (getattr(self.splat_args, "render_width", 1280), getattr(self.splat_args, "render_height", 720))
        return self.render_fn(camera_state, img_wh)

    def snapshot(self, idx: int = 0) -> int:
        state = self.current_camera_state()
        image = self.call_renderer(state, (getattr(self.splat_args, "render_width", 1280), getattr(self.splat_args, "render_height", 720)))
        image_np = image_to_uint8_numpy(image)
        out = Path(f"snapshot{idx}.png")
        Image.fromarray(image_np).save(out)
        self.save_camera_state(self._saved_camera_path)
        print(f"[snapshot] Saved {out}")
        return idx + 1

    def rerender_K(self, fov: float) -> None:
        try:
            for client in self.server.get_clients().values():
                client.camera.fov = float(fov)
        except Exception:
            pass
        self._request_rerender()

    def render_camera_path(
        self,
        camera_path: list[dict[str, Any]],
        output_path: str | Path,
        *,
        width: int,
        height: int,
        fps: int,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames: list[np.ndarray] = []
        for i, item in enumerate(camera_path):
            c2w = np.asarray(item["camera_to_world"], dtype=np.float64).reshape(4, 4)
            fov = _fov_to_radians(float(item.get("fov", math.degrees(math.radians(45.0)))))
            aspect = float(item.get("aspect", width / max(height, 1)))
            state = nerfview.CameraState(c2w=c2w, fov=fov, aspect=aspect)
            image = self.call_renderer(state, (width, height))
            frames.append(image_to_uint8_numpy(image))
            if (i + 1) % 30 == 0 or i + 1 == len(camera_path):
                print(f"[render] {i + 1}/{len(camera_path)} frames")
        imageio.mimsave(output_path, frames, fps=fps, macro_block_size=1)
        print(f"[render] Saved {output_path}")
        return output_path

    def render_camera_path_file(
        self,
        camera_path_json: str | Path,
        output_path: str | Path | None = None,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ) -> Path:
        path = Path(camera_path_json)
        spec = json.loads(path.read_text(encoding="utf8"))
        width = int(width or spec.get("render_width", getattr(self.splat_args, "render_width", 1280)))
        height = int(height or spec.get("render_height", getattr(self.splat_args, "render_height", 720)))
        fps = int(fps or spec.get("fps", getattr(self.splat_args, "render_fps", 30)))
        output_path = output_path or getattr(self.splat_args, "video_output", Path("outputs/render.mp4"))
        return self.render_camera_path(spec["camera_path"], output_path, width=width, height=height, fps=fps)
