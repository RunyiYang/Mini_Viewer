"""Camera placement, Nerfstudio-style camera-path export, and video render."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from core.viewer import _fov_to_degrees, _fov_to_radians, _matrix_to_quaternion_wxyz, _quaternion_wxyz_to_matrix


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return _normalize_quat(q0 + t * (q1 - q0))
    theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return _normalize_quat((s0 * q0) + (s1 * q1))


def _parse_matrix(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    mat = np.asarray(value, dtype=np.float64)
    if mat.shape == (4, 4):
        return mat
    if mat.size == 16:
        return mat.reshape(4, 4)
    raise ValueError(f"Expected 4x4 matrix or 16 flat values, got shape {mat.shape}")


class CameraPathFeature:
    def __init__(self, viewer: Any, splatdata: Any, args: Any) -> None:
        self.viewer = viewer
        self.splatdata = splatdata
        self.args = args
        self.server = viewer.server
        self.keyframes: list[dict[str, Any]] = []
        self._handles: list[Any] = []
        self.camera_path_output = Path(getattr(args, "camera_path", Path("outputs/camera_path.json")))
        self.video_output = Path(getattr(args, "video_output", Path("outputs/render.mp4")))
        self.width = int(getattr(args, "render_width", 1280))
        self.height = int(getattr(args, "render_height", 720))
        self.fps = int(getattr(args, "render_fps", 30))
        self.seconds = float(getattr(args, "render_seconds", 5.0))
        self._setup_gui()

    # ------------------------------------------------------------------ GUI
    def _setup_gui(self) -> None:
        try:
            gui = self.server.gui
            with gui.add_folder("Camera Path"):
                path_text = gui.add_text("Camera JSON", initial_value=str(self.camera_path_output))
                video_text = gui.add_text("Video output", initial_value=str(self.video_output))
                width_slider = gui.add_slider("Width", min=320, max=4096, step=16, initial_value=self.width)
                height_slider = gui.add_slider("Height", min=240, max=4096, step=16, initial_value=self.height)
                fps_slider = gui.add_slider("FPS", min=1, max=120, step=1, initial_value=self.fps)
                seconds_slider = gui.add_slider("Seconds", min=0.1, max=120.0, step=0.1, initial_value=self.seconds)
                add_button = gui.add_button("Add Camera")
                clear_button = gui.add_button("Clear Cameras")
                export_button = gui.add_button("Export Cameras")
                load_button = gui.add_button("Load Cameras")
                render_button = gui.add_button("Render Video")

            @path_text.on_update
            def _path(event: Any) -> None:
                self.camera_path_output = Path(str(event.target.value))

            @video_text.on_update
            def _video(event: Any) -> None:
                self.video_output = Path(str(event.target.value))

            @width_slider.on_update
            def _width(event: Any) -> None:
                self.width = int(event.target.value)

            @height_slider.on_update
            def _height(event: Any) -> None:
                self.height = int(event.target.value)

            @fps_slider.on_update
            def _fps(event: Any) -> None:
                self.fps = int(event.target.value)

            @seconds_slider.on_update
            def _seconds(event: Any) -> None:
                self.seconds = float(event.target.value)

            @add_button.on_click
            def _add(_: Any) -> None:
                self.add_current_camera()

            @clear_button.on_click
            def _clear(_: Any) -> None:
                self.clear_keyframes()

            @export_button.on_click
            def _export(_: Any) -> None:
                out = self.export_camera_path(self.camera_path_output)
                print(f"[camera] Exported {out}")

            @load_button.on_click
            def _load(_: Any) -> None:
                self.load_camera_path(self.camera_path_output)
                print(f"[camera] Loaded {len(self.keyframes)} cameras from {self.camera_path_output}")

            @render_button.on_click
            def _render(_: Any) -> None:
                path = self.export_camera_path(self.camera_path_output)
                spec = json.loads(Path(path).read_text(encoding="utf8"))
                self.viewer.render_camera_path(
                    spec["camera_path"],
                    self.video_output,
                    width=int(spec["render_width"]),
                    height=int(spec["render_height"]),
                    fps=int(spec["fps"]),
                )
        except Exception as exc:
            print(f"[camera] GUI setup warning: {exc}")

    # ------------------------------------------------------------------ keyframes
    def add_current_camera(self) -> None:
        state = self.viewer.current_camera_state()
        c2w = np.asarray(state.c2w, dtype=np.float64).reshape(4, 4)
        keyframe = {
            "c2w": c2w,
            "fov": float(getattr(state, "fov", math.radians(45.0))),
            "aspect": float(getattr(state, "aspect", self.width / max(self.height, 1))),
            "time": len(self.keyframes),
        }
        self.keyframes.append(keyframe)
        self._draw_keyframes()
        print(f"[camera] Added keyframe {len(self.keyframes) - 1}")

    def clear_keyframes(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles = []
        self.keyframes = []
        print("[camera] Cleared keyframes")

    def _draw_keyframes(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles = []
        for i, key in enumerate(self.keyframes):
            c2w = key["c2w"]
            pos = c2w[:3, 3].astype(np.float32)
            q = _matrix_to_quaternion_wxyz(c2w[:3, :3]).astype(np.float32)
            try:
                handle = self.server.scene.add_camera_frustum(
                    f"/camera_path/keyframe_{i}",
                    fov=float(key["fov"]),
                    aspect=float(key["aspect"]),
                    scale=0.25,
                    color=(255, 180, 0),
                    wxyz=q,
                    position=pos,
                )
                self._handles.append(handle)
            except Exception:
                try:
                    handle = self.server.scene.add_frame(
                        f"/camera_path/keyframe_{i}",
                        wxyz=q,
                        position=pos,
                        axes_length=0.25,
                        axes_radius=0.01,
                    )
                    self._handles.append(handle)
                except Exception:
                    pass

    # ------------------------------------------------------------------ export/load
    def _interpolate(self) -> list[dict[str, Any]]:
        if not self.keyframes:
            self.add_current_camera()
        if len(self.keyframes) == 1:
            count = max(1, int(round(self.fps * self.seconds)))
            key = self.keyframes[0]
            return [self._camera_path_item(key["c2w"], key["fov"], key["aspect"]) for _ in range(count)]

        count = max(2, int(round(self.fps * self.seconds)))
        total_segments = len(self.keyframes) - 1
        items: list[dict[str, Any]] = []
        for frame_idx in range(count):
            u = frame_idx / max(count - 1, 1) * total_segments
            seg = min(int(math.floor(u)), total_segments - 1)
            t = float(u - seg)
            k0 = self.keyframes[seg]
            k1 = self.keyframes[seg + 1]
            c0 = np.asarray(k0["c2w"], dtype=np.float64)
            c1 = np.asarray(k1["c2w"], dtype=np.float64)
            q0 = _matrix_to_quaternion_wxyz(c0[:3, :3])
            q1 = _matrix_to_quaternion_wxyz(c1[:3, :3])
            q = _slerp(q0, q1, t)
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = _quaternion_wxyz_to_matrix(q)
            c2w[:3, 3] = (1.0 - t) * c0[:3, 3] + t * c1[:3, 3]
            fov = (1.0 - t) * float(k0["fov"]) + t * float(k1["fov"])
            aspect = (1.0 - t) * float(k0["aspect"]) + t * float(k1["aspect"])
            items.append(self._camera_path_item(c2w, fov, aspect))
        return items

    @staticmethod
    def _camera_path_item(c2w: np.ndarray, fov: float, aspect: float) -> dict[str, Any]:
        return {
            "camera_to_world": np.asarray(c2w, dtype=float).reshape(4, 4).reshape(-1).tolist(),
            "fov": _fov_to_degrees(float(fov)),
            "aspect": float(aspect),
        }

    def build_camera_path_json(self) -> dict[str, Any]:
        camera_path = self._interpolate()
        keyframes = []
        for i, key in enumerate(self.keyframes):
            c2w = np.asarray(key["c2w"], dtype=float).reshape(4, 4)
            fov_deg = _fov_to_degrees(float(key["fov"]))
            keyframes.append(
                {
                    "matrix": str(c2w.reshape(-1).tolist()),
                    "fov": fov_deg,
                    "aspect": float(key["aspect"]),
                    "properties": json.dumps([["FOV", fov_deg], ["NAME", f"Camera {i}"], ["TIME", float(i)]]),
                }
            )
        return {
            "keyframes": keyframes,
            "camera_type": "perspective",
            "render_height": int(self.height),
            "render_width": int(self.width),
            "camera_path": camera_path,
            "fps": int(self.fps),
            "seconds": float(self.seconds),
            "smoothness_value": 0,
            "is_cycle": False,
        }

    def export_camera_path(self, path: str | Path | None = None) -> Path:
        path = Path(path or self.camera_path_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        spec = self.build_camera_path_json()
        path.write_text(json.dumps(spec, indent=2), encoding="utf8")
        return path

    def load_camera_path(self, path: str | Path) -> None:
        path = Path(path)
        spec = json.loads(path.read_text(encoding="utf8"))
        self.clear_keyframes()
        if "keyframes" in spec and spec["keyframes"]:
            for key in spec["keyframes"]:
                c2w = _parse_matrix(key.get("matrix", key.get("camera_to_world")))
                self.keyframes.append(
                    {
                        "c2w": c2w,
                        "fov": _fov_to_radians(float(key.get("fov", 45.0))),
                        "aspect": float(key.get("aspect", spec.get("render_width", 16) / max(spec.get("render_height", 9), 1))),
                        "time": float(len(self.keyframes)),
                    }
                )
        else:
            for item in spec.get("camera_path", []):
                c2w = _parse_matrix(item["camera_to_world"])
                self.keyframes.append(
                    {
                        "c2w": c2w,
                        "fov": _fov_to_radians(float(item.get("fov", 45.0))),
                        "aspect": float(item.get("aspect", spec.get("render_width", 16) / max(spec.get("render_height", 9), 1))),
                        "time": float(len(self.keyframes)),
                    }
                )
        self.width = int(spec.get("render_width", self.width))
        self.height = int(spec.get("render_height", self.height))
        self.fps = int(spec.get("fps", self.fps))
        self.seconds = float(spec.get("seconds", self.seconds))
        self._draw_keyframes()
