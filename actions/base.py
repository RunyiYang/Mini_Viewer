"""Basic viewer controls."""

from __future__ import annotations

from typing import Any


class BasicFeature:
    """RGB/depth/normal/snapshot controls."""

    def __init__(self, viewer: Any, splatdata: Any) -> None:
        self.viewer = viewer
        self.splatdata = splatdata
        try:
            gui = viewer.server.gui
            with gui.add_folder("Basic"):
                rgb_button = gui.add_button("RGB")
                depth_button = gui.add_button("Depth")
                normal_button = gui.add_button("Normal")
                snapshot_button = gui.add_button("Snapshot")

            @rgb_button.on_click
            def _rgb(_: Any) -> None:
                viewer.update_splat_renderer(splatdata, render_mode="rgb")

            @depth_button.on_click
            def _depth(_: Any) -> None:
                viewer.update_splat_renderer(splatdata, render_mode="depth")

            @normal_button.on_click
            def _normal(_: Any) -> None:
                data = splatdata.get_data()
                old = data["colors"]
                data["colors"] = ((data["normals"] + 1.0) * 0.5).clamp(0.0, 1.0)
                viewer.update_splat_renderer(data, render_mode="rgb")
                data["colors"] = old

            @snapshot_button.on_click
            def _snapshot(_: Any) -> None:
                viewer._snapshot_index = viewer.snapshot(viewer._snapshot_index)
        except Exception as exc:
            print(f"[basic] GUI setup warning: {exc}")
