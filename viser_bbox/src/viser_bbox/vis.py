import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import viser

from .geometry import add_bounding_box
from .ply_io import (
    load_gaussian_folder,
    load_ply_file,
    load_point_cloud_folder,
    load_vertices_and_colors,
)
from .script_parser import add_script_bboxes


def create_vis_scene(
    *,
    server: Optional[viser.ViserServer] = None,
    port: Optional[int] = None,
    gaussian_ply: Optional[Path] = None,
    gaussian_path: str = "/gaussian_splats",
    point_cloud_ply: Optional[Path] = None,
    point_cloud_path: str = "/pcd",
    point_size: float = 0.02,
    bbox_script: Optional[str] = None,
    script_path_prefix: str = "/scene",
    bbox_kwargs: Optional[Dict[str, object]] = None,
) -> Tuple[viser.ViserServer, Dict[str, object]]:
    """Create a Viser server populated with Gaussian splats and/or point cloud and optional geometry."""
    if server is None:
        server_kwargs = {}
        if port is not None:
            server_kwargs["port"] = port
        server = viser.ViserServer(**server_kwargs)

    handles: Dict[str, object] = {}

    if gaussian_ply is not None:
        gaussian_ply = Path(gaussian_ply)
        if gaussian_ply.is_dir():
            splat_data = load_gaussian_folder(gaussian_ply, center=False)
        else:
            splat_data = load_ply_file(gaussian_ply, center=False)
        handles["gaussian_splats"] = server.scene.add_gaussian_splats(
            gaussian_path,
            centers=splat_data["centers"],
            covariances=splat_data["covariances"],
            rgbs=splat_data["rgbs"],
            opacities=splat_data["opacities"],
        )

    if point_cloud_ply is not None:
        point_cloud_ply = Path(point_cloud_ply)
        if point_cloud_ply.is_dir():
            points, colors = load_point_cloud_folder(point_cloud_ply)
        else:
            points, colors = load_vertices_and_colors(point_cloud_ply)
        handles["point_cloud"] = server.scene.add_point_cloud(
            point_cloud_path,
            points=points,
            colors=colors,
            point_shape="circle",
            point_size=point_size,
        )

    if bbox_script:
        add_script_bboxes(server, bbox_script, path_prefix=script_path_prefix)

    if bbox_kwargs:
        add_bounding_box(server, **bbox_kwargs)

    return server, handles


def run_vis(
    *,
    gaussian_ply: Optional[Path] = None,
    point_cloud_ply: Optional[Path] = None,
    bbox_script: Optional[str] = None,
    port: Optional[int] = None,
) -> viser.ViserServer:
    """Launch the interactive vis scene."""
    server, _ = create_vis_scene(
        port=port,
        gaussian_ply=gaussian_ply,
        point_cloud_ply=point_cloud_ply,
        bbox_script=bbox_script,
    )

    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        pass

    return server


def main():
    parser = argparse.ArgumentParser(description="Launch Viser demo with Gaussian splats, point cloud, and bounding boxes.")
    parser.add_argument(
        "--gaussian_ply",
        type=Path,
        help="Path to Gaussian splats PLY file or directory of .npy files (optional).",
    )
    parser.add_argument(
        "--point_cloud_ply",
        type=Path,
        help="Path to point cloud PLY file or directory of .npy files (optional).",
    )
    parser.add_argument("--bbox_script", type=Path, help="Path to bounding box script text file (optional).")
    parser.add_argument(
        "--port",
        type=int,
        help="Port to bind the Viser server to (optional).",
    )
    args = parser.parse_args()

    bbox_script_text = None
    if args.bbox_script:
        bbox_script_text = args.bbox_script.read_text()

    run_vis(
        gaussian_ply=args.gaussian_ply,
        point_cloud_ply=args.point_cloud_ply,
        bbox_script=bbox_script_text,
        port=args.port,
    )


if __name__ == "__main__":
    main()

__all__ = ["create_vis_scene", "run_vis"]
