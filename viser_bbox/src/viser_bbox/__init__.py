from .constants import (
    FLOAT,
)
# Remove eager import from .vis to avoid runpy warning
from .geometry import add_bounding_box, get_min_max_coords, quaternion_to_rotation_matrix
from .ply_io import load_gaussian_folder, load_ply_file, load_point_cloud_folder, load_vertices_and_colors
from .script_parser import add_script_bboxes
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vis import create_vis_scene, run_vis  # type: ignore

def __getattr__(name: str):
    if name in ("create_vis_scene", "run_vis"):
        mod = importlib.import_module(".vis", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "FLOAT",
    "add_bounding_box",
    "add_script_bboxes",
    "create_vis_scene",
    "get_min_max_coords",
    "load_gaussian_folder",
    "load_ply_file",
    "load_point_cloud_folder",
    "load_vertices_and_colors",
    "quaternion_to_rotation_matrix",
    "run_vis",
]
