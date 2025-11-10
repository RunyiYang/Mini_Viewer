import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from plyfile import PlyData

from .geometry import quaternion_to_rotation_matrix


def load_ply_file(ply_file_path: Path, center: bool = False) -> Dict[str, np.ndarray]:
    """Load Gaussian splat data from a PLY file."""
    start_time = time.time()
    sh_c0 = 0.28209479177387814  # Spherical harmonic constant

    plydata = PlyData.read(ply_file_path)
    vertices = plydata["vertex"]

    positions = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=-1)
    scales = np.exp(np.stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]], axis=-1))
    wxyzs = np.stack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]], axis=1)
    colors = 0.5 + sh_c0 * np.stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"],], axis=-1)
    opacities = 1.0 / (1.0 + np.exp(-vertices["opacity"][:, None]))

    rotation_mats = quaternion_to_rotation_matrix(wxyzs)

    scale_mats = np.eye(3)[None, :, :] * scales[:, None, :] ** 2
    covariances = np.einsum("nij,njk,nlk->nil", rotation_mats, scale_mats, rotation_mats)

    if center:
        positions -= np.mean(positions, axis=0, keepdims=True)

    print(f"PLY file loaded in {time.time() - start_time:.2f} seconds")
    return {
        "centers": positions,
        "rgbs": colors,
        "opacities": opacities,
        "covariances": covariances,
    }


def load_vertices_and_colors(ply_file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (positions, colors) arrays extracted from a vertex-only PLY file."""
    plydata = PlyData.read(ply_file_path)
    vertices = plydata["vertex"]

    positions = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=-1)
    colors = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=-1)

    return positions, colors


def load_point_cloud_folder(folder_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load point cloud positions and colors from a directory of NumPy arrays."""
    folder = Path(folder_path)
    coords_path = folder / "coords.npy"
    colors_path = folder / "colors.npy"

    missing = [str(path.name) for path in (coords_path, colors_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required point cloud files in {folder}: {', '.join(missing)}")

    positions = np.load(coords_path)
    colors = np.load(colors_path)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"'coords.npy' in {folder} must have shape (N, 3); got {positions.shape}.")

    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"'colors.npy' in {folder} must have shape (N, 3); got {colors.shape}.")

    if colors.shape[0] != positions.shape[0]:
        raise ValueError(
            f"Point and color counts differ in {folder}: coords has {positions.shape[0]}, colors has {colors.shape[0]}."
        )

    return positions, colors


def load_gaussian_folder(folder_path: Path, center: bool = False) -> Dict[str, np.ndarray]:
    """Load Gaussian splat data from a directory of NumPy arrays."""
    start_time = time.time()

    folder = Path(folder_path)
    required_files = {
        "coord": folder / "coord.npy",
        "color": folder / "color.npy",
        "quat": folder / "quat.npy",
        "scale": folder / "scale.npy",
        "opacity": folder / "opacity.npy",
    }

    missing = [name for name, path in required_files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required Gaussian files in {folder}: {', '.join(missing)}"
        )
    from IPython import embed; embed()
    centers = np.load(required_files["coord"])
    rgbs = np.load(required_files["color"]).astype(np.float32) / 255.0
    quaternions = np.load(required_files["quat"])
    scales = np.load(required_files["scale"])
    opacities = np.load(required_files["opacity"])
    if centers.shape[0] == 0:
        raise ValueError(f"No Gaussian entries found in {folder}.")

    num = centers.shape[0]
    for name, array in (("rgbs", rgbs), ("quaternions", quaternions), ("scales", scales), ("opacities", opacities)):
        if array.shape[0] != num:
            raise ValueError(
                f"Array {name} has inconsistent length ({array.shape[0]}) compared to centers ({num})."
            )

    rotation_mats = quaternion_to_rotation_matrix(quaternions)
    scale_mats = np.eye(3)[None, :, :] * np.square(scales)[:, None, :]
    covariances = np.einsum("nij,njk,nlk->nil", rotation_mats, scale_mats, rotation_mats)

    if center:
        centers = centers - np.mean(centers, axis=0, keepdims=True)

    opacities = np.asarray(opacities).reshape(num, 1)

    print(f"Gaussian folder loaded in {time.time() - start_time:.2f} seconds")
    return {
        "centers": centers,
        "rgbs": rgbs,
        "opacities": opacities,
        "covariances": covariances,
    }


__all__ = [
    "load_gaussian_folder",
    "load_ply_file",
    "load_point_cloud_folder",
    "load_vertices_and_colors",
]
