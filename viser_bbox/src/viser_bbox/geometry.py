import math
from typing import Optional, Tuple

import numpy as np
import viser


def quaternion_to_rotation_matrix(quaternions: np.ndarray) -> np.ndarray:
    """Convert quaternions [w, x, y, z] to rotation matrices using pure NumPy."""
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # Normalize quaternions
    norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # Calculate rotation matrix elements
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Build rotation matrices
    R = np.zeros((len(quaternions), 3, 3))
    R[:, 0, 0] = 1 - 2 * (yy + zz)
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = 1 - 2 * (xx + zz)
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = 1 - 2 * (xx + yy)

    return R


def add_bounding_box(
    server: viser.ViserServer,
    position_x: float,
    position_y: float,
    position_z: float,
    angle_z: float,
    scale_x: float,
    scale_y: float,
    scale_z: float,
    path: str = "/bbox",
    color: Tuple[int, int, int] = (0, 255, 0),
    line_width: float = 3.0,
    label: Optional[str] = None,
    label_position: str = "above",  # "above", "below", "front", "back", "left", "right"
) -> None:
    """
    Add a wireframe bounding box to the Viser scene with transform properties and optional label.
    """
    # Create a unit cube centered at origin
    unit_cube = np.array([
        [-0.5, -0.5, -0.5],  # back bottom left
        [0.5, -0.5, -0.5],  # back bottom right
        [0.5, 0.5, -0.5],  # back top right
        [-0.5, 0.5, -0.5],  # back top left
        [-0.5, -0.5, 0.5],  # front bottom left
        [0.5, -0.5, 0.5],  # front bottom right
        [0.5, 0.5, 0.5],  # front top right
        [-0.5, 0.5, 0.5],  # front top left
    ])

    # Apply scaling
    scale_matrix = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, scale_z],
    ])
    scaled_corners = unit_cube @ scale_matrix.T

    # Apply rotation around Z axis
    cos_angle = np.cos(angle_z)
    sin_angle = np.sin(angle_z)
    rotation_matrix = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1],
    ])
    rotated_corners = scaled_corners @ rotation_matrix.T

    # Apply translation
    corners = rotated_corners + np.array([position_x, position_y, position_z])

    # Define the 12 edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # back face
        (4, 5), (5, 6), (6, 7), (7, 4),  # front face
        (0, 4), (1, 5), (2, 6), (3, 7),  # connecting edges
    ]

    # Create line segments
    points = np.array([(corners[i], corners[j]) for (i, j) in edges])
    colors = np.full((len(edges), 2, 3), color)

    server.scene.add_line_segments(
        path,
        points=points,
        colors=colors,
        line_width=line_width,
    )

    if not label:
        return

    label_offset = 0.1  # Small offset from the box

    if label_position == "above":
        label_pos = np.array([position_x, position_y, position_z + scale_z / 2 + label_offset])
    elif label_position == "below":
        label_pos = np.array([position_x, position_y, position_z - scale_z / 2 - label_offset])
    elif label_position == "front":
        label_pos = np.array([position_x, position_y + scale_y / 2 + label_offset, position_z])
    elif label_position == "back":
        label_pos = np.array([position_x, position_y - scale_y / 2 - label_offset, position_z])
    elif label_position == "left":
        label_pos = np.array([position_x - scale_x / 2 - label_offset, position_y, position_z])
    elif label_position == "right":
        label_pos = np.array([position_x + scale_x / 2 + label_offset, position_y, position_z])
    else:
        label_pos = np.array([position_x, position_y, position_z + scale_z / 2 + label_offset])

    server.scene.add_label(
        f"{path}/label",
        text=label,
        position=label_pos,
    )


def get_min_max_coords(
    position_x: float,
    position_y: float,
    position_z: float,
    angle_z: float,
    scale_x: float,
    scale_y: float,
    scale_z: float,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return the axis-aligned bounding extremes after rotation around Z (angle in degrees)."""
    hx, hy, hz = scale_x / 2, scale_y / 2, scale_z / 2

    corners = [
        (hx, hy, hz),
        (hx, -hy, hz),
        (-hx, hy, hz),
        (-hx, -hy, hz),
        (hx, hy, -hz),
        (hx, -hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, -hz),
    ]

    rad = math.radians(angle_z)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rotated = [
        (
            x * cos_a - y * sin_a + position_x,
            x * sin_a + y * cos_a + position_y,
            z + position_z,
        )
        for x, y, z in corners
    ]

    xs = [x for x, _, _ in rotated]
    ys = [y for _, y, _ in rotated]
    zs = [z for _, _, z in rotated]

    min_coords = (min(xs), min(ys), min(zs))
    max_coords = (max(xs), max(ys), max(zs))

    return min_coords, max_coords