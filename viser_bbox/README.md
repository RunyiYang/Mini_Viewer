# viser-bbox

Utilities for building interactive Viser scenes populated with Gaussian splats, point clouds, and scripted bounding boxes. The package bundles helpers for loading 3D assets from PLY files, drawing labeled wireframe boxes, and parsing a simple domain-specific language for rooms, doors, windows, and generic objects.

## Features
- Load Gaussian splat datasets produced by common 3D reconstruction pipelines and feed them into `viser.ViserServer`.
- Render point clouds and axis-aligned wireframe boxes with optional labels and styling.
- Parse a compact text script to procedurally create walls, doors, windows, and arbitrary bounding boxes.
- Ship a ready-to-run demo scene that combines all of the above.

## Installation
The package depends on `viser`, `numpy`, and `plyfile`. Install them into a virtual environment of your choice:

```bash
python -m venv .venv
source .venv/bin/activate
pip install viser numpy plyfile
```

Alternatively, install the project in editable mode (recommended for development):

```fish
python -m venv .venv
source .venv/bin/activate
cd /path/to/parent-of-viser_bbox  # the directory that contains the `viser_bbox` folder
python -m pip install -e ./viser_bbox
```

If you don't want to install, run example scripts from the repository root or export `PYTHONPATH` so Python can find the package:

```fish
cd /path/to/parent-of-viser_bbox
env PYTHONPATH=$PWD python -m viser_bbox.vis
```

## Running the demo

The demo now accepts file paths as command-line arguments. You can launch Viser with any combination of Gaussian splats, point cloud, and bounding box script files:

```fish
python -m viser_bbox.vis --gaussian_ply /path/to/gaussian_source --point_cloud_ply /path/to/pointcloud_source --bbox_script /path/to/bboxes.txt
```

All arguments are optional:
- `--gaussian_ply` (optional): path to Gaussian splats PLY file or directory of NumPy arrays (`coord.npy`, etc.)
- `--point_cloud_ply` (optional): path to point cloud PLY file or directory of NumPy arrays (`coords.npy`, `colors.npy`)
- `--bbox_script` (optional): path to bounding box script text file

Examples:
- Only gaussians:
  ```fish
  python -m viser_bbox.vis --gaussian_ply /path/to/gaussian_source
  ```
- Only point cloud:
  ```fish
  python -m viser_bbox.vis --point_cloud_ply /path/to/pointcloud_source
  ```
- Both:
  ```fish
  python -m viser_bbox.vis --gaussian_ply /path/to/gaussian_source --point_cloud_ply /path/to/pointcloud_source
  ```
- Add bounding box script:
  ```fish
  python -m viser_bbox.vis --bbox_script /path/to/bboxes.txt
  ```

The demo will launch Viser with whatever data you provide.

## API overview

```python
from pathlib import Path
import viser_bbox as vb

server, handles = vb.create_vis_scene(
    gaussian_ply=Path("your_gaussians.ply"),
    point_cloud_ply=Path("your_cloud.ply"),
    bbox_kwargs=dict(
        position_x=0.0,
        position_y=0.0,
        position_z=1.0,
        angle_z=0.0,
        scale_x=1.0,
        scale_y=1.0,
        scale_z=2.0,
        label="Center box",
    ),
)
```

- `viser_bbox.load_ply_file` converts a Gaussian splat PLY file into arrays ready for `ViserServer.scene.add_gaussian_splats`.
- `viser_bbox.load_gaussian_folder` loads the same arrays from a directory of NumPy files (`coord.npy`, `color.npy`, `quat.npy`, `scale.npy`, `opacity.npy`).
- `viser_bbox.load_vertices_and_colors` extracts point positions and RGB colors from a point-cloud PLY.
- `viser_bbox.add_bounding_box` draws labeled boxes using wireframe line segments; use it directly for custom interactions.
- `viser_bbox.add_script_bboxes` parses wall/door/window/bbox definitions and injects them into a server.

Refer to `vis.create_vis_scene` for a full, composable example.

## PLY format (non-standard properties)

Note: this project expects a non-standard PLY layout for Gaussian splat files. The reader functions rely on specific per-vertex property names and shapes. If your PLY files use a different schema, convert or rename properties before loading.

If your splats are stored as NumPy arrays instead of a PLY, place `coord.npy`, `color.npy`, `quat.npy`, `scale.npy`, and `opacity.npy` in a directory and load them with `load_gaussian_folder`.

If your point cloud lives in NumPy arrays, place `coords.npy` and `colors.npy` in a directory and load them with `load_point_cloud_folder`.

1) Gaussian splat PLY (used by `load_ply_file`)

- Required vertex properties (exact names):
  - `x`, `y`, `z` : float - vertex center position
  - `scale_0`, `scale_1`, `scale_2` : float - log-scale components (the code applies exp and squares them)
  - `rot_0`, `rot_1`, `rot_2`, `rot_3` : float - quaternion components (w,x,y,z order expected)
  - `f_dc_0`, `f_dc_1`, `f_dc_2` : float - decorrelated color coefficients (converted to RGB via a constant)
  - `opacity` : float - per-vertex opacity (sigmoid applied)

- Interpretation notes:
  - `scale_*` values are exponentiated and squared in order to form diagonal scale matrices used to compute a 3x3 covariance per splat.
  - Quaternion ordering is stacked as `[rot_0, rot_1, rot_2, rot_3]` and passed to `quaternion_to_rotation_matrix`.
  - Colors are reconstructed from the `f_dc_*` channels using a spherical-harmonic constant; they are not standard 0-255 RGB channels.

- Minimal example PLY header for gaussian splats (binary or ascii PLY following these properties):

```
ply
format ascii 1.0
element vertex N
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
end_header
```

2) Point cloud PLY (used by `load_vertices_and_colors`)

- Expected vertex properties (exact names):
  - `x`, `y`, `z` : float - point position
  - `red`, `green`, `blue` : uchar or float - per-vertex color channels

- Minimal example PLY header for a point cloud:

```
ply
format ascii 1.0
element vertex N
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
```

If your files use different property names (for example `r,g,b` or `nx,ny,nz` for normals), either rename the properties or write a small converter script that reads the original PLY and writes a new file with the expected property names. The `plyfile` package can be helpful for such conversions.

## Script format reference

The bounding box script uses the SpatialLM format. Here is a minimal example:

```
wall0 = Wall(0, 0, 0, 4, 0, 0, 3.0, 0.18)
door0 = Door(wall0, 2.0, 0.0, 0.0, 0.9, 2.1)
bbox0 = Bbox(Sink, 1.5, -0.1, 0.8, 0.0, 0.6, 0.5, 0.9)
```

For more details, see the SpatialLM documentation or examples.

- `Wall(start_x, start_y, start_z, end_x, end_y, end_z, height, thickness?)`
- `Door(wall_name, center_x, center_y, base_z, width, height)`
- `Window(wall_name, center_x, center_y, center_z, width, height)`
- `Bbox(label, center_x, center_y, center_z, yaw_radians, size_x, size_y, size_z)`

Angles are interpreted in radians. Omitted wall thickness falls back to the `default_wall_thickness` parameter (0.15 by default). Opening thickness for doors and windows is `default_opening_thickness` (0.12) unless overridden.

## Development tips
- Enable live reload while experimenting with the demo by keeping the Python process running; Viser updates in place as you toggle geometry.
- The helper `get_min_max_coords` computes axis-aligned bounds for post-rotation boxes, which is useful for spatial indexing or culling.
- Gaussian splat loading prints a quick timing summary; large `.ply` files can take several seconds.
