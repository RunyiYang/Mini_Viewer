# Mini Viewer environment

This patch splits dependencies into a CPU-safe path and a CUDA 12.4 path. Use Python 3.11.

## CPU-only install

```bash
micromamba create -n mini-viewer python=3.11 -y
micromamba activate mini-viewer
pip install -r requirements-cpu.txt
pip install -e ./viser_bbox
```

Run without CUDA:

```bash
python run_viewer.py --ply /path/to/scene.ply --device cpu --backend torch
python run_viewer.py --folder-npy /path/to/npy_folder --device cpu --backend torch
```

CPU rendering is an approximate point-splat fallback. It is meant for inspection, data loading, camera placement, and debugging, not final-quality 3DGS rendering.

## CUDA 12.4 install

```bash
micromamba create -n mini-viewer python=3.11 -y
micromamba activate mini-viewer
pip install -r requirements-cuda124.txt
pip install -r requirements-language.txt
pip install -e ./viser_bbox
```

Run with CUDA + language querying:

```bash
python run_viewer.py \
  --ply /path/to/scene.ply \
  --language-feature /path/to/language_features.pth \
  --feature-type siglip \
  --device cuda \
  --backend gsplat
```

## Data formats

### PLY

The PLY loader accepts Inria/3DGS-style properties:

- `x,y,z`
- `scale_0,scale_1,scale_2` as log scales
- `rot_0,rot_1,rot_2,rot_3`
- `opacity` as opacity or logit
- `f_dc_0,f_dc_1,f_dc_2` or RGB fields

### NumPy folder

The folder loader accepts common aliases:

- coordinates: `coord.npy`, `coords.npy`, `means.npy`, `xyz.npy`
- quaternions: `quat.npy`, `quats.npy`, `rotation.npy`, `rot.npy`
- scales: `scale.npy`, `scales.npy`, `scaling.npy`
- opacity: `opacity.npy`, `opacities.npy`, `alpha.npy`
- color: `color.npy`, `colors.npy`, `rgb.npy`
- optional normals: `normal.npy`, `normals.npy`
- optional feature-valid mask: `valid_feat_mask.npy`

Use `--npy-scale-log` if your NumPy scale files are stored as log-scales.

### Language features

Language features can be `.npy`, `.npz`, `.pt`, or `.pth`. The loader supports raw tensors/arrays, tuples/lists, and dictionaries with keys such as `language_feature`, `features`, `embeddings`, `clip`, or `siglip`.

Text encoders are lazy-loaded. On CPU the viewer does not load SigLIP/CLIP unless `--enable-language-on-cpu` is passed. You can also bypass text encoders with a precomputed query vector:

```bash
python run_viewer.py \
  --folder-npy /path/to/npy_folder \
  --language-feature /path/to/point_features.npy \
  --query-feature /path/to/query_vector.npy \
  --device cpu
```

## Query bbox

After running a text/query feature search, toggle **Show query bbox** in the GUI. The bbox is computed from splats whose normalized cosine score is above the current threshold. Press **Export query bbox** to write:

```text
outputs/query_bbox.json
```

## Camera path and rendering

In the GUI:

1. Place the viewport camera.
2. Press **Add Camera** for each keyframe.
3. Set width, height, FPS, and seconds.
4. Press **Export Cameras** to write a Nerfstudio-style `camera_path.json`.
5. Press **Render Video** to render the MP4.

Headless rendering:

```bash
python scripts/render_camera_path.py \
  --ply /path/to/scene.ply \
  --camera-path outputs/camera_path.json \
  --output outputs/render.mp4 \
  --device cuda
```
