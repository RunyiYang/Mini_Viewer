# Mini Viewer CUDA/CPU cleanup patch

This patch replaces the main viewer path with a cleaner CPU/CUDA split and adds requested feature workflows.

## What changed

- Added CPU-safe dependency files and CUDA 12.4 dependency files.
- Made `gsplat` optional and lazy: CUDA uses `gsplat`, CPU uses a torch/numpy point-splat fallback.
- Made SigLIP/CLIP optional and lazy: text encoders are only loaded for language querying.
- Added robust PLY and NumPy-folder loading.
- Added `.npy`, `.npz`, `.pt`, and `.pth` language/query feature loading.
- Added queried-feature bbox visualization and JSON export.
- Added camera placement, camera-path export, and MP4 rendering in a Nerfstudio-style JSON format.
- Added a headless camera-path rendering script.
- Cleaned imports and separated renderer/data/viewer/actions.

## Apply

From the root of `RunyiYang/Mini_Viewer`:

```bash
bash apply_patch.sh
```

The script backs up replaced files into `.mini_viewer_patch_backup_<timestamp>/`.

## Install

CPU:

```bash
pip install -r requirements-cpu.txt
pip install -e ./viser_bbox
```

CUDA 12.4 + language:

```bash
pip install -r requirements-cuda124.txt
pip install -r requirements-language.txt
pip install -e ./viser_bbox
```

## Run

```bash
python run_viewer.py --ply scene.ply --device auto
python run_viewer.py --folder-npy data/scene --language-feature features.npy --device cuda
```

Headless video:

```bash
python scripts/render_camera_path.py --ply scene.ply --camera-path outputs/camera_path.json --output outputs/render.mp4 --device cuda
```
