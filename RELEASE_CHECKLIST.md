# Release checklist

Use this checklist before tagging a public release.

## Static checks

```bash
python scripts/validate_release.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
ruff check . --select E9,F63,F7,F82
git diff --check
```

## Install checks

```bash
conda env remove -n mini-viewer -y || true
conda env create -f env.yml
conda activate mini-viewer
pip install -e .
pip install -e ./viser_bbox
```

## Runtime smoke checks

CPU loader/render path:

```bash
python scripts/smoke_test_loaders.py \
  --folder-npy /path/to/scene_folder \
  --feature-file /path/to/features.npy \
  --device cpu

python run_viewer.py \
  --folder-npy /path/to/scene_folder \
  --feature-file /path/to/features.npy \
  --device cpu \
  --backend torch
```

CUDA SigLIP2 path:

```bash
python run_viewer.py \
  --folder-npy /path/to/scene_folder \
  --feature-file /path/to/siglip2_features.npy \
  --feature-type siglip2 \
  --device cuda \
  --backend gsplat \
  --cpu-render-fallback
```

CUDA DINOv2 path:

```bash
python run_viewer.py \
  --folder-npy /path/to/scene_folder \
  --dino-feature /path/to/dino_features.npy \
  --feature-type dinov2 \
  --query-image /path/to/query_crop.png \
  --device cuda \
  --backend gsplat \
  --cpu-render-fallback
```

Camera path render:

```bash
python scripts/render_camera_path.py \
  --folder-npy /path/to/scene_folder \
  --camera-path outputs/camera_path.json \
  --output outputs/render.mp4 \
  --device cuda \
  --backend gsplat
```

## Repository checks

- Add an explicit `LICENSE` chosen by the project owner.
- Ensure no `.npy`, `.pth`, `.ply`, `.mp4`, outputs, caches, or patch backup folders are committed.
- Set GitHub description and topics.
- Close or triage open issues.
- Create and push an annotated tag.

```bash
git tag -a v0.3.0 -m "Mini Viewer v0.3.0"
git push origin v0.3.0
```

Suggested release notes:

```text
- Added DINOv2 image-query support for aligned visual features.
- Added generic feature-file loading for SigLIP2, CLIP, DINOv2, and custom embeddings.
- Added CPU/CUDA render fallback controls.
- Added queried-feature bbox export and camera-path video rendering.
- Consolidated release metadata into README.md, env.yml, requirements.txt, and pyproject.toml.
```
