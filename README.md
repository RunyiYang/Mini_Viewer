# Mini Viewer

Realtime Gaussian splat viewer for `.ply` and NumPy splat folders, with CUDA/CPU rendering, SigLIP2/CLIP language-feature querying, queried-feature bounding boxes, and Nerfstudio-style camera-path export/video rendering.

Mini Viewer is designed for large 3DGS scenes where the normal path is CUDA + `gsplat`, but the same environment can still run a CPU/torch fallback for debugging, servers without a visible GPU, or rerender fallback when CUDA rasterization fails.

![Mini Viewer](docs/mini_viewer.png)

## What is included

- Load Gaussian splats from `.ply` or NumPy folders.
- Load language features from `.npy`, `.npz`, `.pt`, or `.pth`.
- Query language features with SigLIP2 by default: `google/siglip2-so400m-patch16-512`.
- Visualize feature maps, RGB, depth, and normals.
- Toggle queried-feature bounding boxes and export them as JSON.
- Add camera keyframes in the viewer and export a Nerfstudio-style `camera_path.json`.
- Render camera-path videos from the GUI or with a headless script.
- Use CUDA/`gsplat` when available, with optional CPU rerender fallback.

## Repository dependency files

The repo should keep only these project-level dependency/config files:

```text
README.md
requirements.txt
env.yml
pyproject.toml
```

Older split files such as `requirements-cpu.txt`, `requirements-cuda124.txt`, `requirements-language.txt`, and `environment-mini-viewer-*.yml` are intentionally replaced by the single full-stack environment below.

## Environment setup

Use Python 3.10. The full environment installs a CUDA 12.4 PyTorch build, `gsplat`, the viewer stack, and language-query dependencies. It can still run CPU rendering via `--device cpu --backend torch`.

From the repo root:

```bash
conda env remove -n mini-viewer -y || true
conda env create -f env.yml
conda activate mini-viewer

# The bbox helper is an editable local package in this repo.
pip install -e ./viser_bbox
```

Verify the install:

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch cuda build:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())

try:
    import gsplat
    print('gsplat: OK')
except Exception as exc:
    print('gsplat import failed:', repr(exc))

try:
    import transformers
    import open_clip
    print('language deps: OK')
except Exception as exc:
    print('language deps failed:', repr(exc))
PY
```

### Optional pip-only install

`requirements.txt` contains the same full-stack pip dependency set used by `env.yml`. Use it only when you are not creating the Conda environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ./viser_bbox
```

## Quick start

### CUDA + gsplat, normal path

```bash
python run_viewer.py \
  --folder-npy /path/to/scene_folder \
  --language-feature /path/to/features.npy \
  --device cuda \
  --backend gsplat \
  --port 8080
```

Open:

```text
http://localhost:8080
```

### PLY input

```bash
python run_viewer.py \
  --ply /path/to/scene.ply \
  --device cuda \
  --backend gsplat
```

### CPU rendering from the same environment

```bash
python run_viewer.py \
  --folder-npy /path/to/scene_folder \
  --device cpu \
  --backend torch
```

### Force CPU rerender fallback while still loading CUDA/language features

Useful when `gsplat` fails during interaction, or while debugging feature maps on a large scene:

```bash
python run_viewer.py \
  --folder-npy /path/to/scene_folder \
  --language-feature /path/to/features.npy \
  --device cuda \
  --backend gsplat \
  --force-cpu-render \
  --cpu-fallback-splats 80000
```

## Data formats

### NumPy folder

Pass the folder through `--folder-npy`. The loader expects:

```text
coord.npy      # required, N x 3 xyz
quat.npy       # required, N x 4 quaternion
scale.npy      # required, N x 3 scale
opacity.npy    # required, N or N x 1 opacity
color.npy      # required, N x 3 RGB, usually 0-255 or 0-1
normal.npy     # optional, N x 3 normals
```

If your NumPy scale values are log-scales, add:

```bash
--npy-scale-log
```

### PLY

Pass an Inria/3DGS-style `.ply` through `--ply`. The loader handles standard Gaussian PLY properties such as positions, opacity, scales, rotations, and color/SH fields.

### Language features

Language features should be aligned with the splats and have shape approximately:

```text
N x D
```

Supported formats:

```text
.npy
.npz
.pt
.pth
```

Run with:

```bash
python run_viewer.py \
  --folder-npy /path/to/scene_folder \
  --language-feature /path/to/language_features.pth \
  --feature-type siglip2 \
  --device cuda
```

If you already have a precomputed text/query embedding, use it directly and avoid loading a text encoder:

```bash
python run_viewer.py \
  --folder-npy /path/to/scene_folder \
  --language-feature /path/to/point_features.npy \
  --query-feature /path/to/query_vector.npy \
  --device cpu \
  --backend torch
```

## SigLIP2 query encoder

The default text-query model is:

```text
google/siglip2-so400m-patch16-512
```

The viewer downloads it lazily on first text query. To pre-download the model once:

```bash
python scripts/download_siglip2.py \
  --cache-dir /path/to/hf_cache
```

Then run with:

```bash
python run_viewer.py \
  --folder-npy /path/to/scene_folder \
  --language-feature /path/to/features.npy \
  --hf-cache-dir /path/to/hf_cache \
  --device cuda
```

CPU text encoding is disabled by default because it is slow. To explicitly allow it:

```bash
--enable-language-on-cpu
```

## Viewer controls

The GUI exposes folders for basic rendering, language features, and camera paths.

### Basic rendering

- RGB rendering.
- Depth rendering.
- Normal rendering.
- Snapshot export.
- CPU fallback toggles for rerender failures.
- Force CPU renderer toggle.
- CPU fallback splat-count control.

### Language feature querying

- Show feature map.
- Query by text using SigLIP2/CLIP.
- Query by precomputed embedding.
- Set score/rate threshold.
- Recolor matched splats.
- Toggle queried-feature bbox.
- Export queried-feature bbox JSON.

A missing language feature file produces:

```text
[language] No language feature tensor loaded; query controls are disabled.
```

That only means `--language-feature` is missing or points to the wrong file.

### Queried-feature bbox

Workflow:

1. Start the viewer with `--language-feature`.
2. Enter a text query or load `--query-feature`.
3. Adjust the threshold/rate.
4. Toggle **Show query bbox**.
5. Press **Export query bbox**.

Default output:

```text
outputs/query_bbox.json
```

## Camera path and video rendering

### In the viewer

1. Move the camera to a desired pose.
2. Press **Add Camera**.
3. Repeat for more keyframes.
4. Press **Export Cameras**.
5. Press **Render Video**.

Default outputs:

```text
outputs/camera_path.json
outputs/render.mp4
```

### Headless render

```bash
python scripts/render_camera_path.py \
  --folder-npy /path/to/scene_folder \
  --camera-path outputs/camera_path.json \
  --output outputs/render.mp4 \
  --device cuda \
  --backend gsplat
```

CPU fallback video render:

```bash
python scripts/render_camera_path.py \
  --folder-npy /path/to/scene_folder \
  --camera-path outputs/camera_path.json \
  --output outputs/render_cpu.mp4 \
  --device cpu \
  --backend torch
```

## Useful CLI arguments

```text
--ply PATH                         Load a .ply scene.
--folder-npy PATH                  Load a NumPy splat folder.
--language-feature PATH            Load aligned language features.
--query-feature PATH               Load a precomputed query embedding.
--feature-type {siglip2,siglip,clip}
--siglip-model MODEL_ID            Default: google/siglip2-so400m-patch16-512.
--hf-cache-dir PATH                Hugging Face model cache directory.
--enable-language-on-cpu           Allow text encoder on CPU.
--bbox-script PATH                 Draw SpatialLM-style bbox script through viser_bbox.
--device {auto,cuda,cpu}           Default: auto.
--backend {auto,gsplat,torch}      Default: auto.
--max-splats N                     Optional global splat downsample while loading.
--max-cpu-splats N                 CPU renderer splat cap. Default: 180000.
--cpu-render-fallback              Retry failed CUDA rerenders on CPU. Enabled by default.
--no-cpu-render-fallback           Disable automatic CPU retry.
--cpu-fallback-splats N            CPU retry cap. Default: 80000.
--force-cpu-render                 Force all viewer rerenders through CPU torch renderer.
--npy-scale-log                    Treat NumPy scale arrays as log-scales.
--camera-path PATH                 Camera-path JSON path. Default: outputs/camera_path.json.
--video-output PATH                GUI video output path. Default: outputs/render.mp4.
--render-width N                   GUI video width. Default: 1280.
--render-height N                  GUI video height. Default: 720.
--render-fps N                     GUI video FPS. Default: 30.
--render-seconds SEC              GUI video duration. Default: 5.0.
--port PORT                        Viser port. Default: 8080.
```

Both hyphen and underscore aliases are accepted for the patched options, for example `--folder-npy` and `--folder_npy`.

## ScanNet++ style example

```bash
python run_viewer.py \
  --folder-npy /work/runyi_yang/Worldcept/example/scannetpp_v2_mcmc_3dgs_lang_large/val/09c1414f1b \
  --language-feature /path/to/features.npy \
  --hf-cache-dir /work/runyi_yang/hf_cache \
  --device cuda \
  --backend gsplat \
  --cpu-render-fallback \
  --cpu-fallback-splats 80000 \
  --port 8080
```

If interaction is unstable on this very large scene, force CPU rerendering:

```bash
python run_viewer.py \
  --folder-npy /work/runyi_yang/Worldcept/example/scannetpp_v2_mcmc_3dgs_lang_large/val/09c1414f1b \
  --language-feature /path/to/features.npy \
  --device cuda \
  --backend gsplat \
  --force-cpu-render \
  --cpu-fallback-splats 80000
```

## Troubleshooting

### `No language feature tensor loaded`

Pass a valid feature file:

```bash
--language-feature /path/to/features.npy
```

To find candidates:

```bash
find /path/to/scene_folder \
  -maxdepth 2 \
  \( -name '*.npy' -o -name '*.npz' -o -name '*.pth' -o -name '*.pt' \) \
  | sort
```

### `gsplat` render error or CUDA fallback message

Keep automatic fallback enabled:

```bash
--cpu-render-fallback --cpu-fallback-splats 80000
```

Or force CPU rendering:

```bash
--force-cpu-render
```

### CPU-only machine

Use the same environment if it installs successfully, then run:

```bash
python run_viewer.py --ply /path/to/scene.ply --device cpu --backend torch
```

If the `gsplat` wheel is unavailable on a CPU-only platform, remove or comment this line from `requirements.txt` and `env.yml`:

```text
gsplat==1.5.3+pt24cu124
```

Then recreate the env and use `--device cpu --backend torch`.

### Hugging Face download/cache issues

Set an explicit cache directory:

```bash
export HF_HOME=/path/to/hf_cache
python scripts/download_siglip2.py --cache-dir /path/to/hf_cache
```

Then pass:

```bash
--hf-cache-dir /path/to/hf_cache
```

## Development

- Main entry point: `run_viewer.py`.
- Splat loading and CPU master copy: `core/splat.py`.
- CUDA/torch/CPU-fallback rendering: `core/renderer.py`.
- Viser/Nerfview integration: `core/viewer.py`.
- Basic controls: `actions/base.py`.
- Language query and bbox controls: `actions/language_feature.py`.
- Camera-path controls: `actions/camera_path.py`.
- SigLIP2/CLIP text embedding: `models/clip_query.py`.
- Headless camera-path rendering: `scripts/render_camera_path.py`.
- SigLIP2 pre-download helper: `scripts/download_siglip2.py`.

Run a syntax check:

```bash
python -m compileall run_viewer.py actions core models scripts utils
```

## Acknowledgements

- Nerfview for interactive viewer scaffolding.
- Viser for the WebGL frontend.
- GSplat for CUDA Gaussian rasterization.
- The in-repo `viser_bbox` utilities for bounding-box overlays.

## Citations

If you use Mini Viewer in research, please consider citing:

```bibtex
@article{wu2023mars,
  author    = {Wu, Zirui and Liu, Tianyu and Luo, Liyi and Zhong, Zhide and Chen, Jianteng and Xiao, Hongmin and Hou, Chao and Lou, Haozhe and Chen, Yuantao and Yang, Runyi and Huang, Yuxin and Ye, Xiaoyu and Yan, Zike and Shi, Yongliang and Liao, Yiyi and Zhao, Hao},
  title     = {MARS: An Instance-aware, Modular and Realistic Simulator for Autonomous Driving},
  journal   = {CICAI},
  year      = {2023}
}

@misc{yang2024spectrally,
  title         = {Spectrally Pruned Gaussian Fields with Neural Compensation},
  author        = {Runyi Yang and Zhenxin Zhu and Zhou Jiang and Baijun Ye and Xiaoxue Chen and Yifei Zhang and Yuantao Chen and Jian Zhao and Hao Zhao},
  year          = {2024},
  eprint        = {2405.00676},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@article{zheng2024gaussiangrasper,
  title     = {Gaussiangrasper: 3d language gaussian splatting for open-vocabulary robotic grasping},
  author    = {Zheng, Yuhang and Chen, Xiangyu and Zheng, Yupeng and Gu, Songen and Yang, Runyi and Jin, Bu and Li, Pengfei and Zhong, Chengliang and Wang, Zengmao and Liu, Lina and others},
  journal   = {IEEE Robotics and Automation Letters},
  year      = {2024},
  publisher = {IEEE}
}

@article{li2025scenesplat,
  title   = {SceneSplat: Gaussian Splatting-based Scene Understanding with Vision-Language Pretraining},
  author  = {Li, Yue and Ma, Qi and Yang, Runyi and Li, Huapeng and Ma, Mengjiao and Ren, Bin and Popovic, Nikola and Sebe, Nicu and Konukoglu, Ender and Gevers, Theo and others},
  journal = {arXiv preprint arXiv:2503.18052},
  year    = {2025}
}
```
