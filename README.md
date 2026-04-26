# Mini Viewer

Realtime Gaussian-splat inspector with CUDA/CPU rendering, SigLIP2/CLIP text querying, DINOv2 image-feature querying, queried-feature bounding boxes, and Nerfstudio-style camera-path video export.

Mini Viewer loads Gaussian scenes from `.ply` files or NumPy folders, overlays bbox scripts through `viser_bbox`, visualizes aligned feature tensors, queries those tensors with text, image prompts, or precomputed vectors, exports query bounding boxes, and renders camera paths to MP4.

![Mini Viewer](docs/mini_viewer.png)

## Release feature set

- **Scene loading:** Inria/3DGS-style `.ply` files or NumPy folders with `coord.npy`, `quat.npy`, `scale.npy`, `opacity.npy`, and `color.npy`.
- **CUDA renderer:** `gsplat` backend with PyTorch CUDA 12.4 wheels.
- **CPU renderer:** torch/numpy fallback backend for CPU-only runs, forced CPU rerendering, or failed CUDA rerenders.
- **Feature tensors:** aligned `.npy`, `.npz`, `.pt`, or `.pth` tensors for SigLIP, CLIP, DINO, or custom embeddings.
- **SigLIP2 text queries:** default text encoder is `google/siglip2-so400m-patch16-512`.
- **CLIP text queries:** optional OpenCLIP path through `--feature-type clip`.
- **DINOv2 image queries:** default visual encoder is `facebook/dinov2-base`; queries use an image path or a precomputed vector.
- **Feature maps:** PCA/preview recoloring of high-dimensional feature tensors.
- **Query bbox:** threshold the query score, toggle bbox visibility, and export `outputs/query_bbox.json`.
- **Camera paths:** place cameras, export a Nerfstudio-style `camera_path.json`, and render MP4 video.
- **Release checks:** static validation script, pytest smoke test, GitHub Actions CI, `.gitattributes`, and `.gitignore`.

## Repository layout

```text
README.md                   Main setup and usage guide.
env.yml                     Full Conda environment: CUDA 12.4, gsplat, viewer, query encoders.
requirements.txt            Pip equivalent of env.yml.
pyproject.toml              Project metadata, console script, Ruff, and pytest config.
CHANGELOG.md                Release notes.
RELEASE_CHECKLIST.md         Manual release checklist.
run_viewer.py               Main viewer CLI.
core/splat.py               PLY/NumPy/aligned-feature loading.
core/renderer.py            CUDA/torch/CPU-fallback rendering.
core/viewer.py              nerfview/viser integration.
actions/language_feature.py Generic SigLIP/CLIP/DINO/query-vector feature UI.
actions/camera_path.py      Camera placement, export, and GUI video rendering.
models/clip_query.py        Lazy SigLIP2, CLIP, and DINOv2 query encoders.
scripts/download_siglip2.py SigLIP2 cache helper.
scripts/download_dino.py    DINOv2 cache helper.
scripts/render_camera_path.py Headless camera-path renderer.
scripts/smoke_test_loaders.py Data-loader smoke test.
scripts/validate_release.py Static release validator.
```

Old split files such as `requirements-common.txt`, `requirements-cpu.txt`, `requirements-cuda124.txt`, `requirements-language.txt`, `environment-mini-viewer-*.yml`, and `README_patch.md` are obsolete.

## Install from scratch with Conda

Use this path for a full CUDA-capable workstation/server environment. The same environment can still run CPU mode.

```bash
git clone git@github.com:RunyiYang/Mini_Viewer.git
cd Mini_Viewer

conda env remove -n mini-viewer -y || true
conda env create -f env.yml
conda activate mini-viewer

pip install -e .
pip install -e ./viser_bbox
```

Validate the installation:

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
    print('query encoder deps: OK')
except Exception as exc:
    print('query encoder deps failed:', repr(exc))
PY
```

## Pip-only install

Conda is preferred because it installs `ffmpeg`, but a pip-only setup is available:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -e ./viser_bbox
```

## CPU-only hosts

`env.yml` and `requirements.txt` use the CUDA 12.4 PyTorch wheel as the superset setup. CPU rendering still works from that environment:

```bash
python run_viewer.py \
  --ply /path/to/scene.ply \
  --device cpu \
  --backend torch
```

On a machine where the CUDA `gsplat` wheel cannot be installed, remove or comment out this line in `requirements.txt` or `env.yml`:

```text
gsplat==1.5.3+pt24cu124
```

Then run only with:

```bash
--device cpu --backend torch
```

## Input data format

### NumPy scene folder

Required files:

```text
coord.npy      float array, shape (N, 3)
quat.npy       float array, shape (N, 4)
scale.npy      float array, shape (N, 3)
opacity.npy    float array, shape (N,) or (N, 1)
color.npy      RGB/color array, float [0,1] or uint8 [0,255]
```

Optional files:

```text
normal.npy
valid_feat_mask.npy
features.npy / features.npz / features.pt / features.pth
```

Use `--npy-scale-log` if NumPy `scale.npy` values are log-scales and should be exponentiated.

### PLY file

The PLY loader expects Inria/3DGS-style vertex properties such as:

```text
x, y, z
scale_0, scale_1, scale_2
rot_0, rot_1, rot_2, rot_3
f_dc_0, f_dc_1, f_dc_2
opacity
```

### Aligned feature files

Use `--feature-file`, `--language-feature`, or `--dino-feature` to load an aligned feature tensor. The loader accepts `.npy`, `.npz`, `.pt`, and `.pth` and will search common keys such as:

```text
features, feature, embeddings, language_feature, clip_features,
siglip_features, dino_features, dinov2_features, image_features, visual_features
```

The first dimension must match the loaded splat count, the original splat count before masks/downsampling, or the `valid_feat_mask.npy` count.

## Run the viewer

### PLY scene, automatic backend

```bash
python run_viewer.py \
  --ply /path/to/scene.ply \
  --device auto \
  --backend auto \
  --port 8080
```

Open:

```text
http://localhost:8080
```

### NumPy scene with CUDA rendering

```bash
python run_viewer.py \
  --folder-npy /path/to/scene_folder \
  --device cuda \
  --backend gsplat \
  --port 8080
```

### SigLIP2 text-query features

```bash
python run_viewer.py \
  --folder-npy /work/runyi_yang/Worldcept/example/scannetpp_v2_mcmc_3dgs_lang_large/val/09c1414f1b \
  --feature-file /path/to/siglip2_features.npy \
  --feature-type siglip2 \
  --siglip-model google/siglip2-so400m-patch16-512 \
  --hf-cache-dir /work/runyi_yang/hf_cache \
  --device cuda \
  --backend gsplat
```

The backward-compatible alias also works:

```bash
--language-feature /path/to/siglip2_features.npy
```

### DINOv2 image-query features

DINOv2 is visual-only: query it with an image path or a precomputed query vector, not text.

```bash
python run_viewer.py \
  --folder-npy /path/to/scene_folder \
  --dino-feature /path/to/dino_features.npy \
  --feature-type dinov2 \
  --query-image /path/to/query_crop.png \
  --dino-model facebook/dinov2-base \
  --hf-cache-dir /work/runyi_yang/hf_cache \
  --device cuda \
  --backend gsplat
```

Inside the viewer, paste another image path into **Text prompt / DINO image path** and press **Query text / image / feature**.

### Query with a precomputed vector

This works for SigLIP2, CLIP, DINOv2, or any custom aligned feature space:

```bash
python run_viewer.py \
  --folder-npy /path/to/scene_folder \
  --feature-file /path/to/point_features.npy \
  --query-feature /path/to/query_vector.npy \
  --feature-type dinov2 \
  --device cpu \
  --backend torch
```

### Force CPU rerendering while keeping CUDA data loading

```bash
python run_viewer.py \
  --folder-npy /path/to/scene_folder \
  --feature-file /path/to/features.npy \
  --device cuda \
  --backend gsplat \
  --force-cpu-render \
  --cpu-fallback-splats 80000
```

### Disable automatic CPU fallback

CPU fallback is enabled by default. Disable it only when CUDA errors should fail loudly:

```bash
python run_viewer.py \
  --folder-npy /path/to/scene_folder \
  --device cuda \
  --backend gsplat \
  --no-cpu-render-fallback
```

## Query model cache helpers

Pre-download SigLIP2:

```bash
python scripts/download_siglip2.py \
  --cache-dir /work/runyi_yang/hf_cache
```

Pre-download DINOv2:

```bash
python scripts/download_dino.py \
  --model facebook/dinov2-base \
  --cache-dir /work/runyi_yang/hf_cache
```

CPU model-backed queries are disabled by default because they are slow. Enable them only for debugging:

```bash
--enable-feature-model-on-cpu
```

## Feature Query GUI

The **Feature Query** folder exposes:

1. **Feature Map**: PCA/preview recoloring of loaded aligned features.
2. **Reset RGB**: return to RGB splat colors.
3. **Normal Map**: visualize normals.
4. **Text prompt / DINO image path**: text for SigLIP/CLIP, image path for DINOv2.
5. **Threshold**: score cutoff for recolor/prune/bbox.
6. **Query text / image / feature**: run cosine query.
7. **Prune by Query**: show only selected splats.
8. **Show query bbox**: draw bbox over selected splats.
9. **Export query bbox**: write `outputs/query_bbox.json`.

## Camera paths and video export

In the viewer:

1. Move the camera to the desired pose.
2. Press **Add Camera**.
3. Repeat for all keyframes.
4. Press **Export Cameras**.
5. Press **Render Video**.

Default outputs:

```text
outputs/camera_path.json
outputs/render.mp4
```

Headless render command:

```bash
python scripts/render_camera_path.py \
  --ply /path/to/scene.ply \
  --camera-path outputs/camera_path.json \
  --output outputs/render.mp4 \
  --device cuda \
  --backend gsplat
```

CPU fallback video rendering:

```bash
python scripts/render_camera_path.py \
  --folder-npy /path/to/scene_folder \
  --camera-path outputs/camera_path.json \
  --output outputs/render_cpu.mp4 \
  --device cpu \
  --backend torch
```

### Render feature maps from saved camera poses

The headless renderer can use the same exported camera pose/path to render feature-space outputs instead of RGB. The projection is defined by the camera JSON; the rendered colors are selected by `--render-layer`.

Supported layers:

```text
rgb          Original splat RGB.
depth        Expected-depth visualization from the same pose.
normal       Per-splat normal colors.
feature      PCA/preview colors from the aligned feature tensor.
query-score  Heatmap of cosine score against a text/image/vector query.
query-mask   Binary threshold mask from the same score.
```

Render one PNG from the current camera saved by **Save current camera**:

```bash
python scripts/render_camera_path.py \
  --folder-npy /path/to/scene_folder \
  --feature-file /path/to/features.npy \
  --render-layer feature \
  --camera-state .tmp/camera_state.json \
  --output outputs/feature_map.png \
  --device cuda \
  --backend gsplat
```

Render a feature-map video from an exported camera path:

```bash
python scripts/render_camera_path.py \
  --folder-npy /path/to/scene_folder \
  --feature-file /path/to/features.npy \
  --render-layer feature \
  --camera-path outputs/camera_path.json \
  --output outputs/feature_map.mp4 \
  --device cuda \
  --backend gsplat
```

Render a SigLIP2 text-query score map from the camera path:

```bash
python scripts/render_camera_path.py \
  --folder-npy /path/to/scene_folder \
  --feature-file /path/to/siglip2_features.npy \
  --feature-type siglip2 \
  --query-text "chair" \
  --render-layer query-score \
  --camera-path outputs/camera_path.json \
  --output outputs/chair_score.mp4 \
  --score-output outputs/chair_scores.npy \
  --device cuda
```

Render a DINOv2 image-query mask from a camera pose:

```bash
python scripts/render_camera_path.py \
  --folder-npy /path/to/scene_folder \
  --dino-feature /path/to/dino_features.npy \
  --feature-type dinov2 \
  --query-image /path/to/query_crop.png \
  --render-layer query-mask \
  --threshold 0.55 \
  --camera-state .tmp/camera_state.json \
  --output outputs/dino_query_mask.png \
  --device cuda
```

For exact frame-by-frame inspection, add:

```bash
--frame-output-dir outputs/feature_frames
```

## Static bbox scripts with `viser_bbox`

Install once:

```bash
pip install -e ./viser_bbox
```

Run with a bbox script:

```bash
python run_viewer.py \
  --folder-npy /path/to/scene_folder \
  --bbox-script docs/bboxes/demo.txt \
  --device cuda
```

Example script syntax:

```text
wall0 = Wall(-2, -2, 0, 2, -2, 0, 3.0, 0.18)
door0 = Door(wall0, 0.0, -2.0, 1.0, 0.9, 2.1)
bbox0 = Bbox(Sofa, 0.8, 0.3, 0.7, 0.0, 1.5, 0.9, 1.0)
```

## Useful CLI options

```text
--ply PATH                         Load a Gaussian PLY.
--folder-npy PATH                  Load NumPy splat arrays.
--feature-file PATH                Load aligned feature tensor.
--language-feature PATH            Backward-compatible alias for --feature-file.
--dino-feature PATH                Backward-compatible DINO alias for --feature-file.
--feature-type {siglip2,siglip,clip,dino,dinov2,dino2}
--query-feature PATH               Load precomputed query embedding.
--query-image PATH                 Initial DINO/DINOv2 image query.
--siglip-model MODEL_ID            Default: google/siglip2-so400m-patch16-512.
--dino-model MODEL_ID              Default: facebook/dinov2-base.
--hf-cache-dir PATH                Hugging Face cache path.
--device {auto,cuda,cpu}
--backend {auto,gsplat,torch}
--cpu-render-fallback              Retry failed CUDA frames on CPU; enabled by default.
--no-cpu-render-fallback           Disable automatic CPU fallback.
--cpu-fallback-splats INT          Downsample cap for CPU fallback frames.
--force-cpu-render                 Render all viewer frames through CPU torch backend.
--max-cpu-splats INT               CPU renderer downsample cap.
--enable-feature-model-on-cpu      Allow SigLIP/CLIP/DINO encoders on CPU.
--camera-path PATH                 Camera-path JSON output/input path.
--camera-state PATH                Headless render from one saved camera pose.
--render-layer LAYER               Headless layer: rgb/depth/normal/feature/query-score/query-mask.
--query-text TEXT                  Headless SigLIP/CLIP text query for score maps.
--score-output PATH                Save normalized per-splat query scores as .npy.
--frame-output-dir PATH            Also save rendered frames as PNGs.
--video-output PATH                GUI video output path.
--render-width INT
--render-height INT
--render-fps INT
--render-seconds FLOAT
--bbox-script PATH
--npy-scale-log
--port INT
```

Both hyphen and underscore aliases are accepted for patched options, for example `--folder-npy` and `--folder_npy`.

## Troubleshooting

### `No aligned feature tensor loaded; query controls are disabled.`

Pass a valid feature file:

```bash
--feature-file /path/to/features.npy
```

The feature tensor must align with the loaded splats after masks/downsampling.

### DINO query says it needs an image path

DINOv2 is not a text model. Use one of these:

```bash
--query-image /path/to/query_crop.png
```

or:

```bash
--query-feature /path/to/query_vector.npy
```

### Query dimension mismatch

The query vector and loaded point features must share the same final dimension. For example, `facebook/dinov2-base` image queries are normally 768-D, so aligned DINO splat features should also be 768-D.

### CUDA rendering fails or falls back repeatedly

Use the CPU fallback controls:

```bash
--cpu-render-fallback --cpu-fallback-splats 80000
```

or force CPU rendering:

```bash
--force-cpu-render --cpu-fallback-splats 80000
```

### Hugging Face model download is slow or unavailable

Pre-download the model while online, then run with `--hf-cache-dir /path/to/hf_cache`.

## Development and release checks

Run the static release check:

```bash
python scripts/validate_release.py
```

Run tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

Run syntax-focused Ruff checks:

```bash
ruff check . --select E9,F63,F7,F82
```

Before publishing a GitHub release, run at least one CPU command and one CUDA command on a real scene:

```bash
python scripts/smoke_test_loaders.py \
  --folder-npy /path/to/scene_folder \
  --feature-file /path/to/features.npy \
  --device cpu

python run_viewer.py \
  --folder-npy /path/to/scene_folder \
  --feature-file /path/to/features.npy \
  --feature-type siglip2 \
  --device cuda \
  --backend gsplat
```

Also test one DINO path if DINO features are part of the release:

```bash
python run_viewer.py \
  --folder-npy /path/to/scene_folder \
  --dino-feature /path/to/dino_features.npy \
  --feature-type dinov2 \
  --query-image /path/to/query_crop.png \
  --device cuda \
  --backend gsplat
```

## License

Add an explicit `LICENSE` file before publishing a public release. The project owner should choose the license; this patch does not assign legal terms automatically.

## Acknowledgements

- Nerfview for the interactive rendering scaffold.
- Viser for the WebGL viewer frontend.
- GSplat for CUDA Gaussian splat rasterization.
- Hugging Face Transformers for SigLIP2 and DINOv2 model loading.

## Citations

If you use Mini Viewer in research, please consider citing:

```bibtex
@article{wu2023mars,
  author  = {Wu, Zirui and Liu, Tianyu and Luo, Liyi and Zhong, Zhide and Chen, Jianteng and Xiao, Hongmin and Hou, Chao and Lou, Haozhe and Chen, Yuantao and Yang, Runyi and Huang, Yuxin and Ye, Xiaoyu and Yan, Zike and Shi, Yongliang and Liao, Yiyi and Zhao, Hao},
  title   = {MARS: An Instance-aware, Modular and Realistic Simulator for Autonomous Driving},
  journal = {CICAI},
  year    = {2023}
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
