# Mini Viewer
> Realtime Gaussian splat inspector with SigLIP/CLIP querying, bounding-box overlays, and a streamlined `viser` / `nerfview` workflow.

![Mini Viewer](docs/mini_viewer.png)

Mini Viewer taps GSplat’s CUDA kernels through Nerfview’s ergonomics and renders them with Viser’s low-latency WebGL front-end. Load `.ply` scenes, NumPy blobs, or ckpt exports, drop in language features, and layer annotated bounding boxes via the in-repo `viser_bbox` toolkit.

## Highlights
- **Fresh viewer stack** – `viser 1.0.15`, `nerfview 0.1.3`, and `gsplat 1.5.3` with PyTorch CUDA 12.4 wheels.
- **Language guidance** – SigLIP/CLIP embeddings recolor or prune splats in real time.
- **One-click shaders** – RGB, depth, normals, screenshots with white background cleanup, and sticky camera poses.
- **Bounding boxes** – Use SpatialLM-style scripts to draw labeled walls, doors, windows, and instance boxes right inside the same Viser server.
- **Batteries included** – `viser_bbox` ships as an editable package inside this repo for custom scripts or tooling.

## Environment Setup
Tested on **Python 3.11**, **CUDA 12.4**, and NVIDIA 30/40/A-series data center GPUs. FlashAttention requires CUDA ≥ 12.3 if you enable it.

```bash
# create env (micromamba shown, conda/venv also fine)
micromamba create -n viewer python=3.11 -y
micromamba activate viewer

# install the viewer stack
pip install -r requirements.txt
pip install -e ./viser_bbox

# optional extras
pip install flash-attn --no-build-isolation
```

Useful CUDA exports (adapt to your toolchain):

```bash
export CC=/usr/bin/gcc-11.5
export CXX=/usr/bin/g++-11.5
export LD=/usr/bin/g++-11.5
export TORCH_CUDA_ARCH_LIST="8.6;8.9"
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.4/bin:$PATH
export CPLUS_INCLUDE_PATH=/usr/local/cuda-12.4/include
```

## Quick Start
1. Prepare a Gaussian splat `.ply`, a set of NumPy arrays (`coord.npy`, `quat.npy`, `scale.npy`, `opacity.npy`, `color.npy`), or a ckpt convertible via `utils/pyl_to_ckpt.py`.
2. (Optional) Pre-compute language features (SigLIP or CLIP embeddings) into a `.pth` or `.npy`.
3. Launch the viewer:

```bash
python run_viewer.py \
  --ply /path/to/scene.ply \
  --language_feature /path/to/langfeat.pth \
  --feature_type siglip \
  --port 8080 \
  --device cuda
```

Then open `http://localhost:8080` to interact with the scene. The server persists camera states inside `.tmp/camera_state.json`, so refreshing keeps your view.

### Example: ScanNet++ scene
This repository includes ScanNet++ samples under `data/scannetpp/val`. To inspect `09c1414f1b`, run:

```bash
python run_viewer.py \
  --folder_npy data/scannetpp/val/09c1414f1b \
  --language_feature language_feature_dummy \
  --device cuda \
  --port 8080
```

### CLI Arguments
- `--ply`: Path to Inria-style Gaussian PLY (alternatively use `--folder_npy` to read NumPy blobs).
- `--folder_npy`: Directory with `{coord,normal?,quat,scale,opacity,color}.npy`.
- `--language_feature`: Torch or NumPy file containing `(N, D)` embeddings aligned with the splats.
- `--feature_type`: `siglip`, `clip`, or path to a saved embedding vector.
- `--prune`: Optional string flag to enable pruning heuristics baked into `SplatData`.
- `--bbox_script`: SpatialLM-style script (see `docs/bboxes/demo.txt`) for drawing bounding boxes via `viser_bbox`.
- `--device`: `cuda` or `cpu`.
- `--port`: Viser server port (default `8080`).

## Bounding Boxes with `viser_bbox`
The `viser_bbox` subpackage ships reusable helpers and an API for drawing labeled wireframe boxes. Install it once (`pip install -e ./viser_bbox`) and pass a script to the viewer:

```bash
python run_viewer.py \
  --folder_npy data/scannetpp/val/09c1414f1b \
  --bbox_script docs/bboxes/demo.txt \
  --device cuda
```

Scripts follow the SpatialLM format:

```
wall0 = Wall(-2, -2, 0, 2, -2, 0, 3.0, 0.18)
door0 = Door(wall0, 0.0, -2.0, 1.0, 0.9, 2.1)
bbox0 = Bbox(Sofa, 0.8, 0.3, 0.7, 0.0, 1.5, 0.9, 1.0)
```

Under the hood we call `viser_bbox.add_script_bboxes`, so you can generate or reload scripts at runtime. For programmatic overlays (e.g., after an object selection) import `add_bounding_box` directly inside any action.

## Language-Driven Editing
Inside the **Language Feature** folder in the GUI you can:
- Visualize the embedding space by sending colors straight to the renderer (`Feature Map`).
- Toggle per-point normals (`Normal Map`) for quick sanity checks.
- Type free-form text prompts (SigLIP/CLIP encoders are loaded dynamically) and recolor matches in red.
- Prune splats by cosine score using the `Rate` field and `Prune based on text prompt` button.

The module keeps both the PCA-compressed 3-channel preview (`language_feature`) and the full high-dimensional tensor (`language_feature_large`) so you can swap render modes without recomputing embeddings.

### Query Gallery
Below are snapshots generated from live language queries. The embeddings come from SigLIP and are executed through the viewer’s cosine-similarity filter.

![Query: vocation art](docs/query/query_vocation_art.png)
![Query: toy add flavor](docs/query/query_toy_addflavor.png)

See `docs/query/text_query.mp4` for a short screen recording of the workflow.

## Working with Data
- **PLY ingestion:** `utils/pyl_to_ckpt.py` converts Inria-format splats into GSplat-ready tensors. We expect vertex properties named `x,y,z`, `scale_*`, `rot_*`, `f_dc_*`, and `opacity`.
- **NumPy ingestion:** drop `coord.npy`, `quat.npy`, `scale.npy`, `opacity.npy`, and `color.npy` (0–255) into a folder and pass it via `--folder_npy`. Optional `normal.npy` and language feature files are respected.
- **Language features:** store embeddings as `.pth` (tuple `(tensor, metadata)`) or `.npy`. The loader masks them alongside the splats so pruning stays consistent.
- **Snapshots:** press *Snapshot* in the GUI to write `./snapshot{idx}.png` with alpha handled automatically and the current camera stored under `.tmp/`.

## Development Notes
- GUI controls live in `actions/*.py`. Drop another folder via `server.gui.add_folder` to extend the toolkit (object selection, path planning, etc.).
- `ViewerEditor` subclasses `nerfview.Viewer`, so any upstream camera improvements land here automatically when you bump the pip dependency.
- `update_splat_renderer` can swap backends (e.g., `backend="torch"`) if you experiment with new renderers.
- Do **not** stream datasets across nodes; latency tanks because GSplat expects local NVMe speed.

## Roadmap
- [x] Object query & removal
- [x] Basic `viser_bbox` overlay support from the CLI
- [ ] Object selection tooling
- [ ] Camera placement and keyframe interpolation
- [ ] In-viewer bbox authoring + editing tools

## Acknowledgements
- [Nerfview](https://github.com/hangg7/nerfview) for the original interactive scaffolding.
- [Viser](https://github.com/nerfstudio-project/viser) for the lightweight WebGL frontend.
- [GSplat](https://github.com/nerfstudio-project/gsplat) for CUDA splat rendering.
- `viser_bbox` utilities were authored in this repo to streamline bounding-box overlays.
## Citations
If you use Mini Viewer in research, please consider citing:

```
@article{wu2023mars,
  author    = {Wu, Zirui and Liu, Tianyu and Luo, Liyi and Zhong, Zhide and Chen, Jianteng and Xiao, Hongmin and Hou, Chao and Lou, Haozhe and Chen, Yuantao and Yang, Runyi and Huang, Yuxin and Ye, Xiaoyu and Yan, Zike and Shi, Yongliang and Liao, Yiyi and Zhao, Hao},
  title     = {MARS: An Instance-aware, Modular and Realistic Simulator for Autonomous Driving},
  journal   = {CICAI},
  year      = {2023},
}

@misc{yang2024spectrally,
      title={Spectrally Pruned Gaussian Fields with Neural Compensation}, 
      author={Runyi Yang and Zhenxin Zhu and Zhou Jiang and Baijun Ye and Xiaoxue Chen and Yifei Zhang and Yuantao Chen and Jian Zhao and Hao Zhao},
      year={2024},
      eprint={2405.00676},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{zheng2024gaussiangrasper,
  title={Gaussiangrasper: 3d language gaussian splatting for open-vocabulary robotic grasping},
  author={Zheng, Yuhang and Chen, Xiangyu and Zheng, Yupeng and Gu, Songen and Yang, Runyi and Jin, Bu and Li, Pengfei and Zhong, Chengliang and Wang, Zengmao and Liu, Lina and others},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}

@article{li2025scenesplat,
  title={SceneSplat: Gaussian Splatting-based Scene Understanding with Vision-Language Pretraining},
  author={Li, Yue and Ma, Qi and Yang, Runyi and Li, Huapeng and Ma, Mengjiao and Ren, Bin and Popovic, Nikola and Sebe, Nicu and Konukoglu, Ender and Gevers, Theo and others},
  journal={arXiv preprint arXiv:2503.18052},
  year={2025}
}
```
