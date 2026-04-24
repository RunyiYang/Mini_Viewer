# Changelog

## 0.3.0 - 2026-04-24

### Added

- DINOv2 aligned-feature support via `--feature-type dinov2` / `--feature-type dino`.
- `--dino-feature` alias for visual feature tensors.
- `--query-image` for DINOv2 image-prompt querying.
- `--dino-model`, defaulting to `facebook/dinov2-base`.
- `scripts/download_dino.py` for offline/cache preparation.
- Generic `--feature-file` option that works for SigLIP2, CLIP, DINOv2, and custom features.
- Release-ready static validation, CI, changelog, release checklist, `.gitattributes`, and normalized `.gitignore`.

### Changed

- Reframed the GUI folder as **Feature Query** because it now supports text, image, and vector queries.
- Query encoders are lazy-loaded only when a model-backed query is used.
- Feature loader now recognizes DINO/DINOv2/visual feature keys in `.npz`, `.pt`, and `.pth` files.
- Consolidated CPU/CUDA/hotfix behavior into normal source files.

### Notes

- DINOv2 is visual-only. Use an image path or `--query-feature`; it does not encode text prompts.
- Add an explicit `LICENSE` before publishing a public GitHub release.

## 0.2.0 - 2026-04-24

### Added

- Unified CUDA/CPU environment via `env.yml` and `requirements.txt`.
- CPU render fallback controls for interactive rerendering.
- SigLIP2 default query encoder: `google/siglip2-so400m-patch16-512`.
- Feature-map visualization, query recoloring, queried-feature bbox export.
- Camera keyframe placement, Nerfstudio-style camera-path JSON export, and video rendering.
