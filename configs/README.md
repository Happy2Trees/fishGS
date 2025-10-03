These YAMLs are minimal overrides for datasets (mirroring OmniGS C++ cfgs). They only list values that differ from the code defaults to keep them readable.

Usage examples:

- Train 360Roam "bar":
  `python train.py -c configs/360roam_bar.yaml`

Notes
- Defaults come from `train.py` and `arguments/` (Model/Pipeline/Optimization). Use `python train.py --print_params` to see full defaults.
- ERP datasets (360Roam/EgoNeRF lonlat) are auto-detected; no need to set `camera_type`.
- Viewer is disabled via `train.disable_viewer: true` for headless runs.
- For resolution variants, use `_r1` or `_r2` files next to each dataset config. These set `model.resolution` (1: 원본, 2: 1/2 스케일).

Folders
- `configs/360roam/` — per-scene configs from OmniGS lonlat cfg. 해상도는 `_r1/_r2` 파일로 제공합니다.
- `configs/egonerf/` — Ricoh360/OmniBlender 템플릿과 `_r1/_r2` 변형.
- `configs/colmap/` — COLMAP 템플릿과 `_r1/_r2` 변형.
