These configs mirror OmniGS C++ cfg for 360Roam (lonlat) with only the differences from Python defaults.

Mapped from `submodules/OmniGS/cfg/lonlat/360roam_lonlat.yaml`:
- optimization.iterations: 32010 (C++ `Optimization.max_num_iterations`)
- optimization.opacity_lr: 0.05 (C++ `Optimization.opacity_lr`)
- optimization.skip_bottom_ratio: 0.063 (C++ `Optimization.skip_bottom_ratio`)

Per‑scene files only change `model.source_path` and `model.model_path`.

Usage:
- `python train.py -c configs/360roam/bar.yaml`
 - Resolution variants live next to each scene file (e.g., `bar_r1.yaml`, `bar_r2.yaml`). Only r1/r2를 제공합니다.

Notes:
- ERP dataset type is auto-detected from 360Roam folder structure.
- For “fullres” variant (C++ cfg `360roam_lonlat_fullres.yaml`), use `skip_bottom_ratio: 0.0`.
