Configs mapped from OmniGS C++ lonlat cfgs:

- Base (differences only):
  - iterations: 32010
  - opacity_lr: 0.05
  - prune_by_extent: false

Fill in `model.source_path` per sequence before running.

Examples:
- `python train.py -c configs/egonerf/ricoh360_template.yaml`
- `python train.py -c configs/egonerf/ricoh360_r1.yaml`
- `python train.py -c configs/egonerf/ricoh360_r2.yaml`
