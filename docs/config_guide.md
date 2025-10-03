# YAML Config Guide

You can now launch training and rendering with a YAML config via `--config/-c`.

Merge precedence (low → high):

1. Built‑in defaults
2. `<model_path>/cfg_args` (if present)
3. YAML file passed via `--config`
4. Command‑line flags

This means CLI flags always override values from the YAML.

## Run Examples

- Train with YAML only:

  `python train.py -c docs/configs/train_sample.yaml`

- Train with YAML but override iterations from CLI:

  `python train.py -c docs/configs/train_sample.yaml --iterations 10000`

- Preview effective params or write them to a file:

  - `python train.py --print_params`
  - `python train.py -c my.yaml --dump_config merged.yaml`

## YAML Structure

The loader accepts both a grouped and a flat style:

- Grouped keys: `model`, `pipeline`, `optimization`, `train`.
- Flat keys (same as CLI flags) also work; nested keys are flattened.

Any unknown key is ignored. Boolean values are regular YAML booleans (`true/false`).

### Minimal Example

```yaml
model:
  source_path: /path/to/dataset
  model_path: output/my_run
  resolution: -1
  white_background: false

optimization:
  iterations: 30000
  optimizer_type: default  # or "sparse_adam" if compiled
  lambda_dssim: 0.2

pipeline:
  rasterizer: omnigs  # or "diff"
  antialiasing: false

train:
  seed: 0
  test_iterations: [7000, 30000]
  save_iterations: [7000, 30000]
  disable_viewer: true
```

### Flat Style (equivalent)

```yaml
source_path: /path/to/dataset
model_path: output/my_run
resolution: -1
iterations: 30000
lambda_dssim: 0.2
rasterizer: omnigs
antialiasing: false
seed: 0
test_iterations: [7000, 30000]
save_iterations: [7000, 30000]
disable_viewer: true
```

## Parameter Reference

- Built‑in defaults are discoverable via:
  - `python train.py --print_params` (training)
  - `python render.py --print_params` (rendering)

Both commands print every available flag with its default value. Use those names as YAML keys (grouped or flat).

