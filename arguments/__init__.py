#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os
from typing import Any, Dict
try:
    import yaml  # type: ignore
    _YAML_AVAILABLE = True
except Exception:
    _YAML_AVAILABLE = False

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        # Camera type override for loader: 'auto'|'pinhole'|'lonlat'
        # Default keeps dataset-driven inference.
        self.camera_type = "auto"
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        # Rasterizer backend selection: 'omnigs' (default) or 'diff'
        self.rasterizer = "omnigs"
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        # OmniGS-aligned extras
        self.densify_min_opacity = 0.005
        self.prune_by_extent = True
        self.prune_big_point_after_iter = 0  # 0 disables until positive iterations
        # ERP options
        self.skip_bottom_ratio = 0.0
        self.erp_weighted_loss = False
        self.erp_weighted_metrics = False
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    """Backward-compatible config merger used by render.py.
    Reads `<model_path>/cfg_args` (if present) and merges with CLI, where CLI wins.
    """
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    except FileNotFoundError:
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

# ---------------------- YAML-enabled argument handling ---------------------- #

def _flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            flat.update(_flatten_dict(v))
        else:
            flat[k] = v
    return flat

def _read_cfgfile_namespace(model_path: str) -> Namespace:
    cfgfilepath = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfgfilepath):
        raise FileNotFoundError(cfgfilepath)
    print("Looking for config file in", cfgfilepath)
    with open(cfgfilepath) as cfg_file:
        print("Config file found: {}".format(cfgfilepath))
        cfgfile_string = cfg_file.read()
    return eval(cfgfile_string)

def get_args_with_yaml(parser: ArgumentParser) -> Namespace:
    """Parse arguments with optional YAML and stored cfg_args merging.

    Merge order (low -> high precedence):
      parser defaults -> `<model_path>/cfg_args` -> YAML `--config/-c` -> CLI

    This keeps CLI as the final override while allowing both stored configs
    and external YAML to pre-populate values.
    """
    argv = sys.argv[1:]

    # Bootstrap parse to discover `--config/-c` and `--model_path/-m`
    bootstrap = ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", "-c", type=str, default=None)
    bootstrap.add_argument("--model_path", "-m", type=str, default=None)
    boot_args, _ = bootstrap.parse_known_args(argv)

    # Load stored cfg_args if available
    stored: Dict[str, Any] = {}
    if boot_args.model_path:
        try:
            stored_ns = _read_cfgfile_namespace(boot_args.model_path)
            stored = vars(stored_ns).copy()
        except Exception:
            stored = {}

    # Load YAML if requested
    yaml_flat: Dict[str, Any] = {}
    if boot_args.config:
        if not _YAML_AVAILABLE:
            raise RuntimeError("PyYAML is not installed. Please `pip install pyyaml`. ")
        with open(boot_args.config, "r") as f:
            yaml_data = yaml.safe_load(f) or {}
        yaml_flat = _flatten_dict(yaml_data)

    # Compose defaults -> stored -> yaml, then parse CLI to finalize
    defaults = vars(parser.parse_args([])).copy()
    merged_defaults = defaults.copy()
    merged_defaults.update(stored)
    merged_defaults.update(yaml_flat)
    parser.set_defaults(**merged_defaults)

    args = parser.parse_args(argv)
    return args

def dump_args_to_yaml(namespace: Namespace, filepath: str) -> None:
    """Dump a Namespace to grouped YAML.

    Groups are inferred from ParamGroup classes. Keys not belonging to a
    known group go under `train`.
    """
    if not _YAML_AVAILABLE:
        raise RuntimeError("PyYAML is not installed. Please `pip install pyyaml`. ")

    # Build group key sets
    dummy = ArgumentParser(add_help=False)
    mp = ModelParams(dummy)
    pp = PipelineParams(dummy)
    op = OptimizationParams(dummy)

    def norm_keys(o: object):
        keys = []
        for k in vars(o).keys():
            if k.startswith("_"):
                keys.append(k[1:])
            else:
                keys.append(k)
        return set(keys)

    groups = {
        "model": norm_keys(mp),
        "pipeline": norm_keys(pp),
        "optimization": norm_keys(op),
    }

    flat = vars(namespace).copy()
    out: Dict[str, Any] = {"model": {}, "pipeline": {}, "optimization": {}, "train": {}}
    for k, v in flat.items():
        if k in groups["model"]:
            out["model"][k] = v
        elif k in groups["pipeline"]:
            out["pipeline"][k] = v
        elif k in groups["optimization"]:
            out["optimization"][k] = v
        elif k not in {"config", "dump_config", "print_params"}:
            out["train"][k] = v

    with open(filepath, "w") as f:
        yaml.safe_dump(out, f, sort_keys=True)
