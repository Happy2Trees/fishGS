import math
import os
import sys
from argparse import ArgumentParser

import pytest
import torch

# Ensure repository root is on path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import get_expon_lr_func


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GaussianModel tensors")
def test_param_groups_and_initial_lrs_match_omnigs():
    # Create a tiny point cloud
    import numpy as np
    pts = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.2], [-0.1, 0.2, 0.9]], dtype=np.float32)
    cols = np.array([[0.5, 0.2, 0.7], [0.1, 0.9, 0.3], [0.3, 0.3, 0.3]], dtype=np.float32)
    nrm = np.zeros_like(pts, dtype=np.float32)
    pcd = BasicPointCloud(points=pts, colors=cols, normals=nrm)

    class _CamInfo:
        def __init__(self, name):
            self.image_name = name

    cam_infos = [_CamInfo("img0"), _CamInfo("img1")]

    gm = GaussianModel(sh_degree=3)
    spatial_lr_scale = 1.23
    gm.create_from_pcd(pcd, cam_infos, spatial_lr_scale)

    parser = ArgumentParser()
    op = OptimizationParams(parser)
    args = parser.parse_args([])
    train_args = op.extract(args)
    gm.training_setup(train_args)

    # Expect 6 param groups with OmniGS order and LRs
    pgs = gm.optimizer.param_groups
    assert len(pgs) == 6
    # group0: xyz
    assert pgs[0]["name"] == "xyz"
    assert math.isclose(pgs[0]["lr"], train_args.position_lr_init * spatial_lr_scale, rel_tol=1e-6, abs_tol=1e-12)
    # group1: features_dc
    assert pgs[1]["name"] == "f_dc"
    assert math.isclose(pgs[1]["lr"], train_args.feature_lr, rel_tol=1e-6, abs_tol=1e-12)
    # group2: features_rest
    assert pgs[2]["name"] == "f_rest"
    assert math.isclose(pgs[2]["lr"], train_args.feature_lr / 20.0, rel_tol=1e-6, abs_tol=1e-12)
    # group3: opacity
    assert pgs[3]["name"] == "opacity"
    assert math.isclose(pgs[3]["lr"], train_args.opacity_lr, rel_tol=1e-6, abs_tol=1e-12)
    # group4: scaling
    assert pgs[4]["name"] == "scaling"
    assert math.isclose(pgs[4]["lr"], train_args.scaling_lr, rel_tol=1e-6, abs_tol=1e-12)
    # group5: rotation
    assert pgs[5]["name"] == "rotation"
    assert math.isclose(pgs[5]["lr"], train_args.rotation_lr, rel_tol=1e-6, abs_tol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GaussianModel tensors")
def test_update_learning_rate_applies_only_to_xyz_group():
    # Setup minimal scene
    import numpy as np
    pts = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.2]], dtype=np.float32)
    cols = np.array([[0.5, 0.2, 0.7], [0.1, 0.9, 0.3]], dtype=np.float32)
    nrm = np.zeros_like(pts, dtype=np.float32)
    pcd = BasicPointCloud(points=pts, colors=cols, normals=nrm)

    class _CamInfo:
        def __init__(self, name):
            self.image_name = name

    cam_infos = [_CamInfo("img0")]

    gm = GaussianModel(sh_degree=3)
    spatial_lr_scale = 1.0
    gm.create_from_pcd(pcd, cam_infos, spatial_lr_scale)

    parser = ArgumentParser()
    op = OptimizationParams(parser)
    args = parser.parse_args([])
    train_args = op.extract(args)
    gm.training_setup(train_args)

    # Compute expected schedule
    sched = get_expon_lr_func(
        lr_init=train_args.position_lr_init * spatial_lr_scale,
        lr_final=train_args.position_lr_final * spatial_lr_scale,
        lr_delay_mult=train_args.position_lr_delay_mult,
        max_steps=train_args.position_lr_max_steps,
    )

    pgs_before = [pg["lr"] for pg in gm.optimizer.param_groups]

    step = 1234
    lr = gm.update_learning_rate(step)
    # xyz lr updated, others unchanged
    pgs_after = [pg["lr"] for pg in gm.optimizer.param_groups]

    assert math.isclose(lr, sched(step), rel_tol=1e-8, abs_tol=1e-12)
    assert math.isclose(pgs_after[0], sched(step), rel_tol=1e-8, abs_tol=1e-12)
    for i in range(1, 6):
        assert math.isclose(pgs_after[i], pgs_before[i], rel_tol=1e-12, abs_tol=1e-12)
