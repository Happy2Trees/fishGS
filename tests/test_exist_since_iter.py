import os
import sys

import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
from argparse import ArgumentParser
from arguments import OptimizationParams


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GaussianModel tensors")
def test_exist_since_iter_init_and_propagation_clone_split_prune():
    # Build a small cloud with varying spacing
    A = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0001], [0.0, 0.0, 0.9999]], dtype=np.float32)
    B = np.array([[1.0, 1.0, 2.0], [1.1, 1.1, 2.2]], dtype=np.float32)
    pts = np.vstack([A, B])
    cols = np.ones_like(pts, dtype=np.float32) * 0.5
    nrm = np.zeros_like(pts, dtype=np.float32)
    pcd = BasicPointCloud(points=pts, colors=cols, normals=nrm)

    class _CamInfo:
        def __init__(self, name):
            self.image_name = name

    gm = GaussianModel(sh_degree=3)
    gm.create_from_pcd(pcd, cam_infos=[_CamInfo("img0")], spatial_lr_scale=1.0)
    parser = ArgumentParser()
    op = OptimizationParams(parser)
    args = parser.parse_args([])
    train_args = op.extract(args)
    gm.training_setup(train_args)

    N0 = gm.get_xyz.shape[0]
    # Initialize exist_since_iter with unique values per point
    base_exist = torch.arange(N0, dtype=torch.int32, device="cuda")
    gm.exist_since_iter = base_exist.clone()

    # Prepare gradients to force densify
    grads = torch.full((N0, 1), 1.0, device="cuda")
    gm.xyz_gradient_accum = grads.clone()
    gm.denom = torch.ones_like(gm.denom)

    # Choose threshold by current scale to split some and clone others
    scales_max = gm.get_scaling.max(dim=1).values.detach()
    threshold_val = torch.median(scales_max).item()
    extent = 1.0
    gm.percent_dense = threshold_val / extent

    # No screen-size prune, no opacity prune
    radii = torch.zeros(N0, device="cuda")
    gm.densify_and_prune(
        max_grad=1e-3,
        min_opacity=0.0,
        extent=extent,
        max_screen_size=0,
        radii=radii,
        prune_by_extent=True,
    )

    # Expected counts
    K_clone = int((scales_max <= threshold_val).sum().item())
    K_split = int((scales_max > threshold_val).sum().item())
    N_expected = N0 + K_clone + K_split
    assert gm.get_xyz.shape[0] == N_expected

    # Check exist_since_iter values propagate correctly
    # Gather indices chosen
    choose_clone_mask = (scales_max <= threshold_val)
    choose_split_mask = ~choose_clone_mask
    cloned_exist = base_exist[choose_clone_mask]
    split_exist = base_exist[choose_split_mask]
    # Layout after densify+prune:
    # [originals that were cloned (not split)] + [their clones] + [split children]
    head = gm.exist_since_iter[:K_clone]
    mid = gm.exist_since_iter[K_clone:K_clone + K_clone]
    tail = gm.exist_since_iter[K_clone + K_clone:]
    assert torch.equal(head, cloned_exist)
    assert torch.equal(mid, cloned_exist)
    N_children = 2  # densify_and_split uses N=2
    # Order follows repeat(N,1): all selected once, then again
    expected_tail = split_exist.repeat(N_children)
    assert torch.equal(tail, expected_tail)

    # Now prune arbitrary two points and ensure exist prunes accordingly
    mask = torch.zeros(gm.exist_since_iter.shape[0], dtype=torch.bool, device="cuda")
    mask[0] = True
    mask[-1] = True
    gm.tmp_radii = torch.zeros_like(gm.max_radii2D)
    before_len = gm.exist_since_iter.shape[0]
    gm.prune_points(mask)
    assert gm.exist_since_iter.shape[0] == before_len - 2
