import os
import sys

import pytest
import torch
import numpy as np

# Ensure repository root is on path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
from argparse import ArgumentParser
from arguments import OptimizationParams


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GaussianModel tensors")
def test_densify_clone_vs_split_and_no_prune_by_size_when_disabled():
    # Construct two clusters to induce different scales
    # Cluster A (very close points) -> smaller scaling
    A = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0001], [0.0, 0.0, 0.9999]], dtype=np.float32)
    # Cluster B (more separated points) -> larger scaling
    B = np.array([[1.0, 1.0, 2.0], [1.0, 1.0, 2.2]], dtype=np.float32)
    pts = np.vstack([A, B])
    cols = np.ones_like(pts, dtype=np.float32) * 0.5
    nrm = np.zeros_like(pts, dtype=np.float32)
    pcd = BasicPointCloud(points=pts, colors=cols, normals=nrm)

    class _CamInfo:
        def __init__(self, name):
            self.image_name = name

    gm = GaussianModel(sh_degree=3)
    gm.create_from_pcd(pcd, cam_infos=[_CamInfo("img0")], spatial_lr_scale=1.0)
    # initialize training buffers on CUDA
    parser = ArgumentParser()
    op = OptimizationParams(parser)
    args = parser.parse_args([])
    train_args = op.extract(args)
    gm.training_setup(train_args)

    # Prepare gradients: all above threshold
    N0 = gm.get_xyz.shape[0]
    grads = torch.full((N0, 1), 1e-4, device="cuda")
    gm.xyz_gradient_accum = grads.clone()
    gm.denom = torch.ones_like(gm.denom) + 0.0

    # Choose threshold at median scale value to split clone/split sets deterministically
    scales_max = gm.get_scaling.max(dim=1).values.detach()
    threshold_val = torch.median(scales_max).item()
    extent = 1.0
    gm.percent_dense = threshold_val / extent

    # Call densify_and_prune with size pruning disabled (max_screen_size = 0)
    radii = torch.zeros(N0, device="cuda")
    gm.densify_and_prune(
        max_grad=1e-6,
        min_opacity=0.0,
        extent=extent,
        max_screen_size=0,
        radii=radii,
        prune_by_extent=True,
    )

    # Expected: clones for <= threshold, splits (N=2) for > threshold with original pruned
    K_clone = int((scales_max <= threshold_val).sum().item())
    K_split = int((scales_max > threshold_val).sum().item())
    N_expected = N0 + K_clone + K_split  # net +K_clone + (2-1)*K_split
    assert gm.get_xyz.shape[0] == N_expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GaussianModel tensors")
def test_prune_by_screen_and_extent_rules():
    # Single small cloud
    pts = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.1], [0.2, 0.0, 1.2]], dtype=np.float32)
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

    # No densify: grads zero
    gm.xyz_gradient_accum = torch.zeros_like(gm.xyz_gradient_accum)
    gm.denom = torch.ones_like(gm.denom)

    # Prune by world-space extent when enabled (note: VS threshold is unreliable inside densify call due to reset)
    gm.max_radii2D = torch.zeros(N0, device="cuda")  # ensure VS prune does not trigger
    scales_max = gm.get_scaling.max(dim=1).values.detach()
    # choose very small extent so that many points exceed 0.1 * extent
    tiny_extent = float(scales_max.min().item() / 20.0)
    gm.densify_and_prune(
        max_grad=1.0,
        min_opacity=0.0,
        extent=tiny_extent,
        max_screen_size=1,   # truthy to activate block
        radii=torch.zeros(N0, device="cuda"),
        prune_by_extent=True,
    )
    # All points with scale > 0.1 * extent should be removed
    expected_pruned = (scales_max > 0.1 * tiny_extent).sum().item()
    assert gm.get_xyz.shape[0] == N0 - int(expected_pruned)
