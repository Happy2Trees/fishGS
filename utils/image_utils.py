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

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def weighted_psnr(img1: torch.Tensor, img2: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
    """Compute PSNR with latitude weighting for ERP images.
    img1, img2: (N,C,H,W) or (C,H,W). weight_map: (1,1,H,W) or (H,W) or (N,1,H,W).
    Returns: (N,1) tensor of PSNR per-image.
    """
    if weight_map.dim() == 2:
        weight_map = weight_map.unsqueeze(0).unsqueeze(0)
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    if weight_map.dim() == 3:
        weight_map = weight_map.unsqueeze(0)
    # broadcast to N
    if weight_map.shape[0] == 1 and img1.shape[0] > 1:
        weight_map = weight_map.expand(img1.shape[0], -1, -1, -1)
    diff2 = (img1 - img2) ** 2  # (N,C,H,W)
    w = weight_map
    # sum over spatial and channel with weights, normalize by sum of weights
    sum_w = torch.clamp(w.sum(dim=(-2, -1), keepdim=True), min=1e-8)  # (N,1,1,1)
    mse_w = (diff2 * w).sum(dim=(-2, -1), keepdim=True) / sum_w  # (N,C,1,1)
    mse_w = mse_w.mean(dim=1, keepdim=True).view(img1.shape[0], 1)  # average over C
    return 20 * torch.log10(1.0 / torch.sqrt(mse_w))
