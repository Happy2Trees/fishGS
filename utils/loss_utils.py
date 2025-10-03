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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


# --- ERP latitude weighting utilities ---
def erp_latitude_weight_map(height: int, width: int, device=None, dtype=None) -> torch.Tensor:
    """Return a latitude cosine weight map for ERP images.
    Shape: (1, 1, H, W). weight(y) = cos(lat), lat in [-pi/2, pi/2].
    """
    if dtype is None:
        dtype = torch.float32
    yy = torch.arange(height, device=device, dtype=dtype) + 0.5
    lat = (yy / height) * torch.pi - (torch.pi / 2.0)  # [-pi/2, pi/2]
    w = torch.cos(lat).clamp(min=0.0)  # (H,)
    w2d = w[:, None].expand(height, width)  # (H, W)
    return w2d.unsqueeze(0).unsqueeze(0)

def weighted_l1(network_output: torch.Tensor, gt: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
    """Compute weighted L1 with per-pixel weights. weight_map can be (1,1,H,W) or (H,W).
    Broadcasts across channel.
    """
    if weight_map.dim() == 2:
        weight_map = weight_map.unsqueeze(0).unsqueeze(0)
    if weight_map.dim() == 3:
        weight_map = weight_map.unsqueeze(0)
    # network_output, gt: (C,H,W) or (N,C,H,W)
    if network_output.dim() == 3:
        network_output = network_output.unsqueeze(0)
        gt = gt.unsqueeze(0)
    l1 = torch.abs(network_output - gt)
    # sum over channel, average spatial with weights
    denom = torch.clamp(weight_map.sum(dim=(-2, -1), keepdim=True), min=1e-8)
    loss = (l1 * weight_map).sum(dim=(-2, -1), keepdim=True) / denom
    return loss.mean()

def erp_skip_bottom(img: torch.Tensor, ratio: float) -> torch.Tensor:
    """Crop out the bottom portion of an image by ratio.
    Accepts (C,H,W) or (N,C,H,W). No-op if ratio<=0 or computed pixels==0.
    """
    if ratio <= 0.0:
        return img
    if img.dim() == 3:
        C, H, W = img.shape
        pix = int(round(H * ratio))
        if pix <= 0:
            return img
        return img[:, : H - pix, :]
    elif img.dim() == 4:
        N, C, H, W = img.shape
        pix = int(round(H * ratio))
        if pix <= 0:
            return img
        return img[:, :, : H - pix, :]
    else:
        return img
