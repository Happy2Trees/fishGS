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
import math
from typing import TYPE_CHECKING

# Optional backends: prefer OmniGS but allow 'diff' for compatibility
try:  # OmniGS backend (preferred)
    from omnigs_rasterization import (
        GaussianRasterizationSettings as OmniGS_Settings,
        GaussianRasterizer as OmniGS_Rasterizer,
    )
    from omnigs_rasterization import mark_visible as _omnigs_mark_visible
except Exception:  # pragma: no cover - tests may stub this
    OmniGS_Settings = None
    OmniGS_Rasterizer = None
    _omnigs_mark_visible = None

try:  # 3DGS original backend
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings as Diff_Settings,
        GaussianRasterizer as Diff_Rasterizer,
    )
except Exception:  # pragma: no cover
    Diff_Settings = None
    Diff_Rasterizer = None

# Public alias for tests to monkeypatch
GaussianRasterizer = OmniGS_Rasterizer if OmniGS_Rasterizer is not None else Diff_Rasterizer
from utils.sh_utils import eval_sh

if TYPE_CHECKING:  # avoid heavy import at runtime
    from scene.gaussian_model import GaussianModel

def render(viewpoint_camera, pc: "GaussianModel", pipe, bg_color: torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    camera_type = getattr(viewpoint_camera, "camera_type", 1)
    if camera_type == 3:
        tanfovx = 1.0
        tanfovy = 1.0
    else:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # Choose rasterizer backend
    backend = getattr(pipe, "rasterizer", "omnigs")
    if backend == "diff":
        if Diff_Settings is None or Diff_Rasterizer is None:
            raise RuntimeError("diff_gaussian_rasterization is not available")
        raster_settings = Diff_Settings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=getattr(pipe, "debug", False),
            antialiasing=getattr(pipe, "antialiasing", False),
        )
        rasterizer = Diff_Rasterizer(raster_settings=raster_settings)
    else:
        if OmniGS_Settings is None or OmniGS_Rasterizer is None:
            raise RuntimeError("omnigs_rasterization is not available")
        raster_settings = OmniGS_Settings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            camera_type=camera_type,
        )
        rasterizer_cls = globals().get("GaussianRasterizer", OmniGS_Rasterizer)
        rasterizer = rasterizer_cls(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        ret = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        ret = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

    # Normalize rasterizer return signature across backends.
    # OmniGS returns (image, radii, depth). diff backend may return only (image, radii).
    if isinstance(ret, tuple) and len(ret) == 3:
        rendered_image, radii, depth_image = ret
    elif isinstance(ret, tuple) and len(ret) == 2:
        rendered_image, radii = ret
        # Create a zero depth map matching HxW
        H = int(viewpoint_camera.image_height)
        W = int(viewpoint_camera.image_width)
        depth_image = torch.zeros(1, H, W, device=rendered_image.device, dtype=rendered_image.dtype)
    else:
        raise RuntimeError("Unexpected rasterizer return format")
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out


def GaussianModel(*args, **kwargs):  
    from scene.gaussian_model import GaussianModel as _GM
    return _GM(*args, **kwargs)


def mark_visible(viewpoint_camera, positions: torch.Tensor) -> torch.Tensor:
    """Call OmniGS rasterizer's markVisible with the current camera.

    - Returns a boolean tensor of shape (P,) indicating visibility.
    - For ERP/LONLAT cameras (camera_type==3), all points are visible by design.
    """
    camera_type = getattr(viewpoint_camera, "camera_type", 1)
    if _omnigs_mark_visible is None:
        raise RuntimeError("omnigs_rasterization is not available: mark_visible requires the extension to be built")
    return _omnigs_mark_visible(
        positions,
        viewpoint_camera.world_view_transform,
        viewpoint_camera.full_proj_transform,
        int(camera_type),
    )
