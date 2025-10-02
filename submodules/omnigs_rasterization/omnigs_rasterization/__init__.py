"""OmniGS rasterizer PyTorch extension (self-contained).

This module exposes a 3DGS-compatible Python API backed by the OmniGS CUDA
rasterizer. It mirrors the common diff_gaussian_rasterization surface while
adding camera_type support and depth rendering via a second forward pass.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence, Tuple, Optional

import torch

try:  # pragma: no cover - during install the extension is unavailable
    from . import _C
except ImportError as import_error:  # pragma: no cover
    _C = None
    _IMPORT_ERROR = import_error
else:
    _IMPORT_ERROR = None


# ---------------------------
# Settings and small helpers
# ---------------------------

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: Sequence[float] | torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    camera_type: int = 1  # 1: PINHOLE, 3: LONLAT
    render_depth: bool = False


def _as_float3_bg(bg: Sequence[float] | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(bg, torch.Tensor):
        if bg.numel() != 3:
            raise ValueError("bg must have 3 elements (RGB)")
        return bg.to(device=device, dtype=torch.float32).contiguous().view(3)
    if len(bg) != 3:
        raise ValueError("bg must have 3 elements (RGB)")
    return torch.tensor(list(bg), device=device, dtype=torch.float32).contiguous()


def _empty(device: torch.device) -> torch.Tensor:
    return torch.empty(0, device=device, dtype=torch.float32)


def _ensure_cuda_f32(t: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not t.is_cuda:
        raise ValueError(f"{name} must be on CUDA device")
    if t.dtype != torch.float32:
        t = t.to(dtype=torch.float32)
    return t.contiguous()


def _check_camera_type(camera_type: int) -> None:
    if camera_type not in (1, 3):
        raise ValueError("camera_type must be 1 (PINHOLE) or 3 (LONLAT)")


# ---------------------------
# Autograd binding
# ---------------------------

def rasterize_gaussians(
    means3D: torch.Tensor,
    means2D: torch.Tensor,
    sh: torch.Tensor,
    colors_precomp: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    cov3Ds_precomp: torch.Tensor,
    raster_settings: GaussianRasterizationSettings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        sh: torch.Tensor,
        colors_precomp: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        cov3Ds_precomp: torch.Tensor,
        raster_settings: GaussianRasterizationSettings,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if _C is None:  # pragma: no cover
            raise RuntimeError("omnigs_rasterization extension is not built yet") from _IMPORT_ERROR

        device = means3D.device

        # Sanity checks similar to 3DGS
        if (sh.numel() == 0 and colors_precomp.numel() == 0) or (sh.numel() != 0 and colors_precomp.numel() != 0):
            raise ValueError("Provide exactly one of SHs or precomputed colors")
        if ((scales.numel() == 0 or rotations.numel() == 0) and cov3Ds_precomp.numel() == 0) or (
            (scales.numel() != 0 or rotations.numel() != 0) and cov3Ds_precomp.numel() != 0
        ):
            raise ValueError("Provide exactly one of (scales+rotations) or cov3D_precomp")

        # Ensure expected dtypes/devices/contiguity
        means3D = _ensure_cuda_f32(means3D, "means3D")
        means2D = _ensure_cuda_f32(means2D, "means2D")  # for gradient only
        opacities = _ensure_cuda_f32(opacities, "opacities")
        scales = _ensure_cuda_f32(scales, "scales") if scales.numel() != 0 else _empty(device)
        rotations = _ensure_cuda_f32(rotations, "rotations") if rotations.numel() != 0 else _empty(device)
        cov3Ds_precomp = _ensure_cuda_f32(cov3Ds_precomp, "cov3D_precomp") if cov3Ds_precomp.numel() != 0 else _empty(device)
        colors_precomp = _ensure_cuda_f32(colors_precomp, "colors_precomp") if colors_precomp.numel() != 0 else _empty(device)
        sh = _ensure_cuda_f32(sh, "sh") if sh.numel() != 0 else _empty(device)

        bg = _as_float3_bg(raster_settings.bg, device)
        viewmatrix = _ensure_cuda_f32(raster_settings.viewmatrix, "viewmatrix")
        projmatrix = _ensure_cuda_f32(raster_settings.projmatrix, "projmatrix")
        campos = _ensure_cuda_f32(raster_settings.campos, "campos")

        _check_camera_type(raster_settings.camera_type)

        # Native forward: color first
        args = (
            bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            float(raster_settings.scale_modifier),
            cov3Ds_precomp,
            viewmatrix,
            projmatrix,
            float(raster_settings.tanfovx),
            float(raster_settings.tanfovy),
            int(raster_settings.image_height),
            int(raster_settings.image_width),
            sh,
            int(raster_settings.sh_degree),
            campos,
            bool(raster_settings.prefiltered),
            int(raster_settings.camera_type),
            False,  # render_depth off for color pass
        )
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Save for backward
        ctx.num_rendered = int(num_rendered)
        ctx.raster_settings = raster_settings
        ctx.save_for_backward(
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            opacities,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            bg,
            viewmatrix,
            projmatrix,
            campos,
        )

        # Optional depth: second pass using same inputs, with zero background
        with torch.no_grad():
            zero_bg = torch.zeros_like(bg)
            depth_args = list(args)
            depth_args[0] = zero_bg
            depth_args[-1] = True  # render_depth
            _, depth_color, _, _, _, _ = _C.rasterize_gaussians(*depth_args)
            # Use first channel as scalar depth map (C,H,W) -> (1,H,W)
            depth = depth_color[:1]

        return color, radii, depth

    @staticmethod
    def backward(
        ctx,
        grad_out_color: torch.Tensor,
        _grad_radii: Optional[torch.Tensor],
        grad_out_depth: Optional[torch.Tensor],
    ):
        # We currently do not propagate depth gradients in v0
        del grad_out_depth

        (
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            opacities,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            bg,
            viewmatrix,
            projmatrix,
            campos,
        ) = ctx.saved_tensors

        rs = ctx.raster_settings
        num_rendered = ctx.num_rendered

        args = (
            bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            float(rs.scale_modifier),
            cov3Ds_precomp,
            viewmatrix,
            projmatrix,
            float(rs.tanfovx),
            float(rs.tanfovy),
            _ensure_cuda_f32(grad_out_color, "grad_out_color"),
            sh,
            int(rs.sh_degree),
            campos,
            geomBuffer,
            int(num_rendered),
            binningBuffer,
            imgBuffer,
            int(rs.camera_type),
        )

        (
            dL_dmeans2D,
            dL_dcolors,
            dL_dopacity,
            dL_dmeans3D,
            dL_dcov3D,
            dL_dsh,
            dL_dscales,
            dL_drotations,
        ) = _C.rasterize_gaussians_backward(*args)

        # Match input order of forward()
        grads = (
            dL_dmeans3D,   # means3D
            dL_dmeans2D,   # means2D
            dL_dsh,        # sh
            dL_dcolors,    # colors_precomp
            dL_dopacity,   # opacities
            dL_dscales,    # scales
            dL_drotations, # rotations
            dL_dcov3D,     # cov3Ds_precomp
            None,          # raster_settings
        )
        return grads


# ---------------------------
# High-level wrapper (3DGS-compatible)
# ---------------------------

class GaussianRasterizer(torch.nn.Module):
    def __init__(self, raster_settings: GaussianRasterizationSettings):
        super().__init__()
        if _C is None:  # pragma: no cover
            raise RuntimeError("omnigs_rasterization extension is not built yet") from _IMPORT_ERROR
        self.raster_settings = raster_settings

    def markVisible(self, positions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            rs = self.raster_settings
            _check_camera_type(rs.camera_type)
            return _C.mark_visible(
                _ensure_cuda_f32(positions, "positions"),
                _ensure_cuda_f32(rs.viewmatrix, "viewmatrix"),
                _ensure_cuda_f32(rs.projmatrix, "projmatrix"),
                int(rs.camera_type),
            )

    def forward(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        opacities: torch.Tensor,
        shs: Optional[torch.Tensor] = None,
        colors_precomp: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        rotations: Optional[torch.Tensor] = None,
        cov3D_precomp: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Mutual exclusivity checks
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise ValueError("Please provide exactly one of SHs or precomputed colors!")
        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise ValueError("Please provide exactly one of (scales+rotations) or precomputed 3D covariance!")

        device = means3D.device
        empty = _empty(device)
        shs = _ensure_cuda_f32(shs, "shs") if shs is not None else empty
        colors_precomp = _ensure_cuda_f32(colors_precomp, "colors_precomp") if colors_precomp is not None else empty
        scales = _ensure_cuda_f32(scales, "scales") if scales is not None else empty
        rotations = _ensure_cuda_f32(rotations, "rotations") if rotations is not None else empty
        cov3D_precomp = _ensure_cuda_f32(cov3D_precomp, "cov3D_precomp") if cov3D_precomp is not None else empty

        return rasterize_gaussians(
            _ensure_cuda_f32(means3D, "means3D"),
            _ensure_cuda_f32(means2D, "means2D"),
            shs,
            colors_precomp,
            _ensure_cuda_f32(opacities, "opacities"),
            scales,
            rotations,
            cov3D_precomp,
            self.raster_settings,
        )


def mark_visible(
    positions: torch.Tensor,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    camera_type: int = 1,
) -> torch.Tensor:
    if _C is None:  # pragma: no cover
        raise RuntimeError("omnigs_rasterization extension is not built yet") from _IMPORT_ERROR
    _check_camera_type(camera_type)
    return _C.mark_visible(
        _ensure_cuda_f32(positions, "positions"),
        _ensure_cuda_f32(viewmatrix, "viewmatrix"),
        _ensure_cuda_f32(projmatrix, "projmatrix"),
        int(camera_type),
    )


__all__ = [
    "GaussianRasterizationSettings",
    "GaussianRasterizer",
    "rasterize_gaussians",
    "mark_visible",
]
