"""OmniGS rasterizer PyTorch extension scaffolding (self-contained)."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch

try:  # pragma: no cover - during install the extension is unavailable
    from . import _C
except ImportError as import_error:  # pragma: no cover
    _C = None
    _IMPORT_ERROR = import_error
else:
    _IMPORT_ERROR = None


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: Sequence[float]
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    camera_type: int = 1
    render_depth: bool = False


class GaussianRasterizer(torch.nn.Module):
    """Thin wrapper that will delegate to the CUDA extension once available."""

    def forward(self, *args, **kwargs):  # type: ignore[override]
        if _C is None:
            raise RuntimeError(
                "omnigs_rasterization extension is not built yet"  # pragma: no cover
            ) from _IMPORT_ERROR
        raise NotImplementedError(
            "Forward scattering into the CUDA extension is pending implementation"
        )


def mark_visible(*args, **kwargs):
    if _C is None:
        raise RuntimeError(
            "omnigs_rasterization extension is not built yet"
        ) from _IMPORT_ERROR
    raise NotImplementedError("mark_visible binding will land with the core implementation")


__all__ = [
    "GaussianRasterizationSettings",
    "GaussianRasterizer",
    "mark_visible",
]

