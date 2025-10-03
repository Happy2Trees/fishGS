from __future__ import annotations

import math
import sys
from typing import Tuple

import torch


def _device() -> torch.device:
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available; ERP wrapper test requires CUDA.")
        sys.exit(0)
    return torch.device("cuda")


def _rand_inputs(P: int, M: int, H: int, W: int, device: torch.device):
    # Random but sane inputs for ERP camera
    means3D = torch.randn(P, 3, device=device, dtype=torch.float32)
    means2D = torch.zeros(P, 3, device=device, dtype=torch.float32, requires_grad=True)
    opacities = torch.sigmoid(torch.randn(P, 1, device=device, dtype=torch.float32))
    scales = torch.exp(torch.randn(P, 3, device=device, dtype=torch.float32) * 0.1)
    rotations = torch.nn.functional.normalize(torch.randn(P, 4, device=device, dtype=torch.float32), dim=-1)
    cov3D_precomp = torch.empty(0, device=device, dtype=torch.float32)
    colors_precomp = torch.empty(0, device=device, dtype=torch.float32)
    sh = torch.empty(0, device=device, dtype=torch.float32)

    viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
    projmatrix = torch.eye(4, device=device, dtype=torch.float32)
    campos = torch.zeros(3, device=device, dtype=torch.float32)
    bg = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)

    return {
        "means3D": means3D,
        "means2D": means2D,
        "opacities": opacities,
        "scales": scales,
        "rotations": rotations,
        "cov3D_precomp": cov3D_precomp,
        "colors_precomp": colors_precomp,
        "sh": sh,
        "viewmatrix": viewmatrix,
        "projmatrix": projmatrix,
        "campos": campos,
        "bg": bg,
        "H": H,
        "W": W,
        "M": M,
    }


def test_forward_backward_erp() -> None:
    from omnigs_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    device = _device()
    P, degree = 128, 3
    M = (degree + 1) ** 2
    H, W = 64, 64
    data = _rand_inputs(P, M, H, W, device)

    # ERP camera settings
    settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=data["bg"],
        scale_modifier=1.0,
        viewmatrix=data["viewmatrix"],
        projmatrix=data["projmatrix"],
        sh_degree=degree,
        campos=data["campos"],
        prefiltered=False,
        camera_type=3,  # LONLAT (ERP)
        render_depth=False,
    )

    raster = GaussianRasterizer(settings)

    # SH path (enable grads before forward)
    sh = torch.randn(P, M, 3, device=device, dtype=torch.float32)
    data["means3D"].requires_grad_(True)
    data["opacities"].requires_grad_(True)
    color, radii, depth = raster(
        means3D=data["means3D"],
        means2D=data["means2D"],
        opacities=data["opacities"],
        shs=sh,
        colors_precomp=None,
        scales=data["scales"],
        rotations=data["rotations"],
        cov3D_precomp=None,
    )
    assert color.shape == (3, H, W)
    assert radii.shape == (P,)
    assert depth.shape[1:] == (H, W)
    assert torch.isfinite(color).all() and torch.isfinite(radii.float()).all() and torch.isfinite(depth).all()

    # Backward smoke (color only)
    loss = color.mean()
    loss.backward()
    assert data["means3D"].grad is not None and torch.isfinite(data["means3D"].grad).all()
    assert data["opacities"].grad is not None and torch.isfinite(data["opacities"].grad).all()

    # colors_precomp path
    colors = torch.rand(P, 3, device=device, dtype=torch.float32)
    color2, radii2, depth2 = raster(
        means3D=data["means3D"].detach(),
        means2D=torch.zeros_like(data["means2D"]).detach(),
        opacities=data["opacities"].detach(),
        shs=None,
        colors_precomp=colors,
        scales=data["scales"].detach(),
        rotations=data["rotations"].detach(),
        cov3D_precomp=None,
    )
    assert color2.shape == (3, H, W)
    assert radii2.shape == (P,)
    assert depth2.shape[1:] == (H, W)
    assert torch.isfinite(color2).all() and torch.isfinite(radii2.float()).all() and torch.isfinite(depth2).all()

    print("[OK] ERP forward/backward/variants passed.")


def test_mark_visible_erp() -> None:
    from omnigs_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    device = _device()
    P, degree = 64, 2
    M = (degree + 1) ** 2
    H, W = 32, 48
    data = _rand_inputs(P, M, H, W, device)

    settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=data["bg"],
        scale_modifier=1.0,
        viewmatrix=data["viewmatrix"],
        projmatrix=data["projmatrix"],
        sh_degree=degree,
        campos=data["campos"],
        prefiltered=False,
        camera_type=3,
        render_depth=False,
    )
    raster = GaussianRasterizer(settings)
    visible = raster.markVisible(data["means3D"].detach())
    # ERP markVisible marks all as visible
    assert visible.shape == (P,) and visible.dtype == torch.bool and bool(visible.all().item())
    print("[OK] ERP markVisible passed.")


def test_argument_validation() -> None:
    from omnigs_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    device = _device()
    P, degree = 8, 1
    M = (degree + 1) ** 2
    H, W = 8, 8
    data = _rand_inputs(P, M, H, W, device)

    settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=data["bg"],
        scale_modifier=1.0,
        viewmatrix=data["viewmatrix"],
        projmatrix=data["projmatrix"],
        sh_degree=degree,
        campos=data["campos"],
        prefiltered=False,
        camera_type=3,
        render_depth=False,
    )
    raster = GaussianRasterizer(settings)

    # Both SH and colors -> error
    try:
        raster(
            means3D=data["means3D"],
            means2D=data["means2D"],
            opacities=data["opacities"],
            shs=torch.zeros(P, M, 3, device=device),
            colors_precomp=torch.zeros(P, 3, device=device),
            scales=data["scales"],
            rotations=data["rotations"],
            cov3D_precomp=None,
        )
        raise AssertionError("Expected ValueError for SH+colors_precomp")
    except ValueError:
        pass

    # Neither SH nor colors -> error
    try:
        raster(
            means3D=data["means3D"],
            means2D=data["means2D"],
            opacities=data["opacities"],
            shs=None,
            colors_precomp=None,
            scales=data["scales"],
            rotations=data["rotations"],
            cov3D_precomp=None,
        )
        raise AssertionError("Expected ValueError for neither SH nor colors_precomp")
    except ValueError:
        pass

    # Both scales/rotations and cov3D_precomp -> error
    try:
        raster(
            means3D=data["means3D"],
            means2D=data["means2D"],
            opacities=data["opacities"],
            shs=torch.zeros(P, M, 3, device=device),
            colors_precomp=None,
            scales=data["scales"],
            rotations=data["rotations"],
            cov3D_precomp=torch.zeros(P, 6, device=device),
        )
        raise AssertionError("Expected ValueError for both scales/rot and cov3D_precomp")
    except ValueError:
        pass

    # Neither scales/rotations nor cov3D_precomp -> error
    try:
        raster(
            means3D=data["means3D"],
            means2D=data["means2D"],
            opacities=data["opacities"],
            shs=torch.zeros(P, M, 3, device=device),
            colors_precomp=None,
            scales=None,
            rotations=None,
            cov3D_precomp=None,
        )
        raise AssertionError("Expected ValueError for neither scales/rot nor cov3D_precomp")
    except ValueError:
        pass

    print("[OK] Argument validation passed.")


def main() -> None:
    # Import smoke of extension
    try:
        from omnigs_rasterization import _C  # type: ignore
        print("[OK] Extension loaded:", _C.__name__)
    except Exception as e:
        print("[FAIL] Extension import:", e)
        raise

    test_forward_backward_erp()
    test_mark_visible_erp()
    test_argument_validation()
    print("[DONE] ERP wrapper tests completed.")


if __name__ == "__main__":
    main()
