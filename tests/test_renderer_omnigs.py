import math
import sys
import types
import unittest
from typing import NamedTuple

import torch

# Inject a minimal stub for omnigs_rasterization before importing gaussian_renderer
stub = types.ModuleType("omnigs_rasterization")

class _Settings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor | list[float]
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    camera_type: int = 1
    render_depth: bool = False

class _StubRasterizer:
    def __init__(self, raster_settings: _Settings):
        raise RuntimeError("Stubbed rasterizer should be monkeypatched in tests")

stub.GaussianRasterizationSettings = _Settings
stub.GaussianRasterizer = _StubRasterizer
sys.modules["omnigs_rasterization"] = stub

import gaussian_renderer as gr


class DummyPC:
    def __init__(self, n=5, sh_degree=0):
        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self._xyz = torch.randn(n, 3)
        self._opacity = torch.sigmoid(torch.randn(n, 1))
        self._scaling = torch.abs(torch.randn(n, 3)) * 0.01 + 1e-3
        self._rotation = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
        # features (N, 3, (sh_degree+1)^2)
        sh_dim = (sh_degree + 1) ** 2
        self._features_dc = torch.randn(n, 1, 3).transpose(1, 2).contiguous()  # (N,3,1)
        if sh_dim > 1:
            self._features_rest = torch.randn(n, sh_dim - 1, 3).transpose(1, 2).contiguous()  # (N,3,sh_dim-1)
        else:
            self._features_rest = torch.empty(n, 3, 0)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_features(self):
        return torch.cat([self._features_dc, self._features_rest], dim=2)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    def get_covariance(self, scaling_modifier):
        # not used when pipe.compute_cov3D_python=False
        return torch.eye(3).repeat(self._xyz.shape[0], 1, 1)


class DummyCam:
    def __init__(self, w=64, h=48, fovx=math.radians(60.0), fovy=math.radians(45.0), camera_type=1):
        self.image_width = w
        self.image_height = h
        self.FoVx = fovx
        self.FoVy = fovy
        self.world_view_transform = torch.eye(4)
        self.full_proj_transform = torch.eye(4)
        self.camera_center = torch.zeros(3)
        self.image_name = "dummy"
        self.camera_type = camera_type


class DummyPipe:
    def __init__(self):
        self.compute_cov3D_python = False
        self.convert_SHs_python = False
        self.debug = False
        self.antialiasing = False


class FakeRasterizer:
    def __init__(self, raster_settings):
        # capture settings for assertions
        gr._captured_rs = raster_settings

    def __call__(self, *, means3D, means2D, opacities, shs=None, colors_precomp=None, scales=None, rotations=None, cov3D_precomp=None):
        h = gr._captured_rs.image_height
        w = gr._captured_rs.image_width
        n = means3D.shape[0]
        color = torch.zeros(3, h, w)
        depth = torch.zeros(1, h, w)
        radii = torch.ones(n)
        return color, radii, depth


class TestOmniGSRasterizerIntegration(unittest.TestCase):
    def setUp(self):
        # patch GaussianRasterizer and torch.zeros_like in module
        self._orig_rast = gr.GaussianRasterizer
        gr.GaussianRasterizer = FakeRasterizer
        # proxy torch to override zeros_like without touching global torch
        self._orig_torch = gr.torch

        class TorchProxy:
            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            def zeros_like(self, x, dtype=None, requires_grad=False, device=None):
                return torch.zeros_like(x, dtype=dtype, requires_grad=requires_grad)

        gr.torch = TorchProxy(torch)

    def tearDown(self):
        gr.GaussianRasterizer = self._orig_rast
        gr.torch = self._orig_torch
        if hasattr(gr, "_captured_rs"):
            delattr(gr, "_captured_rs")

    def test_pinhole_camera_type_and_fovs(self):
        pc = DummyPC(n=5, sh_degree=0)
        cam = DummyCam(camera_type=1)
        pipe = DummyPipe()
        bg = torch.tensor([0.0, 0.0, 0.0])
        out = gr.render(cam, pc, pipe, bg)
        self.assertIn("render", out)
        rs = gr._captured_rs
        self.assertEqual(rs.camera_type, 1)
        self.assertAlmostEqual(rs.tanfovx, math.tan(cam.FoVx * 0.5), places=6)
        self.assertAlmostEqual(rs.tanfovy, math.tan(cam.FoVy * 0.5), places=6)

    def test_erp_camera_type_and_tanfov_defaults(self):
        pc = DummyPC(n=3, sh_degree=0)
        cam = DummyCam(camera_type=3)
        pipe = DummyPipe()
        bg = torch.tensor([1.0, 1.0, 1.0])
        _ = gr.render(cam, pc, pipe, bg)
        rs = gr._captured_rs
        self.assertEqual(rs.camera_type, 3)
        self.assertAlmostEqual(rs.tanfovx, 1.0, places=6)
        self.assertAlmostEqual(rs.tanfovy, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
