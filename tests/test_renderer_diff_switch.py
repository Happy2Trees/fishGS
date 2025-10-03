import math
import sys
import types
import unittest
from typing import NamedTuple

import torch

# Provide a stub for diff_gaussian_rasterization before importing gaussian_renderer
diff_stub = types.ModuleType("diff_gaussian_rasterization")


class DiffSettings(NamedTuple):
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
    debug: bool
    antialiasing: bool


class _StubRasterizer:
    def __init__(self, raster_settings: DiffSettings):
        # capture settings for assertions
        import gaussian_renderer as gr
        gr._captured_rs = raster_settings

    def __call__(
        self,
        *,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):
        import gaussian_renderer as gr
        h = gr._captured_rs.image_height
        w = gr._captured_rs.image_width
        n = means3D.shape[0]
        color = torch.zeros(3, h, w)
        depth = torch.zeros(1, h, w)
        radii = torch.ones(n)
        return color, radii, depth


diff_stub.GaussianRasterizationSettings = DiffSettings
diff_stub.GaussianRasterizer = _StubRasterizer
sys.modules["diff_gaussian_rasterization"] = diff_stub

import gaussian_renderer as gr


class DummyPC:
    def __init__(self, n=4, sh_degree=0):
        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self._xyz = torch.randn(n, 3)
        self._opacity = torch.sigmoid(torch.randn(n, 1))
        self._scaling = torch.abs(torch.randn(n, 3)) * 0.01 + 1e-3
        self._rotation = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
        sh_dim = (sh_degree + 1) ** 2
        self._features_dc = torch.randn(n, 1, 3).transpose(1, 2).contiguous()
        if sh_dim > 1:
            self._features_rest = torch.randn(n, sh_dim - 1, 3).transpose(1, 2).contiguous()
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


class DummyCam:
    def __init__(self, w=32, h=24, fovx=math.radians(50.0), fovy=math.radians(40.0), camera_type=1):
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
        self.debug = True
        self.antialiasing = False
        self.rasterizer = "diff"


class TestDiffRasterizerSwitch(unittest.TestCase):
    def setUp(self):
        # patch torch.zeros_like behavior used in gaussian_renderer to not require cuda
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
        gr.torch = self._orig_torch
        if hasattr(gr, "_captured_rs"):
            delattr(gr, "_captured_rs")

    def test_diff_selected_and_settings_shape(self):
        pc = DummyPC(n=6, sh_degree=0)
        cam = DummyCam(camera_type=1)
        pipe = DummyPipe()
        bg = torch.tensor([0.0, 0.0, 0.0])

        out = gr.render(cam, pc, pipe, bg)
        # Ensure dictionary keys and shapes look right
        self.assertIn("render", out)
        self.assertIn("radii", out)
        self.assertIn("depth", out)
        self.assertEqual(out["render"].shape[0], 3)
        self.assertEqual(out["depth"].shape[0], 1)
        self.assertEqual(out["radii"].shape[0], pc.get_xyz.shape[0])

        # Captured settings should be from diff stub
        rs = getattr(gr, "_captured_rs")
        self.assertIsInstance(rs, DiffSettings)
        self.assertEqual(rs.debug, True)
        self.assertEqual(rs.antialiasing, False)


if __name__ == "__main__":
    unittest.main()

