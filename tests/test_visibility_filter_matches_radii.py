import math
import sys
import types
import unittest

import torch


# Inject a minimal stub for omnigs_rasterization before importing gaussian_renderer
stub = types.ModuleType("omnigs_rasterization")

class _Settings(tuple):
    pass

def _Settings_ctor(**kwargs):
    return types.SimpleNamespace(**kwargs)

class _StubRasterizer:
    def __init__(self, raster_settings):
        # store settings if needed
        pass

    def __call__(self, *, means3D, means2D, opacities, shs=None, colors_precomp=None, scales=None, rotations=None, cov3D_precomp=None):
        n = means3D.shape[0]
        # radii: mark half as zero to simulate invisibility
        radii = torch.ones(n)
        if n > 0:
            radii[::2] = 0.0
        h, w = 8, 10
        color = torch.zeros(3, h, w)
        depth = torch.zeros(1, h, w)
        return color, radii, depth

stub.GaussianRasterizationSettings = _Settings_ctor
stub.GaussianRasterizer = _StubRasterizer
def _mark_visible(positions, viewmatrix, projmatrix, camera_type):
    return torch.ones(positions.shape[0], dtype=torch.bool)
stub.mark_visible = _mark_visible
sys.modules["omnigs_rasterization"] = stub

# Ensure repo root is on sys.path
import os as _os
_repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import gaussian_renderer as gr


class DummyPC:
    def __init__(self, n=6, sh_degree=0):
        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self._xyz = torch.randn(n, 3)
        self._opacity = torch.sigmoid(torch.randn(n, 1))
        self._scaling = torch.abs(torch.randn(n, 3)) * 0.01 + 1e-3
        self._rotation = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
        self._features_dc = torch.randn(n, 1, 3).transpose(1, 2).contiguous()
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
    def __init__(self):
        self.image_width = 10
        self.image_height = 8
        self.FoVx = math.radians(60.0)
        self.FoVy = math.radians(45.0)
        self.world_view_transform = torch.eye(4)
        self.full_proj_transform = torch.eye(4)
        self.camera_center = torch.zeros(3)
        self.image_name = "dummy"
        self.camera_type = 1


class DummyPipe:
    def __init__(self):
        self.compute_cov3D_python = False
        self.convert_SHs_python = False
        self.debug = False
        self.antialiasing = False


class TestVisibilityFilterMatchesRadii(unittest.TestCase):
    def test_visibility_filter_indices_match_radii_positive(self):
        pc = DummyPC(n=6, sh_degree=0)
        cam = DummyCam()
        pipe = DummyPipe()
        bg = torch.tensor([0.0, 0.0, 0.0])
        out = gr.render(cam, pc, pipe, bg)
        radii = out["radii"]
        vf = out["visibility_filter"].squeeze(-1)  # indices of radii>0
        expected = (radii > 0).nonzero().squeeze(-1)
        self.assertTrue(torch.equal(vf, expected))


if __name__ == "__main__":
    unittest.main()
