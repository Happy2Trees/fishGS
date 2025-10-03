import math
import unittest

import torch


class TestOmniGSCUDASmoke(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_forward_pinhole(self):
        import gaussian_renderer as gr

        device = torch.device("cuda")
        N = 64

        class PC:
            def __init__(self):
                self.max_sh_degree = 0
                self.active_sh_degree = 0
                self._xyz = torch.randn(N, 3, device=device)
                self._opacity = torch.rand(N, 1, device=device)
                self._scaling = torch.rand(N, 3, device=device) * 0.01 + 1e-3
                q = torch.randn(N, 4, device=device)
                self._rotation = q / q.norm(dim=-1, keepdim=True)
                self._features = torch.zeros(N, 3, 1, device=device)

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
                return self._features

        pc = PC()
        cam = type("C", (), dict(
            image_width=160,
            image_height=120,
            FoVx=math.radians(60.0),
            FoVy=math.radians(45.0),
            world_view_transform=torch.eye(4, device=device),
            full_proj_transform=torch.eye(4, device=device),
            camera_center=torch.zeros(3, device=device),
            image_name="dummy",
            camera_type=1,
        ))()

        pipe = type("P", (), dict(
            compute_cov3D_python=False,
            convert_SHs_python=False,
            debug=False,
            antialiasing=False,
        ))()

        bg = torch.zeros(3, device=device)

        # If the extension is not built, GaussianRasterizer init raises RuntimeError
        try:
            out = gr.render(cam, pc, pipe, bg)
        except RuntimeError as e:
            self.skipTest(f"omnigs_rasterization not built: {e}")
            return

        img = out["render"]
        depth = out["depth"]
        self.assertEqual(tuple(img.shape), (3, 120, 160))
        self.assertEqual(tuple(depth.shape), (1, 120, 160))


if __name__ == "__main__":
    unittest.main()

