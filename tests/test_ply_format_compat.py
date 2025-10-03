import os
import sys
import unittest

from plyfile import PlyData
import torch


class TestPLYFormatCompat(unittest.TestCase):
    def test_saved_ply_has_omnigs_fields(self):
        # Build a minimal GaussianModel with deterministic small tensors
        # Ensure repo root in path
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from scene.gaussian_model import GaussianModel

        N = 4
        sh_degree = 1  # so (sh+1)^2 = 4, f_rest has 3 coeffs per channel
        gm = GaussianModel(sh_degree)

        gm._xyz = torch.arange(N * 3, dtype=torch.float32).view(N, 3)
        # features_dc: (N,1,3)
        gm._features_dc = torch.arange(N * 3, dtype=torch.float32).view(N, 1, 3)
        # features_rest: (N, (4-1), 3) = (N,3,3)
        gm._features_rest = torch.arange(N * 9, dtype=torch.float32).view(N, 3, 3)
        gm._opacity = torch.full((N, 1), 0.5, dtype=torch.float32)
        gm._scaling = torch.ones(N, 3, dtype=torch.float32)
        gm._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(N, 1)

        # Save ply to tmp path
        out_path = os.path.abspath("./tmp_test_point_cloud.ply")
        try:
            gm.save_ply(out_path)
            ply = PlyData.read(out_path)
            vert = ply.elements[0]

            # Check required properties exist and counts match
            required = [
                "x", "y", "z",
                "nx", "ny", "nz",
                "f_dc_0", "f_dc_1", "f_dc_2",
                "opacity",
                "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3",
            ]
            for name in required:
                self.assertIn(name, vert.data.dtype.names)

            # f_rest length should be (4-1)*3 = 9
            f_rest_names = [n for n in vert.data.dtype.names if n.startswith("f_rest_")]
            self.assertEqual(len(f_rest_names), 9)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)


if __name__ == "__main__":
    unittest.main()
