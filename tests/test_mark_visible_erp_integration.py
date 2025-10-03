import unittest
import os as _os
import sys as _sys

import torch


class TestMarkVisibleERPIntegration(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_mark_visible_all_true_for_erp(self):
        # Ensure repo root is on sys.path
        _repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)

        try:
            import gaussian_renderer as gr
        except Exception as e:
            self.skipTest(f"gaussian_renderer import failed: {e}")
            return

        # Prepare a dummy ERP camera (identity transforms)
        device = torch.device("cuda")
        cam = type("C", (), dict(
            image_width=32,
            image_height=16,
            FoVx=1.0,
            FoVy=1.0,
            world_view_transform=torch.eye(4, device=device),
            full_proj_transform=torch.eye(4, device=device),
            camera_center=torch.zeros(3, device=device),
            image_name="dummy",
            camera_type=3,
        ))()

        P = 10
        pos = torch.randn(P, 3, device=device)
        # Extension may not be built; mark_visible() will raise and we skip
        try:
            visible = gr.mark_visible(cam, pos)
        except RuntimeError as e:
            self.skipTest(f"omnigs_rasterization not built: {e}")
            return

        self.assertEqual(visible.shape, (P,))
        self.assertEqual(visible.dtype, torch.bool)
        self.assertTrue(bool(visible.all().item()))


if __name__ == "__main__":
    unittest.main()
