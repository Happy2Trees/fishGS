import math
import os
import shutil
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from scene.dataset_readers import CameraInfo
from utils import camera_utils
from utils.loss_utils import erp_latitude_weight_map


class DummyArgs:
    def __init__(self):
        self.resolution = -1
        self.train_test_exp = False
        self.data_device = "cuda" if torch.cuda.is_available() else "cpu"


class TestCameraDataPipelineERP(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="fishgs_cam_")
        # Create a small 2:1 ERP-like image (e.g., 128x64)
        w, h = 128, 64
        img = Image.fromarray((np.ones((h, w, 3), dtype=np.uint8) * 127))
        self.img_path = os.path.join(self.tmpdir, "frame000.png")
        img.save(self.img_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_camera_type_propagation(self):
        # Build CameraInfo with camera_type=3 (ERP)
        R = np.eye(3)
        T = np.zeros(3)
        fovx = math.radians(60.0)
        fovy = math.radians(45.0)
        c = CameraInfo(
            uid=0,
            R=R,
            T=T,
            FovY=fovy,
            FovX=fovx,
            depth_params=None,
            image_path=self.img_path,
            image_name="frame000",
            depth_path="",
            width=128,
            height=64,
            is_test=False,
            camera_type=3,
        )

        args = DummyArgs()
        cam = camera_utils.loadCam(args, id=0, cam_info=c, resolution_scale=1.0, is_nerf_synthetic=False, is_test_dataset=False)
        self.assertEqual(getattr(cam, "camera_type", 1), 3)
        self.assertEqual(cam.image_width, 128)
        self.assertEqual(cam.image_height, 64)

    def test_erp_latitude_weight_map(self):
        H, W = 64, 128
        w = erp_latitude_weight_map(H, W, device="cpu")
        self.assertEqual(tuple(w.shape), (1, 1, H, W))
        # Center row should have larger weight than top/bottom rows
        mid = w[0, 0, H // 2, 0].item()
        top = w[0, 0, 0, 0].item()
        bot = w[0, 0, H - 1, 0].item()
        self.assertGreater(mid, top)
        self.assertGreater(mid, bot)
        # Non-negativity
        self.assertGreaterEqual(w.min().item(), 0.0)


if __name__ == "__main__":
    unittest.main()

