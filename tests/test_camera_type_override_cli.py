import unittest
from argparse import ArgumentParser

from arguments import ModelParams
from scene.dataset_readers import CameraInfo
from utils.camera_utils import cameraList_from_camInfos


class DummyArgs:
    def __init__(self, camera_type: str):
        self.camera_type = camera_type
        self.resolution = -1
        self.data_device = "cuda"
        self.train_test_exp = False


def _make_caminfo(camera_type: int = 1) -> CameraInfo:
    import numpy as np
    from PIL import Image
    import tempfile, os

    # create a small temporary image file
    tmpdir = tempfile.mkdtemp(prefix="cam_override_")
    img_path = os.path.join(tmpdir, "im.png")
    Image.new("RGB", (16, 16), color=(0, 0, 0)).save(img_path)

    return CameraInfo(
        uid=0,
        R=np.eye(3),
        T=np.zeros(3),
        FovY=1.0,
        FovX=1.0,
        depth_params=None,
        image_path=img_path,
        image_name="dummy",
        depth_path="",
        width=16,
        height=16,
        is_test=False,
        camera_type=camera_type,
    )


class TestCameraTypeOverride(unittest.TestCase):
    def test_override_lonlat(self):
        args = DummyArgs(camera_type="lonlat")
        cams = cameraList_from_camInfos([_make_caminfo(1)], 1.0, args, is_nerf_synthetic=False, is_test_dataset=False)
        self.assertEqual(getattr(cams[0], "camera_type", 1), 3)

    def test_override_pinhole(self):
        args = DummyArgs(camera_type="pinhole")
        cams = cameraList_from_camInfos([_make_caminfo(3)], 1.0, args, is_nerf_synthetic=False, is_test_dataset=False)
        self.assertEqual(getattr(cams[0], "camera_type", 1), 1)

    def test_auto_keeps_dataset(self):
        args = DummyArgs(camera_type="auto")
        cams = cameraList_from_camInfos([_make_caminfo(3)], 1.0, args, is_nerf_synthetic=False, is_test_dataset=False)
        self.assertEqual(getattr(cams[0], "camera_type", 1), 3)


if __name__ == "__main__":
    unittest.main()
