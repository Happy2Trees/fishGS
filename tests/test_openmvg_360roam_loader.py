import os
import sys
import json
import unittest
from pathlib import Path

import numpy as np

# Ensure repo root on sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scene.dataset_readers import readOpenMVG360SceneInfo, CameraInfo, getNerfppNorm


def _load_train_test_names(scene_root: str, use_eval: bool):
    train_list = os.path.join(scene_root, "train.txt")
    test_list = os.path.join(scene_root, "test.txt")
    def _read_list(p):
        if not os.path.isfile(p):
            return set()
        return {line.strip() for line in open(p, 'r', encoding='utf-8') if line.strip()}
    train = _read_list(train_list)
    test = _read_list(test_list) if use_eval else set()
    return train, test


def _parse_openmvg_views(scene_root: str):
    views_json = os.path.join(scene_root, "data_views.json")
    extr_json = os.path.join(scene_root, "data_extrinsics.json")
    with open(views_json, 'r', encoding='utf-8') as f:
        vobj = json.load(f)
    with open(extr_json, 'r', encoding='utf-8') as f:
        eobj = json.load(f)
    views = vobj.get("views", [])
    extr = {int(it.get("key")): it.get("value", {}) for it in eobj.get("extrinsics", [])}
    # return list of tuples: (filename, id_pose, has_extr)
    out = []
    for it in views:
        data = (((it.get("value") or {}).get("ptr_wrapper") or {}).get("data") or {})
        filename = data.get("filename")
        id_pose = data.get("id_pose")
        if filename is None or id_pose is None:
            continue
        has_extr = int(id_pose) in extr
        out.append((filename, int(id_pose), has_extr))
    return out


class TestOpenMVG360RoamLoader(unittest.TestCase):
    SCENE = "data/360Roam/lab"

    @unittest.skipUnless(os.path.isdir(SCENE), "360Roam sample scene not found")
    def test_eval_true_split_counts(self):
        # eval=True should honor test.txt
        si = readOpenMVG360SceneInfo(self.SCENE, eval=True, train_test_exp=False)
        self.assertGreater(len(si.train_cameras), 0)
        self.assertGreater(len(si.test_cameras), 0)
        # expected counts from lists and extrinsics
        train_names, test_names = _load_train_test_names(self.SCENE, use_eval=True)
        allowed = train_names.union(test_names)
        views = _parse_openmvg_views(self.SCENE)
        expected_train = 0
        expected_test = 0
        for filename, _, has_extr in views:
            stem = Path(filename).stem
            if stem not in allowed or not has_extr:
                continue
            if stem in test_names:
                expected_test += 1
            else:
                expected_train += 1
        self.assertEqual(len(si.train_cameras), expected_train)
        self.assertEqual(len(si.test_cameras), expected_test)

    @unittest.skipUnless(os.path.isdir(SCENE), "360Roam sample scene not found")
    def test_eval_false_merged_counts(self):
        # eval=False should ignore test.txt (all allowed into train)
        si = readOpenMVG360SceneInfo(self.SCENE, eval=False, train_test_exp=False)
        self.assertGreater(len(si.train_cameras), 0)
        self.assertEqual(len(si.test_cameras), 0)
        train_names, test_names = _load_train_test_names(self.SCENE, use_eval=False)
        allowed = train_names.union(test_names) if (train_names or test_names) else None
        views = _parse_openmvg_views(self.SCENE)
        expected_train = 0
        for filename, _, has_extr in views:
            stem = Path(filename).stem
            if allowed is not None and stem not in allowed:
                continue
            if not has_extr:
                continue
            expected_train += 1
        self.assertEqual(len(si.train_cameras), expected_train)

    @unittest.skipUnless(os.path.isdir(SCENE), "360Roam sample scene not found")
    def test_camera_type_and_pcd_usage(self):
        si = readOpenMVG360SceneInfo(self.SCENE, eval=True, train_test_exp=False)
        self.assertTrue(os.path.isfile(si.ply_path))
        self.assertIsNotNone(si.point_cloud)
        self.assertTrue(hasattr(si.point_cloud, 'points'))
        self.assertGreater(np.asarray(si.point_cloud.points).shape[0], 0)
        # ERP camera_type must be 3 for all views
        for cam in (si.train_cameras + si.test_cameras):
            self.assertEqual(getattr(cam, "camera_type", 1), 3)

    @unittest.skipUnless(os.path.isdir(SCENE), "360Roam sample scene not found")
    def test_extent_matches_camera_based_norm(self):
        si = readOpenMVG360SceneInfo(self.SCENE, eval=True, train_test_exp=False)
        # recompute via getNerfppNorm on the returned CameraInfo list
        nn = getNerfppNorm(si.train_cameras)
        self.assertIn("radius", nn)
        self.assertAlmostEqual(float(nn["radius"]), float(si.nerf_normalization["radius"]), places=5)


if __name__ == '__main__':
    unittest.main()
