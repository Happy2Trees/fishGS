import unittest
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams


class TestCLINewArgs(unittest.TestCase):
    def test_defaults_and_overrides(self):
        parser = ArgumentParser()
        mp = ModelParams(parser)
        pp = PipelineParams(parser)

        # Defaults
        args = parser.parse_args([])
        self.assertEqual(args.camera_type, "auto")
        self.assertEqual(args.rasterizer, "omnigs")

        # Overrides
        args = parser.parse_args(["--camera_type", "lonlat", "--rasterizer", "diff"])
        self.assertEqual(args.camera_type, "lonlat")
        self.assertEqual(args.rasterizer, "diff")


if __name__ == "__main__":
    unittest.main()

