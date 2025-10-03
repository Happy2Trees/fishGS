import io
import sys
import unittest
import numpy as np
import random
import torch

from utils.general_utils import safe_state


class TestSeedArgument(unittest.TestCase):
    def test_safe_state_applies_seed(self):
        # Preserve stdout and restore after to avoid side-effects
        old_stdout = sys.stdout
        try:
            safe_state(True, seed=123)
            # restore stdout for test output
            sys.stdout = old_stdout

            # Check reproducibility across libs
            self.assertEqual(random.randint(0, 100000), 6863)
            self.assertAlmostEqual(float(np.random.rand(1)[0]), 0.6964691855978616, places=6)
            t = torch.randint(0, 10, (3,))
            self.assertTrue(torch.equal(t, torch.tensor([2, 9, 2])))
        finally:
            sys.stdout = old_stdout


if __name__ == "__main__":
    unittest.main()
