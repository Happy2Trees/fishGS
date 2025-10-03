import os
import sys

# Ensure repository root is on path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train import compute_size_threshold


def test_compute_size_threshold_matches_omnigs_rule():
    # prune_big_point_after_iter == 0 => always disabled
    assert compute_size_threshold(1, 0) == 0
    assert compute_size_threshold(10_000, 0) == 0

    # enable only strictly after threshold
    thresh = 30000
    assert compute_size_threshold(thresh, thresh) == 0
    assert compute_size_threshold(thresh - 1, thresh) == 0
    assert compute_size_threshold(thresh + 1, thresh) == 20
    # larger iterations stay enabled
    for it in [thresh + 10, thresh + 1000, thresh * 2]:
        assert compute_size_threshold(it, thresh) == 20

