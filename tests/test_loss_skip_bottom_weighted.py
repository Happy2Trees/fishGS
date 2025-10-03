import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.loss_utils import erp_latitude_weight_map, weighted_l1, erp_skip_bottom


def test_erp_skip_bottom_crop():
    img = torch.arange(0, 3*10*4, dtype=torch.float32).view(3, 10, 4)
    # crop bottom 50%
    cropped = erp_skip_bottom(img, 0.5)
    assert tuple(cropped.shape) == (3, 5, 4)
    # crop zero -> no-op
    same = erp_skip_bottom(img, 0.0)
    assert torch.allclose(same, img)


def test_weighted_l1_shapes_and_values():
    C, H, W = 3, 8, 6
    a = torch.ones(C, H, W)
    b = torch.zeros(C, H, W)
    w = erp_latitude_weight_map(H, W, device=a.device, dtype=a.dtype)
    # Weighted L1 should be between 0 and 1 for ones-vs-zeros
    loss = weighted_l1(a, b, w)
    assert 0.0 <= loss.item() <= 1.0
    # If we crop bottom half, new weight map adapts to new H and loss stays finite
    a2 = erp_skip_bottom(a, 0.5)
    b2 = erp_skip_bottom(b, 0.5)
    w2 = erp_latitude_weight_map(a2.shape[1], a2.shape[2], device=a.device, dtype=a.dtype)
    loss2 = weighted_l1(a2, b2, w2)
    assert 0.0 <= loss2.item() <= 1.0

