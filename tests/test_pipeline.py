"""Small shape and protocol checks for the actual image pipeline."""

import numpy as np
import torch
from PIL import Image

from sharpdistill.data import make_pair
from sharpdistill.metrics import scores
from sharpdistill.student import StudentNet


def test_student_keeps_full_resolution_and_identity_at_start():
    model = StudentNet(channels=8, blocks=2)
    x = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape
    assert torch.equal(x, y)


def test_degradation_pair_and_metrics():
    image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (65, 97, 3), dtype=np.uint8))
    degraded, low, target = make_pair(image)
    assert degraded.shape == target.shape == (3, 64, 96)
    assert low.shape == (3, 32, 48)
    assert scores(target, target)["ssim_rgb"] == 1.0
    assert scores(degraded, target)["ssim_rgb"] < 1.0

