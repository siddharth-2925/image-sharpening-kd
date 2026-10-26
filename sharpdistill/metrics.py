"""RGB full-image metrics with consistent data range and crop convention."""

import math

import numpy as np
import torch
from skimage.metrics import structural_similarity


def to_rgb_uint8(image: torch.Tensor) -> np.ndarray:
    return (image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def scores(output: torch.Tensor, target: torch.Tensor, crop_border: int = 2) -> dict[str, float]:
    estimate = to_rgb_uint8(output)
    truth = to_rgb_uint8(target)
    if crop_border:
        estimate = estimate[crop_border:-crop_border, crop_border:-crop_border]
        truth = truth[crop_border:-crop_border, crop_border:-crop_border]
    if min(estimate.shape[:2]) < 7:
        raise ValueError("Images too small for SSIM")
    difference = estimate.astype(np.float64) - truth.astype(np.float64)
    mse = np.mean(difference ** 2)
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)
    ssim = structural_similarity(estimate, truth, channel_axis=2, data_range=255)
    return {"psnr_rgb_db": psnr, "ssim_rgb": float(ssim)}

