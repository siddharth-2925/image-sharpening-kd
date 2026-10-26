"""DIV2K image pairs and deterministic x2 degradation."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def image_paths(folder: Path) -> list[Path]:
    paths = sorted(path for path in folder.iterdir() if path.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f"No images found in {folder}")
    return paths


def rgb_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def make_pair(sharp: Image.Image, method: str = "bicubic") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return full-size degraded input, original x2 low resolution, and sharp target."""
    sharp = sharp.convert("RGB")
    width, height = sharp.size
    width -= width % 2
    height -= height % 2
    sharp = sharp.crop((0, 0, width, height))
    resampling = Image.Resampling.BICUBIC if method == "bicubic" else Image.Resampling.BILINEAR
    low = sharp.resize((width // 2, height // 2), resampling)
    degraded = low.resize((width, height), resampling)
    return rgb_tensor(degraded), rgb_tensor(low), rgb_tensor(sharp)


class PatchDataset(Dataset):
    def __init__(self, paths: list[Path], patch_size: int = 256, training: bool = True,
                 method: str = "bicubic", repeats: int = 1):
        if patch_size < 32 or patch_size % 2:
            raise ValueError("patch_size must be even and at least 32")
        if method not in {"bicubic", "bilinear", "mixed"}:
            raise ValueError("method must be bicubic, bilinear, or mixed")
        self.paths = paths
        self.patch_size = patch_size
        self.training = training
        self.method = method
        self.repeats = repeats

    def __len__(self) -> int:
        return len(self.paths) * self.repeats

    def __getitem__(self, index: int):
        path = self.paths[index % len(self.paths)]
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        size = self.patch_size
        if min(width, height) < size:
            scale = size / min(width, height)
            image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
            width, height = image.size
        if self.training:
            left = random.randrange((width - size) // 2 + 1) * 2
            top = random.randrange((height - size) // 2 + 1) * 2
            image = image.crop((left, top, left + size, top + size))
            if random.getrandbits(1):
                image = ImageOps.mirror(image)
            if random.getrandbits(1):
                image = ImageOps.flip(image)
            if random.getrandbits(1):
                image = image.transpose(Image.Transpose.ROTATE_90)
        else:
            image = image.crop((0, 0, size, size))
        method = random.choice(("bicubic", "bilinear")) if self.method == "mixed" and self.training else self.method
        if method == "mixed":
            method = "bicubic"
        degraded, low, target = make_pair(image, method)
        return degraded, low, target

