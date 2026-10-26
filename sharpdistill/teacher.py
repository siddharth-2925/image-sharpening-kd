"""Official pretrained SwinIR x2 teacher for matching bicubic super-resolution."""

import importlib.util
from pathlib import Path
from urllib.request import urlretrieve

import torch


SOURCE_URL = "https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/models/network_swinir.py"
WEIGHTS_URL = ("https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
               "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth")


def prepare_teacher(root: Path) -> tuple[Path, Path]:
    folder = root / "third_party" / "swinir"
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "network_swinir.py"
    weights = folder / "swinir_lightweight_x2.pth"
    for destination, url in ((source, SOURCE_URL), (weights, WEIGHTS_URL)):
        if not destination.exists() or destination.stat().st_size == 0:
            temporary = destination.with_suffix(destination.suffix + ".part")
            try:
                urlretrieve(url, temporary)
                temporary.replace(destination)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
    return source, weights


def load_teacher(root: Path, device: str):
    source, weights = prepare_teacher(root)
    specification = importlib.util.spec_from_file_location("official_swinir", source)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load SwinIR source at {source}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    model = module.SwinIR(upscale=2, in_chans=3, img_size=64, window_size=8,
                          img_range=1.0, depths=[6] * 4, embed_dim=60,
                          num_heads=[6] * 4, mlp_ratio=2,
                          upsampler="pixelshuffledirect", resi_connection="1conv")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("params", state), strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def teacher_tiled(model, low: torch.Tensor, tile: int = 128, overlap: int = 16) -> torch.Tensor:
    """Tile with overlap; keep each tile's central region to avoid seam artifacts."""
    if low.shape[0] != 1:
        raise ValueError("teacher_tiled expects batch size 1")
    if tile % 8 or overlap % 8 or overlap >= tile // 2:
        raise ValueError("tile and overlap must be multiples of 8; overlap < tile/2")
    _, _, height, width = low.shape
    output = torch.empty((1, 3, height * 2, width * 2), device=low.device, dtype=low.dtype)
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            inner_y1, inner_y2 = y, min(y + tile, height)
            inner_x1, inner_x2 = x, min(x + tile, width)
            y1 = max(0, inner_y1 - overlap)
            x1 = max(0, inner_x1 - overlap)
            y2 = min(height, inner_y2 + overlap)
            x2 = min(width, inner_x2 + overlap)
            # SwinIR requires low-resolution dimensions divisible by 8.
            pad_h = (8 - (y2 - y1) % 8) % 8
            pad_w = (8 - (x2 - x1) % 8) % 8
            patch = torch.nn.functional.pad(low[..., y1:y2, x1:x2], (0, pad_w, 0, pad_h), mode="replicate")
            result = model(patch)
            sy1, sy2 = (inner_y1 - y1) * 2, (inner_y2 - y1) * 2
            sx1, sx2 = (inner_x1 - x1) * 2, (inner_x2 - x1) * 2
            output[..., inner_y1*2:inner_y2*2, inner_x1*2:inner_x2*2] = result[..., sy1:sy2, sx1:sx2]
    return output

