"""Checkpoint loading shared by training, evaluation, and deployment."""

from pathlib import Path

import torch

from .student import StudentNet


def save_student(path: Path, model: StudentNet, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "channels": model.channels,
                "blocks": model.blocks_count, "metadata": metadata}, path)


def load_student(path: Path, device: str = "cpu") -> tuple[StudentNet, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Student checkpoint missing: {path}. Train it first.")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = StudentNet(channels=int(saved["channels"]), blocks=int(saved["blocks"]))
    model.load_state_dict(saved["model"], strict=True)
    model.to(device).eval()
    return model, saved.get("metadata", {})

