"""Frame conversion and PyTorch/OpenVINO inference backends."""

from pathlib import Path

import cv2
import numpy as np


def frame_to_tensor(frame_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
        frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def tensor_to_frame(array: np.ndarray) -> np.ndarray:
    rgb = np.clip(array[0].transpose(1, 2, 0), 0, 1)
    return cv2.cvtColor((rgb * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


class Runtime:
    def __init__(self, backend: str, width: int, height: int, checkpoint: Path, model_xml: Path,
                 device: str = "CPU"):
        self.backend = backend
        self.width, self.height = width, height
        if backend == "openvino":
            if not model_xml.exists():
                raise FileNotFoundError(f"Missing {model_xml}. Run export_openvino.py first.")
            import openvino as ov
            core = ov.Core()
            self.model = core.compile_model(str(model_xml), device)
            self.input_port = self.model.input(0)
            shape = tuple(self.input_port.shape)
            if shape != (1, height, width, 3):
                raise ValueError(f"OpenVINO video model expects {shape}, requested {(1, height, width, 3)}")
            self.output_port = self.model.output(0)
        elif backend == "torch":
            import torch
            from sharpdistill.checkpoint import load_student
            self.torch = torch
            self.model, _ = load_student(checkpoint, "cpu")
        else:
            raise ValueError("backend must be openvino or torch")

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.backend == "openvino":
            if frame_bgr.shape[:2] != (self.height, self.width):
                frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
            result = self.model([frame_bgr[None]])[self.output_port]
            return result[0].copy()
        else:
            array = frame_to_tensor(frame_bgr, self.width, self.height)
            with self.torch.inference_mode():
                result = self.model(self.torch.from_numpy(array)).numpy()
            return tensor_to_frame(result)

