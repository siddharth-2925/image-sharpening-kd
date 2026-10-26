"""Export a trained student to fixed-size ONNX and OpenVINO models."""

import argparse
from pathlib import Path

import torch
from torch import nn

from sharpdistill.checkpoint import load_student


class VideoWrapper(nn.Module):
    """Scale model RGB output to byte-valued BGR for OpenCV deployment."""

    def __init__(self, student):
        super().__init__()
        self.student = student

    def forward(self, rgb):
        output = torch.round(self.student(rgb).clamp(0, 1) * 255.0)
        return output[:, [2, 1, 0], :, :]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/distill_best.pt"))
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    if args.height % 2 or args.width % 2:
        parser.error("height and width must be even")
    model, _ = load_student(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"student_{args.width}x{args.height}"
    onnx_path = args.output_dir / f"{stem}.onnx"
    video_onnx_path = args.output_dir / f"{stem}_video.onnx"
    xml_path = args.output_dir / f"{stem}_video.xml"
    example = torch.rand((1, 3, args.height, args.width), dtype=torch.float32)
    torch.onnx.export(model, example, onnx_path, input_names=["degraded_rgb"],
                      output_names=["sharpened_rgb"], opset_version=17,
                      do_constant_folding=True, dynamo=False)
    torch.onnx.export(VideoWrapper(model), example, video_onnx_path,
                      input_names=["degraded_rgb"], output_names=["sharpened_bgr_0_to_255"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    import openvino as ov
    from openvino.preprocess import ColorFormat, PrePostProcessor
    converted = ov.convert_model(str(video_onnx_path))
    ppp = PrePostProcessor(converted)
    ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC")).set_color_format(ColorFormat.BGR)
    ppp.input().model().set_layout(ov.Layout("NCHW"))
    ppp.input().preprocess().convert_element_type(ov.Type.f32).convert_color(ColorFormat.RGB).scale(255.0)
    ppp.output().model().set_layout(ov.Layout("NCHW"))
    ppp.output().tensor().set_layout(ov.Layout("NHWC")).set_element_type(ov.Type.u8)
    ov.save_model(ppp.build(), str(xml_path))
    print(f"Exported raw RGB ONNX {onnx_path} and OpenCV-ready OpenVINO {xml_path}")


if __name__ == "__main__":
    main()

