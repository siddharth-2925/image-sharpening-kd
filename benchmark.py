"""Time full-HD model inference separately from complete frame processing."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from sharpdistill.runtime import Runtime, frame_to_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["openvino", "torch"], default="openvino")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/distill_best.pt"))
    parser.add_argument("--model-xml", type=Path, default=Path("checkpoints/student_1920x1080_video.xml"))
    parser.add_argument("--device", default="CPU", help="OpenVINO device: CPU, GPU, or AUTO")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args()
    if min(args.width, args.height) < 32 or args.width % 2 or args.height % 2:
        parser.error("dimensions must be even and at least 32")
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    runtime = Runtime(args.backend, args.width, args.height, args.checkpoint, args.model_xml, args.device)
    frame = np.random.default_rng(2110).integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)
    input_array = frame[None] if args.backend == "openvino" else frame_to_tensor(frame, args.width, args.height)

    def inference_only():
        if args.backend == "openvino":
            runtime.model([input_array])
        else:
            with runtime.torch.inference_mode():
                runtime.model(runtime.torch.from_numpy(input_array))

    def measure(function):
        for _ in range(args.warmup):
            function()
        elapsed = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            function()
            elapsed.append((time.perf_counter() - start) * 1000)
        return {"mean_ms": float(np.mean(elapsed)), "median_ms": float(np.median(elapsed)),
                "p95_ms": float(np.percentile(elapsed, 95)),
                "fps_from_mean_latency": float(1000 / np.mean(elapsed))}

    model_timing = measure(inference_only)
    complete_timing = measure(lambda: runtime.process(frame))
    result = {"backend": args.backend, "device": args.device if args.backend == "openvino" else "CPU",
              "input_and_output": f"{args.width}x{args.height} {'BGR uint8' if args.backend == 'openvino' else 'RGB float32'}", "batch_size": 1,
              "warmup": args.warmup, "iterations": args.iterations,
              "engine_only": model_timing, "full_frame_processing": complete_timing,
              "platform": platform.platform(), "processor": platform.processor(),
              "note": "OpenVINO engine includes integrated color/layout/byte conversion. Single synchronous stream; excludes camera capture, display, and video encoding."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

