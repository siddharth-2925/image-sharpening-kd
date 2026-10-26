"""Live webcam or video demo, with measured on-screen throughput."""

import argparse
import time
from pathlib import Path

import cv2

from sharpdistill.runtime import Runtime


def synthetic_degradation(frame):
    height, width = frame.shape[:2]
    low = cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(low, (width, height), interpolation=cv2.INTER_CUBIC)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera number or video file path")
    parser.add_argument("--backend", choices=["openvino", "torch"], default="openvino")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/distill_best.pt"))
    parser.add_argument("--model-xml", type=Path, default=Path("checkpoints/student_1920x1080_video.xml"))
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--preview-width", type=int, default=640,
                        help="Width of each side-by-side preview panel; model still processes full resolution")
    parser.add_argument("--synthetic-degradation", action="store_true",
                        help="Controlled down/up blur for the side-by-side demonstration")
    parser.add_argument("--save", type=Path, help="Optional MP4 output")
    args = parser.parse_args()
    if args.preview_width < 160:
        parser.error("preview width must be at least 160 pixels")
    runtime = Runtime(args.backend, args.width, args.height, args.checkpoint, args.model_xml, args.device)
    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera/video source: {args.source}")
    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    writer = None
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), 20,
                                 (args.width, args.height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot write video to {args.save}")
    degrade = args.synthetic_degradation
    smoothed_fps = None
    window_name = "Real-time Image Sharpening | D: degradation | Q: quit"
    preview_width = min(args.width // 2, args.preview_width)
    preview_height = round(preview_width * args.height / args.width)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, preview_width * 2, preview_height)
    print("Keys: D toggles simulated degradation; Q or Esc quits.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            start = time.perf_counter()
            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
            input_frame = synthetic_degradation(frame) if degrade else frame
            restored = runtime.process(input_frame)
            elapsed = time.perf_counter() - start
            fps = 1 / max(elapsed, 1e-9)
            smoothed_fps = fps if smoothed_fps is None else 0.85 * smoothed_fps + 0.15 * fps
            if writer is not None:
                writer.write(restored)
            left = cv2.resize(input_frame, (preview_width, preview_height))
            right = cv2.resize(restored, (preview_width, preview_height))
            display = cv2.hconcat([left, right])
            cv2.putText(display, "Input" + (" (simulated blur)" if degrade else ""),
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, f"Output | {smoothed_fps:.1f} processing FPS",
                        (preview_width + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                degrade = not degrade
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

