"""Create blinded A/B image pairs and blank participant score sheets."""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from sharpdistill.checkpoint import load_student
from sharpdistill.data import EXTENSIONS, make_pair
from sharpdistill.metrics import to_rgb_uint8


def detail_box(image: Image.Image, requested_size: int) -> tuple[int, int, int, int]:
    """Choose a textured crop from the input, independently of the model output."""
    width, height = image.size
    side = min(requested_size, width, height)
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    xs = np.linspace(0, width - side, num=5, dtype=int)
    ys = np.linspace(0, height - side, num=5, dtype=int)
    best_score, best_xy = -1.0, (0, 0)
    for y in ys:
        for x in xs:
            patch = gray[y:y + side, x:x + side]
            score = float(np.abs(np.diff(patch, axis=0)).mean() +
                          np.abs(np.diff(patch, axis=1)).mean())
            if score > best_score:
                best_score, best_xy = score, (int(x), int(y))
    x, y = best_xy
    return x, y, x + side, y + side


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/custom_test"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/distill_best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/mos"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--crop-size", type=int, default=512,
                        help="Native-resolution square crop size; never resize the comparison")
    parser.add_argument("--seed", type=int, default=2110)
    args = parser.parse_args()
    if args.crop_size < 64:
        parser.error("crop size must be at least 64 pixels")
    paths = sorted(path for path in args.images.rglob("*") if path.suffix.lower() in EXTENSIONS)
    if len(paths) < args.count:
        parser.error(f"Need {args.count} distinct images; found {len(paths)}")
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    model, _ = load_student(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    answer_key = []
    with torch.inference_mode():
        for index, path in enumerate(paths[:args.count], 1):
            with Image.open(path) as image:
                degraded, _, _ = make_pair(image)
            restored = model(degraded.unsqueeze(0))[0]
            left_is_restored = bool(rng.getrandbits(1))
            a = restored if left_is_restored else degraded
            b = degraded if left_is_restored else restored
            a_image = Image.fromarray(to_rgb_uint8(a))
            b_image = Image.fromarray(to_rgb_uint8(b))
            box = detail_box(Image.fromarray(to_rgb_uint8(degraded)), args.crop_size)
            a_image = a_image.crop(box)
            b_image = b_image.crop(box)
            side = a_image.width
            page = Image.new("RGB", (side * 2, side + 40), "#202020")
            page.paste(a_image, (0, 40))
            page.paste(b_image, (side, 40))
            draw = ImageDraw.Draw(page)
            draw.text((15, 12), f"Pair {index:02d} - A", fill="white")
            draw.text((side + 15, 12), f"Pair {index:02d} - B", fill="white")
            page.save(args.output_dir / f"pair_{index:02d}.png")
            answer_key.append({"pair_id": f"pair_{index:02d}", "source": str(path),
                               "A": "student" if left_is_restored else "input",
                               "B": "input" if left_is_restored else "student"})
    with (args.output_dir / "answer_key_PRIVATE.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["pair_id", "source", "A", "B"])
        writer.writeheader()
        writer.writerows(answer_key)
    with (args.output_dir / "participant_scores_TEMPLATE.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["participant_id", "pair_id", "quality_A_1_to_5", "quality_B_1_to_5"])
        writer.writeheader()
        for row in answer_key:
            writer.writerow({"participant_id": "", "pair_id": row["pair_id"],
                             "quality_A_1_to_5": "", "quality_B_1_to_5": ""})
    print(f"Created {len(answer_key)} blinded pairs in {args.output_dir}")
    print("Keep answer_key_PRIVATE.csv away from participants.")


if __name__ == "__main__":
    main()

