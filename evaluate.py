"""Evaluate 101+ untouched images; write per-image and summary results."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sharpdistill.checkpoint import load_student
from sharpdistill.data import EXTENSIONS, make_pair
from sharpdistill.metrics import scores
from sharpdistill.teacher import load_teacher, teacher_tiled


def collect_images(div2k_dir: Path, custom_dir: Path | None):
    entries = [(path, "DIV2K") for path in sorted(div2k_dir.glob("*.png"))]
    if custom_dir is not None and custom_dir.exists():
        entries += [(path, path.parent.name) for path in sorted(custom_dir.rglob("*"))
                    if path.is_file() and path.suffix.lower() in EXTENSIONS]
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--div2k-dir", type=Path, default=Path("data/DIV2K_valid_HR"))
    parser.add_argument("--custom-dir", type=Path, default=Path("data/custom_test"))
    parser.add_argument("--student", type=Path, default=Path("checkpoints/distill_best.pt"))
    parser.add_argument("--baseline", type=Path, default=Path("checkpoints/baseline_best.pt"))
    parser.add_argument("--teacher", action="store_true", help="Also evaluate pretrained SwinIR (slow)")
    parser.add_argument("--min-images", type=int, default=125)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    entries = collect_images(args.div2k_dir, args.custom_dir)
    if len(entries) < args.min_images:
        parser.error(f"Found {len(entries)} images; add custom_test images to reach {args.min_images}")
    categories = {category for _, category in entries if category != "DIV2K"}
    expected_categories = {"people", "text", "nature", "animals", "games"}
    if not expected_categories.issubset(categories):
        parser.error(f"Missing custom test categories: {sorted(expected_categories - categories)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    student, _ = load_student(args.student, device)
    baseline, _ = load_student(args.baseline, device)
    teacher = load_teacher(Path(__file__).resolve().parent, device) if args.teacher else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for index, (path, category) in enumerate(entries, 1):
            with Image.open(path) as image:
                degraded, low, target = make_pair(image)
            reference = target
            methods = {"bicubic_input": degraded,
                       "baseline_student": baseline(degraded.unsqueeze(0).to(device))[0].cpu(),
                       "distilled_student": student(degraded.unsqueeze(0).to(device))[0].cpu()}
            if teacher is not None:
                methods["swinir_teacher"] = teacher_tiled(teacher, low.unsqueeze(0).to(device))[0].cpu()
            for name, output in methods.items():
                result = scores(output, reference)
                rows.append({"image": str(path), "category": category, "method": name, **result})
            print(f"{index}/{len(entries)} {path.name}")
    csv_path = args.output_dir / "image_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "category", "method", "psnr_rgb_db", "ssim_rgb"])
        writer.writeheader()
        writer.writerows(rows)
    by_method = defaultdict(list)
    by_category = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
        by_category[(row["category"], row["method"])].append(row)
    def summarize(group):
        return {"images": len(group), "mean_psnr_rgb_db": float(np.mean([r["psnr_rgb_db"] for r in group])),
                "mean_ssim_rgb": float(np.mean([r["ssim_rgb"] for r in group]))}
    summary = {"protocol": "RGB uint8, 2-pixel border crop, x2 bicubic downscale/upscale",
               "total_distinct_images": len(entries), "overall": {k: summarize(v) for k, v in by_method.items()},
               "by_category": {f"{category}/{method}": summarize(group)
                               for (category, method), group in by_category.items()}}
    summary_path = args.output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(f"Saved {csv_path} and {summary_path}")


if __name__ == "__main__":
    main()

