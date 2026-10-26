"""Create report figures from measured evaluation and training CSV files."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("results/image_metrics.csv"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    rows = read_rows(args.metrics)
    if not rows:
        parser.error("Metrics CSV is empty; run evaluate.py first")
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(float(row["ssim_rgb"]))
    labels = list(by_method)
    means = [np.mean(by_method[name]) for name in labels]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, means, color=["#888888", "#3c6e9e", "#36a77b", "#9a67b1"][:len(labels)])
    ax.axhline(0.90, color="#c94f4f", linestyle="--", linewidth=1.2, label="0.90 target")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean RGB SSIM (higher is better)")
    ax.set_title(f"Image quality on {len(by_method[labels[0]])} test images")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.01, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(args.output_dir / "mean_ssim.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    found_history = False
    for name in ("baseline", "distill"):
        history = args.checkpoints / f"{name}_history.csv"
        if history.exists():
            values = read_rows(history)
            ax.plot([int(row["epoch"]) for row in values],
                    [float(row["val_ssim_rgb"]) for row in values], marker="o", label=name)
            found_history = True
    if found_history:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Held-out validation RGB SSIM")
        ax.set_title("Training comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / "validation_learning_curve.png", dpi=180)
    plt.close(fig)
    print(f"Saved measured figures to {args.output_dir}")


if __name__ == "__main__":
    main()

