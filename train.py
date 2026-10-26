"""Train the plain student or the student guided by pretrained SwinIR.

Examples:
  python train.py --mode baseline --data-dir data/DIV2K_train_HR
  python train.py --mode distill --data-dir data/DIV2K_train_HR
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sharpdistill.checkpoint import save_student
from sharpdistill.data import PatchDataset, image_paths
from sharpdistill.metrics import scores
from sharpdistill.student import StudentNet, count_parameters
from sharpdistill.teacher import load_teacher


def gradients(image: torch.Tensor) -> torch.Tensor:
    horizontal = image[..., :, 1:] - image[..., :, :-1]
    vertical = image[..., 1:, :] - image[..., :-1, :]
    return torch.cat((horizontal[..., :-1, :], vertical[..., :, :-1]), dim=1)


def validate(model, loader, device) -> tuple[float, float]:
    model.eval()
    ssims, psnrs = [], []
    with torch.inference_mode():
        for degraded, _, target in loader:
            output = model(degraded.to(device)).clamp(0, 1)
            result = scores(output[0], target[0])
            ssims.append(result["ssim_rgb"])
            psnrs.append(result["psnr_rgb_db"])
    return float(np.mean(ssims)), float(np.mean(psnrs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "distill"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/DIV2K_train_HR"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--degradation", choices=["bicubic", "bilinear", "mixed"], default="bicubic")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--seed", type=int, default=2110)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    if args.mode == "distill" and args.patch_size % 16:
        parser.error("distillation patch size must be a multiple of 16 for SwinIR windows")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = image_paths(args.data_dir)
    if len(paths) < 60:
        parser.error("At least 60 training images are needed for a 50-image validation split")
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    train_paths, validation_paths = paths[:-50], paths[-50:]
    train_set = PatchDataset(train_paths, args.patch_size, True, args.degradation, args.repeats)
    validation_set = PatchDataset(validation_paths, args.patch_size, False, "bicubic")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device == "cuda")
    validation_loader = DataLoader(validation_set, batch_size=1, num_workers=0)
    student = StudentNet(args.channels, args.blocks).to(device)
    teacher = load_teacher(Path(__file__).resolve().parent, device) if args.mode == "distill" else None
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{args.mode}_history.csv"
    print(f"Device: {device}; student parameters: {count_parameters(student):,}")
    print(f"Training images: {len(train_paths)}; validation images: {len(validation_paths)}")
    best_ssim = -1.0
    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=["epoch", "train_loss", "val_ssim_rgb", "val_psnr_rgb_db"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            student.train()
            running = 0.0
            progress = tqdm(train_loader, desc=f"{args.mode} epoch {epoch}/{args.epochs}")
            for degraded, low, target in progress:
                degraded, low, target = degraded.to(device), low.to(device), target.to(device)
                prediction = student(degraded)
                reconstruction = F.l1_loss(prediction, target)
                edge = F.l1_loss(gradients(prediction), gradients(target))
                loss = reconstruction + 0.05 * edge
                if teacher is not None:
                    with torch.no_grad():
                        teacher_output = teacher(low).clamp(0, 1)
                    loss = loss + 0.2 * F.l1_loss(prediction, teacher_output)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                running += loss.item()
                progress.set_postfix(loss=f"{loss.item():.4f}")
            scheduler.step()
            val_ssim, val_psnr = validate(student, validation_loader, device)
            row = {"epoch": epoch, "train_loss": running / len(train_loader),
                   "val_ssim_rgb": val_ssim, "val_psnr_rgb_db": val_psnr}
            writer.writerow(row)
            log_file.flush()
            print(row)
            metadata = {"mode": args.mode, "epoch": epoch, "seed": args.seed,
                        "degradation": args.degradation, "patch_size": args.patch_size,
                        "train_images": len(train_paths), "validation_images": len(validation_paths),
                        "validation_ssim_rgb": val_ssim, "validation_psnr_rgb_db": val_psnr}
            save_student(output_dir / f"{args.mode}_latest.pt", student, metadata)
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                save_student(output_dir / f"{args.mode}_best.pt", student, metadata)
    print(f"Best {args.mode} validation SSIM: {best_ssim:.4f}; checkpoint: {output_dir / (args.mode + '_best.pt')}")


if __name__ == "__main__":
    main()

