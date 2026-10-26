"""Download the official DIV2K high-resolution train and validation images."""

import argparse
import urllib.request
import zipfile
from pathlib import Path


BASE = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
ARCHIVES = {"train": "DIV2K_train_HR.zip", "validation": "DIV2K_valid_HR.zip"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "validation", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = ARCHIVES if args.split == "both" else {args.split: ARCHIVES[args.split]}
    for split, filename in selected.items():
        folder = args.output_dir / ("DIV2K_train_HR" if split == "train" else "DIV2K_valid_HR")
        if folder.exists() and len(list(folder.glob("*.png"))) >= (800 if split == "train" else 100):
            print(f"Already extracted: {folder}")
            continue
        archive = args.output_dir / filename
        if not archive.exists():
            print(f"Downloading {BASE + filename} (several GB for training)...")
            temporary = archive.with_suffix(".zip.part")
            try:
                urllib.request.urlretrieve(BASE + filename, temporary)
                temporary.replace(archive)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        with zipfile.ZipFile(archive) as file:
            for member in file.infolist():
                destination = (args.output_dir / member.filename).resolve()
                if not destination.is_relative_to(args.output_dir.resolve()):
                    raise ValueError(f"Unsafe path in archive: {member.filename}")
            file.extractall(args.output_dir)
        count = len(list(folder.glob("*.png")))
        expected = 800 if split == "train" else 100
        if count < expected:
            raise RuntimeError(f"Expected {expected} PNGs in {folder}; found {count}")
        print(f"Ready: {folder} ({count} images)")


if __name__ == "__main__":
    main()

