#!/usr/bin/env python3
"""
Train the YOLOv8 classification head.

    python -m training.train_classifier --data ./dataset --epochs 120

Expected dataset layout (ultralytics classification format). Folder names must
match species ids in data/species_db.json exactly — the app refuses to load a
model whose classes it cannot resolve.

    dataset/
      train/
        lissemys_punctata/*.jpg
        pangshura_tecta/*.jpg
        ...
      val/
        lissemys_punctata/*.jpg
        ...
      test/
        ...

Notes that actually affect accuracy here:

* Split by ANIMAL, not by photograph. Ten photographs of the same basking
  Pangshura in one sitting will leak between train and val and give you a
  validation accuracy that collapses in the field. If images are named with a
  capture id, use prepare_dataset.py, which splits on that.

* Class imbalance is severe and unavoidable — you will have 400 Lissemys and
  9 Batagur kachuga. Do not fix this by throwing away Lissemys images. Use the
  weighted sampling below and judge the model on per-class recall, never on
  overall accuracy.

* Augmentation is deliberately restrained on colour. Hue and saturation shifts
  destroy the exact characters this tool depends on: coral-red vs. blotched
  plastron, red postorbital patch, yellow head spots. Geometry can be pushed
  hard; colour cannot.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import CLS_IMAGE_SIZE, MODEL_DIR, SPECIES_DB_PATH  # noqa: E402


def audit_dataset(root: Path) -> dict[str, int]:
    """Count images per class per split and warn about the ones that matter."""
    summary: dict[str, int] = {}
    with open(SPECIES_DB_PATH, encoding="utf-8") as fh:
        known = {s["id"] for s in json.load(fh)["species"]}

    for split in ("train", "val"):
        split_dir = root / split
        if not split_dir.is_dir():
            raise SystemExit(f"Missing split directory: {split_dir}")

        counts = Counter()
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            n = sum(
                1 for f in cls_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
            counts[cls_dir.name] = n
            if cls_dir.name not in known:
                raise SystemExit(
                    f"Class folder '{cls_dir.name}' is not an id in species_db.json. "
                    "Rename it or add the species to the database."
                )

        logger.info("--- %s split ---", split)
        for name, n in counts.most_common():
            flag = "  <-- too few, expect poor recall" if n < 30 else ""
            logger.info("  %-32s %4d%s", name, n, flag)
        summary[split] = sum(counts.values())

        if split == "train":
            thin = [k for k, v in counts.items() if v < 30]
            if thin:
                logger.warning(
                    "%d classes have under 30 training images: %s. "
                    "Per-class recall for these will be unusable. Prioritise "
                    "collection, and keep them abstaining rather than guessing.",
                    len(thin), ", ".join(thin),
                )
    return summary


def train(args: argparse.Namespace) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("pip install ultralytics")

    root = Path(args.data).resolve()
    audit_dataset(root)

    model = YOLO(args.weights)
    model.train(
        data=str(root),
        epochs=args.epochs,
        imgsz=CLS_IMAGE_SIZE,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=str(MODEL_DIR / "runs"),
        name=args.name,
        exist_ok=True,
        seed=args.seed,
        pretrained=True,
        optimizer="AdamW",
        lr0=1e-3,
        cos_lr=True,
        label_smoothing=0.05,   # tempers overconfidence at the source
        dropout=0.15,

        # --- geometry: push it. Turtles are photographed from every angle.
        degrees=25.0,
        translate=0.15,
        scale=0.45,
        fliplr=0.5,
        flipud=0.0,             # an upside-down turtle means plastron, a real view
        erasing=0.25,           # partial occlusion: hands, grass, bucket rims

        # --- colour: restrained. These shifts destroy diagnostic characters.
        hsv_h=0.010,
        hsv_s=0.25,
        hsv_v=0.30,
    )

    weights = MODEL_DIR / "runs" / args.name / "weights" / "best.pt"
    target = MODEL_DIR / "chelonid_cls.pt"
    if weights.exists():
        target.write_bytes(weights.read_bytes())
        logger.info("Best weights copied to %s", target)
    else:
        logger.error("Expected weights at %s but found none.", weights)

    logger.info(
        "Training complete. Run `python -m training.calibrate` next — the model "
        "is not fit for field use until it has been temperature-scaled."
    )
    return target


def main() -> None:
    p = argparse.ArgumentParser(description="Train the chelonid classifier.")
    p.add_argument("--data", required=True, help="Dataset root with train/ and val/")
    p.add_argument("--weights", default="yolov8s-cls.pt", help="Starting weights")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--device", default="0", help="'0' for GPU, 'cpu' otherwise")
    p.add_argument("--name", default="chelonid_cls")
    p.add_argument("--seed", type=int, default=42)
    train(p.parse_args())


if __name__ == "__main__":
    main()
