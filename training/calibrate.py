#!/usr/bin/env python3
"""
Calibrate the trained classifier.

    python -m training.calibrate --data ./dataset --negatives ./negatives

This is the step people skip, and it is the reason so many field ID apps report
94% confidence on a wrong answer.

Two things happen here.

1. TEMPERATURE SCALING (Guo et al. 2017)
   A single scalar T is fitted on the validation split to minimise negative
   log-likelihood. It does not change which class wins, so accuracy is
   untouched, but it makes the reported probability mean something: of the
   determinations reported at 80%, roughly 80% should be right. Expected
   Calibration Error before and after is printed so you can see whether it
   worked.

2. OOD THRESHOLD
   A free-energy threshold is chosen on a set of negative images — other
   reptiles, empty habitat, blurred frames, chelonian species outside the
   training set. The threshold is set at the 95th percentile of in-distribution
   energy, which targets a 5% false-rejection rate. Retaking a photograph costs
   a minute; a confident wrong species on a seizure memo costs a case.

The negatives directory is a flat folder of images. It matters far more than
its size suggests — a model with no negative set will confidently assign a
species to a photograph of a monitor lizard.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("calibrate")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import CALIBRATION_PATH, CLASSIFIER_PATH  # noqa: E402
from core.database import SpeciesDB  # noqa: E402
from core.inference import ChelonidIdentifier, free_energy, softmax  # noqa: E402

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def collect_logits(identifier, image_paths: list[Path]) -> tuple[np.ndarray, bool]:
    """
    Score images through the exact same path inference uses.

    Returns (matrix, are_true_logits). If the logit hook fails, the caller must
    know, because the energy threshold fitted here would then be on a different
    scale from the one applied at inference time.
    """
    from PIL import Image, ImageOps

    rows, true_logits = [], True
    for i, path in enumerate(image_paths, 1):
        try:
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            scores, is_true = identifier.raw_scores(img)
            true_logits = true_logits and is_true
            rows.append(scores)
        except Exception as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
        if i % 100 == 0:
            logger.info("  %d / %d", i, len(image_paths))
    return (np.vstack(rows) if rows else np.empty((0, 0))), true_logits


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Standard ECE: mean |accuracy - confidence| weighted by bin occupancy."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Grid then golden-section search on T. Small problem; no need for autograd."""
    def nll(T: float) -> float:
        total = 0.0
        for row, label in zip(logits, labels):
            total -= np.log(max(softmax(row, T)[label], 1e-12))
        return total / len(labels)

    grid = np.concatenate([np.arange(0.5, 3.0, 0.1), np.arange(3.0, 8.1, 0.5)])
    best_T = float(min(grid, key=nll))

    lo, hi = max(0.05, best_T - 0.15), best_T + 0.15
    phi = (5 ** 0.5 - 1) / 2
    for _ in range(30):
        a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
    return round((lo + hi) / 2, 4)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Dataset root containing val/")
    p.add_argument("--negatives", default=None, help="Flat folder of non-target images")
    p.add_argument("--weights", default=str(CLASSIFIER_PATH))
    p.add_argument("--fpr", type=float, default=0.05,
                   help="Target false-rejection rate for the OOD gate")
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("pip install ultralytics")

    if not Path(args.weights).exists():
        raise SystemExit(f"No weights at {args.weights}. Train the model first.")

    identifier = ChelonidIdentifier(SpeciesDB.load(), classifier_path=Path(args.weights))
    model = identifier._ensure_classifier()
    class_names = [model.names[i] for i in sorted(model.names)]
    index = {name: i for i, name in enumerate(class_names)}

    # ---- validation set ------------------------------------------------
    val_root = Path(args.data) / "val"
    paths, labels = [], []
    for cls_dir in sorted(d for d in val_root.iterdir() if d.is_dir()):
        if cls_dir.name not in index:
            logger.warning("Validation class %s is not in the model; skipping.", cls_dir.name)
            continue
        for f in cls_dir.iterdir():
            if f.suffix.lower() in IMG_SUFFIXES:
                paths.append(f)
                labels.append(index[cls_dir.name])

    if not paths:
        raise SystemExit(f"No validation images found under {val_root}")

    logger.info("Scoring %d validation images...", len(paths))
    logits, true_logits = collect_logits(identifier, paths)
    labels = np.asarray(labels[: len(logits)])

    if not true_logits:
        logger.error(
            "Raw logits could not be captured from the model. The free-energy "
            "threshold fitted below would be on a different scale from the one "
            "used at inference, so it will NOT be written. Temperature scaling "
            "is still valid and will be saved."
        )

    before = np.vstack([softmax(r, 1.0) for r in logits])
    ece_before = expected_calibration_error(before, labels)
    accuracy = float((before.argmax(axis=1) == labels).mean())

    temperature = fit_temperature(logits, labels)
    after = np.vstack([softmax(r, temperature) for r in logits])
    ece_after = expected_calibration_error(after, labels)

    logger.info("Top-1 accuracy      : %.3f", accuracy)
    logger.info("Fitted temperature  : %.4f", temperature)
    logger.info("ECE before / after  : %.4f  ->  %.4f", ece_before, ece_after)
    if temperature > 1.5:
        logger.warning(
            "Temperature above 1.5 means the raw model was badly overconfident. "
            "Usually a sign of too few images per class or leakage between splits."
        )
    if ece_after > 0.10:
        logger.warning(
            "Calibration error is still above 0.10. Reported percentages remain "
            "unreliable; keep the abstention thresholds conservative."
        )

    # ---- OOD threshold -------------------------------------------------
    energies_in = np.array([free_energy(r, temperature) for r in logits])
    threshold = float(np.percentile(energies_in, 100 * (1 - args.fpr)))
    logger.info(
        "In-distribution energy: median %.3f, %d%%ile %.3f",
        float(np.median(energies_in)), int(100 * (1 - args.fpr)), threshold,
    )

    if args.negatives:
        neg_paths = [
            f for f in Path(args.negatives).iterdir()
            if f.suffix.lower() in IMG_SUFFIXES
        ]
        if neg_paths:
            logger.info("Scoring %d negative images...", len(neg_paths))
            neg_logits, _ = collect_logits(identifier, neg_paths)
            energies_out = np.array([free_energy(r, temperature) for r in neg_logits])
            caught = float((energies_out > threshold).mean())
            logger.info(
                "Negatives correctly rejected at this threshold: %.1f%%", 100 * caught
            )
            if caught < 0.7:
                logger.warning(
                    "Under 70%% of negatives are rejected. The model will assign "
                    "species names to non-chelonian images. Add more negatives "
                    "and consider training a detector stage."
                )
        else:
            logger.warning("Negatives folder is empty; OOD gate is untested.")
    else:
        logger.warning(
            "No negatives supplied. The OOD threshold is set from in-distribution "
            "data alone and is UNTESTED against real off-target images."
        )

    payload = {
        "temperature": temperature,
        "energy_threshold": round(threshold, 4) if true_logits else None,
        "energy_threshold_valid": true_logits,
        "validation_accuracy": round(accuracy, 4),
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "n_validation": int(len(labels)),
        "target_false_rejection_rate": args.fpr,
        "classes": class_names,
    }
    CALIBRATION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Calibration written to %s", CALIBRATION_PATH)


if __name__ == "__main__":
    main()
