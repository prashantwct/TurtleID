#!/usr/bin/env python3
"""
Build the train/val/test splits that `train_classifier.py` expects.

    python -m training.prepare_dataset --pool ./pool --out ./dataset
    python -m training.prepare_dataset --pool ./pool --out ./dataset --seed-with-reference-plates

Input is a pool of labelled images, one directory per species id:

    pool/
      lissemys_punctata/IMG_0431.jpg
      lissemys_punctata/IMG_0432.jpg
      pangshura_tecta/kanha-2024-06-11--03.jpg
      ...

Directory names must match ids in `data/species_db.json`; an unrecognised one
is refused rather than silently trained on, because the app resolves model
classes through the same ids and a typo becomes an unloadable model.

WHY THE SPLIT IS BY CAPTURE AND NOT BY FILE
-------------------------------------------
Ten photographs of the same basking *Pangshura* in one sitting are not ten
independent observations. Split them by file and the same animal lands on both
sides; validation accuracy then measures whether the model recognises that
individual, which it does, and the number it reports is a fiction that
collapses the first time the tool meets a different animal.

So files are grouped by capture id — everything before `--` in the filename —
and whole groups move together. A filename without `--` is its own capture,
which is the right default for one photograph of one animal.

    kanha-2024-06-11--03.jpg   ->  capture "kanha-2024-06-11"
    IMG_0431.jpg               ->  capture "IMG_0431"

WHAT THIS SCRIPT WILL NOT DO
----------------------------
It will not manufacture a validation split for a class that has only one
capture. A single image cannot be both trained on and held out, and duplicating
it across the split would produce a validation number that means nothing. Such
classes go to train, and are listed at the end as unmeasurable — the model can
emit them, but nothing here can tell you how often it is right.

That case is not hypothetical. Seeded from the published identification cards
alone, every class has exactly one capture, and the honest summary of the
resulting dataset is that it can train nothing worth deploying. It is a
starting point for field photographs to accumulate against, which is the order
of work the README sets out.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prepare")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import (  # noqa: E402
    ALLOWED_SUFFIXES,
    REFERENCE_IMAGE_DIR,
    REFERENCE_MANIFEST,
    SPECIES_DB_PATH,
)

CAPTURE_SEPARATOR = "--"

# Below this many training images a class is reported as thin. It matches the
# threshold train_classifier.py already warns at, so the two agree.
THIN_CLASS = 30


def known_species_ids() -> set[str]:
    db = json.loads(SPECIES_DB_PATH.read_text(encoding="utf-8"))
    return {sp["id"] for sp in db["species"]}


def capture_id(path: Path) -> str:
    """The capture a file belongs to. Everything before `--`, else the stem."""
    stem = path.stem
    return stem.split(CAPTURE_SEPARATOR)[0] if CAPTURE_SEPARATOR in stem else stem


def collect(pool: Path, extra_roots: list[Path]) -> dict[str, dict[str, list[Path]]]:
    """species id -> capture id -> files, across the pool and any extra roots."""
    by_species: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for root in [pool, *extra_roots]:
        if not root.is_dir():
            continue
        for species_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for image in sorted(species_dir.iterdir()):
                if image.suffix.lower() in ALLOWED_SUFFIXES:
                    by_species[species_dir.name][capture_id(image)].append(image)
    return by_species


def split_captures(
    captures: list[str], val_fraction: float, test_fraction: float, rng: random.Random
) -> tuple[list[str], list[str], list[str]]:
    """Assign whole captures to train/val/test.

    Val is filled before test, so a class with two captures gets a validation
    image rather than a test one — being able to measure the model matters more
    than holding a final set back at this size.
    """
    shuffled = captures[:]
    rng.shuffle(shuffled)
    if len(shuffled) < 2:
        return shuffled, [], []

    n_val = max(1, round(len(shuffled) * val_fraction))
    n_test = round(len(shuffled) * test_fraction) if len(shuffled) >= 3 else 0
    n_val = min(n_val, len(shuffled) - 1)
    n_test = min(n_test, len(shuffled) - n_val - 1)
    return shuffled[n_val + n_test:], shuffled[:n_val], shuffled[n_val:n_val + n_test]


def build(args: argparse.Namespace) -> None:
    known = known_species_ids()
    extra_roots = []
    if args.seed_with_reference_plates:
        if not REFERENCE_IMAGE_DIR.is_dir():
            raise SystemExit(
                f"No reference images at {REFERENCE_IMAGE_DIR}. Run "
                f"`python -m training.extract_id_cards --pdf <source>` first, or drop "
                f"the flag. (The manifest at {REFERENCE_MANIFEST} records what should "
                f"be there.)"
            )
        extra_roots.append(REFERENCE_IMAGE_DIR)

    by_species = collect(args.pool, extra_roots)
    if not by_species:
        raise SystemExit(f"No images found under {args.pool}")

    unknown = sorted(set(by_species) - known)
    if unknown:
        raise SystemExit(
            "Directory names that are not species ids in data/species_db.json: "
            + ", ".join(unknown)
            + ". Rename them, or add the taxon to the database first."
        )

    rng = random.Random(args.seed)
    if args.out.exists():
        if not args.force:
            raise SystemExit(f"{args.out} exists. Pass --force to replace it.")
        shutil.rmtree(args.out)

    totals = {"train": 0, "val": 0, "test": 0}
    unmeasurable: list[str] = []
    thin: list[str] = []

    for species_id in sorted(by_species):
        captures = by_species[species_id]
        train, val, test = split_captures(
            sorted(captures), args.val_fraction, args.test_fraction, rng
        )
        if not val:
            unmeasurable.append(species_id)

        counts = {}
        for split, chosen in (("train", train), ("val", val), ("test", test)):
            files = [f for capture in chosen for f in captures[capture]]
            counts[split] = len(files)
            totals[split] += len(files)
            for source in files:
                target = args.out / split / species_id / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

        if counts["train"] < THIN_CLASS:
            thin.append(f"{species_id} ({counts['train']})")
        logger.info(
            "%-28s %2d captures -> train %d / val %d / test %d",
            species_id, len(captures), counts["train"], counts["val"], counts["test"],
        )

    logger.info("Wrote %s — train %d, val %d, test %d images across %d classes",
                args.out, totals["train"], totals["val"], totals["test"], len(by_species))

    if thin:
        logger.warning("%d of %d classes have under %d training images: %s",
                       len(thin), len(by_species), THIN_CLASS, ", ".join(thin))
    if unmeasurable:
        logger.warning(
            "%d of %d classes have a single capture and therefore no validation "
            "image: %s. The model can emit these classes; nothing in this dataset "
            "can tell you how often it is right about them.",
            len(unmeasurable), len(by_species), ", ".join(unmeasurable),
        )
    if len(unmeasurable) == len(by_species):
        logger.warning(
            "EVERY class is unmeasurable. This dataset cannot be validated, so "
            "training/calibrate.py cannot fit a temperature and the abstention "
            "machinery the app depends on will not function. Do not enable the "
            "photograph tab from a model trained on this — collect field "
            "photographs first."
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--pool", required=True, type=Path,
                   help="Labelled image pool, one directory per species id")
    p.add_argument("--out", required=True, type=Path, help="Dataset root to write")
    p.add_argument("--seed-with-reference-plates", action="store_true",
                   help=f"Also draw from {REFERENCE_IMAGE_DIR}")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=17, help="Split seed, for reproducibility")
    p.add_argument("--force", action="store_true", help="Replace an existing --out")
    build(p.parse_args())


if __name__ == "__main__":
    main()
