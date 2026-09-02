#!/usr/bin/env python3
"""
Build the matching gallery. This is the no-training path.

    python -m training.build_gallery --pool ./pool --seed-with-reference-plates
    python -m training.build_gallery --pool ./pool --negatives ./negatives

There is no model to train here and no train/validation split to make. Every
photograph is embedded once and stored; identifying a new photograph means
finding its nearest neighbours among them. Adding photographs later means
running this again — a few minutes, on a laptop, with no GPU.

WHAT REPLACES THE VALIDATION SPLIT
----------------------------------
Leave-one-capture-out. Each photograph is scored against the gallery with every
photograph of its own animal removed, which answers the question that matters:
would this have been identified correctly from OTHER animals of the species?
Splitting by photograph instead would let one individual vouch for itself and
report a number that collapses in the field.

Two things are fitted from those held-out scores:

* the temperature that turns similarity gaps into probabilities, so that of the
  determinations reported at 80%, roughly 80% are right;
* the similarity floor below which a photograph is rejected as resembling
  nothing in the gallery — the gate that stops a monitor lizard being handed a
  species name. `--negatives` tests it against real off-target images and
  reports what fraction it catches.

A species with only one capture cannot take part: remove its only animal and
nothing of it remains to match against. Those species are listed as
unmeasurable. They stay in the gallery and can still be matched — a single
reference plate is genuinely useful — but nothing here can say how often they
are right, and the summary says so rather than implying otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gallery")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import (  # noqa: E402
    ALLOWED_SUFFIXES,
    GALLERY_NEIGHBOURS,
    GALLERY_PATH,
    PUBLISHED_GALLERY_PATH,
    REFERENCE_IMAGE_DIR,
    REFERENCE_MANIFEST,
    SPECIES_DB_PATH,
)
from core.detect import crop_to_animal, load_detector  # noqa: E402
from core.matcher import (  # noqa: E402
    DEFAULT_BACKBONE,
    Gallery,
    embed_images,
    fit_temperature,
    load_backbone,
)
from training.prepare_dataset import collect  # noqa: E402


def known_species_ids() -> set[str]:
    db = json.loads(SPECIES_DB_PATH.read_text(encoding="utf-8"))
    return {sp["id"] for sp in db["species"]}


def open_images(paths: list[Path], detector=None, counter: dict | None = None):
    """Yield each readable photograph, cropped to the animal where one is found.

    The crop is the same call `core/inference.py` makes on a query, so a
    gallery entry and the photograph searched against it get identical
    treatment. Frames where nothing was detected are embedded whole and
    counted: a detector that finds nothing in half the gallery is a fact the
    build has to report, not absorb.
    """
    from PIL import Image, ImageOps

    for path in paths:
        try:
            with Image.open(path) as handle:
                image = ImageOps.exif_transpose(handle).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop a build
            logger.warning("Skipping %s: %s", path.name, exc)
            continue

        if detector is not None:
            image, confidence = crop_to_animal(image, detector)
            if counter is not None:
                counter["cropped" if confidence is not None else "whole"] += 1
        yield image


def gather(pool: Path, seed_plates: bool) -> tuple[list[Path], list[str], list[str]]:
    """Flatten the pool into parallel lists of path, species and capture."""
    extra_roots = []
    if seed_plates:
        if not REFERENCE_IMAGE_DIR.is_dir():
            raise SystemExit(
                f"No reference images at {REFERENCE_IMAGE_DIR}. Drop the flag, or "
                f"run `python -m training.extract_id_cards --pdf <source>` first. "
                f"({REFERENCE_MANIFEST} records what should be there.)"
            )
        extra_roots.append(REFERENCE_IMAGE_DIR)

    by_species = collect(pool, extra_roots)
    if not by_species:
        raise SystemExit(
            f"No images found under {pool}. File some first — "
            f"`python -m training.import_folders` is the easiest way."
        )

    unknown = sorted(set(by_species) - known_species_ids())
    if unknown:
        raise SystemExit(
            "Directory names that are not species ids in data/species_db.json: "
            + ", ".join(unknown)
            + ". Rename them, or add the taxon to the database first."
        )

    paths, species, captures = [], [], []
    for species_id in sorted(by_species):
        for capture_id in sorted(by_species[species_id]):
            for path in sorted(by_species[species_id][capture_id]):
                if path.suffix.lower() in ALLOWED_SUFFIXES:
                    paths.append(path)
                    species.append(species_id)
                    captures.append(capture_id)
    return paths, species, captures


def evaluate(gallery: Gallery, fpr: float) -> dict:
    """Leave-one-capture-out over every species that has more than one animal."""
    capture_count = Counter()
    for species_id, _ in {(str(s), str(c)) for s, c in zip(gallery.species, gallery.captures)}:
        capture_count[species_id] += 1

    measurable = {s for s, n in capture_count.items() if n >= 2}
    unmeasurable = sorted(set(gallery.classes) - measurable)

    rows = [i for i, s in enumerate(gallery.species) if str(s) in measurable]
    if not rows:
        return {
            "measurable": False,
            "unmeasurable_classes": unmeasurable,
            "n_evaluated": 0,
        }

    index = {name: i for i, name in enumerate(gallery.classes)}
    query = gallery.vectors[rows]
    keys = gallery.keys[rows]
    labels = np.array([index[str(gallery.species[i])] for i in rows])

    scores, best_similarity = gallery.score(query, exclude_keys=keys)
    predicted = scores.argmax(axis=1)
    accuracy = float((predicted == labels).mean())

    per_class: dict[str, dict] = {}
    for species_id in sorted(measurable):
        mask = labels == index[species_id]
        per_class[species_id] = {
            "n": int(mask.sum()),
            "recall": round(float((predicted[mask] == labels[mask]).mean()), 4),
        }

    temperature = fit_temperature(scores, labels)
    floor = float(np.percentile(best_similarity, 100 * fpr))

    # The bar an identifier has to clear is not "calibrated", it is "better
    # than guessing". A gallery can be beautifully calibrated about knowing
    # nothing: fit a temperature to scores that carry no signal and it reports
    # 1/n for everything, with an excellent calibration error.
    chance = 1.0 / len(gallery.classes)

    from core.inference import softmax
    from training.calibrate import expected_calibration_error

    before = np.vstack([softmax(r, 1.0) for r in scores])
    after = np.vstack([softmax(r, temperature) for r in scores])

    return {
        "measurable": True,
        "reliable": accuracy > chance,
        "chance": round(chance, 4),
        "unmeasurable_classes": unmeasurable,
        "n_evaluated": len(rows),
        "accuracy": round(accuracy, 4),
        "per_class_recall": per_class,
        "temperature": temperature,
        "similarity_floor": round(floor, 4),
        "median_similarity": round(float(np.median(best_similarity)), 4),
        "ece_before": round(expected_calibration_error(before, labels), 4),
        "ece_after": round(expected_calibration_error(after, labels), 4),
        "target_false_rejection_rate": fpr,
    }


def check_negatives(gallery: Gallery, negatives: Path, model, transform) -> float | None:
    paths = [p for p in sorted(negatives.iterdir()) if p.suffix.lower() in ALLOWED_SUFFIXES]
    if not paths:
        logger.warning("Negatives folder %s is empty; the rejection gate is UNTESTED.", negatives)
        return None

    logger.info("Embedding %d negative images...", len(paths))
    vectors = embed_images(open_images(paths), model, transform)
    if vectors.size == 0:
        return None
    _, best = gallery.score(vectors)
    caught = float((best < gallery.similarity_floor).mean())
    logger.info("Negatives correctly rejected: %.1f%% of %d", 100 * caught, len(best))
    if caught < 0.7:
        logger.warning(
            "Under 70%% of negatives are rejected. The matcher will assign species "
            "names to photographs with no chelonian in them. Add more negatives, "
            "and keep the photograph tab treated as advisory."
        )
    return round(caught, 4)


def build(args: argparse.Namespace) -> None:
    paths, species, captures = gather(args.pool, args.seed_with_reference_plates)
    logger.info("Embedding %d photographs with %s (first run downloads weights)...",
                len(paths), args.backbone)

    detector = load_detector(args.detector) if args.detector else None
    if args.detector and detector is None:
        raise SystemExit(
            f"No detector could be loaded from {args.detector}. Check the path, "
            f"or drop --detector to embed whole frames."
        )
    if detector is not None:
        logger.info("Cropping to the animal with %s before embedding.", args.detector)

    tally = {"cropped": 0, "whole": 0}
    model, transform = load_backbone(args.backbone)
    vectors = embed_images(open_images(paths, detector, tally), model, transform,
                           batch_size=args.batch)
    if vectors.shape[0] != len(paths):
        # open_images drops unreadable files; keep the labels aligned with what
        # was actually embedded rather than writing a gallery that is off by one.
        raise SystemExit(
            f"Embedded {vectors.shape[0]} of {len(paths)} photographs. Some files "
            f"could not be read — the warnings above name them. HEIC needs "
            f"converting to JPEG first; anything else is likely a truncated file. "
            f"Remove or convert them and run again."
        )

    if detector is not None:
        logger.info("Detector found an animal in %d of %d photographs.",
                    tally["cropped"], tally["cropped"] + tally["whole"])
        if tally["whole"]:
            logger.warning(
                "%d photograph(s) had no detection and were embedded whole. They "
                "sit in a different space from the crops around them, which is "
                "the very mismatch cropping is meant to remove — check them, or "
                "lower DET_CONF_THRESHOLD in config.py.", tally["whole"],
            )

    gallery = Gallery(
        vectors=vectors,
        species=np.array(species),
        captures=np.array(captures),
        classes=sorted(set(species)),
        backbone=args.backbone,
        neighbours=args.neighbours,
        cropped=detector is not None,
    )

    logger.info("Leave-one-capture-out over %d species...", len(gallery.classes))
    metrics = evaluate(gallery, args.fpr)

    if metrics["measurable"]:
        gallery.temperature = metrics["temperature"]
        gallery.similarity_floor = metrics["similarity_floor"]
        gallery.calibrated = True
        gallery.reliable = metrics["reliable"]

        logger.info("Held-out top-1 accuracy : %.3f  (%d photographs)",
                    metrics["accuracy"], metrics["n_evaluated"])
        logger.info("Fitted temperature      : %.5f", metrics["temperature"])
        logger.info("ECE before / after      : %.4f  ->  %.4f",
                    metrics["ece_before"], metrics["ece_after"])
        logger.info("Similarity floor        : %.4f (median match %.4f)",
                    metrics["similarity_floor"], metrics["median_similarity"])
        logger.info("--- per-class recall (held out) ---")
        for species_id, row in sorted(metrics["per_class_recall"].items(),
                                      key=lambda kv: kv[1]["recall"]):
            flag = "  <-- unreliable" if row["recall"] < 0.5 else ""
            logger.info("  %-32s %4.2f  (n=%d)%s", species_id, row["recall"], row["n"], flag)
        if not metrics["reliable"]:
            logger.error(
                "THIS GALLERY DOES NOT IDENTIFY ANYTHING. Held-out accuracy is "
                "%.3f against %.3f for picking a species at random. Every "
                "photograph it was asked about, with its own animal held out, "
                "it got wrong or no better than a coin. The photograph tab will "
                "refuse to use it, which is the correct outcome: a calibrated "
                "gallery that knows nothing still reports a species, and that "
                "species goes on a form.",
                metrics["accuracy"], metrics["chance"],
            )
            logger.error(
                "The usual cause is that the gallery's photographs and the "
                "photographs people submit are not alike — scanned reference "
                "plates against phone photographs taken in the field. Generic "
                "image features separate those two far more strongly than they "
                "separate one species from another, so a field photograph "
                "matches other field photographs whatever animal is in them. "
                "The fix is field photographs of the species themselves, from "
                "more than one animal each."
            )
        if metrics["temperature"] <= 0.011 or metrics["temperature"] >= 1.0:
            logger.warning(
                "The fitted temperature landed at the edge of the search range. "
                "That means the similarity gaps between species carry almost no "
                "information — every photograph looks about equally close to "
                "everything. Expect the matcher to abstain constantly. More "
                "photographs, and more different animals, is the only fix."
            )
        if metrics["similarity_floor"] > 0.95:
            logger.warning(
                "The rejection floor fitted at %.4f, which is very close to a "
                "perfect match. Real photographs that are not already in the "
                "gallery will fall below it and be rejected. Treat the gate as "
                "untrustworthy until the gallery holds more animals.",
                metrics["similarity_floor"],
            )
        if metrics["ece_after"] > 0.10:
            logger.warning(
                "Calibration error is still above 0.10. Reported percentages "
                "remain unreliable; keep every determination advisory."
            )
    else:
        logger.warning(
            "No species has photographs of two different animals, so nothing "
            "could be held out and NOTHING here is measured. The gallery is "
            "written and will match, but its percentages are guesses and it "
            "reports itself as uncalibrated. Photograph a second individual of "
            "any species and run this again — that single addition is what turns "
            "the number into a measurement."
        )

    if metrics["unmeasurable_classes"]:
        logger.warning(
            "%d of %d species have only one animal in the gallery, so their "
            "accuracy is unmeasured: %s",
            len(metrics["unmeasurable_classes"]), len(gallery.classes),
            ", ".join(metrics["unmeasurable_classes"]),
        )

    if args.negatives:
        metrics["negatives_rejected"] = check_negatives(
            gallery, Path(args.negatives), model, transform
        )
    else:
        logger.warning(
            "No --negatives supplied. The rejection gate is fitted from turtle "
            "photographs alone and is UNTESTED against monitor lizards, empty "
            "habitat and unusable frames."
        )

    per_species = Counter(species)
    gallery.metrics = {
        **metrics,
        "n_photographs": int(vectors.shape[0]),
        "n_captures": len({f"{s}/{c}" for s, c in zip(species, captures)}),
        "per_species_photographs": dict(sorted(per_species.items())),
    }
    gallery.save(args.out)

    logger.info(
        "Wrote %s — %d photographs, %d captures, %d species.",
        args.out, vectors.shape[0], gallery.metrics["n_captures"], len(gallery.classes),
    )
    logger.info("The photograph tab will use this. Run `streamlit run app.py`.")

    if args.publish:
        gallery.published().save(args.publish_to)
        logger.info(
            "Wrote %s for committing — same vectors, capture ids removed. "
            "A hosted deployment can only see what is in the repository, so "
            "commit this one and push:",
            args.publish_to,
        )
        logger.info("    git add %s && git commit -m 'Update the published gallery'",
                    args.publish_to)
        logger.warning(
            "Committing it publishes which species you hold photographs of and "
            "how many. That is in the repository already by way of the "
            "contribution counts; the capture ids, which name places, are not "
            "and do not go in."
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--pool", type=Path, default=REPO_ROOT / "pool",
                   help="Labelled image pool, one directory per species id")
    p.add_argument("--out", type=Path, default=GALLERY_PATH)
    p.add_argument("--seed-with-reference-plates", action="store_true",
                   help=f"Also draw from {REFERENCE_IMAGE_DIR}")
    p.add_argument("--negatives", default=None,
                   help="Flat folder of non-chelonian images, to test the rejection gate")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--neighbours", type=int, default=GALLERY_NEIGHBOURS,
                   help="Per species, how many best matches are averaged")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--fpr", type=float, default=0.05,
                   help="Target false-rejection rate for the similarity floor")
    p.add_argument("--detector", default=None,
                   help="Detector weights (e.g. models/chelonid_det.pt). Crops "
                        "each photograph to the animal before embedding, the "
                        "same way the app crops a query. See core/detect.py.")
    p.add_argument("--publish", action="store_true",
                   help="Also write a committable copy with capture ids stripped")
    p.add_argument("--publish-to", type=Path, default=PUBLISHED_GALLERY_PATH)
    build(p.parse_args())


if __name__ == "__main__":
    main()
