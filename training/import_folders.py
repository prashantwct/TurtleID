#!/usr/bin/env python3
"""
File photographs into the pool from a folder tree you arrange by hand.

    python -m training.import_folders --setup      # make the folders
    ...drop photographs in with a file manager...
    python -m training.import_folders --dry-run    # see what would happen
    python -m training.import_folders              # file them

This is `ingest_field_images.py` for people who would rather drag photographs
into folders than type a command per animal. It does exactly the same filing,
through the same EXIF scrubber, and produces the same `pool/` layout — the only
difference is where it learns the species and the capture from.

THE LAYOUT
----------
One folder per species. Inside it, ONE FOLDER PER ANIMAL:

    incoming/
      lissemys_punctata/
        chambal-aug-19/        <- one animal, photographed once
          IMG_0431.jpg
          IMG_0432.jpg
        chambal-aug-20/        <- a different animal
          IMG_0455.jpg
      Indian Roofed Turtle/    <- common names work too
        rescue-crate-3/
          a.jpg

Species folders may be named with the id from `data/species_db.json`, the
scientific name, or the English common name; anything else is reported and
skipped rather than guessed at. The animal folder's name becomes the capture
id, tidied into something usable as a filename.

WHY THE ANIMAL FOLDER MATTERS
-----------------------------
`prepare_dataset.py` moves whole captures between train and val, and the animal
folder is what tells it which photographs are of the same individual. Ten
photographs of one basking Pangshura are not ten observations; split apart, the
same animal lands on both sides of the split and validation accuracy stops
measuring anything you can use in the field.

Photographs dropped loose in a species folder, with no animal folder around
them, are each treated as their own animal. That is right when every loose
photograph really is a different individual, and wrong the moment two are not
— so it is counted and reported every run.

RUNNING IT TWICE IS SAFE
------------------------
A capture already present in the pool is left alone, not filed again. Add more
animals, run it again, and only the new folders are picked up. `--reimport`
replaces captures that are already there, for when you have corrected what is
inside a folder.

Nothing is moved or deleted: `incoming/` keeps your originals, and the copies
written into `pool/` are stripped of metadata on the way. Both directories are
gitignored, and the privacy CI job fails if either is ever tracked.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("import")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import ALLOWED_SUFFIXES, BASE_DIR, SPECIES_DB_PATH  # noqa: E402
from training.ingest_field_images import (  # noqa: E402
    POOL_DIR,
    FilingResult,
    capture_id_error,
    existing_frames,
    file_capture,
    pool_summary,
    report_gps,
    report_heic,
)

INCOMING_DIR = BASE_DIR / "incoming"
GUIDE_FILE = "WHICH-FOLDER.txt"


# ------------------------------------------------------------------ naming

def normalise(text: str) -> str:
    """Fold a human-typed name to something comparable: 'Indian Roofed' -> indian_roofed."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def species_aliases() -> dict[str, str]:
    """Every name a species folder may carry, mapped to its id."""
    db = json.loads(SPECIES_DB_PATH.read_text(encoding="utf-8"))
    aliases: dict[str, str] = {}
    for sp in db["species"]:
        for name in (sp["id"], sp["scientific_name"], sp.get("common_en") or ""):
            if name:
                aliases[normalise(name)] = sp["id"]
    return aliases


def capture_from_name(name: str) -> str:
    """Turn a folder or file name into a capture id `ingest_field_images` accepts."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    cleaned = re.sub(r"-{2,}", "-", cleaned)          # '--' separates capture from frame
    cleaned = re.sub(r"^[^A-Za-z0-9]+", "", cleaned)  # must open on a letter or digit
    cleaned = cleaned.rstrip("-")
    return cleaned if cleaned and capture_id_error(cleaned) is None else ""


def unique_capture(taken: set[str], candidate: str) -> str:
    """Keep two folders that tidy to the same id from colliding in the pool."""
    if candidate not in taken:
        return candidate
    for n in range(2, 1000):
        alternative = f"{candidate}-{n}"
        if alternative not in taken:
            return alternative
    raise SystemExit(f"Could not find a free capture id for {candidate!r}")


# ------------------------------------------------------------------ scanning

def images_in(root: Path, recursive: bool) -> list[Path]:
    """Usable photographs, ignoring hidden files and file-manager leftovers."""
    walk = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        p for p in walk
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in ALLOWED_SUFFIXES
    )


class Plan:
    """What one run would file, worked out before anything is written."""

    def __init__(self) -> None:
        self.captures: list[tuple[str, str, list[Path], str]] = []
        self.unknown_folders: list[Path] = []
        self.loose: list[tuple[str, str]] = []
        self.already_present: list[tuple[str, str]] = []
        self.empty_folders: list[Path] = []


def scan(incoming: Path, aliases: dict[str, str], reimport: bool) -> Plan:
    plan = Plan()
    taken: dict[str, set[str]] = defaultdict(set)

    for species_dir in sorted(p for p in incoming.iterdir() if p.is_dir()):
        if species_dir.name.startswith("."):
            continue
        species_id = aliases.get(normalise(species_dir.name))
        if species_id is None:
            plan.unknown_folders.append(species_dir)
            continue

        animal_dirs = sorted(
            p for p in species_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
        loose_files = images_in(species_dir, recursive=False)

        sources: list[tuple[str, list[Path], str]] = []
        for animal_dir in animal_dirs:
            files = images_in(animal_dir, recursive=True)
            if not files:
                plan.empty_folders.append(animal_dir)
                continue
            sources.append((animal_dir.name, files, str(animal_dir.relative_to(incoming))))
        for image in loose_files:
            sources.append((image.stem, [image], str(image.relative_to(incoming))))
            plan.loose.append((species_id, image.name))

        for raw_name, files, label in sources:
            capture = capture_from_name(raw_name)
            if not capture:
                capture = capture_from_name(f"capture-{len(taken[species_id]) + 1}")
                logger.warning(
                    "%s: %r has no usable capture id, filing it as %r",
                    species_id, raw_name, capture,
                )
            capture = unique_capture(taken[species_id], capture)
            taken[species_id].add(capture)

            if existing_frames(species_id, capture) and not reimport:
                plan.already_present.append((species_id, capture))
                continue
            plan.captures.append((species_id, capture, files, label))

    return plan


# ------------------------------------------------------------------ actions

def setup(incoming: Path) -> None:
    """Create one folder per species, plus a note saying which is which."""
    db = json.loads(SPECIES_DB_PATH.read_text(encoding="utf-8"))
    incoming.mkdir(parents=True, exist_ok=True)

    lines = [
        "Drop photographs into the folder for the species, in a sub-folder per animal:",
        "",
        "    incoming/lissemys_punctata/chambal-aug-19/IMG_0431.jpg",
        "    incoming/lissemys_punctata/chambal-aug-19/IMG_0432.jpg   <- same animal",
        "    incoming/lissemys_punctata/chambal-aug-20/IMG_0455.jpg   <- different animal",
        "",
        "One folder per ANIMAL, not per day or per trip. That folder is what keeps",
        "the same individual from landing on both sides of the validation split.",
        "",
        "Then run:  python -m training.import_folders",
        "",
        "You can rename a folder to the common name if that is easier to find;",
        "both work. Delete any folder you do not need.",
        "",
        f"{'FOLDER':<30} {'SPECIES':<30} COMMON NAME",
    ]
    for sp in db["species"]:
        (incoming / sp["id"]).mkdir(exist_ok=True)
        lines.append(f"{sp['id']:<30} {sp['scientific_name']:<30} {sp.get('common_en', '')}")

    (incoming / GUIDE_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Created %d species folders under %s", len(db["species"]), incoming)
    logger.info("Which folder is which: %s", incoming / GUIDE_FILE)
    logger.info("Drop photographs in — one folder per animal — then run "
                "`python -m training.import_folders`")


def describe(plan: Plan, incoming: Path) -> None:
    """Everything worth saying about a scan, whether or not it gets filed."""
    aliases = species_aliases() if plan.unknown_folders else {}
    for species_dir in plan.unknown_folders:
        wanted = normalise(species_dir.name)
        close = difflib.get_close_matches(wanted, aliases, n=3)
        close += [a for a in aliases if wanted and wanted in a]
        matches = sorted({aliases[c] for c in close})
        suggestion = f" Did you mean: {', '.join(matches[:3])}?" if matches else ""
        logger.error(
            "Folder %r is not a species this database knows, so nothing in it was "
            "filed.%s See %s.",
            species_dir.name, suggestion, incoming / GUIDE_FILE,
        )
    for folder in plan.empty_folders:
        logger.info("Empty (no usable images): %s", folder.relative_to(incoming))
    if plan.already_present:
        logger.info(
            "%d capture(s) already in the pool, left alone: %s. Pass --reimport to "
            "replace them.",
            len(plan.already_present),
            ", ".join(f"{s}/{c}" for s, c in plan.already_present[:5]),
        )
    if plan.loose:
        by_species: dict[str, int] = defaultdict(int)
        for species_id, _ in plan.loose:
            by_species[species_id] += 1
        logger.warning(
            "%d photograph(s) sat loose in a species folder with no animal folder "
            "around them (%s). Each is filed as a SEPARATE animal. If any two of "
            "them are the same individual, put those in one folder together and "
            "re-run with --reimport — otherwise the split will quietly overstate "
            "how good the model is.",
            len(plan.loose),
            ", ".join(f"{s}: {n}" for s, n in sorted(by_species.items())),
        )


def run(incoming: Path, dry_run: bool, reimport: bool) -> None:
    if not incoming.is_dir():
        raise SystemExit(
            f"No folder at {incoming}. Run `python -m training.import_folders --setup` "
            f"to create it, or pass --incoming with the path you are using."
        )

    plan = scan(incoming, species_aliases(), reimport)
    describe(plan, incoming)

    if not plan.captures:
        if plan.already_present:
            # A no-op re-run is the normal case once everything has been filed,
            # not a failure. Say so and leave the exit status clean.
            logger.info("Everything here is already in the pool; nothing new to file.")
            return
        raise SystemExit(
            "Nothing to file. Put photographs inside a species folder, in a "
            f"sub-folder per animal, and run this again. See {incoming / GUIDE_FILE}."
        )

    if dry_run:
        logger.info("--- dry run: nothing written ---")
        for species_id, capture, files, label in plan.captures:
            logger.info("%-28s %-24s %2d photograph(s)  <- %s",
                        species_id, capture, len(files), label)
        species = {s for s, _, _, _ in plan.captures}
        logger.info("Would file %d photographs as %d captures across %d species.",
                    sum(len(f) for _, _, f, _ in plan.captures), len(plan.captures),
                    len(species))
        return

    totals = FilingResult()
    for species_id, capture, files, label in plan.captures:
        if reimport:
            for stale in (POOL_DIR / species_id).glob(f"{capture}--*"):
                stale.unlink()
        result = file_capture(species_id, capture, files)
        logger.info("%-28s %-24s %2d filed  <- %s",
                    species_id, capture, result.written, label)
        totals.written += result.written
        totals.had_gps += result.had_gps
        totals.unreadable.extend(result.unreadable)
        totals.heic.extend(result.heic)

    logger.info("Filed %d photographs as %d captures.", totals.written, len(plan.captures))
    report_gps(totals)
    report_heic(totals.heic)
    if totals.unreadable:
        logger.warning("%d file(s) could not be read: %s",
                       len(totals.unreadable),
                       ", ".join(p.name for p in totals.unreadable[:5]))

    for species_id in sorted({s for s, _, _, _ in plan.captures}):
        pool_summary(species_id)

    logger.info("Next: python -m training.prepare_dataset --pool ./pool --out ./dataset --force")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--incoming", type=Path, default=INCOMING_DIR,
                   help=f"Folder tree to read (default {INCOMING_DIR})")
    p.add_argument("--setup", action="store_true",
                   help="Create a folder per species, then exit")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be filed without writing anything")
    p.add_argument("--reimport", action="store_true",
                   help="Replace captures already in the pool instead of skipping them")
    args = p.parse_args()

    if args.setup:
        setup(args.incoming)
        return
    run(args.incoming, args.dry_run, args.reimport)


if __name__ == "__main__":
    main()
