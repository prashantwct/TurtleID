#!/usr/bin/env python3
"""
Add field photographs to the labelled pool.

    python -m training.ingest_field_images --species lissemys_punctata \\
        --capture chambal-2026-08-19 ~/Downloads/rescue/*.jpg

Run this once per animal, not once per batch. Everything in a single
invocation is recorded as one capture, and `prepare_dataset.py` moves a whole
capture between splits — so if two different animals go in under one capture
id, their photographs can never be separated again and the validation split
quietly stops meaning anything.

    pool/lissemys_punctata/chambal-2026-08-19--01.jpg
    pool/lissemys_punctata/chambal-2026-08-19--02.jpg

If naming a capture per animal on the command line is more bookkeeping than you
want, `training/import_folders.py` does the same filing from a folder tree you
arrange in a file manager — one folder per animal — and calls straight into
this module to do it.

`pool/` is gitignored and must stay that way. These are photographs of
threatened species with locality in the EXIF and, often, in the frame. Every
file is written through `core.contributions.strip_exif`, the same scrubber the
Contribute tab uses, which re-encodes the pixels and carries no metadata
across; files that arrived with GPS are counted at the end so you know what was
in the batch you were sent.

Species ids are checked against `data/species_db.json`. A photograph filed
under a name the database does not know is worse than one not filed at all: it
becomes a class the model can emit and the app cannot resolve.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import ALLOWED_SUFFIXES, BASE_DIR, SPECIES_DB_PATH  # noqa: E402

POOL_DIR = BASE_DIR / "pool"

# Capture ids become filenames and are split on "--", so neither is allowed
# inside one.
CAPTURE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._]*(-[A-Za-z0-9._]+)*$")

# Pillow decodes these without help. .heic is in ALLOWED_SUFFIXES because the
# app accepts what a phone produces, but decoding it needs a plugin that is not
# in requirements.txt, so filing one fails here and is reported as its own case
# rather than as a generic unreadable file.
HEIC_SUFFIXES = {".heic", ".heif"}


@dataclass
class FilingResult:
    """What one capture's worth of filing actually did."""

    written: int = 0
    had_gps: int = 0
    unreadable: list[Path] = field(default_factory=list)
    heic: list[Path] = field(default_factory=list)


def known_species_ids() -> set[str]:
    db = json.loads(SPECIES_DB_PATH.read_text(encoding="utf-8"))
    return {sp["id"] for sp in db["species"]}


def capture_id_error(capture: str) -> str | None:
    """Why this capture id is unusable, or None if it is fine."""
    if not CAPTURE_RE.match(capture) or "--" in capture:
        return (
            f"Capture id {capture!r} must be letters, digits, dots, single "
            f"hyphens and underscores — no '--', which separates the capture "
            f"from the frame number."
        )
    return None


def existing_frames(species_id: str, capture: str) -> int:
    """How many frames this capture already has in the pool."""
    target_dir = POOL_DIR / species_id
    if not target_dir.is_dir():
        return 0
    return len(list(target_dir.glob(f"{capture}--*")))


def file_capture(species_id: str, capture: str, sources: list[Path]) -> FilingResult:
    """Scrub and copy `sources` into the pool as one capture.

    Frames are numbered on from whatever the capture already holds, so calling
    this twice for the same capture appends rather than overwrites. Callers
    decide whether appending is what the user meant; this function does not.
    """
    from core.contributions import strip_exif

    target_dir = POOL_DIR / species_id
    target_dir.mkdir(parents=True, exist_ok=True)
    start = existing_frames(species_id, capture)

    result = FilingResult()
    for offset, source in enumerate(sorted(sources), start=start + 1):
        try:
            clean, gps = strip_exif(source.read_bytes())
        except Exception as exc:  # unreadable or not an image after all
            if source.suffix.lower() in HEIC_SUFFIXES:
                result.heic.append(source)
                logger.warning("skipped %s: HEIC could not be decoded", source.name)
            else:
                result.unreadable.append(source)
                logger.warning("skipped %s: %s", source.name, exc)
            continue
        target = target_dir / f"{capture}--{offset:02d}.jpg"
        target.write_bytes(clean)
        result.had_gps += gps
        result.written += 1
        logger.info("%s -> %s%s", source.name, target.relative_to(BASE_DIR),
                    "  (GPS removed)" if gps else "")
    return result


def report_gps(result: FilingResult) -> None:
    """The warning that matters most in a batch someone else sent you."""
    if result.had_gps:
        logger.warning(
            "%d of %d carried GPS coordinates, now removed. Worth telling "
            "whoever sent them — their camera embeds locality on every "
            "photograph, which matters well beyond this pool.",
            result.had_gps, result.written,
        )


def report_heic(paths: list[Path]) -> None:
    """HEIC needs converting, and saying so beats an unexplained skip."""
    if not paths:
        return
    logger.warning(
        "%d HEIC file(s) could not be read and were NOT filed: %s. Convert them "
        "to JPEG first — `sips -s format jpeg *.heic --out jpgs/` on macOS, or "
        "`magick mogrify -format jpg *.heic` — then run this again.",
        len(paths), ", ".join(p.name for p in paths[:5]),
    )


def pool_summary(species_id: str) -> None:
    target_dir = POOL_DIR / species_id
    files = [p for p in target_dir.iterdir() if p.is_file()]
    captures = len({p.stem.split("--")[0] for p in files})
    logger.info("%s now holds %d photographs across %d captures",
                species_id, len(files), captures)


def ingest(args: argparse.Namespace) -> None:
    if args.species not in known_species_ids():
        raise SystemExit(
            f"{args.species!r} is not an id in data/species_db.json. "
            f"Check the spelling, or add the taxon to the database first."
        )
    problem = capture_id_error(args.capture)
    if problem:
        raise SystemExit(problem)

    sources = [p for p in args.images if p.suffix.lower() in ALLOWED_SUFFIXES]
    skipped = [p for p in args.images if p not in sources]
    if not sources:
        raise SystemExit(
            f"No usable images. Accepted suffixes: {', '.join(sorted(ALLOWED_SUFFIXES))}"
        )

    existing = existing_frames(args.species, args.capture)
    if existing and not args.add_to_capture:
        raise SystemExit(
            f"Capture {args.capture!r} already has {existing} frames for "
            f"{args.species}. If these are more photographs of the SAME animal, "
            f"pass --add-to-capture. If it is a different animal, give it its "
            f"own capture id — that distinction is what keeps the validation "
            f"split honest."
        )

    result = file_capture(args.species, args.capture, sources)

    logger.info("Filed %d photographs as capture %r for %s",
                result.written, args.capture, args.species)
    report_gps(result)
    report_heic(result.heic)
    if skipped:
        logger.info("Ignored %d file(s) with an unsupported suffix: %s",
                    len(skipped), ", ".join(p.name for p in skipped[:5]))
    pool_summary(args.species)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--species", required=True, help="Species id from data/species_db.json")
    p.add_argument("--capture", required=True,
                   help="Capture id — one animal, one sitting, e.g. chambal-2026-08-19")
    p.add_argument("--add-to-capture", action="store_true",
                   help="Append to an existing capture; only for the same animal")
    p.add_argument("images", nargs="+", type=Path, help="Photograph files")
    ingest(p.parse_args())


if __name__ == "__main__":
    main()
