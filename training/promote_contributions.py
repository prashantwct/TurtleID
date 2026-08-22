#!/usr/bin/env python3
"""
Move reviewed contributed photographs into the training pool.

    python -m training.promote_contributions --list
    python -m training.promote_contributions --accept 4f2a9c1e --accept 8b0d33a1
    python -m training.promote_contributions --accept-all-confident

Contributions from the Contribute tab land in `contributions/`, which is where
they stop. Nothing has ever carried them into `pool/`, so an image can be
submitted, acknowledged, counted in the coverage map, and still never reach a
model. This is that missing step.

REVIEW IS NOT OPTIONAL
----------------------
`scripts/review_contributions.py` refuses to apply proposals automatically
because the database decides what a Range Officer is told. The same argument
applies here for a different reason: a photograph filed under the wrong species
does not announce itself. It becomes a class the model learns wrongly, and the
error surfaces later as a confident misidentification with no trace back to its
cause. So nothing is promoted unless it is named, and images submitted as
anything other than `confident` are refused outright — they need a
determination first, not a promotion.

CAPTURE GROUPING
----------------
The pool is split by capture, and the split is by animal rather than by
photograph. Contributions carry no capture id, so one is derived from the
contributor, the species and the submission date.

That deliberately errs toward grouping. Two animals merged into one capture
costs a little split flexibility. One animal split across two captures puts the
same individual in train and val, and the validation number stops measuring
anything — the failure this whole scheme exists to prevent. If a contributor
really did send two different animals on one day, promote them separately with
`--capture-suffix`.

The append-only contribution log is never rewritten. What has been promoted is
recorded in `contributions/promoted.json`, so re-running is safe and promotes
only what is new.
"""

from __future__ import annotations

import argparse
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
logger = logging.getLogger("promote")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import BASE_DIR, SPECIES_DB_PATH  # noqa: E402
from core.contributions import IMAGE_DIR, read_proposals  # noqa: E402

POOL_DIR = BASE_DIR / "pool"
LEDGER = BASE_DIR / "contributions" / "promoted.json"

SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(text: str) -> str:
    return SLUG_RE.sub("-", text.lower()).strip("-") or "anonymous"


def known_species_ids() -> set[str]:
    db = json.loads(SPECIES_DB_PATH.read_text(encoding="utf-8"))
    return {sp["id"] for sp in db["species"]}


def load_ledger() -> dict[str, str]:
    if not LEDGER.is_file():
        return {}
    try:
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {LEDGER}: {exc}")


def save_ledger(ledger: dict[str, str]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def image_records() -> list[dict]:
    return [r for r in read_proposals() if r.get("kind") == "image"]


def capture_for(record: dict, suffix: str | None) -> str:
    day = (record.get("submitted_utc") or "")[:10] or "undated"
    parts = [slug(record.get("contributor") or "anonymous"), day]
    if suffix:
        parts.append(slug(suffix))
    return "-".join(parts)


def show_list(records: list[dict], ledger: dict[str, str]) -> None:
    if not records:
        logger.info("No contributed images found in %s", IMAGE_DIR.parent)
        return

    known = known_species_ids()
    pending: dict[str, list[dict]] = defaultdict(list)
    promoted = blocked = 0

    for record in records:
        if record["id"] in ledger:
            promoted += 1
            continue
        species_id = record.get("species_id")
        if species_id not in known or record.get("certainty") != "confident":
            blocked += 1
            logger.info(
                "  %-10s HELD    %-28s %s",
                record["id"], species_id or "unidentified",
                "not a species id" if species_id not in known
                else f"certainty={record.get('certainty')}",
            )
            continue
        pending[species_id].append(record)

    for species_id in sorted(pending):
        for record in pending[species_id]:
            present = (IMAGE_DIR / record["image_file"]).is_file()
            logger.info(
                "  %-10s %-7s %-28s %s%s",
                record["id"], "READY" if present else "NO FILE", species_id,
                record.get("contributor") or "anonymous",
                "" if present else "   (image missing from contributions/images)",
            )

    total = sum(len(v) for v in pending.values())
    logger.info("%d ready to promote across %d taxa, %d already promoted, %d held",
                total, len(pending), promoted, blocked)


def promote(records: list[dict], ledger: dict[str, str], args) -> None:
    known = known_species_ids()
    chosen = []

    for record in records:
        if record["id"] in ledger:
            continue
        if record.get("species_id") not in known:
            continue
        if record.get("certainty") != "confident":
            continue
        if args.accept_all_confident or record["id"] in set(args.accept):
            chosen.append(record)

    unknown_ids = set(args.accept) - {r["id"] for r in records}
    if unknown_ids:
        raise SystemExit(
            "No contributed image has id " + ", ".join(sorted(unknown_ids))
            + ". Run --list to see what is pending."
        )
    if not chosen:
        logger.info("Nothing to promote. Run --list to see what is pending and why.")
        return

    by_capture: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in chosen:
        by_capture[(record["species_id"], capture_for(record, args.capture_suffix))].append(record)

    written = 0
    for (species_id, capture), group in sorted(by_capture.items()):
        target_dir = POOL_DIR / species_id
        target_dir.mkdir(parents=True, exist_ok=True)
        offset = len(list(target_dir.glob(f"{capture}--*")))

        for record in sorted(group, key=lambda r: r["id"]):
            source = IMAGE_DIR / record["image_file"]
            if not source.is_file():
                logger.warning("%s: %s is not in contributions/images — skipped",
                               record["id"], record["image_file"])
                continue
            offset += 1
            target = target_dir / f"{capture}--{offset:02d}.jpg"
            # Already scrubbed of EXIF on submission; copy the bytes as they are.
            target.write_bytes(source.read_bytes())
            ledger[record["id"]] = str(target.relative_to(BASE_DIR))
            written += 1
            logger.info("%s -> %s", record["id"], target.relative_to(BASE_DIR))

        logger.info("  %s: capture %r now holds %d photographs",
                    species_id, capture, offset)

    save_ledger(ledger)
    logger.info("Promoted %d photographs into %s. Ledger: %s",
                written, POOL_DIR, LEDGER)
    logger.info("Next: python -m training.prepare_dataset --pool ./pool --out ./dataset")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--list", action="store_true", help="Show what is pending and why")
    p.add_argument("--accept", action="append", default=[],
                   help="Contribution id to promote; repeatable")
    p.add_argument("--accept-all-confident", action="store_true",
                   help="Promote every named, confident, unpromoted image")
    p.add_argument("--capture-suffix", default=None,
                   help="Distinguish a second animal from the same contributor on the same day")
    args = p.parse_args()

    records = image_records()
    ledger = load_ledger()

    if args.list or (not args.accept and not args.accept_all_confident):
        show_list(records, ledger)
        return
    promote(records, ledger, args)


if __name__ == "__main__":
    main()
