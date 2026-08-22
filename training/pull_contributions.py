#!/usr/bin/env python3
"""
Bring contributions down from durable storage.

    python -m training.pull_contributions --list
    python -m training.pull_contributions

The deployed app writes every submission to object storage because its own disk
does not survive a reboot. This fetches what is there into the local
`contributions/` directory, in the same shape the app would have written it, so
that `promote_contributions.py` works identically whether the photographs were
submitted on this machine or on the hosted one.

Records already present locally are left alone and their photographs are not
re-downloaded, so this is cheap to re-run and safe to interrupt.

A record whose photograph is missing from the bucket is reported rather than
skipped silently — that combination means a submission was accepted and its
image lost, which is worth knowing about immediately rather than discovering as
a gap in the training set months later.

Configuration is the same as the app's; see `core/storage.py`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pull")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import storage  # noqa: E402
from core.contributions import (  # noqa: E402
    CONTRIB_DIR,
    IMAGE_DIR,
    PROPOSAL_FILE,
    read_proposals,
)


def local_record_ids() -> set[str]:
    return {r["id"] for r in read_proposals() if r.get("id")}


def pull(args: argparse.Namespace) -> None:
    try:
        config = storage.settings()
    except storage.StorageError as exc:
        raise SystemExit(str(exc))
    if config is None:
        raise SystemExit(
            "Durable storage is not configured, so there is nothing to pull. "
            "See core/storage.py for the settings, or work with the local "
            "contributions/ directory directly."
        )

    s3 = storage.client(config)
    logger.info("Reading %s", storage.describe())
    keys = storage.list_records(s3=s3, config=config)
    if not keys:
        logger.info("No contribution records in the bucket.")
        return

    known = local_record_ids()
    fetched = skipped = missing = 0
    new_records = []

    for key in sorted(keys):
        record_id = key.rsplit("/", 1)[-1].removesuffix(".json")
        if record_id in known:
            skipped += 1
            continue
        try:
            record = json.loads(storage.fetch(key, s3=s3, config=config))
        except (storage.StorageError, json.JSONDecodeError) as exc:
            logger.warning("%s: could not be read (%s)", key, exc)
            continue

        image_name = record.get("image_file")
        if args.list:
            logger.info("  %-10s %-28s %s", record.get("id"),
                        record.get("species_id") or "unidentified",
                        record.get("contributor") or "anonymous")
            fetched += 1
            continue

        if image_name:
            target = IMAGE_DIR / image_name
            if not target.is_file():
                image_key = config.key_for(f"{storage.IMAGE_PREFIX}{image_name}")
                try:
                    data = storage.fetch(image_key, s3=s3, config=config)
                except storage.StorageError as exc:
                    missing += 1
                    logger.warning(
                        "%s: record kept but its photograph is not in the "
                        "bucket (%s). The submission was accepted and the image "
                        "lost — worth chasing.", record.get("id"), exc,
                    )
                    data = None
                if data is not None:
                    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)

        new_records.append(record)
        fetched += 1

    if args.list:
        logger.info("%d record(s) in the bucket not present locally, %d already here",
                    fetched, skipped)
        return

    if new_records:
        CONTRIB_DIR.mkdir(parents=True, exist_ok=True)
        with open(PROPOSAL_FILE, "a", encoding="utf-8") as fh:
            for record in new_records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Pulled %d new contribution(s), %d already local, %d with a missing image",
                fetched, skipped, missing)
    if fetched:
        logger.info("Next: python -m training.promote_contributions --list")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--list", action="store_true",
                   help="Show what would be pulled; download nothing")
    pull(p.parse_args())


if __name__ == "__main__":
    main()
