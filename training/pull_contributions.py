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

Two shapes, matching the two backends.

* **GitHub.** There is nothing to download — `git pull` has already brought
  every submission into the working tree. This copies them out of
  `submissions/` into the local layout `promote_contributions` reads. Chosen
  automatically when the GitHub backend is configured or a `submissions/`
  directory is simply present; force it with `--from-repo`.
* **Object storage.** Records and photographs are fetched from the bucket.

Either way it is idempotent: records already in the local log are left alone
and their photographs are not re-copied.

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

from core import github_storage, storage  # noqa: E402
from core.contributions import (  # noqa: E402
    CONTRIB_DIR,
    IMAGE_DIR,
    PROPOSAL_FILE,
    read_proposals,
)


def local_record_ids() -> set[str]:
    return {r["id"] for r in read_proposals() if r.get("id")}


def absorb(new_records: list[dict]) -> None:
    """Append records to the local log the rest of the pipeline reads."""
    if not new_records:
        return
    CONTRIB_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROPOSAL_FILE, "a", encoding="utf-8") as fh:
        for record in new_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def pull_from_repo(args: argparse.Namespace) -> None:
    """Take up submissions that are already here, committed by the app.

    With the GitHub backend there is nothing to download: `git pull` has
    already brought every submission into the working tree. This copies them
    into the local layout `promote_contributions` reads, which is the only
    reason the step exists at all.
    """
    records_dir = args.submissions / "records"
    images_dir = args.submissions / "images"
    if not records_dir.is_dir():
        raise SystemExit(
            f"No submissions at {records_dir}. If the app commits to a "
            f"repository, `git pull` first — that is what brings them here. If "
            f"it commits somewhere else, pass --submissions with that path."
        )

    known = local_record_ids()
    new_records: list[dict] = []
    found = skipped = missing = 0

    for path in sorted(records_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("%s: could not be read (%s)", path.name, exc)
            continue

        record_id = record.get("id")
        if record_id in known:
            skipped += 1
            continue

        if args.list:
            logger.info("  %-10s %-28s %s", record_id,
                        record.get("species_id") or "unidentified",
                        record.get("contributor") or "anonymous")
            found += 1
            continue

        image_name = record.get("image_file")
        if image_name:
            source = images_dir / image_name
            target = IMAGE_DIR / image_name
            if source.is_file():
                if not target.is_file():
                    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read_bytes())
            elif not target.is_file():
                missing += 1
                logger.warning(
                    "%s: record kept but its photograph is not at %s. The "
                    "submission was accepted and the image lost — worth chasing.",
                    record_id, source,
                )

        new_records.append(record)
        found += 1

    if args.list:
        logger.info("%d submission(s) here and not yet local, %d already local",
                    found, skipped)
        return

    absorb(new_records)
    logger.info("Took up %d new contribution(s) from %s, %d already local, "
                "%d with a missing image", found, args.submissions, skipped, missing)
    if found:
        logger.info("Next: python -m training.promote_contributions --list")


def pull(args: argparse.Namespace) -> None:
    if args.from_repo or _repo_is_the_source(args):
        return pull_from_repo(args)

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

    absorb(new_records)

    logger.info("Pulled %d new contribution(s), %d already local, %d with a missing image",
                fetched, skipped, missing)
    if fetched:
        logger.info("Next: python -m training.promote_contributions --list")


def _repo_is_the_source(args: argparse.Namespace) -> bool:
    """Whether submissions arrive by git rather than by download.

    True when the app is configured to commit them, or when a submissions
    directory is simply sitting here — which is what a maintainer who cloned
    the repository and ran nothing else will have.
    """
    if github_storage.configured():
        return True
    return (args.submissions / "records").is_dir()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--list", action="store_true",
                   help="Show what would be pulled; download nothing")
    p.add_argument("--from-repo", action="store_true",
                   help="Read submissions committed to this repository, after a "
                        "git pull, rather than downloading from object storage")
    p.add_argument("--submissions", type=Path, default=REPO_ROOT / "submissions",
                   help="Where committed submissions live (default: submissions/)")
    pull(p.parse_args())


if __name__ == "__main__":
    main()
