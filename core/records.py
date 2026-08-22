"""
Append-only determination log.

Every identification is written to a JSON Lines file before the result is shown.
Two reasons, both operational rather than technical:

  1. If a determination later turns out to be wrong, you need to know what the
     tool actually said at the time, not what someone remembers it saying.
  2. The log doubles as the training-data queue. Every INDETERMINATE and
     REJECTED record is a photograph the model needs, and the retraining
     pipeline reads this file to find them.

Writes are atomic and never raise into the UI: a failed log entry degrades the
audit trail, it does not stop a Range Officer identifying a turtle.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import APP_VERSION, RECORD_FILE

logger = logging.getLogger(__name__)


def image_fingerprint(image_bytes: bytes) -> str:
    """SHA-256 of the uploaded image. Links the log entry to a specific file."""
    return hashlib.sha256(image_bytes).hexdigest()[:16]


def _atomic_append(path: Path, line: str) -> None:
    """
    Append one line without risking a truncated file if the process dies.

    Appending a single line under 4 KiB to a file opened in append mode is
    atomic on POSIX, but we fsync so a field laptop losing power mid-write
    does not leave a partial record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def log_determination(
    determination_record: dict[str, Any],
    *,
    image_hash: str | None = None,
    observer: str | None = None,
    location_note: str | None = None,
    method: str = "model",
    path: Path = RECORD_FILE,
) -> bool:
    """Write one determination. Returns True on success, False on failure."""
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "app_version": APP_VERSION,
        "method": method,
        "observer": observer or "unrecorded",
        "location_note": location_note or "",
        "image_hash": image_hash or "",
        "determination": determination_record,
    }
    try:
        _atomic_append(path, json.dumps(entry, ensure_ascii=False))
        return True
    except OSError as exc:
        logger.error("Could not write determination record: %s", exc)
        return False


def read_records(path: Path = RECORD_FILE, limit: int | None = None) -> list[dict[str, Any]]:
    """Read back the log, skipping any corrupt lines rather than failing."""
    if not Path(path).exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed record at line %d", lineno)
    except OSError as exc:
        logger.error("Could not read determination records: %s", exc)
        return []
    return entries[-limit:] if limit else entries


def retraining_queue(path: Path = RECORD_FILE) -> list[dict[str, Any]]:
    """Records the model could not resolve — the images worth labelling next."""
    return [
        r for r in read_records(path)
        if r.get("determination", {}).get("tier") in {"INDETERMINATE", "REJECTED", "TENTATIVE"}
    ]
