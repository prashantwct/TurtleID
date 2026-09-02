"""
Contribution intake.

Field staff and researchers contribute three things, and they are handled
differently because they carry different risk.

**Images.** The most valuable contribution and the most dangerous to mishandle.
Photographs carry EXIF GPS, and a coordinate for a Batagur kachuga nesting bank
in the Chambal or an Indotestudo elongata in a tiger reserve is poaching-
relevant. So: EXIF is stripped on intake, images are written outside the
repository tree, and the manifest records a hash and a state-level locality
only. Nothing finer than a state ever enters a file that could be committed.

**Corrections to species records.** A diagnostic character that misleads, a
distribution that does not match a published source, a status that has been
reassessed. These become a structured proposal with a citation requirement,
which a maintainer applies by hand. Nothing edits species_db.json directly —
the geographic prior and the legal status shown to a Range Officer both come
out of that file.

**Field determinations.** Already handled by core/records.py.

Contributions land in contributions/ as JSON, which is gitignored. A maintainer
runs scripts/review_contributions.py, decides, and edits the database.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import APP_VERSION, BASE_DIR, INDIAN_STATES
from core import storage

logger = logging.getLogger(__name__)

CONTRIB_DIR = BASE_DIR / "contributions"
IMAGE_DIR = CONTRIB_DIR / "images"
PROPOSAL_FILE = CONTRIB_DIR / "proposals.jsonl"

# Strip anything that could be a coordinate, a phone number, or an ID.
COORD_PATTERN = re.compile(
    r"\b\d{1,3}\.\d{4,}\s*[,°]?\s*\d{1,3}\.\d{4,}\b|"
    r"\b\d{1,3}°\s*\d{1,2}['′]\s*[\d.]+[\"″]?\s*[NSEW]\b",
    re.IGNORECASE,
)
LONG_DIGITS = re.compile(r"\b\d{7,}\b")

CONTRIBUTION_KINDS = {
    "image": "Photograph for the training set",
    "correction": "Correction to an existing species record",
    "new_species": "A taxon this database does not cover",
    "distribution": "New or disputed distribution record",
}


class ContributionError(Exception):
    """Raised when a submission cannot be accepted."""


# ------------------------------------------------------------------ scrubbing

def scrub_free_text(text: str) -> tuple[str, list[str]]:
    """
    Remove coordinates and long digit strings from a free-text field.

    Contributors paste coordinates into notes fields constantly, usually with
    good intentions. Returns the cleaned text and a list of what was removed so
    the UI can tell them rather than silently editing what they wrote.
    """
    removed: list[str] = []
    cleaned = text or ""

    for match in COORD_PATTERN.findall(cleaned):
        removed.append("coordinates")
    cleaned = COORD_PATTERN.sub("[coordinates removed]", cleaned)

    for match in LONG_DIGITS.findall(cleaned):
        removed.append("long numeric string")
    cleaned = LONG_DIGITS.sub("[number removed]", cleaned)

    return cleaned.strip(), sorted(set(removed))


def strip_exif(image_bytes: bytes) -> tuple[bytes, bool]:
    """
    Re-encode an image without any metadata.

    Returns (bytes, had_gps). Rather than editing EXIF tags selectively, the
    pixel data is copied into a fresh image — anything not pixels is gone,
    including maker notes and thumbnails that survive naive EXIF stripping.
    """
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ContributionError("Pillow is required to process images.") from exc

    try:
        original = Image.open(io.BytesIO(image_bytes))
        had_gps = False
        try:
            exif = original.getexif()
            had_gps = bool(exif and exif.get_ifd(0x8825))
        except Exception:
            pass

        original = ImageOps.exif_transpose(original).convert("RGB")
        clean = Image.new("RGB", original.size)
        clean.putdata(list(original.getdata()))

        buffer = io.BytesIO()
        clean.save(buffer, format="JPEG", quality=92, optimize=True)
        return buffer.getvalue(), had_gps
    except ContributionError:
        raise
    except Exception as exc:
        raise ContributionError(f"Could not process that image: {exc}") from exc


# ------------------------------------------------------------------ intake

def _write(record: dict[str, Any]) -> bool:
    try:
        CONTRIB_DIR.mkdir(parents=True, exist_ok=True)
        with open(PROPOSAL_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except OSError as exc:
        logger.error("Could not write contribution: %s", exc)
        return False


def submit_image(
    image_bytes: bytes,
    *,
    species_id: str | None,
    view: str,
    state: str | None,
    contributor: str,
    notes: str = "",
    certainty: str = "confident",
) -> dict[str, Any]:
    """Accept a training photograph. EXIF is stripped before anything is written."""
    if state and state not in INDIAN_STATES:
        raise ContributionError(f"Unrecognised state: {state}")

    clean_bytes, had_gps = strip_exif(image_bytes)
    digest = hashlib.sha256(clean_bytes).hexdigest()[:16]

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    target = IMAGE_DIR / f"{species_id or 'unidentified'}_{digest}.jpg"
    if target.exists():
        raise ContributionError("This image has already been contributed.")

    # Durable storage first, when it is configured. The local directory does not
    # survive a reboot on a hosted deployment, so a photograph that exists only
    # there has not really been received — better to refuse it and let the
    # contributor try again than to thank them for something already lost.
    stored_key = None
    if storage.configured():
        try:
            stored_key = storage.put_image(target.name, clean_bytes)
        except storage.StorageError as exc:
            logger.error("Durable storage rejected a contribution: %s", exc)
            # The reason is the whole point: it names what a maintainer has to
            # fix. Logging it and showing "please try again" sends someone to
            # hunt through deployment logs for a line already in hand.
            raise ContributionError(
                "This photograph could not be stored and has not been kept. "
                "Nothing has been lost from your device.\n\n"
                f"**Reason:** {storage.safe_reason(exc)}\n\n"
                "If you maintain this deployment, that line says what to change. "
                "Otherwise please pass it on — retrying will not help until it "
                "is fixed."
            ) from exc

    target.write_bytes(clean_bytes)

    clean_notes, redactions = scrub_free_text(notes)

    record = {
        "id": str(uuid.uuid4())[:8],
        "kind": "image",
        "submitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "app_version": APP_VERSION,
        "contributor": (contributor or "anonymous")[:120],
        "species_id": species_id,
        "certainty": certainty,
        "view": view,
        "state": state,
        "image_hash": digest,
        "image_file": target.name,
        "notes": clean_notes,
        "exif_gps_present_and_removed": had_gps,
        "redactions": redactions,
        "status": "pending",
        "stored_key": stored_key,
    }
    _write(record)

    if stored_key:
        try:
            storage.put_record(record)
        except storage.StorageError as exc:
            # The photograph is already safe; losing its metadata costs the
            # species label, which pull_contributions reports rather than
            # guessing at. Not worth failing the submission over.
            logger.error("Stored %s but not its record: %s", stored_key, exc)

    return record


def submit_proposal(
    *,
    kind: str,
    species_id: str | None,
    field: str,
    current_value: str,
    proposed_value: str,
    citation: str,
    contributor: str,
    rationale: str = "",
) -> dict[str, Any]:
    """
    Accept a proposed change to a species record.

    A citation is mandatory and is not a formality. Every line in the database
    is traceable to a published source; a change without one would break that,
    and the field characters are what a Range Officer acts on.
    """
    if kind not in CONTRIBUTION_KINDS:
        raise ContributionError(f"Unknown contribution kind: {kind}")
    if len(citation.strip()) < 15:
        raise ContributionError(
            "A citation is required — a paper, handbook, official notification, "
            "or an institutional record with a specimen or accession number. "
            "Personal observation alone is not enough to change a field "
            "character other people will act on, though it is very welcome as "
            "a photograph contribution."
        )
    if not proposed_value.strip():
        raise ContributionError("The proposed value is empty.")

    clean_rationale, redactions = scrub_free_text(rationale)

    record = {
        "id": str(uuid.uuid4())[:8],
        "kind": kind,
        "submitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "app_version": APP_VERSION,
        "contributor": (contributor or "anonymous")[:120],
        "species_id": species_id,
        "field": field,
        "current_value": current_value[:2000],
        "proposed_value": proposed_value[:4000],
        "citation": citation.strip()[:1000],
        "rationale": clean_rationale[:2000],
        "redactions": redactions,
        "status": "pending",
    }
    _write(record)
    return record


# ------------------------------------------------------------------ review

def read_proposals(status: str | None = None) -> list[dict[str, Any]]:
    if not PROPOSAL_FILE.exists():
        return []
    out = []
    try:
        for lineno, line in enumerate(
            PROPOSAL_FILE.read_text(encoding="utf-8").splitlines(), 1
        ):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed contribution at line %d", lineno)
                continue
            if status is None or record.get("status") == status:
                out.append(record)
    except OSError as exc:
        logger.error("Could not read contributions: %s", exc)
    return out


def summarise() -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in read_proposals():
        key = record.get("kind", "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def image_coverage(species_ids: list[str]) -> dict[str, int]:
    """How many contributed images exist per species — the collection gap map."""
    counts = {sid: 0 for sid in species_ids}
    for record in read_proposals():
        if record.get("kind") == "image":
            sid = record.get("species_id")
            if sid in counts:
                counts[sid] += 1
    return counts
