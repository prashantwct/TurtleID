"""
Reference photographs.

These are published identification plates, extracted by
`training/extract_id_cards.py` and recorded in `data/reference_images.json`.
The manifest is tracked; the image files are not, because their copyright sits
with the photographers and publishers named in it (see PUBLISHING.md).

That split is deliberate and everything here is written around it: a manifest
entry whose file is absent is normal, not an error. A checkout that has never
run the extractor — including the deployed app — simply shows no plates, and
the species reference works exactly as it did before.

Nothing in this module imports Pillow or reads image data. It resolves paths
and provenance; the caller displays the file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from config import REFERENCE_IMAGE_DIR, REFERENCE_MANIFEST

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Plate:
    """One reference photograph that is present on disk."""

    species_id: str
    path: Path
    credit: str | None
    source_title: str
    source_publisher: str
    source_year: int | None
    page: int | None
    rights: str

    @property
    def attribution(self) -> str:
        """One line naming the photographer and the publication."""
        who = self.credit or "Photographer not credited in source"
        where = self.source_title
        if self.source_year:
            where = f"{where} ({self.source_year})"
        return f"{who} — {where}"


@lru_cache(maxsize=1)
def _manifest() -> dict:
    if not REFERENCE_MANIFEST.is_file():
        return {}
    try:
        return json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # A malformed manifest must not take down the species reference.
        logger.warning("Could not read %s: %s", REFERENCE_MANIFEST, exc)
        return {}


def plates_for(species_id: str) -> list[Plate]:
    """Every reference photograph for a species whose file is actually present.

    Entries whose file is missing are skipped silently — that is the normal
    state of a fresh checkout, not a fault worth logging on every rerun.
    """
    manifest = _manifest()
    sources = manifest.get("sources", {})
    found = []

    for entry in manifest.get("images", {}).get(species_id, []):
        path = REFERENCE_IMAGE_DIR / entry["file"]
        if not path.is_file():
            continue
        source = sources.get(entry.get("source"), {})
        found.append(Plate(
            species_id=species_id,
            path=path,
            credit=entry.get("credit"),
            source_title=source.get("title", "Unrecorded source"),
            source_publisher=source.get("publisher", ""),
            source_year=source.get("year"),
            page=entry.get("page"),
            rights=source.get("rights", ""),
        ))
    return found


def coverage() -> tuple[int, int]:
    """(taxa with a plate on disk, taxa listed in the manifest).

    The two differ whenever the manifest is present but the images have not
    been extracted, which is what the deployed app sees.
    """
    listed = _manifest().get("images", {})
    present = sum(1 for species_id in listed if plates_for(species_id))
    return present, len(listed)
