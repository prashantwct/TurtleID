#!/usr/bin/env python3
"""
Validate the species database and the morphological key matrix.

    python -m scripts.validate_db

Run by CI on every pull request. Also the fastest way for a contributor to
check their edit before opening one.

Exit code 0 means the database will load. Non-zero means the app would fail at
startup, or would show a species panel with a broken cross-reference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import SPECIES_DB_PATH  # noqa: E402
from core.database import SpeciesDB, SpeciesDBError  # noqa: E402
from core.morphkey import CHARACTERS, MATRIX  # noqa: E402

VALID_WPA = {"Schedule I", "Schedule II", "Schedule III", "Schedule IV"}


def check(label: str, problems: list[str]) -> bool:
    if problems:
        print(f"  FAIL  {label}")
        for p in problems:
            print(f"          - {p}")
        return False
    print(f"  ok    {label}")
    return True


def main() -> int:
    print("\nValidating Chelonid-ID reference data\n" + "-" * 44)
    ok = True

    # 1. JSON parses
    try:
        raw = json.loads(SPECIES_DB_PATH.read_text(encoding="utf-8"))
        print(f"  ok    species_db.json parses ({len(raw['species'])} taxa)")
    except (json.JSONDecodeError, OSError, KeyError) as exc:
        print(f"  FAIL  species_db.json unreadable: {exc}")
        return 1

    # 2. Schema validation via the real loader
    try:
        db = SpeciesDB.load()
        print(f"  ok    schema validation ({len(db)} taxa loaded)")
    except SpeciesDBError as exc:
        print(f"  FAIL  schema validation:\n{exc}")
        return 1

    # 3. Key matrix covers every species, and vice versa
    ok &= check("every species has a key profile",
                [f"{i} missing from MATRIX in core/morphkey.py" for i in db.ids
                 if i not in MATRIX])
    ok &= check("key matrix has no orphans",
                [f"{i} in MATRIX but not in species_db.json" for i in MATRIX
                 if i not in db.ids])

    # 4. Key character states are declared
    bad_states = []
    for sid, profile in MATRIX.items():
        for char, expected in profile.items():
            if char not in CHARACTERS:
                bad_states.append(f"{sid}: unknown character '{char}'")
                continue
            if expected is None:
                continue
            for state in (expected if isinstance(expected, tuple) else (expected,)):
                if state not in CHARACTERS[char][1]:
                    bad_states.append(f"{sid}.{char}: undeclared state '{state}'")
    ok &= check("key character states are declared", bad_states)

    # 5. Legal fields are well formed
    legal = []
    for sp in db:
        if not any(sp.wpa.startswith(s) for s in VALID_WPA) and "Not a Schedule" not in sp.wpa:
            legal.append(f"{sp.id}: unrecognised WPA value '{sp.wpa}'")
        if "Appendix" not in sp.cites and sp.cites.lower() not in {"not listed", "none"}:
            legal.append(f"{sp.id}: unrecognised CITES value '{sp.cites}'")
    ok &= check("legal status fields well formed", legal)

    # 6. Every species is citable and identifiable
    sourcing = []
    for sp in db:
        if not sp.references:
            sourcing.append(f"{sp.id}: no references")
        if not sp.key_character.strip():
            sourcing.append(f"{sp.id}: empty key_character")
        if len(sp.diagnostics) < 2:
            sourcing.append(f"{sp.id}: fewer than two diagnostic characters")
        if not sp.states:
            sourcing.append(f"{sp.id}: no distribution states")
    ok &= check("sourcing and diagnostics present", sourcing)

    # 7. Confusion pairs resolve and are informative
    pairs = []
    for sp in db:
        for pair in sp.confusion_with:
            if pair.get("species_id") not in db:
                pairs.append(f"{sp.id}: points at unknown '{pair.get('species_id')}'")
            elif len(pair.get("discriminator", "")) < 20:
                pairs.append(f"{sp.id} vs {pair['species_id']}: discriminator too vague")
    ok &= check("confusion pairs resolve", pairs)

    # 8. Distribution states are recognised
    from config import STATE_CENTROIDS
    geo = [f"{sp.id}: unrecognised state '{s}'"
           for sp in db for s in sp.states if s not in STATE_CENTROIDS]
    ok &= check("distribution states recognised", geo)

    print("-" * 44)
    if ok:
        print("PASS — the database will load cleanly.\n")
        return 0
    print("FAIL — fix the items above before opening a pull request.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
