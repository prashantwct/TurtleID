#!/usr/bin/env python3
"""
Review contributions.

    python -m scripts.review_contributions              # summary
    python -m scripts.review_contributions --proposals  # pending edits, in full
    python -m scripts.review_contributions --coverage   # image gap map

This prints; it does not apply. Every accepted proposal is edited into
data/species_db.json by hand, then scripts/validate_db.py is run.

That is deliberate friction. The database determines the WPA schedule and the
diagnostic characters shown to someone who may be writing a seizure memo, and
the distribution lists feed the geographic prior in core/inference.py. An
automated apply path would let a well-meaning contributor change what the tool
tells a Range Officer without anyone reading it first.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.contributions import (  # noqa: E402
    CONTRIBUTION_KINDS,
    image_coverage,
    read_proposals,
)
from core.database import SpeciesDB  # noqa: E402

TARGET_PER_CLASS = 30


def show_coverage(db: SpeciesDB) -> None:
    coverage = image_coverage(db.ids)
    print("\nIMAGE COVERAGE\n" + "-" * 62)
    ready = 0
    for sid, n in sorted(coverage.items(), key=lambda kv: kv[1]):
        sp = db.get(sid)
        bar = "#" * min(30, n)
        if n >= TARGET_PER_CLASS:
            ready += 1
            flag = ""
        elif n == 0:
            flag = "  <-- nothing yet"
        else:
            flag = f"  <-- {TARGET_PER_CLASS - n} short"
        print(f"  {sp.scientific_name:32} {n:4} {bar}{flag}")
    print("-" * 62)
    print(f"  {ready} of {len(coverage)} classes at or above {TARGET_PER_CLASS} images.")
    print(
        "\n  Classes under the target will have poor recall. Leave them in with\n"
        "  conservative thresholds so the tool abstains rather than guesses —\n"
        "  INDETERMINATE is the true answer for a class with nine images.\n"
    )


def show_proposals(db: SpeciesDB) -> None:
    pending = [p for p in read_proposals("pending") if p.get("kind") != "image"]
    if not pending:
        print("\nNo pending proposals.\n")
        return

    print(f"\n{len(pending)} PENDING PROPOSAL(S)\n" + "=" * 62)
    for p in pending:
        name = db.get(p["species_id"]).scientific_name if p.get("species_id") in db else "(new taxon)"
        print(f"\n  [{p['id']}]  {CONTRIBUTION_KINDS.get(p['kind'], p['kind'])}")
        print(f"  species   : {name}")
        print(f"  field     : {p.get('field')}")
        print(f"  from      : {p.get('contributor')}  ({p.get('submitted_utc')})")
        print(f"\n  CURRENT   : {(p.get('current_value') or '(none)')[:400]}")
        print(f"  PROPOSED  : {p.get('proposed_value', '')[:400]}")
        print(f"  CITATION  : {p.get('citation', '')[:400]}")
        if p.get("rationale"):
            print(f"  RATIONALE : {p['rationale'][:300]}")
        if p.get("redactions"):
            print(f"  REDACTED  : {', '.join(p['redactions'])}")
        if p.get("field") in {"wpa_2022", "cites"}:
            print("  !! LEGAL FIELD — verify against the Gazette or the CITES")
            print("     appendices directly, not against a secondary source.")
        if p.get("field") == "distribution_states":
            print("  !! Changes the geographic prior. A new state raises that")
            print("     species' posterior for every animal found there.")
        print("  " + "-" * 58)
    print(
        "\n  To apply: edit data/species_db.json by hand, then run\n"
        "  python -m scripts.validate_db\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    args = ap.parse_args()

    db = SpeciesDB.load()
    records = read_proposals()

    if not args.proposals and not args.coverage:
        kinds = Counter(r.get("kind") for r in records)
        states = Counter(r.get("state") for r in records if r.get("state"))
        gps = sum(1 for r in records if r.get("exif_gps_present_and_removed"))

        print("\nCONTRIBUTION SUMMARY\n" + "-" * 62)
        if not records:
            print("  Nothing yet.")
        for kind, n in kinds.most_common():
            print(f"  {CONTRIBUTION_KINDS.get(kind, kind):42} {n:4}")
        if states:
            print("\n  By state")
            for state, n in states.most_common(8):
                print(f"    {state:36} {n:4}")
        if gps:
            print(f"\n  {gps} image(s) arrived carrying GPS metadata, since removed.")
        print("\n  --proposals for pending edits, --coverage for the image gap map.\n")
        return 0

    if args.coverage:
        show_coverage(db)
    if args.proposals:
        show_proposals(db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
