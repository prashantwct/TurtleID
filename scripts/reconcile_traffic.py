#!/usr/bin/env python3
"""
Reconcile the curated database against the TRAFFIC/TSA-India 2023 ID Cards.

    python -m scripts.reconcile_traffic

Why this exists as a report rather than an import.

The TRAFFIC 2023 cards are the reference actually issued to Indian enforcement
agencies. If a Range Officer has a printed card in their pocket and this app
says something different, that divergence needs to be visible and explained,
not silently resolved in favour of whichever source the code happened to
prefer.

Two kinds of divergence appear, and they mean opposite things:

**IUCN status.** TRAFFIC 2023 reproduces assessments that were current when the
cards were compiled. Several have since been reassessed, generally upward:
Chitra indica, Indotestudo elongata and Nilssonia leithii are now Critically
Endangered on cards that read Endangered or Vulnerable. Here the local database
is the more current value and TRAFFIC is the stale one. The card is not wrong
about the law, only about the Red List.

**Distribution.** Here TRAFFIC is the stronger authority and the local database
is the claim that needs defending. TRAFFIC's state lists were compiled by TSA
India from verified records. Where this database asserts a state TRAFFIC does
not, that assertion needs a citation or it should be downgraded — because the
geographic prior in core/inference.py multiplies a candidate's probability by
occurrence class, so an unsupported state listing directly changes what the
model reports.

Nothing here is applied automatically. Both directions require a human.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import DATA_DIR  # noqa: E402
from core.database import SpeciesDB  # noqa: E402

CROSSWALK = DATA_DIR / "traffic_2023.json"


def main() -> int:
    db = SpeciesDB.load()
    payload = json.loads(CROSSWALK.read_text(encoding="utf-8"))
    ref = payload["taxa"]
    src = payload["source"]

    print()
    print("=" * 74)
    print("RECONCILIATION vs TRAFFIC / TSA-India / WWF-India ID Cards (2023)")
    print("=" * 74)
    print(f"  {src['authors']}")
    print(f"  {src['publisher']}, {src['year']}")

    status_diff, cites_diff, wpa_diff = [], [], []
    extra_states, missing_states, absent = [], [], []

    for sp in sorted(db, key=lambda s: s.scientific_name):
        entry = ref.get(sp.id)
        if not entry:
            if sp.id != "trachemys_scripta_elegans":
                absent.append(sp)
            continue

        if entry["iucn"] != sp.iucn_status:
            status_diff.append((sp, entry["iucn"]))
        if entry["cites"] != sp.cites:
            cites_diff.append((sp, entry["cites"]))
        if entry["wpa"] != sp.wpa:
            wpa_diff.append((sp, entry["wpa"]))

        ours, theirs = set(sp.states), set(entry["states"])
        if ours - theirs:
            extra_states.append((sp, sorted(ours - theirs)))
        if theirs - ours:
            missing_states.append((sp, sorted(theirs - ours)))

    # ---- legal status: this must not diverge -------------------------
    print("\n" + "-" * 74)
    print("WPA SCHEDULE  (must agree — this determines the offence category)")
    print("-" * 74)
    if wpa_diff:
        for sp, theirs in wpa_diff:
            print(f"  MISMATCH  {sp.scientific_name}: local {sp.wpa} / TRAFFIC {theirs}")
        print("\n  Resolve against the Gazette before anything else ships.")
    else:
        print("  All schedules agree, including the three Schedule II taxa")
        print("  (Melanochelys trijuga, Pangshura smithii, Cyclemys gemeli).")

    print("\n" + "-" * 74)
    print("CITES APPENDIX")
    print("-" * 74)
    if cites_diff:
        for sp, theirs in cites_diff:
            print(f"  {sp.scientific_name}: local {sp.cites} / TRAFFIC {theirs}")
    else:
        print("  No divergence.")

    # ---- IUCN: local expected to be newer ----------------------------
    print("\n" + "-" * 74)
    print("IUCN STATUS  (local is generally the newer assessment)")
    print("-" * 74)
    if status_diff:
        for sp, theirs in status_diff:
            note = ""
            order = ["LC", "NT", "VU", "EN", "CR", "EW", "EX"]
            if sp.iucn_status in order and theirs in order:
                if order.index(sp.iucn_status) > order.index(theirs):
                    note = "  <-- uplisted since the cards were printed"
                elif order.index(sp.iucn_status) < order.index(theirs):
                    note = "  <-- downlisted since the cards were printed"
            print(f"  {sp.scientific_name:32} card {theirs:3}  ->  now {sp.iucn_status}{note}")
        print("\n  Verify each against the current Red List before relying on it.")
        print("  Field staff carrying printed cards will see the older category.")
    else:
        print("  No divergence.")

    # ---- distribution: TRAFFIC is the stronger authority --------------
    print("\n" + "-" * 74)
    print("DISTRIBUTION — states this database claims that TRAFFIC does not")
    print("-" * 74)
    if extra_states:
        for sp, states in extra_states:
            flag = "  ** affects the geographic prior **" if "Madhya Pradesh" in states else ""
            print(f"  {sp.scientific_name}: {', '.join(states)}{flag}")
        print("\n  Each needs a citation or a downgrade. An unsupported state listing")
        print("  raises that species' posterior for any animal found there.")
    else:
        print("  None.")

    print("\n" + "-" * 74)
    print("DISTRIBUTION — states TRAFFIC lists that this database omits")
    print("-" * 74)
    if missing_states:
        for sp, states in missing_states:
            print(f"  {sp.scientific_name}: {', '.join(states)}")
        print("\n  Safe to add. Omissions here suppress a real candidate.")
    else:
        print("  None.")

    if absent:
        print("\n" + "-" * 74)
        print("IN THIS DATABASE BUT NOT ON THE CARDS")
        print("-" * 74)
        for sp in absent:
            print(f"  {sp.scientific_name} ({sp.common_en})")

    print()
    total = len(wpa_diff) + len(cites_diff) + len(status_diff) + len(extra_states)
    print(f"{total} item(s) need a human decision. Nothing has been changed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
