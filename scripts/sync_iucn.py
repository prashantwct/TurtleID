#!/usr/bin/env python3
"""
Sync IUCN Red List data for every species in the local database.

    export IUCN_API_TOKEN="your-token"
    python -m scripts.sync_iucn                 # fetch and cache
    python -m scripts.sync_iucn --report        # show divergences, no fetching
    python -m scripts.sync_iucn --species "Batagur kachuga"

Deliberately does NOT edit data/species_db.json.

IUCN supplies the conservation category, criteria, population trend, threats
and habitat coding. It does not supply the WPA-2022 schedule, the diagnostic
characters, the confusion pairs, or the Madhya Pradesh occurrence notes — which
is most of what makes the local database useful. An automatic overwrite would
be able to damage those fields and would remove the human check on a status
change that alters what a Range Officer is told.

So the sync writes to data/iucn_cache.json and prints a divergence report. A
person applies the changes. On thirty taxa that is a few minutes of work per
Red List release, twice a year.

Requesting a token: https://api.iucnredlist.org/users/sign_up
Citation: IUCN. IUCN Red List of Threatened Species. https://www.iucnredlist.org
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("sync_iucn")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.database import SpeciesDB  # noqa: E402
from core.iucn import SIGNUP_URL, TOKEN_ENV, IUCNClient, compare_with_local  # noqa: E402


def report(db: SpeciesDB, client: IUCNClient) -> int:
    """Print a divergence report from cached data. Returns the divergence count."""
    matches, divergent, missing = [], [], []

    for sp in sorted(db, key=lambda s: s.scientific_name):
        cached = client.cached(sp.scientific_name)
        verdict = compare_with_local(sp.iucn_status, cached)
        if verdict["status"] == "match":
            matches.append(sp)
        elif verdict["status"] == "divergent":
            divergent.append((sp, cached, verdict))
        else:
            missing.append(sp)

    print()
    print("=" * 72)
    print(f"IUCN SYNC REPORT   Red List version: {client.red_list_version or 'unknown'}")
    print("=" * 72)
    print(f"  agree      {len(matches):3d}")
    print(f"  divergent  {len(divergent):3d}")
    print(f"  no data    {len(missing):3d}")

    if divergent:
        print("\n--- REVIEW REQUIRED " + "-" * 52)
        for sp, cached, verdict in divergent:
            print(f"\n  {sp.scientific_name}")
            print(f"    local : {verdict['local_category']}")
            print(f"    IUCN  : {verdict['iucn_category']} "
                  f"({cached.get('year_published', '?')})")
            if cached.get("criteria"):
                print(f"    crit  : {cached['criteria']}")
            hist = cached.get("history") or []
            if len(hist) > 1:
                trail = " -> ".join(
                    f"{h['category']}({h['year']})" for h in hist if h.get("category")
                )
                print(f"    trail : {trail}")
            print(f"    edit  : data/species_db.json -> id '{sp.id}' -> iucn.status")

    if missing:
        print("\n--- NO IUCN RECORD CACHED " + "-" * 46)
        for sp in missing:
            print(f"  {sp.scientific_name}")
        print("\n  Usually means the sync has not run, or IUCN uses a different")
        print("  name for this taxon. Check the accepted binomial before assuming")
        print("  the species is unassessed.")

    print()
    return len(divergent)


def main() -> None:
    p = argparse.ArgumentParser(description="Sync IUCN Red List data.")
    p.add_argument("--report", action="store_true",
                   help="Report from cache only; make no network requests")
    p.add_argument("--species", help="Sync a single binomial")
    p.add_argument("--force", action="store_true",
                   help="Re-fetch species already cached")
    args = p.parse_args()

    db = SpeciesDB.load()
    client = IUCNClient()

    if args.report:
        report(db, client)
        return

    if not client.configured:
        print(
            f"\nNo IUCN token found.\n\n"
            f"  1. Request one at {SIGNUP_URL}\n"
            f"  2. export {TOKEN_ENV}=\"your-token\"   "
            f"(or put it in a .env file, which is gitignored)\n\n"
            f"Reporting from cache instead.\n"
        )
        report(db, client)
        return

    version = client.fetch_red_list_version()
    if version:
        logger.info("IUCN Red List version: %s", version)
        client._cache.setdefault("_meta", {})["red_list_version"] = version

    targets = (
        [sp for sp in db if sp.scientific_name == args.species]
        if args.species else list(db)
    )
    if args.species and not targets:
        raise SystemExit(f"{args.species!r} is not in the local database.")

    fetched = skipped = failed = 0
    for i, sp in enumerate(targets, 1):
        if not args.force and client.cached(sp.scientific_name):
            skipped += 1
            continue
        logger.info("[%d/%d] %s", i, len(targets), sp.scientific_name)
        if client.fetch_species(sp.scientific_name):
            fetched += 1
        else:
            failed += 1

    logger.info("Fetched %d, skipped %d (already cached), failed %d",
                fetched, skipped, failed)
    client._save_cache()

    divergences = report(db, client)
    if divergences:
        print(f"{divergences} status divergence(s) need a human decision.")
        print("Nothing has been changed in data/species_db.json.\n")


if __name__ == "__main__":
    main()
