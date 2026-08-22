#!/usr/bin/env python3
"""
Pull the species photographs out of a published identification-card PDF.

    python -m training.extract_id_cards --pdf "Tortoise and Freshwater Turtles ID Cards 2023.pdf"

Written against the TRAFFIC India / TSA-India / WWF-India *Identification
Cards: Tortoises and Freshwater Turtles of India* (2023), the same document
`data/traffic_2023.json` is reconciled against. Each species card is one page
carrying three images: the photograph, a distribution map, and the publisher
logo strip repeated on every card.

Only the photograph is taken.

* The **logo** is dropped because it is byte-identical across pages, which is
  also how it is detected — no size threshold to tune.

* The **distribution map** is dropped for two reasons. The app already draws
  its own map from `states` in `species_db.json`, so the printed one is
  redundant; and at least one of them is wrong in the source. The map on the
  *Lissemys punctata* card (p16 of the 2023 edition) is legended *Chitra
  indica*. Importing that would put a mislabelled range map next to a Schedule I
  species, which is the kind of error this tool exists to avoid.

* The **photograph** is the largest image on the page by pixel area, which
  separates it from the map by a factor of five in this document.

A page counts as a species card only if it carries the card's own headings, so
the front matter, the hardshell/softshell key plates and the glossary are never
considered. Cards are then matched to a species by the scientific name in their
text, taking synonyms from the `data/traffic_2023.json` crosswalk. A card that
matches nothing is reported and skipped rather than guessed at.

The images are written to `reference_images/<species_id>/` which is gitignored.
`data/reference_images.json` records where every file came from, who took it,
and under what terms, and is tracked. Committing the photographs themselves is
a licensing decision, not an engineering one: see PUBLISHING.md.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("extract")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import DATA_DIR, REFERENCE_IMAGE_DIR, REFERENCE_MANIFEST, SPECIES_DB_PATH  # noqa: E402

# A page is a species card only if it carries the card's own headings. In the
# 2023 edition this selects pages 10-37 exactly, and keeps the introduction out
# — that page discusses several species by name in prose and would otherwise
# match one of them.
CARD_MARKER = "IDENTIFICATION FEATURES"

TRAFFIC_CROSSWALK = DATA_DIR / "traffic_2023.json"
SYNONYM_RE = re.compile(r"^as ([A-Z][a-z]+ [a-z]+)$")

# Source descriptor written into the manifest. Update it if you point this at a
# different publication, and keep `rights` honest — it is the only record of
# what may be done with the files.
DEFAULT_SOURCE = {
    "id": "traffic_2023",
    "title": "Identification Cards: Tortoises and Freshwater Turtles of India",
    "publisher": "TRAFFIC India Office, Turtle Survival Alliance (TSA)-India, WWF-India",
    "year": 2023,
    "rights": (
        "Copyright rests with the publishers and with the individual "
        "photographers credited per image. Extracted here for reference use "
        "within this tool. Not cleared for redistribution — the image files are "
        "gitignored, and clearing them is a decision for the project owner "
        "(see PUBLISHING.md)."
    ),
}

CREDIT_RE = re.compile(r"©\s*([^\n©]{2,60})")


def load_species_index() -> dict[str, str]:
    """Scientific name (lowercased) -> species id, including printed synonyms.

    The cards use the name current when they were published, which is not
    always the name the database carries — the 2023 edition prints *Amyda
    cartilaginea* for what is here `amyda_ornata`. Those are already reconciled
    in `data/traffic_2023.json` as `"note": "as Amyda cartilaginea"`, so the
    synonyms come from there rather than from a taxonomy call made in this
    script.
    """
    db = json.loads(SPECIES_DB_PATH.read_text(encoding="utf-8"))
    index = {sp["scientific_name"].lower(): sp["id"] for sp in db["species"]}

    if TRAFFIC_CROSSWALK.is_file():
        crosswalk = json.loads(TRAFFIC_CROSSWALK.read_text(encoding="utf-8"))
        for species_id, entry in crosswalk["taxa"].items():
            m = SYNONYM_RE.match((entry.get("note") or "").strip())
            if m and species_id in index.values():
                index.setdefault(m.group(1).lower(), species_id)
    return index


def repeated_image_hashes(doc) -> set[str]:
    """Digests of images that appear on more than one page.

    Page furniture — logos, rules, the publisher strip — is repeated verbatim.
    Photographs are not.
    """
    seen: Counter[str] = Counter()
    for page in doc:
        for digest in {
            hashlib.sha256(doc.extract_image(img[0])["image"]).hexdigest()
            for img in page.get_images(full=True)
        }:
            seen[digest] += 1
    return {digest for digest, n in seen.items() if n > 1}


def match_species(text: str, index: dict[str, str]) -> str | None:
    """The species id whose scientific name appears in the page text.

    Longest name first, so that a binomial is preferred over a genus that is a
    prefix of it.
    """
    lowered = text.lower()
    for name in sorted(index, key=len, reverse=True):
        if name in lowered:
            return index[name]
    return None


def page_credit(text: str) -> str | None:
    m = CREDIT_RE.search(text)
    return f"© {m.group(1).strip()}" if m else None


def extract(pdf_path: Path, out_dir: Path, source: dict, dry_run: bool) -> dict:
    import pymupdf  # imported here so the rest of the repo never needs it
    from PIL import Image

    index = load_species_index()
    doc = pymupdf.open(pdf_path)
    furniture = repeated_image_hashes(doc)
    logger.info(
        "%s: %d pages, %d names recognised, %d repeated page-furniture images",
        pdf_path.name, doc.page_count, len(index), len(furniture),
    )

    images: dict[str, list[dict]] = {}
    unmatched: list[int] = []
    seen_species: set[str] = set()

    for number, page in enumerate(doc, start=1):
        text = page.get_text()
        if CARD_MARKER not in text:
            continue

        species_id = match_species(text, index)
        if species_id is None:
            unmatched.append(number)
            continue

        candidates = []
        for img in page.get_images(full=True):
            info = doc.extract_image(img[0])
            digest = hashlib.sha256(info["image"]).hexdigest()
            if digest in furniture:
                continue
            picture = Image.open(io.BytesIO(info["image"]))
            candidates.append((picture.width * picture.height, picture, digest))

        if not candidates:
            logger.warning("p%d (%s): no image left after dropping furniture",
                           number, species_id)
            continue

        # Largest by area is the photograph; everything smaller on these cards
        # is the distribution map, which is deliberately not imported.
        _, picture, digest = max(candidates, key=lambda c: c[0])
        dropped = len(candidates) - 1

        relative = Path(species_id) / f"{source['id']}-p{number:02d}.jpg"
        record = {
            "file": relative.as_posix(),
            "source": source["id"],
            "page": number,
            "credit": page_credit(text),
            "width": picture.width,
            "height": picture.height,
            "sha256": digest,
        }
        images.setdefault(species_id, []).append(record)
        seen_species.add(species_id)

        if not dry_run:
            target = out_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # The cards are CMYK JPEGs; the app and the training pipeline both
            # want RGB.
            picture.convert("RGB").save(target, "JPEG", quality=95)

        logger.info("p%d %-28s %dx%d  %s%s", number, species_id,
                    picture.width, picture.height, record["credit"] or "uncredited",
                    f"  (dropped {dropped} smaller)" if dropped else "")

    if unmatched:
        logger.warning(
            "species cards naming a taxon this database does not carry (skipped): %s. "
            "Add the taxon, or record the printed name in data/traffic_2023.json "
            "as \"note\": \"as <printed name>\".",
            ", ".join(str(n) for n in unmatched),
        )

    missing = sorted(set(index.values()) - seen_species)
    if missing:
        logger.warning("%d of %d taxa have no photograph in this document: %s",
                       len(missing), len(index), ", ".join(missing))

    return {
        "schema_version": 1,
        "generated": date.today().isoformat(),
        "generated_by": "training/extract_id_cards.py",
        "note": (
            "Provenance record for reference photographs. The image files live "
            "in reference_images/ and are not tracked; re-create them by "
            "running this script against the source PDF."
        ),
        "sources": {source["id"]: {k: v for k, v in source.items() if k != "id"}},
        "images": {k: images[k] for k in sorted(images)},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--pdf", required=True, type=Path, help="Source identification-card PDF")
    p.add_argument("--out", type=Path, default=REFERENCE_IMAGE_DIR,
                   help=f"Image output directory (default: {REFERENCE_IMAGE_DIR})")
    p.add_argument("--manifest", type=Path, default=REFERENCE_MANIFEST,
                   help=f"Manifest to write (default: {REFERENCE_MANIFEST})")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be extracted; write nothing")
    args = p.parse_args()

    if not args.pdf.is_file():
        raise SystemExit(f"No such PDF: {args.pdf}")

    manifest = extract(args.pdf, args.out, DEFAULT_SOURCE, args.dry_run)
    total = sum(len(v) for v in manifest["images"].values())

    if args.dry_run:
        logger.info("Dry run: %d photographs for %d taxa, nothing written",
                    total, len(manifest["images"]))
        return

    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Wrote %d photographs for %d taxa to %s", total,
                len(manifest["images"]), args.out)
    logger.info("Wrote manifest %s", args.manifest)


if __name__ == "__main__":
    main()
