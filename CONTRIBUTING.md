# Contributing

This tool exists to help forest department staff identify turtles and
tortoises correctly, including in casework where the answer determines an
offence category. Contributions are welcome from anyone — field staff,
researchers, veterinarians, rescue centres, enforcement officers.

Please read the locality section before submitting photographs.

---

## Locality data: read this first

**Never submit coordinates, and check your camera.**

Photographs of threatened species carry GPS coordinates in EXIF by default on
most phones. A coordinate for a *Batagur kachuga* nesting bank in the Chambal,
or an *Indotestudo elongata* in a tiger reserve, is directly useful to a
poacher. Several species covered here are targeted specifically because they
are locatable.

What this repository does about it:

- EXIF is stripped on intake. The image is re-encoded from pixel data, so
  maker notes and embedded thumbnails go too, not just the GPS tags.
- Coordinates typed into notes fields are redacted, and you are told.
- Nothing finer than a **state** is stored anywhere.
- `contributions/`, `records/` and `logs/` are gitignored and must stay that
  way. Never `git add -f` them.

The model does not need a coordinate to learn a species. If you have precise
locality data worth preserving, it belongs in your institutional records or
with TSA-India, not in a public repository.

---

## What to contribute

### 1. Photographs — the highest-value contribution

The model does not exist yet, and the only reason is images. Everything else
is built.

**Most needed, in order:**

1. **Ventral (plastron) views.** Scarcest and most valuable. The
   *Pangshura tecta* / *P. smithii* split is a plastron character, and those
   two sit on different WPA schedules — a wrong call changes the offence.
2. **Rare species.** Under 30 images a class will have unusable recall. The
   Contribute tab shows the current gap list, or run
   `python -m scripts.review_contributions --coverage`.
3. **Seizure and rescue photographs.** Real casework conditions — bad light,
   buckets, tarpaulin, hands in frame. These match deployment far better than
   clean field photographs do.
4. **Juveniles.** Most references illustrate adults. Juvenile *Nilssonia*
   ocelli and *Batagur dhongoka* stripes fade with age, so a model trained
   only on adults fails on exactly the animals most often trafficked.
5. **Negatives** — monitor lizards, frogs, crabs, empty riverbank, blurred
   frames. Without these the model confidently names a species in photographs
   containing no turtle.

**Capture protocol:** dorsal, ventral, lateral profile, head close-up, scale.
Even shade, no flash. Direct sun blows out the pale rays on a star tortoise
and turns a coral plastron white.

**Unsure of the species?** Submit anyway and mark it unidentified. An
unidentified photograph of a real animal in real conditions is worth more than
a confident wrong label, which actively damages the model.

**Licensing:** by submitting you confirm you took the photograph or have the
right to contribute it, and that it may be used to train this model and be
redistributed as part of the training set. Say so if either is restricted.

### 2. Corrections to species records

A diagnostic character that misleads in practice, a distribution that does not
match a published source, a status that has been reassessed.

**A citation is required.** Not bureaucracy: every line in
`data/species_db.json` is traceable to a published source, the WPA schedule
shown determines the offence category, and the distribution list feeds the
geographic prior that changes what the model reports. A paper, handbook,
gazette notification, or an institutional record with an accession number.

Personal field observation is genuinely valuable and often more current than
the literature — please submit it as a **photograph**, where it carries weight
without needing a citation.

### 3. Field determinations

Just use the tool. Every determination is logged locally, and the ones it
could not resolve are the images most worth labelling next.

---

## How to submit

**Through the app** (easiest, and the EXIF stripping is automatic):
Contribute tab → choose type → submit. Records land in `contributions/` on
your machine. Send that folder to the maintainer.

**Through GitHub:** open an issue using one of the templates. Do not attach
photographs with intact metadata to a public issue.

**Directly:** for bulk image contributions, contact the maintainer rather than
opening a pull request with hundreds of files.

---

## Pull requests

Small, focused, one concern each.

```bash
python -m scripts.validate_db     # must pass
python -m pytest test_smoke.py -q # must pass
```

CI runs both on every PR.

Changing `data/species_db.json` also means adding a row to `MATRIX` in
`core/morphkey.py` if it is a new species — the validator enforces this.

**Changes to `wpa_2022` or `cites` need a primary source.** Verify against the
Gazette notification or the CITES appendices directly, not a secondary
compilation. If you are correcting one of these, say in the PR which document
you checked.

---

## What gets rejected

- Photographs with locality data attached in any form
- Species record changes without a citation
- Anything that removes an abstention path or raises a confidence threshold so
  the tool reports a species more often. The tiering in `core/inference.py` is
  conservative on purpose: a wrong species on a seizure memo costs a case,
  while an expert referral costs a phone call.
- Automated apply paths for contributed data. A human reads every proposal.
- Removing the uncalibrated-model warning

---

## Setting up

```bash
git clone <repo> && cd chelonid-id
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m scripts.validate_db
streamlit run app.py
```

Optional, for IUCN sync:

```bash
export IUCN_API_TOKEN="your-token"   # request at https://api.iucnredlist.org/users/sign_up
python -m scripts.sync_iucn
```

The token goes in the environment or a `.env` file. `.env` is gitignored.
Never commit it — IUCN revokes tokens found in public repositories.

---

## Sources and attribution

Legal status follows the Wild Life (Protection) Amendment Act, 2022, as
compiled in the ZSI *Fauna of India Checklist: Reptilia* v1.0
(Mohapatra et al. 2024), cross-checked against the TRAFFIC / TSA-India /
WWF-India *Identification Cards: Tortoises and Freshwater Turtles of India*
(Singh, Badola & Fernandes 2023) — the reference issued to Indian enforcement
agencies. The schedules agree in full.

The TRAFFIC cards are copyright TSA-India and TRAFFIC, and reproduction
requires publisher permission. Status codes and state lists are transcribed as
factual data in `data/traffic_2023.json` with attribution; all diagnostic text
in this repository is written independently. **Do not paste text from that
publication, or from any handbook, into this database.** Write the character in
your own words and cite the source.

Taxonomy follows Rhodin et al. (2021), *Turtles of the World*, 9th edition.

---

## Conduct

Assume competence. A Range Officer who has handled more flapshells than most
herpetologists have seen may not write in the register of a journal, and a
correction phrased informally is still a correction. Engage with the substance.

Disagreements about a character are resolved by going back to the specimen and
the literature, not by seniority.
