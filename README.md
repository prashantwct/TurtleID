# Chelonid-ID

Field identification of Indian turtles and tortoises for forest department use.
YOLOv8 image classification, backed by a reference database of 30 taxa compiled
from published literature, IUCN assessments and standard handbooks, with a
morphological key that works without the model.

Covers the non-marine chelonians of India — Geoemydidae (16), Trionychidae (8),
Testudinidae (5) — plus the exotic Red-eared Slider, which is included
deliberately because confusing it with a native Schedule I species is the most
common identification error in Indian enforcement casework. Marine turtles are
out of scope.

---

## Read this before deploying

**The model does not exist yet.** This repository ships the reference database,
the identification logic, the morphological key and the training pipeline. It
does not ship trained weights, because no adequately labelled Indian chelonian
image set exists that I can package for you. Everything except the photograph
tab works on first run.

That is not a gap to paper over. A tool that gives a Range Officer a species
name and a percentage, on a model trained on 40 scraped images, is worse than
no tool: it converts an honest "I don't know" into a confident wrong entry on a
seizure memo. The abstention machinery in `core/inference.py` exists to make
the tool say "I don't know" more often than a naive classifier would.

Order of work:

1. Deploy today with the **morphological key** and **species reference**. These
   are useful immediately and need nothing but Python.
2. Collect images (below). This is the long pole — budget months, not weeks.
3. Build a **matching gallery** as soon as the first photographs exist. No
   training, minutes to run, and it improves every time a photograph is added.
4. Train a classifier once there are dozens of photographs of *different animals*
   per species, calibrate it, and it takes over the photograph tab.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Python 3.10+. Runs on a field laptop without a GPU; inference on CPU is
around 200 ms per image at 320 px.

### Deploying on Streamlit Community Cloud

Community Cloud chooses the interpreter, not the repository — the version is
set per app under **Advanced settings → Python version**, and an app left on a
version that falls out of support is moved forward without warning. That is a
dependency problem, because numpy, pandas and pillow are compiled packages: a
pin whose wheels predate the interpreter is built from source on the runner,
and the runner has no zlib headers, so the deploy dies during
`pip install` and the app never leaves "Your app is in the oven".

`requirements.txt` therefore splits those three pins on `python_version`. Both
branches install from wheels on 3.10 through 3.14. Pin a fourth package there
only after checking a wheel exists for the Python the app is deployed on.

If a deploy hangs, read the log from the top: the first compiled package that
starts "Building wheel" is the failure, whatever error appears later.

---

## How a determination is produced

```
photograph
  └─ detector (optional)     locate and crop the animal
      └─ classifier          YOLOv8-cls → raw logits (captured by forward hook)
          └─ temperature     scalar fitted on validation; makes probabilities mean something
              └─ OOD gate    free energy; rejects non-chelonians and untrained species
                  └─ geography  weak prior from the recorded state
                      └─ entropy   spread across candidates forces a downgrade
                          └─ TIER
```

### Why not just report the softmax score

A cross-entropy-trained classification head is systematically overconfident,
and on a species it has never seen it will still return 0.97 for something. The
raw number is not a probability of being right. Three corrections are applied:

**Temperature scaling** (Guo et al. 2017). One scalar, fitted on the validation
split. Does not change which class wins, so accuracy is untouched, but it makes
the reported figure calibrated: of determinations reported at 80%, roughly 80%
should be correct. `training/calibrate.py` prints Expected Calibration Error
before and after so you can verify it worked rather than assume it.

**Free-energy out-of-distribution gate** (Liu et al. 2020). Softmax cannot tell
you "none of the above" — it always sums to 1. Free energy can. This is what
rejects a photograph of a monitor lizard, an empty bucket, or a chelonian
species outside the training set.

> Implementation note worth knowing if you modify this: Ultralytics exposes only
> post-softmax probabilities. Softmax is shift-invariant, so `log(p)` always
> gives `logsumexp == 0` and free energy collapses to a constant that rejects
> nothing. `core/inference.py` registers a forward hook on the classification
> head to capture true logits. `training/calibrate.py` scores through the same
> method for exactly this reason — fitting the threshold on one scale and
> applying it on another gives a gate that silently misfires.

**Geographic prior.** The recorded state multiplies each candidate: resident
1.00, introduced 0.90, marginal 0.35, absent 0.06. The floor is deliberately
non-zero. An out-of-range animal is not an error to be suppressed — it is what
a trade seizure looks like, and the tool flags it as a trade record rather than
hiding it.

### Determination tiers

| Tier | Posterior | What it means |
|---|---|---|
| `CONFIRMED` | ≥ 0.85 | Record the species; retain voucher photographs |
| `PROBABLE` | ≥ 0.60 | Verify the named discriminating character first |
| `TENTATIVE` | ≥ 0.45 | Genus or family only; run the morphological key |
| `INDETERMINATE` | < 0.45 | Do not enter a species on any form; refer to a specialist |
| `REJECTED` | — | Not a species this model knows; retake or use the key |

Normalised entropy above 0.55 forces a downgrade regardless of the top-1 value,
so a model that is 0.5/0.45 between two species cannot report `PROBABLE`.

Thresholds live in `config.py`. Changing them changes what the tool tells a
Range Officer — log any change.

### Automatic flags

- **Legal divergence.** If the top two candidates sit on different WPA
  schedules, this is stated explicitly. *Pangshura tecta* is Schedule I,
  *P. smithii* is Schedule II — the offence category differs, and the two are
  separated by plastron colour and one scute.
- **Out of range.** Reported as a trade, transport or release record, not a
  wild occurrence.
- **Threatened species.** CR/EN/VU determinations prompt DFO notification.
- **Confusion pair.** The published discriminating character is surfaced from
  the database for the user to check on the animal.

---

## Building the training set

The dataset is the project. The architecture is the easy part.

### Layout

```
dataset/
  train/<species_id>/*.jpg      folder names must match ids in data/species_db.json
  val/<species_id>/*.jpg
negatives/*.jpg                 non-chelonian images; see below
```

### Rules that decide whether this works

**Split by animal, not by photograph.** Ten frames of the same basking
*Pangshura* from one sitting, spread across train and val, give you a
validation accuracy that evaporates in the field. If you have capture ids, split
on them.

**Negatives are not optional.** A few hundred images of monitor lizards, frogs,
crabs, empty riverbank, hands, buckets, tarpaulin, and out-of-focus frames. This
set is what stops the model confidently naming a species in a photograph that
contains no turtle. Without it, the OOD gate is untuned and untested.

**Expect brutal imbalance and do not fix it by discarding data.** You will have
hundreds of *Lissemys punctata* and single figures of *Batagur kachuga*. Judge
the model on per-class recall, never on overall accuracy — a model that predicts
"flapshell" for everything scores well on accuracy and is useless.

**Rare species should abstain, not guess.** Under ~30 images, a class will have
unusable recall. Leaving it in with conservative thresholds is correct: the tool
reports `INDETERMINATE`, which is the true answer.

**Photograph both surfaces.** A carapace-only dataset cannot learn the
tecta/smithii split, because that character is on the plastron.

### Sources

Verified specimen photographs from MPFD rescue and seizure records, TSA-India,
the National Chambal Sanctuary, and research collections. Public aggregators
(GBIF, iNaturalist research-grade, India Biodiversity Portal) are usable but
must be re-verified against the key — misidentified *Pangshura* are common in
citizen-science data, and training on them bakes the error in. Check licences
before redistribution.

### Seeding from published identification cards

`training/extract_id_cards.py` pulls the species photographs out of the TRAFFIC
India / TSA-India / WWF-India *Identification Cards: Tortoises and Freshwater
Turtles of India* (2023) — the same document `data/traffic_2023.json` is
reconciled against.

```bash
python -m training.extract_id_cards --pdf "path/to/ID Cards 2023.pdf" --dry-run
python -m training.extract_id_cards --pdf "path/to/ID Cards 2023.pdf"
```

It takes one photograph per species card, resolving the printed name through
the crosswalk in `data/traffic_2023.json` — the 2023 edition prints *Amyda
cartilaginea* for what this database calls `amyda_ornata`. It leaves the
distribution maps alone: the app draws its own from `states`, and the printed
map on the *Lissemys punctata* card is legended *Chitra indica*, which is an
error in the source.

The 2023 edition yields **28 photographs covering 28 of the 30 taxa** — one each,
missing only *Manouria impressa* and the Red-eared Slider.

**One image per species is a reference library, not a training set.** Feeding it
to the classifier gives every class a single capture, which means no validation
split, which means `calibrate.py` cannot fit a temperature and the abstention
machinery does not function. `prepare_dataset.py` says so in as many words and
does not pretend otherwise:

```bash
python -m training.prepare_dataset --pool ./pool --out ./dataset --seed-with-reference-plates
```

The plates are worth having for what they are — a known-good photograph beside
each species in the reference tab, which is useful in the field today — and as
the thing field photographs accumulate against.

**On copyright.** The photographs are tracked, and so is
`data/reference_images.json`, which records the source, page, photographer
credit and rights for every one of them. They remain the property of the
publishers and of the photographers credited on each card, and the app prints
that credit under every plate. Including them here is not a licence to reuse
them: anyone wanting to do that needs the publishers' permission. If the
arrangement ever needs unwinding, `reference_images/` can go back into
`.gitignore` and nothing else breaks — the app already treats an absent file as
normal. See PUBLISHING.md.

### Promoting contributed photographs

Images submitted through the Contribute tab land in `contributions/`. That is
where they stop — being contributed does not put an image in front of the
model. This carries them across:

```bash
python -m training.promote_contributions --list
python -m training.promote_contributions --accept 4f2a9c1e
python -m training.promote_contributions --accept-all-confident
```

Two kinds of contribution are held back rather than promoted: those submitted
as anything other than `confident`, and those with no species id. Both need a
determination first. A photograph filed under the wrong species does not
announce itself — it becomes a class the model learns wrongly, surfacing later
as a confident misidentification with nothing pointing back at the cause.

Capture ids are derived from contributor and submission date, which errs toward
grouping: two animals merged into one capture costs a little split flexibility,
whereas one animal split across two captures puts the same individual in train
and val. Use `--capture-suffix` when one contributor really did send two
animals on one day.

`contributions/promoted.json` records what has already moved, so re-running
promotes only what is new. The contribution log itself is append-only and is
never rewritten.

### Durable contribution storage

`contributions/` is written to the disk of whatever machine runs the app. On a
field laptop that persists. On Streamlit Community Cloud it does not — every
reboot and every push to the deployed branch re-clones the repository, and
anything submitted since is gone. An app that accepts a photograph, thanks the
contributor and then loses it is worse than one with no Contribute tab, because
the contributor believes the job is done.

So when object storage is configured, every submission is written there too,
and a photograph that cannot be stored durably is **refused** rather than
accepted into a directory that will not survive the week. The sidebar states
which mode is in force.

There are two backends. **GitHub** commits each submission to a repository, so
review runs through the same history and pull requests as everything else:

```
CHELONID_GITHUB_REPO     required; "owner/repo". Setting it switches this on
CHELONID_GITHUB_TOKEN    required; fine-grained, Contents:write, that repo only
CHELONID_GITHUB_BRANCH   optional, defaults to main
CHELONID_GITHUB_PATH     optional, defaults to submissions/
```

> **A public repository publishes every contribution, permanently.** The
> photograph, the contributor's name and the state become world-readable the
> moment they are submitted, and a commit cannot be withdrawn — deleting a file
> later leaves it in the history and does nothing about copies already taken.
> EXIF stripping and note redaction still run, and neither can do anything about
> locality *visible in the frame*: a recognisable bank, a signboard, a number
> plate. For a Schedule I species that is the disclosure the rest of this
> codebase exists to prevent.
>
> The app checks the repository's visibility and, when it is public or cannot be
> determined, refuses to submit without an explicit acknowledgement from the
> contributor. **A private repository gives the identical workflow with none of
> the exposure**, and is the right default unless publishing is a decision you
> have taken deliberately.

Submissions land as one file each, so simultaneous contributors never collide:

```
submissions/images/<species_id>_<digest>.jpg
submissions/records/<record_id>.json
```

Commit messages carry the record id and the species only — never the
contributor, the state, or any note.

**Object storage** is the other backend, and the one that publishes nothing. Any
S3-compatible service works — AWS S3, Cloudflare R2, Backblaze B2, Wasabi,
MinIO. Set these in the deployment's secrets, in the environment, or in a
gitignored `.env`:

```
CHELONID_S3_BUCKET       required; setting it switches durable mode on
CHELONID_S3_ACCESS_KEY   required
CHELONID_S3_SECRET_KEY   required
CHELONID_S3_ENDPOINT     required for anything that is not AWS S3
CHELONID_S3_REGION       optional, defaults to us-east-1
CHELONID_S3_PREFIX       optional key prefix
```

Check it without submitting a photograph:

```bash
python -m scripts.check_github_storage
```

It asks the four questions that separate every failure this backend produces —
is the token one GitHub recognises, can it see the repository, may it push, does
the branch exist — and writes nothing. The token is never printed, only its
length and prefix, which is what tells a truncated paste from a revoked token.

The two failures worth telling apart: **401** means the token value itself is
wrong (revoked, expired, or incompletely copied) and permissions are not
involved; **403** means it authenticates but lacks Contents:write. Secrets are
read once at startup, so reboot after changing one.

If both backends are configured, GitHub wins and the app says so in the log.

A bucket named without credentials is treated as a misconfiguration and
reported, not quietly ignored — that mistake otherwise looks identical to
working correctly, right up until the first restart.

Then take up what has been submitted and promote it:

```bash
git pull                                        # GitHub backend: this IS the download
python -m training.pull_contributions --list
python -m training.pull_contributions
python -m training.promote_contributions --list
python -m training.promote_contributions --accept <id>
python -m training.build_gallery --publish      # fold them into the gallery
```

With the GitHub backend there is nothing to fetch — `git pull` has already
brought every submission into the working tree, and `pull_contributions` copies
them out of `submissions/` into the local layout the rest of the pipeline reads.
It picks that route automatically when GitHub is configured or a `submissions/`
directory is present; `--from-repo` forces it, `--submissions` points it
elsewhere. With object storage it downloads, as before.

Pulling is idempotent: records already present locally are left alone and their
photographs are not re-downloaded. A record whose photograph is missing from
the bucket is reported rather than skipped, because that combination means a
submission was accepted and its image lost.

### Dropping photographs into folders

If naming a capture per animal on the command line is more bookkeeping than the
job deserves, arrange the photographs in a file manager instead and let the
importer do the filing:

```bash
python -m training.import_folders --setup      # one folder per species
python -m training.import_folders --dry-run    # what would be filed
python -m training.import_folders              # file it
```

`--setup` writes a folder per species under `incoming/`, plus a
`WHICH-FOLDER.txt` naming each one in English. Drop photographs in, **one
folder per animal**:

```
incoming/
  lissemys_punctata/
    chambal-aug-19/        <- one animal
      IMG_0431.jpg
      IMG_0432.jpg
    chambal-aug-20/        <- a different animal
      IMG_0455.jpg
  Indian Roofed Turtle/    <- common and scientific names work too
    rescue-crate-3/
      a.jpg
```

The animal folder is the whole point of the layout: it is what tells
`prepare_dataset.py` which photographs are of the same individual, and it is
the difference between a validation number you can act on and one that measures
whether the model recognises one turtle it has already seen. A photograph left
loose in a species folder is filed as its own animal — right when every loose
photograph is a different individual, wrong the moment two are not, so the
count is reported on every run rather than assumed.

A species folder whose name the database does not recognise is reported and
skipped, with the near misses listed. Nothing is guessed at.

Filing goes through the same EXIF scrubber as everything else, and re-running
is safe: captures already in the pool are left alone, so adding another animal
folder and running again picks up only what is new. `--reimport` replaces
captures that are already there, for when you have corrected a folder's
contents. `incoming/` keeps your originals and is gitignored alongside `pool/`;
the privacy CI job fails if either is ever tracked.

HEIC files are reported rather than silently dropped — Pillow cannot decode
them without a plugin that is not in `requirements.txt`, so convert them to
JPEG first (`sips -s format jpeg *.heic --out jpgs/` on macOS,
`magick mogrify -format jpg *.heic` elsewhere).

### Filing field photographs as they arrive

```bash
python -m training.ingest_field_images --species lissemys_punctata \
    --capture chambal-2026-08-19 ~/rescue/*.jpg
```

**Once per animal, not once per batch.** Everything in one invocation becomes
one capture, and captures move between splits whole. Two animals filed under a
single capture id can never be separated again, and the validation split
quietly stops meaning anything from that point on.

Every file is rewritten through the same EXIF scrubber the Contribute tab uses,
and the count that arrived carrying GPS is reported — worth passing back to
whoever sent them, because their camera is embedding locality on every
photograph they take, not just these. `pool/` is gitignored and the privacy CI
job fails if it is ever tracked.

Species ids are checked against the database on the way in. A photograph filed
under a name the database does not know becomes a class the model can emit and
the app cannot resolve.

### Building the matching gallery (no training)

```bash
python -m training.build_gallery --seed-with-reference-plates
python -m training.build_gallery --negatives ./negatives
```

This is the path that works on the day the first photographs arrive. Every
photograph in `pool/` is passed once through an ImageNet-pretrained backbone and
stored as a vector; a new photograph is identified by finding its nearest
neighbours and reading off which species they belong to. There is no training
step, no epochs and no train/validation split, and it runs on a laptop CPU in
minutes. Adding photographs means running it again — the gallery *is* the model.

**What replaces the validation split** is leave-one-capture-out. Every
photograph is scored against the gallery with every photograph of its own animal
removed, which answers the question that matters: would this have been
identified from *other* animals of the species? That held-out score is what fits
the temperature and the similarity floor, so the same guarantees hold as on the
trained path — a reported 80% should be right about 80% of the time, and a
photograph resembling nothing in the gallery is rejected rather than named.

#### Cropping to the animal first

```bash
python -m training.build_gallery --detector models/chelonid_det.pt --publish
```

The camera-trap pipeline, and what Addax AI / EcoAssist does: a class-agnostic
detector finds the animal, everything else is discarded, and only the crop is
embedded. It is the obvious answer to the measurement above — if what separates
a plate from a field photograph is paper, grass, hands and light, then removing
all of it should leave the animal to be matched on.

Nothing here ships a detector. COCO has no turtle class, so a stock YOLO is no
use; what this needs is an animal detector, and
[MegaDetector](https://github.com/agentmorris/MegaDetector) is the usual free
choice. Put the weights at `models/chelonid_det.pt`.

**Whatever is done to a gallery photograph must be done to a query.** A gallery
of crops searched with whole frames differs from every entry in exactly the way
cropping exists to remove, and is worse than either arrangement applied
consistently. `Gallery.cropped` records which it was and the app refuses the
mismatch rather than quietly serving it, so adding a detector after building a
gallery means rebuilding the gallery.

Whether it helps is a measurement, not a promise. The build prints held-out
accuracy against chance either way, and that is the number that answers it.

**The number that decides whether it works is held-out accuracy against
chance.** A gallery can be perfectly calibrated and know nothing: fit a
temperature to scores carrying no signal and it reports 1/n for everything,
with an excellent calibration error. So a gallery that did not beat chance is
recorded as unreliable, and the photograph tab refuses to use it — it would
otherwise return a species for anything, with a confidence figure attached,
and that figure would go on a form.

The first real gallery here measured **0% against 4% for guessing**. The cause
is worth knowing because it is not a shortage of photographs: generic image
features separate a scanned reference plate from a phone photograph taken in
the field far more strongly than they separate one species from another. Plates
sit at 0.56 mean similarity to each other, field photographs at 0.51 to each
other, and the two groups at 0.42 across the gap — so a field photograph
matches other field photographs whatever animal is in them, and the plate it
should be matching is unreachable. Seeding a gallery with reference plates does
not help a tool whose queries are field photographs. What helps is field
photographs of each species, from more than one animal.

It reports honestly on its own limits. Species with only one animal in the
gallery are listed as unmeasurable: they can still be matched, and a single
reference plate is genuinely useful, but nothing can say how often they are
right. Per-class recall is printed, lowest first, because overall accuracy on a
set this imbalanced tells you almost nothing.

**What it cannot do.** ImageNet features separate a softshell from a tortoise
easily. They do not reliably separate *Pangshura tecta* from *P. smithii*, which
differ in plastron colour — coral-red versus dark-blotched, and the difference
between Schedule I and Schedule II. The app labels every gallery determination
as advisory and points at the morphological key, and the confusable-pair warning
still fires. Treat the gallery as a fast first opinion, not as the record.

The trained classifier below is strictly better once the photographs exist to
support it, and takes over automatically: when both `models/chelonid_cls.pt` and
`models/gallery.npz` are present, the classifier wins.

#### Getting the gallery onto a hosted deployment

`models/` is gitignored, and Streamlit Community Cloud re-clones the repository
on every restart — so a gallery built locally is invisible to the hosted app,
which reports *Identification: not installed* however many times you rebuild.

```bash
python -m training.build_gallery --publish
git add data/gallery.npz && git commit -m "Update the published gallery" && git push
```

`--publish` writes a second copy with the capture ids replaced by a constant.
They exist for leave-one-capture-out, which happens at build time; identification
never reads them, and they are the one field in the file that names places. The
vectors, species, fitted temperature and floor all carry over, so the deployed
app knows whether it is calibrated instead of having to assume the worst.

About 7 KB per photograph. The local gallery wins wherever both exist, so
publishing never shadows a fresher build on your own machine.

**Restart after publishing.** Streamlit caches the identifier and does not
re-import modules on a source change, so a deployment given a new gallery keeps
serving the one already in memory until the process restarts. The sidebar states
how many photographs the *loaded* gallery holds, which is how you tell the two
apart: if the count has not moved after a rebuild, the app has not restarted.

### Train and calibrate

The second path, and the better one once the photographs exist: dozens per
species, across different animals. Until then `prepare_dataset.py` will tell you
the dataset cannot be validated, and that assessment is not pessimism — a model
fitted to a handful of images reports confident numbers that mean nothing.

```bash
python -m training.train_classifier --data ./dataset --epochs 120
python -m training.calibrate --data ./dataset --negatives ./negatives
```

Calibration is not an optional polish step. Until it runs, the app displays an
uncalibrated warning on every determination, which is the honest state.

Augmentation is deliberately asymmetric: geometry is pushed hard (rotation,
scale, occlusion), colour is kept tight. Hue and saturation jitter destroys the
exact characters the tool depends on — coral-red versus dark-blotched plastron,
the red postorbital patch, yellow head spots.

---

## Photograph capture protocol

Print this for field staff. Most failed identifications are failed photographs.

1. **Dorsal** — carapace square-on from directly above, whole shell, no
   foreshortening.
2. **Ventral** — plastron flat and whole. This one frame separates Schedule I
   from Schedule II in *Pangshura*.
3. **Lateral profile** — at eye level. The only view showing whether the third
   vertebral scute is spined.
4. **Head close-up** — filling the frame, in shade, no flash. Head markings
   carry more diagnostic weight than shell colour.
5. **Scale** — ruler or a ten-rupee coin in the plane of the carapace.

Shoot in even shade. Direct sun blows out the pale rays on a star tortoise and
turns a coral plastron white.

---

## Records

Every determination is appended to `records/determinations.jsonl` before the
result is displayed, with an image hash, observer, location and the full audit
trail including model probabilities, the geographic multiplier applied, entropy
and energy.

Two purposes. If a determination is later challenged you need what the tool
actually said, not what someone remembers. And every `INDETERMINATE`,
`REJECTED` and `TENTATIVE` entry is a photograph the model needs — the Records
tab surfaces this queue. Those images are worth far more per image than another
hundred flapshells.

---

## Sources for the reference database

Legal status follows the Wild Life (Protection) Amendment Act, 2022, as
compiled in the ZSI *Fauna of India Checklist: Reptilia* v1.0
(Mohapatra et al. 2024), Table 4. Under that amendment all Indian chelonians
are Schedule I except three — *Melanochelys trijuga*, *Pangshura smithii* and
*Cyclemys gemeli* — which are Schedule II.

Taxonomy and conservation status follow Rhodin et al. (2021), *Turtles of the
World*, 9th edition (Chelonian Research Monographs 8) and current IUCN Red List
assessments. Diagnostic characters are drawn from Das (1995) *Turtles and
Tortoises of India*, Ahmed & Das (2010) *Turtles and Tortoises of Northeast
India*, the TFTSG *Conservation Biology of Freshwater Turtles and Tortoises*
species accounts, and Moll (1987) for *Pangshura*. Per-species citations and
links are in `data/species_db.json` and shown in the app.

**The schedule shown is for field triage.** For any FIR, seizure memo or court
submission, verify against the Gazette notification. This tool assists
identification; it does not make determinations of law.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version:

- **Photographs are the bottleneck.** Everything else is built. Ventral
  (plastron) views and rare species first.
- **Never submit coordinates.** EXIF is stripped on intake and coordinates in
  notes fields are redacted, but check your camera anyway — a coordinate for a
  *Batagur kachuga* nesting bank is a poaching risk by any route.
- **Species record changes need a citation.** Personal observation is valuable
  and should come in as a photograph, where it carries weight without one.
- The Contribute tab handles all of this; records land in `contributions/`,
  which is gitignored.

Maintainers: `python -m scripts.review_contributions --proposals --coverage`.
Nothing applies automatically — every proposal is edited in by hand and
validated, because this file determines the offence category shown to a Range
Officer.

## IUCN Red List integration

```bash
export IUCN_API_TOKEN="your-token"   # https://api.iucnredlist.org/users/sign_up
python -m scripts.sync_iucn          # fetch and cache
python -m scripts.sync_iucn --report # divergences, no network
```

Uses Red List **API v4** (Bearer auth). Caches to disk keyed by Red List
version, which is what IUCN's terms ask API users to do rather than re-query
between releases. Fully offline-tolerant: no token or no connectivity means the
app falls back to the curated database, which is authoritative here anyway.

The sync never writes to `data/species_db.json`. IUCN supplies conservation
status; it does not supply WPA schedules, diagnostic characters, or confusion
pairs. It prints divergences for a person to resolve.

## Reference data provenance

Legal status follows the Wild Life (Protection) Amendment Act, 2022 as compiled
in the ZSI *Fauna of India Checklist: Reptilia* v1.0 (Mohapatra et al. 2024),
cross-checked against the TRAFFIC / TSA-India / WWF-India *Identification
Cards: Tortoises and Freshwater Turtles of India* (Singh, Badola & Fernandes
2023) — the reference issued to Indian enforcement agencies.

**Schedules agree in full**, including the three Schedule II taxa
(*Melanochelys trijuga*, *Pangshura smithii*, *Cyclemys gemeli*). A test
enforces this.

The 2023 cards predate 15 IUCN reassessments, mostly upward — *Chitra indica*,
*Indotestudo elongata* and *Nilssonia leithii* are now CR on cards reading EN
or VU. Staff carrying printed cards will see the older category.
`python -m scripts.reconcile_traffic` prints the full comparison.

The cards are copyright TSA-India and TRAFFIC; reproduction requires publisher
permission. Status codes and state lists are transcribed as factual data in
`data/traffic_2023.json` with attribution. All diagnostic text here is written
independently.

## Files

```
app.py                       Streamlit interface
config.py                    all thresholds, paths, priors
data/species_db.json         the reference database — the authoritative content
core/database.py             loading with schema validation
core/inference.py            detection, calibration, OOD, geography, tiering
core/morphkey.py             multi-access morphological key
core/records.py              atomic append-only determination log
core/iucn.py                 Red List API v4 client, cached and offline-tolerant
core/contributions.py        contribution intake, EXIF stripping, coordinate scrubbing
data/traffic_2023.json       TRAFFIC 2023 crosswalk for reconciliation
core/matcher.py              gallery embedding and nearest-neighbour matching
training/import_folders.py   filing from a hand-arranged folder tree
training/build_gallery.py    gallery build, leave-one-capture-out fitting
training/train_classifier.py YOLOv8-cls training with a dataset audit
training/calibrate.py        temperature scaling and OOD threshold fitting
scripts/validate_db.py       schema and cross-reference validator (CI)
scripts/sync_iucn.py         IUCN sync and divergence report
scripts/reconcile_traffic.py reconciliation against the enforcement cards
scripts/review_contributions.py  maintainer review and image gap map
```

Adding a species means editing `data/species_db.json` and adding a row to
`MATRIX` in `core/morphkey.py`. The database is validated on load: missing
fields, unknown IUCN categories, absent citations and unresolvable
cross-references all fail loudly at startup rather than surfacing as a blank
panel in the field.
