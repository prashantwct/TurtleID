# Quick Start — Running Chelonid-ID

## Installation (first time only)

### 1. Get the code

Download `chelonid-id.bundle` and clone it:
```bash
git clone chelonid-id.bundle chelonid-id
cd chelonid-id
```

Or unzip `chelonid-id.zip`:
```bash
unzip chelonid-id.zip
cd chelonid-id
```

### 2. Install Python and dependencies

**Windows:**
- Download Python 3.10+ from python.org (tick "Add to PATH" during install)
- Open Command Prompt and run:
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt
  ```

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you get `ModuleNotFoundError: No module named pip`, install it first:
```bash
python -m ensurepip --upgrade
```

### 3. Start the app

From inside the `chelonid-id` folder with the virtual environment activated:

```bash
streamlit run app.py
```

Your browser will open to `http://localhost:8501` automatically. If it doesn't, copy that URL into your browser.

---

## What works today (without a trained model)

### Morphological key
Select any characters you can observe on the animal (carapace, plastron, head,
feet) in any order. The tool ranks species by how many characters they match.
Contradicted taxa are ranked down but never hidden — field characters get
misread and silently deleting the right answer is worse than ambiguity.

Completely functional from day one.

### Species reference
Browse 30 taxa by family, IUCN status, state occurrence. Read the diagnostics,
key character, habitat, maximum size, Madhya Pradesh notes, confusion pairs with
the discriminator, and citations. State-level distribution shown as a map.

Fully functional.

### Determination log
Every determination is recorded with the observer's name, location, timestamp,
and the model's confidence/entropy/energy if the photograph tab was used. Shows
how many determinations have been made and how many were ambiguous — the
unresolved cases are the images most worth labelling for model training.

Functional. Can download as JSONL for data analysis.

---

## Enabling the photograph tab (requires trained model)

The photo identification tab is disabled until a trained YOLOv8 classifier is
installed. Here's how to train one:

### 1. Collect images

Organize them by species id (folder names must match `data/species_db.json`):

```
dataset/
  train/
    lissemys_punctata/
      photo1.jpg
      photo2.jpg
      ...
    pangshura_tecta/
      ...
  val/
    lissemys_punctata/
      photo1.jpg
      ...
    pangshura_tecta/
      ...

negatives/
  monitor_lizard1.jpg
  frog1.jpg
  empty_river.jpg
  ...
```

**Critical rules:**
- Split by **animal**, not photograph. If you have 10 photos of one basking
  turtle, put them all in train OR all in val, not split.
- Include plastron and carapace views — the plastron separates *Pangshura
  tecta* (Schedule I, coral-red) from *P. smithii* (Schedule II, dark-blotched).
- Negatives are not optional. Monitor lizards, frogs, crabs, hands, buckets,
  empty habitat, blurred frames — this set stops the model confidently naming a
  species in a photo with no turtle.
- Classes with under 30 images will have poor recall. Don't discard data to
  balance the set; let the model abstain on rare species.

### 2. Train the classifier

With your virtual environment activated and inside the `chelonid-id` folder:

```bash
python -m training.train_classifier --data ./dataset --epochs 120
```

This will take 30–60 minutes on CPU, less on GPU. Best weights are saved to
`models/chelonid_cls.pt`.

### 3. Calibrate (required before field use)

```bash
python -m training.calibrate --data ./dataset --negatives ./negatives
```

This does two critical things:
- **Temperature scaling:** Fits one scalar so reported percentages mean
  something. An 80% determination should be right ~80% of the time.
- **Out-of-distribution threshold:** Tests the model on negatives and sets a
  gate that rejects non-chelonian images.

Prints calibration quality metrics (Expected Calibration Error before/after,
percentage of negatives rejected). If ECE stays above 0.10 or less than 70% of
negatives are rejected, add more negatives and retrain.

Writes `models/calibration.json`.

### 4. Restart the app

```bash
streamlit run app.py
```

The photograph tab is now enabled. Any image it cannot identify is recorded in
`records/determinations.jsonl` — these are the ones worth labelling next.

---

## Field workflow

### With model:
1. Take 5 photographs (dorsal, ventral, lateral, head close-up, scale).
2. Upload the best one.
3. Record observer name and location notes.
4. Select the state where the animal was found (if known).
5. Read the tier (CONFIRMED / PROBABLE / TENTATIVE / INDETERMINATE / REJECTED).
6. If it says CONFIRMED or PROBABLE, check the listed discriminator against
   the live animal.
7. Record the species.

### Without model (or when the model says INDETERMINATE):
1. Go to **Morphological key**.
2. Record whichever characters you can observe.
3. The key ranks candidates. Select one and check the full species card.
4. Record the species.

---

## Troubleshooting

### "No module named streamlit"
The virtual environment is not activated. Run `source .venv/bin/activate`
(Mac/Linux) or `.venv\Scripts\activate` (Windows).

### "ModuleNotFoundError: No module named ultralytics"
Run `pip install -r requirements.txt` again, or just `pip install ultralytics`.

### "No trained model is installed"
This is expected on first run. The photograph tab is disabled. Use the
morphological key and species reference instead, or collect images and train
a model (above).

### "Model is UNCALIBRATED"
You trained a model but skipped `training/calibrate.py`. Run that step before
field deployment — percentages are unreliable without it.

### The app won't start / crashes on startup
Check the terminal for an error message. Common causes:
- Python version is too old (need 3.10+): run `python --version`
- `data/species_db.json` is corrupted: run `python -c "import json; json.load(open('data/species_db.json'))"`
- A required package didn't install: run `pip install -r requirements.txt` again

### Slow on CPU
This is expected. 320-px images on CPU take ~200 ms. Inference is not the
bottleneck in the field — your bandwidth for observing the animal is. If
you have a GPU, install the GPU version of PyTorch (`pip install torch
torchvision -f https://download.pytorch.org/whl/torch_stable.html`) and it
will be ~10× faster.

---

## Data privacy

- **Determination logs** (`records/determinations.jsonl`) are saved locally,
  never uploaded. They contain image hashes, observer names, and location
  notes. Guard them — locality data on CR species is poaching-relevant.
- **The species database** (`data/species_db.json`) is compiled from published
  sources; distribution maps are state-level, not GPS coordinates.
- **No telemetry or cloud upload** — the app is fully offline. It works without
  internet once it starts.

---

## Next steps

- Read `README.md` for the technical architecture and how the model reaches
  determinations.
- Read `PUBLISHING.md` if you plan to share this with other forest departments.
- See `core/morphkey.py` to understand the multi-access key logic, or
  `data/species_db.json` to modify species records.
- Run `pytest test_smoke.py` to verify the installation is correct.
