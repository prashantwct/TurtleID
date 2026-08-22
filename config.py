"""
Central configuration for the Chelonid-ID tool.

Every threshold that affects a determination lives here, not scattered through
the code. Anything you change here should be recorded in the deployment log,
because it changes what the tool reports to a Range Officer.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RECORD_DIR = BASE_DIR / "records"
LOG_DIR = BASE_DIR / "logs"

SPECIES_DB_PATH = DATA_DIR / "species_db.json"
CLASSIFIER_PATH = MODEL_DIR / "chelonid_cls.pt"      # YOLOv8-cls weights
DETECTOR_PATH = MODEL_DIR / "chelonid_det.pt"        # YOLOv8 detection weights (optional)
CALIBRATION_PATH = MODEL_DIR / "calibration.json"    # temperature + energy threshold

for _d in (MODEL_DIR, RECORD_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- image handling
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
CLS_IMAGE_SIZE = 320          # YOLOv8-cls input edge
DET_CONF_THRESHOLD = 0.35     # minimum detection confidence to accept a crop
DET_CROP_PADDING = 0.08       # fractional padding around the detected box

# ---------------------------------------------------------------- confidence policy
#
# Determination tiers. These map a posterior probability onto an action, and
# they are deliberately conservative: a wrong species on a seizure memo is a
# far more expensive error than an "expert referral" that turns out easy.
#
TIER_CONFIRMED = 0.85    # report species; still recommend a voucher photograph
TIER_PROBABLE = 0.60     # report species as probable; check the discriminator
TIER_TENTATIVE = 0.45    # report genus/family; run the morphological key
# below TIER_TENTATIVE -> INDETERMINATE, expert referral

# Normalised Shannon entropy above which the output is treated as ambiguous
# regardless of the top-1 value.
MAX_NORMALISED_ENTROPY = 0.55

# Free-energy out-of-distribution score. Above this the image probably does not
# contain an Indian chelonian at all (wrong animal, unusable photograph, a
# species not in the training set). Calibrated on a held-out negative set.
DEFAULT_ENERGY_THRESHOLD = -4.0

# Fallback gate used only when true logits cannot be captured from the model.
# Maximum softmax probability below this is treated as out of distribution.
# Weaker than free energy — an overconfident head can sail past it.
MSP_OOD_FLOOR = 0.30

# Default softmax temperature. Overwritten by calibration.json after temperature
# scaling on the validation split. 1.0 means uncalibrated -- the app warns.
DEFAULT_TEMPERATURE = 1.0

# ---------------------------------------------------------------- geographic prior
#
# Multipliers applied to the model's probability given the recorded location.
# The tool never rules a species out on geography alone -- an out-of-range
# animal is exactly what a trade seizure looks like -- so the floor is non-zero.
#
GEO_PRIOR = {
    "resident": 1.00,
    "introduced": 0.90,
    "marginal": 0.35,
    "absent": 0.06,       # low, not zero: seizures and releases happen
    "unknown": 0.60,
}

# When the location is not supplied at all, no prior is applied.
GEO_PRIOR_NEUTRAL = 1.0

# ---------------------------------------------------------------- state centroids
# Used to draw the offline distribution map. Approximate geographic centres,
# sufficient for a presence/absence bubble map; not for any spatial analysis.
STATE_CENTROIDS = {
    "Andhra Pradesh": (15.91, 79.74), "Arunachal Pradesh": (28.22, 94.73),
    "Assam": (26.20, 92.94), "Bihar": (25.10, 85.31),
    "Chhattisgarh": (21.28, 81.87), "Goa": (15.30, 74.12),
    "Gujarat": (22.26, 71.19), "Haryana": (29.06, 76.09),
    "Himachal Pradesh": (31.10, 77.17), "Jharkhand": (23.61, 85.28),
    "Karnataka": (15.32, 75.71), "Kerala": (10.85, 76.27),
    "Madhya Pradesh": (22.97, 78.66), "Maharashtra": (19.75, 75.71),
    "Manipur": (24.66, 93.91), "Meghalaya": (25.47, 91.37),
    "Mizoram": (23.16, 92.94), "Nagaland": (26.16, 94.56),
    "Odisha": (20.95, 85.10), "Punjab": (31.15, 75.34),
    "Rajasthan": (27.02, 74.22), "Sikkim": (27.53, 88.51),
    "Tamil Nadu": (11.13, 78.66), "Telangana": (18.11, 79.02),
    "Tripura": (23.94, 91.99), "Uttar Pradesh": (26.85, 80.95),
    "Uttarakhand": (30.07, 79.09), "West Bengal": (22.99, 87.86),
    "Delhi": (28.70, 77.10), "Andaman & Nicobar Islands": (11.74, 92.66),
}

INDIAN_STATES = sorted(STATE_CENTROIDS.keys())

# ---------------------------------------------------------------- external links
IUCN_SEARCH = "https://www.iucnredlist.org/search?query={name}&searchType=species"
GBIF_SEARCH = "https://www.gbif.org/species/search?q={name}"
TFTSG_CHECKLIST = "https://iucn-tftsg.org/checklist/"
INDIA_BIODIVERSITY = "https://indiabiodiversity.org/species/search?q={name}"

# ---------------------------------------------------------------- record keeping
RECORD_FILE = RECORD_DIR / "determinations.jsonl"
APP_LOG_FILE = LOG_DIR / "chelonid.log"
APP_VERSION = "1.0.0"
