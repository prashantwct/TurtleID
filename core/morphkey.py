"""
Multi-access morphological key.

This is not a fallback bolted on to satisfy a checklist. It is the part of the
tool that works on day one, before a single training image has been collected,
and it is what the model output has to be checked against afterwards.

A strict dichotomous key fails in the field because the first couplet often
asks for a character the observer cannot see: the animal is in a bucket, or
only the carapace is photographed, or it is a shell fragment from a poaching
camp. So this is a multi-access (synoptic) key. The user states whichever
characters they can actually observe, in any order, and candidates that
positively contradict an observation are eliminated. Characters recorded as
variable or unknown for a taxon never eliminate it.

Character states use the vocabulary a Range Officer would recognise, not
formal osteology.
"""

from __future__ import annotations

from dataclasses import dataclass

# ------------------------------------------------------------------ characters
# Each character: key -> (question, {state_code: label})

CHARACTERS: dict[str, tuple[str, dict[str, str]]] = {
    "shell_surface": (
        "Shell surface",
        {
            "hard": "Hard, divided into horny scutes with visible seams",
            "soft": "Soft and leathery, no scutes, skin-covered edge",
        },
    ),
    "limbs": (
        "Feet and limbs",
        {
            "webbed": "Webbed or paddle-like, with claws — aquatic",
            "columnar": "Columnar, club-shaped, elephantine — a tortoise",
        },
    ),
    "femoral_flaps": (
        "Hinged flaps on the rear plastron covering the withdrawn hindlimbs",
        {"yes": "Present", "no": "Absent"},
    ),
    "head_shape": (
        "Head proportion",
        {
            "needle": "Strikingly narrow and elongate, eyes tiny and near the snout tip",
            "normal": "Normally proportioned",
            "broad": "Broad and flattened, snout very short",
        },
    ),
    "keels": (
        "Longitudinal keels (ridges) on the carapace",
        {"none": "None or barely traceable", "one": "One (vertebral only)", "three": "Three"},
    ),
    "third_vertebral": (
        "Third vertebral scute in side profile",
        {
            "spined": "Strongly raised, drawn into a backward-pointing knob or spine",
            "low": "Low and smoothly rounded",
        },
    ),
    "plastron_colour": (
        "Plastron (belly shell) colour",
        {
            "coral": "Coral-red, orange or pink",
            "blotched": "Yellow or cream with large dark blotches",
            "plain_pale": "Uniform pale yellow, cream or pinkish, essentially unmarked",
            "dark": "Dark brown to black overall",
            "radiating": "Pale with dark radiating streaks or star rays",
        },
    ),
    "head_marking": (
        "Head and neck markings",
        {
            "red_ear": "Discrete red or orange patch immediately BEHIND the eye",
            "red_crown": "Red or orange arrow / crescent on TOP of the head",
            "bold_stripes": "Two or three bold clean yellow stripes along each side",
            "fine_lines": "Many fine yellow or green longitudinal lines",
            "yellow_spots": "Discrete round yellow spots on the crown",
            "dark_streaks": "Dark streaks radiating back from the eye and snout",
            "irregular_spots": "Irregular yellow-orange spots and blotches",
            "plain": "Plain, essentially unmarked",
        },
    ),
    "carapace_pattern": (
        "Carapace pattern",
        {
            "star": "Symmetrical yellow rays bursting from the centre of each scute",
            "stripes3": "Three dark longitudinal stripes",
            "ocelli": "Large dark-centred, pale-ringed eyespots (ocelli)",
            "white_streaks": "Bold white or cream radiating streaks on black",
            "blotch_per_scute": "One irregular dark blotch near the centre of each scute",
            "plain": "Plain or nearly so",
            "fine_lines": "Fine wavy longitudinal lines or reticulation",
        },
    ),
    "plastral_hinge": (
        "Hinge across the plastron allowing the shell to close",
        {"yes": "Present", "no": "Absent"},
    ),
    "nuchal_scute": (
        "Nuchal scute (small scute at the front midline, above the neck) — tortoises",
        {"present": "Present", "absent": "Absent"},
    ),
    "size_class": (
        "Straight carapace length",
        {
            "tiny": "Under 200 mm",
            "small": "200–350 mm",
            "medium": "350–600 mm",
            "large": "Over 600 mm",
        },
    ),
}

# ------------------------------------------------------------------ matrix
# None means variable / not applicable / unrecorded -> never eliminates.
# A tuple means any of those states is acceptable.

MATRIX: dict[str, dict[str, object]] = {
    "lissemys_punctata": {
        "shell_surface": "soft", "limbs": "webbed", "femoral_flaps": "yes",
        "head_shape": "normal", "keels": "none", "plastron_colour": "plain_pale",
        "head_marking": ("plain", "fine_lines"), "carapace_pattern": "plain",
        "plastral_hinge": "no", "size_class": ("tiny", "small"),
    },
    "nilssonia_gangetica": {
        "shell_surface": "soft", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "none", "plastron_colour": "plain_pale",
        "head_marking": "dark_streaks", "carapace_pattern": ("ocelli", "plain"),
        "plastral_hinge": "no", "size_class": ("medium", "large"),
    },
    "nilssonia_hurum": {
        "shell_surface": "soft", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "none", "plastron_colour": "plain_pale",
        "head_marking": "yellow_spots", "carapace_pattern": "ocelli",
        "plastral_hinge": "no", "size_class": ("small", "medium"),
    },
    "nilssonia_leithii": {
        "shell_surface": "soft", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "none", "plastron_colour": "plain_pale",
        "head_marking": ("plain", "fine_lines"), "carapace_pattern": ("ocelli", "plain"),
        "plastral_hinge": "no", "size_class": ("medium", "large"),
    },
    "nilssonia_nigricans": {
        "shell_surface": "soft", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "none", "plastron_colour": "plain_pale",
        "head_marking": ("yellow_spots", "plain"), "carapace_pattern": "plain",
        "plastral_hinge": "no", "size_class": ("medium", "large"),
    },
    "amyda_ornata": {
        "shell_surface": "soft", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "none", "plastron_colour": "plain_pale",
        "head_marking": "dark_streaks", "carapace_pattern": None,
        "plastral_hinge": "no", "size_class": ("medium", "large"),
    },
    "chitra_indica": {
        "shell_surface": "soft", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "needle", "keels": "none", "plastron_colour": "plain_pale",
        "head_marking": "fine_lines", "carapace_pattern": "fine_lines",
        "plastral_hinge": "no", "size_class": "large",
    },
    "pelochelys_cantorii": {
        "shell_surface": "soft", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "broad", "keels": "none", "plastron_colour": "plain_pale",
        "head_marking": "plain", "carapace_pattern": "plain",
        "plastral_hinge": "no", "size_class": "large",
    },
    "pangshura_tecta": {
        "shell_surface": "hard", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "one", "third_vertebral": "spined",
        "plastron_colour": "coral", "head_marking": "red_crown",
        "carapace_pattern": "plain", "plastral_hinge": "no", "size_class": "tiny",
    },
    "pangshura_tentoria": {
        "shell_surface": "hard", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "one", "third_vertebral": "low",
        "plastron_colour": "blotched", "head_marking": ("fine_lines", "plain"),
        "carapace_pattern": "plain", "plastral_hinge": "no", "size_class": ("tiny", "small"),
    },
    "pangshura_smithii": {
        "shell_surface": "hard", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "one", "third_vertebral": "low",
        "plastron_colour": "blotched", "head_marking": "fine_lines",
        "carapace_pattern": "plain", "plastral_hinge": "no", "size_class": ("tiny", "small"),
    },
    "pangshura_sylhetensis": {
        "shell_surface": "hard", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "one", "third_vertebral": "low",
        "plastron_colour": ("blotched", "plain_pale"), "head_marking": "fine_lines",
        "carapace_pattern": "plain", "plastral_hinge": "no", "size_class": "tiny",
    },
    "batagur_kachuga": {
        "shell_surface": "hard", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "normal", "keels": ("none", "one"), "third_vertebral": "low",
        "plastron_colour": ("plain_pale", "coral"), "head_marking": None,
        "carapace_pattern": "plain", "plastral_hinge": "no", "size_class": "medium",
    },
    "batagur_dhongoka": {
        "shell_surface": "hard", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "normal", "keels": ("none", "one"), "third_vertebral": "low",
        "plastron_colour": "plain_pale", "head_marking": None,
        "carapace_pattern": ("stripes3", "plain"), "plastral_hinge": "no",
        "size_class": "medium",
    },
    "batagur_baska": {
        "shell_surface": "hard", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "none", "third_vertebral": "low",
        "plastron_colour": ("plain_pale", "dark"), "head_marking": "plain",
        "carapace_pattern": "plain", "plastral_hinge": "no", "size_class": "medium",
    },
    "hardella_thurjii": {
        "shell_surface": "hard", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "three", "third_vertebral": "low",
        "plastron_colour": "dark", "head_marking": "bold_stripes",
        "carapace_pattern": "plain", "plastral_hinge": "no",
        "size_class": ("tiny", "medium"),
    },
    "melanochelys_trijuga": {
        "shell_surface": "hard", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "three", "third_vertebral": "low",
        "plastron_colour": "dark", "head_marking": "irregular_spots",
        "carapace_pattern": "plain", "plastral_hinge": "no",
        "size_class": ("tiny", "small"),
    },
    "melanochelys_tricarinata": {
        "shell_surface": "hard", "limbs": "columnar", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "three", "third_vertebral": "low",
        "plastron_colour": "plain_pale", "head_marking": "fine_lines",
        "carapace_pattern": "plain", "plastral_hinge": "no", "size_class": "tiny",
    },
    "geoclemys_hamiltonii": {
        "shell_surface": "hard", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "three", "third_vertebral": "low",
        "plastron_colour": "dark", "head_marking": "irregular_spots",
        "carapace_pattern": "white_streaks", "plastral_hinge": "no",
        "size_class": ("small", "medium"),
    },
    "morenia_petersi": {
        "shell_surface": "hard", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "one", "third_vertebral": "low",
        "plastron_colour": "plain_pale", "head_marking": "fine_lines",
        "carapace_pattern": "ocelli", "plastral_hinge": "no", "size_class": "tiny",
    },
    "cuora_amboinensis": {
        "shell_surface": "hard", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "normal", "keels": ("none", "one"), "third_vertebral": "low",
        "plastron_colour": ("plain_pale", "blotched"), "head_marking": "bold_stripes",
        "carapace_pattern": "plain", "plastral_hinge": "yes", "size_class": ("tiny", "small"),
    },
    "cuora_mouhotii": {
        "shell_surface": "hard", "limbs": "columnar", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "three", "third_vertebral": "low",
        "plastron_colour": ("plain_pale", "blotched"), "head_marking": "plain",
        "carapace_pattern": "plain", "plastral_hinge": "yes", "size_class": "tiny",
    },
    "cyclemys_gemeli": {
        "shell_surface": "hard", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "one", "third_vertebral": "low",
        "plastron_colour": "radiating", "head_marking": "fine_lines",
        "carapace_pattern": "plain", "plastral_hinge": "yes", "size_class": ("tiny", "small"),
    },
    "vijayachelys_silvatica": {
        "shell_surface": "hard", "limbs": "columnar", "femoral_flaps": "no",
        "head_shape": "normal", "keels": ("none", "one"), "third_vertebral": "low",
        "plastron_colour": ("plain_pale", "blotched"), "head_marking": None,
        "carapace_pattern": "plain", "plastral_hinge": "no", "size_class": "tiny",
    },
    "indotestudo_elongata": {
        "shell_surface": "hard", "limbs": "columnar", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "none", "plastron_colour": ("plain_pale", "blotched"),
        "head_marking": "plain", "carapace_pattern": "blotch_per_scute",
        "plastral_hinge": "no", "nuchal_scute": "present", "size_class": ("small", "medium"),
    },
    "geochelone_elegans": {
        "shell_surface": "hard", "limbs": "columnar", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "none", "plastron_colour": "radiating",
        "head_marking": "plain", "carapace_pattern": "star",
        "plastral_hinge": "no", "nuchal_scute": "absent", "size_class": ("small", "medium"),
    },
    "indotestudo_travancorica": {
        "shell_surface": "hard", "limbs": "columnar", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "none", "plastron_colour": ("plain_pale", "blotched"),
        "head_marking": "plain", "carapace_pattern": ("blotch_per_scute", "plain"),
        "plastral_hinge": "no", "nuchal_scute": "present", "size_class": ("small", "medium"),
    },
    "manouria_emys": {
        "shell_surface": "hard", "limbs": "columnar", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "none", "plastron_colour": "dark",
        "head_marking": "plain", "carapace_pattern": "plain",
        "plastral_hinge": "no", "nuchal_scute": "present", "size_class": ("medium", "large"),
    },
    "manouria_impressa": {
        "shell_surface": "hard", "limbs": "columnar", "femoral_flaps": "no",
        "head_shape": "normal", "keels": "none", "plastron_colour": "radiating",
        "head_marking": "plain", "carapace_pattern": "blotch_per_scute",
        "plastral_hinge": "no", "nuchal_scute": "present", "size_class": ("small", "medium"),
    },
    "trachemys_scripta_elegans": {
        "shell_surface": "hard", "limbs": "webbed", "femoral_flaps": "no",
        "head_shape": "normal", "keels": ("none", "one"), "third_vertebral": "low",
        "plastron_colour": "blotched", "head_marking": "red_ear",
        "carapace_pattern": ("plain", "ocelli"), "plastral_hinge": "no",
        "size_class": ("tiny", "small"),
    },
}


@dataclass
class KeyResult:
    species_id: str
    matched: int
    total_scored: int
    contradicted: bool
    unresolved: list[str]

    @property
    def score(self) -> float:
        return self.matched / self.total_scored if self.total_scored else 0.0


def _state_matches(expected, observed: str) -> bool | None:
    """True = match, False = contradiction, None = uninformative for this taxon."""
    if expected is None:
        return None
    if isinstance(expected, tuple):
        return observed in expected
    return expected == observed


def run_key(observations: dict[str, str]) -> list[KeyResult]:
    """
    Score every taxon against the observed character states.

    Returns all taxa, sorted: non-contradicted first, then by proportion of
    observations matched. Contradicted taxa are retained rather than deleted,
    because field observers misread characters and a hard elimination that
    silently drops the right answer is worse than a ranked list.
    """
    observations = {k: v for k, v in observations.items() if v}
    results: list[KeyResult] = []

    for species_id, profile in MATRIX.items():
        matched = 0
        scored = 0
        contradicted = False
        unresolved: list[str] = []

        for char, observed in observations.items():
            verdict = _state_matches(profile.get(char), observed)
            if verdict is None:
                unresolved.append(char)
                continue
            scored += 1
            if verdict:
                matched += 1
            else:
                contradicted = True

        results.append(
            KeyResult(species_id, matched, scored, contradicted, unresolved)
        )

    results.sort(key=lambda r: (r.contradicted, -r.score, -r.matched))
    return results


def most_discriminating(observations: dict[str, str], candidates: list[str]) -> str | None:
    """
    Suggest which character to observe next: the unrecorded character that
    splits the surviving candidate set most evenly.
    """
    remaining = [c for c in CHARACTERS if c not in observations]
    best, best_balance = None, -1.0

    for char in remaining:
        buckets: dict[str, int] = {}
        informative = 0
        for sid in candidates:
            expected = MATRIX.get(sid, {}).get(char)
            if expected is None:
                continue
            informative += 1
            states = expected if isinstance(expected, tuple) else (expected,)
            for s in states:
                buckets[s] = buckets.get(s, 0) + 1

        if informative < 2 or len(buckets) < 2:
            continue
        # Balance: 1.0 when the character splits candidates evenly.
        largest = max(buckets.values())
        balance = 1.0 - (largest / informative)
        if balance > best_balance:
            best, best_balance = char, balance

    return best
