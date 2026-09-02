"""
Identification engine.

Design note, because it matters more than the model architecture:

A bare YOLOv8 softmax score is not a confidence estimate. Classification heads
trained with cross-entropy are systematically overconfident, and on a field
photograph of a species that was never in the training set they will still
return 0.97 for something. Reporting that number to a Range Officer who is
about to write a seizure memo is the failure mode this module exists to avoid.

So the pipeline is:

    detect -> crop -> classify -> temperature-scale -> geographic prior
           -> out-of-distribution check -> entropy check -> tier

and the output is a determination tier with an explicit action, not a bare
percentage. Every step is inspectable in the returned object.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from config import (
    CALIBRATION_PATH,
    CLASSIFIER_PATH,
    CLS_IMAGE_SIZE,
    DEFAULT_ENERGY_THRESHOLD,
    DEFAULT_TEMPERATURE,
    DETECTOR_PATH,
    DET_CONF_THRESHOLD,
    DET_CROP_PADDING,
    MAX_NORMALISED_ENTROPY,
    GALLERY_PATH,
    MSP_OOD_FLOOR,
    PUBLISHED_GALLERY_PATH,
    TIER_CONFIRMED,
    TIER_PROBABLE,
    TIER_TENTATIVE,
)
from core.database import SpeciesDB
from core.matcher import Gallery, MatcherError, embed_images, load_backbone

logger = logging.getLogger(__name__)

# Returned by `backend_of` when the identifier predates the code asking. See
# there for why that happens.
STALE_BACKEND = "stale"


def backend_summary(identifier) -> str:
    """One line naming what is actually loaded, or "" when nothing is.

    A deployment that has just been given a new gallery looks identical to one
    still running the old one, and the question "is it using my photographs
    yet" has no answer anywhere on screen. Reading the counts off the file that
    is loaded answers it: the number moves when a rebuild lands, and does not
    when a restart has been forgotten.

    Never raises. A gallery that cannot be read is a caption that says so, not
    a page that fails to render.
    """
    try:
        backend = backend_of(identifier)
        if backend == "classifier":
            return identifier.classifier_path.name
        if backend != "gallery":
            return ""
        gallery = identifier._ensure_gallery()
    except Exception as exc:  # noqa: BLE001 - a status line must not take the page down
        logger.warning("Could not summarise the identification backend: %s", exc)
        return "could not be read"

    return (
        f"{gallery.vectors.shape[0]} photographs, {len(gallery.classes)} species"
    )


def gallery_is_unfit(identifier) -> dict | None:
    """Metrics of a gallery that failed its own held-out test, else None.

    A gallery measured at or below chance is not an identifier. It still
    returns a species for every photograph, still reports a probability, and
    with a fitted temperature that probability is even honest — honestly
    uniform. Nothing downstream can tell that apart from a working gallery, so
    it has to be caught here and refused rather than presented.
    """
    try:
        if backend_of(identifier) != "gallery":
            return None
        gallery = identifier._ensure_gallery()
    except Exception:  # noqa: BLE001 - a guard must not itself take the page down
        return None
    if gallery.reliable:
        return None
    # A gallery built before the check recorded no chance rate, and the caller
    # renders both numbers; derive it rather than hand back a None to format.
    chance = gallery.metrics.get("chance")
    if not isinstance(chance, (int, float)) and gallery.classes:
        chance = 1.0 / len(gallery.classes)
    accuracy = gallery.metrics.get("accuracy")
    return {
        "accuracy": float(accuracy) if isinstance(accuracy, (int, float)) else 0.0,
        "chance": float(chance) if isinstance(chance, (int, float)) else 0.0,
        "n_evaluated": gallery.metrics.get("n_evaluated", 0),
        "per_class_recall": gallery.metrics.get("per_class_recall", {}),
    }


def gallery_species_counts(identifier) -> dict[str, int]:
    """Photographs per species in the loaded gallery, or {} if unavailable."""
    try:
        if backend_of(identifier) != "gallery":
            return {}
        gallery = identifier._ensure_gallery()
    except Exception:  # noqa: BLE001 - diagnostics must never be load-bearing
        return {}
    counts: dict[str, int] = {}
    for species_id in gallery.species:
        counts[str(species_id)] = counts.get(str(species_id), 0) + 1
    return counts


def backend_of(identifier) -> str | None:
    """Which scorer an identifier is using, tolerating one built by older code.

    Streamlit re-executes its script when the source changes but does not
    re-import modules already in `sys.modules`, and `@st.cache_resource` hands
    back the object it built earlier — from the class as it was then. So after
    an update to this file, a long-running app can hold an identifier that
    predates `backend` entirely, and reading it raises AttributeError from
    somewhere that looks unrelated.

    Only restarting the process fixes it. Reporting STALE_BACKEND lets the
    caller say that plainly instead of taking the whole app down over a status
    line, and keeps the mistake from being read as "no model installed".
    """
    try:
        return identifier.backend
    except AttributeError:
        return STALE_BACKEND


# ====================================================================== results

@dataclass
class Candidate:
    species_id: str
    scientific_name: str
    common_en: str
    model_probability: float      # after temperature scaling, before geography
    geo_multiplier: float
    posterior: float              # after geography, renormalised
    occurrence: str               # resident / marginal / absent / introduced


@dataclass
class Determination:
    """The complete, auditable output of one identification."""

    candidates: list[Candidate]
    tier: str                     # CONFIRMED / PROBABLE / TENTATIVE / INDETERMINATE / REJECTED
    action: str                   # what the user should do about it
    warnings: list[str] = field(default_factory=list)

    # diagnostics for audit
    energy: float = 0.0
    similarity: float = float("nan")   # gallery path: best cosine match
    normalised_entropy: float = 0.0
    temperature: float = 1.0
    detector_confidence: float | None = None
    calibrated: bool = False
    state: str | None = None
    model_version: str = ""

    @property
    def top(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def confidence_pct(self) -> float:
        return round(100.0 * self.top.posterior, 1) if self.top else 0.0

    def to_record(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "action": self.action,
            "energy": None if math.isnan(self.energy) else round(self.energy, 4),
            "similarity": None if math.isnan(self.similarity) else round(self.similarity, 4),
            "normalised_entropy": round(self.normalised_entropy, 4),
            "temperature": self.temperature,
            "calibrated": self.calibrated,
            "detector_confidence": self.detector_confidence,
            "state": self.state,
            "model_version": self.model_version,
            "warnings": self.warnings,
            "candidates": [
                {
                    "species_id": c.species_id,
                    "scientific_name": c.scientific_name,
                    "model_probability": round(c.model_probability, 4),
                    "geo_multiplier": c.geo_multiplier,
                    "posterior": round(c.posterior, 4),
                    "occurrence": c.occurrence,
                }
                for c in self.candidates
            ],
        }


# ====================================================================== maths

def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Temperature-scaled softmax. T > 1 softens an overconfident head."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    z = logits.astype(np.float64) / temperature
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def free_energy(logits: np.ndarray, temperature: float = 1.0) -> float:
    """
    Free-energy OOD score (Liu et al. 2020), E(x) = -T * logsumexp(logits / T).

    In-distribution inputs give low (more negative) energy. An image of a
    monitor lizard, a blurred hand, or a chelonian species absent from the
    training set gives high energy even when the softmax looks decisive.
    """
    z = logits.astype(np.float64) / temperature
    m = z.max()
    return float(-temperature * (m + math.log(np.exp(z - m).sum())))


def normalised_entropy(probs: np.ndarray) -> float:
    """Shannon entropy scaled to [0, 1]. 0 = certain, 1 = uniform."""
    p = probs[probs > 0]
    if p.size <= 1:
        return 0.0
    h = float(-(p * np.log(p)).sum())
    return h / math.log(len(probs))


# ====================================================================== engine

class ChelonidIdentifier:
    """
    Wraps the YOLOv8 classification model (and optional detector) with the
    calibration and abstention logic described above.

    Loading is lazy so the Streamlit app starts and shows the reference
    material even when no trained weights are present yet.
    """

    def __init__(
        self,
        db: SpeciesDB,
        classifier_path: Path = CLASSIFIER_PATH,
        detector_path: Path = DETECTOR_PATH,
        calibration_path: Path = CALIBRATION_PATH,
        gallery_path: Path = GALLERY_PATH,
        published_gallery_path: Path = PUBLISHED_GALLERY_PATH,
    ):
        self.db = db
        self.classifier_path = Path(classifier_path)
        self.detector_path = Path(detector_path)
        self.gallery_path = Path(gallery_path)
        self.published_gallery_path = Path(published_gallery_path)
        self._classifier = None
        self._detector = None
        self._gallery = None
        self._embedder = None
        self._class_names: list[str] = []
        self.model_version = ""
        self._hook_handle = None
        self._captured_logits: np.ndarray | None = None

        self.temperature = DEFAULT_TEMPERATURE
        self.energy_threshold = DEFAULT_ENERGY_THRESHOLD
        self.calibrated = False
        if self.backend == "classifier":
            self._load_calibration(Path(calibration_path))

    # -- setup ---------------------------------------------------------
    def _load_calibration(self, path: Path) -> None:
        if not path.exists():
            logger.warning(
                "No calibration file at %s. Confidence values are UNCALIBRATED "
                "and will be overconfident. Run training/calibrate.py.", path
            )
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.temperature = float(payload["temperature"])
            threshold = payload.get("energy_threshold")
            if threshold is None or payload.get("energy_threshold_valid") is False:
                logger.warning(
                    "Calibration contains no valid energy threshold. Temperature "
                    "scaling applies, but out-of-distribution screening falls "
                    "back to the untuned default."
                )
            else:
                self.energy_threshold = float(threshold)
            self.calibrated = True
            logger.info(
                "Calibration loaded: T=%.3f, energy threshold=%.3f",
                self.temperature, self.energy_threshold,
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            logger.error("Calibration file unusable (%s); falling back to defaults.", exc)

    @property
    def backend(self) -> str | None:
        """Which scorer is in force.

        A trained classifier wins when both are present: it was fitted on these
        species, where the gallery leans on features learned from ImageNet. The
        gallery is what makes the photograph tab usable before that model
        exists, not a fallback to prefer once it does.
        """
        if self.classifier_path.exists():
            return "classifier"
        if self.active_gallery_path is not None:
            return "gallery"
        return None

    @property
    def active_gallery_path(self) -> Path | None:
        """The gallery in force, or None.

        A locally built gallery wins over the committed one: it is the newer of
        the two by construction, since publishing is a step you take after
        building. The committed fallback is what makes the photograph tab work
        on a hosted deployment, which re-clones the repository on every restart
        and never sees the gitignored one.
        """
        for path in (self.gallery_path, self.published_gallery_path):
            if path.exists():
                return path
        return None

    @property
    def available(self) -> bool:
        return self.backend is not None

    def _ensure_classifier(self):
        if self._classifier is not None:
            return self._classifier
        if not self.classifier_path.exists():
            raise FileNotFoundError(
                f"No classifier weights at {self.classifier_path}. "
                "Train one with training/train_classifier.py, or use the "
                "morphological key instead."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. pip install ultralytics"
            ) from exc

        self._classifier = YOLO(str(self.classifier_path))
        names = self._classifier.names
        self._class_names = [names[i] for i in sorted(names)]
        self.model_version = self.classifier_path.name

        unknown = [n for n in self._class_names if n not in self.db]
        if unknown:
            raise ValueError(
                "Model classes are not present in the species database: "
                f"{unknown}. The model and the database are out of sync."
            )
        logger.info("Classifier loaded with %d classes", len(self._class_names))
        return self._classifier

    def _ensure_detector(self):
        if self._detector is not None or not self.detector_path.exists():
            return self._detector
        try:
            from ultralytics import YOLO
            self._detector = YOLO(str(self.detector_path))
            logger.info("Detector loaded from %s", self.detector_path)
        except Exception as exc:  # detector is optional; never fatal
            logger.warning("Detector unavailable (%s); classifying full frame.", exc)
            self._detector = None
        return self._detector

    def _ensure_gallery(self) -> Gallery:
        if self._gallery is not None:
            return self._gallery
        path = self.active_gallery_path
        if path is None:
            raise FileNotFoundError(
                f"No gallery at {self.gallery_path} or {self.published_gallery_path}. "
                "Build one with training/build_gallery.py, or use the "
                "morphological key instead."
            )
        gallery = Gallery.load(path)

        unknown = [n for n in gallery.classes if n not in self.db]
        if unknown:
            raise ValueError(
                "Gallery species are not present in the species database: "
                f"{unknown}. The gallery and the database are out of sync."
            )

        self._gallery = gallery
        self._class_names = list(gallery.classes)
        self.temperature = gallery.temperature
        self.calibrated = gallery.calibrated
        self.model_version = (
            f"{path.name} [{gallery.backbone}, "
            f"{gallery.vectors.shape[0]} photographs, {len(gallery.classes)} species]"
        )
        logger.info("Gallery loaded: %s", self.model_version)
        return gallery

    def _ensure_embedder(self):
        if self._embedder is None:
            gallery = self._ensure_gallery()
            self._embedder = load_backbone(gallery.backbone)
        return self._embedder

    def _gallery_scores(self, image) -> tuple[np.ndarray, float, bool]:
        """Probabilities over species, the best match found, and whether to reject."""
        gallery = self._ensure_gallery()
        model, transform = self._ensure_embedder()

        vectors = embed_images([image], model, transform)
        if vectors.size == 0:
            raise MatcherError("The photograph could not be embedded.")

        scores, best = gallery.score(vectors)
        similarity = float(best[0])
        return (
            softmax(scores[0], self.temperature),
            similarity,
            similarity < gallery.similarity_floor,
        )

    # -- pipeline ------------------------------------------------------
    def _locate(self, image) -> tuple[Any, float | None]:
        """Crop to the animal if a detector is available. Returns (image, conf)."""
        det = self._ensure_detector()
        if det is None:
            return image, None
        try:
            result = det.predict(image, conf=DET_CONF_THRESHOLD, verbose=False)[0]
            if len(result.boxes) == 0:
                return image, None
            confs = result.boxes.conf.cpu().numpy()
            best = int(np.argmax(confs))
            x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy()[best]
            w, h = x2 - x1, y2 - y1
            px, py = w * DET_CROP_PADDING, h * DET_CROP_PADDING
            arr = np.asarray(image)
            H, W = arr.shape[:2]
            x1 = max(0, int(x1 - px)); y1 = max(0, int(y1 - py))
            x2 = min(W, int(x2 + px)); y2 = min(H, int(y2 + py))
            if x2 - x1 < 16 or y2 - y1 < 16:
                return image, float(confs[best])
            from PIL import Image
            return Image.fromarray(arr[y1:y2, x1:x2]), float(confs[best])
        except Exception as exc:
            logger.warning("Detection failed (%s); classifying full frame.", exc)
            return image, None

    def _attach_logit_hook(self, model) -> bool:
        """
        Capture pre-softmax logits from the classification head.

        This matters more than it looks. Ultralytics exposes only post-softmax
        probabilities, and softmax is shift-invariant, so the raw logit scale
        cannot be recovered from them: log(p) always gives logsumexp == 0 and
        the free-energy OOD score becomes a constant zero that rejects nothing.
        Hooking the final Linear layer gets the real values.

        Returns True if the hook attached; False means fall back to an
        entropy-based OOD gate, which is weaker but not silently broken.
        """
        if self._hook_handle is not None:
            return True
        try:
            import torch.nn as nn
            head = None
            for module in reversed(list(model.model.modules())):
                if isinstance(module, nn.Linear):
                    head = module
                    break
            if head is None:
                return False

            def _capture(_module, _inputs, output):
                self._captured_logits = output.detach().cpu().numpy().astype(np.float64)

            self._hook_handle = head.register_forward_hook(_capture)
            logger.info("Logit hook attached to %s", head)
            return True
        except Exception as exc:
            logger.warning(
                "Could not attach a logit hook (%s). Falling back to an "
                "entropy-based out-of-distribution gate.", exc
            )
            return False

    def _logits(self, image) -> tuple[np.ndarray, bool]:
        """
        Returns (scores, are_true_logits).

        When are_true_logits is False the values are log-probabilities: usable
        for temperature scaling and ranking, but NOT for free energy.
        """
        model = self._ensure_classifier()
        hooked = self._attach_logit_hook(model)
        self._captured_logits = None

        result = model.predict(image, imgsz=CLS_IMAGE_SIZE, verbose=False)[0]

        if hooked and self._captured_logits is not None:
            raw = self._captured_logits
            return (raw[0] if raw.ndim > 1 else raw), True

        probs = np.clip(result.probs.data.cpu().numpy().astype(np.float64), 1e-12, 1.0)
        return np.log(probs), False

    # -- public --------------------------------------------------------
    def raw_scores(self, image) -> tuple[np.ndarray, bool]:
        """
        Public access to the scoring path used by identify().

        training/calibrate.py MUST use this rather than re-deriving scores from
        probabilities. If calibration fits a threshold on log-probabilities
        while inference measures true logits, the two are on different scales
        and the out-of-distribution gate silently misfires in the field.
        """
        return self._logits(image)

    def identify(
        self,
        image,
        state: str | None = None,
        top_k: int = 4,
    ) -> Determination:
        """Run one identification. `image` is a PIL Image or a path."""
        cropped, det_conf = self._locate(image)
        warnings: list[str] = []
        energy = float("nan")
        similarity = float("nan")

        if self.backend == "gallery":
            # No trained head, so no logits and no free energy. The gate here is
            # the similarity to the nearest photograph in the gallery.
            probs, similarity, reject = self._gallery_scores(cropped)
            warnings.append(
                "Identified by matching against the reference gallery, not by a "
                "trained model. Fine distinctions between similar species are "
                "weaker on this path than the percentages suggest; confirm the "
                "confusable pairs against the morphological key."
            )
        else:
            scores, true_logits = self._logits(cropped)
            probs = softmax(scores, self.temperature)

            # -- out-of-distribution gate ------------------------------
            # Free energy is only meaningful on true logits. Without them, fall
            # back to a maximum-softmax-probability gate and say so, rather than
            # reporting an energy value that is structurally always zero.
            if true_logits:
                energy = free_energy(scores, self.temperature)
                reject = energy > self.energy_threshold
            else:
                reject = float(probs.max()) < MSP_OOD_FLOOR
                warnings.append(
                    "Raw logits were unavailable, so out-of-distribution screening "
                    "is running in a weaker fallback mode. Off-target images are "
                    "more likely to be given a species name."
                )

        # Determined by the backend above: the gallery reads it off the file it
        # was built with, the classifier off calibration.json.
        if not self.calibrated:
            warnings.insert(0, (
                "Model is UNCALIBRATED. Treat all probabilities as upper bounds "
                "and confirm every determination against the morphological key."
            ))

        if reject:
            return Determination(
                candidates=[],
                tier="REJECTED",
                action=(
                    f"The image does not resemble {'anything in the reference gallery' if self.backend == 'gallery' else 'any species this model was trained on'}. "
                    "Retake the photograph following the capture protocol (dorsal, "
                    "ventral, lateral, head close-up, scale reference), or use the "
                    "morphological key. If the animal is genuinely not an Indian "
                    "chelonian, this rejection is correct."
                ),
                energy=energy,
                similarity=similarity,
                temperature=self.temperature,
                detector_confidence=det_conf,
                calibrated=self.calibrated,
                state=state,
                model_version=self.model_version,
                warnings=warnings,
            )

        # -- geographic prior ------------------------------------------
        posteriors = probs.copy()
        multipliers = np.ones_like(probs)
        if state:
            for i, name in enumerate(self._class_names):
                multipliers[i] = self.db.get(name).geo_prior(state)
            posteriors = probs * multipliers
            total = posteriors.sum()
            posteriors = posteriors / total if total > 0 else probs.copy()

        order = np.argsort(posteriors)[::-1][:top_k]
        candidates = [
            Candidate(
                species_id=self._class_names[i],
                scientific_name=self.db.get(self._class_names[i]).scientific_name,
                common_en=self.db.get(self._class_names[i]).common_en,
                model_probability=float(probs[i]),
                geo_multiplier=float(multipliers[i]),
                posterior=float(posteriors[i]),
                occurrence=self.db.get(self._class_names[i]).occurs_in(state),
            )
            for i in order
        ]

        ent = normalised_entropy(posteriors)
        tier, action, extra = self._assign_tier(candidates, ent, state)
        warnings.extend(extra)

        return Determination(
            candidates=candidates,
            tier=tier,
            action=action,
            warnings=warnings,
            energy=energy,
            similarity=similarity,
            normalised_entropy=ent,
            temperature=self.temperature,
            detector_confidence=det_conf,
            calibrated=self.calibrated,
            state=state,
            model_version=self.model_version,
        )

    # -- tiering -------------------------------------------------------
    def _assign_tier(
        self, candidates: Sequence[Candidate], entropy: float, state: str | None
    ) -> tuple[str, str, list[str]]:
        warnings: list[str] = []
        top = candidates[0]
        sp = self.db.get(top.species_id)

        # Geography flags
        if state and top.occurrence == "absent":
            warnings.append(
                f"{sp.scientific_name} is not known from {state}. If the "
                "determination holds, this is a trade, transport or release "
                "record and should be reported as such, not logged as a wild "
                "occurrence."
            )
        elif state and top.occurrence == "marginal":
            warnings.append(
                f"{sp.scientific_name} is marginal in {state}. Retain voucher "
                "photographs before recording."
            )

        # Confusion-pair flag
        if len(candidates) > 1:
            runner = candidates[1]
            disc = self.db.discriminator(top.species_id, runner.species_id)
            if disc and runner.posterior > 0.15:
                warnings.append(
                    f"Close call against {runner.scientific_name}. "
                    f"Check this character: {disc}"
                )
            # Legal-status divergence is worth its own line.
            other = self.db.get(runner.species_id)
            if runner.posterior > 0.15 and sp.wpa != other.wpa:
                warnings.append(
                    f"LEGAL DIVERGENCE: {sp.scientific_name} is {sp.wpa} but "
                    f"{other.scientific_name} is {other.wpa}. The offence "
                    "category differs between the top two candidates. Do not "
                    "finalise paperwork on this determination alone."
                )

        # Threat-status flag
        if sp.is_threatened and top.posterior >= TIER_PROBABLE:
            warnings.append(
                f"{sp.scientific_name} is {sp.iucn_label} ({sp.iucn_status}) and "
                f"{sp.wpa}. Notify the Divisional Forest Officer and retain the "
                "record regardless of outcome."
            )

        # Tier assignment
        if entropy > MAX_NORMALISED_ENTROPY and top.posterior < TIER_CONFIRMED:
            return (
                "TENTATIVE",
                "The model is spread across several species. Report to family or "
                "genus only and work through the morphological key on the live "
                "animal or a better photograph.",
                warnings,
            )
        if top.posterior >= TIER_CONFIRMED:
            return (
                "CONFIRMED",
                "Record the species. Retain the photographs as a voucher and "
                "attach them to the field register entry.",
                warnings,
            )
        if top.posterior >= TIER_PROBABLE:
            return (
                "PROBABLE",
                "Record as a probable determination. Verify the discriminating "
                "character listed below before it goes on any official form.",
                warnings,
            )
        if top.posterior >= TIER_TENTATIVE:
            return (
                "TENTATIVE",
                "Insufficient for a species-level record. Report to genus, run "
                "the morphological key, and retake photographs per the capture "
                "protocol.",
                warnings,
            )
        return (
            "INDETERMINATE",
            "No candidate reaches the reporting threshold. Refer to a chelonian "
            "specialist (Turtle Survival Alliance India, WII, or the state Chief "
            "Wildlife Warden's office). Retain the animal safely and photograph "
            "it fully. Do not enter a species on any form.",
            warnings,
        )
