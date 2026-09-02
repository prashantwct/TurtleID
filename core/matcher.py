"""
Identification by matching, not by training.

The classifier in `core/inference.py` learns a decision boundary from hundreds
of photographs per species. That is the right tool once the photographs exist.
Before they do, it has nothing to learn from, and the honest report on a model
trained from a handful of images is that it cannot be validated at all.

This module takes the other approach. Every photograph is passed once through a
network pretrained on ImageNet and reduced to a vector; identifying a new
photograph means finding the vectors nearest to it and reading off which species
they belong to. There is no training step, no epochs, and no train/validation
split. Adding a photograph means adding one vector — the gallery IS the model.

WHAT THIS BUYS, AND WHAT IT COSTS
---------------------------------
It works from two or three photographs per species, on a laptop, in minutes.
That is the entire point: it is usable on the day the first photographs arrive,
and it improves every time one is added, with no retraining.

What it costs is fine discrimination. ImageNet features separate a softshell
from a tortoise easily. They do not reliably separate *Pangshura tecta* from
*P. smithii*, which differ in plastron colour — a coral-red versus dark-blotched
character that a generic feature vector barely encodes, and the difference
between Schedule I and Schedule II. So the abstention machinery matters more
here, not less, and the confusable pairs belong in the morphological key.

HOW A SPECIES SCORE IS FORMED
-----------------------------
Cosine similarity against every vector in the gallery, then per species the mean
of its `neighbours` best matches. Taking the single best match makes the result
hostage to one lucky photograph; averaging over all of a species' photographs
buries a genuine match among its bad angles. A small top-k mean is the usual
compromise and it is what HotSpotter-style systems settle on too.

Those scores become probabilities through a temperature fitted by
leave-one-capture-out on the gallery itself, which is what replaces the
validation split. The same procedure sets the similarity floor below which a
photograph is rejected as not resembling anything in the gallery — the analogue
of the free-energy gate on the trained path, and the thing that stops a monitor
lizard being handed a species name.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from config import (
    GALLERY_IMAGE_SIZE,
    GALLERY_NEIGHBOURS,
    GALLERY_SIMILARITY_FLOOR,
    GALLERY_TEMPERATURE,
)

logger = logging.getLogger(__name__)

# Backbone name -> (torchvision constructor, weights enum attribute, feature attr)
BACKBONES = {
    "resnet50": ("resnet50", "ResNet50_Weights", "fc"),
    "resnet18": ("resnet18", "ResNet18_Weights", "fc"),
    "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights", "classifier"),
}
DEFAULT_BACKBONE = "resnet50"

# Similarity assigned to a species with nothing in the gallery. Cosine
# similarity cannot reach -2, so such a species can never win a comparison and
# never needs a special case downstream.
ABSENT = -2.0


class MatcherError(RuntimeError):
    """Raised where a clear message helps more than a traceback."""


# ====================================================================== embedding

def load_backbone(name: str = DEFAULT_BACKBONE, pretrained: bool = True):
    """Return (model in eval mode, preprocessing transform).

    `pretrained=False` exists for tests: it builds the same architecture without
    fetching weights. The vectors it produces are meaningless for
    identification, so nothing that writes a gallery should use it.
    """
    if name not in BACKBONES:
        raise MatcherError(
            f"Unknown backbone {name!r}. Available: {', '.join(sorted(BACKBONES))}"
        )
    try:
        import torch
        import torchvision
        from torchvision import transforms
    except ImportError as exc:
        raise MatcherError(
            "torch and torchvision are required to build or query a gallery. "
            "pip install -r requirements.txt"
        ) from exc

    constructor, weights_attr, head = BACKBONES[name]
    weights = None
    if pretrained:
        try:
            weights = getattr(torchvision.models, weights_attr).DEFAULT
        except Exception as exc:  # noqa: BLE001 - torchvision version drift
            raise MatcherError(
                f"Could not resolve pretrained weights for {name}: {exc}"
            ) from exc

    try:
        model = getattr(torchvision.models, constructor)(weights=weights)
    except Exception as exc:  # noqa: BLE001 - almost always a failed download
        raise MatcherError(
            f"Could not load {name} with pretrained weights ({exc}). The weights "
            f"are downloaded once from download.pytorch.org and cached in "
            f"~/.cache/torch; this needs network access on the first run."
        ) from exc

    setattr(model, head, torch.nn.Identity())
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(int(GALLERY_IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(GALLERY_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


def embed_images(
    images: Iterable[Any],
    model,
    transform,
    batch_size: int = 16,
) -> np.ndarray:
    """Embed PIL images into L2-normalised row vectors.

    Each image is embedded together with its mirror and the two averaged. A
    turtle photographed from the left and the same turtle from the right should
    land in the same place, and averaging the pair costs one extra forward pass.
    """
    import torch

    batch: list[Any] = []
    out: list[np.ndarray] = []

    def flush() -> None:
        if not batch:
            return
        tensor = torch.stack(batch)
        with torch.no_grad():
            features = model(tensor) + model(torch.flip(tensor, dims=[3]))
        out.append(features.cpu().numpy().astype(np.float32))
        batch.clear()

    for image in images:
        batch.append(transform(image.convert("RGB")))
        if len(batch) >= batch_size:
            flush()
    flush()

    if not out:
        return np.empty((0, 0), dtype=np.float32)
    return normalise_rows(np.vstack(out))


def normalise_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, so a dot product is a cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


# ====================================================================== scoring

def species_scores(
    similarities: np.ndarray,
    entry_species: np.ndarray,
    classes: Sequence[str],
    neighbours: int = GALLERY_NEIGHBOURS,
) -> np.ndarray:
    """Reduce per-photograph similarities to one score per species.

    `similarities` is (n_queries, n_entries); the result is (n_queries, n_classes).
    Entries masked out by the caller (leave-one-out) should already be -inf.
    """
    scores = np.full((similarities.shape[0], len(classes)), ABSENT, dtype=np.float64)
    for c, species_id in enumerate(classes):
        columns = similarities[:, entry_species == species_id]
        if columns.size == 0:
            continue
        k = min(neighbours, columns.shape[1])
        best = np.sort(columns, axis=1)[:, -k:]
        with np.errstate(invalid="ignore"):
            means = best.mean(axis=1)
        # A row whose every entry was masked out averages to -inf or nan; that
        # species simply has no evidence for this query.
        scores[:, c] = np.where(np.isfinite(means), means, ABSENT)
    return scores


def fit_temperature(scores: np.ndarray, labels: np.ndarray) -> float:
    """Fit the scalar that turns similarity gaps into probabilities.

    Cosine similarities live in a much narrower range than logits, so the grid
    is correspondingly finer and lower than the one `training/calibrate.py` uses
    on the trained path. Minimises negative log-likelihood, same as there.
    """
    from core.inference import softmax

    def nll(temperature: float) -> float:
        total = 0.0
        for row, label in zip(scores, labels):
            total -= np.log(max(softmax(row, temperature)[label], 1e-12))
        return total / len(labels)

    grid = np.concatenate([
        np.arange(0.01, 0.20, 0.005),
        np.arange(0.20, 1.05, 0.05),
    ])
    best = float(min(grid, key=nll))

    lo, hi = max(0.005, best - 0.01), best + 0.01
    phi = (5 ** 0.5 - 1) / 2
    for _ in range(30):
        a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
    return round((lo + hi) / 2, 5)


# ====================================================================== gallery

def _reliability_of(meta: dict) -> bool:
    """Whether a stored gallery beat chance, deciding it if it never recorded so.

    Galleries written before the check carry no verdict but do carry the
    accuracy it would have been drawn from, so read it off rather than waiting
    for a rebuild — the ones most in need of the verdict are the ones already
    deployed. A gallery with nothing to read stays unproven, not condemned:
    absent evidence is not evidence of failure.
    """
    if "reliable" in meta:
        return bool(meta["reliable"])

    metrics = meta.get("metrics") or {}
    accuracy = metrics.get("accuracy")
    classes = meta.get("classes") or []
    if not isinstance(accuracy, (int, float)) or not classes:
        return True
    return float(accuracy) > 1.0 / len(classes)


@dataclass
class Gallery:
    """Every embedded photograph, and what is known about how well it matches."""

    vectors: np.ndarray          # (N, D) float32, L2-normalised
    species: np.ndarray          # (N,) species id per vector
    captures: np.ndarray         # (N,) capture id per vector
    classes: list[str]           # sorted species present
    backbone: str = DEFAULT_BACKBONE
    neighbours: int = GALLERY_NEIGHBOURS
    temperature: float = GALLERY_TEMPERATURE
    similarity_floor: float = GALLERY_SIMILARITY_FLOOR
    calibrated: bool = False
    # Whether held-out accuracy beat picking a species at random. A gallery
    # that did not is not an identifier, however well calibrated it is:
    # calibration only means the reported probability matches reality, and a
    # gallery can be perfectly calibrated about knowing nothing.
    reliable: bool = True
    metrics: dict = None

    def __post_init__(self) -> None:
        if self.metrics is None:
            self.metrics = {}

    # -- persistence ---------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vectors=self.vectors,
            species=self.species,
            captures=self.captures,
            meta=np.array(json.dumps({
                "classes": self.classes,
                "backbone": self.backbone,
                "neighbours": self.neighbours,
                "temperature": self.temperature,
                "similarity_floor": self.similarity_floor,
                "calibrated": self.calibrated,
                "reliable": self.reliable,
                "metrics": self.metrics,
            })),
        )

    @classmethod
    def load(cls, path: Path) -> "Gallery":
        with np.load(path, allow_pickle=False) as payload:
            meta = json.loads(str(payload["meta"]))
            return cls(
                vectors=payload["vectors"],
                species=payload["species"],
                captures=payload["captures"],
                classes=list(meta["classes"]),
                backbone=meta["backbone"],
                neighbours=int(meta["neighbours"]),
                temperature=float(meta["temperature"]),
                similarity_floor=float(meta["similarity_floor"]),
                calibrated=bool(meta["calibrated"]),
                reliable=_reliability_of(meta),
                metrics=meta.get("metrics", {}),
            )

    def published(self) -> "Gallery":
        """A copy safe to commit: capture ids replaced with a constant.

        Capture ids exist for leave-one-capture-out, which happens at build
        time. Identification never reads them. They are also the one field here
        that carries locality — an id like `chambal-2026-08-19` names a river —
        so publishing keeps the vectors and the species and drops them.

        The fitted temperature, floor and metrics are kept: a deployment that
        could not tell whether its gallery was calibrated would have to assume
        the worst and warn on every determination.
        """
        clone = Gallery(
            vectors=self.vectors,
            species=self.species,
            captures=np.full(len(self.captures), "unpublished"),
            classes=list(self.classes),
            backbone=self.backbone,
            neighbours=self.neighbours,
            temperature=self.temperature,
            similarity_floor=self.similarity_floor,
            calibrated=self.calibrated,
            reliable=self.reliable,
            metrics=dict(self.metrics),
        )
        clone.reliable = self.reliable
        clone.metrics["published"] = True
        return clone

    # -- querying ------------------------------------------------------
    @property
    def keys(self) -> np.ndarray:
        """One id per capture, unique across species."""
        return np.array([f"{s}/{c}" for s, c in zip(self.species, self.captures)])

    def score(self, vectors: np.ndarray, exclude_keys: Sequence[str] | None = None):
        """Score query vectors against the gallery.

        Returns (scores, best_similarity). `exclude_keys` masks out the capture
        each query came from, which is how leave-one-capture-out is done: a
        photograph must never be identified by another photograph of the same
        individual.
        """
        if self.vectors.size == 0:
            raise MatcherError("The gallery is empty.")
        similarities = vectors @ self.vectors.T

        if exclude_keys is not None:
            keys = self.keys
            for row, key in enumerate(exclude_keys):
                similarities[row, keys == key] = -np.inf

        best = similarities.max(axis=1)
        scores = species_scores(similarities, self.species, self.classes, self.neighbours)
        return scores, np.where(np.isfinite(best), best, ABSENT)
