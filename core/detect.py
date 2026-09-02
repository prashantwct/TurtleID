"""
Find the animal in a photograph and crop to it.

The pipeline camera-trap work settled on years ago, and what Addax AI / EcoAssist
does: a class-agnostic detector finds the animal, everything else is thrown away,
and only the crop reaches the classifier. Two reasons it matters here.

WHY CROPPING IS NOT COSMETIC
----------------------------
A generic image feature encodes the whole frame. Given a scanned reference plate
and a phone photograph taken on a riverbank, the strongest thing separating them
is not the animal — it is the paper, the layout, the grass, the hands, the light.
This project measured that directly: plates sat at 0.56 mean similarity to each
other, field photographs at 0.51 to each other, and the two groups at 0.42 across
the gap, so a field photograph matched other field photographs whatever animal
was in them. Cropping to the animal removes most of what was doing the
separating.

It is a plausible fix rather than a proven one. The honest test is the held-out
accuracy `training/build_gallery.py` prints, against chance.

THE INVARIANT
-------------
Whatever is done to a gallery photograph must be done to a query. A gallery of
crops searched with whole frames is worse than either alone, because now the
query differs from every entry in exactly the way this module exists to remove.
`Gallery.cropped` records which it was, and `core/inference.py` refuses the
mismatch rather than quietly serving it.

GETTING A DETECTOR
------------------
Nothing here ships one. Any Ultralytics-loadable model works; what this needs is
a detector that finds animals rather than one trained on 80 everyday objects —
COCO has no turtle class. MegaDetector is the usual choice, is free, and is
exactly what the camera-trap pipelines above use:

    https://github.com/agentmorris/MegaDetector

Put the weights at `models/chelonid_det.pt` and both the gallery build and the
app pick them up.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from config import DET_CONF_THRESHOLD, DET_CROP_PADDING

logger = logging.getLogger(__name__)

# Below this many pixels a side, a crop is not worth having: it is a detection
# on a speck, and the full frame carries more.
MIN_CROP_EDGE = 16


def load_detector(path) -> Any | None:
    """An Ultralytics detector, or None if there is none to load.

    Never raises. A detector is an improvement, not a dependency; a broken one
    should degrade to whole-frame matching rather than take the app down.
    """
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return None
    try:
        from ultralytics import YOLO

        detector = YOLO(str(path))
        logger.info("Detector loaded from %s", path)
        return detector
    except Exception as exc:  # noqa: BLE001 - optional stage, never fatal
        logger.warning("Detector at %s unavailable (%s); using whole frames.", path, exc)
        return None


def crop_to_animal(image, detector) -> tuple[Any, float | None]:
    """Crop to the highest-confidence detection. Returns (image, confidence).

    Falls back to the frame it was given — unchanged, with a confidence of None
    — whenever there is nothing to crop to. The caller cannot tell a failed
    detection from an absent detector, and does not need to: both mean this
    photograph is being matched whole.
    """
    if detector is None:
        return image, None
    try:
        result = detector.predict(image, conf=DET_CONF_THRESHOLD, verbose=False)[0]
        if len(result.boxes) == 0:
            return image, None

        confidences = result.boxes.conf.cpu().numpy()
        best = int(np.argmax(confidences))
        x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy()[best]

        width, height = x2 - x1, y2 - y1
        pad_x, pad_y = width * DET_CROP_PADDING, height * DET_CROP_PADDING
        frame = np.asarray(image)
        rows, columns = frame.shape[:2]

        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(columns, int(x2 + pad_x))
        y2 = min(rows, int(y2 + pad_y))
        if x2 - x1 < MIN_CROP_EDGE or y2 - y1 < MIN_CROP_EDGE:
            return image, float(confidences[best])

        from PIL import Image

        return Image.fromarray(frame[y1:y2, x1:x2]), float(confidences[best])
    except Exception as exc:  # noqa: BLE001 - a bad detection must not lose the photograph
        logger.warning("Detection failed (%s); using the whole frame.", exc)
        return image, None
