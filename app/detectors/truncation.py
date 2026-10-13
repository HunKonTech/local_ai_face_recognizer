"""Truncated-face filter — drops "half faces" cut off by the photo edge.

A person standing at the edge of a photo is often sliced in half by the frame.
The detectors still fire on the visible half (and YuNet / SCRFD regress a box
that extends past the border), but the resulting crop is a half face: it
embeds poorly, clusters into junk "Unknown" groups and is never useful for
recognition.

Two signals, both from the detector's *raw* output:

* ``Detection.truncated_fraction`` — the share of the raw box area that
  :meth:`Detection.clamp` cut off at the image border;
* the 5 landmarks (kept unclamped) — an eye landmark outside the image means
  that eye is not in the photo at all.

Fails open: landmark-less detectors (Coral / Caffe / Haar) are judged by the
box fraction alone, and a detector that clips its own boxes to the image
(fraction always 0) is never penalised.
"""

from __future__ import annotations

import math
from typing import Optional

from app.detectors.base import Detection

# Canonical YuNet / ArcFace order: right eye, left eye, nose, mouth corners.
_EYE_INDICES = (0, 1)


def truncation_reject_reason(
    det: Detection,
    img_w: int,
    img_h: int,
    max_outside_fraction: float,
) -> Optional[str]:
    """Return why *det* is a truncated (half) face, or ``None`` to keep it."""
    if det.truncated_fraction > max_outside_fraction:
        return "box_outside_image"
    lms = det.landmarks
    if lms is None or len(lms) != 5:
        return None
    for idx in _EYE_INDICES:
        px, py = lms[idx][0], lms[idx][1]
        if not (math.isfinite(px) and math.isfinite(py)):
            return None
        if not (0.0 <= px < img_w and 0.0 <= py < img_h):
            return "eye_outside_image"
    return None
