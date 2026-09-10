"""Shared geometry / embedding helpers for same-image duplicate detection.

Several passes need the same question answered — "are these two boxes on one
image the *same physical face* detected twice, rather than two real faces?" —
with the same safety valve: a second, genuinely separate appearance of a person
(a mirror, a portrait hanging on the wall, a photo-in-photo) must never be
collapsed.  The rule here mirrors the proven one in
:class:`app.services.overlap_resolution_service.OverlapResolutionService`:

* the two boxes must **geometrically overlap** (IoU or containment) — a
  non-overlapping second box is always a real, separate face;
* when both faces have embeddings, a moderate overlap additionally requires the
  embeddings to agree (guards two different people photographed cheek-to-cheek);
  above ``dup_hard_iou_threshold`` the boxes are the same spot and the pair is
  the same face regardless of embedding.
"""

from __future__ import annotations

from typing import Optional, Protocol, Tuple

import numpy as np

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


class DupThresholds(Protocol):
    """Duck-typed config carrying the duplicate thresholds."""

    dup_iou_threshold: float
    dup_containment_threshold: float
    dup_embedding_guard: float
    dup_hard_iou_threshold: float


def bbox_iou_containment(a: BBox, b: BBox) -> Tuple[float, float]:
    """Return ``(iou, containment)`` for two ``(x, y, w, h)`` boxes.

    ``containment`` is ``intersection / area(smaller box)`` — high when one box
    sits nested inside a larger one (a case where IoU stays low).
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0, 0.0
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    smaller = min(area_a, area_b)
    containment = inter / smaller if smaller > 0 else 0.0
    return iou, containment


def unit(embedding: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return the L2-normalised vector, or ``None`` for a missing/zero vector."""
    if embedding is None:
        return None
    vec = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    return vec / norm


def is_same_physical_face(
    a_bbox: BBox,
    a_vec: Optional[np.ndarray],
    b_bbox: BBox,
    b_vec: Optional[np.ndarray],
    cfg: DupThresholds,
) -> bool:
    """True when the two boxes are one physical face detected twice on an image.

    ``a_vec`` / ``b_vec`` are expected to be unit vectors (see :func:`unit`) or
    ``None``.  Geometric overlap is required; a non-overlapping second box is a
    genuine separate appearance (mirror, wall portrait) and is never collapsed.
    """
    iou, containment = bbox_iou_containment(a_bbox, b_bbox)
    if iou < cfg.dup_iou_threshold and containment < cfg.dup_containment_threshold:
        return False
    if a_vec is not None and b_vec is not None:
        if (
            float(np.dot(a_vec, b_vec)) < cfg.dup_embedding_guard
            and iou < cfg.dup_hard_iou_threshold
        ):
            return False
    return True
