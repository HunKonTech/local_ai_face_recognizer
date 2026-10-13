"""Learned "not a face" classifier on face embeddings.

The detectors' false positives — carpet / wallpaper patterns, foliage, half
faces cut by the photo edge — share one trait: their embeddings land far from
every real face and close to each other.  The user's "Not a face" marks are
negative examples, human-confirmed faces are positives; a regularised logistic
regression on the L2-normalised embeddings then generalises to *new* patterns,
which the exact-match ignore list cannot do.

Precision first — a real face must never be hidden:

* the decision threshold is calibrated out-of-fold so at most
  ``max_positive_fpr`` of the confirmed real faces would be hidden, and is
  never below ``min_probability``;
* a candidate is only flagged when, in addition, its nearest labelled
  neighbour is a non-face — so the face of a stranger (a real face, just not
  in the positives) stays closer to the positives than to the patterns;
* the model stays off until enough examples of both kinds exist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Candidates are scored in chunks so the nearest-neighbour matrices stay small.
_CHUNK = 1024


def normalise_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row (zero rows stay zero)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


@dataclass
class NonFaceModel:
    """A trained non-face classifier plus its calibrated decision rule."""

    classifier: object
    threshold: float
    positives: np.ndarray          # normalised, for the nearest-neighbour check
    negatives: np.ndarray
    cv_negative_recall: float      # held-out share of non-faces it catches
    cv_positive_fpr: float         # held-out share of real faces it would hide

    @property
    def dim(self) -> int:
        return int(self.positives.shape[1])

    def non_face_mask(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(is_non_face, probability)`` for each row of *embeddings*."""
        x = normalise_rows(embeddings)
        prob = self.classifier.predict_proba(x)[:, 1]
        mask = prob >= self.threshold
        for start in range(0, len(x), _CHUNK):
            chunk = x[start:start + _CHUNK]
            near_neg = (chunk @ self.negatives.T).max(axis=1)
            near_pos = (chunk @ self.positives.T).max(axis=1)
            mask[start:start + _CHUNK] &= near_neg > near_pos
        return mask, prob


def _make_classifier(seed: int):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        C=10.0, class_weight="balanced", max_iter=5000, random_state=seed
    )


def train_non_face_model(
    positives: np.ndarray,
    negatives: np.ndarray,
    *,
    min_negatives: int = 5,
    min_positives: int = 30,
    min_probability: float = 0.90,
    max_positive_fpr: float = 0.005,
    max_positives: int = 5000,
    seed: int = 0,
) -> Optional[NonFaceModel]:
    """Train and calibrate the classifier; ``None`` when data is insufficient."""
    from sklearn.model_selection import StratifiedKFold

    if len(negatives) < min_negatives or len(positives) < min_positives:
        return None
    rng = np.random.default_rng(seed)
    if len(positives) > max_positives:
        positives = positives[rng.choice(len(positives), max_positives, replace=False)]
    pos = normalise_rows(positives)
    neg = normalise_rows(negatives)
    x = np.vstack([pos, neg])
    y = np.concatenate([np.zeros(len(pos), int), np.ones(len(neg), int)])

    # Out-of-fold probabilities calibrate the threshold on data the model did
    # not see — in-sample scores would make it far too permissive.
    oof = np.zeros(len(x), dtype=np.float64)
    folds = StratifiedKFold(n_splits=min(5, len(neg)), shuffle=True, random_state=seed)
    for train_idx, test_idx in folds.split(x, y):
        clf = _make_classifier(seed).fit(x[train_idx], y[train_idx])
        oof[test_idx] = clf.predict_proba(x[test_idx])[:, 1]

    oof_pos, oof_neg = oof[y == 0], oof[y == 1]
    threshold = max(
        float(min_probability),
        float(np.quantile(oof_pos, 1.0 - max_positive_fpr)) + 1e-6,
    )
    model = NonFaceModel(
        classifier=_make_classifier(seed).fit(x, y),
        threshold=threshold,
        positives=pos,
        negatives=neg,
        cv_negative_recall=float(np.mean(oof_neg >= threshold)),
        cv_positive_fpr=float(np.mean(oof_pos >= threshold)),
    )
    log.info(
        "Non-face model: %d positive(s), %d negative(s), threshold=%.3f, "
        "held-out recall=%.2f, held-out real-face FPR=%.4f",
        len(pos), len(neg), threshold, model.cv_negative_recall, model.cv_positive_fpr,
    )
    return model
