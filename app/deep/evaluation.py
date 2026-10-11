"""Honest accuracy measurement for the deep recognition engine.

Two complementary measurements:

* :func:`evaluate_open_set` — offline, on the user's own labels.  A share of
  the persons is held out *entirely* (they play strangers: any name they get
  is a false accept), the remaining persons are split into folds that keep
  photo sessions together (no burst-shot leakage), and the full engine —
  training, calibration and every gate — is retrained per fold.  Results are
  broken down by face size and quality, because small / poor faces are where
  recognition fails first.

* :func:`feedback_precision` — online, from the review tab: of the automatic
  assignments the user already judged, how many were confirmed versus
  corrected / reverted.  This is the precision the user actually experiences.

Both are cheap to call from tests and from ``tools/deep_eval.py``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.deep.dataset import ProbeMeta, TrainingDataset

log = logging.getLogger(__name__)


@dataclass
class BucketStats:
    """Decision counts for one slice of the evaluation faces."""

    n_known: int = 0          # held-in faces of known persons
    n_correct: int = 0        # … assigned to the right person
    n_wrong: int = 0          # … assigned to a wrong person
    n_strangers: int = 0      # faces of held-out persons
    n_false_accept: int = 0   # … that still got a name

    @property
    def coverage(self) -> Optional[float]:
        return self.n_correct / self.n_known if self.n_known else None

    @property
    def known_precision(self) -> Optional[float]:
        decided = self.n_correct + self.n_wrong
        return self.n_correct / decided if decided else None

    @property
    def false_accept_rate(self) -> Optional[float]:
        return self.n_false_accept / self.n_strangers if self.n_strangers else None

    def precision_at(self, unknown_ratio: float) -> Optional[float]:
        """Precision expected when *unknown_ratio* of probes are strangers."""
        if not self.n_known:
            return None
        known_share = 1.0 - unknown_ratio
        tp = known_share * self.n_correct / self.n_known
        fp = known_share * self.n_wrong / self.n_known
        if self.n_strangers:
            fp += unknown_ratio * self.n_false_accept / self.n_strangers
        return tp / (tp + fp) if (tp + fp) > 0 else None

    def add(self, other: "BucketStats") -> None:
        for name in ("n_known", "n_correct", "n_wrong", "n_strangers", "n_false_accept"):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def to_dict(self, unknown_ratio: float) -> Dict:
        data = asdict(self)
        data.update({
            "coverage": self.coverage,
            "known_precision": self.known_precision,
            "false_accept_rate": self.false_accept_rate,
            "precision": self.precision_at(unknown_ratio),
        })
        return data


@dataclass
class EvaluationReport:
    """Result of :func:`evaluate_open_set`."""

    n_folds: int = 0
    n_known_persons: int = 0
    n_holdout_persons: int = 0
    unknown_ratio: float = 0.5
    overall: BucketStats = field(default_factory=BucketStats)
    by_size: Dict[str, BucketStats] = field(default_factory=dict)
    by_quality: Dict[str, BucketStats] = field(default_factory=dict)
    rejection_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        r = self.unknown_ratio
        return {
            "n_folds": self.n_folds,
            "n_known_persons": self.n_known_persons,
            "n_holdout_persons": self.n_holdout_persons,
            "unknown_ratio": r,
            "overall": self.overall.to_dict(r),
            "by_size": {k: v.to_dict(r) for k, v in self.by_size.items()},
            "by_quality": {k: v.to_dict(r) for k, v in self.by_quality.items()},
            "rejection_reasons": dict(self.rejection_reasons),
        }


def size_bucket(probe: ProbeMeta) -> str:
    if probe.size is None:
        return "unknown"
    if probe.size < 48:
        return "small(<48px)"
    if probe.size < 96:
        return "medium(48-96px)"
    return "large(>=96px)"


def quality_bucket(probe: ProbeMeta) -> str:
    if probe.quality is None:
        return "unknown"
    return "low(<0.5)" if probe.quality < 0.5 else "good(>=0.5)"


def _holdout_persons(dataset: TrainingDataset, fraction: float, seed: int) -> List[int]:
    counts = dataset.examples_per_person()
    persons = sorted(counts)
    if len(persons) < 3 or fraction <= 0:
        return []
    n_holdout = max(1, int(round(fraction * len(persons))))
    n_holdout = min(n_holdout, len(persons) - 2)  # keep ≥2 known persons
    rng = np.random.default_rng(seed)
    return sorted(int(p) for p in rng.choice(persons, size=n_holdout, replace=False))


def evaluate_open_set(
    dataset: TrainingDataset,
    config,
    *,
    n_folds: int = 3,
    holdout_fraction: float = 0.2,
    unknown_ratio: float = 0.5,
    seed: int = 0,
    progress_cb=None,
) -> EvaluationReport:
    """Retrain the full engine per fold and grade it like real use.

    Every decision gate is exercised exactly as in production, so the numbers
    reflect the whole engine, not just the network's top-1 accuracy.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    from app.deep.classifier import DeepFaceClassifier

    report = EvaluationReport(unknown_ratio=unknown_ratio)
    if dataset.n_examples == 0:
        return report
    holdout = set(_holdout_persons(dataset, holdout_fraction, seed))
    labels = dataset.labels
    known_mask = np.asarray([int(lab) not in holdout for lab in labels])
    stranger_idx = np.flatnonzero(~known_mask)
    known_idx = np.flatnonzero(known_mask)
    report.n_holdout_persons = len(holdout)
    report.n_known_persons = len(set(labels[known_idx].tolist()))
    if report.n_known_persons < 1 or known_idx.size < 2:
        return report

    known = dataset.subset(known_mask)
    counts = known.examples_per_person()
    groups = known.groups()
    folds_wanted = max(2, min(n_folds, min(counts.values()), len(np.unique(groups))))
    if min(counts.values()) < 2 or len(np.unique(groups)) < 2:
        return report
    splitter = StratifiedGroupKFold(n_splits=folds_wanted, shuffle=True, random_state=seed)
    try:
        folds = list(splitter.split(known.embeddings, known.labels, groups))
    except ValueError as exc:
        log.warning("Evaluation folds unavailable: %s", exc)
        return report
    report.n_folds = len(folds)

    for fold_no, (train_idx, test_idx) in enumerate(folds):
        if progress_cb is not None:
            progress_cb(fold_no + 1, len(folds), f"evaluation fold {fold_no + 1}")
        mask = np.zeros(known.n_examples, dtype=bool)
        mask[train_idx] = True
        train_set = known.subset(mask)
        clf = DeepFaceClassifier(config)
        clf.train(train_set)
        if not clf.is_trained:
            continue
        trained = set(clf.person_ids)
        for i in test_idx:
            probe = known.probe(int(i))
            pred = clf.predict(known.embeddings[i], probe)
            truth = int(known.labels[i])
            if truth not in trained:
                # The person only appeared in the test fold → a stranger here.
                _record(report, probe, BucketStats(
                    n_strangers=1, n_false_accept=int(pred.person_id is not None)
                ))
                continue
            stats = BucketStats(n_known=1)
            if pred.person_id is None:
                report.rejection_reasons[pred.reason] = (
                    report.rejection_reasons.get(pred.reason, 0) + 1
                )
            elif pred.person_id == truth:
                stats.n_correct = 1
            else:
                stats.n_wrong = 1
            _record(report, probe, stats)
        # Held-out persons: every one of their faces is a stranger.
        for i in stranger_idx:
            probe = dataset.probe(int(i))
            pred = clf.predict(dataset.embeddings[i], probe)
            _record(
                report, probe,
                BucketStats(n_strangers=1, n_false_accept=int(pred.person_id is not None)),
            )
    return report


def _record(report: EvaluationReport, probe: ProbeMeta, stats: BucketStats) -> None:
    report.overall.add(stats)
    report.by_size.setdefault(size_bucket(probe), BucketStats()).add(stats)
    report.by_quality.setdefault(quality_bucket(probe), BucketStats()).add(stats)


def feedback_precision(session, window: int = 300) -> Tuple[Optional[float], int]:
    """Precision of the last *window* reviewed automatic assignments.

    Confirmed = correct; corrected / reverted = wrong.  "Near" suggestions are
    not automatic assignments and are ignored.  Returns ``(precision, n)``;
    precision is None when nothing was reviewed yet.
    """
    from app.db.models import (
        AUTO_ASSIGN_STATUS_CONFIRMED,
        AUTO_ASSIGN_STATUS_CORRECTED,
        AUTO_ASSIGN_STATUS_REVERTED,
        AutoAssignment,
    )

    rows = (
        session.query(AutoAssignment.status)
        .filter(
            AutoAssignment.status.in_((
                AUTO_ASSIGN_STATUS_CONFIRMED,
                AUTO_ASSIGN_STATUS_CORRECTED,
                AUTO_ASSIGN_STATUS_REVERTED,
            ))
        )
        # A near-match suggestion that was accepted/rejected is stored with
        # the same statuses but never had a score-based assignment: those rows
        # carry a "near" origin in decision_json — excluded below.
        .filter(
            (AutoAssignment.decision_json.is_(None))
            | (~AutoAssignment.decision_json.like('%"origin": "near"%'))
        )
        .order_by(AutoAssignment.decided_at.desc(), AutoAssignment.id.desc())
        .limit(max(1, int(window)))
        .all()
    )
    n = len(rows)
    if n == 0:
        return None, 0
    good = sum(1 for (status,) in rows if status == AUTO_ASSIGN_STATUS_CONFIRMED)
    return good / n, n
