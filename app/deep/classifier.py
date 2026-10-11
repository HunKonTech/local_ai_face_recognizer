"""DeepFaceClassifier — neural-network identity classifier with open-set rejection.

The classifier is the learning core of the new recognition path:

* **Model** — an ensemble of feed-forward neural networks (multi-layer
  perceptrons) trained on the deep CNN face embeddings of every labeled face.
  Ensemble averaging smooths out individual network quirks.  When enough
  "nobody we know" faces exist, the networks also learn an explicit *unknown*
  output, so strangers stop being forced onto the closest known person.

* **Continual learning** — the ensemble is retrained from scratch on every
  run with *all* trusted labels, so each newly confirmed face immediately
  improves the model.  A fingerprint of the labeled data lets callers skip
  retraining when nothing changed.

* **Calibrated decisions** — cross-validation keeps whole photo sessions
  together (no burst-shot leakage) and simulates strangers by hiding each
  test face's true person.  From those out-of-fold decisions a small
  logistic *meta-classifier* learns P(correct) from the network probability,
  the embedding similarities, the k-NN agreement and the face quality; its
  threshold is set where the measured precision reaches ``target_precision``.
  Real mistakes the user corrected are added as weighted hard negatives.

* **Open-set rejection** — independent gates keep strangers / non-faces from
  being force-assigned:

  1. *outlier gate* — the embedding must resemble at least one known training
     example at all;
  2. *unknown gate* — the explicit unknown output must not win;
  3. *calibrated gate* — P(correct) (or, without a meta-model, the ensemble
     probability) must exceed its threshold, with a clear margin;
  4. *prototype gates* — cosine-close to a real example of the winner, and
     clearly closer to the winner than to anyone else;
  5. *k-NN gate* — most nearest training faces must belong to the winner, so
     one mislabeled training face cannot attract its lookalikes.

* **Training-data hygiene** — labeled faces that look far more like another
  person are flagged as *suspicious* (excluded until the user decides), and
  near-identical burst shots collapse to one example.

* **Accuracy over speed** — training uses every CPU core (ensemble members and
  calibration folds run in parallel) and is allowed to take its time.
"""

from __future__ import annotations

import functools
import logging
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.deep.dataset import ProbeMeta, TrainingDataset

log = logging.getLogger(__name__)

_MODEL_FILE_VERSION = 2

# Training label of the explicit "nobody we know" output.
UNKNOWN_LABEL = -1

# Meta-classifier input features, in order.
FEATURE_NAMES = (
    "p_top",        # ensemble probability of the candidate
    "p_margin",     # gap to the runner-up person
    "p_unknown",    # probability of the "unknown" output
    "sim1",         # best cosine similarity to the candidate's examples
    "simk",         # mean of the candidate's top-3 similarities
    "sim_center",   # best similarity to the candidate's sub-centers
    "sim_margin",   # sim1 minus the best similarity to anyone else
    "knn_agree",    # share of the k nearest training faces that are the candidate
    "quality",      # face quality score
    "size",         # log face size
    "consistency",  # flip-TTA embedding agreement
)


@functools.lru_cache(maxsize=1)
def _default_config():
    from app.config import DeepRecognitionConfig

    return DeepRecognitionConfig()


@dataclass
class DeepPrediction:
    """Result of classifying one embedding."""

    person_id: Optional[int]      # None → rejected (stays unknown)
    person_name: Optional[str]    # winner's name (also set for rejections)
    score: float                  # confidence [0, 1] (calibrated P(correct) with a meta-model)
    probability: float            # raw ensemble probability of the winner
    similarity: float             # best cosine similarity to the winner's examples
    margin: float                 # probability gap to the runner-up person
    reason: str                   # "assigned" | rejection reason
    # The best candidate even when rejected (near-match suggestions, voting).
    candidate_id: Optional[int] = None
    knn_agreement: float = 0.0
    p_correct: Optional[float] = None
    # Best similarity to the winner minus the best similarity to anyone else.
    sim_margin: float = 0.0


@dataclass
class DeepTrainingReport:
    """Diagnostics from one training run."""

    n_persons: int = 0
    n_examples: int = 0
    n_augmented: int = 0
    validation_accuracy: Optional[float] = None
    duration_seconds: float = 0.0
    fingerprint: str = ""
    mode: str = "ensemble"        # "ensemble" | "prototype" | "empty"
    n_negatives: int = 0
    n_unknown: int = 0
    n_deduplicated: int = 0
    # Label-noise filter: (face_id, own person, lookalike person, own sim, other sim)
    suspicious: List[Tuple[int, int, int, float, float]] = field(default_factory=list)
    # Calibration diagnostics (JSON-safe).
    metrics: Dict = field(default_factory=dict)

    @property
    def suspicious_face_ids(self) -> List[int]:
        return [row[0] for row in self.suspicious]


@dataclass
class _PrototypeIndex:
    """Training embeddings sorted by person, with per-person sub-centers."""

    matrix: np.ndarray
    labels: np.ndarray
    manual: np.ndarray
    classes: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    centers: List[np.ndarray]

    @classmethod
    def build(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        manual: Optional[np.ndarray] = None,
        faces_per_center: int = 8,
        max_centers: int = 4,
    ) -> "_PrototypeIndex":
        order = np.argsort(y, kind="stable")
        x, y = x[order], y[order]
        manual_sorted = (
            np.asarray(manual, dtype=bool)[order]
            if manual is not None and len(manual) == len(order)
            else np.zeros(len(order), dtype=bool)
        )
        classes, starts, counts = np.unique(y, return_index=True, return_counts=True)
        centers = [
            _sub_centers(x[s:s + n], faces_per_center, max_centers)
            for s, n in zip(starts, counts)
        ]
        return cls(
            matrix=x.astype(np.float32),
            labels=y.astype(np.int64),
            manual=manual_sorted,
            classes=classes.astype(np.int64),
            starts=starts.astype(np.int64),
            counts=counts.astype(np.int64),
            centers=centers,
        )

    @property
    def n_classes(self) -> int:
        return int(self.classes.shape[0])

    def class_index(self, label: int) -> Optional[int]:
        pos = int(np.searchsorted(self.classes, label))
        if pos < self.n_classes and int(self.classes[pos]) == int(label):
            return pos
        return None

    def class_max(self, sims: np.ndarray) -> np.ndarray:
        if sims.size == 0:
            return np.zeros(0, dtype=np.float64)
        return np.maximum.reduceat(sims, self.starts).astype(np.float64)

    def class_sims(self, sims: np.ndarray, c: int) -> np.ndarray:
        s = int(self.starts[c])
        return sims[s:s + int(self.counts[c])]

    def has_manual_anchor(self, sims: np.ndarray, c: int, threshold: float) -> bool:
        s, n = int(self.starts[c]), int(self.counts[c])
        mask = self.manual[s:s + n]
        if not mask.any():
            return False
        return float(np.max(sims[s:s + n][mask])) >= threshold

    def knn_agreement(
        self, sims: np.ndarray, c: int, k: int, exclude: Optional[int] = None
    ) -> float:
        """Share of the k nearest training faces belonging to class *c*.

        ``k`` shrinks to the class size so a person with one example can
        still reach full agreement.
        """
        k_eff = max(1, min(int(k), int(self.counts[c])))
        work = sims
        if exclude is not None:
            work = sims.copy()
            s = int(self.starts[exclude])
            work[s:s + int(self.counts[exclude])] = -np.inf
        if k_eff >= work.shape[0]:
            top = np.arange(work.shape[0])
        else:
            top = np.argpartition(-work, k_eff - 1)[:k_eff]
        return float(np.mean(self.labels[top] == self.classes[c]))


def _sub_centers(rows: np.ndarray, faces_per_center: int, max_centers: int) -> np.ndarray:
    """1–``max_centers`` unit-norm centers of one person (age / look modes)."""
    k = max(1, min(int(max_centers), rows.shape[0] // max(1, int(faces_per_center))))
    if k <= 1 or rows.shape[0] < 2 * k:
        center = rows.mean(axis=0, keepdims=True)
    else:
        from sklearn.cluster import KMeans

        center = KMeans(n_clusters=k, n_init=3, random_state=0).fit(rows).cluster_centers_
    norms = np.linalg.norm(center, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return (center / norms).astype(np.float32)


@dataclass
class _TrainedState:
    """Everything predict() needs, picklable as one unit."""

    mode: str = "empty"
    classes: List[int] = field(default_factory=list)
    person_names: Dict[int, str] = field(default_factory=dict)
    members: List[object] = field(default_factory=list)
    index: Optional[_PrototypeIndex] = None
    has_unknown_class: bool = False
    prob_thresholds: Dict[int, float] = field(default_factory=dict)
    sim_floors: Dict[int, float] = field(default_factory=dict)
    meta: Optional[object] = None          # sklearn Pipeline → P(correct)
    meta_threshold: Optional[float] = None
    fingerprint: str = ""
    trained_at: float = 0.0
    report: Optional[DeepTrainingReport] = None

    # Read-only views kept for callers/tests written against the old layout.
    @property
    def train_matrix(self) -> np.ndarray:
        return self.index.matrix if self.index is not None else np.zeros((0, 0), np.float32)

    @property
    def train_labels(self) -> np.ndarray:
        return self.index.labels if self.index is not None else np.zeros(0, np.int64)


@dataclass
class _Features:
    """Feature vector of one (probe, candidate person) decision."""

    values: Dict[str, float]
    sim1: float
    runner_sim: float
    p_top: float
    p_margin: float
    p_unknown: float
    knn: float

    def vector(self) -> np.ndarray:
        return np.asarray([[self.values[n] for n in FEATURE_NAMES]], dtype=np.float64)


class DeepFaceClassifier:
    """Ensemble MLP classifier over face embeddings with open-set rejection."""

    def __init__(self, config) -> None:
        # ``config`` is an app.config.DeepRecognitionConfig; kept duck-typed so
        # tests can pass a small stub (missing fields fall back to defaults).
        self._config = config
        self._state = _TrainedState()
        # Extra strictness from the feedback loop (raises every threshold).
        self._strictness = 0.0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._state.mode in ("ensemble", "prototype")

    @property
    def person_ids(self) -> tuple[int, ...]:
        """Person IDs this model was trained to recognise (its class labels).

        Used to detect a *stale* model — one whose people no longer exist in the
        current database (e.g. after the DB was reset but the model kept).
        """
        return tuple(int(c) for c in self._state.classes)

    @property
    def fingerprint(self) -> str:
        return self._state.fingerprint

    @property
    def trained_hidden_layers(self) -> Optional[tuple]:
        """Hidden-layer layout of the persisted ensemble (None without MLPs)."""
        if not self._state.members:
            return None
        sizes = getattr(self._state.members[0], "hidden_layer_sizes", None)
        if sizes is None:
            return None
        if isinstance(sizes, (list, tuple)):
            return tuple(int(s) for s in sizes)
        return (int(sizes),)

    @property
    def report(self) -> Optional[DeepTrainingReport]:
        return self._state.report

    @property
    def has_meta_model(self) -> bool:
        return self._state.meta is not None and self._state.meta_threshold is not None

    @property
    def meta_threshold(self) -> Optional[float]:
        return self._state.meta_threshold if self.has_meta_model else None

    @property
    def strictness(self) -> float:
        return self._strictness

    def set_strictness(self, value: float) -> None:
        """Raise every decision threshold by *value* (feedback loop)."""
        self._strictness = max(0.0, float(value))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        dataset: TrainingDataset,
        progress_cb=None,
    ) -> DeepTrainingReport:
        """(Re)train the model from *dataset*; returns a diagnostics report.

        progress_cb: optional ``(current: int, total: int, detail: str)``.
        """
        started = time.monotonic()
        report = DeepTrainingReport(
            n_persons=dataset.n_persons,
            n_examples=dataset.n_examples,
            fingerprint=dataset.fingerprint(),
            n_negatives=dataset.n_negatives,
            n_unknown=dataset.n_unknown,
        )

        if dataset.n_examples == 0 or dataset.n_persons == 0:
            self._state = _TrainedState(mode="empty", report=report)
            report.mode = "empty"
            log.info("Deep training skipped: no labeled examples")
            return report

        # --- Training-data hygiene ---------------------------------------
        keep = np.ones(dataset.n_examples, dtype=bool)
        report.suspicious = self._find_suspicious(dataset)
        flagged = set(report.suspicious_face_ids)
        if flagged:
            keep &= np.asarray([fid not in flagged for fid in dataset.face_ids])
        dedup = self._dedup_mask(dataset, keep)
        report.n_deduplicated = int(np.sum(keep & ~dedup))
        clean = dataset.subset(keep & dedup)
        if clean.n_examples == 0:  # defensive: never train on nothing
            clean = dataset

        manual_mask = (
            np.asarray(clean.manual_flags, dtype=bool) if clean.manual_flags else None
        )
        index = _PrototypeIndex.build(
            clean.embeddings,
            clean.labels,
            manual_mask,
            faces_per_center=int(self._cfg("faces_per_sub_center")),
            max_centers=int(self._cfg("max_sub_centers")),
        )
        state = _TrainedState(
            mode="prototype",
            classes=[int(c) for c in index.classes],
            person_names=dict(clean.person_names),
            index=index,
            fingerprint=report.fingerprint,
            trained_at=time.time(),
        )
        state.sim_floors = self._calibrate_sim_floors(clean)

        # A discriminative network needs a real cohort to produce meaningful
        # probabilities: with a handful of people (or examples) the softmax is
        # grossly overconfident for the dominant person and open-set rejection
        # degenerates — stay in pure prototype matching until there is data.
        enough_for_ensemble = (
            clean.n_persons >= int(self._cfg("min_persons_for_ensemble", 4))
            and clean.n_examples >= int(self._cfg("min_examples_for_ensemble", 30))
        )
        if enough_for_ensemble:
            x_aug, y_aug, n_synthetic = self._augment(clean.embeddings, clean.labels)
            report.n_augmented = n_synthetic
            unknown_x = self._unknown_rows(clean)
            state.has_unknown_class = unknown_x is not None
            if unknown_x is not None:
                x_aug = np.vstack([x_aug, unknown_x])
                y_aug = np.concatenate(
                    [y_aug, np.full(unknown_x.shape[0], UNKNOWN_LABEL, dtype=y_aug.dtype)]
                )

            total_steps = self._ensemble_size() + self._n_folds(clean) + 2
            step = [0]

            def tick(detail: str) -> None:
                step[0] += 1
                if progress_cb is not None:
                    progress_cb(step[0], total_steps, detail)

            state.members = self._fit_ensemble(x_aug, y_aug, tick)
            state.mode = "ensemble"
            report.mode = "ensemble"
            # Calibration below needs the gates of the final model.
            self._state = state

            cv = self._cross_validate(clean, unknown_x, tick)
            state.prob_thresholds = cv["thresholds"]
            report.validation_accuracy = cv["accuracy"]
            tick("calibrating confidence")
            self._fit_meta(state, clean, cv, report)
            tick("finalising")
        else:
            # Too few people/examples for a discriminative network — pure
            # prototype matching with the hard similarity floor is safer.
            report.mode = "prototype"
            if progress_cb is not None:
                progress_cb(1, 1, "prototype mode")

        report.duration_seconds = time.monotonic() - started
        state.report = report
        self._state = state
        log.info(
            "Deep training done: mode=%s persons=%d examples=%d (+%d synthetic, "
            "-%d duplicates, %d suspicious) negatives=%d unknown=%d val_acc=%s "
            "meta_thr=%s in %.1fs",
            report.mode, report.n_persons, report.n_examples, report.n_augmented,
            report.n_deduplicated, len(report.suspicious), report.n_negatives,
            report.n_unknown,
            f"{report.validation_accuracy:.3f}" if report.validation_accuracy is not None else "n/a",
            f"{state.meta_threshold:.3f}" if state.meta_threshold is not None else "n/a",
            report.duration_seconds,
        )
        return report

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        embedding: Optional[np.ndarray],
        probe: Optional[ProbeMeta] = None,
        relief: float = 0.0,
    ) -> DeepPrediction:
        """Classify one embedding; rejected predictions carry person_id=None.

        *probe* carries the face's quality / size / TTA consistency — used by
        the calibrated meta-classifier (neutral defaults when omitted).
        *relief* lowers only the calibrated / probability threshold — used
        when independent context evidence (unknown-group vote, same-day
        event) supports the candidate; every other gate still applies.
        """
        rejected = DeepPrediction(
            person_id=None, person_name=None,
            score=0.0, probability=0.0, similarity=0.0, margin=0.0,
            reason="untrained",
        )
        if not self.is_trained:
            return rejected
        vec = self._unit(embedding)
        if vec is None or self._dim_mismatch(vec):
            rejected.reason = "no_embedding"
            return rejected
        return self._decide(vec, probe, None, relief)

    def class_probabilities(self, embedding: Optional[np.ndarray]) -> Dict[int, float]:
        """Return ``{person_id: ensemble probability}`` for *embedding*.

        This is the open-set softmax the AI engine itself uses — useful for
        showing a real "AI probability" next to each candidate in
        person-selector popups.  With an explicit "unknown" output the values
        sum to ``1 − P(unknown)``.

        Only the **ensemble** mode produces meaningful probabilities; prototype
        mode (too few people/examples for a discriminative network) has no
        softmax, so an empty mapping is returned and callers fall back to plain
        similarity ordering.  An empty mapping is likewise returned for a
        missing / degenerate embedding or an untrained model.
        """
        state = self._state
        vec = self._unit(embedding)
        if vec is None or state.mode != "ensemble" or not state.members:
            return {}
        if self._dim_mismatch(vec):
            return {}
        probs = self._ensemble_proba(vec.reshape(1, -1))[0]
        return {
            int(state.classes[i]): float(p)
            for i, p in enumerate(probs)
        }

    def predict_debug(
        self,
        embedding: Optional[np.ndarray],
        face_id: int = -1,
        crop_path: Optional[str] = None,
        probe: Optional[ProbeMeta] = None,
    ) -> Tuple["DeepPrediction", "DeepDebugInfo"]:
        """Like :meth:`predict` but also returns a :class:`DeepDebugInfo` object.

        The extra work (activation forward pass, gate recording) only happens
        here — the hot path :meth:`predict` is not affected.  Both run the very
        same decision code, so the recorded gates always explain the result.
        """
        from app.deep.debug_info import (
            DeepDebugInfo,
            GateResult,
            compute_decision_path,
            compute_layer_activations,
        )

        state = self._state
        gates: List[GateResult] = []
        layer_activations: List[np.ndarray] = []
        output_probs: Dict[int, float] = {}
        all_similarities: Dict[str, float] = {}

        vec = self._unit(embedding)
        if vec is None or not self.is_trained or self._dim_mismatch(vec):
            pred = self.predict(embedding, probe)
            return pred, DeepDebugInfo(
                face_id=face_id, crop_path=crop_path,
                embedding_norm=0.0, embedding_top_dims=[],
                all_similarities={}, layer_activations=[],
                output_probs={}, gates=[], prediction=pred,
                mode=state.mode,
            )

        index = state.index
        sims = index.matrix @ vec
        class_max = index.class_max(sims)
        for c, pid in enumerate(index.classes):
            all_similarities[state.person_names.get(int(pid), str(pid))] = float(class_max[c])

        if state.mode == "ensemble" and state.members:
            layer_activations = compute_layer_activations(state.members[0], vec)
            probs_avg = self._ensemble_proba(vec.reshape(1, -1))[0]
            output_probs = {
                state.person_names.get(int(state.classes[i]), str(state.classes[i])): float(p)
                for i, p in enumerate(probs_avg)
            }

        pred = self._decide(vec, probe, gates)

        # Embedding stats
        emb_arr = np.asarray(embedding, dtype=np.float32).ravel()
        emb_norm = float(np.linalg.norm(emb_arr)) if emb_arr.size else 0.0
        top_idx_arr = np.argsort(np.abs(emb_arr))[::-1][:10]
        emb_top = [(int(i), float(emb_arr[i])) for i in top_idx_arr]

        # Contribution path to the winning output (green overlay in the NN graph)
        decision_path = None
        if state.mode == "ensemble" and state.members and layer_activations:
            member = state.members[0]
            class_names = [
                state.person_names.get(int(c), "(unknown)" if int(c) == UNKNOWN_LABEL else str(c))
                for c in member.classes_
            ]
            decision_path = compute_decision_path(
                member, vec, layer_activations, emb_top, output_probs, class_names
            )

        info = DeepDebugInfo(
            face_id=face_id,
            crop_path=crop_path,
            embedding_norm=emb_norm,
            embedding_top_dims=emb_top,
            all_similarities=all_similarities,
            layer_activations=layer_activations,
            output_probs=output_probs,
            gates=gates,
            prediction=pred,
            mode=state.mode,
            decision_path=decision_path,
        )
        return pred, info

    def _decide(
        self,
        vec: np.ndarray,
        probe: Optional[ProbeMeta],
        gates: Optional[list],
        relief: float = 0.0,
    ) -> DeepPrediction:
        """The single decision procedure behind predict() and predict_debug()."""
        from app.deep.debug_info import GateResult

        state = self._state
        index = state.index
        strict = self._strictness
        # Relief only ever lowers the calibrated / probability threshold.
        threshold_shift = strict - max(0.0, float(relief))

        def record(name: str, passed: bool, value: float, threshold: float) -> bool:
            if gates is not None:
                gates.append(GateResult(name, bool(passed), float(value), float(threshold)))
            return passed

        sims = index.matrix @ vec
        best_global_sim = float(np.max(sims)) if sims.size else 0.0
        # Gate 1: must resemble *someone* we know — non-faces and complete
        # strangers fail here and are never force-assigned.
        outlier_thr = float(self._cfg("outlier_similarity", 0.42))
        if not record("outlier", best_global_sim >= outlier_thr, best_global_sim, outlier_thr):
            return DeepPrediction(
                person_id=None, person_name=None, score=0.0,
                probability=0.0, similarity=best_global_sim, margin=0.0,
                reason="outlier",
            )

        class_max = index.class_max(sims)
        probs_full = None
        if state.mode == "ensemble" and state.members:
            probs_full = self._ensemble_proba_full(vec.reshape(1, -1))[0]
        feats = self._features(index, vec, sims, class_max, probs_full, probe, None)
        if feats is None:
            return DeepPrediction(
                person_id=None, person_name=None, score=0.0, probability=0.0,
                similarity=best_global_sim, margin=0.0, reason="outlier",
            )
        c, f = feats
        person_id = int(index.classes[c])
        name = state.person_names.get(person_id)
        ensemble = probs_full is not None

        p_correct: Optional[float] = None
        if ensemble and self.has_meta_model:
            p_correct = float(state.meta.predict_proba(f.vector())[0, 1])
            score = p_correct
        elif ensemble:
            score = 0.5 * f.p_top + 0.5 * f.sim1
        else:
            score = f.sim1

        sim_margin = f.sim1 - f.runner_sim if index.n_classes > 1 else f.sim1

        def reject(reason: str) -> DeepPrediction:
            return DeepPrediction(
                person_id=None, person_name=name, score=score,
                probability=f.p_top if ensemble else 0.0, similarity=f.sim1,
                margin=f.p_margin if ensemble else sim_margin,
                reason=reason, candidate_id=person_id, knn_agreement=f.knn,
                p_correct=p_correct, sim_margin=sim_margin,
            )

        if ensemble:
            if state.has_unknown_class and not record(
                "unknown", f.p_unknown < f.p_top, f.p_unknown, f.p_top
            ):
                return reject("unknown_class")
            if p_correct is not None:
                thr = float(np.clip(
                    float(state.meta_threshold) + threshold_shift, 0.35, 0.995
                ))
                if not record("meta", p_correct >= thr, p_correct, thr):
                    return reject("below_threshold")
            else:
                thr = state.prob_thresholds.get(
                    person_id, float(self._cfg("base_prob_threshold", 0.60))
                ) + threshold_shift
                thr = max(0.25, thr)
                if not record("prob", f.p_top >= thr, f.p_top, thr):
                    return reject("below_threshold")
            min_margin = float(self._cfg("min_margin", 0.10))
            if not record("margin", f.p_margin >= min_margin, f.p_margin, min_margin):
                return reject("margin")

        # The configured minimum is a hard floor even when a stale persisted
        # model carries a lower calibrated value.
        default_floor = float(self._cfg("min_prototype_similarity", 0.55))
        floor = max(state.sim_floors.get(person_id, default_floor), default_floor)
        floor += 0.5 * strict
        # When the nearest match is a manually confirmed face, the sim_floor gate
        # is bypassed: a human labelled this face as ground truth, so we trust it.
        anchored = index.has_manual_anchor(
            sims, c, float(self._cfg("manual_anchor_min_similarity", 0.48))
        )
        if not record("sim_floor", f.sim1 >= floor or anchored, f.sim1, floor):
            return reject("prototype")
        # Embedding similarity must also clearly favour the winner over every
        # other known person (probabilities alone are blind to lookalikes).
        if index.n_classes > 1:
            min_sim_margin = float(self._cfg("min_sim_margin", 0.05))
            gap = f.sim1 - f.runner_sim
            if not record("sim_margin", gap >= min_sim_margin, gap, min_sim_margin):
                return reject("margin")
        # One mislabeled training face must not attract its lookalikes: most
        # of the nearest training faces have to agree on the winner.
        min_knn = float(self._cfg("min_knn_agreement"))
        if not record("knn", f.knn >= min_knn, f.knn, min_knn):
            return reject("knn")

        return DeepPrediction(
            person_id=person_id, person_name=name, score=score,
            probability=f.p_top if ensemble else 1.0, similarity=f.sim1,
            margin=f.p_margin if ensemble else sim_margin,
            reason="assigned", candidate_id=person_id, knn_agreement=f.knn,
            p_correct=p_correct, sim_margin=sim_margin,
        )

    def _features(
        self,
        index: _PrototypeIndex,
        vec: np.ndarray,
        sims: np.ndarray,
        class_max: np.ndarray,
        probs_full: Optional[np.ndarray],
        probe: Optional[ProbeMeta],
        exclude: Optional[int],
        candidate: Optional[int] = None,
    ) -> Optional[Tuple[int, _Features]]:
        """Pick the candidate person and compute its decision features.

        ``probs_full`` is aligned to ``index.classes`` plus a trailing
        "unknown" column (None in prototype mode).  ``exclude`` hides one class
        entirely — the open-set simulation used during calibration ("what would
        this face get if its true person were unknown?").  ``candidate`` forces
        the candidate class (hard negatives).
        """
        n = index.n_classes
        cmax = class_max.copy()
        if exclude is not None:
            cmax[exclude] = -np.inf
        if probs_full is not None:
            probs = np.asarray(probs_full[:n], dtype=np.float64).copy()
            p_unknown = float(probs_full[n]) if probs_full.shape[0] > n else 0.0
            if exclude is not None:
                probs[exclude] = 0.0
                total = float(probs.sum()) + p_unknown
                if total > 1e-12:
                    probs /= total
                    p_unknown /= total
            if candidate is None:
                candidate = int(np.argmax(probs))
            p_top = float(probs[candidate])
            others = np.delete(probs, candidate)
            p_margin = p_top - (float(np.max(others)) if others.size else 0.0)
        else:
            p_unknown, p_margin = 0.0, 0.0
            if candidate is None:
                candidate = int(np.argmax(cmax))
            p_top = 1.0
        if exclude is not None and candidate == exclude:
            return None
        if not np.isfinite(cmax[candidate]):
            return None

        sim1 = float(cmax[candidate])
        others = np.delete(cmax, candidate)
        others = others[np.isfinite(others)]
        runner_sim = float(np.max(others)) if others.size else 0.0
        own = index.class_sims(sims, candidate)
        k3 = min(3, own.shape[0])
        simk = float(np.mean(np.sort(own)[-k3:]))
        sim_center = float(np.max(index.centers[candidate] @ vec))
        knn = index.knn_agreement(
            sims, candidate, int(self._cfg("knn_k")), exclude=exclude
        )
        probe = probe or ProbeMeta()
        size = probe.size
        values = {
            "p_top": p_top,
            "p_margin": p_margin,
            "p_unknown": p_unknown,
            "sim1": sim1,
            "simk": simk,
            "sim_center": sim_center,
            "sim_margin": sim1 - runner_sim if others.size else sim1,
            "knn_agree": knn,
            "quality": float(probe.quality) if probe.quality is not None else 0.5,
            "size": (
                float(np.clip(np.log(max(size, 1.0)) / np.log(256.0), 0.0, 1.2))
                if size is not None else 0.6
            ),
            "consistency": (
                float(probe.consistency) if probe.consistency is not None else 0.9
            ),
        }
        return candidate, _Features(
            values=values, sim1=sim1, runner_sim=runner_sim, p_top=p_top,
            p_margin=p_margin, p_unknown=p_unknown, knn=knn,
        )

    def _passes_hard_gates(
        self, index: _PrototypeIndex, sims: np.ndarray, c: int, f: _Features,
        sim_floors: Dict[int, float], has_unknown: bool,
    ) -> bool:
        """The non-calibrated gates of :meth:`_decide` (used for calibration)."""
        if has_unknown and f.p_unknown >= f.p_top:
            return False
        if f.p_margin < float(self._cfg("min_margin", 0.10)):
            return False
        default_floor = float(self._cfg("min_prototype_similarity", 0.55))
        pid = int(index.classes[c])
        floor = max(sim_floors.get(pid, default_floor), default_floor)
        anchored = index.has_manual_anchor(
            sims, c, float(self._cfg("manual_anchor_min_similarity", 0.48))
        )
        if f.sim1 < floor and not anchored:
            return False
        if index.n_classes > 1 and f.sim1 - f.runner_sim < float(self._cfg("min_sim_margin", 0.05)):
            return False
        return f.knn >= float(self._cfg("min_knn_agreement"))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": _MODEL_FILE_VERSION, "state": self._state}
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Deep model saved: %s", path)

    def load(self, path: Path | str) -> bool:
        """Load a previously saved model; returns False when unusable."""
        path = Path(path)
        if not path.exists():
            return False
        try:
            with open(path, "rb") as fh:
                payload = pickle.load(fh)
            if payload.get("version") != _MODEL_FILE_VERSION:
                log.warning("Deep model version mismatch — retraining required")
                return False
            self._state = payload["state"]
            return self.is_trained
        except Exception as exc:  # noqa: BLE001 — a corrupt model just retrains
            log.warning("Failed to load deep model %s: %s", path, exc)
            return False

    # ------------------------------------------------------------------
    # Training internals
    # ------------------------------------------------------------------

    def _ensemble_size(self) -> int:
        return max(1, int(self._cfg("ensemble_size", 5)))

    def _n_folds(self, dataset: TrainingDataset) -> int:
        counts = dataset.examples_per_person()
        min_count = min(counts.values()) if counts else 0
        n_groups = len(np.unique(dataset.groups())) if dataset.n_examples else 0
        return max(0, min(int(self._cfg("calibration_folds", 3)), min_count, n_groups))

    def _make_mlp(self, seed: int, max_iter: Optional[int] = None):
        from sklearn.neural_network import MLPClassifier

        hidden = tuple(self._cfg("hidden_layers", (256, 128)))
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=int(max_iter or self._cfg("max_iter", 600)),
            shuffle=True,
            random_state=seed,
            early_stopping=False,
            n_iter_no_change=25,
            tol=1e-5,
        )

    def _workers(self, n_jobs: int) -> int:
        from app.tasks.resource_governor import get_resource_governor

        workers = max(1, min(n_jobs, os.cpu_count() or 1))
        return get_resource_governor().recommended_workers(workers)

    def _fit_ensemble(self, x: np.ndarray, y: np.ndarray, tick) -> List[object]:
        """Train all ensemble members in parallel — saturates the CPU on purpose."""
        members: List[Optional[object]] = [None] * self._ensemble_size()

        def fit_one(idx: int):
            model = self._make_mlp(seed=1000 + idx)
            model.fit(x, y)
            tick(f"network {idx + 1}/{len(members)}")
            return idx, model

        with ThreadPoolExecutor(max_workers=self._workers(len(members))) as pool:
            for idx, model in pool.map(fit_one, range(len(members))):
                members[idx] = model
        return [m for m in members if m is not None]

    def _ensemble_proba_full(self, x: np.ndarray) -> np.ndarray:
        """Average member probabilities: person columns + one "unknown" column."""
        state = self._state
        classes = np.asarray(state.classes)
        total = np.zeros((x.shape[0], len(classes) + 1), dtype=np.float64)
        for model in state.members:
            total += _aligned_proba(model, x, classes)
        total /= max(1, len(state.members))
        return total

    def _ensemble_proba(self, x: np.ndarray) -> np.ndarray:
        return self._ensemble_proba_full(x)[:, : len(self._state.classes)]

    def _unknown_rows(self, dataset: TrainingDataset) -> Optional[np.ndarray]:
        """Unknown-background rows for the explicit "unknown" output, or None."""
        if not bool(self._cfg("use_unknown_class")):
            return None
        rows = dataset.unknown_embeddings
        if rows.ndim != 2 or rows.shape[0] < int(self._cfg("min_unknown_examples")):
            return None
        if dataset.embeddings.ndim == 2 and rows.shape[1] != dataset.embeddings.shape[1]:
            return None
        return rows[: int(self._cfg("max_unknown_examples"))]

    def _augment(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Grow small classes with jittered / interpolated synthetic embeddings."""
        rng = np.random.default_rng(42)
        min_size = max(2, int(self._cfg("min_class_size", 8)))
        sigma = float(self._cfg("augment_noise_sigma", 0.03))

        extra_x: List[np.ndarray] = []
        extra_y: List[int] = []
        for label in sorted(set(y.tolist())):
            rows = x[y == label]
            need = min_size - rows.shape[0]
            for k in range(max(0, need)):
                if rows.shape[0] >= 2 and k % 2 == 1:
                    # Convex combination of two real examples of the person.
                    i, j = rng.choice(rows.shape[0], size=2, replace=False)
                    w = rng.uniform(0.3, 0.7)
                    sample = w * rows[i] + (1.0 - w) * rows[j]
                else:
                    sample = rows[k % rows.shape[0]] + rng.normal(
                        0.0, sigma, size=rows.shape[1]
                    )
                norm = float(np.linalg.norm(sample))
                if norm > 1e-8:
                    extra_x.append((sample / norm).astype(np.float32))
                    extra_y.append(label)

        if not extra_x:
            return x, y, 0
        x_aug = np.vstack([x, np.vstack(extra_x)])
        y_aug = np.concatenate([y, np.asarray(extra_y, dtype=y.dtype)])
        return x_aug, y_aug, len(extra_x)

    def _cross_validate(
        self, dataset: TrainingDataset, unknown_x: Optional[np.ndarray], tick
    ) -> Dict:
        """Out-of-fold decisions: per-person thresholds, accuracy, meta rows.

        Folds keep photo sessions together (StratifiedGroupKFold) so a model
        is never graded on a near-copy of its own training data.  Every test
        face yields a *closed-set* row (the model's own pick, correct or not)
        and an *open-set* row (its true person hidden → simulated stranger).
        """
        result: Dict = {"thresholds": {}, "accuracy": None, "rows": [], "labels": [], "weights": []}
        n_folds = self._n_folds(dataset)
        if n_folds < 2:
            return result

        from sklearn.model_selection import StratifiedGroupKFold

        x, y = dataset.embeddings, dataset.labels
        manual = np.asarray(dataset.manual_flags, dtype=bool) if dataset.manual_flags else None
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=7)
        try:
            folds = [
                (tr, te) for tr, te in splitter.split(x, y, dataset.groups())
                if tr.size and te.size
            ]
        except ValueError as exc:
            log.warning("Calibration folds unavailable: %s", exc)
            return result
        if not folds:
            return result

        def run_fold(fold_idx: int):
            train_idx, test_idx = folds[fold_idx]
            x_tr, y_tr = x[train_idx], y[train_idx]
            if len(set(y_tr.tolist())) < 2:
                tick(f"calibration {fold_idx + 1}/{len(folds)}")
                return None
            x_aug, y_aug, _ = self._augment(x_tr, y_tr)
            if unknown_x is not None:
                x_aug = np.vstack([x_aug, unknown_x])
                y_aug = np.concatenate(
                    [y_aug, np.full(unknown_x.shape[0], UNKNOWN_LABEL, dtype=y_aug.dtype)]
                )
            model = self._make_mlp(seed=2000 + fold_idx)
            model.fit(x_aug, y_aug)
            index = _PrototypeIndex.build(
                x_tr, y_tr, manual[train_idx] if manual is not None else None,
                faces_per_center=int(self._cfg("faces_per_sub_center")),
                max_centers=int(self._cfg("max_sub_centers")),
            )
            proba = _aligned_proba(model, x[test_idx], index.classes)
            tick(f"calibration {fold_idx + 1}/{len(folds)}")
            return test_idx, index, proba

        with ThreadPoolExecutor(max_workers=self._workers(len(folds))) as pool:
            outputs = [o for o in pool.map(run_fold, range(len(folds))) if o is not None]

        has_unknown = unknown_x is not None
        sim_floors = self._state.sim_floors
        ratio = float(np.clip(self._cfg("expected_unknown_ratio"), 0.0, 0.95))
        w_open = ratio / max(1e-6, 1.0 - ratio)
        true_probs: Dict[int, List[float]] = {}
        n_correct = n_total = 0
        rows, labels, weights = result["rows"], result["labels"], result["weights"]

        def add_row(index, vec, sims, cmax, probs, i, exclude, correct_class, weight):
            """One meta row; label 1 only when the pick equals *correct_class*."""
            picked = self._features(
                index, vec, sims, cmax, probs, dataset.probe(i), exclude
            )
            if picked is None:
                return
            c, f = picked
            if not self._passes_hard_gates(index, sims, c, f, sim_floors, has_unknown):
                return
            rows.append(f.vector()[0])
            labels.append(int(correct_class is not None and c == correct_class))
            weights.append(weight)

        for test_idx, index, proba in outputs:
            for row, i in enumerate(test_idx):
                vec = x[i]
                sims = index.matrix @ vec
                cmax = index.class_max(sims)
                probs = proba[row]
                true_c = index.class_index(int(y[i]))
                if true_c is None:
                    # The whole person sat in this test fold — a real stranger.
                    add_row(index, vec, sims, cmax, probs, i, None, None, w_open)
                    continue
                n_total += 1
                top = int(np.argmax(probs[: index.n_classes]))
                n_correct += int(top == true_c)
                true_probs.setdefault(int(y[i]), []).append(float(probs[true_c]))
                # Closed set: was the model's own pick right?
                add_row(index, vec, sims, cmax, probs, i, None, true_c, 1.0)
                # Open set: hide the true person → whatever wins is wrong.
                if index.n_classes > 1:
                    add_row(index, vec, sims, cmax, probs, i, true_c, None, w_open)

        base = float(self._cfg("base_prob_threshold", 0.60))
        floor = float(self._cfg("min_prob_threshold", 0.35))
        shrink = max(0.0, float(self._cfg("threshold_shrinkage")))
        thresholds: Dict[int, float] = {}
        for label, probs in true_probs.items():
            if len(probs) < 2:
                continue
            # 20th percentile of genuine probabilities (with headroom), shrunk
            # towards the global base for people with few faces, clamped into
            # [floor, base] — never stricter than the global base.
            q = float(np.percentile(probs, 20)) * 0.8
            n = float(len(probs))
            shrunk = (n * q + shrink * base) / (n + shrink) if (n + shrink) > 0 else q
            thresholds[label] = float(np.clip(shrunk, floor, base))
        result["thresholds"] = thresholds
        result["accuracy"] = (n_correct / n_total) if n_total else None
        return result

    def _fit_meta(
        self,
        state: _TrainedState,
        dataset: TrainingDataset,
        cv: Dict,
        report: DeepTrainingReport,
    ) -> None:
        """Fit the calibrated P(correct) model and its precision threshold."""
        metrics = report.metrics
        rows = list(cv["rows"])
        labels = list(cv["labels"])
        weights = list(cv["weights"])
        metrics["n_oof_rows"] = len(rows)

        # Real mistakes the user corrected: features of the wrong person on the
        # final model, weighted up — the engine must not repeat them.
        n_hard = 0
        hard_w = float(self._cfg("hard_negative_weight"))
        index = state.index
        if dataset.n_negatives and dataset.neg_embeddings.ndim == 2:
            for i, pid in enumerate(dataset.neg_person_ids):
                c = index.class_index(int(pid))
                vec = self._unit(dataset.neg_embeddings[i])
                if c is None or vec is None or self._dim_mismatch(vec):
                    continue
                sims = index.matrix @ vec
                probs = self._ensemble_proba_full(vec.reshape(1, -1))[0]
                picked = self._features(
                    index, vec, sims, index.class_max(sims), probs,
                    dataset.neg_probes[i] if i < len(dataset.neg_probes) else None,
                    None, candidate=c,
                )
                if picked is None:
                    continue
                _, f = picked
                if not self._passes_hard_gates(
                    index, sims, c, f, state.sim_floors, state.has_unknown_class
                ):
                    continue
                rows.append(f.vector()[0])
                labels.append(0)
                weights.append(hard_w)
                n_hard += 1
        metrics["n_hard_negative_rows"] = n_hard

        y = np.asarray(labels, dtype=np.int64)
        n_pos, n_neg = int(np.sum(y == 1)), int(np.sum(y == 0))
        metrics["n_meta_pos"], metrics["n_meta_neg"] = n_pos, n_neg
        if (
            not bool(self._cfg("use_meta_classifier"))
            or len(rows) < int(self._cfg("meta_min_rows"))
            or n_pos < 5
            or n_neg < 5
        ):
            metrics["meta"] = "disabled" if not self._cfg("use_meta_classifier") else "insufficient_data"
            return

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        x = np.vstack(rows)
        w = np.asarray(weights, dtype=np.float64)

        def make_model():
            return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))

        # Honest scores for picking the threshold: out-of-fold where possible.
        try:
            n_splits = min(5, n_pos, n_neg)
            scores = cross_val_predict(
                make_model(), x, y,
                cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=11),
                method="predict_proba",
                params={"logisticregression__sample_weight": w},
            )[:, 1]
        except Exception as exc:  # noqa: BLE001 — fall back to in-sample scores
            log.debug("Meta cross_val_predict failed (%s) — in-sample threshold", exc)
            scores = None
        model = make_model()
        model.fit(x, y, logisticregression__sample_weight=w)
        if scores is None:
            scores = model.predict_proba(x)[:, 1]

        target = float(self._cfg("target_precision"))
        threshold, precision, recall = _precision_threshold(scores, y, w, target)
        lo, hi = float(self._cfg("meta_min_threshold")), float(self._cfg("meta_max_threshold"))
        reached = threshold is not None
        threshold = float(np.clip(threshold if reached else hi, lo, hi))
        state.meta = model
        state.meta_threshold = threshold
        sel = scores >= threshold
        tp = float(np.sum(w[sel & (y == 1)]))
        metrics.update({
            "meta": "active",
            "meta_threshold": threshold,
            "target_precision": target,
            "target_reached": bool(reached),
            "oof_precision": tp / float(np.sum(w[sel])) if sel.any() else None,
            "oof_recall": tp / float(np.sum(w[y == 1])) if n_pos else None,
        })

    def _find_suspicious(
        self, dataset: TrainingDataset
    ) -> List[Tuple[int, int, int, float, float]]:
        """Labeled faces that look far more like another person than their own.

        Leave-one-out: own = mean of the face's top-3 similarities to the rest
        of its person; other = the same for the most similar other person.
        Only persons with ≥5 faces are checked, at most
        ``label_noise_max_fraction`` of them are flagged (most extreme first),
        and faces the user explicitly kept are never flagged again.
        """
        if dataset.n_persons < 2 or dataset.n_examples < 6:
            return []
        counts = dataset.examples_per_person()
        min_other = float(self._cfg("label_noise_min_other_similarity"))
        margin = float(self._cfg("label_noise_margin"))
        max_frac = float(self._cfg("label_noise_max_fraction"))
        kept = dataset.kept_flags if len(dataset.kept_flags) == dataset.n_examples else None
        # Sorted by person so per-person maxima are one vectorised reduceat.
        order = np.argsort(dataset.labels, kind="stable")
        x, y = dataset.embeddings[order], dataset.labels[order]
        classes, starts, sizes = np.unique(y, return_index=True, return_counts=True)
        class_pos = {int(c): k for k, c in enumerate(classes)}

        def top3_mean(sims: np.ndarray, k: int, skip: Optional[int] = None) -> float:
            part = sims[starts[k]:starts[k] + sizes[k]]
            if skip is not None:
                part = np.delete(part, skip - starts[k])
            n = min(3, part.size)
            return float(np.mean(np.partition(part, part.size - n)[-n:])) if n else -1.0

        candidates: Dict[int, List[Tuple[float, int, int, float, float]]] = {}
        for start in range(0, x.shape[0], 512):
            block = x[start:start + 512] @ x.T
            maxima = np.maximum.reduceat(block, starts, axis=1)
            for r in range(block.shape[0]):
                i = start + r
                own_label = int(y[i])
                if counts[own_label] < 5 or (kept is not None and kept[order[i]]):
                    continue
                own_k = class_pos[own_label]
                own = top3_mean(block[r], own_k, skip=i)
                row_max = maxima[r].copy()
                row_max[own_k] = -np.inf
                other_k = int(np.argmax(row_max))
                best_other = top3_mean(block[r], other_k)
                if best_other >= min_other and best_other - own >= margin:
                    candidates.setdefault(own_label, []).append(
                        (best_other - own, int(order[i]), int(classes[other_k]), own, best_other)
                    )

        flagged: List[Tuple[int, int, int, float, float]] = []
        for own_label, items in candidates.items():
            cap = int(np.floor(max_frac * counts[own_label]))
            for _gap, i, other_label, own, other in sorted(items, reverse=True)[:cap]:
                flagged.append(
                    (int(dataset.face_ids[i]), own_label, other_label, own, other)
                )
        return flagged

    def _dedup_mask(self, dataset: TrainingDataset, keep: np.ndarray) -> np.ndarray:
        """Collapse near-identical faces of one person (burst shots).

        Manual faces are preferred as the surviving representative.  Returns a
        mask over the dataset rows (True = keep).
        """
        threshold = float(self._cfg("dedup_similarity"))
        mask = np.ones(dataset.n_examples, dtype=bool)
        if threshold >= 1.0 or dataset.n_examples < 2:
            return mask
        x, y = dataset.embeddings, dataset.labels
        manual = (
            np.asarray(dataset.manual_flags, dtype=bool)
            if len(dataset.manual_flags) == dataset.n_examples
            else np.zeros(dataset.n_examples, dtype=bool)
        )
        for label in set(y.tolist()):
            rows = np.flatnonzero((y == label) & keep)
            if rows.size < 2:
                continue
            rows = rows[np.argsort(~manual[rows], kind="stable")]  # manual first
            sims = x[rows] @ x[rows].T
            survivors = np.zeros(rows.size, dtype=bool)
            for j in range(rows.size):
                if survivors.any() and float(np.max(sims[j, survivors])) >= threshold:
                    mask[rows[j]] = False
                else:
                    survivors[j] = True
        return mask

    def _calibrate_sim_floors(self, dataset: TrainingDataset) -> Dict[int, float]:
        """Derive per-person cosine-similarity floors from intra-class spread."""
        default_floor = float(self._cfg("min_prototype_similarity", 0.55))
        floors: Dict[int, float] = {}
        x, y = dataset.embeddings, dataset.labels
        for label in sorted(set(y.tolist())):
            rows = x[y == label]
            if rows.shape[0] < 2:
                continue
            sims = rows @ rows.T
            iu = np.triu_indices(rows.shape[0], k=1)
            pair_sims = sims[iu]
            if pair_sims.size == 0:
                continue
            # A new face of this person should look at least as similar as the
            # person's own weaker pairs (with slack).  Calibration may only
            # TIGHTEN the gate: a person whose labeled faces are noisy/diverse
            # must not drag the floor below the configured minimum, or every
            # vaguely face-like crop ends up assigned to them.  The upper cap
            # keeps burst-photo training sets (near-duplicate crops, intra-pair
            # sims ≈ 1.0) from rejecting every genuinely new photo.
            q = float(np.percentile(pair_sims, 10)) - 0.08
            floors[label] = float(np.clip(q, default_floor, 0.80))
        return floors

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _dim_mismatch(self, vec: np.ndarray) -> bool:
        index = self._state.index
        return index is not None and index.matrix.ndim == 2 and index.matrix.shape[1] != vec.shape[0]

    def _cfg(self, name: str, default=None):
        if self._config is not None and hasattr(self._config, name):
            return getattr(self._config, name)
        if default is not None:
            return default
        return getattr(_default_config(), name)

    @staticmethod
    def _unit(vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if vector is None:
            return None
        vec = np.asarray(vector, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return None
        return vec / norm


def _aligned_proba(model, x: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Map one member's ``predict_proba`` onto ``classes`` + an unknown column."""
    proba = model.predict_proba(x)
    out = np.zeros((x.shape[0], len(classes) + 1), dtype=np.float64)
    col_of = {int(c): i for i, c in enumerate(model.classes_)}
    for j, cls in enumerate(classes):
        col = col_of.get(int(cls))
        if col is not None:
            out[:, j] = proba[:, col]
    unknown_col = col_of.get(UNKNOWN_LABEL)
    if unknown_col is not None:
        out[:, -1] = proba[:, unknown_col]
    return out


def _precision_threshold(
    scores: np.ndarray, y: np.ndarray, w: np.ndarray, target: float
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Lowest score threshold whose weighted precision reaches *target*.

    Returns ``(threshold, precision, recall)``; threshold is None when no
    threshold reaches the target.  At least three accepted positives are
    required so a single lucky row cannot define the threshold.
    """
    order = np.argsort(-scores, kind="stable")
    s, yy, ww = scores[order], y[order], w[order]
    tp = np.cumsum(ww * (yy == 1))
    total = np.cumsum(ww)
    n_pos_seen = np.cumsum(yy == 1)
    precision = tp / np.maximum(total, 1e-12)
    pos_total = float(np.sum(ww * (yy == 1)))
    best: Optional[int] = None
    for k in range(len(s)):
        # Only cut between distinct scores.
        if k + 1 < len(s) and s[k + 1] == s[k]:
            continue
        if n_pos_seen[k] >= 3 and precision[k] >= target:
            best = k
    if best is None:
        return None, None, None
    recall = float(tp[best] / pos_total) if pos_total > 0 else None
    return float(s[best]), float(precision[best]), recall
