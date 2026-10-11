"""Deep-learning recognition service (the "new" recognition path).

Orchestrates the :class:`~app.deep.classifier.DeepFaceClassifier` against the
database:

* :meth:`train` — collects every trusted labeled face, retrains the neural
  ensemble (continual learning) and records a :class:`TrainingRun`;
* :meth:`recognize` — classifies the still-unknown faces and assigns the
  confident matches, logging every decision as an :class:`AutoAssignment`
  for later review;
* :meth:`confirm_assignment` / :meth:`correct_assignment` /
  :meth:`revert_assignment` — the review actions of the "automatic
  groupings" tab.  Confirmations become trusted training data; corrections
  additionally record :class:`FaceVeto` / :class:`FaceCorrection`
  judgements so the same mistake is not repeated.
* near-match suggestions and suspicious-label review — the active-learning
  side: the engine asks about exactly the faces it is unsure of.

Safety rules
------------
* A face already assigned to a *named* person is never touched — an earlier
  recognition always wins over a newer one.
* Faces failing the open-set gates (strangers, non-faces) stay unknown.
* The user's "different person" judgements are hard vetoes, and a photo
  taken before a person's birth / after their death never gets their name.
* When reviewed precision drops below the target, every threshold tightens.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.config import DeepRecognitionConfig, RecognitionIdentityGuardConfig
from app.db.models import (
    AUTO_ASSIGN_STATUS_AUTO,
    AUTO_ASSIGN_STATUS_CONFIRMED,
    AUTO_ASSIGN_STATUS_CORRECTED,
    AUTO_ASSIGN_STATUS_NEAR,
    AUTO_ASSIGN_STATUS_REVERTED,
    LABEL_REVIEW_KEPT,
    LABEL_REVIEW_SUSPICIOUS,
    AutoAssignment,
    Face,
    FaceCorrection,
    FaceVeto,
    Image,
    Person,
    TrainingRun,
)
from app.deep.classifier import DeepFaceClassifier, DeepPrediction
from app.deep.dataset import ProbeMeta, build_training_dataset
from app.deep.evaluation import evaluate_open_set, feedback_precision
from app.services._image_dup_geometry import is_same_physical_face
from app.services._image_dup_geometry import unit as _unit_vec
from app.utils.fuzzy_date import FuzzyDate, Precision

log = logging.getLogger(__name__)

_MODEL_FILENAME = "deep_face_model.pkl"

# Assignment source written by this engine.
DEEP_ASSIGNMENT_SOURCE = "deep_recognition"
# Source after the user confirmed an automatic assignment (trusted training).
DEEP_CONFIRMED_SOURCE = "deep_confirmed"

# Assignment sources marking a human decision — these incumbents are never
# displaced by the per-image identity guard (legacy rows carry NULL).
_HUMAN_ASSIGNMENT_SOURCES = {
    "manual",
    "manual_merge",
    "suggestion_approved",
    "deep_confirmed",
}


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@dataclass
class DeepTrainStats:
    """Counters returned by :meth:`DeepRecognitionService.train`."""

    n_persons: int = 0
    n_examples: int = 0
    n_augmented: int = 0
    validation_accuracy: Optional[float] = None
    duration_seconds: float = 0.0
    reused_existing_model: bool = False


@dataclass
class DeepRecognitionStats:
    """Counters returned by :meth:`DeepRecognitionService.recognize`."""

    n_candidates: int = 0
    n_assigned: int = 0
    n_rejected_outlier: int = 0       # stranger / non-face — stayed unknown
    n_rejected_threshold: int = 0
    n_rejected_margin: int = 0
    n_rejected_prototype: int = 0
    n_rejected_correction: int = 0    # vetoed by a user "different" judgement
    n_skipped_low_confidence: int = 0  # detector confidence too low to trust
    n_no_embedding: int = 0
    # Predicted a person that no longer exists in the DB — a stale model trained
    # on a previous database. Skipped instead of creating a ghost assignment.
    n_skipped_unknown_person: int = 0
    # Per-image identity guard: the predicted person already has an overlapping
    # face on this image, and the incumbent was kept (human / higher score).
    n_skipped_duplicate_identity: int = 0
    # Per-image identity guard: a weaker auto-recognition incumbent box was
    # deleted in favour of this stronger prediction.
    n_replaced_duplicate_identity: int = 0
    # Explicit "unknown" output of the network won.
    n_rejected_unknown_class: int = 0
    # Nearest training faces disagreed with the winner (k-NN gate).
    n_rejected_knn: int = 0
    # Photo date outside the person's lifetime.
    n_rejected_date: int = 0
    # Second appearance of the person on one photo without strong evidence.
    n_rejected_repeat_on_image: int = 0
    # Unsure faces accepted thanks to their unknown group's vote / same-day event.
    n_cluster_vote: int = 0
    n_event_prior: int = 0
    # Close-but-unsure predictions offered as one-click suggestions.
    n_near_suggestions: int = 0
    # Extra assignments made by self-training rounds.
    n_self_training: int = 0
    # Threshold tightening applied from review feedback.
    strictness: float = 0.0
    measured_precision: Optional[float] = None
    n_reviewed: int = 0


@dataclass
class _Candidate:
    """One unknown face with its (possibly revised) prediction."""

    face: Face
    prediction: DeepPrediction
    debug_info: object = None
    origin: str = "model"        # "model" | "cluster_vote" | "event"
    blocked: bool = False        # candidate person vetoed (user / date)


@dataclass(eq=False)
class _Incumbent:
    """A face already attached to a real named person on some image."""

    face_id: int
    person_id: int
    bbox: Tuple[int, int, int, int]
    vec: Optional[np.ndarray]
    is_human: bool
    confidence: Optional[float]


@dataclass
class AutoAssignmentDTO:
    """Display payload for one reviewable automatic assignment."""

    assignment_id: int
    face_id: int
    person_id: int
    person_name: str
    previous_person_name: Optional[str]
    score: float
    status: str
    crop_path: Optional[str]
    image_path: Optional[str]
    bbox: Optional[Tuple[int, int, int, int]]
    run_id: int = 0
    run_finished_at: Optional[datetime] = None
    corrected_person_name: Optional[str] = None
    decided_at: Optional[datetime] = None
    # Parsed deep-engine decision graph (None for rows predating capture).
    decision: Optional[Dict] = None


_NEAR_ORIGIN_MARKER = '%"origin": "near"%'


def _real_assignment_filter():
    """SQL filter: rows that were real automatic assignments, not near-match
    suggestions (open or already answered)."""
    return (AutoAssignment.status != AUTO_ASSIGN_STATUS_NEAR) & (
        AutoAssignment.decision_json.is_(None)
        | ~AutoAssignment.decision_json.like(_NEAR_ORIGIN_MARKER)
    )


def _parse_decision_json(raw: Optional[str]) -> Optional[Dict]:
    """Parse a stored decision JSON, tolerating missing/corrupt values.

    A row that only carries provenance (no gate list, e.g. a near-match
    suggestion made without the debug view) yields None, so the review tab
    recomputes the full graph on demand.
    """
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or "gates" not in data:
        return None
    return data


@dataclass
class SuspiciousLabelDTO:
    """A labeled face the noise filter thinks belongs to someone else."""

    face_id: int
    person_id: int
    person_name: str
    lookalike_person_id: Optional[int]
    lookalike_name: Optional[str]
    own_similarity: Optional[float]
    other_similarity: Optional[float]
    crop_path: Optional[str]
    image_path: Optional[str]
    bbox: Optional[Tuple[int, int, int, int]]


@dataclass
class TrainAndRecognizeResult:
    """Combined result of one full train + recognize cycle."""

    training_run_id: Optional[int] = None
    train: DeepTrainStats = field(default_factory=DeepTrainStats)
    recognition: DeepRecognitionStats = field(default_factory=DeepRecognitionStats)


class DeepRecognitionService:
    """DB-facing orchestration of the deep recognition engine."""

    def __init__(
        self,
        session: Session,
        config: Optional[DeepRecognitionConfig] = None,
        model_dir: Optional[Path | str] = None,
        identity_guard: Optional[RecognitionIdentityGuardConfig] = None,
        embedder_id: str = "",
    ) -> None:
        self._session = session
        self._config = config or DeepRecognitionConfig()
        self._identity_guard = identity_guard or RecognitionIdentityGuardConfig()
        self._model_dir = Path(model_dir) if model_dir else Path(self._config.model_dir)
        # Identifies the embedding model / crop geometry / TTA — part of the
        # training fingerprint, so a re-embed always retrains the model.
        self._embedder_id = embedder_id
        self._image_dates: Dict[int, FuzzyDate] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @property
    def model_path(self) -> Path:
        return self._model_dir / _MODEL_FILENAME

    def delete_model(self) -> bool:
        """Delete the persisted model file so the next train starts from scratch.

        Returns True when a file was removed.  Used by the "rebuild from scratch"
        flow: without this, an empty/unlabeled DB would leave the previous
        model on disk, and any disk-loading path would still trust its stale,
        now-deleted people.  Never raises.
        """
        path = self.model_path
        try:
            if path.exists():
                path.unlink()
                log.info("Deep model deleted (rebuild from scratch): %s", path)
                return True
        except OSError as exc:
            log.warning("Could not delete deep model %s: %s", path, exc)
        return False

    def decision_for_face(self, face_id: int) -> Optional[Dict]:
        """Recompute the deep decision graph for *face_id* from the saved model.

        Used by the review tab to show a graph for assignments made *before*
        the per-row decision was persisted (their ``decision_json`` is NULL).
        Returns None — never raises — when there is no trained model, no face,
        or no embedding, so the caller can show a friendly "no graph" notice.
        """
        from app.deep.debug_info import decision_to_dict

        info = self.debug_info_for_face(face_id)
        if info is None:
            return None
        decision = decision_to_dict(info)
        decision["recomputed"] = True
        return decision

    def debug_info_for_face(self, face_id: int):
        """Run a debug forward pass for *face_id* on the saved model.

        Returns the full :class:`~app.deep.debug_info.DeepDebugInfo` (layer
        activations, decision path, gates) or None — never raises — when there
        is no trained model, no face, or no embedding.
        """
        from app.deep.classifier import DeepFaceClassifier

        try:
            face = self._session.get(Face, face_id)
            if face is None:
                return None
            embedding = face.get_embedding()
            if embedding is None or not self.model_path.exists():
                return None
            classifier = DeepFaceClassifier(self._config)
            if not classifier.load(self.model_path) or not classifier.is_trained:
                return None
            crop_path = str(face.crop_path) if face.crop_path else None
            _pred, info = classifier.predict_debug(
                embedding, face_id=face.id, crop_path=crop_path,
                probe=ProbeMeta.from_face(face),
            )
            return info
        except Exception:  # noqa: BLE001
            log.warning("Could not recompute debug info for face %s", face_id)
            return None

    def debug_sample_face(self):
        """Debug info for a representative embedded face on the *current* model.

        Lets the neural-net graph refresh to the live model — its current set of
        persons — without needing a full visualization scan.  Prefers a face
        assigned to a real (named, unprotected) person so the output layer shows
        a meaningful prediction; falls back to any embedded face.  Returns None
        (never raises) when there is no trained model or no embedded face.
        """
        if not self.model_path.exists():
            return None
        try:
            face = (
                self._session.query(Face)
                .join(Person, Face.person_id == Person.id)
                .filter(Person.is_auto_named == False)  # noqa: E712
                .filter(Person.is_protected == False)  # noqa: E712
                .filter(Face.embedding_exists())
                .order_by(Face.assignment_confidence.desc())
                .first()
            )
            if face is None:
                face = (
                    self._session.query(Face)
                    .filter(Face.embedding_exists())
                    .first()
                )
        except Exception:  # noqa: BLE001
            log.warning("debug_sample_face: face lookup failed", exc_info=True)
            return None
        if face is None:
            return None
        return self.debug_info_for_face(face.id)

    # ------------------------------------------------------------------
    # Model export / import
    # ------------------------------------------------------------------

    def export_model(self, dest_path: Path | str) -> bool:
        """Copy the current model file to *dest_path*.

        Returns True on success, False when no trained model exists yet.
        """
        dest = Path(dest_path)
        src = self.model_path
        if not src.exists():
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        log.info("Deep model exported: %s → %s", src, dest)
        return True

    def import_model(self, src_path: Path | str) -> bool:
        """Replace the active model with the file at *src_path*.

        Validates the file version before installing.  Returns True on success,
        False when the file is missing, corrupt, or a version mismatch.
        """
        import pickle

        from app.deep.classifier import _MODEL_FILE_VERSION

        src = Path(src_path)
        if not src.exists():
            log.warning("Import failed — file not found: %s", src)
            return False
        try:
            with open(src, "rb") as fh:
                payload = pickle.load(fh)
            if payload.get("version") != _MODEL_FILE_VERSION:
                log.warning(
                    "Import failed — version mismatch (got %s, expected %s)",
                    payload.get("version"),
                    _MODEL_FILE_VERSION,
                )
                return False
        except Exception as exc:  # noqa: BLE001
            log.warning("Import failed — cannot read model file: %s", exc)
            return False

        dest = self.model_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        log.info("Deep model imported: %s → %s", src, dest)
        return True

    def train(
        self,
        mode: str = "rescan",
        progress_cb=None,
        force: bool = False,
        run: Optional[TrainingRun] = None,
    ) -> Tuple[TrainingRun, DeepTrainStats, DeepFaceClassifier]:
        """Retrain the neural ensemble on every trusted labeled face.

        Returns the persisted :class:`TrainingRun`, the statistics and the
        ready-to-use classifier.  When the labeled data is unchanged since the
        last run and ``skip_unchanged`` is on, the saved model is reused —
        unless ``force`` is set (the explicit "train the model" action must
        always retrain, e.g. to pick up new engine defaults).  *run* re-uses an
        existing TrainingRun row (self-training rounds of one cycle).
        """
        cfg = self._config
        dataset = build_training_dataset(
            self._session,
            auto_min_confidence=cfg.auto_training_min_confidence,
            use_auto_assignments=cfg.use_auto_assignments_for_training,
            exclude_low_quality=cfg.strict_quality_filter,
            embedder_id=self._embedder_id,
            collect_unknown=cfg.use_unknown_class,
            unknown_max_similarity=cfg.unknown_class_max_similarity,
            max_unknown=cfg.max_unknown_examples,
        )

        if run is None:
            run = TrainingRun(mode=mode, started_at=_utcnow_naive())
            self._session.add(run)

        classifier = DeepFaceClassifier(self._config)
        stats = DeepTrainStats(
            n_persons=dataset.n_persons, n_examples=dataset.n_examples
        )

        fingerprint = dataset.fingerprint()
        if (
            not force
            and self._config.skip_unchanged
            and classifier.load(self.model_path)
            and classifier.fingerprint == fingerprint
            and self._architecture_unchanged(classifier)
            and self._model_persons_exist(classifier)
        ):
            stats.reused_existing_model = True
            report = classifier.report
            if report is not None:
                stats.n_augmented = report.n_augmented
                stats.validation_accuracy = report.validation_accuracy
            log.info("Deep training skipped: labeled data unchanged — model reused")
        else:
            report = classifier.train(dataset, progress_cb=progress_cb)
            stats.n_augmented = report.n_augmented
            stats.validation_accuracy = report.validation_accuracy
            stats.duration_seconds = report.duration_seconds
            if classifier.is_trained:
                try:
                    classifier.save(self.model_path)
                except OSError as exc:
                    log.warning("Could not persist deep model: %s", exc)
            self._flag_suspicious_labels(report)
            if cfg.evaluate_after_training and classifier.is_trained:
                try:
                    evaluation = evaluate_open_set(
                        dataset, cfg,
                        n_folds=cfg.calibration_folds,
                        holdout_fraction=cfg.evaluation_holdout_fraction,
                        unknown_ratio=cfg.expected_unknown_ratio,
                    )
                    report.metrics["evaluation"] = evaluation.to_dict()
                except Exception:  # noqa: BLE001 — diagnostics never block training
                    log.warning("Open-set evaluation failed", exc_info=True)

        run.finished_at = _utcnow_naive()
        run.n_persons = stats.n_persons
        run.n_examples = stats.n_examples
        run.n_augmented = stats.n_augmented
        run.validation_accuracy = stats.validation_accuracy
        run.data_fingerprint = fingerprint
        run.model_path = str(self.model_path)
        run.metrics_json = self._metrics_json(classifier.report)
        self._session.commit()
        return run, stats, classifier

    @staticmethod
    def _metrics_json(report) -> Optional[str]:
        if report is None:
            return None
        data = dict(report.metrics or {})
        data.update({
            "mode": report.mode,
            "n_deduplicated": report.n_deduplicated,
            "n_negatives": report.n_negatives,
            "n_unknown": report.n_unknown,
            "suspicious": [list(row) for row in report.suspicious],
        })
        try:
            return json.dumps(data, ensure_ascii=False, default=float)
        except (TypeError, ValueError):
            return None

    def _flag_suspicious_labels(self, report) -> None:
        """Mark label-noise suspects for review (never re-flag a kept label)."""
        ids = report.suspicious_face_ids if report is not None else []
        if not ids:
            return
        for face in self._session.query(Face).filter(Face.id.in_(ids)).all():
            if face.label_review_status != LABEL_REVIEW_KEPT:
                face.label_review_status = LABEL_REVIEW_SUSPICIOUS
        log.info("Deep training: %d suspicious label(s) flagged for review", len(ids))

    def _model_persons_exist(self, classifier: DeepFaceClassifier) -> bool:
        """False when none of the model's people still exist in the database.

        Prevents reusing a stale on-disk model (trained on a previous DB) after
        the database was reset — such a model must be retrained from scratch.
        """
        ids = set(classifier.person_ids)
        if not ids:
            return True  # empty model carries no stale people
        existing = {pid for (pid,) in self._session.query(Person.id).all()}
        return bool(ids & existing)

    def _architecture_unchanged(self, classifier: DeepFaceClassifier) -> bool:
        """False when the saved ensemble was trained with a different layer
        layout than the current config — the model must then be rebuilt even
        though the labeled data is unchanged."""
        trained = classifier.trained_hidden_layers
        return trained is None or trained == tuple(self._config.hidden_layers)

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def recognize(
        self,
        classifier: DeepFaceClassifier,
        training_run: TrainingRun,
        progress_cb=None,
        debug_cb=None,
    ) -> DeepRecognitionStats:
        """Assign still-unknown faces to known people where the model is sure.

        Faces already assigned to a named person are never candidates — an
        existing recognition always survives a newer run untouched.

        The run has five passes: (1) predict every candidate, (2) apply the
        user's vetoes and the lifetime check, (3) give unsure faces a second
        chance when their unknown group or a same-day photo supports the
        candidate, (4) apply the assignments strongest-first under the
        per-image guards, (5) offer the close misses as near-match suggestions.
        """
        cfg = self._config
        stats = DeepRecognitionStats()
        if not classifier.is_trained:
            log.info("Deep recognition skipped: no trained model")
            return stats

        # Guard against a stale model: one trained on a previous database whose
        # people no longer exist (e.g. the DB was reset but the model kept).
        # Assigning to those vanished person IDs would create ghost links.
        valid_person_ids = {
            pid for (pid,) in self._session.query(Person.id).all()
        }
        model_person_ids = set(classifier.person_ids)
        if model_person_ids and not (model_person_ids & valid_person_ids):
            log.warning(
                "Deep recognition skipped: the model was trained on people that "
                "no longer exist in this database (model persons=%s). Rebuild the "
                "AI model from scratch.",
                sorted(model_person_ids),
            )
            return stats

        self._apply_feedback_strictness(classifier, training_run, stats)
        self._drop_stale_near_suggestions()

        candidates = self._load_candidate_faces()
        stats.n_candidates = len(candidates)
        if not candidates:
            self._session.commit()
            return stats

        vetoes = self._load_correction_vetoes(candidates)
        lifespans = self._load_lifespans() if cfg.date_veto_enabled else {}

        # --- Pass 1: predict ------------------------------------------------
        decided: List[_Candidate] = []
        for idx, face in enumerate(candidates):
            if progress_cb is not None:
                progress_cb(idx + 1, len(candidates), f"face #{face.id}")

            # "Never recognise non-faces": a weak detection may not even be a
            # face — it may stay in the unknown pool but is never auto-assigned.
            if (
                face.confidence is not None
                and face.confidence < cfg.min_face_confidence
            ):
                stats.n_skipped_low_confidence += 1
                continue

            probe = ProbeMeta.from_face(face)
            debug_info = None
            if debug_cb is not None:
                crop_path = str(face.crop_path) if face.crop_path else None
                prediction, debug_info = classifier.predict_debug(
                    face.get_embedding(), face_id=face.id, crop_path=crop_path,
                    probe=probe,
                )
                debug_cb(debug_info)
            else:
                prediction = classifier.predict(face.get_embedding(), probe)
            if prediction.reason == "no_embedding":
                stats.n_no_embedding += 1
                continue
            decided.append(_Candidate(face, prediction, debug_info))

        # --- Pass 2: user vetoes + lifetime check ---------------------------
        for cand in decided:
            pid = cand.prediction.candidate_id
            if pid is None:
                continue
            if pid in vetoes.get(cand.face.id, set()):
                cand.blocked = True
                if cand.prediction.person_id is not None:
                    stats.n_rejected_correction += 1
                    cand.prediction = self._rejected(cand.prediction, "correction")
            elif lifespans and self._outside_lifetime(cand.face, pid, lifespans):
                cand.blocked = True
                if cand.prediction.person_id is not None:
                    stats.n_rejected_date += 1
                    cand.prediction = self._rejected(cand.prediction, "date")

        # --- Pass 3: context second chance ----------------------------------
        if cfg.cluster_vote_enabled:
            stats.n_cluster_vote = self._cluster_vote(classifier, decided)
        if cfg.event_prior_bonus > 0:
            stats.n_event_prior = self._event_prior(classifier, decided)

        # --- Pass 4: apply, strongest first ---------------------------------
        now = _utcnow_naive()
        # Per-image identity guard: index the faces already attached to a real
        # named person, keyed by image, so we never label one physical face with
        # a person who is already placed on that photo.
        incumbents = self._load_incumbents_by_image()
        meta_threshold = classifier.meta_threshold
        assigned_ids: Set[int] = set()
        order = sorted(
            (c for c in decided if c.prediction.person_id is not None),
            key=lambda c: -c.prediction.score,
        )
        for cand in order:
            face, prediction = cand.face, cand.prediction
            # Never attach a face to a person that no longer exists (stale model).
            if prediction.person_id not in valid_person_ids:
                stats.n_skipped_unknown_person += 1
                cand.blocked = True
                continue

            # Per-image identity guard: does this person already own an
            # overlapping face on this image?
            if self._identity_guard.enabled:
                dup = self._find_identity_dup(face, prediction.person_id, incumbents)
                if dup is not None:
                    incumbent_conf = (
                        dup.confidence if dup.confidence is not None else -1.0
                    )
                    if dup.is_human or incumbent_conf >= prediction.score:
                        # A second box of an already-named face — never a
                        # suggestion either.
                        stats.n_skipped_duplicate_identity += 1
                        cand.blocked = True
                        continue
                    # Weaker auto-recognition incumbent — drop its box (its
                    # AutoAssignment row cascades) so the clusterer cannot
                    # re-pick it within this run, then let the stronger
                    # prediction take over below.
                    self._session.query(Face).filter(Face.id == dup.face_id).delete(
                        synchronize_session="fetch"
                    )
                    incumbents[face.image_id].remove(dup)
                    stats.n_replaced_duplicate_identity += 1

            # A second, separate appearance of the person on this photo is
            # usually a lookalike relative — demand stronger evidence.
            if cfg.repeat_on_image_strict and any(
                inc.person_id == prediction.person_id
                for inc in incumbents.get(face.image_id, [])
            ):
                if not self._strong_enough_for_repeat(prediction, meta_threshold, classifier):
                    stats.n_rejected_repeat_on_image += 1
                    cand.prediction = self._rejected(prediction, "repeat_on_image")
                    continue

            previous_person_id = face.person_id
            previous_person_name = (
                face.person.name if face.person is not None else None
            )
            face.person_id = prediction.person_id
            face.assignment_source = DEEP_ASSIGNMENT_SOURCE
            face.assignment_confidence = prediction.score
            face.assigned_at = now

            # Capture the decision graph only when it was already computed for
            # the live visualization.  Otherwise leave it NULL: the review tab
            # recomputes the graph on demand via ``decision_for_face`` when a
            # row is opened, so the hot path avoids a second full forward pass
            # per assigned face — roughly halving recognition time on large
            # archives where most runs have the viz off.
            decision_json = (
                self._serialize_decision(
                    classifier, face, cand.debug_info, origin=cand.origin
                )
                if cand.debug_info is not None
                else self._origin_json(cand.origin)
            )

            self._session.add(
                AutoAssignment(
                    training_run=training_run,
                    face_id=face.id,
                    person_id=prediction.person_id,
                    previous_person_id=previous_person_id,
                    previous_person_name=previous_person_name,
                    score=prediction.score,
                    status=AUTO_ASSIGN_STATUS_AUTO,
                    decision_json=decision_json,
                    created_at=now,
                )
            )
            stats.n_assigned += 1
            assigned_ids.add(face.id)

            incumbents.setdefault(face.image_id, []).append(
                _Incumbent(
                    face_id=face.id,
                    person_id=prediction.person_id,
                    bbox=(face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h),
                    vec=_unit_vec(face.get_embedding()),
                    is_human=False,
                    confidence=prediction.score,
                )
            )

        for cand in decided:
            if cand.face.id in assigned_ids:
                continue
            reason = cand.prediction.reason
            if reason == "outlier":
                stats.n_rejected_outlier += 1
            elif reason == "below_threshold":
                stats.n_rejected_threshold += 1
            elif reason == "margin":
                stats.n_rejected_margin += 1
            elif reason == "prototype":
                stats.n_rejected_prototype += 1
            elif reason == "unknown_class":
                stats.n_rejected_unknown_class += 1
            elif reason == "knn":
                stats.n_rejected_knn += 1

        # --- Pass 5: near-match suggestions (active learning) ---------------
        if cfg.near_suggestions_enabled:
            stats.n_near_suggestions = self._add_near_suggestions(
                decided, assigned_ids, valid_person_ids, training_run, now
            )

        self._delete_orphan_auto_persons()
        self._session.commit()
        log.info(
            "Deep recognition: %d/%d assigned (outlier=%d, threshold=%d, margin=%d, "
            "prototype=%d, unknown=%d, knn=%d, correction-veto=%d, date-veto=%d, "
            "repeat-on-image=%d, low-conf=%d, no-emb=%d, unknown-person=%d, "
            "dup-identity-skip=%d, dup-identity-replace=%d, cluster-vote=%d, "
            "event=%d, near=%d, strictness=%.3f)",
            stats.n_assigned, stats.n_candidates,
            stats.n_rejected_outlier, stats.n_rejected_threshold,
            stats.n_rejected_margin, stats.n_rejected_prototype,
            stats.n_rejected_unknown_class, stats.n_rejected_knn,
            stats.n_rejected_correction, stats.n_rejected_date,
            stats.n_rejected_repeat_on_image, stats.n_skipped_low_confidence,
            stats.n_no_embedding, stats.n_skipped_unknown_person,
            stats.n_skipped_duplicate_identity, stats.n_replaced_duplicate_identity,
            stats.n_cluster_vote, stats.n_event_prior, stats.n_near_suggestions,
            stats.strictness,
        )
        return stats

    # ------------------------------------------------------------------
    # Recognition context helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rejected(prediction: DeepPrediction, reason: str) -> DeepPrediction:
        """Copy of *prediction* turned into a rejection (candidate kept)."""
        from dataclasses import replace

        return replace(prediction, person_id=None, reason=reason)

    def _apply_feedback_strictness(
        self,
        classifier: DeepFaceClassifier,
        training_run: TrainingRun,
        stats: DeepRecognitionStats,
    ) -> None:
        """Tighten every threshold when reviewed precision is below target."""
        cfg = self._config
        precision, n = feedback_precision(self._session, cfg.feedback_window)
        stats.measured_precision, stats.n_reviewed = precision, n
        training_run.measured_precision = precision
        training_run.n_reviewed = n
        strictness = 0.0
        if (
            precision is not None
            and n >= cfg.feedback_min_reviewed
            and precision < cfg.target_precision
        ):
            strictness = min(
                cfg.max_strictness,
                (cfg.target_precision - precision) * cfg.feedback_tighten_gain,
            )
            log.info(
                "Deep recognition: reviewed precision %.3f < target %.3f over %d "
                "decisions — tightening thresholds by %.3f",
                precision, cfg.target_precision, n, strictness,
            )
        classifier.set_strictness(max(classifier.strictness, strictness))
        stats.strictness = classifier.strictness

    def _strong_enough_for_repeat(
        self,
        prediction: DeepPrediction,
        meta_threshold: Optional[float],
        classifier: DeepFaceClassifier,
    ) -> bool:
        cfg = self._config
        if prediction.knn_agreement < 0.999:
            return False
        if prediction.sim_margin < 2.0 * cfg.min_sim_margin:
            return False
        if prediction.p_correct is not None and meta_threshold is not None:
            needed = min(
                0.995,
                meta_threshold + classifier.strictness + cfg.repeat_extra_confidence,
            )
            return prediction.p_correct >= needed
        return True

    def _image_date(self, face: Face) -> Optional[FuzzyDate]:
        image_id = face.image_id
        if image_id not in self._image_dates:
            image = face.image
            raw = (image.photo_date or image.estimated_date) if image is not None else None
            self._image_dates[image_id] = FuzzyDate.parse(raw)
        parsed = self._image_dates[image_id]
        return parsed if parsed.is_known else None

    def _load_lifespans(self) -> Dict[int, Tuple[Optional[int], Optional[int]]]:
        """person id → (earliest birth year, latest death year) when known."""
        spans: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
        rows = (
            self._session.query(Person.id, Person.birth_date, Person.death_date)
            .filter(or_(Person.birth_date.isnot(None), Person.death_date.isnot(None)))
            .all()
        )
        for pid, birth_raw, death_raw in rows:
            birth = FuzzyDate.parse(birth_raw)
            death = FuzzyDate.parse(death_raw)
            born = birth.earliest.year if birth.earliest is not None else None
            died = death.latest.year if death.latest is not None else None
            if born is not None or died is not None:
                spans[int(pid)] = (born, died)
        return spans

    def _outside_lifetime(
        self,
        face: Face,
        person_id: int,
        lifespans: Dict[int, Tuple[Optional[int], Optional[int]]],
    ) -> bool:
        """True when the photo was certainly taken before birth / after death."""
        span = lifespans.get(int(person_id))
        if span is None:
            return False
        taken = self._image_date(face)
        if taken is None:
            return False
        tol = int(self._config.date_veto_tolerance_years)
        born, died = span
        if born is not None and taken.latest is not None and taken.latest.year < born - tol:
            return True
        if died is not None and taken.earliest is not None and taken.earliest.year > died + tol:
            return True
        return False

    def _cluster_vote(
        self, classifier: DeepFaceClassifier, decided: List[_Candidate]
    ) -> int:
        """Unsure members of an unknown group follow the group's clear majority.

        A group whose decided members overwhelmingly went to one person is
        most likely that person; its unsure members (only threshold-unsure —
        every other gate must still pass) are re-evaluated with a relaxed
        calibrated threshold.  Mixed groups never propagate a name.
        """
        cfg = self._config
        groups: Dict[int, List[_Candidate]] = {}
        for cand in decided:
            person = cand.face.person
            if person is not None and person.is_auto_named:
                groups.setdefault(person.id, []).append(cand)

        accepted = 0
        for members in groups.values():
            if len(members) < cfg.cluster_vote_min_size:
                continue
            votes = Counter(
                c.prediction.person_id
                for c in members
                if c.prediction.person_id is not None
            )
            if not votes:
                continue
            winner, n_winner = votes.most_common(1)[0]
            if (
                n_winner < cfg.cluster_vote_min_assigned
                or n_winner / float(sum(votes.values())) < cfg.cluster_vote_min_purity
            ):
                continue
            for cand in members:
                pred = cand.prediction
                if (
                    pred.person_id is not None
                    or cand.blocked
                    or pred.candidate_id != winner
                    or pred.reason != "below_threshold"
                    or pred.similarity < cfg.cluster_vote_min_similarity
                ):
                    continue
                retry = classifier.predict(
                    cand.face.get_embedding(),
                    ProbeMeta.from_face(cand.face),
                    relief=cfg.cluster_vote_relief,
                )
                if retry.person_id == winner:
                    cand.prediction = retry
                    cand.origin = "cluster_vote"
                    accepted += 1
        return accepted

    def _event_prior(
        self, classifier: DeepFaceClassifier, decided: List[_Candidate]
    ) -> int:
        """A person confirmed on another photo of the same day lowers the bar.

        Only exact-day photo dates count, only human-confirmed appearances,
        and the relief is small (``event_prior_bonus``) — the face must still
        pass every other gate.
        """
        unsure = [
            c for c in decided
            if c.prediction.person_id is None
            and not c.blocked
            and c.prediction.reason == "below_threshold"
            and c.prediction.candidate_id is not None
        ]
        if not unsure:
            return 0
        day_persons = self._confirmed_persons_by_day()
        if not day_persons:
            return 0
        accepted = 0
        for cand in unsure:
            taken = self._image_date(cand.face)
            if taken is None or taken.precision is not Precision.EXACT:
                continue
            day = taken.earliest
            others = day_persons.get(day, {})
            pid = cand.prediction.candidate_id
            # The supporting appearance must be on a *different* photo.
            if not (others.get(pid, set()) - {cand.face.image_id}):
                continue
            retry = classifier.predict(
                cand.face.get_embedding(),
                ProbeMeta.from_face(cand.face),
                relief=self._config.event_prior_bonus,
            )
            if retry.person_id == pid:
                cand.prediction = retry
                cand.origin = "event"
                accepted += 1
        return accepted

    def _confirmed_persons_by_day(self) -> Dict[date, Dict[int, Set[int]]]:
        """day → person id → image ids where a human confirmed that person."""
        rows = (
            self._session.query(Face.person_id, Face.image_id, Image.photo_date)
            .join(Image, Face.image_id == Image.id)
            .join(Person, Face.person_id == Person.id)
            .filter(Person.is_auto_named == False)  # noqa: E712
            .filter(Person.is_protected == False)  # noqa: E712
            .filter(Face.is_excluded == False)  # noqa: E712
            .filter(Image.photo_date.isnot(None))
            .filter(
                or_(
                    Face.assignment_source.is_(None),
                    Face.assignment_source.in_(_HUMAN_ASSIGNMENT_SOURCES),
                )
            )
            .all()
        )
        by_day: Dict[date, Dict[int, Set[int]]] = {}
        for pid, image_id, raw in rows:
            parsed = FuzzyDate.parse(raw)
            if parsed.precision is not Precision.EXACT or parsed.earliest is None:
                continue
            by_day.setdefault(parsed.earliest, {}).setdefault(int(pid), set()).add(
                int(image_id)
            )
        return by_day

    def _add_near_suggestions(
        self,
        decided: List[_Candidate],
        assigned_ids: Set[int],
        valid_person_ids: Set[int],
        training_run: TrainingRun,
        now: datetime,
    ) -> int:
        """Store the most informative close misses as one-click suggestions."""
        cfg = self._config
        near: List[_Candidate] = []
        for cand in decided:
            pred = cand.prediction
            if (
                cand.face.id in assigned_ids
                or cand.blocked
                or pred.candidate_id is None
                or pred.candidate_id not in valid_person_ids
                or pred.reason in ("outlier", "no_embedding", "untrained")
                or pred.similarity < cfg.near_min_similarity
                or pred.score < cfg.near_min_score
            ):
                continue
            near.append(cand)
        near.sort(key=lambda c: -c.prediction.score)
        for cand in near[: max(0, int(cfg.max_near_suggestions))]:
            face, pred = cand.face, cand.prediction
            self._session.add(
                AutoAssignment(
                    training_run=training_run,
                    face_id=face.id,
                    person_id=pred.candidate_id,
                    previous_person_id=face.person_id,
                    previous_person_name=(
                        face.person.name if face.person is not None else None
                    ),
                    score=pred.score,
                    status=AUTO_ASSIGN_STATUS_NEAR,
                    decision_json=json.dumps(
                        {"origin": "near", "reason": pred.reason}, ensure_ascii=False
                    ),
                    created_at=now,
                )
            )
        return min(len(near), max(0, int(cfg.max_near_suggestions)))

    def _drop_stale_near_suggestions(self) -> None:
        """Unanswered near suggestions of older runs are superseded."""
        self._session.query(AutoAssignment).filter(
            AutoAssignment.status == AUTO_ASSIGN_STATUS_NEAR
        ).delete(synchronize_session="fetch")

    @staticmethod
    def _origin_json(origin: str) -> Optional[str]:
        """Provenance marker for assignments made via context (no graph)."""
        if origin == "model":
            return None
        return json.dumps({"origin": origin}, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Per-image identity guard
    # ------------------------------------------------------------------

    def _load_incumbents_by_image(self) -> Dict[int, List["_Incumbent"]]:
        """Index faces already attached to a real named person, by image id."""
        rows: List[Face] = (
            self._session.query(Face)
            .join(Person, Face.person_id == Person.id)
            .filter(Face.is_excluded == False)  # noqa: E712
            .filter(Person.is_auto_named == False)  # noqa: E712
            .filter(Person.is_protected == False)  # noqa: E712
            .all()
        )
        by_image: Dict[int, List[_Incumbent]] = {}
        for face in rows:
            is_human = (
                face.detector_backend == "manual"
                or face.assignment_source is None
                or face.assignment_source in _HUMAN_ASSIGNMENT_SOURCES
            )
            by_image.setdefault(face.image_id, []).append(
                _Incumbent(
                    face_id=face.id,
                    person_id=face.person_id,
                    bbox=(face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h),
                    vec=_unit_vec(face.get_embedding()),
                    is_human=is_human,
                    confidence=face.assignment_confidence,
                )
            )
        return by_image

    def _find_identity_dup(
        self,
        face: Face,
        predicted_person_id: int,
        incumbents: Dict[int, List["_Incumbent"]],
    ) -> Optional["_Incumbent"]:
        """Return the incumbent on *face*'s image that this prediction duplicates.

        A duplicate = same predicted person + geometric/embedding overlap.  When
        several match, the highest-confidence one is returned (human incumbents
        rank above any score).
        """
        same_person = [
            inc
            for inc in incumbents.get(face.image_id, [])
            if inc.person_id == predicted_person_id
        ]
        if not same_person:
            return None
        face_bbox = (face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h)
        face_vec = _unit_vec(face.get_embedding())
        dups = [
            inc
            for inc in same_person
            if is_same_physical_face(
                face_bbox, face_vec, inc.bbox, inc.vec, self._identity_guard
            )
        ]
        if not dups:
            return None
        return max(
            dups,
            key=lambda inc: (
                1 if inc.is_human else 0,
                inc.confidence if inc.confidence is not None else -1.0,
            ),
        )

    @staticmethod
    def _serialize_decision(
        classifier: DeepFaceClassifier, face, debug_info, origin: str = "model"
    ) -> Optional[str]:
        """JSON for *face*'s assignment decision graph, or None on failure.

        Reuses *debug_info* when it was already computed for the live viz;
        otherwise runs ``predict_debug`` once for this single assigned face.
        Never raises — a diagnostics blob must not abort a recognition run.
        """
        from app.deep.debug_info import DeepDebugInfo, decision_to_dict

        try:
            if not isinstance(debug_info, DeepDebugInfo):
                crop_path = str(face.crop_path) if face.crop_path else None
                _pred, debug_info = classifier.predict_debug(
                    face.get_embedding(), face_id=face.id, crop_path=crop_path,
                    probe=ProbeMeta.from_face(face),
                )
            decision = decision_to_dict(debug_info)
            if origin != "model":
                decision["origin"] = origin
            return json.dumps(decision, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            log.warning("Could not serialize decision graph for face %s", face.id)
            return None

    def train_and_recognize(
        self,
        mode: str = "rescan",
        train_progress_cb=None,
        recognize_progress_cb=None,
        debug_cb=None,
    ) -> TrainAndRecognizeResult:
        """Convenience wrapper: one full continual-learning cycle.

        A *rebuild* always retrains from scratch (``force``), never reusing the
        previous on-disk model — the rebuild flow has already deleted it.
        """
        result = TrainAndRecognizeResult()
        run, train_stats, classifier = self.train(
            mode=mode,
            progress_cb=train_progress_cb,
            force=(mode in ("rebuild", "rebuild_model")),
        )
        result.training_run_id = run.id
        result.train = train_stats
        result.recognition = self.recognize(
            classifier, run,
            progress_cb=recognize_progress_cb,
            debug_cb=debug_cb,
        )

        # Self-training: this run's most confident assignments (score ≥
        # auto_training_min_confidence) are now trusted training data — a
        # retrained model may recognise more of the remaining faces.  Stops as
        # soon as a round adds nothing or the data did not change.
        rec = result.recognition
        for _round in range(max(0, int(self._config.self_training_rounds))):
            if rec.n_assigned == 0:
                break
            _, round_stats, round_clf = self.train(
                mode=mode, progress_cb=train_progress_cb, run=run
            )
            if round_stats.reused_existing_model:
                break
            round_clf.set_strictness(classifier.strictness)
            rec = self.recognize(round_clf, run, progress_cb=recognize_progress_cb)
            result.recognition.n_assigned += rec.n_assigned
            result.recognition.n_self_training += rec.n_assigned
            result.recognition.n_near_suggestions = rec.n_near_suggestions
            result.train = round_stats
        return result

    # ------------------------------------------------------------------
    # Review actions ("automatic groupings" tab)
    # ------------------------------------------------------------------

    def latest_run(self) -> Optional[TrainingRun]:
        return (
            self._session.query(TrainingRun)
            .filter(TrainingRun.finished_at.isnot(None))
            .order_by(TrainingRun.id.desc())
            .first()
        )

    def latest_assignment_run_id(self) -> Optional[int]:
        """Id of the newest run that actually produced auto-assignments.

        Train-only runs create a :class:`TrainingRun` without assignments;
        the review tab must keep showing the last run that has any.
        """
        row = (
            self._session.query(AutoAssignment.training_run_id)
            .filter(_real_assignment_filter())
            .order_by(AutoAssignment.training_run_id.desc())
            .first()
        )
        return row[0] if row is not None else None

    def list_auto_assignments(
        self,
        run_id: Optional[int] = None,
        only_open: bool = True,
    ) -> List[AutoAssignmentDTO]:
        """Reviewable assignments of *run_id* (default: last run with any)."""
        if run_id is None:
            run_id = self.latest_assignment_run_id()
            if run_id is None:
                return []

        query = (
            self._session.query(AutoAssignment)
            .filter(AutoAssignment.training_run_id == run_id)
            .filter(_real_assignment_filter())
        )
        if only_open:
            query = query.filter(AutoAssignment.status == AUTO_ASSIGN_STATUS_AUTO)

        dtos: List[AutoAssignmentDTO] = []
        for assignment in query.order_by(AutoAssignment.score.desc()).all():
            face = assignment.face
            if face is None:
                continue
            # Skip stale rows: the face was meanwhile moved elsewhere manually.
            if (
                assignment.status == AUTO_ASSIGN_STATUS_AUTO
                and face.person_id != assignment.person_id
            ):
                continue
            person = assignment.person
            dtos.append(
                AutoAssignmentDTO(
                    assignment_id=assignment.id,
                    face_id=face.id,
                    person_id=assignment.person_id,
                    person_name=person.name if person else "?",
                    previous_person_name=(
                        assignment.previous_person_name
                        or (
                            assignment.previous_person.name
                            if assignment.previous_person is not None
                            else None
                        )
                    ),
                    score=assignment.score,
                    status=assignment.status,
                    crop_path=face.crop_path,
                    image_path=face.image.file_path if face.image else None,
                    bbox=(face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h),
                    run_id=run_id,
                    run_finished_at=(
                        assignment.training_run.finished_at
                        if assignment.training_run
                        else None
                    ),
                    decision=_parse_decision_json(assignment.decision_json),
                )
            )
        return dtos

    def list_decided_assignments(self, limit: int = 100) -> List[AutoAssignmentDTO]:
        """Already-reviewed assignments (any run), newest decision first."""
        rows = (
            self._session.query(AutoAssignment)
            .filter(
                AutoAssignment.status.notin_(
                    (AUTO_ASSIGN_STATUS_AUTO, AUTO_ASSIGN_STATUS_NEAR)
                )
            )
            .order_by(AutoAssignment.decided_at.desc(), AutoAssignment.id.desc())
            .limit(limit)
            .all()
        )
        dtos: List[AutoAssignmentDTO] = []
        for assignment in rows:
            face = assignment.face
            person = assignment.person
            dtos.append(
                AutoAssignmentDTO(
                    assignment_id=assignment.id,
                    face_id=assignment.face_id,
                    person_id=assignment.person_id,
                    person_name=person.name if person else "?",
                    previous_person_name=(
                        assignment.previous_person_name
                        or (
                            assignment.previous_person.name
                            if assignment.previous_person is not None
                            else None
                        )
                    ),
                    score=assignment.score,
                    status=assignment.status,
                    crop_path=face.crop_path if face is not None else None,
                    image_path=(
                        face.image.file_path
                        if face is not None and face.image
                        else None
                    ),
                    bbox=(
                        (face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h)
                        if face is not None
                        else None
                    ),
                    run_id=assignment.training_run_id,
                    run_finished_at=(
                        assignment.training_run.finished_at
                        if assignment.training_run
                        else None
                    ),
                    corrected_person_name=(
                        assignment.corrected_person.name
                        if assignment.corrected_person is not None
                        else None
                    ),
                    decided_at=assignment.decided_at,
                    decision=_parse_decision_json(assignment.decision_json),
                )
            )
        return dtos

    def count_open_assignments(self, run_id: Optional[int] = None) -> int:
        if run_id is None:
            run_id = self.latest_assignment_run_id()
            if run_id is None:
                return 0
        return (
            self._session.query(AutoAssignment)
            .filter(AutoAssignment.training_run_id == run_id)
            .filter(AutoAssignment.status == AUTO_ASSIGN_STATUS_AUTO)
            .count()
        )

    def confirm_assignment(self, assignment_id: int) -> AutoAssignment:
        """User says the grouping is right → becomes trusted training data.

        For a near-match suggestion this is where the face actually moves to
        the suggested person.
        """
        assignment = self._require_assignment(assignment_id)
        self._confirm_one(assignment)
        self._delete_orphan_auto_persons()
        self._session.commit()
        return assignment

    def _confirm_one(self, assignment: AutoAssignment) -> None:
        """Shared confirm core; caller commits."""
        face = assignment.face
        if face is not None and assignment.status == AUTO_ASSIGN_STATUS_NEAR:
            # Only move a face that is still unknown / in an auto group — the
            # user may have named it elsewhere meanwhile.
            person = face.person
            if person is None or person.is_auto_named or person.is_protected:
                face.person_id = assignment.person_id
                face.assigned_at = _utcnow_naive()
        if face is not None and face.person_id == assignment.person_id:
            face.assignment_source = DEEP_CONFIRMED_SOURCE
            face.assignment_confidence = 1.0
        assignment.status = AUTO_ASSIGN_STATUS_CONFIRMED
        assignment.decided_at = _utcnow_naive()

    def confirm_assignments(self, assignment_ids: List[int]) -> int:
        """Confirm many groupings in one transaction. Returns the count confirmed.

        Skips ids that no longer exist (e.g. removed by a meanwhile-running
        re-scan), so a stale selection can never abort the whole batch.
        """
        confirmed = 0
        for assignment_id in assignment_ids:
            assignment = self._session.get(AutoAssignment, assignment_id)
            if assignment is None:
                continue
            self._confirm_one(assignment)
            confirmed += 1
        self._delete_orphan_auto_persons()
        self._session.commit()
        log.info("Deep recognition: batch-confirmed %d assignments", confirmed)
        return confirmed

    def correct_assignment(
        self, assignment_id: int, new_person_id: int
    ) -> AutoAssignment:
        """User moves the face to the right person — and the engine learns.

        Records a "different person" judgement against the wrong person and a
        "same person" judgement towards the right one, so future runs are
        vetoed from repeating the mistake and gain a confirmed example.
        """
        assignment = self._require_assignment(assignment_id)
        if assignment.face is None:
            raise ValueError(f"Face for assignment {assignment_id} no longer exists")
        new_person = self._session.get(Person, new_person_id)
        if new_person is None:
            raise ValueError(f"Person id={new_person_id} not found")

        self._correct_one(assignment, new_person_id)
        self._delete_orphan_auto_persons()
        self._session.commit()
        return assignment

    def correct_assignment_to_new_person(
        self, assignment_id: int, new_person_name: str
    ) -> AutoAssignment:
        """Correction variant that first creates a brand-new named person."""
        name = new_person_name.strip()
        if not name:
            raise ValueError("New person name must not be empty")
        person = Person(name=name, is_auto_named=False)
        self._session.add(person)
        self._session.flush()
        return self.correct_assignment(assignment_id, person.id)

    def correct_assignments(
        self, assignment_ids: List[int], new_person_id: int
    ) -> int:
        """Move many groupings to *new_person_id* in one transaction.

        Skips ids whose row or face no longer exists. Returns the count moved.
        """
        new_person = self._session.get(Person, new_person_id)
        if new_person is None:
            raise ValueError(f"Person id={new_person_id} not found")
        moved = 0
        for assignment_id in assignment_ids:
            assignment = self._session.get(AutoAssignment, assignment_id)
            if assignment is None or assignment.face is None:
                continue
            self._correct_one(assignment, new_person_id)
            moved += 1
        self._delete_orphan_auto_persons()
        self._session.commit()
        log.info("Deep recognition: batch-corrected %d assignments", moved)
        return moved

    def correct_assignments_to_new_person(
        self, assignment_ids: List[int], new_person_name: str
    ) -> int:
        """Batch correction that first creates a brand-new named person."""
        name = new_person_name.strip()
        if not name:
            raise ValueError("New person name must not be empty")
        person = Person(name=name, is_auto_named=False)
        self._session.add(person)
        self._session.flush()
        return self.correct_assignments(assignment_ids, person.id)

    def _correct_one(self, assignment: AutoAssignment, new_person_id: int) -> None:
        """Shared correction core; caller validates the person and commits.

        Records a "different person" judgement against the wrong person and a
        "same person" judgement towards the right one, so future runs are
        vetoed from repeating the mistake and gain a confirmed example.
        """
        face = assignment.face
        wrong_person_id = assignment.person_id
        face.person_id = new_person_id
        face.assignment_source = "manual"
        face.assignment_confidence = 1.0
        face.assigned_at = _utcnow_naive()

        self._record_correction_pair(face, wrong_person_id, same_person=False)
        self._record_correction_pair(face, new_person_id, same_person=True)
        if wrong_person_id != new_person_id:
            self._add_veto(face.id, wrong_person_id, "correct")
        self._remove_veto(face.id, new_person_id)

        assignment.status = AUTO_ASSIGN_STATUS_CORRECTED
        assignment.corrected_person_id = new_person_id
        assignment.decided_at = _utcnow_naive()

    def revert_assignment(self, assignment_id: int) -> AutoAssignment:
        """User sends the face back to where it was before the run."""
        assignment = self._require_assignment(assignment_id)
        self._revert_one(assignment)
        self._session.commit()
        return assignment

    def revert_assignments(self, assignment_ids: List[int]) -> int:
        """Reject (undo) many groupings in one transaction.

        Skips ids that no longer exist. Returns the count reverted.
        """
        reverted = 0
        for assignment_id in assignment_ids:
            assignment = self._session.get(AutoAssignment, assignment_id)
            if assignment is None:
                continue
            self._revert_one(assignment)
            reverted += 1
        self._session.commit()
        log.info("Deep recognition: batch-reverted %d assignments", reverted)
        return reverted

    def revert_all_open(self, run_id: Optional[int] = None) -> int:
        """Revert every still-open (auto) assignment of *run_id* in one go.

        Recovery hatch for a run gone wrong (e.g. an avalanche of faces onto
        one person): each face returns to where it was before the run and a
        "different person" judgement is recorded so the engine learns from
        the mass rejection too. Returns the number of reverted assignments.
        """
        if run_id is None:
            run_id = self.latest_assignment_run_id()
            if run_id is None:
                return 0
        open_assignments = (
            self._session.query(AutoAssignment)
            .filter(AutoAssignment.training_run_id == run_id)
            .filter(AutoAssignment.status == AUTO_ASSIGN_STATUS_AUTO)
            .all()
        )
        for assignment in open_assignments:
            self._revert_one(assignment)
        self._session.commit()
        log.info(
            "Deep recognition: bulk-reverted %d open assignments of run %s",
            len(open_assignments), run_id,
        )
        return len(open_assignments)

    def _revert_one(self, assignment: AutoAssignment) -> None:
        """Shared revert core; caller commits.

        A rejected near-match suggestion leaves the face where it is and only
        records the "not this person" judgement.
        """
        face = assignment.face
        if face is not None and assignment.status == AUTO_ASSIGN_STATUS_NEAR:
            self._record_correction_pair(face, assignment.person_id, same_person=False)
            self._add_veto(face.id, assignment.person_id, "near_reject")
        elif face is not None and face.person_id == assignment.person_id:
            previous = (
                self._session.get(Person, assignment.previous_person_id)
                if assignment.previous_person_id is not None
                else None
            )
            face.person_id = previous.id if previous is not None else None
            face.assignment_source = None
            face.assignment_confidence = None
            face.assigned_at = None
            self._record_correction_pair(
                face, assignment.person_id, same_person=False
            )
            self._add_veto(face.id, assignment.person_id, "revert")
        assignment.status = AUTO_ASSIGN_STATUS_REVERTED
        assignment.decided_at = _utcnow_naive()

    # ------------------------------------------------------------------
    # Active learning: near-match suggestions
    # ------------------------------------------------------------------

    def list_near_suggestions(self, limit: int = 200) -> List[AutoAssignmentDTO]:
        """Open near-match suggestions, most promising first.

        A suggestion goes stale (hidden) once the face got a named person
        some other way.
        """
        rows = (
            self._session.query(AutoAssignment)
            .filter(AutoAssignment.status == AUTO_ASSIGN_STATUS_NEAR)
            .order_by(AutoAssignment.score.desc(), AutoAssignment.id)
            .limit(limit)
            .all()
        )
        dtos: List[AutoAssignmentDTO] = []
        for assignment in rows:
            face = assignment.face
            if face is None:
                continue
            owner = face.person
            if owner is not None and not owner.is_auto_named and not owner.is_protected:
                continue
            person = assignment.person
            dtos.append(
                AutoAssignmentDTO(
                    assignment_id=assignment.id,
                    face_id=face.id,
                    person_id=assignment.person_id,
                    person_name=person.name if person else "?",
                    previous_person_name=assignment.previous_person_name,
                    score=assignment.score,
                    status=assignment.status,
                    crop_path=face.crop_path,
                    image_path=face.image.file_path if face.image else None,
                    bbox=(face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h),
                    run_id=assignment.training_run_id,
                    run_finished_at=(
                        assignment.training_run.finished_at
                        if assignment.training_run
                        else None
                    ),
                    decision=None,
                )
            )
        return dtos

    # ------------------------------------------------------------------
    # Training-data hygiene: suspicious label review
    # ------------------------------------------------------------------

    def list_suspicious_labels(self) -> List[SuspiciousLabelDTO]:
        """Labeled faces the noise filter flagged, with the lookalike person."""
        faces: List[Face] = (
            self._session.query(Face)
            .filter(Face.label_review_status == LABEL_REVIEW_SUSPICIOUS)
            .filter(Face.person_id.isnot(None))
            .order_by(Face.id)
            .all()
        )
        if not faces:
            return []
        details: Dict[int, Tuple[int, float, float]] = {}
        for run in (
            self._session.query(TrainingRun)
            .filter(TrainingRun.metrics_json.isnot(None))
            .order_by(TrainingRun.id.desc())
            .limit(20)
            .all()
        ):
            try:
                rows = json.loads(run.metrics_json).get("suspicious") or []
            except (ValueError, TypeError, AttributeError):
                continue
            for row in rows:
                if len(row) >= 5 and int(row[0]) not in details:
                    details[int(row[0])] = (int(row[2]), float(row[3]), float(row[4]))
        names = dict(self._session.query(Person.id, Person.name).all())
        result: List[SuspiciousLabelDTO] = []
        for face in faces:
            other_id, own_sim, other_sim = details.get(face.id, (None, None, None))
            result.append(
                SuspiciousLabelDTO(
                    face_id=face.id,
                    person_id=face.person_id,
                    person_name=names.get(face.person_id, "?"),
                    lookalike_person_id=other_id,
                    lookalike_name=names.get(other_id) if other_id is not None else None,
                    own_similarity=own_sim,
                    other_similarity=other_sim,
                    crop_path=face.crop_path,
                    image_path=face.image.file_path if face.image else None,
                    bbox=(face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h),
                )
            )
        return result

    def count_suspicious_labels(self) -> int:
        return (
            self._session.query(Face)
            .filter(Face.label_review_status == LABEL_REVIEW_SUSPICIOUS)
            .filter(Face.person_id.isnot(None))
            .count()
        )

    def keep_label(self, face_id: int) -> None:
        """The label is right after all — train on it and never flag it again."""
        face = self._session.get(Face, face_id)
        if face is None:
            raise ValueError(f"Face id={face_id} not found")
        face.label_review_status = LABEL_REVIEW_KEPT
        self._session.commit()

    def remove_label(self, face_id: int) -> None:
        """The label is wrong — the face returns to the unknown pool.

        A veto keeps the engine from putting it back onto the same person.
        """
        face = self._session.get(Face, face_id)
        if face is None:
            raise ValueError(f"Face id={face_id} not found")
        wrong_person_id = face.person_id
        if wrong_person_id is not None:
            self._record_correction_pair(face, wrong_person_id, same_person=False)
            self._add_veto(face.id, wrong_person_id, "label_review")
        face.person_id = None
        face.assignment_source = None
        face.assignment_confidence = None
        face.assigned_at = None
        face.label_review_status = None
        self._session.commit()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_assignment(self, assignment_id: int) -> AutoAssignment:
        assignment = self._session.get(AutoAssignment, assignment_id)
        if assignment is None:
            raise ValueError(f"AutoAssignment id={assignment_id} not found")
        return assignment

    def _add_veto(self, face_id: int, person_id: Optional[int], source: str) -> None:
        """Record "face is NOT person" (idempotent)."""
        if person_id is None:
            return
        exists = (
            self._session.query(FaceVeto.id)
            .filter(FaceVeto.face_id == face_id, FaceVeto.person_id == person_id)
            .first()
        )
        if exists is None:
            self._session.add(
                FaceVeto(face_id=face_id, person_id=person_id, source=source)
            )
            self._session.flush()

    def _remove_veto(self, face_id: int, person_id: int) -> None:
        """The user now says the face IS this person — drop an old veto."""
        self._session.query(FaceVeto).filter(
            FaceVeto.face_id == face_id, FaceVeto.person_id == person_id
        ).delete(synchronize_session="fetch")

    def _record_correction_pair(
        self, face: Face, person_id: int, *, same_person: bool
    ) -> None:
        """Persist a FaceCorrection between *face* and a face of *person_id*."""
        other = (
            self._session.query(Face)
            .filter(Face.person_id == person_id)
            .filter(Face.id != face.id)
            .filter(Face.is_excluded == False)  # noqa: E712
            .order_by(Face.id)
            .first()
        )
        if other is None:
            return
        a, b = (face.id, other.id) if face.id < other.id else (other.id, face.id)
        existing = (
            self._session.query(FaceCorrection)
            .filter(FaceCorrection.face_id_a == a)
            .filter(FaceCorrection.face_id_b == b)
            .first()
        )
        if existing is not None:
            existing.same_person = same_person
            return
        self._session.add(
            FaceCorrection(face_id_a=a, face_id_b=b, same_person=same_person)
        )

    def _load_candidate_faces(self) -> List[Face]:
        """Embedded faces that are still unknown (or in an auto-named group)."""
        query = (
            self._session.query(Face)
            .outerjoin(Person, Face.person_id == Person.id)
            .filter(Face.embedding_exists())
            .filter(Face.is_excluded == False)  # noqa: E712
            .filter(
                or_(
                    Face.person_id.is_(None),
                    Person.is_auto_named == True,  # noqa: E712
                    Person.is_protected == True,  # noqa: E712
                )
            )
        )
        if self._config.strict_quality_filter:
            query = query.filter(
                or_(Face.is_low_quality.is_(None), Face.is_low_quality == False)  # noqa: E712
            )
        return query.all()

    def _load_correction_vetoes(
        self, candidates: List[Face]
    ) -> Dict[int, Set[int]]:
        """Map candidate face id → person ids vetoed by 'different' judgements.

        Union of the durable :class:`FaceVeto` rows and the legacy pair-based
        judgements (face ↔ the partner face's current person).
        """
        candidate_ids = {face.id for face in candidates}
        if not candidate_ids:
            return {}

        vetoes: Dict[int, Set[int]] = {}
        id_list = list(candidate_ids)
        for start in range(0, len(id_list), 900):
            for face_id, person_id in (
                self._session.query(FaceVeto.face_id, FaceVeto.person_id)
                .filter(FaceVeto.face_id.in_(id_list[start:start + 900]))
                .all()
            ):
                vetoes.setdefault(int(face_id), set()).add(int(person_id))

        corrections: List[FaceCorrection] = (
            self._session.query(FaceCorrection)
            .filter(FaceCorrection.same_person == False)  # noqa: E712
            .filter(
                or_(
                    FaceCorrection.face_id_a.in_(candidate_ids),
                    FaceCorrection.face_id_b.in_(candidate_ids),
                )
            )
            .all()
        )
        if not corrections:
            return vetoes

        other_ids: Set[int] = set()
        for corr in corrections:
            if corr.face_id_a in candidate_ids:
                other_ids.add(corr.face_id_b)
            if corr.face_id_b in candidate_ids:
                other_ids.add(corr.face_id_a)

        person_of: Dict[int, Optional[int]] = {
            face.id: face.person_id
            for face in self._session.query(Face)
            .filter(Face.id.in_(other_ids))
            .all()
        }

        for corr in corrections:
            if corr.face_id_a in candidate_ids:
                target = person_of.get(corr.face_id_b)
                if target is not None:
                    vetoes.setdefault(corr.face_id_a, set()).add(target)
            if corr.face_id_b in candidate_ids:
                target = person_of.get(corr.face_id_a)
                if target is not None:
                    vetoes.setdefault(corr.face_id_b, set()).add(target)
        return vetoes

    def _delete_orphan_auto_persons(self) -> int:
        orphans: List[Person] = (
            self._session.query(Person)
            .filter(Person.is_auto_named == True)  # noqa: E712
            .filter(~Person.faces.any())
            .all()
        )
        for person in orphans:
            self._session.delete(person)
        return len(orphans)
