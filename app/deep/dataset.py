"""Training dataset construction for the deep recognition engine.

Collects every *trusted* labeled face from the database and turns it into a
matrix of L2-normalised embeddings plus person labels.  The trust rules mirror
the rest of the application:

* faces assigned to a real (non-auto-named, non-protected) person count;
* manual sources are always trusted — the user explicitly chose those faces;
* automatic assignments only count above a confidence floor, so one early
  mistake cannot poison the model;
* excluded faces, faces without embeddings and faces whose label is under
  "suspicious" review never count.

Besides the positives the dataset carries what the engine needs to be
*accurate* rather than merely confident:

* **group keys** — faces of one photo session (same day, else same image) are
  kept together during cross-validation so near-duplicate burst shots cannot
  leak between the training and the test side of a fold;
* **probe metadata** — quality, face size and embedding consistency of every
  example (features of the calibrated meta-classifier);
* **hard negatives** — real "this face is NOT person X" judgements (vetoes,
  reverted / corrected automatic assignments, "different person" pairs);
* **unknown background** — unlabeled faces that resemble no known person,
  used to train an explicit "unknown" output.

The more faces a person has, the more training examples the network sees for
them — recognition quality grows with every confirmed face.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.db.models import (
    AUTO_ASSIGN_STATUS_CORRECTED,
    AUTO_ASSIGN_STATUS_REVERTED,
    LABEL_REVIEW_KEPT,
    LABEL_REVIEW_SUSPICIOUS,
    AutoAssignment,
    Face,
    FaceCorrection,
    FaceVeto,
    Image,
    Person,
)

log = logging.getLogger(__name__)

# Bump when the engine's training/decision logic changes in a way that must
# invalidate every persisted model, even if the labeled data is unchanged.
ENGINE_VERSION = "2"

# Sources that mark a human decision (legacy rows have NULL source).
TRUSTED_MANUAL_SOURCES = {
    None,
    "manual",
    "manual_merge",
    "suggestion_approved",
    "deep_confirmed",
}

# Automatic sources that may be reused for training above a confidence floor.
AUTO_SOURCES = {"recognition", "deep_recognition", "clustering"}

# How many unlabeled faces are sampled when looking for "unknown" background.
_UNKNOWN_SAMPLE_FACTOR = 5


@dataclass
class ProbeMeta:
    """Per-face side information used as meta-classifier features."""

    quality: Optional[float] = None       # Face.quality_score (0..1)
    size: Optional[float] = None          # sqrt(bbox area) in pixels
    consistency: Optional[float] = None   # flip-TTA embedding agreement

    @classmethod
    def from_face(cls, face: Face) -> "ProbeMeta":
        try:
            size = float(np.sqrt(max(0, face.bbox_w) * max(0, face.bbox_h)))
        except TypeError:
            size = None
        return cls(
            quality=face.quality_score,
            size=size,
            consistency=getattr(face, "embedding_consistency", None),
        )


@dataclass
class TrainingDataset:
    """Labeled embeddings ready for classifier training."""

    embeddings: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    face_ids: List[int] = field(default_factory=list)
    person_names: Dict[int, str] = field(default_factory=dict)
    # True for each training example that comes from a manual (human-confirmed) source.
    manual_flags: List[bool] = field(default_factory=list)
    # True when the user explicitly kept a label the noise filter flagged.
    kept_flags: List[bool] = field(default_factory=list)
    # Cross-validation group per example (photo day / image); empty → one
    # group per example.
    group_keys: List[str] = field(default_factory=list)
    probes: List[ProbeMeta] = field(default_factory=list)
    # face_id pairs the user marked as "different person" / "same person"
    diff_pairs: Set[Tuple[int, int]] = field(default_factory=set)
    same_pairs: Set[Tuple[int, int]] = field(default_factory=set)
    # Hard negatives: embedding i is NOT person neg_person_ids[i].
    neg_embeddings: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    neg_person_ids: List[int] = field(default_factory=list)
    neg_face_ids: List[int] = field(default_factory=list)
    neg_probes: List[ProbeMeta] = field(default_factory=list)
    # Unlabeled faces that resemble nobody known ("unknown" background class).
    unknown_embeddings: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    unknown_face_ids: List[int] = field(default_factory=list)
    # Identifier of the embedding model/geometry (part of the fingerprint).
    embedder_id: str = ""

    @property
    def n_examples(self) -> int:
        return int(self.labels.shape[0])

    @property
    def n_persons(self) -> int:
        return len(set(self.labels.tolist())) if self.n_examples else 0

    @property
    def n_negatives(self) -> int:
        return len(self.neg_person_ids)

    @property
    def n_unknown(self) -> int:
        return len(self.unknown_face_ids)

    def examples_per_person(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for label in self.labels.tolist():
            counts[label] = counts.get(label, 0) + 1
        return counts

    def groups(self) -> np.ndarray:
        """Cross-validation group id per example (one group each by default)."""
        if len(self.group_keys) == self.n_examples:
            _, inverse = np.unique(
                np.asarray(self.group_keys, dtype=str), return_inverse=True
            )
            return inverse.astype(np.int64)
        return np.arange(self.n_examples, dtype=np.int64)

    def probe(self, idx: int) -> ProbeMeta:
        return self.probes[idx] if idx < len(self.probes) else ProbeMeta()

    def subset(self, mask: np.ndarray) -> "TrainingDataset":
        """A copy holding only the positives selected by boolean *mask*.

        Negatives / unknown background / judgements are shared unchanged.
        """
        mask = np.asarray(mask, dtype=bool)
        idx = np.flatnonzero(mask)

        def pick(values: list) -> list:
            return [values[i] for i in idx] if len(values) == len(mask) else []

        labels = self.labels[mask]
        kept_ids = set(labels.tolist())
        return TrainingDataset(
            embeddings=self.embeddings[mask] if self.n_examples else self.embeddings,
            labels=labels,
            face_ids=pick(self.face_ids),
            person_names={
                pid: name for pid, name in self.person_names.items() if pid in kept_ids
            },
            manual_flags=pick(self.manual_flags),
            kept_flags=pick(self.kept_flags),
            group_keys=pick(self.group_keys),
            probes=pick(self.probes),
            diff_pairs=self.diff_pairs,
            same_pairs=self.same_pairs,
            neg_embeddings=self.neg_embeddings,
            neg_person_ids=self.neg_person_ids,
            neg_face_ids=self.neg_face_ids,
            neg_probes=self.neg_probes,
            unknown_embeddings=self.unknown_embeddings,
            unknown_face_ids=self.unknown_face_ids,
            embedder_id=self.embedder_id,
        )

    def fingerprint(self) -> str:
        """Stable hash of everything the model learns from.

        Covers the labels, the embedding *values* (a re-embed with another
        model, crop mode or TTA must retrain even when ids are unchanged), the
        hard negatives, the unknown background and the engine version.
        """
        hasher = hashlib.sha256()
        hasher.update(f"engine={ENGINE_VERSION};emb={self.embedder_id}".encode())
        order = np.argsort(np.asarray(self.face_ids))
        for idx in order:
            hasher.update(str(self.face_ids[idx]).encode())
            hasher.update(str(int(self.labels[idx])).encode())
            if self.embeddings.ndim == 2 and idx < self.embeddings.shape[0]:
                # float16 rounding: insensitive to float noise, sensitive to
                # any real re-embedding.
                hasher.update(self.embeddings[idx].astype(np.float16).tobytes())
        hasher.update(str(self.embeddings.shape).encode())
        for fid, pid in sorted(zip(self.neg_face_ids, self.neg_person_ids)):
            hasher.update(f"n{fid}:{pid}".encode())
        for fid in sorted(self.unknown_face_ids):
            hasher.update(f"u{fid}".encode())
        return hasher.hexdigest()


def _normalise(vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if vector is None:
        return None
    vec = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    return vec / norm


def is_trusted_training_face(
    face: Face,
    *,
    auto_min_confidence: float,
    use_auto_assignments: bool,
    exclude_low_quality: bool,
) -> bool:
    """Whether *face* may serve as a training example for its person."""
    if face.is_excluded or face.person is None:
        return False
    if face.person.is_auto_named or face.person.is_protected:
        return False
    if getattr(face, "label_review_status", None) == LABEL_REVIEW_SUSPICIOUS:
        # Flagged by the label-noise filter, awaiting the user's decision.
        return False
    if face.get_embedding() is None:
        return False

    if face.assignment_source in TRUSTED_MANUAL_SOURCES:
        # Human-confirmed faces bypass the quality filter on purpose.
        return True

    if not use_auto_assignments:
        return False
    if exclude_low_quality and face.is_low_quality:
        return False
    if face.assignment_source not in AUTO_SOURCES:
        return False
    confidence = face.assignment_confidence
    return confidence is not None and confidence >= auto_min_confidence


def session_group_key(face: Face) -> str:
    """Cross-validation group of *face*: its photo day, else its image.

    Faces from one day are usually one event (burst shots, same clothes and
    light) — splitting them across folds would grade the model on near-copies
    of its own training data and make every threshold too permissive.
    """
    image: Optional[Image] = face.image
    raw = None
    if image is not None:
        raw = image.photo_date or image.estimated_date
    if raw:
        from app.utils.fuzzy_date import FuzzyDate, Precision

        parsed = FuzzyDate.parse(raw)
        if parsed.precision is Precision.EXACT and parsed.earliest is not None:
            return f"d:{parsed.earliest.isoformat()}"
    return f"i:{face.image_id}"


def build_training_dataset(
    session: Session,
    *,
    auto_min_confidence: float = 0.97,
    use_auto_assignments: bool = True,
    exclude_low_quality: bool = True,
    embedder_id: str = "",
    collect_unknown: bool = True,
    unknown_max_similarity: float = 0.40,
    max_unknown: int = 400,
) -> TrainingDataset:
    """Collect all trusted labeled faces into a :class:`TrainingDataset`."""
    persons: List[Person] = (
        session.query(Person)
        .filter(Person.is_auto_named == False)  # noqa: E712
        .filter(Person.is_protected == False)  # noqa: E712
        .all()
    )

    vectors: List[np.ndarray] = []
    labels: List[int] = []
    face_ids: List[int] = []
    manual_flags: List[bool] = []
    kept_flags: List[bool] = []
    group_keys: List[str] = []
    probes: List[ProbeMeta] = []
    person_names: Dict[int, str] = {}

    for person in persons:
        for face in person.faces:
            if not is_trusted_training_face(
                face,
                auto_min_confidence=auto_min_confidence,
                use_auto_assignments=use_auto_assignments,
                exclude_low_quality=exclude_low_quality,
            ):
                continue
            vec = _normalise(face.get_embedding())
            if vec is None:
                continue
            vectors.append(vec)
            labels.append(person.id)
            face_ids.append(face.id)
            manual_flags.append(face.assignment_source in TRUSTED_MANUAL_SOURCES)
            kept_flags.append(
                getattr(face, "label_review_status", None) == LABEL_REVIEW_KEPT
            )
            group_keys.append(session_group_key(face))
            probes.append(ProbeMeta.from_face(face))
            person_names[person.id] = person.name

    dataset = TrainingDataset(
        embeddings=(
            np.vstack(vectors).astype(np.float32)
            if vectors
            else np.zeros((0, 0), dtype=np.float32)
        ),
        labels=np.asarray(labels, dtype=np.int64),
        face_ids=face_ids,
        manual_flags=manual_flags,
        kept_flags=kept_flags,
        group_keys=group_keys,
        probes=probes,
        person_names=person_names,
        embedder_id=embedder_id,
    )
    _attach_correction_pairs(session, dataset)
    _attach_hard_negatives(session, dataset)
    if collect_unknown:
        _attach_unknown_background(
            session, dataset,
            max_similarity=unknown_max_similarity,
            max_unknown=max_unknown,
        )

    log.info(
        "Deep training dataset: %d example(s) across %d person(s), "
        "%d hard negative(s), %d unknown background face(s)",
        dataset.n_examples,
        dataset.n_persons,
        dataset.n_negatives,
        dataset.n_unknown,
    )
    return dataset


def _attach_correction_pairs(session: Session, dataset: TrainingDataset) -> None:
    """Load the user's same/not-same judgements as training constraints."""
    corrections: List[FaceCorrection] = session.query(FaceCorrection).all()
    for corr in corrections:
        pair = (
            (corr.face_id_a, corr.face_id_b)
            if corr.face_id_a < corr.face_id_b
            else (corr.face_id_b, corr.face_id_a)
        )
        if corr.same_person:
            dataset.same_pairs.add(pair)
        else:
            dataset.diff_pairs.add(pair)


def _attach_hard_negatives(session: Session, dataset: TrainingDataset) -> None:
    """Collect every real "face F is NOT person P" judgement.

    Sources: :class:`FaceVeto` rows, reverted / corrected automatic
    assignments and "different person" pairs (face ↔ the partner's current
    person).  Only persons the model knows matter; a judgement contradicted by
    the face's current label is dropped (the user later changed their mind).
    """
    known = set(dataset.labels.tolist())
    if not known:
        return
    pairs: Set[Tuple[int, int]] = set()

    for face_id, person_id in session.query(FaceVeto.face_id, FaceVeto.person_id).all():
        pairs.add((int(face_id), int(person_id)))
    for face_id, person_id in (
        session.query(AutoAssignment.face_id, AutoAssignment.person_id)
        .filter(
            AutoAssignment.status.in_(
                (AUTO_ASSIGN_STATUS_REVERTED, AUTO_ASSIGN_STATUS_CORRECTED)
            )
        )
        .all()
    ):
        pairs.add((int(face_id), int(person_id)))
    if dataset.diff_pairs:
        ids = {fid for pair in dataset.diff_pairs for fid in pair}
        person_of = dict(
            session.query(Face.id, Face.person_id).filter(Face.id.in_(ids)).all()
        )
        for a, b in dataset.diff_pairs:
            pb, pa = person_of.get(b), person_of.get(a)
            if pb is not None:
                pairs.add((a, int(pb)))
            if pa is not None:
                pairs.add((b, int(pa)))

    pairs = {(f, p) for f, p in pairs if p in known}
    if not pairs:
        return
    faces = {
        face.id: face
        for face in session.query(Face).filter(
            Face.id.in_({f for f, _ in pairs})
        ).all()
    }
    vectors: List[np.ndarray] = []
    for face_id, person_id in sorted(pairs):
        face = faces.get(face_id)
        if face is None or face.is_excluded or face.person_id == person_id:
            continue
        vec = _normalise(face.get_embedding())
        if vec is None:
            continue
        if dataset.embeddings.ndim == 2 and vec.shape[0] != dataset.embeddings.shape[1]:
            continue
        vectors.append(vec)
        dataset.neg_face_ids.append(face_id)
        dataset.neg_person_ids.append(person_id)
        dataset.neg_probes.append(ProbeMeta.from_face(face))
    if vectors:
        dataset.neg_embeddings = np.vstack(vectors).astype(np.float32)


def _attach_unknown_background(
    session: Session,
    dataset: TrainingDataset,
    *,
    max_similarity: float,
    max_unknown: int,
) -> None:
    """Pick unlabeled faces that resemble no labeled face at all.

    These are strangers (or non-faces that slipped through) — exactly what the
    network must learn to answer "nobody I know" for.  Faces even moderately
    similar to a known person are skipped: they may be that person's not yet
    recognised photos.  A deterministic, evenly spread sample keeps the cost
    bounded on large archives.
    """
    if dataset.n_examples == 0 or max_unknown <= 0:
        return
    candidate_ids = [
        fid
        for (fid,) in session.query(Face.id)
        .outerjoin(Person, Face.person_id == Person.id)
        .filter(Face.embedding_exists())
        .filter(Face.is_excluded == False)  # noqa: E712
        .filter(or_(Face.person_id.is_(None), Person.is_auto_named == True))  # noqa: E712
        .order_by(Face.id)
        .all()
    ]
    if not candidate_ids:
        return
    budget = max_unknown * _UNKNOWN_SAMPLE_FACTOR
    if len(candidate_ids) > budget:
        step = len(candidate_ids) / float(budget)
        candidate_ids = [candidate_ids[int(i * step)] for i in range(budget)]

    train = dataset.embeddings
    vectors: List[np.ndarray] = []
    ids: List[int] = []
    for start in range(0, len(candidate_ids), 500):
        chunk = candidate_ids[start:start + 500]
        for face in session.query(Face).filter(Face.id.in_(chunk)).all():
            vec = _normalise(face.get_embedding())
            if vec is None or vec.shape[0] != train.shape[1]:
                continue
            if float(np.max(train @ vec)) >= max_similarity:
                continue
            vectors.append(vec)
            ids.append(face.id)
        if len(ids) >= max_unknown:
            break
    if not vectors:
        return
    order = np.argsort(np.asarray(ids))[:max_unknown]
    dataset.unknown_embeddings = np.vstack([vectors[i] for i in order]).astype(np.float32)
    dataset.unknown_face_ids = [ids[i] for i in order]
