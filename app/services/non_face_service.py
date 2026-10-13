"""The "Not a face" feedback and the learned non-face filter.

* :meth:`NonFaceService.mark_not_face` — the user's "Not a face" action: the
  box is hidden and its embedding stored as a negative example
  (``IgnoredFace.kind == "not_face"``, so the ignore list's near-duplicate
  suppression covers it too).
* :meth:`NonFaceService.apply` — runs in every AI scan: trains
  :mod:`app.deep.non_face` from the examples and hides new / Unknown boxes it
  is sure are not faces (``assignment_source == "non_face_model"``).  Every run
  re-evaluates the earlier hides, so removing an example un-hides what it
  caused.
* :meth:`NonFaceService.confirm_face` — the user restores a wrongly hidden box:
  it comes back and its embedding becomes a positive ("is a face") example, so
  the model never hides it again.

Faces of named people, manually drawn boxes and human-confirmed assignments
are never hidden automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable, List, Optional, Sequence

import numpy as np
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app.config import NonFaceConfig
from app.db.models import (
    IGNORED_KIND_IS_FACE,
    IGNORED_KIND_NOT_FACE,
    Face,
    IgnoredFace,
    Person,
)

log = logging.getLogger(__name__)

# Face.assignment_source values written here.
SOURCE_NOT_FACE = "not_face"          # the user's "Not a face" mark
SOURCE_MODEL = "non_face_model"       # hidden by the learned filter

# Explicit human decisions — such faces are never hidden automatically.
_HUMAN_SOURCES = frozenset(
    {"manual", "manual_merge", "suggestion_approved", "deep_confirmed"}
)


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@dataclass
class NonFaceFilterStats:
    """Outcome of one :meth:`NonFaceService.apply` run."""

    active: bool = False
    reason: str = ""            # why the model is inactive
    n_positives: int = 0
    n_negatives: int = 0
    n_candidates: int = 0
    n_hidden: int = 0           # newly hidden this run
    n_restored: int = 0         # earlier hides the model no longer stands by
    threshold: Optional[float] = None


class NonFaceService:
    """User feedback + learned filter for boxes that are not faces."""

    def __init__(self, session: Session, config: Optional[NonFaceConfig] = None) -> None:
        self._session = session
        self._config = config or NonFaceConfig()

    # ------------------------------------------------------------------
    # User feedback
    # ------------------------------------------------------------------

    def mark_not_face(self, face_ids: Iterable[int]) -> int:
        """Hide each face and store it as a "not a face" example.

        Faces without an embedding are skipped (there is nothing to learn
        from yet).  Returns the number of faces marked.
        """
        marked = 0
        emptied: set[int] = set()
        now = _utcnow_naive()
        for face_id in face_ids:
            face = self._session.get(Face, face_id)
            if face is None or face.get_embedding() is None:
                continue
            if face.assignment_source == SOURCE_NOT_FACE:
                marked += 1  # already an example
                continue
            person = face.person
            self._add_example(face, IGNORED_KIND_NOT_FACE, person.name if person else None)
            face.is_excluded = True
            face.person_id = None
            face.assignment_source = SOURCE_NOT_FACE
            face.assignment_confidence = None
            face.assigned_at = now
            if person is not None:
                emptied.add(person.id)
            marked += 1
        self._session.flush()
        self._delete_empty_auto_persons(emptied)
        self._session.commit()
        log.info("Marked %d face(s) as not a face", marked)
        return marked

    def confirm_face(self, face_id: int) -> bool:
        """Restore a box the model hid; it becomes an "is a face" example."""
        face = self._session.get(Face, face_id)
        if face is None or face.assignment_source != SOURCE_MODEL:
            return False
        if face.get_embedding() is not None:
            self._add_example(face, IGNORED_KIND_IS_FACE, None)
        face.is_excluded = False
        face.assignment_source = None
        face.assignment_confidence = None
        face.assigned_at = None
        self._session.commit()
        log.info("Face %d restored from the non-face filter (now a face example)", face_id)
        return True

    def list_model_hidden(self) -> List[Face]:
        """Faces the learned filter currently hides, most confident first."""
        return (
            self._session.query(Face)
            .filter(Face.assignment_source == SOURCE_MODEL)
            .filter(Face.is_excluded == True)  # noqa: E712
            .order_by(Face.assignment_confidence.desc(), Face.id)
            .all()
        )

    # ------------------------------------------------------------------
    # Pipeline filter
    # ------------------------------------------------------------------

    def apply(self) -> NonFaceFilterStats:
        """Train on the current examples and (re)decide every candidate box."""
        from app.deep.non_face import train_non_face_model

        stats = NonFaceFilterStats()
        cfg = self._config
        if not cfg.enabled:
            stats.reason = "disabled"
            stats.n_restored = self._restore_all_model_hidden()
            return stats

        positives, dim = self._load_positives()
        negatives = self._load_examples(IGNORED_KIND_NOT_FACE, dim)
        stats.n_positives, stats.n_negatives = len(positives), len(negatives)
        model = None
        if dim is not None:
            model = train_non_face_model(
                positives,
                negatives,
                min_negatives=cfg.min_negatives,
                min_positives=cfg.min_positives,
                min_probability=cfg.min_probability,
                max_positive_fpr=cfg.max_positive_fpr,
                max_positives=cfg.max_positives,
            )
        if model is None:
            stats.reason = (
                f"needs {cfg.min_negatives} 'not a face' and {cfg.min_positives} "
                f"confirmed face example(s) (have {len(negatives)} / {len(positives)})"
            )
            stats.n_restored = self._restore_all_model_hidden()
            return stats

        stats.active = True
        stats.threshold = model.threshold
        candidates = [f for f in self._candidates() if _dim(f) == model.dim]
        stats.n_candidates = len(candidates)
        if candidates:
            matrix = np.vstack([f.get_embedding() for f in candidates])
            mask, prob = model.non_face_mask(matrix)
            now = _utcnow_naive()
            emptied: set[int] = set()
            for face, flagged, p in zip(candidates, mask, prob):
                hidden = face.assignment_source == SOURCE_MODEL
                if flagged and not hidden:
                    if face.person_id is not None:
                        emptied.add(face.person_id)
                    face.is_excluded = True
                    face.person_id = None
                    face.assignment_source = SOURCE_MODEL
                    face.assignment_confidence = float(p)
                    face.assigned_at = now
                    stats.n_hidden += 1
                elif hidden and not flagged:
                    self._unhide(face)
                    stats.n_restored += 1
                elif hidden:
                    face.assignment_confidence = float(p)
            self._session.flush()
            self._delete_empty_auto_persons(emptied)
        self._session.commit()
        log.info(
            "Non-face filter: %d candidate(s), %d hidden, %d restored",
            stats.n_candidates, stats.n_hidden, stats.n_restored,
        )
        return stats

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _add_example(self, face: Face, kind: str, source_name: Optional[str]) -> None:
        entry = IgnoredFace(
            kind=kind,
            thumbnail_path=face.crop_path,
            source_face_id=face.id,
            source_person_name=source_name,
        )
        entry.set_embedding(face.get_embedding())
        self._session.add(entry)

    def _load_positives(self) -> tuple[np.ndarray, Optional[int]]:
        """Confirmed real faces: human-assigned faces of named people, manually
        drawn boxes and restored ("is a face") examples.  Returns the matrix and
        the embedding dimension in use (the most common one)."""
        faces = (
            self._session.query(Face)
            .outerjoin(Person, Face.person_id == Person.id)
            .filter(Face.embedding_exists())
            .filter(Face.is_excluded == False)  # noqa: E712
            .filter(
                or_(
                    Face.detector_backend == "manual",
                    and_(
                        Person.is_auto_named == False,  # noqa: E712
                        Person.is_protected == False,  # noqa: E712
                        _human_source(),
                    ),
                )
            )
            .all()
        )
        vectors = [f.get_embedding() for f in faces]
        vectors += [
            e.get_embedding()
            for e in self._session.query(IgnoredFace)
            .filter(IgnoredFace.kind == IGNORED_KIND_IS_FACE)
            .all()
        ]
        vectors = [v for v in vectors if v is not None and v.size]
        if not vectors:
            return np.zeros((0, 0), np.float32), None
        dims = [v.size for v in vectors]
        dim = max(set(dims), key=dims.count)
        return _stack([v for v in vectors if v.size == dim], dim), dim

    def _load_examples(self, kind: str, dim: Optional[int]) -> np.ndarray:
        vectors = [
            e.get_embedding()
            for e in self._session.query(IgnoredFace).filter(IgnoredFace.kind == kind).all()
        ]
        if dim is None:
            return np.zeros((0, 0), np.float32)
        return _stack([v for v in vectors if v is not None and v.size == dim], dim)

    def _candidates(self) -> List[Face]:
        """Boxes the filter may decide on: automatic detections that are
        unassigned or in an "Unknown N" group, plus the boxes it hid before."""
        return (
            self._session.query(Face)
            .outerjoin(Person, Face.person_id == Person.id)
            .filter(Face.embedding_exists())
            .filter(Face.detector_backend != "manual")
            .filter(
                or_(
                    Face.assignment_source == SOURCE_MODEL,
                    and_(
                        Face.is_excluded == False,  # noqa: E712
                        or_(
                            Face.assignment_source.is_(None),
                            Face.assignment_source.notin_(_HUMAN_SOURCES),
                        ),
                        or_(
                            Face.person_id.is_(None),
                            and_(
                                Person.is_auto_named == True,  # noqa: E712
                                Person.is_protected == False,  # noqa: E712
                            ),
                        ),
                    ),
                )
            )
            .all()
        )

    def _restore_all_model_hidden(self) -> int:
        faces = self.list_model_hidden()
        for face in faces:
            self._unhide(face)
        if faces:
            self._session.commit()
            log.info("Non-face filter inactive — restored %d hidden box(es)", len(faces))
        return len(faces)

    @staticmethod
    def _unhide(face: Face) -> None:
        face.is_excluded = False
        face.assignment_source = None
        face.assignment_confidence = None
        face.assigned_at = None

    def _delete_empty_auto_persons(self, person_ids: Sequence[int]) -> None:
        """Remove "Unknown N" persons the hiding left without faces."""
        for pid in person_ids:
            person = self._session.get(Person, pid)
            if person is None or not person.is_auto_named or person.is_protected:
                continue
            if self._session.query(Face).filter(Face.person_id == pid).count() == 0:
                self._session.delete(person)


def _human_source():
    """SQL: the assignment is a human decision (legacy rows have NULL)."""
    return or_(
        Face.assignment_source.is_(None),
        Face.assignment_source.in_(_HUMAN_SOURCES),
    )


def _dim(face: Face) -> int:
    vec = face.get_embedding()
    return 0 if vec is None else int(vec.size)


def _stack(vectors: List[np.ndarray], dim: int) -> np.ndarray:
    if not vectors:
        return np.zeros((0, dim), np.float32)
    return np.vstack(vectors).astype(np.float32)
