"""Tests for resetting automatically created Unknown identities."""

from __future__ import annotations

import numpy as np
import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, FaceBlob, Image, MergeSuggestion, Person
from app.services.unknown_person_reset_service import (
    UnknownPersonResetOptions,
    UnknownPersonResetService,
)


@pytest.fixture()
def tmp_db(tmp_path):
    db_file = tmp_path / "test.db"
    init_db(db_file)
    return db_file


def _add_face(
    session,
    image_id: int,
    person_id: int | None,
    source: str | None,
    crop_path: str | None = None,
) -> Face:
    face = Face(
        image_id=image_id,
        person_id=person_id,
        bbox_x=0,
        bbox_y=0,
        bbox_w=20,
        bbox_h=20,
        confidence=1.0,
        detector_backend="cpu",
        assignment_source=source,
        assignment_confidence=0.8 if source else None,
        crop_path=crop_path,
    )
    face.set_embedding(np.array([1.0, 0.0], dtype=np.float32))
    session.add(face)
    session.flush()
    return face


def _seed(session, *, unknown_crop: str | None = None) -> dict[str, int]:
    """One named, one auto-named ("Unknown") and one protected person, each with
    a single face.  Returns the ids the assertions need."""
    image = Image(file_path="/i.jpg", file_hash="h", file_mtime=0.0)
    named = Person(name="Anna", is_auto_named=False)
    unknown = Person(name="Unknown 1", is_auto_named=True)
    protected = Person(name="Ismeretlen", is_auto_named=False, is_protected=True)
    session.add_all([image, named, unknown, protected])
    session.flush()

    return {
        "unknown_person": unknown.id,
        "unknown_face": _add_face(
            session, image.id, unknown.id, "clustering", unknown_crop
        ).id,
        "named_face": _add_face(session, image.id, named.id, "manual").id,
        "protected_face": _add_face(session, image.id, protected.id, "manual").id,
    }


def test_reset_deletes_auto_named_persons_and_unassigns_their_faces(tmp_db):
    with session_scope() as session:
        image = Image(file_path="/i.jpg", file_hash="h", file_mtime=0.0)
        named = Person(name="Anna", is_auto_named=False)
        unknown = Person(name="Unknown 1", is_auto_named=True)
        protected = Person(name="Ismeretlen", is_auto_named=False, is_protected=True)
        session.add_all([image, named, unknown, protected])
        session.flush()

        unknown_face = _add_face(session, image.id, unknown.id, "clustering")
        named_face = _add_face(session, image.id, named.id, "manual")
        protected_face = _add_face(session, image.id, protected.id, "manual")
        session.add(
            MergeSuggestion(
                source_person_id=min(named.id, unknown.id),
                target_person_id=max(named.id, unknown.id),
                confidence=0.9,
            )
        )
        unknown_id = unknown.id
        unknown_face_id = unknown_face.id
        named_face_id = named_face.id
        protected_face_id = protected_face.id

    with session_scope() as session:
        result = UnknownPersonResetService(session).reset()

    assert result.deleted_persons == 1
    assert result.unassigned_faces == 1
    assert result.deleted_faces == 0
    with session_scope() as session:
        assert session.get(Person, unknown_id) is None
        reset_face = session.get(Face, unknown_face_id)
        assert reset_face is not None
        assert reset_face.person_id is None
        assert reset_face.assignment_source is None
        assert reset_face.assignment_confidence is None
        assert reset_face.get_embedding() is not None
        assert session.get(Face, named_face_id).person_id is not None
        assert session.get(Face, protected_face_id).person_id is not None
        assert session.query(MergeSuggestion).count() == 0


def test_reset_is_a_noop_without_auto_named_persons(tmp_db):
    with session_scope() as session:
        result = UnknownPersonResetService(session).reset()

    assert result.deleted_persons == 0
    assert result.unassigned_faces == 0


def test_delete_face_data_hard_deletes_the_unknown_faces(tmp_db):
    """The option used to be a silent no-op: it queried faces *after* they had
    already been unassigned, and wrote to columns that do not exist."""
    with session_scope() as session:
        ids = _seed(session)

    with session_scope() as session:
        result = UnknownPersonResetService(session).reset(
            UnknownPersonResetOptions(delete_face_data=True)
        )

    assert result.deleted_faces == 1
    with session_scope() as session:
        assert session.get(Face, ids["unknown_face"]) is None
        assert session.get(Person, ids["unknown_person"]) is None
        # The blob rows go with the face (cascade="all, delete-orphan").
        assert session.query(FaceBlob).count() == 2
        # Named and protected people keep every face they had.
        assert session.get(Face, ids["named_face"]) is not None
        assert session.get(Face, ids["protected_face"]) is not None


def test_delete_face_data_does_not_require_unassigning(tmp_db):
    """delete_face_data used to be gated behind delete_face_assignments, so
    turning off the unrelated checkbox silently disabled it."""
    with session_scope() as session:
        ids = _seed(session)

    with session_scope() as session:
        result = UnknownPersonResetService(session).reset(
            UnknownPersonResetOptions(
                delete_face_assignments=False, delete_face_data=True
            )
        )

    assert result.deleted_faces == 1
    with session_scope() as session:
        assert session.get(Face, ids["unknown_face"]) is None


def test_delete_face_data_unlinks_crop_files(tmp_db, tmp_path):
    crop = tmp_path / "unknown_1.jpg"
    crop.write_bytes(b"jpeg")

    with session_scope() as session:
        _seed(session, unknown_crop=str(crop))

    with session_scope() as session:
        result = UnknownPersonResetService(session).reset(
            UnknownPersonResetOptions(delete_face_data=True)
        )

    assert result.n_crops_removed == 1
    assert not crop.exists()


def test_delete_face_data_off_keeps_the_face_rows(tmp_db):
    with session_scope() as session:
        ids = _seed(session)

    with session_scope() as session:
        UnknownPersonResetService(session).reset(
            UnknownPersonResetOptions(delete_face_data=False)
        )

    with session_scope() as session:
        face = session.get(Face, ids["unknown_face"])
        assert face is not None
        assert face.get_embedding() is not None


def test_reset_with_every_option_off_changes_nothing(tmp_db):
    with session_scope() as session:
        ids = _seed(session)

    with session_scope() as session:
        result = UnknownPersonResetService(session).reset(
            UnknownPersonResetOptions(
                delete_unknown_persons=False,
                delete_face_assignments=False,
                delete_face_data=False,
            )
        )

    assert result.deleted_persons == 0
    assert result.deleted_faces == 0
    with session_scope() as session:
        assert session.get(Person, ids["unknown_person"]) is not None
        assert session.get(Face, ids["unknown_face"]).person_id is not None
