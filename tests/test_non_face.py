"""Tests for the learned "not a face" filter (classifier + service)."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import NonFaceConfig
from app.db.database import init_db, session_scope
from app.db.models import (
    IGNORED_KIND_IS_FACE,
    IGNORED_KIND_NOT_FACE,
    Face,
    IgnoredFace,
    Image,
    Person,
)
from app.deep.non_face import train_non_face_model
from app.services.ignored_face_service import IgnoredFaceService
from app.services.non_face_service import SOURCE_MODEL, NonFaceService

DIM = 64


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def _face(identity: int, rng: np.random.Generator) -> np.ndarray:
    """A real face: shared 'face' direction + an identity part + noise."""
    ident = np.random.default_rng(1000 + identity).normal(0, 1, 31)
    v = np.zeros(DIM)
    v[0] = 1.5
    v[1:32] = ident / np.linalg.norm(ident) + rng.normal(0, 0.15, 31)
    return _unit(v)


def _pattern(rng: np.random.Generator) -> np.ndarray:
    """A non-face (carpet / half face): its own region of the space."""
    v = np.zeros(DIM)
    v[32] = 1.5
    v[33:] = rng.normal(0, 0.5, 31)
    return _unit(v)


# ── Classifier ───────────────────────────────────────────────────────────────

def test_model_needs_enough_examples():
    rng = np.random.default_rng(0)
    pos = np.vstack([_face(i % 8, rng) for i in range(80)])
    neg = np.vstack([_pattern(rng) for _ in range(3)])
    assert train_non_face_model(pos, neg, min_negatives=5) is None


def test_model_flags_new_patterns_but_not_unseen_people():
    rng = np.random.default_rng(1)
    pos = np.vstack([_face(i % 8, rng) for i in range(80)])
    neg = np.vstack([_pattern(rng) for _ in range(8)])
    model = train_non_face_model(pos, neg, min_negatives=5, min_positives=30)
    assert model is not None and model.threshold >= 0.90

    new_patterns = np.vstack([_pattern(rng) for _ in range(20)])
    strangers = np.vstack([_face(100 + i, rng) for i in range(20)])  # unseen ids
    assert model.non_face_mask(new_patterns)[0].mean() >= 0.9
    assert not model.non_face_mask(strangers)[0].any()


# ── Service ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(tmp_path):
    init_db(tmp_path / "test.db")


def _add_face(session, image_id, vec, person_id=None, source=None, backend="cpu") -> int:
    face = Face(
        image_id=image_id, person_id=person_id,
        bbox_x=0, bbox_y=0, bbox_w=40, bbox_h=40, confidence=0.9,
        detector_backend=backend, assignment_source=source,
    )
    face.set_embedding(vec)
    session.add(face)
    session.flush()
    return face.id


def _seed(session):
    """Kati (named, 40 manual faces), strangers, and 12 pattern boxes."""
    rng = np.random.default_rng(7)
    img = Image(file_path="/a.jpg", file_hash="a", file_mtime=0.0)
    session.add(img)
    session.flush()
    kati = Person(name="Kati", is_auto_named=False)
    unknown = Person(name="Unknown 1", is_auto_named=True)
    session.add_all([kati, unknown])
    session.flush()
    ids = {
        "kati": [
            _add_face(session, img.id, _face(i % 4, rng), kati.id, "manual")
            for i in range(40)
        ],
        "strangers": [_add_face(session, img.id, _face(50 + i, rng)) for i in range(5)],
        "marked": [_add_face(session, img.id, _pattern(rng)) for _ in range(6)],
        "patterns": [_add_face(session, img.id, _pattern(rng)) for _ in range(5)],
        "unknown_pattern": [
            _add_face(session, img.id, _pattern(rng), unknown.id, "clustering")
        ],
        "kati_ai_pattern": [
            _add_face(session, img.id, _pattern(rng), kati.id, "deep_recognition")
        ],
    }
    session.commit()
    return ids, unknown.id


_CFG = NonFaceConfig(min_negatives=5, min_positives=30)


def test_mark_not_face_hides_box_and_stores_example(tmp_db):
    with session_scope() as s:
        ids, _ = _seed(s)
        assert NonFaceService(s, _CFG).mark_not_face(ids["marked"]) == 6
        face = s.get(Face, ids["marked"][0])
        assert face.is_excluded and face.assignment_source == "not_face"
        kinds = {e.kind for e in s.query(IgnoredFace).all()}
        assert kinds == {IGNORED_KIND_NOT_FACE}


def test_apply_hides_similar_boxes_only(tmp_db):
    with session_scope() as s:
        ids, unknown_id = _seed(s)
        svc = NonFaceService(s, _CFG)
        svc.mark_not_face(ids["marked"])
        stats = svc.apply()
        assert stats.active
        hidden = {f.id for f in svc.list_model_hidden()}
        assert hidden == set(ids["patterns"] + ids["unknown_pattern"])
        # Strangers (real faces, not in the positives) and Kati stay visible;
        # a named person's AI-assigned box is left to the user.
        for fid in ids["strangers"] + ids["kati"] + ids["kati_ai_pattern"]:
            assert not s.get(Face, fid).is_excluded
        # The emptied "Unknown 1" group is removed.
        assert s.get(Person, unknown_id) is None


def test_restored_box_becomes_face_example_and_stays(tmp_db):
    with session_scope() as s:
        ids, _ = _seed(s)
        svc = NonFaceService(s, _CFG)
        svc.mark_not_face(ids["marked"])
        svc.apply()
        fid = ids["patterns"][0]
        assert svc.confirm_face(fid)
        assert s.query(IgnoredFace).filter_by(kind=IGNORED_KIND_IS_FACE).count() == 1
        svc.apply()
        face = s.get(Face, fid)
        assert not face.is_excluded and face.assignment_source != SOURCE_MODEL


def test_too_few_examples_restores_model_hidden(tmp_db):
    with session_scope() as s:
        ids, _ = _seed(s)
        svc = NonFaceService(s, _CFG)
        svc.mark_not_face(ids["marked"])
        svc.apply()
        assert svc.list_model_hidden()
        ignored = IgnoredFaceService(s)
        for entry in ignored.list_ignored()[:3]:
            ignored.unignore(entry.id)
        stats = svc.apply()
        assert not stats.active and stats.n_restored == 6
        assert not svc.list_model_hidden()


def test_unignore_not_face_restores_the_box(tmp_db):
    with session_scope() as s:
        ids, _ = _seed(s)
        NonFaceService(s, _CFG).mark_not_face(ids["marked"][:1])
        ignored = IgnoredFaceService(s)
        entry = ignored.list_ignored()[0]
        assert ignored.unignore(entry.id)
        assert not s.get(Face, ids["marked"][0]).is_excluded


def test_face_examples_are_not_on_the_ignore_list(tmp_db):
    with session_scope() as s:
        ids, _ = _seed(s)
        svc = NonFaceService(s, _CFG)
        svc.mark_not_face(ids["marked"])
        svc.apply()
        svc.confirm_face(ids["patterns"][0])
        ignored = IgnoredFaceService(s)
        assert all(e.kind != IGNORED_KIND_IS_FACE for e in ignored.list_ignored())
        assert ignored.count_ignored() == 6


# ── Ignored Faces dialog ─────────────────────────────────────────────────────

def test_dialog_lists_marks_and_restores_model_hidden(tmp_db, qtbot, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QMessageBox

    from app.config import AppConfig
    from app.ui.dialogs.ignored_faces_dialog import IgnoredFacesDialog

    with session_scope() as s:
        ids, _ = _seed(s)
        svc = NonFaceService(s, _CFG)
        svc.mark_not_face(ids["marked"])
        svc.apply()

    dlg = IgnoredFacesDialog(AppConfig())
    qtbot.addWidget(dlg)
    keys = [dlg._list.item(i).data(Qt.UserRole) for i in range(dlg._list.count())]
    assert sum(1 for k in keys if k[0] == "entry") == 6
    auto_rows = [i for i, k in enumerate(keys) if k[0] == "auto"]
    assert len(auto_rows) == 6

    restored_id = int(keys[auto_rows[0]][1])
    dlg._list.item(auto_rows[0]).setSelected(True)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    dlg._on_unignore()
    assert dlg.changed()
    with session_scope() as s:
        assert not s.get(Face, restored_id).is_excluded
