"""Tests for the deep pipeline worker's rebuild reset logic.

The full pipeline needs detector/embedder models, so these tests cover the
destructive part in isolation: what survives a "rebuild from scratch".
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import AppConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.workers.deep_pipeline_worker import (
    MODE_REBUILD,
    MODE_REBUILD_MODEL,
    MODE_RESCAN,
    MODE_TRAIN,
    DeepPipelineWorker,
)

DIM = 16


@pytest.fixture()
def tmp_db(tmp_path):
    db_file = tmp_path / "test.db"
    init_db(db_file)
    return db_file


def _vec(axis: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[axis] = 1.0
    return v


def _add_image(session, path: str) -> int:
    image = Image(
        file_path=path, file_hash=path, file_mtime=0.0,
        detection_done=True, embedding_done=True,
    )
    session.add(image)
    session.flush()
    return image.id


def _add_person(session, name: str, *, auto=False) -> int:
    person = Person(name=name, is_auto_named=auto)
    session.add(person)
    session.flush()
    return person.id


def _add_face(session, image_id, person_id, *, source=None, backend="cpu") -> int:
    face = Face(
        image_id=image_id, person_id=person_id,
        bbox_x=0, bbox_y=0, bbox_w=20, bbox_h=20,
        confidence=0.9, detector_backend=backend,
        assignment_source=source,
    )
    face.set_embedding(_vec(0))
    session.add(face)
    session.flush()
    return face.id


class _Zero:
    """Stand-in stage result: every attribute is a falsy zero."""

    def __getattr__(self, name):
        return self

    def __add__(self, other):
        return other

    __radd__ = __add__

    def __mul__(self, other):
        return 0

    __rmul__ = __mul__

    def __bool__(self):
        return False

    def __format__(self, spec):
        return format(0, spec) if spec else "0"


def test_overlap_resolution_runs_again_after_recognition(tmp_db, monkeypatch):
    """The pipeline resolves overlapping boxes a second time, *after* the AI has
    assigned names, so two boxes of one face labelled as different people get
    collapsed."""
    worker = DeepPipelineWorker(
        root_folders=["/x"], config=AppConfig(), mode=MODE_RESCAN
    )
    calls: list[str] = []

    def rec(name, ret=None):
        def _fn(*_a, **_k):
            calls.append(name)
            return _Zero() if ret is None else ret
        return _fn

    for name in (
        "_run_detection", "_run_ai_face_detection", "_run_embedding",
        "_run_overlap_resolution", "_run_multistage_cleanup",
        "_run_ignored_filter", "_run_deep_train_and_recognize",
        "_run_clustering", "_run_intra_image_consistency", "_run_suggestions",
    ):
        monkeypatch.setattr(worker, name, rec(name))
    monkeypatch.setattr(worker, "_run_scan", rec("_run_scan", []))
    monkeypatch.setattr(worker, "_get_pending_detection_ids", lambda: [])
    monkeypatch.setattr(worker, "_checkpoint", lambda: None)

    worker._run_pipeline()

    overlap_calls = [i for i, n in enumerate(calls) if n == "_run_overlap_resolution"]
    rec_call = calls.index("_run_deep_train_and_recognize")
    cluster_call = calls.index("_run_clustering")
    assert len(overlap_calls) == 2
    assert overlap_calls[0] < rec_call < overlap_calls[1] < cluster_call


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        DeepPipelineWorker(root_folders=["/x"], config=AppConfig(), mode="bogus")


def test_valid_modes_construct():
    for mode in (MODE_RESCAN, MODE_REBUILD, MODE_TRAIN, MODE_REBUILD_MODEL):
        worker = DeepPipelineWorker(root_folders=["/x"], config=AppConfig(), mode=mode)
        assert worker.mode == mode


def test_rebuild_reset_deletes_stale_model_file(tmp_db, tmp_path):
    """A full rebuild also wipes the model file so it cannot linger/be reused."""
    model_dir = tmp_path / "deep_model"
    model_dir.mkdir()
    model_file = model_dir / "deep_face_model.pkl"
    model_file.write_bytes(b"stale-model")

    cfg = AppConfig()
    cfg.deep_recognition.model_dir = str(model_dir)

    with session_scope() as s:
        img = _add_image(s, "/a.jpg")
        _add_face(s, img, None)  # one auto face so the reset has work to do

    worker = DeepPipelineWorker(root_folders=["/x"], config=cfg, mode=MODE_REBUILD)
    worker._reset_for_rebuild()

    assert not model_file.exists()


def test_ai_stage_threads_detection_config(monkeypatch):
    """The in-pipeline AI stage must pass DetectionConfig so the AI pass runs
    the verification/geometry gate instead of the unprotected legacy path."""
    from unittest.mock import MagicMock, patch

    cfg = AppConfig()
    w = DeepPipelineWorker(root_folders=["/x"], config=cfg, mode=MODE_RESCAN)
    svc = MagicMock()
    svc.detect_images.return_value = MagicMock(
        available=True, images_processed=1, faces_found=0, error=None
    )
    monkeypatch.setattr(
        "app.workers.deep_pipeline_worker.session_scope",
        lambda: MagicMock(__enter__=lambda s: svc, __exit__=MagicMock()),
    )
    with patch(
        "app.services.ai_face_detection_service.AiFaceDetectionService",
        return_value=svc,
    ) as mock_cls:
        w._run_ai_face_detection([1])

    mock_cls.assert_called_once()
    assert mock_cls.call_args.kwargs["detection_config"] is cfg.detection


def test_rebuild_reset_keeps_human_decisions(tmp_db):
    with session_scope() as s:
        img = _add_image(s, "/a.jpg")
        anna = _add_person(s, "Anna")
        unknown = _add_person(s, "Unknown 1", auto=True)

        manual_box = _add_face(s, img, None, backend="manual")
        manual_assigned = _add_face(s, img, anna, source="manual")
        legacy_assigned = _add_face(s, img, anna, source=None)
        confirmed = _add_face(s, img, anna, source="deep_confirmed")
        auto_recognized = _add_face(s, img, anna, source="recognition")
        unknown_member = _add_face(s, img, unknown, source="clustering")
        unassigned = _add_face(s, img, None)

    worker = DeepPipelineWorker(
        root_folders=["/x"], config=AppConfig(), mode=MODE_REBUILD
    )
    n_deleted, n_kept = worker._reset_for_rebuild()

    assert n_kept == 4   # manual box + manual/legacy/confirmed assignments
    assert n_deleted == 3  # auto-recognized + unknown member + unassigned

    with session_scope() as s:
        assert s.get(Face, manual_box) is not None
        assert s.get(Face, manual_assigned) is not None
        assert s.get(Face, legacy_assigned) is not None
        assert s.get(Face, confirmed) is not None
        assert s.get(Face, auto_recognized) is None
        assert s.get(Face, unknown_member) is None
        assert s.get(Face, unassigned) is None

        # The emptied Unknown group is removed; the named person survives.
        assert s.get(Person, unknown) is None
        assert s.get(Person, anna) is not None

        # Every image is queued for re-detection and re-embedding.
        image = s.get(Image, img)
        assert image.detection_done is False
        assert image.embedding_done is False
