"""Tests for the object-matching background worker and review dialog (#164)."""

import os
import tempfile

import cv2
import pytest

from app.db.database import init_db, session_scope
from app.db.models import Image
from app.services.object_matching_service import ObjectMatchingService
from app.services.object_service import ObjectService
from app.workers.object_match_worker import ObjectMatchWorker
from tests.test_object_matching_service import _collage_library


class _FakeToken:
    def __init__(self, cancelled: bool = False) -> None:
        self._cancelled = cancelled

    def cancelled(self) -> bool:
        return self._cancelled


class _FakeContext:
    """Stands in for TaskContext: records progress, counts checkpoints."""

    def __init__(self, cancelled: bool = False) -> None:
        self.token = _FakeToken(cancelled)
        self.reports = []
        self.checkpoints = 0

    def report(self, percent: int, message: str = "") -> None:
        self.reports.append((percent, message))

    def checkpoint(self) -> None:
        self.checkpoints += 1


@pytest.fixture()
def library():
    directory = tempfile.mkdtemp()
    init_db(os.path.join(directory, "objects.db"))
    source_path, collage_path, other_path, box = _collage_library(directory)
    with session_scope() as session:
        source = Image(file_path=source_path, file_hash="a", file_mtime=0.0)
        collage = Image(file_path=collage_path, file_hash="b", file_mtime=0.0)
        other = Image(file_path=other_path, file_hash="c", file_mtime=0.0)
        session.add_all([source, collage, other])
        session.flush()

        objects = ObjectService(session)
        obj = objects.create_object(name="Vitorlás")
        objects.add_occurrence_bbox(obj.id, source.id, 150, 110, 300, 220)
        ids = (obj.id, source.id, collage.id, other.id)
    yield directory, ids, box


def test_worker_reports_progress_and_finds_the_copy(library):
    _directory, (object_id, _source, collage_id, _other), _box = library
    ctx = _FakeContext()

    stats = ObjectMatchWorker(object_id=object_id).run_in_task(ctx)

    assert stats.suggestions_created == 1
    assert ctx.checkpoints > 0
    assert ctx.reports[-1] == (100, "")
    assert all(0 <= percent <= 100 for percent, _ in ctx.reports)

    with session_scope() as session:
        pending = ObjectMatchingService(session).list_pending(object_id=object_id)
    assert [p.image_id for p in pending] == [collage_id]


def test_worker_stops_when_the_task_is_cancelled(library):
    _directory, (object_id, *_rest), _box = library
    ctx = _FakeContext(cancelled=True)

    stats = ObjectMatchWorker(object_id=object_id).run_in_task(ctx)

    assert stats.cancelled
    assert stats.suggestions_created == 0


def test_worker_batch_mode_covers_every_object(library):
    directory, (object_id, source_id, _collage, _other), _box = library
    with session_scope() as session:
        second = ObjectService(session).create_object(name="Második")
        ObjectService(session).add_occurrence_bbox(second.id, source_id, 10, 10, 120, 100)

    stats = ObjectMatchWorker().run_in_task(_FakeContext())
    assert stats.objects_searched == 2
    assert object_id in stats.object_ids


def test_worker_scope_limits_the_images(library):
    _directory, (object_id, _source, _collage, other_id), _box = library
    stats = ObjectMatchWorker(object_id=object_id, image_ids=[other_id]).run_in_task(
        _FakeContext()
    )
    assert stats.images_scanned == 1
    assert stats.suggestions_created == 0


def test_review_dialog_lists_and_decides(library, qtbot):
    from app.ui.dialogs.object_match_review_dialog import ObjectMatchReviewDialog

    _directory, (object_id, *_rest), _box = library
    ObjectMatchWorker(object_id=object_id).run_in_task(_FakeContext())

    dialog = ObjectMatchReviewDialog(object_id=object_id)
    qtbot.addWidget(dialog)
    assert len(dialog._cards) == 1

    suggestion_id = dialog._cards[0].info.suggestion_id
    dialog.decide(suggestion_id, accept=True)

    assert dialog.accepted_count == 1
    assert dialog._cards == []
    with session_scope() as session:
        assert ObjectMatchingService(session).pending_count(object_id) == 0


def test_review_dialog_rejects_without_marking(library, qtbot):
    from app.ui.dialogs.object_match_review_dialog import ObjectMatchReviewDialog

    _directory, (object_id, *_rest), _box = library
    ObjectMatchWorker(object_id=object_id).run_in_task(_FakeContext())

    dialog = ObjectMatchReviewDialog(object_id=object_id)
    qtbot.addWidget(dialog)
    dialog.decide(dialog._cards[0].info.suggestion_id, accept=False)

    assert dialog.rejected_count == 1
    with session_scope() as session:
        occurrences = ObjectService(session).get_occurrences(object_id)
    assert len(occurrences) == 1  # only the original manual marking


def test_review_dialog_is_empty_when_nothing_pending(library, qtbot):
    from app.ui.dialogs.object_match_review_dialog import ObjectMatchReviewDialog

    _directory, (object_id, *_rest), _box = library
    dialog = ObjectMatchReviewDialog(object_id=object_id)
    qtbot.addWidget(dialog)
    assert dialog._cards == []
    assert dialog._empty.isVisibleTo(dialog)


def test_collage_fixture_really_contains_a_shrunk_copy(library):
    """Guards the fixture itself: the collage copy must be clearly smaller."""
    directory, _ids, box = library
    collage = cv2.imread(os.path.join(directory, "collage.png"))
    assert collage is not None
    assert box[2] < 200 and box[3] < 150
