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
from tests.test_object_matching_service import REVIEW_ONLY, _collage_library


class _FakeToken:
    def __init__(self, cancelled: bool = False) -> None:
        self._cancelled = cancelled

    @property
    def cancelled(self) -> bool:
        # A property, like the real CancellationToken (#188).
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

    stats = ObjectMatchWorker(object_id=object_id, config=REVIEW_ONLY).run_in_task(ctx)

    assert stats.suggestions_created == 1
    assert ctx.checkpoints > 0
    assert ctx.reports[-1] == (100, "")
    assert all(0 <= percent <= 100 for percent, _ in ctx.reports)

    with session_scope() as session:
        pending = ObjectMatchingService(session).list_pending(object_id=object_id)
    assert [p.image_id for p in pending] == [collage_id]


class _WritingContext(_FakeContext):
    """Saves an object marking from another session at every checkpoint."""

    def __init__(self, object_id: int, image_id: int) -> None:
        super().__init__()
        self._object_id = object_id
        self._image_id = image_id
        self.writes = 0

    def checkpoint(self) -> None:
        super().checkpoint()
        from sqlalchemy import text

        with session_scope() as session:
            # Fail fast instead of the app's 5 s wait if the lock is held.
            session.execute(text("PRAGMA busy_timeout=100"))
            ObjectService(session).add_occurrence_bbox(
                self._object_id, self._image_id, 1, 1, 5, 5
            )
        self.writes += 1


def test_worker_does_not_hold_the_write_lock(library):
    """The UI can save markings while a search runs (#189)."""
    _directory, (object_id, _source, _collage, other_id), _box = library
    ctx = _WritingContext(object_id, other_id)

    ObjectMatchWorker(object_id=object_id, config=REVIEW_ONLY).run_in_task(ctx)

    assert ctx.writes == ctx.checkpoints > 0


def test_worker_stops_when_the_task_is_cancelled(library):
    _directory, (object_id, *_rest), _box = library
    ctx = _FakeContext(cancelled=True)

    stats = ObjectMatchWorker(object_id=object_id, config=REVIEW_ONLY).run_in_task(ctx)

    assert stats.cancelled
    assert stats.suggestions_created == 0


def test_worker_batch_mode_covers_every_object(library):
    directory, (object_id, source_id, _collage, _other), _box = library
    with session_scope() as session:
        second = ObjectService(session).create_object(name="Második")
        ObjectService(session).add_occurrence_bbox(second.id, source_id, 10, 10, 120, 100)

    stats = ObjectMatchWorker(config=REVIEW_ONLY).run_in_task(_FakeContext())
    assert stats.objects_searched == 2
    assert object_id in stats.object_ids


def test_worker_scope_limits_the_images(library):
    _directory, (object_id, _source, _collage, other_id), _box = library
    stats = ObjectMatchWorker(
        object_id=object_id, image_ids=[other_id], config=REVIEW_ONLY
    ).run_in_task(
        _FakeContext()
    )
    assert stats.images_scanned == 1
    assert stats.suggestions_created == 0


def test_review_dialog_lists_and_decides(library, qtbot):
    from app.ui.dialogs.object_match_review_dialog import ObjectMatchReviewDialog

    _directory, (object_id, *_rest), _box = library
    ObjectMatchWorker(object_id=object_id, config=REVIEW_ONLY).run_in_task(_FakeContext())

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
    ObjectMatchWorker(object_id=object_id, config=REVIEW_ONLY).run_in_task(_FakeContext())

    dialog = ObjectMatchReviewDialog(object_id=object_id)
    qtbot.addWidget(dialog)
    dialog.decide(dialog._cards[0].info.suggestion_id, accept=False)

    assert dialog.rejected_count == 1
    with session_scope() as session:
        occurrences = ObjectService(session).get_occurrences(object_id)
    assert len(occurrences) == 1  # only the original manual marking


def test_worker_invalidates_changed_images_first(library):
    _directory, (object_id, _source, collage_id, _other), _box = library
    from app.db.models import ImageFeatures

    ObjectMatchWorker(object_id=object_id, config=REVIEW_ONLY).run_in_task(
        _FakeContext()
    )
    with session_scope() as session:
        ObjectMatchingService(session).clear_pending(object_id)
        # Simulate cached features describing the file's old, featureless
        # content: without invalidation the copy could not be found again.
        row = session.get(ImageFeatures, collage_id)
        row.descriptors = None
        row.keypoints = None
        row.n_features = 0

    def run(invalidate):
        ObjectMatchWorker(
            object_id=object_id,
            image_ids=[collage_id],
            config=REVIEW_ONLY,
            invalidate_image_ids=invalidate,
        ).run_in_task(_FakeContext())
        with session_scope() as session:
            return ObjectMatchingService(session).pending_count(object_id)

    assert run(None) == 0
    assert run([collage_id]) == 1


def test_review_dialog_lists_auto_tags_with_undo(library, qtbot):
    from app.ui.dialogs.object_match_review_dialog import ObjectMatchReviewDialog

    _directory, (object_id, _source, collage_id, _other), _box = library
    stats = ObjectMatchWorker(object_id=object_id).run_in_task(_FakeContext())
    assert stats.auto_accepted == 1

    dialog = ObjectMatchReviewDialog(object_id=object_id)
    qtbot.addWidget(dialog)
    assert [card.is_auto for card in dialog._cards] == [True]
    assert dialog._confirm_all_btn.isVisibleTo(dialog)
    assert not dialog._accept_above_btn.isEnabled()

    dialog.decide(dialog._cards[0].info.suggestion_id, accept=False)

    assert dialog.rejected_count == 1
    assert dialog._cards == []
    with session_scope() as session:
        occurrences = ObjectService(session).get_occurrences(object_id)
    assert collage_id not in [o.image_id for o in occurrences]


def test_review_dialog_confirms_auto_tags(library, qtbot):
    from app.ui.dialogs.object_match_review_dialog import ObjectMatchReviewDialog

    _directory, (object_id, _source, collage_id, _other), _box = library
    ObjectMatchWorker(object_id=object_id).run_in_task(_FakeContext())

    dialog = ObjectMatchReviewDialog(object_id=object_id)
    qtbot.addWidget(dialog)
    dialog._on_confirm_all_auto()

    assert dialog.accepted_count == 1
    assert dialog._cards == []
    with session_scope() as session:
        occurrences = ObjectService(session).get_occurrences(object_id)
    assert collage_id in [o.image_id for o in occurrences]


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
