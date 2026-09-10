"""Tests for "recognise the same object region in other images" (#164).

The scenario the feature exists for: a photo is tagged with an object, and the
same picture turns up elsewhere at a different size — most typically shrunk into
a collage.  The matcher must find it there and propose the same object.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest

from app.config import ObjectMatchingConfig
from app.db.database import init_db, session_scope
from app.db.models import (
    OBJECT_MATCH_ACCEPTED,
    OBJECT_MATCH_REJECTED,
    Image,
    ObjectMatchSuggestion,
    ObjectOccurrence,
)
from app.services.object_feature_service import (
    ObjectFeatureService,
    extract_features,
    extract_patch_features,
)
from app.services.object_matching_service import (
    ObjectMatchingService,
    bbox_iou,
    match_patch_in_image,
)
from app.services.object_service import ObjectService
from tests.test_object_feature_service import make_texture

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def db():
    directory = tempfile.mkdtemp()
    init_db(os.path.join(directory, "objects.db"))
    yield directory


def _add_image(session, path: str) -> Image:
    image = Image(file_path=path, file_hash="h" + path, file_mtime=0.0)
    session.add(image)
    session.flush()
    return image


def paste(canvas: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    """Paste *patch* into *canvas* with its top-left corner at ``(x, y)``."""
    h, w = patch.shape[:2]
    canvas[y : y + h, x : x + w] = patch


def scaled(patch: np.ndarray, factor: float) -> np.ndarray:
    h, w = patch.shape[:2]
    return cv2.resize(
        patch,
        (max(1, int(w * factor)), max(1, int(h * factor))),
        interpolation=cv2.INTER_AREA,
    )


def background(width: int, height: int, seed: int) -> np.ndarray:
    """A distractor canvas, so a hit has to beat unrelated texture."""
    return make_texture(width, height, seed)


# ---------------------------------------------------------------------------
# The matcher itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factor", [1.0, 0.5, 0.25])
def test_finds_the_same_region_at_any_size(factor):
    """The core promise of #164: same region, different scale, still found."""
    config = ObjectMatchingConfig()
    patch = make_texture(320, 240, seed=101)

    canvas = background(900, 700, seed=202)
    copy = scaled(patch, factor) if factor != 1.0 else patch
    at_x, at_y = 120, 90
    paste(canvas, copy, at_x, at_y)

    patch_fs = extract_patch_features(patch, config)
    target_fs = extract_features(canvas, config)
    result = match_patch_in_image(patch_fs, target_fs, config)

    assert result is not None, f"no match at scale {factor}"
    expected = (at_x, at_y, copy.shape[1], copy.shape[0])
    assert bbox_iou(result.bbox, expected) >= 0.7
    assert result.scale == pytest.approx(factor, rel=0.2)


def test_unrelated_image_produces_no_match():
    config = ObjectMatchingConfig()
    patch_fs = extract_patch_features(make_texture(300, 220, seed=303), config)
    other_fs = extract_features(background(800, 600, seed=404), config)
    assert match_patch_in_image(patch_fs, other_fs, config) is None


def test_rotated_copy_is_still_found():
    """A similarity transform covers rotation, which collages sometimes apply."""
    config = ObjectMatchingConfig()
    patch = make_texture(300, 300, seed=505)
    canvas = background(900, 900, seed=606)
    paste(canvas, cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE), 200, 250)

    result = match_patch_in_image(
        extract_patch_features(patch, config),
        extract_features(canvas, config),
        config,
    )
    assert result is not None
    assert bbox_iou(result.bbox, (200, 250, 300, 300)) >= 0.7


def test_min_score_gate_rejects_weak_hits():
    config = ObjectMatchingConfig()
    patch = make_texture(320, 240, seed=707)
    canvas = background(800, 600, seed=808)
    paste(canvas, patch, 100, 100)

    patch_fs = extract_patch_features(patch, config)
    target_fs = extract_features(canvas, config)
    assert match_patch_in_image(patch_fs, target_fs, config) is not None

    strict = ObjectMatchingConfig(min_score=1.01)
    assert match_patch_in_image(patch_fs, target_fs, strict) is None


def test_bbox_iou_basics():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert bbox_iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


# ---------------------------------------------------------------------------
# End-to-end search
# ---------------------------------------------------------------------------

def _collage_library(directory, factor=0.4):
    """Build a source photo, a collage containing it shrunk, and a distractor."""
    patch = make_texture(300, 220, seed=909)

    source = background(600, 450, seed=111)
    paste(source, patch, 150, 110)
    source_path = os.path.join(directory, "source.png")
    cv2.imwrite(source_path, source)

    small = scaled(patch, factor)
    collage = background(900, 700, seed=222)
    collage_box = (400, 300, small.shape[1], small.shape[0])
    paste(collage, small, collage_box[0], collage_box[1])
    collage_path = os.path.join(directory, "collage.png")
    cv2.imwrite(collage_path, collage)

    other_path = os.path.join(directory, "other.png")
    cv2.imwrite(other_path, background(700, 500, seed=333))

    return source_path, collage_path, other_path, collage_box


def test_find_object_suggests_the_shrunk_copy(db):
    source_path, collage_path, other_path, collage_box = _collage_library(db)
    with session_scope() as session:
        source = _add_image(session, source_path)
        collage = _add_image(session, collage_path)
        other = _add_image(session, other_path)

        objects = ObjectService(session)
        obj = objects.create_object(name="Szemesi asztal")
        objects.add_occurrence_bbox(obj.id, source.id, 150, 110, 300, 220)

        service = ObjectMatchingService(session)
        stats = service.find_object(obj.id)

        assert stats.suggestions_created == 1
        pending = service.list_pending(object_id=obj.id)
        assert len(pending) == 1
        hit = pending[0]
        assert hit.image_id == collage.id
        assert hit.object_name == "Szemesi asztal"
        assert bbox_iou(hit.bbox, collage_box) >= 0.7
        assert hit.scale < 0.6

        # The distractor image was scanned but produced nothing.
        assert other.id not in [p.image_id for p in pending]
        assert stats.images_scanned == 2  # source holds the reference, so skipped


def test_accepting_creates_an_ai_occurrence_and_a_new_reference(db):
    source_path, collage_path, _other, _box = _collage_library(db)
    with session_scope() as session:
        source = _add_image(session, source_path)
        collage = _add_image(session, collage_path)

        objects = ObjectService(session)
        obj = objects.create_object(name="Klotild")
        objects.add_occurrence_bbox(obj.id, source.id, 150, 110, 300, 220)

        service = ObjectMatchingService(session)
        service.find_object(obj.id)
        suggestion = service.list_pending(object_id=obj.id)[0]
        occurrence = service.accept_suggestion(suggestion.suggestion_id)

        assert occurrence.detection_source == "ai"
        assert occurrence.confidence == pytest.approx(suggestion.score)
        assert occurrence.image_id == collage.id

        row = session.get(ObjectMatchSuggestion, suggestion.suggestion_id)
        assert row.status == OBJECT_MATCH_ACCEPTED
        assert row.created_occurrence_id == occurrence.id

        # This is the learning: the confirmed hit is now a reference sample too.
        refs = ObjectFeatureService(session).reference_occurrences(obj.id)
        assert occurrence.id in [r.id for r in refs]


def test_rejected_pairing_is_never_suggested_again(db):
    source_path, collage_path, _other, _box = _collage_library(db)
    with session_scope() as session:
        source = _add_image(session, source_path)
        _add_image(session, collage_path)

        objects = ObjectService(session)
        obj = objects.create_object(name="Hajó")
        objects.add_occurrence_bbox(obj.id, source.id, 150, 110, 300, 220)

        service = ObjectMatchingService(session)
        service.find_object(obj.id)
        suggestion = service.list_pending(object_id=obj.id)[0]
        service.reject_suggestion(suggestion.suggestion_id)

        assert (
            session.get(ObjectMatchSuggestion, suggestion.suggestion_id).status
            == OBJECT_MATCH_REJECTED
        )
        assert service.list_pending(object_id=obj.id) == []

        again = service.find_object(obj.id)
        assert again.suggestions_created == 0
        assert service.list_pending(object_id=obj.id) == []


def test_already_marked_image_is_not_suggested_again(db):
    source_path, collage_path, _other, box = _collage_library(db)
    with session_scope() as session:
        source = _add_image(session, source_path)
        collage = _add_image(session, collage_path)

        objects = ObjectService(session)
        obj = objects.create_object(name="Kerékpár")
        objects.add_occurrence_bbox(obj.id, source.id, 150, 110, 300, 220)
        objects.add_occurrence_bbox(obj.id, collage.id, *box)

        service = ObjectMatchingService(session)
        stats = service.find_object(obj.id)
        assert stats.suggestions_created == 0


def test_search_scope_limits_the_images_scanned(db):
    source_path, collage_path, other_path, _box = _collage_library(db)
    with session_scope() as session:
        source = _add_image(session, source_path)
        _add_image(session, collage_path)
        other = _add_image(session, other_path)

        objects = ObjectService(session)
        obj = objects.create_object(name="Szobor")
        objects.add_occurrence_bbox(obj.id, source.id, 150, 110, 300, 220)

        stats = ObjectMatchingService(session).find_object(
            obj.id, image_ids=[other.id]
        )
        assert stats.images_scanned == 1
        assert stats.suggestions_created == 0


def test_object_without_bbox_reference_is_skipped(db):
    with session_scope() as session:
        path = os.path.join(db, "plain.png")
        cv2.imwrite(path, background(400, 300, seed=444))
        image = _add_image(session, path)

        objects = ObjectService(session)
        obj = objects.create_object(name="Csak pont")
        objects.add_occurrence(obj.id, image.id, 20, 20)

        stats = ObjectMatchingService(session).find_object(obj.id)
        assert stats.skipped_no_reference == 1
        assert stats.suggestions_created == 0


def test_find_all_objects_covers_every_matchable_object(db):
    source_path, collage_path, _other, _box = _collage_library(db)
    with session_scope() as session:
        source = _add_image(session, source_path)
        _add_image(session, collage_path)

        objects = ObjectService(session)
        first = objects.create_object(name="Egy")
        objects.add_occurrence_bbox(first.id, source.id, 150, 110, 300, 220)
        second = objects.create_object(name="Kettő")
        objects.add_occurrence_bbox(second.id, source.id, 10, 10, 120, 100)

        service = ObjectMatchingService(session)
        assert sorted(service.matchable_object_ids()) == sorted([first.id, second.id])

        stats = service.find_all_objects()
        assert stats.objects_searched == 2


def test_search_honours_cancellation(db):
    source_path, collage_path, _other, _box = _collage_library(db)
    with session_scope() as session:
        source = _add_image(session, source_path)
        _add_image(session, collage_path)

        objects = ObjectService(session)
        obj = objects.create_object(name="Megszakítás")
        objects.add_occurrence_bbox(obj.id, source.id, 150, 110, 300, 220)

        stats = ObjectMatchingService(session).find_object(
            obj.id, cancel_check=lambda: True
        )
        assert stats.cancelled
        assert stats.suggestions_created == 0


def test_accept_above_and_clear_pending(db):
    source_path, collage_path, _other, _box = _collage_library(db)
    with session_scope() as session:
        source = _add_image(session, source_path)
        _add_image(session, collage_path)

        objects = ObjectService(session)
        obj = objects.create_object(name="Küszöb")
        objects.add_occurrence_bbox(obj.id, source.id, 150, 110, 300, 220)

        service = ObjectMatchingService(session)
        service.find_object(obj.id)
        assert service.pending_count(obj.id) == 1

        # Nothing clears a threshold nobody can reach.
        assert service.accept_above(1.01, object_id=obj.id) == 0
        assert service.accept_above(0.0, object_id=obj.id) == 1
        assert service.pending_count(obj.id) == 0

        service.find_object(obj.id)
        assert service.clear_pending(obj.id) == 0


def test_clear_pending_leaves_no_negative_memory(db):
    source_path, collage_path, _other, _box = _collage_library(db)
    with session_scope() as session:
        source = _add_image(session, source_path)
        _add_image(session, collage_path)

        objects = ObjectService(session)
        obj = objects.create_object(name="Elvetve")
        objects.add_occurrence_bbox(obj.id, source.id, 150, 110, 300, 220)

        service = ObjectMatchingService(session)
        service.find_object(obj.id)
        assert service.clear_pending(obj.id) == 1

        # Discarding without judging must not teach the matcher anything.
        stats = service.find_object(obj.id)
        assert stats.suggestions_created == 1


def test_occurrences_stay_out_of_face_recognition(db):
    """Object markings must never leak into the faces domain."""
    source_path, collage_path, _other, _box = _collage_library(db)
    with session_scope() as session:
        source = _add_image(session, source_path)
        _add_image(session, collage_path)

        objects = ObjectService(session)
        obj = objects.create_object(name="Ellenőrzés")
        objects.add_occurrence_bbox(obj.id, source.id, 150, 110, 300, 220)

        service = ObjectMatchingService(session)
        service.find_object(obj.id)
        service.accept_suggestion(service.list_pending(obj.id)[0].suggestion_id)

        assert session.query(ObjectOccurrence).count() == 2
        from app.db.models import Face

        assert session.query(Face).count() == 0
