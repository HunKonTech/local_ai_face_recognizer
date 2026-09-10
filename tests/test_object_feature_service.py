"""Tests for ORB feature extraction and caching behind object matching (#164)."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from app.config import ObjectMatchingConfig
from app.db.database import init_db, session_scope
from app.db.models import Image, ImageFeatures, ObjectPatchFeatures
from app.services.object_feature_service import (
    ObjectFeatureService,
    crop_bbox,
    extract_features,
    pack_features,
    params_hash,
    unpack_features,
)
from app.services.object_service import ObjectService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_texture(width: int, height: int, seed: int) -> np.ndarray:
    """A synthetic image with structure that survives downscaling.

    Random shapes rather than per-pixel noise: noise disappears when an image is
    shrunk, so it could never exercise the scale-invariance this feature is for.
    """
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 30, dtype=np.uint8)
    for _ in range(90):
        color = tuple(int(c) for c in rng.integers(40, 255, size=3))
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        if rng.random() < 0.5:
            w = int(rng.integers(width // 20 + 2, width // 4 + 3))
            h = int(rng.integers(height // 20 + 2, height // 4 + 3))
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        else:
            r = int(rng.integers(width // 20 + 2, width // 6 + 3))
            cv2.circle(img, (x, y), r, color, -1)
    return img


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


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def test_extract_features_returns_keypoints():
    config = ObjectMatchingConfig()
    fs = extract_features(make_texture(320, 240, seed=1), config)
    assert fs is not None
    assert fs.usable
    assert fs.width == 320 and fs.height == 240
    assert fs.descriptors.shape[1] == 32
    assert fs.keypoints.shape[0] == fs.descriptors.shape[0]


def test_extract_features_rescales_keypoints_to_original_pixels():
    """A large image is downscaled for speed, but coordinates come back full-size."""
    config = ObjectMatchingConfig(max_work_edge=200)
    fs = extract_features(make_texture(800, 600, seed=2), config)
    assert fs is not None
    assert fs.work_scale == pytest.approx(0.25)
    assert fs.keypoints[:, 0].max() > 200


def test_extract_features_rejects_empty_input():
    config = ObjectMatchingConfig()
    assert extract_features(np.zeros((4, 4, 3), dtype=np.uint8), config) is None


def test_pack_unpack_roundtrip():
    config = ObjectMatchingConfig()
    fs = extract_features(make_texture(300, 200, seed=3), config)
    assert fs is not None
    kp_blob, desc_blob = pack_features(fs)
    restored = unpack_features(kp_blob, desc_blob, fs.width, fs.height, fs.work_scale)
    assert restored is not None
    assert restored.count == fs.count
    np.testing.assert_allclose(restored.keypoints, fs.keypoints)
    np.testing.assert_array_equal(restored.descriptors, fs.descriptors)


def test_unpack_features_handles_missing_blobs():
    assert unpack_features(None, None, 10, 10) is None
    assert unpack_features(b"", b"", 10, 10) is None


def test_params_hash_changes_with_settings():
    base = ObjectMatchingConfig()
    other = ObjectMatchingConfig(pyramid_levels=6)
    assert params_hash(base, "image") != params_hash(other, "image")
    assert params_hash(base, "image") != params_hash(base, "patch")


def test_crop_bbox_clamps_to_image():
    img = make_texture(100, 80, seed=4)
    crop = crop_bbox(img, (90, 70, 50, 50))
    assert crop is not None
    assert crop.shape[0] <= 10 and crop.shape[1] <= 10


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def test_ensure_image_features_caches_and_reuses(db):
    path = os.path.join(db, "scene.png")
    cv2.imwrite(path, make_texture(400, 300, seed=5))
    with session_scope() as session:
        image = _add_image(session, path)
        service = ObjectFeatureService(session)
        first = service.ensure_image_features([image.id])
        assert first.processed == 1
        assert service.load_image_features(image.id) is not None

        second = service.ensure_image_features([image.id])
        assert second.processed == 0
        assert second.reused == 1


def test_stale_params_hash_forces_recompute(db):
    path = os.path.join(db, "scene.png")
    cv2.imwrite(path, make_texture(400, 300, seed=6))
    with session_scope() as session:
        image = _add_image(session, path)
        ObjectFeatureService(session).ensure_image_features([image.id])
        row = session.get(ImageFeatures, image.id)
        row.params_hash = "outdated"
        session.flush()

        service = ObjectFeatureService(session)
        assert service.load_image_features(image.id) is None
        assert service.pending_image_ids([image.id]) == [image.id]


def test_unreadable_image_is_marked_not_retried(db):
    with session_scope() as session:
        image = _add_image(session, os.path.join(db, "missing.png"))
        stats = ObjectFeatureService(session).ensure_image_features([image.id])
        assert stats.failed == 1
        # The empty marker means the next run does not decode the file again.
        assert session.get(ImageFeatures, image.id) is not None
        assert ObjectFeatureService(session).pending_image_ids([image.id]) == []


def test_ensure_image_features_honours_cancel(db):
    paths = []
    for index in range(6):
        path = os.path.join(db, f"scene{index}.png")
        cv2.imwrite(path, make_texture(300, 220, seed=10 + index))
        paths.append(path)
    with session_scope() as session:
        ids = [_add_image(session, p).id for p in paths]
        service = ObjectFeatureService(session)
        service.config = ObjectMatchingConfig(max_workers=1)
        stats = service.ensure_image_features(ids, cancel_check=lambda: True)
        assert stats.cancelled
        assert stats.processed == 0


def test_ensure_patch_features_extracts_reference_crops(db):
    path = os.path.join(db, "scene.png")
    scene = np.full((400, 500, 3), 20, dtype=np.uint8)
    scene[50:250, 100:400] = make_texture(300, 200, seed=7)
    cv2.imwrite(path, scene)
    with session_scope() as session:
        image = _add_image(session, path)
        objects = ObjectService(session)
        obj = objects.create_object(name="Klotild")
        occ = objects.add_occurrence_bbox(obj.id, image.id, 100, 50, 300, 200)

        service = ObjectFeatureService(session)
        patches = service.ensure_patch_features([occ])
        assert occ.id in patches
        assert patches[occ.id].width == 300
        assert session.get(ObjectPatchFeatures, occ.id).n_features > 0

        service.invalidate_patch(occ.id)
        assert session.get(ObjectPatchFeatures, occ.id) is None


def test_reference_occurrences_only_returns_bbox_markings(db):
    path = os.path.join(db, "scene.png")
    cv2.imwrite(path, make_texture(400, 300, seed=8))
    with session_scope() as session:
        image = _add_image(session, path)
        objects = ObjectService(session)
        obj = objects.create_object(name="Asztal")
        objects.add_occurrence(obj.id, image.id, 10, 10)
        boxed = objects.add_occurrence_bbox(obj.id, image.id, 40, 40, 100, 80)

        refs = ObjectFeatureService(session).reference_occurrences(obj.id)
        assert [r.id for r in refs] == [boxed.id]
