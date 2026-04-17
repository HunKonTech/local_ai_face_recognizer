"""Unit tests for database models and session management."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, FaceCorrection, Image, Person


@pytest.fixture()
def tmp_db(tmp_path):
    """Initialise a fresh in-memory SQLite database for each test."""
    db_file = tmp_path / "test.db"
    init_db(db_file)
    return db_file


class TestImageModel:
    def test_create_and_retrieve(self, tmp_db):
        with session_scope() as s:
            img = Image(
                file_path="/tmp/photo.jpg",
                file_hash="abc123",
                file_mtime=1234567890.0,
            )
            s.add(img)

        with session_scope() as s:
            result = s.query(Image).filter(Image.file_path == "/tmp/photo.jpg").first()
            assert result is not None
            assert result.file_hash == "abc123"
            assert result.detection_done is False

    def test_unique_file_path(self, tmp_db):
        """Duplicate file_path should raise an integrity error."""
        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError):
            with session_scope() as s:
                s.add(Image(file_path="/dup.jpg", file_hash="x", file_mtime=1.0))
            with session_scope() as s:
                s.add(Image(file_path="/dup.jpg", file_hash="y", file_mtime=2.0))


class TestFaceEmbedding:
    def test_embedding_roundtrip(self, tmp_db):
        """set_embedding / get_embedding should round-trip without loss."""
        with session_scope() as s:
            img = Image(file_path="/tmp/img.jpg", file_hash="h", file_mtime=0.0)
            s.add(img)
            s.flush()

            face = Face(
                image_id=img.id,
                bbox_x=10, bbox_y=20, bbox_w=80, bbox_h=80,
                confidence=0.95,
                detector_backend="cpu",
            )
            original = np.random.default_rng(0).random(192).astype(np.float32)
            face.set_embedding(original)
            s.add(face)

        with session_scope() as s:
            face = s.query(Face).first()
            recovered = face.get_embedding()
            assert recovered is not None
            np.testing.assert_array_almost_equal(recovered, original, decimal=6)

    def test_null_embedding_returns_none(self, tmp_db):
        with session_scope() as s:
            img = Image(file_path="/tmp/img2.jpg", file_hash="h2", file_mtime=0.0)
            s.add(img)
            s.flush()
            face = Face(
                image_id=img.id,
                bbox_x=0, bbox_y=0, bbox_w=50, bbox_h=50,
                confidence=0.8,
                detector_backend="cpu",
            )
            s.add(face)

        with session_scope() as s:
            face = s.query(Face).first()
            assert face.get_embedding() is None


class TestPersonModel:
    def test_create_person(self, tmp_db):
        with session_scope() as s:
            p = Person(name="Alice", is_auto_named=False)
            s.add(p)

        with session_scope() as s:
            result = s.query(Person).filter(Person.name == "Alice").first()
            assert result is not None
            assert result.is_auto_named is False

    def test_cascade_delete_faces(self, tmp_db):
        """Deleting an Image should cascade to its Faces."""
        with session_scope() as s:
            img = Image(file_path="/tmp/del.jpg", file_hash="h", file_mtime=0.0)
            s.add(img)
            s.flush()
            face = Face(
                image_id=img.id,
                bbox_x=0, bbox_y=0, bbox_w=50, bbox_h=50,
                confidence=0.9,
                detector_backend="cpu",
            )
            s.add(face)
            s.flush()
            img_id = img.id
            face_id = face.id

        with session_scope() as s:
            img = s.get(Image, img_id)
            s.delete(img)

        with session_scope() as s:
            assert s.get(Face, face_id) is None
