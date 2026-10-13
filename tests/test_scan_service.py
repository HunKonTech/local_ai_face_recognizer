"""Unit tests for ScanService."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.config import ScanConfig
from app.db.database import init_db, session_scope
from app.db.models import Image
from app.services.scan_service import ScanService, hash_file


@pytest.fixture()
def tmp_db(tmp_path):
    db_file = tmp_path / "test_scan.db"
    init_db(db_file)
    return db_file


@pytest.fixture()
def image_folder(tmp_path):
    """Create a small folder tree with fake image files."""
    folder = tmp_path / "photos"
    (folder / "sub").mkdir(parents=True)

    for name in ["a.jpg", "b.jpeg", "c.png", "d.webp"]:
        (folder / name).write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

    # A file that should NOT be picked up
    (folder / "README.txt").write_text("not an image")

    # Subfolder image
    (folder / "sub" / "e.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

    return folder


class TestHashFile:
    def test_consistent_hash(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        assert hash_file(f) == hash_file(f)

    def test_different_content_different_hash(self, tmp_path):
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"aaa")
        b.write_bytes(b"bbb")
        assert hash_file(a) != hash_file(b)


class TestScanService:
    def test_scan_finds_correct_files(self, tmp_db, image_folder):
        with session_scope() as s:
            svc = ScanService(s, ScanConfig())
            ids = svc.scan(str(image_folder))

        # 4 root + 1 subfolder = 5 images, txt excluded
        assert len(ids) == 5

    def test_rescan_skips_unchanged(self, tmp_db, image_folder):
        with session_scope() as s:
            svc = ScanService(s, ScanConfig())
            first_ids = svc.scan(str(image_folder))

        # Mark all as detection_done so they look processed
        with session_scope() as s:
            s.query(Image).update({Image.detection_done: True})

        with session_scope() as s:
            svc = ScanService(s, ScanConfig())
            second_ids = svc.scan(str(image_folder))

        assert len(second_ids) == 0, "No new/changed files should be returned"

    def test_modified_file_requeued(self, tmp_db, image_folder):
        with session_scope() as s:
            svc = ScanService(s, ScanConfig())
            svc.scan(str(image_folder))

        # Mark all as done
        with session_scope() as s:
            s.query(Image).update({Image.detection_done: True})

        # Modify one file
        target = image_folder / "a.jpg"
        target.write_bytes(b"\xff\xd8\xff" + b"\x01" * 200)

        with session_scope() as s:
            svc = ScanService(s, ScanConfig())
            ids = svc.scan(str(image_folder))

        assert len(ids) == 1

    def test_custom_extensions(self, tmp_db, image_folder):
        """Only .png files should be indexed with a restricted extension list."""
        with session_scope() as s:
            cfg = ScanConfig(image_extensions=[".png"])
            svc = ScanService(s, cfg)
            ids = svc.scan(str(image_folder))

        assert len(ids) == 1


class TestMovedFiles:
    """#196 — a moved and renamed file keeps its existing record."""

    def test_moved_and_renamed_file_relinks_record(self, tmp_db, image_folder):
        target = image_folder / "a.jpg"
        target.write_bytes(b"\xff\xd8\xff" + b"\x07" * 300)
        with session_scope() as s:
            ScanService(s, ScanConfig()).scan(str(image_folder))
        with session_scope() as s:
            s.query(Image).update({Image.detection_done: True})
            original_id = s.query(Image.id).filter(Image.file_path == str(target)).scalar()

        moved = image_folder / "sub" / "deeper" / "renamed.jpg"
        moved.parent.mkdir(parents=True)
        target.rename(moved)

        with session_scope() as s:
            ids = ScanService(s, ScanConfig()).scan(str(image_folder))
        assert ids == []
        with session_scope() as s:
            assert s.query(Image).count() == 5
            assert s.get(Image, original_id).file_path == str(moved)

    def test_copy_with_original_present_gets_own_record(self, tmp_db, image_folder):
        with session_scope() as s:
            ScanService(s, ScanConfig()).scan(str(image_folder))
        with session_scope() as s:
            s.query(Image).update({Image.detection_done: True})
        (image_folder / "copy.jpg").write_bytes((image_folder / "a.jpg").read_bytes())
        with session_scope() as s:
            ids = ScanService(s, ScanConfig()).scan(str(image_folder))
        assert len(ids) == 1
        with session_scope() as s:
            assert s.query(Image).count() == 6
