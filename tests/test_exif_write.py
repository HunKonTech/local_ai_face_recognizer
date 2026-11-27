"""Tests for the lock-tolerant EXIF GPS/date writers in app.utils.exif."""

from __future__ import annotations

from datetime import datetime

from PIL import Image as PilImage

from app.utils.exif import read_exif_gps, write_exif_date, write_exif_gps


def _read_datetime_original(path):
    """DateTimeOriginal as a string, read without depending on piexif."""
    with PilImage.open(path) as im:
        return im.getexif().get_ifd(0x8769).get(0x9003)


def _make_jpeg(path, *, exif_bytes: bytes | None = None) -> None:
    img = PilImage.new("RGB", (120, 90), (30, 60, 90))
    if exif_bytes:
        img.save(path, format="JPEG", exif=exif_bytes)
    else:
        img.save(path, format="JPEG")


def test_write_gps_and_date_roundtrip(tmp_path):
    jpg = tmp_path / "p.jpg"
    _make_jpeg(jpg)

    assert write_exif_gps(jpg, 47.5, 19.05) is True
    assert write_exif_date(jpg, datetime(1985, 7, 20, 0, 0, 0)) is True

    lat, lon = read_exif_gps(jpg)
    assert round(lat, 3) == 47.5
    assert round(lon, 3) == 19.05

    assert _read_datetime_original(jpg) == "1985:07:20 00:00:00"


def test_gps_and_date_coexist(tmp_path):
    """Writing the date must not wipe the GPS written earlier (and vice versa)."""
    jpg = tmp_path / "p.jpg"
    _make_jpeg(jpg)

    write_exif_gps(jpg, 47.5, 19.05)
    write_exif_date(jpg, datetime(2001, 2, 3))

    assert read_exif_gps(jpg) is not None
    assert _read_datetime_original(jpg) == "2001:02:03 00:00:00"


def test_write_gps_locked_file_left_intact(tmp_path, monkeypatch):
    """A persistent Windows lock → False, original untouched, no temp file."""
    jpg = tmp_path / "locked.jpg"
    _make_jpeg(jpg)
    before = jpg.read_bytes()

    import app.utils.image_metadata as meta

    real_replace = meta.os.replace

    def fake_replace(src, dst):
        if str(dst).endswith("locked.jpg"):
            raise PermissionError("simulated Windows lock")
        return real_replace(src, dst)

    monkeypatch.setattr(meta.os, "replace", fake_replace)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    assert write_exif_gps(jpg, 47.5, 19.05) is False
    assert jpg.read_bytes() == before
    assert not (tmp_path / "locked.jpg.facelocal.tmp").exists()


def test_write_gps_rides_out_short_lock(tmp_path, monkeypatch):
    """A lock that clears after a few retries still gets the write in."""
    jpg = tmp_path / "busy.jpg"
    _make_jpeg(jpg)

    import app.utils.image_metadata as meta

    real_replace = meta.os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("indexer busy")
        return real_replace(src, dst)

    monkeypatch.setattr(meta.os, "replace", flaky_replace)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    assert write_exif_gps(jpg, 47.5, 19.05) is True
    assert read_exif_gps(jpg) is not None


def test_write_gps_and_date_keep_pixels_and_icc(tmp_path):
    """GPS/date writes splice metadata only: scan data and ICC are byte-identical."""
    from PIL import ImageCms

    from app.utils import jpeg_segments as js

    jpg = tmp_path / "icc.jpg"
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    PilImage.new("RGB", (80, 60), (5, 99, 200)).save(jpg, format="JPEG", icc_profile=icc)
    before = jpg.read_bytes()

    assert write_exif_gps(jpg, 47.5, 19.05)
    assert write_exif_date(jpg, datetime(2001, 2, 3))

    after = jpg.read_bytes()
    assert after[js._parse(after)[1]:] == before[js._parse(before)[1]:]
    with PilImage.open(jpg) as im:
        assert im.info.get("icc_profile") == icc


def test_write_gps_quality_preserved_no_recompress_blowup(tmp_path):
    """A repeated GPS write should not keep shrinking/inflating the JPEG wildly."""
    jpg = tmp_path / "p.jpg"
    _make_jpeg(jpg)
    write_exif_gps(jpg, 47.5, 19.05)
    size1 = jpg.stat().st_size
    write_exif_gps(jpg, 47.6, 19.06)
    size2 = jpg.stat().st_size
    # Lossless splice: only the EXIF segment changes, its size stays put.
    assert abs(size2 - size1) < 64


def test_missing_file_returns_false(tmp_path):
    assert write_exif_gps(tmp_path / "nope.jpg", 1.0, 2.0) is False
    assert write_exif_date(tmp_path / "nope.jpg", datetime(2000, 1, 1)) is False
