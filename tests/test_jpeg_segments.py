"""Tests for the lossless JPEG metadata splicer."""

from __future__ import annotations

import io

import pytest
from PIL import Image as PilImage
from PIL import ImageCms

from app.utils import jpeg_segments as js


def _jpeg(*, exif: bytes | None = None, icc: bool = True, xmp: bytes | None = None) -> bytes:
    img = PilImage.new("RGB", (64, 48), (200, 100, 50))
    kwargs: dict = {"format": "JPEG", "quality": 90}
    if exif:
        kwargs["exif"] = exif
    if icc:
        kwargs["icc_profile"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    if xmp:
        kwargs["xmp"] = xmp
    buf = io.BytesIO()
    img.save(buf, **kwargs)
    return buf.getvalue()


def _scan(data: bytes) -> bytes:
    _, sos = js._parse(data)
    return data[sos:]


def _icc(data: bytes) -> bytes | None:
    with PilImage.open(io.BytesIO(data)) as im:
        return im.info.get("icc_profile")


def _exif_with_make(make: str) -> bytes:
    e = PilImage.Exif()
    e[0x010F] = make
    return e.tobytes()


def test_insert_exif_and_xmp_keeps_scan_and_icc():
    src = _jpeg()
    out = js.splice(src, exif=_exif_with_make("Cam"), xmp=b"<x:xmpmeta/>")

    assert _scan(out) == _scan(src)
    assert _icc(out) == _icc(src) is not None
    exif, xmp = js.read_metadata(out)
    assert exif == _exif_with_make("Cam")
    assert xmp == b"<x:xmpmeta/>"
    with PilImage.open(io.BytesIO(out)) as im:
        assert im.getexif().get(0x010F) == "Cam"
        im.load()  # still decodes


def test_replace_existing_segments_without_duplicates():
    src = _jpeg(exif=_exif_with_make("Old"), xmp=b"<old/>")
    out = js.splice(src, exif=_exif_with_make("New"), xmp=b"<new/>")
    out = js.splice(out, exif=_exif_with_make("Newer"), xmp=b"<newer/>")

    segs, _ = js._parse(out)
    assert sum(js._is_exif(s) for s in segs) == 1
    assert sum(js._is_xmp(s) for s in segs) == 1
    assert js.read_metadata(out) == (_exif_with_make("Newer"), b"<newer/>")
    assert _scan(out) == _scan(src)


def test_keep_and_remove():
    src = _jpeg(exif=_exif_with_make("Keep"), xmp=b"<x/>")
    out = js.splice(src, xmp=None)
    assert js.read_metadata(out) == (_exif_with_make("Keep"), None)
    assert js.splice(src) == src


def test_exif_inserted_after_app0():
    src = _jpeg(icc=False)
    out = js.splice(src, exif=_exif_with_make("Cam"))
    segs, _ = js._parse(out)
    assert segs[0].marker == 0xE0  # JFIF stays first
    assert js._is_exif(segs[1])


@pytest.mark.parametrize("bad", [b"", b"not a jpeg", b"\xff\xd8\xff\xe1\x00\x40abc"])
def test_malformed_input_raises(bad):
    with pytest.raises(js.JpegStructureError):
        js.read_metadata(bad)
    with pytest.raises(js.JpegStructureError):
        js.splice(bad, xmp=b"<x/>")


def test_truncated_before_scan_raises():
    src = _jpeg()
    _, sos = js._parse(src)
    with pytest.raises(js.JpegStructureError):
        js.splice(src[:sos], xmp=b"<x/>")


def test_oversize_payload_raises():
    src = _jpeg()
    with pytest.raises(js.SegmentTooLargeError):
        js.splice(src, xmp=b"x" * 70_000)
    with pytest.raises(js.SegmentTooLargeError):
        js.check_exif_size(js.EXIF_HEADER + b"x" * 70_000)
    # Exactly at the limit still fits.
    js.check_exif_size(b"x" * js.MAX_SEGMENT_PAYLOAD)
