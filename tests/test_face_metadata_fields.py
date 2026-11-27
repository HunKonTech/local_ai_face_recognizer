"""Field-level tests for embedded face metadata (XMP, XP tags, UserComment)."""

from __future__ import annotations

import builtins
import io
import json
import sys

import pytest
from PIL import Image as PilImage

from app.utils import exif_fields as ef
from app.utils import image_metadata as meta
from app.utils import jpeg_segments as js
from app.utils import xmp_edit as xe


def _payload(*names, unknown: int = 0, box=None) -> dict:
    faces = [
        {"person_id": i + 1, "person_name": n,
         "box": box or {"x": 10 + 50 * i, "y": 20, "width": 40, "height": 30}}
        for i, n in enumerate(names)
    ]
    faces += [{"unknown": True, "box": {"x": 1, "y": 1, "width": 5, "height": 5}}] * unknown
    return {"schema": meta.FACELOCAL_SCHEMA, "faces": faces}


def _jpeg(path, *, size=(200, 100), exif: bytes | None = None, xmp: bytes | None = None):
    kwargs: dict = {"format": "JPEG"}
    if exif:
        kwargs["exif"] = exif
    if xmp:
        kwargs["xmp"] = xmp
    PilImage.new("RGB", size, (50, 60, 70)).save(path, **kwargs)
    return path


def _segments(path):
    return js.read_metadata(path.read_bytes())


def _exif(path):
    exif, _ = _segments(path)
    parsed = PilImage.Exif()
    parsed.load(exif or b"")
    return exif or b"", parsed


@pytest.fixture()
def no_piexif(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "piexif":
            raise ImportError("piexif is not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "piexif", raising=False)


# ---------------------------------------------------------------------------
# 3.1 XMP editor
# ---------------------------------------------------------------------------

_FOREIGN_XMP = (
    b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>'
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Test">'
    b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    b'<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="4">'
    b"<xmp:Label>Red</xmp:Label></rdf:Description>"
    b"</rdf:RDF></x:xmpmeta>"
    b'<?xpacket end="w"?>'
)


def test_foreign_xmp_survives_and_owned_props_not_duplicated():
    regions = [xe.FaceRegion("A", 0.1, 0.1, 0.2, 0.2)]
    out = _FOREIGN_XMP
    for _ in range(3):
        out = xe.edit_xmp(out, json_text='{"x": 1}', regions=regions, applied_to=(100, 50))
    text = out.decode("utf-8")
    assert 'xmp:Rating="4"' in text and "<xmp:Label>Red</xmp:Label>" in text
    assert text.count("facelocal:data") == 2  # open + close tag, once
    assert text.count("MP:RegionInfo") == 2
    assert text.count("mwg-rs:Regions") == 2
    assert xe.read_facelocal_json(out) == '{"x": 1}'


def test_removing_regions_and_json():
    out = xe.edit_xmp(None, json_text="{}", regions=[xe.FaceRegion("A", 0, 0, 1, 1)],
                      applied_to=(10, 10))
    out = xe.edit_xmp(out, json_text=None, regions=[], applied_to=(10, 10))
    assert xe.read_region_names(out) == ([], [])
    assert xe.read_facelocal_json(out) is None


def test_unparsable_xmp_raises():
    with pytest.raises(xe.XmpParseError):
        xe.edit_xmp(b"<x:xmpmeta><broken", json_text="{}", regions=[], applied_to=(1, 1))


def test_unparsable_xmp_leaves_image_untouched(tmp_path):
    jpg = _jpeg(tmp_path / "bad.jpg", xmp=b"<x:xmpmeta><broken")
    before = jpg.read_bytes()
    result = meta.write_face_metadata(jpg, _payload("Alice"))
    assert result.write_mode == meta.WRITE_MODE_FAILED
    assert "XMP" in (result.error_message or "")
    assert jpg.read_bytes() == before
    assert result.sidecar_path and result.sidecar_path.exists()


# ---------------------------------------------------------------------------
# 3.2 XPComment / XPKeywords, no ImageDescription
# ---------------------------------------------------------------------------

def test_xp_tags_written_and_decoded(tmp_path, no_piexif):
    jpg = _jpeg(tmp_path / "xp.jpg")
    meta.write_face_metadata(jpg, _payload("Szilvay Géza", "Kiss Anna", unknown=1))
    _, exif = _exif(jpg)
    assert ef.decode_xp(exif.get(ef.TAG_XP_COMMENT)) == "Szilvay Géza; Kiss Anna"
    assert ef.decode_xp(exif.get(ef.TAG_XP_KEYWORDS)) == "Szilvay Géza;Kiss Anna"
    assert exif.get(ef.TAG_IMAGE_DESCRIPTION) is None


def test_xp_tags_removed_when_no_names(tmp_path):
    jpg = _jpeg(tmp_path / "xp.jpg")
    meta.write_face_metadata(jpg, _payload("Alice"))
    no_names = _payload("Alice")
    del no_names["faces"][0]["person_name"]
    meta.write_face_metadata(jpg, no_names)
    _, exif = _exif(jpg)
    assert exif.get(ef.TAG_XP_COMMENT) is None
    assert exif.get(ef.TAG_XP_KEYWORDS) is None


def test_reexport_drops_unassigned_person(tmp_path):
    jpg = _jpeg(tmp_path / "re.jpg")
    meta.write_face_metadata(jpg, _payload("A", "B"))
    meta.write_face_metadata(jpg, _payload("A"))
    _, exif = _exif(jpg)
    assert ef.decode_xp(exif.get(ef.TAG_XP_KEYWORDS)) == "A"
    _, xmp = _segments(jpg)
    assert xe.read_region_names(xmp) == (["A"], ["A"])


# ---------------------------------------------------------------------------
# 3.3 Region geometry (orientation-aware)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "orientation, expected_dims",
    [(1, (200, 100)), (3, (200, 100)), (6, (100, 200)), (8, (100, 200))],
)
def test_region_normalisation(orientation, expected_dims):
    dims = xe.oriented_size(200, 100, orientation)
    assert dims == expected_dims
    box = {"x": 20, "y": 10, "width": 40, "height": 20}
    (region,) = xe.regions_from_payload(_payload("A", box=box), *dims)
    w, h = dims
    assert region.x == pytest.approx(20 / w)
    assert region.y == pytest.approx(10 / h)
    assert region.w == pytest.approx(40 / w)
    assert region.h == pytest.approx(20 / h)


def test_regions_clamped_and_skipped():
    big = {"x": 150, "y": 80, "width": 100, "height": 100}
    (r,) = xe.regions_from_payload(_payload("A", box=big), 200, 100)
    assert r.x + r.w <= 1.0 and r.y + r.h <= 1.0
    assert xe.regions_from_payload(_payload(unknown=2), 200, 100) == []


def test_rotated_jpeg_regions_use_oriented_dims(tmp_path):
    exif = PilImage.Exif()
    exif[ef.TAG_ORIENTATION] = 6
    jpg = _jpeg(tmp_path / "rot.jpg", size=(200, 100), exif=exif.tobytes())
    meta.write_face_metadata(jpg, _payload("A", box={"x": 10, "y": 50, "width": 50, "height": 100}))
    _, xmp = _segments(jpg)
    text = xmp.decode("utf-8")
    assert 'stDim:w="100"' in text and 'stDim:h="200"' in text
    assert "0.100000, 0.250000, 0.500000, 0.500000" in text  # MP: left, top, w, h
    assert 'stArea:x="0.350000"' in text and 'stArea:y="0.500000"' in text  # MWG centre


# ---------------------------------------------------------------------------
# 3.4 JSON in UserComment and XMP
# ---------------------------------------------------------------------------

def test_user_comment_and_xmp_hold_same_document(tmp_path, no_piexif):
    jpg = _jpeg(tmp_path / "both.jpg")
    payload = _payload("Szilvay Géza")
    result = meta.write_face_metadata(jpg, payload)
    assert result.write_mode == meta.WRITE_MODE_EXIF_USER_COMMENT

    exif, xmp = _segments(jpg)
    uc = ef.read_user_comment(exif)
    assert json.loads(uc) == json.loads(xe.read_facelocal_json(xmp)) == payload
    assert "Szilvay Géza" in uc  # readable, not mojibake / escapes


def test_legacy_ascii_user_comment_rewritten_as_unicode(tmp_path):
    legacy = PilImage.Exif()
    sub = legacy.get_ifd(ef.TAG_EXIF_IFD)
    sub[ef.TAG_USER_COMMENT] = ef.UC_ASCII + json.dumps(_payload("Old"), ensure_ascii=False).encode()
    legacy[ef.TAG_EXIF_IFD] = sub
    jpg = _jpeg(tmp_path / "legacy.jpg", exif=legacy.tobytes())

    meta.write_face_metadata(jpg, _payload("Új Név"))
    _, exif = _exif(jpg)
    raw = exif.get_ifd(ef.TAG_EXIF_IFD).get(ef.TAG_USER_COMMENT)
    assert raw.startswith(ef.UC_UNICODE)
    assert meta.read_face_metadata(jpg)["faces"][0]["person_name"] == "Új Név"


def test_foreign_user_comment_untouched(tmp_path):
    foreign = PilImage.Exif()
    sub = foreign.get_ifd(ef.TAG_EXIF_IFD)
    sub[ef.TAG_USER_COMMENT] = ef.UC_ASCII + b"Holiday 1998"
    foreign[ef.TAG_EXIF_IFD] = sub
    jpg = _jpeg(tmp_path / "foreign.jpg", exif=foreign.tobytes())

    result = meta.write_face_metadata(jpg, _payload("Alice"))
    assert result.write_mode == meta.WRITE_MODE_XMP
    assert "UserComment" in (result.error_message or "")
    exif, _ = _segments(jpg)
    assert ef.read_user_comment(exif) == "Holiday 1998"
    assert meta.read_face_metadata(jpg)["faces"][0]["person_name"] == "Alice"


def test_exif_comment_disabled_writes_xmp_only_and_drops_stale_copy(tmp_path):
    jpg = _jpeg(tmp_path / "x.jpg")
    meta.write_face_metadata(jpg, _payload("Alice"))
    result = meta.write_face_metadata(jpg, _payload("Bob"), prefer_exif_comment=False)
    assert result.write_mode == meta.WRITE_MODE_XMP
    exif, _ = _segments(jpg)
    assert ef.read_user_comment(exif) is None
    assert meta.read_face_metadata(jpg)["faces"][0]["person_name"] == "Bob"


def test_oversize_user_comment_steps_down(tmp_path, no_piexif):
    jpg = _jpeg(tmp_path / "big.jpg")
    # ~40 KB of JSON: UTF-16 (~80 KB) does not fit, ASCII-escaped does.
    payload = _payload("Szilvay Géza")
    payload["filler"] = "a" * 40_000
    result = meta.write_face_metadata(jpg, payload)
    assert result.write_mode == meta.WRITE_MODE_EXIF_USER_COMMENT
    assert "ASCII-escaped" in (result.error_message or "")
    exif, xmp = _segments(jpg)
    assert json.loads(ef.read_user_comment(exif)) == payload

    # Too big even for ASCII → UserComment omitted, XMP still has it.
    # UTF-8 ~60 KB (fits XMP); UTF-16 ~100 KB and \u-escaped ~100 KB don't fit EXIF.
    payload["filler"] = "a" * 40_000
    payload["more"] = "é" * 10_000
    result = meta.write_face_metadata(jpg, payload)
    assert result.write_mode == meta.WRITE_MODE_XMP
    assert "omitted" in (result.error_message or "")
    exif, xmp = _segments(jpg)
    assert ef.read_user_comment(exif) is None
    assert json.loads(xe.read_facelocal_json(xmp)) == payload


# ---------------------------------------------------------------------------
# 3.5 Legacy reads
# ---------------------------------------------------------------------------

def _exif_with_comment(raw: bytes) -> bytes:
    e = PilImage.Exif()
    sub = e.get_ifd(ef.TAG_EXIF_IFD)
    sub[ef.TAG_USER_COMMENT] = raw
    e[ef.TAG_EXIF_IFD] = sub
    return e.tobytes()


@pytest.mark.parametrize("codec", ["utf-16-le", "utf-16-be"])
def test_read_unicode_user_comment_any_byte_order(tmp_path, codec):
    text = json.dumps(_payload("Géza"), ensure_ascii=False)
    jpg = _jpeg(tmp_path / "u.jpg", exif=_exif_with_comment(ef.UC_UNICODE + text.encode(codec)))
    assert meta.read_face_metadata(jpg)["faces"][0]["person_name"] == "Géza"


def test_read_legacy_ascii_utf8_user_comment(tmp_path):
    text = json.dumps(_payload("Géza"), ensure_ascii=False)
    jpg = _jpeg(tmp_path / "a.jpg", exif=_exif_with_comment(ef.UC_ASCII + text.encode("utf-8")))
    assert meta.read_face_metadata(jpg)["faces"][0]["person_name"] == "Géza"


def test_read_legacy_regex_xmp(tmp_path):
    legacy = (
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        b'<rdf:Description rdf:about="" xmlns:facelocal="http://facelocal.app/ns/faces/1.0/">'
        b"<facelocal:data>{&quot;schema&quot;: &quot;facelocal.faces.v1&quot;, &quot;faces&quot;: []}</facelocal:data>"
        b"</rdf:Description></rdf:RDF></x:xmpmeta>"
    )
    jpg = _jpeg(tmp_path / "x.jpg", xmp=legacy)
    assert meta.read_face_metadata(jpg) == {"schema": meta.FACELOCAL_SCHEMA, "faces": []}


def test_png_gets_xmp_regions_and_keeps_icc(tmp_path):
    from PIL import ImageCms

    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    png = tmp_path / "p.png"
    PilImage.new("RGB", (100, 80)).save(png, format="PNG", icc_profile=icc)
    result = meta.write_face_metadata(png, _payload("Alice"))
    assert result.write_mode == meta.WRITE_MODE_XMP
    with PilImage.open(png) as im:
        assert im.info.get("icc_profile") == icc
        xmp = im.text.get("XML:com.adobe.xmp")
    assert xe.read_region_names(xmp) == (["Alice"], ["Alice"])
    assert meta.read_face_metadata(png)["faces"][0]["person_name"] == "Alice"


def test_jpeg_scan_bytes_unchanged_by_face_export(tmp_path):
    jpg = _jpeg(tmp_path / "s.jpg")
    before = jpg.read_bytes()
    meta.write_face_metadata(jpg, _payload("Alice"))
    after = jpg.read_bytes()
    assert after[js._parse(after)[1]:] == before[js._parse(before)[1]:]
    with PilImage.open(io.BytesIO(after)) as im:
        im.load()
