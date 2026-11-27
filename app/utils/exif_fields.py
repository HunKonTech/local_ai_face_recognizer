"""Build EXIF blocks with Face-Local's edits applied.

One function, :func:`build_exif`, takes the existing EXIF payload (``Exif\\0\\0``
header included, as Pillow exposes it) plus a set of edits and returns the new
payload. ``piexif`` is used when installed (it round-trips every original tag,
including the IFD1 thumbnail); otherwise Pillow's own ``Image.Exif`` writer is
used, so the dependency stays optional.

Encodings follow the EXIF / Windows conventions:

* ``UserComment`` carries an 8-byte character code. Unicode text is written as
  ``UNICODE\\0`` + UTF-16 in the byte order of the EXIF block; pure ASCII text
  may use the ``ASCII\\0\\0\\0`` code.
* ``XPComment`` / ``XPKeywords`` (Windows Explorer "Comments" / "Tags") are
  always UTF-16LE with a terminating NUL, independent of the block's byte order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

EXIF_HEADER = b"Exif\x00\x00"

# Sentinel: leave the field untouched.
KEEP = object()

# 0th IFD
TAG_IMAGE_DESCRIPTION = 0x010E
TAG_ORIENTATION = 0x0112
TAG_DATETIME = 0x0132
TAG_EXIF_IFD = 0x8769
TAG_GPS_IFD = 0x8825
TAG_XP_COMMENT = 0x9C9C
TAG_XP_KEYWORDS = 0x9C9E
# Exif IFD
TAG_DATETIME_ORIGINAL = 0x9003
TAG_DATETIME_DIGITIZED = 0x9004
TAG_USER_COMMENT = 0x9286
# GPS IFD
TAG_GPS_LAT_REF, TAG_GPS_LAT = 0x0001, 0x0002
TAG_GPS_LON_REF, TAG_GPS_LON = 0x0003, 0x0004

UC_ASCII = b"ASCII\x00\x00\x00"
UC_UNICODE = b"UNICODE\x00"


@dataclass
class UserCommentText:
    """Text to store in ``UserComment``; ``ascii=True`` uses the ASCII code."""

    text: str
    ascii: bool = False


@dataclass
class ExifEdits:
    """Edits applied by :func:`build_exif`.

    ``KEEP`` leaves a field untouched, ``None`` removes it.
    """

    gps: Optional[tuple[float, float]] = None
    date: Optional[datetime] = None
    user_comment: object = KEEP  # KEEP | None | UserCommentText
    xp_comment: object = KEEP    # KEEP | None | str
    xp_keywords: object = KEEP   # KEEP | None | str

    def is_empty(self) -> bool:
        return (
            self.gps is None and self.date is None
            and self.user_comment is KEEP
            and self.xp_comment is KEEP and self.xp_keywords is KEEP
        )


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def byte_order(exif_payload: bytes) -> str:
    """``"<"`` (Intel, ``II``) or ``">"`` (Motorola, ``MM``) of an EXIF payload."""
    body = exif_payload[len(EXIF_HEADER):] if exif_payload.startswith(EXIF_HEADER) else exif_payload
    return "<" if body[:2] == b"II" else ">"


def encode_user_comment(value: UserCommentText, order: str) -> bytes:
    if value.ascii:
        return UC_ASCII + value.text.encode("ascii")
    codec = "utf-16-le" if order == "<" else "utf-16-be"
    return UC_UNICODE + value.text.encode(codec)


def decode_user_comment(raw, order: str = ">") -> Optional[str]:
    """Return the text of a ``UserComment`` value, or None if undecodable.

    Tolerates the legacy Face-Local encoding (``ASCII`` code + UTF-8 bytes) and
    UTF-16 written in either byte order (a BOM wins over *order*).
    """
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="replace")
    if isinstance(raw, tuple):
        raw = bytes(raw)
    if not isinstance(raw, (bytes, bytearray)) or len(raw) < 8:
        return None
    code, body = bytes(raw[:8]), bytes(raw[8:])
    if code == UC_ASCII:
        try:
            return body.decode("utf-8").rstrip("\x00")
        except UnicodeDecodeError:
            return body.decode("latin-1").rstrip("\x00")
    if code == UC_UNICODE:
        if body[:2] in (b"\xff\xfe", b"\xfe\xff"):
            codec = "utf-16"
        else:
            codec = "utf-16-le" if order == "<" else "utf-16-be"
            # Heuristic for writers that ignore the block's byte order: JSON
            # and names start with an ASCII char, so the zero byte tells.
            if len(body) >= 2 and body[0] == 0 and body[1] != 0:
                codec = "utf-16-be"
            elif len(body) >= 2 and body[1] == 0 and body[0] != 0:
                codec = "utf-16-le"
        try:
            return body.decode(codec).rstrip("\x00")
        except UnicodeDecodeError:
            return None
    return None


def encode_xp(text: str) -> bytes:
    return text.encode("utf-16-le") + b"\x00\x00"


def decode_xp(raw) -> Optional[str]:
    if isinstance(raw, tuple):
        raw = bytes(raw)
    if not isinstance(raw, (bytes, bytearray)):
        return None
    try:
        return bytes(raw).decode("utf-16-le").rstrip("\x00")
    except UnicodeDecodeError:
        return None


def _dms(value: float) -> list[tuple[int, int]]:
    value = abs(value)
    d = int(value)
    m = int((value - d) * 60)
    s = round((value - d - m / 60) * 3600 * 10_000)
    return [(d, 1), (m, 1), (s, 10_000)]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _pillow_exif(exif_payload: bytes):
    from PIL import Image as PilImage

    exif = PilImage.Exif()
    if exif_payload:
        exif.load(exif_payload)
    return exif


def read_user_comment(exif_payload: bytes) -> Optional[str]:
    """Decoded ``UserComment`` text of *exif_payload* (None if absent)."""
    if not exif_payload:
        return None
    try:
        raw = _pillow_exif(exif_payload).get_ifd(TAG_EXIF_IFD).get(TAG_USER_COMMENT)
    except Exception:  # noqa: BLE001
        return None
    return decode_user_comment(raw, byte_order(exif_payload)) if raw else None


def read_orientation(exif_payload: bytes) -> int:
    if not exif_payload:
        return 1
    try:
        value = int(_pillow_exif(exif_payload).get(TAG_ORIENTATION, 1))
    except Exception:  # noqa: BLE001
        return 1
    return value if 1 <= value <= 8 else 1


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def build_exif(existing: bytes, edits: ExifEdits) -> bytes:
    """Return a new EXIF payload: *existing* with *edits* applied."""
    try:
        import piexif  # noqa: F401
    except ImportError:
        build = _build_pillow
    else:
        build = _build_piexif

    # UserComment's UTF-16 byte order must match the block we produce; build
    # once, check the resulting order, rebuild only if it differs.
    out = build(existing, edits, "<")
    if isinstance(edits.user_comment, UserCommentText) and not edits.user_comment.ascii:
        order = byte_order(out)
        if order != "<":
            out = build(existing, edits, order)
    return out


def _build_pillow(existing: bytes, edits: ExifEdits, order: str) -> bytes:
    from PIL.TiffImagePlugin import IFDRational

    exif = _pillow_exif(b"")
    if existing:
        try:
            exif.load(existing)
        except Exception as exc:  # noqa: BLE001
            log.debug("Unparsable existing EXIF (%s) — writing a fresh block", exc)

    if edits.gps is not None:
        lat, lon = edits.gps
        gps_ifd = exif.get_ifd(TAG_GPS_IFD)
        gps_ifd[TAG_GPS_LAT_REF] = "N" if lat >= 0 else "S"
        gps_ifd[TAG_GPS_LAT] = tuple(IFDRational(n, d) for n, d in _dms(lat))
        gps_ifd[TAG_GPS_LON_REF] = "E" if lon >= 0 else "W"
        gps_ifd[TAG_GPS_LON] = tuple(IFDRational(n, d) for n, d in _dms(lon))
        exif[TAG_GPS_IFD] = gps_ifd

    sub_ifd = exif.get_ifd(TAG_EXIF_IFD)
    if edits.date is not None:
        stamp = edits.date.strftime("%Y:%m:%d %H:%M:%S")
        exif[TAG_DATETIME] = stamp
        sub_ifd[TAG_DATETIME_ORIGINAL] = stamp
        sub_ifd[TAG_DATETIME_DIGITIZED] = stamp

    if edits.user_comment is None:
        sub_ifd.pop(TAG_USER_COMMENT, None)
    elif isinstance(edits.user_comment, UserCommentText):
        sub_ifd[TAG_USER_COMMENT] = encode_user_comment(edits.user_comment, order)
    if sub_ifd:
        exif[TAG_EXIF_IFD] = sub_ifd

    for tag, value in ((TAG_XP_COMMENT, edits.xp_comment), (TAG_XP_KEYWORDS, edits.xp_keywords)):
        if value is None:
            if tag in exif:
                del exif[tag]
        elif isinstance(value, str):
            exif[tag] = encode_xp(value)

    return exif.tobytes()


def _build_piexif(existing: bytes, edits: ExifEdits, order: str) -> bytes:
    import piexif

    try:
        exif_dict = piexif.load(existing) if existing else None
    except Exception as exc:  # noqa: BLE001
        log.debug("piexif could not parse existing EXIF (%s) — writing a fresh block", exc)
        exif_dict = None
    if not exif_dict:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    for key in ("0th", "Exif", "GPS"):
        exif_dict.setdefault(key, {})

    if edits.gps is not None:
        lat, lon = edits.gps
        exif_dict["GPS"][TAG_GPS_LAT_REF] = b"N" if lat >= 0 else b"S"
        exif_dict["GPS"][TAG_GPS_LAT] = _dms(lat)
        exif_dict["GPS"][TAG_GPS_LON_REF] = b"E" if lon >= 0 else b"W"
        exif_dict["GPS"][TAG_GPS_LON] = _dms(lon)

    if edits.date is not None:
        stamp = edits.date.strftime("%Y:%m:%d %H:%M:%S").encode()
        exif_dict["0th"][TAG_DATETIME] = stamp
        exif_dict["Exif"][TAG_DATETIME_ORIGINAL] = stamp
        exif_dict["Exif"][TAG_DATETIME_DIGITIZED] = stamp

    if edits.user_comment is None:
        exif_dict["Exif"].pop(TAG_USER_COMMENT, None)
    elif isinstance(edits.user_comment, UserCommentText):
        exif_dict["Exif"][TAG_USER_COMMENT] = encode_user_comment(edits.user_comment, order)

    for tag, value in ((TAG_XP_COMMENT, edits.xp_comment), (TAG_XP_KEYWORDS, edits.xp_keywords)):
        if value is None:
            exif_dict["0th"].pop(tag, None)
        elif isinstance(value, str):
            exif_dict["0th"][tag] = encode_xp(value)

    return piexif.dump(exif_dict)
