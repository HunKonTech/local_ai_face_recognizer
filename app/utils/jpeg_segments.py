"""Lossless JPEG metadata splicing (stdlib only).

Reads and replaces the APP1 ``Exif`` and APP1 XMP segments of a JPEG without
decoding or re-encoding the image: every other byte (APP0/JFIF, APP2 ICC
profile, MPF, quantisation tables, scan data, …) is copied verbatim.

Payload conventions:

* EXIF payload = the full APP1 body including the ``b"Exif\\x00\\x00"`` header
  (the same bytes Pillow exposes as ``img.info["exif"]`` and produces from
  ``Image.Exif.tobytes()``).
* XMP payload = the XMP packet only, *without* the namespace header.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

EXIF_HEADER = b"Exif\x00\x00"
XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"

# Largest payload a single marker segment can carry (length field is 16-bit
# and counts its own two bytes).
MAX_SEGMENT_PAYLOAD = 0xFFFF - 2

_SOI = b"\xff\xd8"
_APP0 = 0xE0
_APP1 = 0xE1
_SOS = 0xDA
_EOI = 0xD9
# Markers without a length field.
_STANDALONE = {0x01} | set(range(0xD0, 0xD8))

# Sentinel: leave this segment as it is.
KEEP = object()


class JpegStructureError(ValueError):
    """The data is not a JPEG we can safely parse."""


class SegmentTooLargeError(ValueError):
    """A payload does not fit into a single JPEG marker segment."""


@dataclass
class _Segment:
    marker: int
    start: int  # offset of the 0xFF marker byte
    end: int    # offset just past the segment
    payload: bytes


def _parse(data: bytes) -> tuple[list[_Segment], int]:
    """Return the header segments (up to, not including, SOS) and SOS offset."""
    if not data.startswith(_SOI):
        raise JpegStructureError("missing JPEG SOI marker")

    segments: list[_Segment] = []
    pos = 2
    n = len(data)
    while True:
        if pos >= n:
            raise JpegStructureError("JPEG ended before the image data (no SOS)")
        if data[pos] != 0xFF:
            raise JpegStructureError(f"expected a marker at offset {pos}")
        start = pos
        # Skip fill bytes (0xFF 0xFF …).
        while pos < n and data[pos] == 0xFF:
            pos += 1
        if pos >= n:
            raise JpegStructureError("truncated JPEG marker")
        marker = data[pos]
        pos += 1
        if marker == _SOS:
            return segments, start
        if marker == _EOI:
            raise JpegStructureError("JPEG has no image data (EOI before SOS)")
        if marker in _STANDALONE:
            segments.append(_Segment(marker, start, pos, b""))
            continue
        if pos + 2 > n:
            raise JpegStructureError("truncated JPEG segment length")
        length = int.from_bytes(data[pos:pos + 2], "big")
        if length < 2 or pos + length > n:
            raise JpegStructureError(f"truncated JPEG segment at offset {start}")
        payload = data[pos + 2:pos + length]
        pos += length
        segments.append(_Segment(marker, start, pos, payload))


def _is_exif(seg: _Segment) -> bool:
    return seg.marker == _APP1 and seg.payload.startswith(EXIF_HEADER)


def _is_xmp(seg: _Segment) -> bool:
    return seg.marker == _APP1 and seg.payload.startswith(XMP_HEADER)


def read_metadata(data: bytes) -> tuple[Optional[bytes], Optional[bytes]]:
    """Return ``(exif_payload, xmp_packet)`` of a JPEG (each may be None)."""
    segments, _ = _parse(data)
    exif = next((s.payload for s in segments if _is_exif(s)), None)
    xmp_seg = next((s for s in segments if _is_xmp(s)), None)
    xmp = xmp_seg.payload[len(XMP_HEADER):] if xmp_seg else None
    return exif, xmp


def _segment_bytes(payload: bytes) -> bytes:
    if len(payload) > MAX_SEGMENT_PAYLOAD:
        raise SegmentTooLargeError(
            f"metadata segment of {len(payload)} bytes exceeds the "
            f"{MAX_SEGMENT_PAYLOAD}-byte JPEG limit"
        )
    return b"\xff" + bytes([_APP1]) + (len(payload) + 2).to_bytes(2, "big") + payload


def check_exif_size(exif_payload: bytes) -> None:
    """Raise :class:`SegmentTooLargeError` when *exif_payload* cannot fit."""
    _segment_bytes(exif_payload)


def splice(data: bytes, *, exif=KEEP, xmp=KEEP) -> bytes:
    """Return a copy of JPEG *data* with replaced metadata segments.

    Args:
        exif: ``KEEP`` (default), ``None`` to remove the EXIF segment, or the new
            EXIF payload (must start with ``b"Exif\x00\x00"``).
        xmp: ``KEEP``, ``None`` to remove the XMP segment, or the new XMP packet
            bytes (without the namespace header).

    A replaced segment keeps the position of the one it replaces; a new one is
    inserted after the leading APP0 (JFIF) segments, XMP after EXIF.
    """
    segments, sos = _parse(data)

    if exif is not KEEP and exif is not None and not exif.startswith(EXIF_HEADER):
        raise ValueError("EXIF payload must start with the Exif header")
    new_exif = _segment_bytes(exif) if exif is not KEEP and exif is not None else None
    new_xmp = _segment_bytes(XMP_HEADER + xmp) if xmp is not KEEP and xmp is not None else None

    # (kind, bytes) chunks; kind marks our segments so we can place new ones.
    chunks: list[tuple[str, bytes]] = []
    for seg in segments:
        raw = data[seg.start:seg.end]
        if _is_exif(seg):
            if exif is KEEP:
                chunks.append(("exif", raw))
            elif not any(k == "exif" for k, _ in chunks):
                chunks.append(("exif", new_exif or b""))
            continue  # duplicates of a replaced segment are dropped
        if _is_xmp(seg):
            if xmp is KEEP:
                chunks.append(("xmp", raw))
            elif not any(k == "xmp" for k, _ in chunks):
                chunks.append(("xmp", new_xmp or b""))
            continue
        chunks.append(("app0" if seg.marker == _APP0 else "other", raw))

    def insert_at() -> int:
        idx = 0
        while idx < len(chunks) and chunks[idx][0] == "app0":
            idx += 1
        return idx

    if new_exif is not None and not any(k == "exif" for k, _ in chunks):
        chunks.insert(insert_at(), ("exif", new_exif))
    if new_xmp is not None and not any(k == "xmp" for k, _ in chunks):
        exif_idx = next((i for i, (k, _) in enumerate(chunks) if k == "exif"), None)
        chunks.insert(exif_idx + 1 if exif_idx is not None else insert_at(), ("xmp", new_xmp))

    return _SOI + b"".join(raw for _, raw in chunks) + data[sos:]
