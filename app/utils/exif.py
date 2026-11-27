"""Small, failure-tolerant EXIF helpers."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

log = logging.getLogger(__name__)


def _rational_to_float(value) -> float:  # noqa: ANN001
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return float(value.numerator) / float(value.denominator)
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def _dms_to_decimal(values, ref: str) -> Optional[float]:  # noqa: ANN001
    try:
        deg = _rational_to_float(values[0])
        minutes = _rational_to_float(values[1])
        seconds = _rational_to_float(values[2])
        decimal = deg + minutes / 60.0 + seconds / 3600.0
        if ref in {"S", "W"}:
            decimal *= -1
        return decimal
    except Exception as exc:  # noqa: BLE001
        log.debug("Invalid EXIF GPS DMS data: %s", exc)
        return None


def read_exif_gps(path: str | Path) -> Optional[Tuple[float, float]]:
    """Return ``(latitude, longitude)`` from EXIF GPS data, or ``None``.

    The caller can use this during import safely: unreadable files, missing GPS
    tags, or malformed coordinates are logged and reported as no GPS data.
    """
    try:
        from PIL import Image as PilImage
        from PIL.ExifTags import GPSTAGS

        with PilImage.open(path) as img:
            exif = img.getexif()
            gps_ifd = exif.get_ifd(0x8825) if exif else {}
        if not gps_ifd:
            return None

        gps = {GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
        lat = _dms_to_decimal(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef", "N"))
        lon = _dms_to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef", "E"))
        if lat is None or lon is None:
            return None
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            log.info("Ignoring out-of-range EXIF GPS for %s: %s, %s", path, lat, lon)
            return None
        return lat, lon
    except Exception as exc:  # noqa: BLE001
        log.info("EXIF GPS read failed for %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# EXIF writers — thin wrappers over app.utils.image_metadata, which applies
# every edit in one lossless file write (JPEG segment splice, no re-encode)
# and handles Windows file locks.
# ---------------------------------------------------------------------------

def write_exif_gps(path: str | Path, lat: float, lon: float) -> bool:
    """Write *lat*/*lon* into the image's EXIF GPS block.

    Returns True when the file was updated; False when it is missing,
    read-only, locked by another program, or cannot hold EXIF.
    """
    return _apply(path, "GPS", gps=(lat, lon))


def write_exif_date(path: str | Path, dt: datetime) -> bool:
    """Write *dt* into EXIF DateTimeOriginal (and DateTimeDigitized / DateTime)."""
    return _apply(path, "date", date=dt)


def _apply(path: str | Path, what: str, **edits) -> bool:
    from app.utils.image_metadata import apply_image_metadata

    path = Path(path)
    if not path.exists():
        log.warning("EXIF %s write: file not found: %s", what, path)
        return False
    result = apply_image_metadata(path, **edits)
    if result.embedded:
        log.info("EXIF %s written for %s", what, path)
        return True
    log.error("EXIF %s write failed for %s: %s (%s)", what, path,
              result.error_message, result.write_mode)
    return False


# Flexible date string patterns: "1954.03.12", "1954-03-12", "1954.03", "1954"
_DATE_PATTERNS: list[tuple[str, str]] = [
    (r"^\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}$", None),   # YYYY.MM.DD / YYYY-MM-DD
    (r"^\d{4}[.\-/]\d{1,2}$",               None),   # YYYY.MM
    (r"^\d{4}$",                             None),   # YYYY
]


def parse_flexible_date(date_str: str) -> Optional[datetime]:
    """Parse a flexible date string into a :class:`datetime`, or return ``None``.

    Handles: "1954.03.12", "1954-03-12", "1954/03/12", "1954:03:12", "1954.03",
    "1954", and any of those followed by a clock time (EXIF
    ``"1954:03:12 14:22:01"`` / filename ``"1954.03.12 14.22.01"``); the time is
    ignored.  Year-only → Jan 1st; year-month-only → 1st of that month.
    """
    if not date_str:
        return None

    s = date_str.strip()

    # Keep only the date portion: drop anything from the first space or "T".
    s = re.split(r"[ T]", s, maxsplit=1)[0]

    # Normalise separators (including EXIF colons) to "-"
    normalised = re.sub(r"[.:/]", "-", s)

    formats = ["%Y-%m-%d", "%Y-%m", "%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(normalised, fmt)
        except ValueError:
            continue

    return None


def validate_coords(text: str) -> Optional[Tuple[float, float]]:
    """Parse ``"lat, lon"`` text and validate ranges. Returns ``(lat, lon)`` or ``None``."""
    text = text.strip()
    if not text:
        return None
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        return None
    try:
        lat = float(parts[0])
        lon = float(parts[1])
    except ValueError:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return lat, lon
