"""Small, failure-tolerant EXIF helpers."""

from __future__ import annotations

import logging
import os
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
# EXIF write helpers
# ---------------------------------------------------------------------------

def _decimal_to_dms_rational(value: float) -> list:
    """Convert a decimal-degrees float to a list of (numerator, denominator) tuples."""
    value = abs(value)
    d = int(value)
    m = int((value - d) * 60)
    s = round((value - d - m / 60) * 3600 * 10_000)
    return [(d, 1), (m, 1), (s, 10_000)]


def _rewrite_with_exif(path: Path, new_exif: bytes) -> None:
    """Re-save *path* with *new_exif*, safely on Windows.

    Writes to a temp file (never the locked target) and then performs a
    lock-tolerant atomic replace — the same strategy used for embedded face
    metadata. ``quality="keep"`` avoids re-compressing the JPEG. Reuses
    :func:`app.utils.image_metadata._replace_atomic` so the Windows file-lock
    handling lives in one place.
    """
    from PIL import Image as PilImage

    from app.utils.image_metadata import _replace_atomic, _tmp_for

    tmp = _tmp_for(path)
    with PilImage.open(path) as src:
        fmt = src.format or "JPEG"
        src.load()
        save_kwargs: dict = {"format": fmt, "exif": new_exif}
        if fmt == "JPEG":
            save_kwargs["quality"] = "keep"
        try:
            src.save(tmp, **save_kwargs)
        except (ValueError, OSError):
            # "keep" needs the original quant tables; fall back to high quality.
            save_kwargs.pop("quality", None)
            if fmt == "JPEG":
                save_kwargs["quality"] = 95
            src.save(tmp, **save_kwargs)
    _replace_atomic(tmp, path)


# Tag numbers for the piexif-free write path.
_GPS_IFD_TAG = 0x8825
_EXIF_IFD_TAG = 0x8769
_TAG_GPS_LAT_REF, _TAG_GPS_LAT = 0x0001, 0x0002
_TAG_GPS_LON_REF, _TAG_GPS_LON = 0x0003, 0x0004
_TAG_DATETIME = 0x0132
_TAG_DATETIME_ORIGINAL = 0x9003
_TAG_DATETIME_DIGITIZED = 0x9004


def _load_pillow_exif(path: Path):
    """Return a ``PIL.Image.Exif`` holding the file's current EXIF (or empty)."""
    from PIL import Image as PilImage

    exif = PilImage.Exif()
    with PilImage.open(path) as img:
        raw = img.info.get("exif", b"")
    if raw:
        try:
            exif.load(raw)
        except Exception as exc:  # noqa: BLE001
            log.debug("Unparsable existing EXIF in %s (%s) — starting fresh", path, exc)
    return exif


def _write_exif_gps_pillow(path: Path, lat: float, lon: float) -> bool:
    """GPS writer used when piexif is not installed."""
    try:
        from PIL.TiffImagePlugin import IFDRational

        def dms(value: float):
            deg, minutes, seconds = _decimal_to_dms_rational(abs(value))
            return tuple(IFDRational(n, d) for n, d in (deg, minutes, seconds))

        exif = _load_pillow_exif(path)
        gps_ifd = exif.get_ifd(_GPS_IFD_TAG)
        gps_ifd[_TAG_GPS_LAT_REF] = "N" if lat >= 0 else "S"
        gps_ifd[_TAG_GPS_LAT] = dms(lat)
        gps_ifd[_TAG_GPS_LON_REF] = "E" if lon >= 0 else "W"
        gps_ifd[_TAG_GPS_LON] = dms(lon)
        exif[_GPS_IFD_TAG] = gps_ifd

        _rewrite_with_exif(path, exif.tobytes())
        log.info("EXIF GPS written (Pillow) for %s: %.6f, %.6f", path, lat, lon)
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("EXIF GPS write failed for %s: %s", path, exc)
        return False


def _write_exif_date_pillow(path: Path, dt: datetime) -> bool:
    """Date writer used when piexif is not installed."""
    try:
        stamp = dt.strftime("%Y:%m:%d %H:%M:%S")
        exif = _load_pillow_exif(path)
        exif[_TAG_DATETIME] = stamp
        sub_ifd = exif.get_ifd(_EXIF_IFD_TAG)
        sub_ifd[_TAG_DATETIME_ORIGINAL] = stamp
        sub_ifd[_TAG_DATETIME_DIGITIZED] = stamp
        exif[_EXIF_IFD_TAG] = sub_ifd

        _rewrite_with_exif(path, exif.tobytes())
        log.info("EXIF date written (Pillow) for %s: %s", path, stamp)
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("EXIF date write failed for %s: %s", path, exc)
        return False


def write_exif_gps(path: str | Path, lat: float, lon: float) -> bool:
    """Write GPS coordinates into the EXIF of an image file.

    Uses ``piexif`` when installed and Pillow's own EXIF writer otherwise, so
    the write still happens on installs without the optional package.
    The file must exist and be writable. Returns True on success.
    """
    path = Path(path)
    if not path.exists():
        log.warning("EXIF GPS write: file not found: %s", path)
        return False
    if not os.access(path, os.W_OK):
        log.warning("EXIF GPS write: file not writable: %s", path)
        return False

    try:
        import piexif  # type: ignore[import]
    except ImportError:
        log.debug("piexif not installed — writing EXIF GPS via Pillow for %s", path)
        return _write_exif_gps_pillow(path, lat, lon)

    try:
        from PIL import Image as PilImage

        with PilImage.open(path) as img:
            raw = img.info.get("exif", b"")

        exif_dict: dict = piexif.load(raw) if raw else {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

        exif_dict.setdefault("GPS", {})
        exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = (b"N" if lat >= 0 else b"S")
        exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = _decimal_to_dms_rational(lat)
        exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = (b"E" if lon >= 0 else b"W")
        exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = _decimal_to_dms_rational(lon)

        new_exif = piexif.dump(exif_dict)
        _rewrite_with_exif(path, new_exif)

        log.info("EXIF GPS written for %s: %.6f, %.6f", path, lat, lon)
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("EXIF GPS write failed for %s: %s", path, exc)
        return False


def write_exif_date(path: str | Path, dt: datetime) -> bool:
    """Write *dt* into the EXIF DateTimeOriginal (and DateTimeDigitized / DateTime) fields.

    Uses ``piexif`` when installed and Pillow's own EXIF writer otherwise.
    """
    path = Path(path)
    if not path.exists():
        log.warning("EXIF date write: file not found: %s", path)
        return False
    if not os.access(path, os.W_OK):
        log.warning("EXIF date write: file not writable: %s", path)
        return False

    try:
        import piexif  # type: ignore[import]
    except ImportError:
        log.debug("piexif not installed — writing EXIF date via Pillow for %s", path)
        return _write_exif_date_pillow(path, dt)

    try:
        from PIL import Image as PilImage

        dt_bytes = dt.strftime("%Y:%m:%d %H:%M:%S").encode()

        with PilImage.open(path) as img:
            raw = img.info.get("exif", b"")

        exif_dict: dict = piexif.load(raw) if raw else {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

        exif_dict.setdefault("0th", {})
        exif_dict.setdefault("Exif", {})
        exif_dict["0th"][piexif.ImageIFD.DateTime] = dt_bytes
        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = dt_bytes
        exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = dt_bytes

        new_exif = piexif.dump(exif_dict)
        _rewrite_with_exif(path, new_exif)

        log.info("EXIF date written for %s: %s", path, dt.strftime("%Y:%m:%d %H:%M:%S"))
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("EXIF date write failed for %s: %s", path, exc)
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
