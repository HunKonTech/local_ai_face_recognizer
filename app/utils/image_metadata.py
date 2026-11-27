"""Image metadata writer for face-local annotations, GPS and dates.

All metadata edits for one image go through :func:`apply_image_metadata`,
which reads the file once, applies every edit and replaces the file once.

What is written for a ``facelocal.faces.v1`` payload:

* **JPEG** — lossless segment splice (no re-encode; ICC, MPF etc. kept):
  - EXIF ``UserComment``: the JSON (``UNICODE`` code, UTF-16) — optional,
  - EXIF ``XPComment`` / ``XPKeywords``: known person names (Windows
    Explorer "Comments" / "Tags"),
  - XMP ``facelocal:data`` (the JSON), ``MP:RegionInfo`` and
    ``mwg-rs:Regions`` (named face rectangles — Windows "People", MWG tools).
* **PNG** — the XMP part (lossless re-save, ICC/EXIF kept).
* **Anything else / read-only / locked** — a sidecar ``<image>.facelocal.json``.

Safety: the new file is written to a temp file next to the image, fsynced and
swapped in with ``os.replace``. If another program keeps the image open
(Windows), the result is :data:`WRITE_MODE_LOCKED`, the original stays
untouched and the payload is preserved in the sidecar.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.utils import exif_fields as ef
from app.utils import jpeg_segments as js
from app.utils import xmp_edit as xe

log = logging.getLogger(__name__)

# Schema identifier carried inside every payload — bump when the shape changes.
FACELOCAL_SCHEMA = "facelocal.faces.v1"

# Write-mode result constants.
WRITE_MODE_XMP = "xmp"
WRITE_MODE_EXIF_USER_COMMENT = "exif_user_comment"  # JSON in UserComment *and* XMP
WRITE_MODE_EXIF_IMAGE_DESCRIPTION = "exif_image_description"  # legacy, read-only
WRITE_MODE_EXIF = "exif"  # GPS/date only, no face payload
WRITE_MODE_SIDECAR = "sidecar"
WRITE_MODE_LOCKED = "locked"
WRITE_MODE_SKIPPED = "skipped"
WRITE_MODE_FAILED = "failed"

# File extensions we will *attempt* to embed face data into.  Everything else
# (RAW, GIF, HEIC, …) falls back to a sidecar JSON file.
_EMBEDDABLE_EXTS = {".jpg", ".jpeg", ".jpe", ".png"}

# Standard XMP-in-PNG iTXt keyword.
_PNG_XMP_KEY = "XML:com.adobe.xmp"

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Replace retry schedule (seconds) — about 2.8 s in total, enough for the
# Search indexer / antivirus to let go of a freshly written file.
_REPLACE_BACKOFF = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

# Windows error codes meaning "another process has the file open".
_LOCK_WINERRORS = {5, 32, 33}


class FileLockedError(OSError):
    """The image could not be replaced because another program holds it."""


@dataclass
class MetadataWriteResult:
    """Outcome of a single metadata write."""

    write_mode: str
    sidecar_path: Optional[Path] = None
    error_message: Optional[str] = None
    # True when the image file itself was rewritten (GPS/date/face fields).
    exif_written: bool = False

    @property
    def ok(self) -> bool:
        """True when the data is preserved somewhere (file, XMP or sidecar)."""
        return self.write_mode not in (WRITE_MODE_FAILED,)

    @property
    def embedded(self) -> bool:
        return self.write_mode in (
            WRITE_MODE_XMP, WRITE_MODE_EXIF_USER_COMMENT, WRITE_MODE_EXIF,
        )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def sidecar_path_for(image_path: str | Path) -> Path:
    """Return the sidecar JSON path for *image_path* (``name.facelocal.json``)."""
    p = Path(image_path)
    return p.with_name(p.name + ".facelocal.json")


def is_embeddable(image_path: str | Path) -> bool:
    """True when the file extension is one we attempt to embed metadata into."""
    return Path(image_path).suffix.lower() in _EMBEDDABLE_EXTS


def write_face_metadata(
    image_path: str | Path,
    payload: dict,
    *,
    prefer_sidecar_only: bool = False,
    prefer_exif_comment: bool = True,
    set_finder_comment: bool = True,
) -> MetadataWriteResult:
    """Persist a ``facelocal.faces.v1`` *payload* for *image_path*.

    Thin wrapper over :func:`apply_image_metadata` (see there for details).
    """
    return apply_image_metadata(
        image_path,
        faces_payload=payload,
        prefer_sidecar_only=prefer_sidecar_only,
        prefer_exif_comment=prefer_exif_comment,
        set_finder_comment=set_finder_comment,
    )


def apply_image_metadata(
    image_path: str | Path,
    *,
    gps: Optional[tuple[float, float]] = None,
    date: Optional[datetime] = None,
    faces_payload: Optional[dict] = None,
    prefer_sidecar_only: bool = False,
    prefer_exif_comment: bool = True,
    set_finder_comment: bool = True,
) -> MetadataWriteResult:
    """Apply GPS, date and face metadata to *image_path* in one file write.

    Args:
        gps: ``(lat, lon)`` to store in EXIF GPS, or None to leave it.
        date: Capture date for DateTime/DateTimeOriginal/DateTimeDigitized.
        faces_payload: A ``facelocal.faces.v1`` document, or None.
        prefer_sidecar_only: Never put the face payload into the image; write
            the sidecar instead (GPS/date edits are still applied).
        prefer_exif_comment: JPEG only — also keep a copy of the JSON in the
            EXIF ``UserComment`` (default). The XMP copy is always written.
        set_finder_comment: On macOS also set the Finder comment (best effort).

    Returns:
        The write mode: an embedded mode, ``sidecar`` (face data could not or
        should not go into the file), ``locked`` (another program holds the
        file; face data is in the sidecar), ``skipped`` (nothing to do) or
        ``failed``. ``error_message`` carries reasons and partial-write notes.
    """
    path = Path(image_path)
    if not path.exists():
        return MetadataWriteResult(WRITE_MODE_FAILED, error_message=f"file not found: {path}")

    has_faces = faces_payload is not None
    embed_faces = has_faces and not prefer_sidecar_only and is_embeddable(path)
    has_exif = gps is not None or date is not None

    sidecar_reason: Optional[str] = None
    if has_faces and not embed_faces and not prefer_sidecar_only:
        sidecar_reason = f"format {path.suffix or '?'} cannot hold embedded metadata"
        log.debug("Not embeddable (%s) → sidecar: %s", path.suffix, path)

    if not embed_faces and not has_exif:
        if has_faces:
            return _write_sidecar_result(path, faces_payload, reason=sidecar_reason)
        return MetadataWriteResult(WRITE_MODE_SKIPPED)

    if not os.access(path, os.W_OK):
        log.info("Image not writable: %s", path)
        if has_faces:
            return _write_sidecar_result(path, faces_payload, reason="image file is read-only")
        return MetadataWriteResult(WRITE_MODE_FAILED, error_message="image file is read-only")

    try:
        mode, notes = _write_embedded(
            path,
            gps=gps,
            date=date,
            payload=faces_payload if embed_faces else None,
            prefer_exif_comment=prefer_exif_comment,
        )
    except FileLockedError as exc:
        message = f"image is in use by another program: {exc}"
        log.warning("Metadata write blocked by a file lock: %s", path)
        if has_faces:
            sidecar = _write_sidecar_result(path, faces_payload, reason=message)
            if sidecar.write_mode == WRITE_MODE_FAILED:
                return MetadataWriteResult(
                    WRITE_MODE_FAILED, error_message=f"{message}; {sidecar.error_message}"
                )
            return MetadataWriteResult(
                WRITE_MODE_LOCKED, sidecar_path=sidecar.sidecar_path, error_message=message
            )
        return MetadataWriteResult(WRITE_MODE_LOCKED, error_message=message)
    except xe.XmpParseError as exc:
        # Never drop another tool's XMP: leave the file untouched.
        log.warning("Existing XMP unreadable in %s: %s", path, exc)
        if has_faces:
            sidecar = _write_sidecar_result(path, faces_payload)
            return MetadataWriteResult(
                WRITE_MODE_FAILED, sidecar_path=sidecar.sidecar_path, error_message=str(exc)
            )
        return MetadataWriteResult(WRITE_MODE_FAILED, error_message=str(exc))
    except Exception as exc:  # noqa: BLE001
        log.warning("Embedding metadata failed for %s: %s", path, exc, exc_info=True)
        if has_faces:
            return _write_sidecar_result(path, faces_payload, reason=f"embedding failed: {exc}")
        return MetadataWriteResult(WRITE_MODE_FAILED, error_message=str(exc))

    if has_faces and not embed_faces:
        # GPS/date went into the file; the face data goes next to it.
        result = _write_sidecar_result(path, faces_payload, reason=sidecar_reason)
        result.exif_written = True
        return result

    if embed_faces and set_finder_comment:
        write_macos_finder_comment(path, json.dumps(faces_payload, ensure_ascii=False))
    return MetadataWriteResult(mode, error_message="; ".join(notes) or None, exif_written=True)


def write_macos_finder_comment(image_path: str | Path, text: str) -> bool:
    """Set the macOS Finder/Spotlight comment for *image_path* to *text*.

    This is the comment shown in Finder → Get Info → "Comments" (the field most
    users look at), which is a separate channel from EXIF/XMP. Implemented via
    AppleScript so Finder's own store is updated too; falls back to writing the
    ``kMDItemFinderComment`` extended attribute directly.

    Best-effort: returns False (and logs at debug) on non-macOS systems or any
    failure. The first call may trigger a one-time "control Finder" automation
    prompt.
    """
    if sys.platform != "darwin":
        return False
    # Avoid triggering Finder automation prompts / subprocesses during tests.
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False

    import subprocess

    path_str = str(image_path)
    script = (
        "on run argv\n"
        "  set p to POSIX file (item 1 of argv) as alias\n"
        '  tell application "Finder" to set comment of p to (item 2 of argv)\n'
        "end run"
    )
    try:
        subprocess.run(
            ["osascript", "-e", script, path_str, text],
            check=True,
            capture_output=True,
            timeout=20,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug("Finder comment via AppleScript failed for %s: %s", path_str, exc)

    # Fallback: write the extended attribute directly (Spotlight-indexed).
    try:
        import plistlib

        data = plistlib.dumps(text, fmt=plistlib.FMT_BINARY)
        subprocess.run(
            ["xattr", "-w", "-x", "com.apple.metadata:kMDItemFinderComment",
             data.hex(), path_str],
            check=True,
            capture_output=True,
            timeout=20,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug("Finder comment via xattr failed for %s: %s", path_str, exc)
        return False



def read_face_metadata(image_path: str | Path) -> Optional[dict]:
    """Read back a previously written ``facelocal`` payload, or ``None``.

    Checks, in order: embedded XMP, embedded EXIF (UserComment, legacy
    ImageDescription), then the sidecar JSON file.
    """
    path = Path(image_path)

    if is_embeddable(path) and path.exists():
        try:
            embedded = _read_embedded(path)
            if embedded is not None:
                return embedded
        except Exception as exc:  # noqa: BLE001
            log.debug("Reading embedded metadata failed for %s: %s", path, exc)

    sidecar = sidecar_path_for(path)
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.debug("Reading sidecar failed for %s: %s", sidecar, exc)

    return None


# ---------------------------------------------------------------------------
# Sidecar
# ---------------------------------------------------------------------------

def _write_sidecar_result(
    path: Path, payload: dict, *, reason: Optional[str] = None
) -> MetadataWriteResult:
    try:
        sidecar = write_sidecar(path, payload)
        return MetadataWriteResult(WRITE_MODE_SIDECAR, sidecar_path=sidecar, error_message=reason)
    except Exception as exc:  # noqa: BLE001
        log.error("Sidecar write failed for %s: %s", path, exc)
        return MetadataWriteResult(WRITE_MODE_FAILED, error_message=str(exc))


def write_sidecar(image_path: str | Path, payload: dict) -> Path:
    """Atomically write *payload* to ``<image>.facelocal.json`` and return it."""
    sidecar = sidecar_path_for(image_path)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    _atomic_write_bytes(sidecar, text.encode("utf-8"))
    log.info("Face metadata sidecar written: %s", sidecar)
    return sidecar


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _known_names(payload: dict) -> list[str]:
    names: list[str] = []
    for face in payload.get("faces") or []:
        name = face.get("person_name")
        if face.get("unknown") or not name:
            continue
        name = str(name).strip()
        if name and name not in names:
            names.append(name)
    return names


def _is_facelocal_json(text: Optional[str]) -> bool:
    if not text:
        return False
    try:
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(parsed, dict) and parsed.get("schema") == FACELOCAL_SCHEMA


def _face_edits(
    payload: dict, exif_old: bytes, json_text: str, prefer_exif_comment: bool
) -> tuple[ef.ExifEdits, bool, list[str]]:
    """EXIF edits for a face payload → ``(edits, user_comment_written, notes)``."""
    notes: list[str] = []
    names = _known_names(payload)
    edits = ef.ExifEdits(
        xp_comment="; ".join(names) if names else None,
        xp_keywords=";".join(names) if names else None,
    )
    existing_uc = ef.read_user_comment(exif_old)
    uc_is_ours = _is_facelocal_json(existing_uc)
    uc_free = not (existing_uc or "").strip() or uc_is_ours

    if prefer_exif_comment:
        if uc_free:
            edits.user_comment = ef.UserCommentText(json_text)
            return edits, True, notes
        notes.append("EXIF UserComment holds other data — JSON stored in XMP only")
    elif uc_is_ours:
        edits.user_comment = None  # no stale copy next to the fresh XMP
    return edits, False, notes


def _fit_exif(
    exif_old: bytes, edits: ef.ExifEdits, payload: Optional[dict], uc_written: bool
) -> tuple[bytes, bool, list[str]]:
    """Build EXIF that fits one JPEG segment, stepping down when too large."""
    notes: list[str] = []

    def fits(blob: bytes) -> bool:
        return len(blob) <= js.MAX_SEGMENT_PAYLOAD

    out = ef.build_exif(exif_old, edits)
    if fits(out) or payload is None:
        if not fits(out):
            raise js.SegmentTooLargeError("EXIF block too large")
        return out, uc_written, notes

    if uc_written:
        edits.user_comment = ef.UserCommentText(
            json.dumps(payload, ensure_ascii=True), ascii=True
        )
        out = ef.build_exif(exif_old, edits)
        if fits(out):
            notes.append("UserComment stored ASCII-escaped (EXIF size limit)")
            return out, True, notes
        edits.user_comment = None if _is_facelocal_json(ef.read_user_comment(exif_old)) else ef.KEEP
        uc_written = False
        notes.append("UserComment copy omitted (EXIF size limit) — JSON in XMP only")
        out = ef.build_exif(exif_old, edits)
        if fits(out):
            return out, uc_written, notes

    edits.xp_comment = ef.KEEP
    edits.xp_keywords = ef.KEEP
    notes.append("Windows Comments/Tags omitted (EXIF size limit)")
    out = ef.build_exif(exif_old, edits)
    if not fits(out):
        raise js.SegmentTooLargeError("EXIF block too large")
    return out, uc_written, notes


def _write_embedded(
    path: Path,
    *,
    gps: Optional[tuple[float, float]],
    date: Optional[datetime],
    payload: Optional[dict],
    prefer_exif_comment: bool,
) -> tuple[str, list[str]]:
    data = path.read_bytes()
    if data.startswith(b"\xff\xd8"):
        return _write_jpeg(path, data, gps, date, payload, prefer_exif_comment)
    return _write_with_pillow(path, data, gps, date, payload)


def _image_size(data: bytes) -> tuple[int, int]:
    from PIL import Image as PilImage

    with PilImage.open(io.BytesIO(data)) as img:  # header only, no decode
        return img.size


def _write_jpeg(
    path: Path,
    data: bytes,
    gps: Optional[tuple[float, float]],
    date: Optional[datetime],
    payload: Optional[dict],
    prefer_exif_comment: bool,
) -> tuple[str, list[str]]:
    exif_old, xmp_old = js.read_metadata(data)
    exif_old = exif_old or b""
    notes: list[str] = []
    edits = ef.ExifEdits()
    uc_written = False
    new_xmp = js.KEEP

    if payload is not None:
        json_text = json.dumps(payload, ensure_ascii=False)
        edits, uc_written, notes = _face_edits(payload, exif_old, json_text, prefer_exif_comment)
        ow, oh = xe.oriented_size(*_image_size(data), ef.read_orientation(exif_old))
        new_xmp = xe.edit_xmp(
            xmp_old,
            json_text=json_text,
            regions=xe.regions_from_payload(payload, ow, oh),
            applied_to=(ow, oh),
        )
        if len(js.XMP_HEADER) + len(new_xmp) > js.MAX_SEGMENT_PAYLOAD:
            raise js.SegmentTooLargeError("XMP too large for one JPEG segment")

    edits.gps = gps
    edits.date = date
    new_exif = js.KEEP
    if not edits.is_empty():
        new_exif, uc_written, more = _fit_exif(exif_old, edits, payload, uc_written)
        notes.extend(more)

    _replace_with_bytes(path, js.splice(data, exif=new_exif, xmp=new_xmp))

    if payload is None:
        return WRITE_MODE_EXIF, notes
    return (WRITE_MODE_EXIF_USER_COMMENT if uc_written else WRITE_MODE_XMP), notes


def _write_with_pillow(
    path: Path,
    data: bytes,
    gps: Optional[tuple[float, float]],
    date: Optional[datetime],
    payload: Optional[dict],
) -> tuple[str, list[str]]:
    """PNG (and, for GPS/date only, other Pillow formats): lossless re-save."""
    from PIL import Image as PilImage

    with PilImage.open(io.BytesIO(data)) as img:
        fmt = (img.format or "").upper()
        info = dict(img.info)
        png_text = dict(getattr(img, "text", {}) or {})
        img.load()
        work = img.copy()

    exif_old = info.get("exif", b"") or b""
    if isinstance(exif_old, str):
        exif_old = exif_old.encode("latin-1", errors="replace")
    if exif_old and not exif_old.startswith(ef.EXIF_HEADER):
        exif_old = ef.EXIF_HEADER + exif_old

    edits = ef.ExifEdits(gps=gps, date=date)
    new_exif = ef.build_exif(exif_old, edits) if not edits.is_empty() else exif_old

    if fmt == "PNG":
        new_xmp: Optional[str] = None
        if payload is not None:
            ow, oh = xe.oriented_size(*work.size, ef.read_orientation(exif_old))
            existing_xmp = png_text.get(_PNG_XMP_KEY) or info.get("xmp")
            new_xmp = xe.edit_xmp(
                existing_xmp,
                json_text=json.dumps(payload, ensure_ascii=False),
                regions=xe.regions_from_payload(payload, ow, oh),
                applied_to=(ow, oh),
            ).decode("utf-8")
        _save_png(path, work, png_text, xmp=new_xmp, exif=new_exif,
                  icc=info.get("icc_profile"))
        return (WRITE_MODE_XMP if payload is not None else WRITE_MODE_EXIF), []

    if payload is not None:
        raise ValueError(f"cannot embed face metadata into {fmt or 'this format'}")
    save_kwargs: dict = {"format": fmt or None, "exif": new_exif}
    if info.get("icc_profile"):
        save_kwargs["icc_profile"] = info["icc_profile"]
    _replace_with_pillow(path, work, save_kwargs)
    return WRITE_MODE_EXIF, []


def _save_png(path: Path, img, existing_text: dict, *, xmp: Optional[str],
              exif: bytes, icc) -> None:
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    for key, value in existing_text.items():
        if key == _PNG_XMP_KEY and xmp is not None:
            continue
        try:
            info.add_itxt(key, value)
        except Exception:  # noqa: BLE001
            try:
                info.add_text(key, value)
            except Exception:  # noqa: BLE001
                log.debug("Skipping unpreservable PNG text chunk %r", key)
    if xmp is not None:
        info.add_itxt(_PNG_XMP_KEY, xmp)

    save_kwargs: dict = {"format": "PNG", "pnginfo": info}
    if exif:
        save_kwargs["exif"] = exif
    if icc:
        save_kwargs["icc_profile"] = icc
    _replace_with_pillow(path, img, save_kwargs)


# ---------------------------------------------------------------------------
# Reading embedded payloads back
# ---------------------------------------------------------------------------

def _loads(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001
        return None
    return parsed if isinstance(parsed, dict) else None


def _read_embedded(path: Path) -> Optional[dict]:
    data = path.read_bytes()
    if data.startswith(b"\xff\xd8"):
        exif, xmp = js.read_metadata(data)
    else:
        from PIL import Image as PilImage

        with PilImage.open(io.BytesIO(data)) as img:
            exif = img.info.get("exif", b"")
            xmp = (getattr(img, "text", {}) or {}).get(_PNG_XMP_KEY) or img.info.get("xmp")

    payload = _loads(xe.read_facelocal_json(xmp)) if xmp else None
    if payload is not None:
        return payload
    if exif:
        payload = _loads(ef.read_user_comment(exif))
        if payload is not None:
            return payload
        try:  # legacy ImageDescription copy
            from PIL import Image as PilImage

            parsed = PilImage.Exif()
            parsed.load(exif)
            return _loads(parsed.get(ef.TAG_IMAGE_DESCRIPTION))
        except Exception:  # noqa: BLE001
            return None
    return None


# ---------------------------------------------------------------------------
# Safe replace
# ---------------------------------------------------------------------------

def _tmp_for(path: Path) -> Path:
    return path.with_name(path.name + ".facelocal.tmp")


def _cleanup(tmp: Path) -> None:
    try:
        tmp.unlink()
    except OSError:
        pass


def _replace_with_bytes(path: Path, data: bytes) -> None:
    tmp = _tmp_for(path)
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        _cleanup(tmp)
        raise
    _replace_atomic(tmp, path)


def _replace_with_pillow(path: Path, img, save_kwargs: dict) -> None:
    tmp = _tmp_for(path)
    try:
        with open(tmp, "wb") as fh:
            img.save(fh, **save_kwargs)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        _cleanup(tmp)
        raise
    _replace_atomic(tmp, path)


def _is_lock_error(exc: OSError) -> bool:
    return isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in _LOCK_WINERRORS


def _replace_atomic(tmp: Path, path: Path) -> None:
    """Atomically replace *path* with *tmp*, tolerating short Windows locks.

    ``os.replace`` over a file that another process holds open fails on
    Windows (Search indexer, antivirus, Explorer thumbnails, a viewer). Short
    locks are ridden out with a backoff; a lock that persists raises
    :class:`FileLockedError`. The original is never overwritten in place, so
    a failure always leaves it intact, and the temp file is always removed.
    """
    import time

    last_exc: Optional[OSError] = None
    for attempt in range(len(_REPLACE_BACKOFF) + 1):
        try:
            os.replace(tmp, path)
            return
        except OSError as exc:
            if not _is_lock_error(exc):
                _cleanup(tmp)
                raise
            last_exc = exc
            if attempt < len(_REPLACE_BACKOFF):
                time.sleep(_REPLACE_BACKOFF[attempt])

    _cleanup(tmp)
    raise FileLockedError(str(last_exc)) from last_exc


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)
