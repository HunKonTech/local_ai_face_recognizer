"""Windows-only integration checks for embedded face metadata.

* A file held open by another program with share-read only (Photos app,
  many viewers, antivirus) → ``locked`` result, sidecar written, image intact.
* Windows Explorer shows the person names as Comments / Tags / People.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from PIL import Image as PilImage

from app.utils import image_metadata as meta

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


def _payload(*names: str) -> dict:
    return {
        "schema": meta.FACELOCAL_SCHEMA,
        "coordinate_space": "exif_oriented",
        "faces": [
            {"person_id": i + 1, "person_name": n,
             "box": {"x": 10 + 60 * i, "y": 20, "width": 40, "height": 50}}
            for i, n in enumerate(names)
        ],
    }


def _open_share_read(path):
    """Open *path* like .NET FileShare.Read does: others may read, not write/delete."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING = 0x80000000, 0x1, 3
    handle = kernel32.CreateFileW(str(path), GENERIC_READ, FILE_SHARE_READ, None,
                                  OPEN_EXISTING, 0, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        raise OSError(ctypes.get_last_error(), "CreateFileW failed")
    return kernel32, handle


def test_share_read_lock_reports_locked(tmp_path, monkeypatch):
    monkeypatch.setattr(meta, "_REPLACE_BACKOFF", (0.01, 0.01))
    jpg = tmp_path / "held.jpg"
    PilImage.new("RGB", (200, 120), (1, 2, 3)).save(jpg, format="JPEG")
    before = jpg.read_bytes()

    kernel32, handle = _open_share_read(jpg)
    try:
        result = meta.write_face_metadata(jpg, _payload("Szilvay Géza"))
    finally:
        kernel32.CloseHandle(handle)

    assert result.write_mode == meta.WRITE_MODE_LOCKED
    assert result.sidecar_path is not None and result.sidecar_path.exists()
    assert jpg.read_bytes() == before
    assert not (tmp_path / "held.jpg.facelocal.tmp").exists()

    # Lock released → embeds.
    again = meta.write_face_metadata(jpg, _payload("Szilvay Géza"))
    assert again.write_mode == meta.WRITE_MODE_EXIF_USER_COMMENT


def _shell_properties(path) -> dict[str, str]:
    script = (
        "$ErrorActionPreference='Stop';"
        "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
        "$sh=New-Object -ComObject Shell.Application;"
        f"$f=$sh.Namespace('{path.parent}').ParseName('{path.name}');"
        "foreach($k in 'System.Comment','System.Keywords','System.Photo.PeopleNames')"
        "{ '{0}={1}' -f $k, ($f.ExtendedProperty($k) -join '|') }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, timeout=60, check=True,
        ).stdout.decode("utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"Shell property system unavailable: {exc}")
    props: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            props[key.strip()] = value.strip()
    return props


def test_explorer_shows_names(tmp_path):
    jpg = tmp_path / "people.jpg"
    PilImage.new("RGB", (240, 160), (90, 90, 90)).save(jpg, format="JPEG")

    result = meta.write_face_metadata(jpg, _payload("Szilvay Géza", "Kiss Anna"))
    assert result.write_mode == meta.WRITE_MODE_EXIF_USER_COMMENT

    props = _shell_properties(jpg)
    assert "Szilvay Géza" in props.get("System.Comment", "")
    assert "Kiss Anna" in props.get("System.Comment", "")
    assert set(props.get("System.Keywords", "").split("|")) >= {"Szilvay Géza", "Kiss Anna"}
    assert set(props.get("System.Photo.PeopleNames", "").split("|")) >= {"Szilvay Géza", "Kiss Anna"}
