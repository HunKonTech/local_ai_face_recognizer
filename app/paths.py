"""Path helpers for local development and packaged builds."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "Face-Local"
APP_SLUG = "face-local"
APP_ICON_RELATIVE_PATH = "assets/icons/app-icon.png"


def is_frozen() -> bool:
    """Return True when running from a bundled executable."""
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    """Return the repository root during local development."""
    return Path(__file__).resolve().parent.parent


def bundle_root() -> Path:
    """Return the directory that contains bundled read-only resources."""
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return project_root()


def resource_path(relative_path: str) -> Path:
    """Resolve *relative_path* inside the bundled resource directory."""
    return bundle_root() / relative_path


def app_icon_path() -> Path:
    """Return the bundled application icon path."""
    return resource_path(APP_ICON_RELATIVE_PATH)


def user_config_dir() -> Path:
    """Return the per-user configuration directory."""
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / APP_NAME
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", home / "AppData" / "Roaming")) / APP_NAME
    return Path(os.environ.get("XDG_CONFIG_HOME", home / ".config")) / APP_SLUG


def user_data_dir() -> Path:
    """Return the per-user writable data directory."""
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / APP_NAME
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local")) / APP_NAME
    return Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share")) / APP_SLUG


def default_log_file() -> Path:
    """Return the default log file path for the current runtime."""
    if is_frozen():
        return user_data_dir() / "logs" / "face_local.log"
    return Path("data/face_local.log")
