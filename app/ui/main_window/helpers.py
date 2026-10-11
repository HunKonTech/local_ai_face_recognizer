"""Remembered-directory helpers for file dialogs."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

def _last_dir(key: str, default: str = "") -> str:
    """Read a remembered directory path from persistent settings."""
    from app.app_settings import app_qsettings

    return app_qsettings().value(key, default, type=str)


def _save_dir(key: str, path: str) -> None:
    """Persist a directory path so the next file dialog reopens there."""
    from app.app_settings import app_qsettings

    app_qsettings().setValue(key, path)
