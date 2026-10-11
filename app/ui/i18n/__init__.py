"""Minimal translation module — English / Hungarian."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

from .strings import build_strings

log = logging.getLogger(__name__)

SUPPORTED: Dict[str, str] = {"en": "English", "hu": "Magyar"}
_LEGACY_PREFS_FILE = Path.home() / ".face_local_prefs.json"
_lang: str = "en"


def _prefs_file() -> Path:
    """Return the active language preferences file path, creating its directory."""
    try:
        from app.paths import ensure_settings_dir
        return ensure_settings_dir() / "language_prefs.json"
    except Exception:  # noqa: BLE001
        return _LEGACY_PREFS_FILE

_STRINGS: Dict[str, Dict[str, str]] = build_strings()


def t(key: str, **kwargs: object) -> str:
    """Return the translated string for *key* in the current language."""
    entry = _STRINGS.get(key)
    if entry is None:
        log.warning("i18n: unknown key %r", key)
        return key
    text = entry.get(_lang) or entry.get("en") or key
    return text.format(**kwargs) if kwargs else text


def current_language() -> str:
    return _lang


def set_language(lang: str) -> None:
    global _lang
    if lang not in SUPPORTED:
        log.warning("i18n: unsupported language %r — falling back to 'en'", lang)
        lang = "en"
    _lang = lang
    _save_prefs()


def load_prefs() -> None:
    global _lang
    try:
        target = _prefs_file()
        # One-time migration: move legacy ~/.face_local_prefs.json to new location.
        if not target.exists() and _LEGACY_PREFS_FILE.exists():
            try:
                import shutil
                shutil.copy2(_LEGACY_PREFS_FILE, target)
                log.info("i18n: migrated language prefs from %s to %s", _LEGACY_PREFS_FILE, target)
            except Exception as mig_exc:  # noqa: BLE001
                log.warning("i18n: could not migrate prefs: %s", mig_exc)
        if target.exists():
            data = json.loads(target.read_text(encoding="utf-8"))
            lang = data.get("language", "en")
            if lang in SUPPORTED:
                _lang = lang
    except Exception as exc:  # noqa: BLE001
        log.warning("i18n: could not load prefs: %s", exc)


def _save_prefs() -> None:
    try:
        target = _prefs_file()
        data: dict = {}
        if target.exists():
            data = json.loads(target.read_text(encoding="utf-8"))
        data["language"] = _lang
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log.warning("i18n: could not save prefs: %s", exc)
