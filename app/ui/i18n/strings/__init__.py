"""Merge all string parts into one table. Parts are applied in order (later wins)."""

from __future__ import annotations

from typing import Dict

from . import (
    s01_toolbar,
    s02_family_search,
    s03_ai_model_tab_settings,
    s04_person_details,
    s05_rerecognition_review_dialog,
    s06_face_quality_filter_settings,
    s07_already_reviewed_history_lis,
    s08_simplified_scan_dialog_workf,
    s09_keyboard_shortcuts_settings,
)

_PARTS = (
    s01_toolbar,
    s02_family_search,
    s03_ai_model_tab_settings,
    s04_person_details,
    s05_rerecognition_review_dialog,
    s06_face_quality_filter_settings,
    s07_already_reviewed_history_lis,
    s08_simplified_scan_dialog_workf,
    s09_keyboard_shortcuts_settings,
)


def build_strings() -> Dict[str, Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}
    for part in _PARTS:
        merged.update(part.STRINGS)
    return merged
