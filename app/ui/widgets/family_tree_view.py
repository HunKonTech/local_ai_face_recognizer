"""Compatibility re-export.

The family-tree view now lives in the :mod:`app.ui.widgets.family_tree`
package (mixins split by topic). This module keeps the old import path and
the private colour names that tests import.
"""

from app.ui.widgets.family_tree import FamilyTreeView
from app.ui.widgets.family_tree.geometry import _bus_tracks, _elide, _marriage_label, _years
from app.ui.widgets.family_tree.style import (
    _ANCESTOR_BORDER,
    _BAND_A,
    _BAND_B,
    _DESCENDANT_BORDER,
    _FEMALE_BORDER,
    _FEMALE_FILL,
    _FOCUS_BORDER,
    _GEN_NUMBER,
    _LINE,
    _MALE_BORDER,
    _MALE_FILL,
    _ROLE_PID,
    _SELECTED_BORDER,
    _SPOUSE_BORDER,
    _TEXT,
    _UNKNOWN_BORDER,
    _UNKNOWN_FILL,
)

__all__ = [
    "FamilyTreeView",
    "_ANCESTOR_BORDER", "_BAND_A", "_BAND_B", "_DESCENDANT_BORDER",
    "_FEMALE_BORDER", "_FEMALE_FILL", "_FOCUS_BORDER", "_GEN_NUMBER", "_LINE",
    "_MALE_BORDER", "_MALE_FILL", "_ROLE_PID", "_SELECTED_BORDER",
    "_SPOUSE_BORDER", "_TEXT", "_UNKNOWN_BORDER", "_UNKNOWN_FILL",
    "_bus_tracks", "_elide", "_marriage_label", "_years",
]
