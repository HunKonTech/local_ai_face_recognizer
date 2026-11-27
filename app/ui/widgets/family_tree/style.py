"""Colours and item-data roles shared by the family-tree view modules."""

from __future__ import annotations

from PySide6.QtGui import QColor

# Colours (kept close to the reference chart).
_MALE_FILL = QColor("#bcdcf2")
_MALE_BORDER = QColor("#5b9bd5")
_FEMALE_FILL = QColor("#f6d08a")
_FEMALE_BORDER = QColor("#d99a2b")
_UNKNOWN_FILL = QColor("#dcdcdc")
_UNKNOWN_BORDER = QColor("#9a9a9a")
_SELECTED_BORDER = QColor("#e03030")    # the person clicked right now
_FOCUS_BORDER = QColor("#8e44ad")       # the focus/root person (purple)
_ANCESTOR_BORDER = QColor("#1f6fbf")    # selected person's ancestors
_DESCENDANT_BORDER = QColor("#2e9e4f")  # selected person's descendants
_SPOUSE_BORDER = QColor("#c0392b")      # selected person's spouses
_LINE = QColor("#7a7a7a")
_BAND_A = QColor("#ffffff")
_BAND_B = QColor("#faf0dd")
_TEXT = QColor("#222222")
_GEN_NUMBER = QColor("#c8a050")

_ROLE_PID = 0
