"""Smoke test: the scan & maintenance dialog builds and wires its signals.

ScanModesDialog has two tabs:
  * "AI recognition" — three primary workflows, emitted via scan_workflow_started.
  * "Maintenance (developer)" — cleanup/repair tools that also run automatically,
    emitted via maintenance_action_started.

This guards against missing i18n keys or broken card wiring after the restructure.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QPushButton, QTabWidget

from app.app_settings import app_qsettings
from app.services.duplicate_unknown_face_finder import (
    DEFAULT_OVERLAP_SENSITIVITY,
    OVERLAP_SENSITIVITIES,
)
from app.ui.dialogs.scan_modes_dialog import ScanModesDialog


@pytest.fixture
def dialog(qtbot):
    dlg = ScanModesDialog(config=None)
    qtbot.addWidget(dlg)
    return dlg


def test_dialog_has_two_tabs(dialog):
    tabs = dialog.findChild(QTabWidget)
    assert tabs is not None
    assert tabs.count() == 2


def test_dialog_builds_with_all_cards(dialog):
    # 3 AI workflow cards + 6 maintenance cards + close button.
    buttons = dialog.findChildren(QPushButton)
    assert len(buttons) >= 10


def test_workflow_signal_emitted(dialog, qtbot):
    with qtbot.waitSignal(dialog.scan_workflow_started, timeout=1000) as blocker:
        dialog._launch_workflow("face_detection")
    assert blocker.args == ["face_detection"]


def test_maintenance_signal_emitted(dialog, qtbot):
    with qtbot.waitSignal(dialog.maintenance_action_started, timeout=1000) as blocker:
        dialog._launch_maintenance("overlap_cleanup")
    assert blocker.args == ["overlap_cleanup"]


def test_overlap_card_offers_every_sensitivity(dialog):
    combo = dialog._overlap_sensitivity_combo
    keys = [combo.itemData(i) for i in range(combo.count())]
    assert keys == [p.key for p in OVERLAP_SENSITIVITIES]
    assert combo.currentData() in keys


def test_overlap_sensitivity_choice_is_persisted(dialog):
    combo = dialog._overlap_sensitivity_combo
    combo.setCurrentIndex(combo.findData("any"))
    assert app_qsettings().value("overlap_cleanup/sensitivity") == "any"
    combo.setCurrentIndex(combo.findData(DEFAULT_OVERLAP_SENSITIVITY))
