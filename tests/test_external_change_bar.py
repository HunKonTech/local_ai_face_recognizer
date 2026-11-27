"""Offscreen tests for the external-change notice bar."""

from __future__ import annotations

import app.ui.i18n as i18n
from app.ui.widgets.external_change_bar import ExternalChangeBar, summarize_changes


def test_summary_text_en_and_hu(monkeypatch):
    monkeypatch.setattr(i18n, "_lang", "en")
    assert summarize_changes({"person": 3, "relationship": 1}) == "3 persons and 1 relationship"
    assert summarize_changes({"relationship": 1, "person": 1}) == "1 person and 1 relationship"
    assert summarize_changes({"person": 2, "field": 1, "relationship": 2}) == (
        "2 persons, 2 relationships and 1 custom field")
    monkeypatch.setattr(i18n, "_lang", "hu")
    assert summarize_changes({"person": 3, "relationship": 1}) == "3 személy és 1 kapcsolat"


def test_bar_shows_and_refresh_emits(qtbot, monkeypatch):
    monkeypatch.setattr(i18n, "_lang", "en")
    bar = ExternalChangeBar()
    qtbot.addWidget(bar)
    assert bar.isHidden()
    bar.show_changes({"person": 3, "relationship": 1})
    assert not bar.isHidden()
    assert "3 persons and 1 relationship" in bar.text()
    with qtbot.waitSignal(bar.refresh_requested, timeout=1000):
        bar._refresh_btn.click()
    assert bar.isHidden()
