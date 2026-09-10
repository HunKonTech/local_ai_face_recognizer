"""Tests for the unknown-person marking in the sidebar's "Összes arc" grid."""

from __future__ import annotations

import pytest
from PySide6.QtGui import QColor, QPixmap

from app.ui.panels import sidebar_panel
from app.ui.panels.sidebar_panel import (
    FaceData,
    SidebarPanel,
    SidebarPerson,
    _draw_unknown_badge,
)


def _person(pid: int, name: str, *, auto_named: bool = False,
            protected: bool = False) -> SidebarPerson:
    return SidebarPerson(
        id=pid,
        name=name,
        is_protected=protected,
        face_count=3,
        face=FaceData(
            face_id=pid * 10,
            person_id=pid,
            crop_path=f"C:/nowhere/{pid}.jpg",
            image_path=f"C:/nowhere/src{pid}.jpg",
            bbox=(1, 2, 3, 4),
        ),
        is_auto_named=auto_named,
    )


@pytest.fixture
def panel(qtbot, monkeypatch, tmp_path):
    # Keep the toggle out of the user's real settings file.
    from PySide6.QtCore import QSettings

    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(
        sidebar_panel,
        "app_qsettings",
        lambda: QSettings(ini, QSettings.Format.IniFormat),
    )
    p = SidebarPanel()
    qtbot.addWidget(p)
    p.resize(240, 400)
    return p


def test_badge_is_drawn_on_a_copy(qapp):  # noqa: ARG001 — QPixmap needs a QApplication
    source = QPixmap(64, 64)
    source.fill(QColor("#123456"))
    badged = _draw_unknown_badge(source)

    assert badged is not source
    assert badged.size() == source.size()
    # The source pixmap keeps its original top-right corner.
    assert source.toImage().pixelColor(56, 6) == QColor("#123456")
    # The copy has the amber badge painted there.
    assert badged.toImage().pixelColor(56, 6) == QColor(
        sidebar_panel._UNKNOWN_BADGE_COLOR
    )


def test_unknown_flag_follows_auto_named(panel):
    panel.populate([_person(1, "Unknown 7", auto_named=True), _person(2, "Bori")])

    assert panel._thumbs[1]._is_unknown is True
    assert panel._thumbs[2]._is_unknown is False
    assert panel._thumbs[1].toolTip()
    assert not panel._thumbs[2].toolTip()


def test_naming_a_person_rebuilds_the_thumb(panel):
    panel.populate([_person(1, "Unknown 7", auto_named=True)])
    before = panel._thumbs[1]

    # Same face, same crop — only the auto-named flag and name change.
    panel.populate([_person(1, "Dani")])
    after = panel._thumbs[1]

    assert after is not before
    assert after._is_unknown is False


def test_only_unknown_filter(panel):
    persons = [
        _person(1, "Unknown 7", auto_named=True),
        _person(2, "Bori"),
        _person(3, "Ismeretlen", protected=True),
    ]
    panel.populate(persons)
    assert set(panel._thumbs) == {1, 2}

    panel._only_unknown_check.setChecked(True)
    assert set(panel._thumbs) == {1}

    panel._only_unknown_check.setChecked(False)
    assert set(panel._thumbs) == {1, 2}


def test_filter_state_is_persisted(panel):
    panel._only_unknown_check.setChecked(True)
    assert panel._load_only_unknown_pref() is True
