"""Tests for display enumeration and active-window bounds."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication, QWidget

from app.services.screen_recorder_service import RecordingDisplayInfo
from app.ui.display_utils import active_window_bounds, enumerate_displays


class TestEnumerateDisplays:
    def test_returns_recording_display_info_list(self, qapp):
        displays = enumerate_displays()
        assert isinstance(displays, list)
        if displays:
            d = displays[0]
            assert isinstance(d, RecordingDisplayInfo)
            assert d.width > 0 and d.height > 0
            assert d.av_index is None  # filled later by probe_screen_indices

    def test_survives_broken_qgui(self, monkeypatch):
        """A broken Qt must never raise; off-Windows it also means "no data"."""
        def _boom():
            raise RuntimeError("no display")

        monkeypatch.setattr(
            "PySide6.QtGui.QGuiApplication.screens", staticmethod(_boom)
        )
        monkeypatch.setattr(
            "app.services.windows_display_probe.enumerate_physical_monitors",
            lambda: [],
        )
        assert enumerate_displays() == []

    def test_windows_uses_the_native_probe(self, monkeypatch):
        """On Windows the Win32 rectangles and DXGI indices win over Qt's."""
        from app.services import windows_display_probe as probe

        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr(
            probe, "enumerate_physical_monitors",
            lambda: [
                probe.PhysicalMonitor(
                    device_name=r"\.\DISPLAY1", x=0, y=0,
                    width=3840, height=2160, is_primary=True,
                )
            ],
        )
        monkeypatch.setattr(
            probe, "enumerate_dxgi_outputs",
            lambda: [
                probe.DxgiOutput(
                    adapter_index=0, output_index=2, device_name=r"\.\DISPLAY1",
                    x=0, y=0, width=3840, height=2160,
                )
            ],
        )
        displays = enumerate_displays()
        assert len(displays) == 1
        d = displays[0]
        assert d.id == r"\.\DISPLAY1"
        assert (d.physical_width, d.physical_height) == (3840, 2160)
        assert (d.dxgi_adapter_index, d.dxgi_output_index) == (0, 2)

    def test_windows_falls_back_to_qt_when_the_probe_is_empty(self, qapp, monkeypatch):
        from app.services import windows_display_probe as probe

        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr(probe, "enumerate_physical_monitors", lambda: [])
        displays = enumerate_displays()
        # Whatever Qt reported is returned unchanged — no physical data.
        assert all(d.physical_width is None for d in displays)


class TestActiveWindowBounds:
    def test_returns_geometry_for_top_level_window(self, qapp, qtbot):
        win = QWidget()
        win.setGeometry(QRect(50, 40, 300, 200))
        win.show()
        qtbot.addWidget(win)
        qtbot.waitExposed(win)

        bounds = active_window_bounds(win)
        assert bounds is not None
        x, y, w, h = bounds
        assert w == 300
        assert h == 200
        assert x >= 0 and y >= 0

    def test_none_for_missing_widget(self):
        assert active_window_bounds(None) is None

    def test_none_on_failure(self, monkeypatch):
        class _BrokenWidget:
            def window(self):
                raise RuntimeError("no window")

        assert active_window_bounds(_BrokenWidget()) is None
