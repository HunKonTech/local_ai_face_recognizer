"""Monitor enumeration for the screen recorder.

Bridges the windowing system to the Qt-free
:class:`~app.services.screen_recorder_service.RecordingDisplayInfo` used by the
recorder's pure capture-resolution logic.

On Windows the Win32 view is the source of truth: Qt reports monitors in
logical (DPI-scaled) pixels and names them by their friendly model string, while
ffmpeg's grabbers need physical pixels and DXGI identifies outputs by the GDI
device name.  Everywhere else the Qt enumeration is used unchanged.

Kept tiny and defensive: a flaky windowing layer must never crash a recording
start or the settings dialog.
"""

from __future__ import annotations

import logging
import sys
from typing import Dict, List, Optional

from app.services.screen_recorder_service import RecordingDisplayInfo

log = logging.getLogger(__name__)


def _qt_screens() -> List[RecordingDisplayInfo]:
    """Enumerate monitors through Qt, in logical pixels."""
    try:
        from PySide6.QtGui import QGuiApplication

        screens = QGuiApplication.screens()
        primary = QGuiApplication.primaryScreen()
    except Exception:  # noqa: BLE001 — no display / headless / Qt issue
        log.debug("display enumeration failed", exc_info=True)
        return []

    displays: List[RecordingDisplayInfo] = []
    for idx, screen in enumerate(screens):
        try:
            geo = screen.geometry()
            name = screen.name() or ""
            display_id = name or f"display-{idx}"
            displays.append(
                RecordingDisplayInfo(
                    id=display_id,
                    name=name,
                    width=int(geo.width()),
                    height=int(geo.height()),
                    is_primary=(screen is primary),
                    x=int(geo.x()),
                    y=int(geo.y()),
                    av_index=None,
                )
            )
        except Exception:  # noqa: BLE001 — skip a misbehaving screen
            log.debug("skipping unreadable screen %d", idx, exc_info=True)
    return displays


def _friendly_name(monitor, qt_displays: List[RecordingDisplayInfo]) -> str:
    """Best-effort human label for a Win32 monitor, from the Qt screen list.

    Purely cosmetic — the physical rectangle and the DXGI index come from Win32.
    Qt's logical origin scales to the physical one by the monitor's own device
    pixel ratio, so the origins line up whenever the layout is unambiguous.
    """
    for disp in qt_displays:
        if (disp.x, disp.y) == (monitor.x, monitor.y):
            return disp.name
        for ratio in (1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0):
            if (
                abs(disp.x * ratio - monitor.x) <= 2
                and abs(disp.y * ratio - monitor.y) <= 2
            ):
                return disp.name
    return ""


def enumerate_displays(
    dxgi_overrides: Optional[Dict[str, str]] = None,
) -> List[RecordingDisplayInfo]:
    """Return the connected monitors, or ``[]`` if enumeration fails.

    ``av_index`` is left ``None`` here — on macOS the avfoundation
    "Capture screen N" device index is **not** the monitor ordinal (capture
    devices sit after the cameras in the device list), so the caller populates
    it from :func:`~app.services.screen_recorder_service.probe_screen_indices`.

    On Windows the physical geometry and the DXGI adapter/output indices are
    filled in as well; ``dxgi_overrides`` maps a GDI device name to an
    ``"adapter:output"`` string for the rare machine where the automatic join
    picks the wrong output.
    """
    qt_displays = _qt_screens()
    if not sys.platform.startswith("win"):
        return qt_displays

    try:
        from app.services.windows_display_probe import probe_monitors

        monitors = probe_monitors(dxgi_overrides)
    except Exception:  # noqa: BLE001
        log.debug("windows display probe failed", exc_info=True)
        monitors = []
    if not monitors:
        log.info("falling back to Qt display geometry (no Win32 monitor data)")
        return qt_displays

    displays: List[RecordingDisplayInfo] = []
    for monitor in monitors:
        displays.append(
            RecordingDisplayInfo(
                id=monitor.device_name,
                name=_friendly_name(monitor, qt_displays),
                width=monitor.width,
                height=monitor.height,
                is_primary=monitor.is_primary,
                x=monitor.x,
                y=monitor.y,
                physical_x=monitor.x,
                physical_y=monitor.y,
                physical_width=monitor.width,
                physical_height=monitor.height,
                dxgi_adapter_index=monitor.dxgi_adapter_index,
                dxgi_output_index=monitor.dxgi_output_index,
            )
        )
    return displays


def active_window_bounds(widget) -> Optional[tuple]:
    """Return ``(x, y, w, h)`` of *widget*'s top-level window in screen coords.

    Windows returns physical pixels (what the grabbers address) via the DWM
    frame bounds; everywhere else the Qt logical geometry is used.  ``None``
    when the geometry cannot be determined.
    """
    try:
        win = widget.window() if widget is not None else None
        if win is None:
            return None
        if sys.platform.startswith("win"):
            from app.services.windows_display_probe import physical_window_rect

            rect = physical_window_rect(int(win.winId()))
            if rect is not None:
                return rect
        top_left = win.mapToGlobal(win.rect().topLeft())
        return (
            int(top_left.x()),
            int(top_left.y()),
            int(win.width()),
            int(win.height()),
        )
    except Exception:  # noqa: BLE001
        log.debug("active window bounds lookup failed", exc_info=True)
        return None
