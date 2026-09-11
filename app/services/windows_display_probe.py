"""Windows-native monitor geometry, DXGI output indices and window rectangles.

Qt reports monitors in *logical* (DPI-scaled) pixels and names them by their
friendly model string, while ffmpeg's grabbers address the desktop in physical
pixels and DXGI identifies outputs by the GDI device name (``\\\\.\\DISPLAY1``).
Recording therefore needs the Win32 view of the desktop, not the Qt one — see
``app/services/screen_recorder_service.py`` for how the two are joined.

Everything here is defensive: a missing D3D runtime, a locked session or a
ctypes surprise must degrade to "unknown", never take a recording start down.
Only :func:`merge_monitors_with_dxgi` and :func:`virtual_desktop_rect` contain
real logic, and both are pure so they can be unit tested off-Windows.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class PhysicalMonitor:
    """A monitor as Windows sees it, in physical virtual-desktop pixels."""

    device_name: str            # ``\\.\DISPLAY1`` — the join key with DXGI
    x: int
    y: int
    width: int
    height: int
    is_primary: bool = False
    dxgi_adapter_index: Optional[int] = None
    dxgi_output_index: Optional[int] = None


@dataclass
class DxgiOutput:
    """One DXGI output, i.e. one ``ddagrab`` capture target."""

    adapter_index: int
    output_index: int
    device_name: str
    x: int
    y: int
    width: int
    height: int
    attached: bool = True


def is_windows() -> bool:
    return sys.platform.startswith("win")


# ---------------------------------------------------------------------------
# Pure helpers (unit tested)
# ---------------------------------------------------------------------------

def virtual_desktop_rect(
    monitors: List[PhysicalMonitor],
) -> Optional[Tuple[int, int, int, int]]:
    """Return ``(x, y, width, height)`` spanning *monitors*, or ``None``."""
    if not monitors:
        return None
    x0 = min(m.x for m in monitors)
    y0 = min(m.y for m in monitors)
    x1 = max(m.x + m.width for m in monitors)
    y1 = max(m.y + m.height for m in monitors)
    return (x0, y0, x1 - x0, y1 - y0)


def merge_monitors_with_dxgi(
    monitors: List[PhysicalMonitor],
    outputs: List[DxgiOutput],
    overrides: Optional[Dict[str, str]] = None,
) -> List[PhysicalMonitor]:
    """Attach a DXGI adapter/output index to each monitor.

    Matching goes from strongest to weakest evidence: the shared GDI device
    name, then an identical rectangle, then the plain enumeration ordinal.  A
    monitor that matches nothing keeps ``None`` indices, which downstream means
    "ddagrab cannot capture this one".  ``overrides`` maps a device name to an
    ``"adapter:output"`` string and always wins.
    """
    by_name = {o.device_name: o for o in outputs if o.device_name}
    by_rect = {(o.x, o.y, o.width, o.height): o for o in outputs}
    merged: List[PhysicalMonitor] = []
    for ordinal, mon in enumerate(monitors):
        match = by_name.get(mon.device_name)
        if match is None:
            match = by_rect.get((mon.x, mon.y, mon.width, mon.height))
        if match is None and len(outputs) == len(monitors):
            match = outputs[ordinal]
        resolved = PhysicalMonitor(**vars(mon))
        if match is not None:
            resolved.dxgi_adapter_index = match.adapter_index
            resolved.dxgi_output_index = match.output_index
        merged.append(resolved)

    for device_name, spec in (overrides or {}).items():
        try:
            adapter_text, _, output_text = str(spec).partition(":")
            adapter = int(adapter_text)
            output = int(output_text) if output_text else 0
        except (TypeError, ValueError):
            log.warning("ignoring malformed DXGI override %r for %s", spec, device_name)
            continue
        for mon in merged:
            if mon.device_name == device_name:
                mon.dxgi_adapter_index = adapter
                mon.dxgi_output_index = output
    return merged


# ---------------------------------------------------------------------------
# Win32 / DXGI probing
# ---------------------------------------------------------------------------

def enumerate_physical_monitors() -> List[PhysicalMonitor]:
    """Return the attached monitors in physical pixels, or ``[]`` on failure.

    The process is already per-monitor DPI aware (Qt sets that up), so the
    rectangles ``GetMonitorInfoW`` hands back are the physical ones.
    """
    if not is_windows():
        return []
    try:
        import ctypes
        import ctypes.wintypes as wintypes

        user32 = ctypes.windll.user32

        class MONITORINFOEXW(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD),
                ("szDevice", wintypes.WCHAR * 32),
            ]

        monitors: List[PhysicalMonitor] = []

        callback_type = ctypes.WINFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.RECT),
            ctypes.c_double,
        )

        def _collect(handle, _hdc, _rect, _data) -> int:
            info = MONITORINFOEXW()
            info.cbSize = ctypes.sizeof(MONITORINFOEXW)
            if user32.GetMonitorInfoW(ctypes.c_void_p(handle), ctypes.byref(info)):
                rect = info.rcMonitor
                monitors.append(
                    PhysicalMonitor(
                        device_name=info.szDevice,
                        x=int(rect.left),
                        y=int(rect.top),
                        width=int(rect.right - rect.left),
                        height=int(rect.bottom - rect.top),
                        is_primary=bool(info.dwFlags & 0x1),  # MONITORINFOF_PRIMARY
                    )
                )
            return 1

        user32.EnumDisplayMonitors(None, None, callback_type(_collect), 0)
        return monitors
    except Exception:  # noqa: BLE001 — a probe must never break recording
        log.debug("physical monitor enumeration failed", exc_info=True)
        return []


# IDXGIFactory1 — {770aae78-f26f-4dba-a829-253c83d1b387}
_IID_IDXGIFactory1 = (
    0x770AAE78, 0xF26F, 0x4DBA,
    (0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
)
# COM vtable slots used below (ABI-stable):
_VT_RELEASE = 2            # IUnknown::Release
_VT_ENUM_ADAPTERS1 = 12    # IDXGIFactory1::EnumAdapters1
_VT_ENUM_OUTPUTS = 7       # IDXGIAdapter::EnumOutputs
_VT_OUTPUT_GET_DESC = 7    # IDXGIOutput::GetDesc


def enumerate_dxgi_outputs() -> List[DxgiOutput]:
    """Return every DXGI output (adapter index, output index, rectangle).

    ``ddagrab``'s ``output_idx`` counts outputs *within one adapter*, so both
    halves are needed to address a monitor on a hybrid-GPU machine.
    """
    if not is_windows():
        return []
    try:
        import ctypes
        import ctypes.wintypes as wintypes

        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", ctypes.c_uint32),
                ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        class DXGI_OUTPUT_DESC(ctypes.Structure):
            _fields_ = [
                ("DeviceName", ctypes.c_wchar * 32),
                ("DesktopCoordinates", wintypes.RECT),
                ("AttachedToDesktop", ctypes.c_int),
                ("Rotation", ctypes.c_uint),
                ("Monitor", ctypes.c_void_p),
            ]

        def call(interface, slot, restype, argtypes, *args):
            vtable = ctypes.cast(
                interface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
            )
            proto = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
            return proto(vtable[0][slot])(interface, *args)

        def release(interface) -> None:
            call(interface, _VT_RELEASE, ctypes.c_ulong, ())

        dxgi = ctypes.windll.dxgi
        iid = GUID(*_IID_IDXGIFactory1[:3], (ctypes.c_ubyte * 8)(*_IID_IDXGIFactory1[3]))
        factory = ctypes.c_void_p()
        if dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory)) != 0:
            return []

        outputs: List[DxgiOutput] = []
        try:
            adapter_index = 0
            while True:
                adapter = ctypes.c_void_p()
                hr = call(
                    factory, _VT_ENUM_ADAPTERS1, ctypes.c_long,
                    (ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)),
                    adapter_index, ctypes.byref(adapter),
                )
                if hr != 0 or not adapter:
                    break
                try:
                    output_index = 0
                    while True:
                        output = ctypes.c_void_p()
                        hr = call(
                            adapter, _VT_ENUM_OUTPUTS, ctypes.c_long,
                            (ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)),
                            output_index, ctypes.byref(output),
                        )
                        if hr != 0 or not output:
                            break
                        try:
                            desc = DXGI_OUTPUT_DESC()
                            if call(
                                output, _VT_OUTPUT_GET_DESC, ctypes.c_long,
                                (ctypes.POINTER(DXGI_OUTPUT_DESC),),
                                ctypes.byref(desc),
                            ) == 0:
                                rect = desc.DesktopCoordinates
                                outputs.append(
                                    DxgiOutput(
                                        adapter_index=adapter_index,
                                        output_index=output_index,
                                        device_name=desc.DeviceName,
                                        x=int(rect.left),
                                        y=int(rect.top),
                                        width=int(rect.right - rect.left),
                                        height=int(rect.bottom - rect.top),
                                        attached=bool(desc.AttachedToDesktop),
                                    )
                                )
                        finally:
                            release(output)
                        output_index += 1
                finally:
                    release(adapter)
                adapter_index += 1
        finally:
            release(factory)
        return [o for o in outputs if o.attached]
    except Exception:  # noqa: BLE001
        log.debug("DXGI output enumeration failed", exc_info=True)
        return []


def physical_window_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
    """Return a window's ``(x, y, w, h)`` in physical pixels, or ``None``.

    Prefers the DWM's extended frame bounds: ``GetWindowRect`` includes the
    invisible resize border, which would otherwise show up as a black margin
    around an active-window capture.
    """
    if not is_windows() or not hwnd:
        return None
    try:
        import ctypes
        import ctypes.wintypes as wintypes

        rect = wintypes.RECT()
        ok = False
        try:
            # DWMWA_EXTENDED_FRAME_BOUNDS == 9
            ok = ctypes.windll.dwmapi.DwmGetWindowAttribute(
                wintypes.HWND(hwnd), ctypes.c_uint(9),
                ctypes.byref(rect), ctypes.sizeof(rect),
            ) == 0
        except Exception:  # noqa: BLE001 — dwmapi missing / attribute refused
            ok = False
        if not ok and not ctypes.windll.user32.GetWindowRect(
            wintypes.HWND(hwnd), ctypes.byref(rect)
        ):
            return None
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width <= 0 or height <= 0:
            return None
        return (int(rect.left), int(rect.top), width, height)
    except Exception:  # noqa: BLE001
        log.debug("window rect lookup failed", exc_info=True)
        return None


def probe_monitors(
    overrides: Optional[Dict[str, str]] = None,
) -> List[PhysicalMonitor]:
    """Physical monitors with their DXGI indices resolved; ``[]`` off-Windows."""
    monitors = enumerate_physical_monitors()
    if not monitors:
        return []
    return merge_monitors_with_dxgi(monitors, enumerate_dxgi_outputs(), overrides)
