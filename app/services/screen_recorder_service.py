"""Screen recorder backed by the system ``ffmpeg`` binary.

The recorder captures the screen (with cursor), the microphone and — best
effort, where a virtual loopback device exists — the system/speaker audio,
writing the result as a sequence of short, independently playable MP4
**segments** (crash protection).  When recording stops the segments are
optionally concatenated into a single ``recording.mp4``.

Design notes
------------
* ffmpeg is driven through :class:`~PySide6.QtCore.QProcess` so it integrates
  with the Qt event loop (``finished`` / ``errorOccurred`` signals, no polling).
* Pause/resume is implemented by stopping the current ffmpeg process (which
  finalizes its open segment) and starting a fresh one that keeps numbering
  segments in the same directory.  The elapsed clock excludes paused time, so
  the video and the :class:`RecordingTimelineLog` stay in sync.
* The argument-building / device-parsing helpers below are plain module-level
  functions with no Qt dependency, so they can be unit tested without ffmpeg.
"""

from __future__ import annotations

import logging
import re
import shutil
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers (no Qt / no subprocess) — unit tested
# ---------------------------------------------------------------------------

# crf: lower = better quality/bigger; scale_height: output height (px, -2 keeps
# aspect with even width); audio_bitrate: AAC bitrate.
_QUALITY_PRESETS: Dict[str, Dict[str, object]] = {
    "low":    {"crf": 30, "scale_height": 540,  "audio_bitrate": "64k"},
    "normal": {"crf": 26, "scale_height": 720,  "audio_bitrate": "96k"},
    "better": {"crf": 22, "scale_height": 1080, "audio_bitrate": "128k"},
}


def quality_preset(name: str) -> Dict[str, object]:
    """Return the encoder parameters for a named quality preset.

    Unknown names fall back to ``"normal"``.
    """
    return dict(_QUALITY_PRESETS.get(name, _QUALITY_PRESETS["normal"]))


@dataclass
class CaptureDevices:
    """Resolved capture device identifiers for the current platform.

    ``screen`` is the avfoundation video index (macOS) or ``"desktop"``
    (Windows gdigrab).  Audio fields are ``None`` when unavailable.  The
    ``*_name`` fields carry the human-readable device names purely for
    diagnostics/logging (the ``microphone``/``system_audio`` ids are what
    actually get passed to ffmpeg).
    """

    screen: str
    microphone: Optional[str] = None
    system_audio: Optional[str] = None
    microphone_name: Optional[str] = None
    system_audio_name: Optional[str] = None
    # Why a system-audio device could not be found (no loopback installed,
    # disabled by config, …) — surfaced in diagnostics so "no system sound"
    # is never silent.
    system_audio_note: Optional[str] = None


@dataclass
class RecordingOptions:
    """Encoding/capture options derived from :class:`RecordingConfig`."""

    fps: int = 18
    quality: str = "normal"
    segment_seconds: int = 8
    capture_cursor: bool = True
    # Audio mix controls.  Volumes are linear gain multipliers (1.0 = unity);
    # mutes drop the source from the mix entirely.  Output is always coerced to
    # ``audio_sample_rate`` / ``audio_channels`` (48 kHz stereo by default).
    mic_volume: float = 1.0
    system_volume: float = 1.0
    mute_microphone: bool = False
    mute_system_audio: bool = False
    audio_sample_rate: int = 48000
    audio_channels: int = 2
    # Emit live peak-level metering on ffmpeg's stdout (drives the VU meter).
    meter_audio: bool = False
    # Windows desktop grabber: ``"auto"`` | ``"ddagrab"`` | ``"gdigrab"``.
    # Ignored on every other platform.
    windows_backend: str = "auto"
    # Force a DXGI adapter for ``ddagrab`` (``None`` → derive from the region).
    windows_dxgi_adapter: Optional[int] = None


class RecordingDisplayMode(Enum):
    """Which part of the screen to record."""

    ACTIVE_WINDOW = "active_window"   # only the application window
    ALL_DISPLAYS = "all"             # every monitor on one canvas
    SELECTED_DISPLAYS = "selected"   # a user-chosen subset of monitors

    @classmethod
    def from_value(cls, value: object) -> "RecordingDisplayMode":
        """Coerce a stored string into a mode, defaulting to ALL_DISPLAYS."""
        for mode in cls:
            if mode.value == value:
                return mode
        return cls.ALL_DISPLAYS


@dataclass
class RecordingDisplayInfo:
    """A monitor as reported by the windowing system.

    ``av_index`` is the macOS avfoundation "Capture screen N" index (best-effort
    ordinal mapping); it is ``None`` on other platforms / when unknown.
    """

    id: str
    name: str
    width: int
    height: int
    is_primary: bool = False
    x: int = 0
    y: int = 0
    av_index: Optional[int] = None
    # Windows only: the monitor rectangle in *physical* pixels.  Qt reports
    # logical (DPI-scaled) coordinates, while ffmpeg's grabbers address the
    # desktop in physical pixels — on a scaled multi-monitor layout the two
    # disagree and a crop computed from the logical values lands off-screen
    # (which gdigrab fills with black).  ``None`` → the logical values above
    # are the best we have (non-Windows, or the native query failed).
    physical_x: Optional[int] = None
    physical_y: Optional[int] = None
    physical_width: Optional[int] = None
    physical_height: Optional[int] = None
    # Windows only: the DXGI adapter/output pair addressed by ``ddagrab``.
    # ``output_idx`` is an index *within an adapter*, so both halves matter on a
    # hybrid-GPU machine.  ``None`` → ddagrab cannot address this monitor.
    dxgi_adapter_index: Optional[int] = None
    dxgi_output_index: Optional[int] = None


def physical_rect(info: RecordingDisplayInfo) -> tuple:
    """Return *info*'s ``(x, y, width, height)`` in physical pixels.

    Falls back to the logical geometry when the native query was unavailable,
    which keeps every non-Windows caller on its previous behaviour.
    """
    if (
        info.physical_width is not None
        and info.physical_height is not None
        and info.physical_x is not None
        and info.physical_y is not None
    ):
        return (
            int(info.physical_x),
            int(info.physical_y),
            int(info.physical_width),
            int(info.physical_height),
        )
    return (int(info.x), int(info.y), int(info.width), int(info.height))


class WindowsCaptureBackend(Enum):
    """Which Windows desktop grabber ffmpeg should use.

    ``gdigrab`` is the legacy GDI/BitBlt grabber: it works everywhere but
    returns all-black frames on many hardware-accelerated, hybrid-GPU or HDR
    setups.  ``ddagrab`` drives the DXGI Desktop Duplication API and therefore
    sees exactly what the compositor puts on the monitor, at true physical
    resolution.  ``auto`` prefers ddagrab and falls back to gdigrab.
    """

    AUTO = "auto"
    DDAGRAB = "ddagrab"
    GDIGRAB = "gdigrab"

    @classmethod
    def from_value(cls, value: object) -> "WindowsCaptureBackend":
        """Coerce a stored string into a backend, defaulting to AUTO."""
        for backend in cls:
            if backend.value == value:
                return backend
        return cls.AUTO


@dataclass
class DdaTarget:
    """One ``ddagrab`` source: a DXGI output and the slice of it to capture.

    ``offset_x/offset_y`` and ``width/height`` are relative to the monitor's own
    top-left corner (ddagrab addresses each output separately).  ``canvas_x`` /
    ``canvas_y`` place the grabbed rectangle on the combined canvas; both are 0
    for a single-monitor capture.  Zero ``width``/``height`` means "the whole
    output".
    """

    output_idx: int
    adapter_idx: int = 0
    width: int = 0
    height: int = 0
    offset_x: int = 0
    offset_y: int = 0
    canvas_x: int = 0
    canvas_y: int = 0


@dataclass
class CaptureRegion:
    """Resolved capture target for a given display mode + platform.

    Exactly one capture strategy is expressed:

    * ``screen_index`` — a macOS avfoundation video device index.
    * ``offset_x/offset_y/width/height`` — a Windows ``gdigrab`` crop rectangle,
      in physical pixels.
    * ``window_title`` — a Windows ``gdigrab`` single-window capture.

    ``dda_targets`` carries the same target expressed for ``ddagrab``; it is
    empty when the region cannot be captured that way (a window-title grab, or
    an unknown DXGI output index), which forces the gdigrab path.

    All ``None`` means "platform default" (macOS: probed screen device;
    Windows: the whole virtual desktop).
    """

    screen_index: Optional[str] = None
    offset_x: Optional[int] = None
    offset_y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    window_title: Optional[str] = None
    dda_targets: List[DdaTarget] = field(default_factory=list)
    canvas_width: Optional[int] = None
    canvas_height: Optional[int] = None


def is_macos(platform: str) -> bool:
    return platform.startswith("darwin")


def is_windows(platform: str) -> bool:
    return platform.startswith("win")


def format_display_label(
    info: RecordingDisplayInfo,
    ordinal: int,
    monitor_word: str = "monitor",
    primary_marker: str = "(primary)",
) -> str:
    """Human-readable monitor label, e.g. ``"1. monitor (primary) — 2560x1440"``."""
    primary = f" {primary_marker}" if info.is_primary else ""
    name = f" {info.name}" if info.name else ""
    return f"{ordinal}. {monitor_word}{primary}{name} — {info.width}x{info.height}"


def _even(value: int) -> int:
    """Round *value* down to the nearest even integer (libx264 needs even dims)."""
    v = max(2, int(value))
    return v - (v % 2)


def _primary_display(
    displays: List[RecordingDisplayInfo],
) -> Optional[RecordingDisplayInfo]:
    if not displays:
        return None
    return next((d for d in displays if d.is_primary), displays[0])


def selected_displays(
    mode: RecordingDisplayMode,
    displays: List[RecordingDisplayInfo],
    selected_ids: List[str],
) -> List[RecordingDisplayInfo]:
    """Return the monitors a non-window mode would capture.

    Falls back to the primary monitor when a SELECTED mode has no valid ids.
    """
    if mode is RecordingDisplayMode.ALL_DISPLAYS:
        return list(displays)
    if mode is RecordingDisplayMode.SELECTED_DISPLAYS:
        wanted = set(selected_ids or [])
        # Ids were once the Qt screen name and are now the GDI device name on
        # Windows, so a stored selection is matched against either.
        chosen = [d for d in displays if d.id in wanted or (d.name and d.name in wanted)]
        if chosen:
            return chosen
        primary = _primary_display(displays)
        return [primary] if primary else []
    # ACTIVE_WINDOW captures no full monitor.
    return []


def displays_bounding_box(
    displays: List[RecordingDisplayInfo],
    physical: bool = True,
) -> Optional[tuple]:
    """Return ``(x, y, width, height)`` spanning *displays*, or ``None``.

    ``physical`` (the default) measures in physical pixels — the coordinate
    space every ffmpeg grabber works in.  Pass ``False`` for logical/Qt units.
    """
    if not displays:
        return None
    rects = [
        physical_rect(d) if physical else (d.x, d.y, d.width, d.height)
        for d in displays
    ]
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[0] + r[2] for r in rects)
    y1 = max(r[1] + r[3] for r in rects)
    return (x0, y0, x1 - x0, y1 - y0)


# gdigrab's banner, e.g.
#   [gdigrab @ 0000...] Capturing whole desktop as 1680x1050x32 at (0,0)
_GDIGRAB_RECT_RE = re.compile(
    r"Capturing whole desktop as (\d+)x(\d+)x\d+ at \((-?\d+)\s*,\s*(-?\d+)\)"
)


def parse_gdigrab_desktop_rect(stderr_text: str) -> Optional[Tuple[int, int, int, int]]:
    """Return the desktop rectangle gdigrab reports, or ``None``.

    ffmpeg.exe ships without a per-monitor DPI manifest, so Windows hands it a
    *virtualized* desktop: on a scaled machine gdigrab's coordinates are neither
    Qt's logical pixels nor the true physical ones.  Rather than guess the
    conversion we ask gdigrab what it sees and calibrate against it.
    """
    match = _GDIGRAB_RECT_RE.search(stderr_text or "")
    if not match:
        return None
    w, h, x, y = (int(g) for g in match.groups())
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def gdigrab_coordinate_scale(
    gdigrab_rect: Optional[Tuple[int, int, int, int]],
    physical: Optional[Tuple[int, int, int, int]],
) -> float:
    """Factor converting physical desktop pixels into gdigrab's coordinates.

    Returns ``1.0`` whenever either rectangle is unknown or the ratio looks
    implausible, which keeps the unscaled machines on their previous behaviour.
    """
    if not gdigrab_rect or not physical:
        return 1.0
    gw, gh = gdigrab_rect[2], gdigrab_rect[3]
    pw, ph = physical[2], physical[3]
    if pw <= 0 or ph <= 0 or gw <= 0 or gh <= 0:
        return 1.0
    scale = gw / pw
    # Windows scaling only ever shrinks the virtualized desktop, and the two
    # axes must agree; anything else means we misread one of the rectangles.
    if not 0.2 <= scale <= 1.0:
        return 1.0
    if abs(scale - (gh / ph)) > 0.02:
        return 1.0
    return scale


def scale_region_for_gdigrab(region: CaptureRegion, scale: float) -> CaptureRegion:
    """Return *region* with its gdigrab crop expressed in gdigrab coordinates.

    The ddagrab targets are left untouched — they always address true physical
    pixels, per DXGI output.
    """
    if scale == 1.0 or region.width is None or region.height is None:
        return region
    scaled = CaptureRegion(**vars(region))
    scaled.offset_x = int(round((region.offset_x or 0) * scale))
    scaled.offset_y = int(round((region.offset_y or 0) * scale))
    scaled.width = _even(round(region.width * scale))
    scaled.height = _even(round(region.height * scale))
    return scaled


def _display_containing(
    displays: List[RecordingDisplayInfo], rect: tuple
) -> Optional[RecordingDisplayInfo]:
    """Return the monitor holding most of *rect* (physical px), or ``None``."""
    x, y, w, h = rect
    best = None
    best_area = 0
    for disp in displays:
        dx, dy, dw, dh = physical_rect(disp)
        ow = max(0, min(x + w, dx + dw) - max(x, dx))
        oh = max(0, min(y + h, dy + dh) - max(y, dy))
        area = ow * oh
        if area > best_area:
            best, best_area = disp, area
    return best


def _dda_targets_for(
    displays: List[RecordingDisplayInfo], box: tuple
) -> tuple:
    """Build the ``ddagrab`` targets covering *box* (physical px).

    Returns ``(targets, canvas_width, canvas_height)``.  The targets are empty
    when any needed monitor has no known DXGI output index — ddagrab cannot
    address it, so the caller must stay on gdigrab.
    """
    bx, by, bw, bh = box
    targets: List[DdaTarget] = []
    for disp in displays:
        if disp.dxgi_output_index is None:
            return ([], None, None)
        dx, dy, dw, dh = physical_rect(disp)
        # Intersect the monitor with the requested box, then express the slice
        # both in monitor-local and canvas coordinates.
        ix0, iy0 = max(bx, dx), max(by, dy)
        ix1, iy1 = min(bx + bw, dx + dw), min(by + bh, dy + dh)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        full = (ix0, iy0, ix1 - ix0, iy1 - iy0) == (dx, dy, dw, dh)
        targets.append(
            DdaTarget(
                output_idx=int(disp.dxgi_output_index),
                adapter_idx=int(disp.dxgi_adapter_index or 0),
                width=0 if full else _even(ix1 - ix0),
                height=0 if full else _even(iy1 - iy0),
                offset_x=ix0 - dx,
                offset_y=iy0 - dy,
                canvas_x=ix0 - bx,
                canvas_y=iy0 - by,
            )
        )
    if not targets:
        return ([], None, None)
    return (targets, _even(bw), _even(bh))


def resolve_capture_region(
    mode: RecordingDisplayMode,
    displays: List[RecordingDisplayInfo],
    selected_ids: List[str],
    active_window_bounds: Optional[tuple],
    platform: str = sys.platform,
) -> CaptureRegion:
    """Resolve the capture rectangle/device for *mode* on *platform*.

    ``active_window_bounds`` is ``(x, y, w, h)`` of the app window in *physical*
    pixels (used only by Windows ACTIVE_WINDOW; macOS avfoundation cannot crop
    to a window, so it falls back to the primary screen device — the caller logs
    that limitation).

    On Windows the returned region carries both strategies: a gdigrab crop
    rectangle *and* the equivalent ``ddagrab`` targets, so the caller can switch
    grabbers without re-resolving anything.
    """
    if not isinstance(mode, RecordingDisplayMode):
        mode = RecordingDisplayMode.from_value(mode)
    mac = is_macos(platform)
    win = is_windows(platform)

    if mode is RecordingDisplayMode.ACTIVE_WINDOW:
        if win:
            if active_window_bounds:
                x, y, w, h = active_window_bounds
                box = (int(x), int(y), _even(w), _even(h))
                host = _display_containing(displays, box)
                targets, cw, ch = (
                    _dda_targets_for([host], box) if host is not None else ([], None, None)
                )
                return CaptureRegion(
                    offset_x=box[0], offset_y=box[1],
                    width=box[2], height=box[3],
                    dda_targets=targets, canvas_width=cw, canvas_height=ch,
                )
            return CaptureRegion()  # no bounds → whole desktop fallback
        if mac:
            primary = _primary_display(displays)
            idx = (
                str(primary.av_index)
                if primary is not None and primary.av_index is not None
                else None
            )
            return CaptureRegion(screen_index=idx)
        return CaptureRegion()

    chosen = selected_displays(mode, displays, selected_ids)
    if win:
        box = displays_bounding_box(chosen)
        if box is None:
            return CaptureRegion()  # whole desktop
        x, y, w, h = box
        box = (int(x), int(y), _even(w), _even(h))
        targets, cw, ch = _dda_targets_for(chosen, box)
        return CaptureRegion(
            offset_x=box[0], offset_y=box[1], width=box[2], height=box[3],
            dda_targets=targets, canvas_width=cw, canvas_height=ch,
        )
    if mac:
        # avfoundation records a single screen device.  For ALL_DISPLAYS there
        # is no merged canvas, so capture the primary monitor; for SELECTED take
        # the first chosen monitor.  ``screen_index`` stays None when the
        # avfoundation index is unknown — the caller then keeps the probed
        # default screen device (build_ffmpeg_args falls back to devices.screen).
        if mode is RecordingDisplayMode.ALL_DISPLAYS:
            target = _primary_display(chosen)
        else:
            target = next((d for d in chosen if d.av_index is not None), None)
        idx = (
            str(target.av_index)
            if target is not None and target.av_index is not None
            else None
        )
        return CaptureRegion(screen_index=idx)
    return CaptureRegion()


def effective_fps(
    base_fps: int,
    mode: RecordingDisplayMode,
    displays: List[RecordingDisplayInfo],
    selected_ids: List[str],
    *,
    auto_reduce: bool = True,
    multi_monitor_cap: int = 15,
) -> int:
    """Cap the frame rate when capturing more than one monitor.

    Multi-monitor captures produce a large canvas; dropping to a lower fps keeps
    the file size sane.  Single-monitor / active-window captures are unaffected.
    """
    if not auto_reduce or mode is RecordingDisplayMode.ACTIVE_WINDOW:
        return base_fps
    if len(selected_displays(mode, displays, selected_ids)) > 1:
        return max(1, min(base_fps, multi_monitor_cap))
    return base_fps


@dataclass
class AudioSource:
    """One audio leg feeding the mixer.

    ``label`` is the ffmpeg input pad (e.g. ``"[0:a]"``); ``volume`` is the
    linear gain; ``kind`` is ``"mic"`` or ``"sys"`` (diagnostics only).
    """

    label: str
    volume: float
    kind: str


# Peak-level metering tap appended to the mix bus.  ``astats`` recomputes the
# peak every ~0.1 s and ``ametadata`` prints it to ffmpeg's *stdout* (``file=-``)
# independently of ``-loglevel`` so the GUI can drive a live VU meter.  The
# parser looks for ``METER_KEY=<dBFS>`` lines.
METER_KEY = "lavfi.astats.Overall.Peak_level"
_METER_TAP = (
    "astats=metadata=1:reset=1:length=0.1,"
    f"ametadata=mode=print:key={METER_KEY}:file=-,"
    "anullsink"
)


def build_audio_filtergraph(
    sources: List[AudioSource],
    *,
    sample_rate: int = 48000,
    channels: int = 2,
    meter: bool = False,
) -> tuple:
    """Build the ``-filter_complex`` graph for the audio mix.

    Returns ``(graph, out_label)`` where *graph* is the filter_complex string
    (or ``None`` when there is no audio) and *out_label* is the pad to map into
    the encoder (``"[aout]"`` normally, ``"[aenc]"`` when *meter* splits off a
    metering tap).

    Each source is volume-adjusted, resampled to *sample_rate* and coerced to a
    *channels*-channel layout; two or more sources are mixed with ``amix`` and
    then run through ``alimiter`` so summed peaks cannot clip.
    """
    if not sources:
        return None, None

    layout = "stereo" if channels == 2 else "mono"
    legs: List[str] = []
    mixed_labels: List[str] = []
    for i, src in enumerate(sources):
        chain = [
            f"aresample={sample_rate}",
            f"aformat=sample_fmts=fltp:channel_layouts={layout}",
        ]
        if abs(src.volume - 1.0) > 1e-3:
            chain.insert(0, f"volume={src.volume:g}")
        out = f"[a{i}]"
        legs.append(f"{src.label}{','.join(chain)}{out}")
        mixed_labels.append(out)

    if len(sources) == 1:
        # Rename the single leg's output pad to the mix-bus label.
        legs[0] = legs[0][: -len(mixed_labels[0])] + "[aout]"
    else:
        # normalize=0 keeps user volumes intact; alimiter prevents the summed
        # signal from clipping past full scale.
        legs.append(
            f"{''.join(mixed_labels)}"
            f"amix=inputs={len(sources)}:duration=longest:normalize=0,"
            "alimiter=limit=0.95[aout]"
        )

    out_label = "[aout]"
    if meter:
        legs.append(f"[aout]asplit=2[aenc][amet];[amet]{_METER_TAP}")
        out_label = "[aenc]"
    return ";".join(legs), out_label


def parse_meter_peak_db(text: str) -> Optional[float]:
    """Extract the most recent peak level (dBFS) from metering stdout, if any."""
    last: Optional[float] = None
    for line in text.splitlines():
        if line.startswith(METER_KEY):
            _, _, value = line.partition("=")
            try:
                last = float(value.strip())
            except ValueError:
                continue
    return last


def probe_gdigrab_rect(
    ffmpeg_path: Optional[str], timeout: int = 10
) -> Optional[Tuple[int, int, int, int]]:
    """Ask gdigrab what desktop rectangle it sees, or ``None``.

    A 0.2 s capture to ``-f null -`` is enough: the banner we need is printed
    while the input is being opened.  Never raises.
    """
    if not ffmpeg_path:
        return None
    import subprocess  # local import: only needed when a recording starts

    try:
        proc = subprocess.run(
            [
                ffmpeg_path,
                "-hide_banner", "-loglevel", "info",
                "-f", "gdigrab", "-framerate", "5", "-i", "desktop",
                "-t", "0.2", "-f", "null", "-",
            ],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except Exception:  # noqa: BLE001
        log.debug("gdigrab desktop probe failed", exc_info=True)
        return None
    return parse_gdigrab_desktop_rect(proc.stderr or "")


def resolve_windows_backend(
    value: object, region: Optional[CaptureRegion]
) -> WindowsCaptureBackend:
    """Decide which Windows grabber to use for *region*.

    ``auto`` prefers ddagrab, but only when the region was resolved into DXGI
    targets — a window-title grab or an unknown output index leaves gdigrab as
    the only option.
    """
    backend = WindowsCaptureBackend.from_value(value)
    has_targets = region is not None and bool(region.dda_targets)
    if backend is WindowsCaptureBackend.DDAGRAB and not has_targets:
        return WindowsCaptureBackend.GDIGRAB
    if backend is WindowsCaptureBackend.AUTO:
        return (
            WindowsCaptureBackend.DDAGRAB
            if has_targets
            else WindowsCaptureBackend.GDIGRAB
        )
    return backend


def ddagrab_adapter_index(
    region: Optional[CaptureRegion], override: Optional[int] = None
) -> int:
    """Return the DXGI adapter index to bind the d3d11 device to.

    ``ddagrab``'s ``output_idx`` is relative to the adapter the hardware device
    was created on, so a hybrid-GPU machine needs the right adapter or the
    indices address the wrong monitors (or nothing at all).
    """
    if override is not None:
        return int(override)
    if region is not None and region.dda_targets:
        return int(region.dda_targets[0].adapter_idx)
    return 0


def build_ddagrab_graph(
    options: RecordingOptions,
    region: Optional[CaptureRegion],
    scale_height: int,
    out_label: str = "[vout]",
) -> str:
    """Build the ``ddagrab`` video filter chain ending in *out_label*.

    One target is a straight grab.  Several targets (a multi-monitor canvas) are
    composited by padding the first monitor out to the canvas size and
    overlaying the rest at their canvas offsets — padding a live stream rather
    than overlaying onto a synthetic ``color`` source keeps the whole graph
    driven by real capture timing.
    """
    draw_mouse = "1" if options.capture_cursor else "0"
    targets = list(region.dda_targets) if region is not None else []
    if not targets:
        targets = [DdaTarget(output_idx=0)]

    def source(target: DdaTarget) -> str:
        # ``allow_fallback=1`` keeps an HDR / 10-bit desktop from erroring the
        # filter outright, and the ``bgra|x2bgr10`` download list accepts either
        # surface format before the final conversion to plain BGRA.
        src = (
            f"ddagrab=output_idx={target.output_idx}"
            f":framerate={options.fps}"
            f":draw_mouse={draw_mouse}"
            ":allow_fallback=1"
        )
        if target.width and target.height:
            src += (
                f":offset_x={target.offset_x}:offset_y={target.offset_y}"
                f":video_size={target.width}x{target.height}"
            )
        return f"{src},hwdownload,format=bgra|x2bgr10,format=bgra"

    if len(targets) == 1:
        return f"{source(targets[0])},scale=-2:{scale_height}{out_label}"

    canvas_w = region.canvas_width if region is not None else 0
    canvas_h = region.canvas_height if region is not None else 0
    base, rest = targets[0], targets[1:]
    parts = [
        f"{source(base)},"
        f"pad={canvas_w}:{canvas_h}:{base.canvas_x}:{base.canvas_y}:color=black"
        "[dda_base]"
    ]
    for i, target in enumerate(rest):
        parts.append(f"{source(target)}[dda{i}]")
    prev = "[dda_base]"
    for i, target in enumerate(rest):
        step = f"{prev}[dda{i}]overlay=x={target.canvas_x}:y={target.canvas_y}"
        if i == len(rest) - 1:
            parts.append(f"{step},scale=-2:{scale_height}{out_label}")
        else:
            prev = f"[dda_mix{i}]"
            parts.append(f"{step}{prev}")
    return ";".join(parts)


def build_ffmpeg_args(
    platform: str,
    devices: CaptureDevices,
    options: RecordingOptions,
    segment_pattern: str,
    region: Optional[CaptureRegion] = None,
) -> List[str]:
    """Build the ffmpeg argument list (excluding the binary itself).

    Produces a segmented MP4 capture.  ``segment_pattern`` is an ffmpeg output
    pattern such as ``/out/seg_%05d.mp4``.  ``region`` selects which monitor /
    window / desktop crop to capture; ``None`` keeps the platform default
    (macOS: the probed screen device; Windows: the whole virtual desktop).
    """
    preset = quality_preset(options.quality)
    crf = preset["crf"]
    height = preset["scale_height"]
    audio_bitrate = preset["audio_bitrate"]

    args: List[str] = ["-hide_banner", "-loglevel", "warning", "-y"]

    # Resolve which audio sources are actually mixed in.  A muted source is
    # dropped before opening it so we neither capture nor pay for it.
    use_mic = devices.microphone is not None and not options.mute_microphone
    use_sys = devices.system_audio is not None and not options.mute_system_audio

    # Audio legs feeding the mixer, with the ffmpeg input pad each maps to.
    audio_sources: List[AudioSource] = []
    input_index = 0  # advances for every ``-i`` we append
    # ddagrab is a *source filter*, not an input: its video arrives through the
    # filtergraph, so the encoder maps a graph label instead of an input stream
    # and the scaling happens inside the graph rather than in ``-vf``.
    video_graph: Optional[str] = None
    video_map = "0:v"

    if is_macos(platform):
        cursor = "1" if options.capture_cursor else "0"
        # A region may override which avfoundation screen device is captured.
        screen = devices.screen
        if region is not None and region.screen_index:
            screen = region.screen_index
        # On avfoundation the microphone shares input 0 with the screen video.
        video_audio = f"{screen}:{devices.microphone}" if use_mic else screen
        args += [
            "-f", "avfoundation",
            "-capture_cursor", cursor,
            "-framerate", str(options.fps),
            "-i", video_audio,
        ]
        if use_mic:
            audio_sources.append(
                AudioSource(f"[{input_index}:a]", options.mic_volume, "mic")
            )
        input_index += 1
        if use_sys:
            args += ["-f", "avfoundation", "-i", f":{devices.system_audio}"]
            audio_sources.append(
                AudioSource(f"[{input_index}:a]", options.system_volume, "sys")
            )
            input_index += 1
    elif is_windows(platform):
        backend = resolve_windows_backend(options.windows_backend, region)
        if backend is WindowsCaptureBackend.DDAGRAB:
            adapter = ddagrab_adapter_index(region, options.windows_dxgi_adapter)
            args += ["-init_hw_device", f"d3d11va:{adapter}"]
            video_graph = build_ddagrab_graph(options, region, int(height))
            video_map = "[vout]"
            # No video ``-i`` — the audio inputs start at index 0.
        else:
            draw_mouse = "1" if options.capture_cursor else "0"
            # gdigrab input options (crop offset / size) must precede ``-i``.
            grab: List[str] = [
                "-f", "gdigrab",
                "-draw_mouse", draw_mouse,
                "-framerate", str(options.fps),
            ]
            if region is not None and region.window_title:
                args += grab + ["-i", f"title={region.window_title}"]
            elif region is not None and region.width and region.height:
                args += grab + [
                    "-offset_x", str(region.offset_x or 0),
                    "-offset_y", str(region.offset_y or 0),
                    "-video_size", f"{region.width}x{region.height}",
                    "-i", "desktop",
                ]
            else:
                args += grab + ["-i", "desktop"]
            input_index += 1  # gdigrab desktop is input 0 (video only)
        if use_mic:
            args += ["-f", "dshow", "-i", f"audio={devices.microphone}"]
            audio_sources.append(
                AudioSource(f"[{input_index}:a]", options.mic_volume, "mic")
            )
            input_index += 1
        if use_sys:
            args += ["-f", "dshow", "-i", f"audio={devices.system_audio}"]
            audio_sources.append(
                AudioSource(f"[{input_index}:a]", options.system_volume, "sys")
            )
            input_index += 1
    else:
        raise ValueError(f"unsupported platform for recording: {platform!r}")

    # Video encode (shared).  Force a keyframe at every segment boundary —
    # the segment muxer can only cut on keyframes, so without this the output
    # would not actually split into short files (breaking crash protection).
    args += [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
    ]
    if video_graph is None:
        args += ["-vf", f"scale=-2:{height}"]
    args += [
        "-r", str(options.fps),
        "-g", str(max(1, options.fps * options.segment_seconds)),
        "-force_key_frames",
        f"expr:gte(t,n_forced*{options.segment_seconds})",
    ]

    # Audio mapping.  Every present source is volume-adjusted, resampled and
    # (when >1) mixed with clipping protection into a single ``[aout]`` bus that
    # is encoded as one AAC track at the configured sample-rate/channels.  When
    # there is no audio at all we mux video only (``-an``).
    graph, out_label = build_audio_filtergraph(
        audio_sources,
        sample_rate=options.audio_sample_rate,
        channels=options.audio_channels,
        meter=options.meter_audio,
    )
    # The video and audio chains share a single ``-filter_complex``.
    combined = ";".join(part for part in (video_graph, graph) if part)
    if combined:
        args += ["-filter_complex", combined]
    args += ["-map", video_map]
    if graph is not None:
        args += [
            "-map", out_label,
            "-c:a", "aac",
            "-b:a", str(audio_bitrate),
            "-ar", str(options.audio_sample_rate),
            "-ac", str(options.audio_channels),
        ]
    else:
        args += ["-an"]

    # Segment muxer — each closed segment is an independently playable file.
    args += [
        "-f", "segment",
        "-segment_time", str(options.segment_seconds),
        "-reset_timestamps", "1",
        "-segment_format", "mp4",
        segment_pattern,
    ]
    return args


def build_concat_list(segment_paths: List[Path]) -> str:
    """Build the body of an ffmpeg concat-demuxer list file.

    Each line is ``file '<absolute-path>'`` with single quotes escaped.
    """
    lines = []
    for p in segment_paths:
        escaped = str(p).replace("'", r"'\''")
        lines.append(f"file '{escaped}'")
    return "\n".join(lines) + ("\n" if lines else "")


def build_concat_args(list_file: Path, output_file: Path) -> List[str]:
    """Build ffmpeg args that stream-copy a concat list into one file."""
    return [
        "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_file),
    ]


def parse_avfoundation_devices(stderr_text: str) -> Dict[str, List[tuple]]:
    """Parse ``ffmpeg -f avfoundation -list_devices true`` stderr output.

    Returns ``{"video": [(index, name), ...], "audio": [(index, name), ...]}``.
    """
    video: List[tuple] = []
    audio: List[tuple] = []
    bucket: Optional[List[tuple]] = None
    line_re = re.compile(r"\[(\d+)\]\s+(.*)$")
    for raw in stderr_text.splitlines():
        low = raw.lower()
        if "avfoundation video devices" in low:
            bucket = video
            continue
        if "avfoundation audio devices" in low:
            bucket = audio
            continue
        if bucket is None:
            continue
        m = line_re.search(raw.strip())
        if m:
            bucket.append((m.group(1), m.group(2).strip()))
    return {"video": video, "audio": audio}


def capture_screen_indices(video_devices: List[tuple]) -> List[str]:
    """avfoundation device indices of the "Capture screen N" entries, in order.

    ``video_devices`` is ``[(index, name), ...]`` as returned by
    :func:`parse_avfoundation_devices`.  Screen-capture devices are listed
    *after* cameras (e.g. ``[3] Capture screen 0``), so their device index is
    **not** the monitor ordinal — this maps monitor order → real device index.
    """
    screens = [
        (idx, name)
        for idx, name in video_devices
        if "capture screen" in name.lower()
    ]

    def _screen_num(name: str) -> int:
        m = re.search(r"(\d+)\s*$", name)
        return int(m.group(1)) if m else 0

    screens.sort(key=lambda p: _screen_num(p[1]))
    return [idx for idx, _ in screens]


def parse_dshow_audio_devices(stderr_text: str) -> List[str]:
    """Parse ``ffmpeg -f dshow -list_devices true`` stderr for audio names."""
    names: List[str] = []
    in_audio = False
    name_re = re.compile(r'"([^"]+)"')
    for raw in stderr_text.splitlines():
        low = raw.lower()
        if "audio devices" in low:
            in_audio = True
            continue
        if "video devices" in low:
            in_audio = False
            continue
        # New FFmpeg
        if "(audio)" in low:
            m = name_re.search(raw)
            if m:
                names.append(m.group(1))
            continue
        # Old FFmpeg
        if in_audio:
            m = name_re.search(raw)
            if m:
                names.append(m.group(1))
    return names


# Heuristic names that identify a virtual loopback (system-audio) device.
# Includes localized "Stereo Mix" variants — on a non-English Windows the
# device is renamed (e.g. Hungarian "Sztereó keverő"), so matching only the
# English string would silently skip a perfectly usable loopback.
_LOOPBACK_HINTS = (
    "blackhole",
    "loopback",
    "soundflower",
    "virtual-audio-capturer",
    "virtual audio",
    "vb-audio",
    "voicemeeter",
    "cable output",
    "what u hear",
    "what you hear",
    "wave out mix",
    "wave out",
    # "Stereo Mix" across locales (en / hu / de / fr / es / it / nl / pl / …).
    "stereo mix",
    "stereomix",
    "sztereó keverő",
    "szteró keverő",
    "sztereó keverés",
    "hangkeverő",
    "stereomischung",
    "mixage stéréo",
    "mezcla estéreo",
    "missaggio stereo",
    "stereo-mix",
)


def pick_system_audio(candidates: List[str]) -> Optional[str]:
    """Return the first candidate device name that looks like a loopback."""
    for name in candidates:
        low = name.lower()
        if any(hint in low for hint in _LOOPBACK_HINTS):
            return name
    return None


# Continuity / wireless devices that frequently fail to deliver samples when
# picked as the default capture mic (they may be asleep or off-network).
_FLAKY_MIC_HINTS = ("iphone", "ipad", "apple watch", "continuity")
# Names that indicate a reliable built-in / wired microphone.
_PREFERRED_MIC_HINTS = ("macbook", "built-in", "built in", "internal", "imac")


def pick_microphone(audio: List[tuple]) -> Optional[str]:
    """Choose the best microphone index from ``[(index, name), ...]``.

    Prefers a built-in mic, then any non-Continuity device, and only falls back
    to a flaky wireless/Continuity device (e.g. an iPhone mic) when nothing
    better exists — those often deliver no audio samples and can stall ffmpeg.
    """
    if not audio:
        return None
    for idx, name in audio:
        if any(h in name.lower() for h in _PREFERRED_MIC_HINTS):
            return idx
    for idx, name in audio:
        if not any(h in name.lower() for h in _FLAKY_MIC_HINTS):
            return idx
    return audio[0][0]


def audio_diagnostics(
    devices: CaptureDevices,
    options: RecordingOptions,
    platform: str = sys.platform,
) -> List[str]:
    """Build the ``[Audio] …`` diagnostic lines for *devices*/*options*.

    Returned as a list so callers can log each line; this makes a silent
    "no audio" outcome explainable (which devices were chosen, what was
    skipped and why, the mix format).
    """
    lines: List[str] = []
    plat = "macOS avfoundation" if is_macos(platform) else (
        "Windows dshow" if is_windows(platform) else platform
    )
    lines.append(f"[Audio] Platform: {plat}")

    if devices.microphone is not None:
        name = devices.microphone_name or devices.microphone
        if options.mute_microphone:
            lines.append(f"[Audio] Microphone MUTED (would use: {name})")
        else:
            lines.append(
                f"[Audio] Microphone capture: {name} "
                f"(volume {options.mic_volume:g})"
            )
    else:
        lines.append("[Audio] Microphone: none detected / disabled")

    if devices.system_audio is not None:
        name = devices.system_audio_name or devices.system_audio
        if options.mute_system_audio:
            lines.append(f"[Audio] System audio MUTED (would use: {name})")
        else:
            lines.append(
                f"[Audio] System audio capture: {name} "
                f"(volume {options.system_volume:g})"
            )
    else:
        note = devices.system_audio_note or "no loopback device available"
        lines.append(f"[Audio] System audio: NOT captured — {note}")

    n = sum(
        1
        for present, muted in (
            (devices.microphone is not None, options.mute_microphone),
            (devices.system_audio is not None, options.mute_system_audio),
        )
        if present and not muted
    )
    if n >= 2:
        lines.append("[Audio] Mixer initialized: 2 sources → 1 track")
    elif n == 1:
        lines.append("[Audio] Mixer initialized: 1 source → 1 track")
    else:
        lines.append("[Audio] Mixer initialized: NO audio sources → silent video")
    lines.append(
        f"[Audio] Output format: {options.audio_sample_rate} Hz, "
        f"{options.audio_channels} ch, AAC"
    )
    return lines


@dataclass
class AudioValidation:
    """Result of inspecting the final recording for a usable audio track."""

    has_audio: bool
    codec: Optional[str] = None
    duration: Optional[float] = None
    bit_rate: Optional[int] = None
    channels: Optional[int] = None
    sample_rate: Optional[int] = None
    error: Optional[str] = None

    def summary(self) -> str:
        if self.error:
            return f"[Audio] Final mux validation failed: {self.error}"
        if not self.has_audio:
            return "[Audio] Final mux contains audio=false — NO audio track"
        return (
            "[Audio] Final mux contains audio=true "
            f"(codec={self.codec}, {self.sample_rate} Hz, {self.channels} ch, "
            f"{self.duration:.1f}s, {self.bit_rate or 0} bit/s)"
        )


def resolve_ffprobe(ffmpeg_path: Optional[str]) -> Optional[str]:
    """Locate ``ffprobe`` — next to *ffmpeg_path* first, then on PATH."""
    if ffmpeg_path:
        cand = Path(ffmpeg_path).with_name(
            "ffprobe.exe" if ffmpeg_path.lower().endswith(".exe") else "ffprobe"
        )
        if cand.exists():
            return str(cand)
    return shutil.which("ffprobe")


def parse_ffprobe_audio(stdout_text: str) -> AudioValidation:
    """Parse ``ffprobe -show_streams`` flat key=value output for the audio track.

    Looks for an ``audio`` stream and reports its codec, duration, bitrate,
    channels and sample-rate.  ``has_audio`` is ``True`` only when an audio
    stream exists *and* its duration is greater than zero.
    """
    streams: List[Dict[str, str]] = []
    current: Optional[Dict[str, str]] = None
    for raw in stdout_text.splitlines():
        line = raw.strip()
        if line == "[STREAM]":
            current = {}
            continue
        if line == "[/STREAM]":
            if current is not None:
                streams.append(current)
            current = None
            continue
        if current is not None and "=" in line:
            key, _, value = line.partition("=")
            current[key.strip()] = value.strip()

    def _to_float(value: Optional[str]) -> Optional[float]:
        try:
            return float(value) if value not in (None, "", "N/A") else None
        except ValueError:
            return None

    def _to_int(value: Optional[str]) -> Optional[int]:
        f = _to_float(value)
        return int(f) if f is not None else None

    for st in streams:
        if st.get("codec_type") != "audio":
            continue
        duration = _to_float(st.get("duration"))
        return AudioValidation(
            has_audio=duration is not None and duration > 0,
            codec=st.get("codec_name"),
            duration=duration,
            bit_rate=_to_int(st.get("bit_rate")),
            channels=_to_int(st.get("channels")),
            sample_rate=_to_int(st.get("sample_rate")),
        )
    return AudioValidation(has_audio=False)


def validate_recording_audio(
    ffprobe_path: Optional[str], mp4_path: Path
) -> AudioValidation:
    """Probe *mp4_path* and report whether it carries a usable audio track.

    Never raises — a missing ffprobe / probe failure is returned as an
    ``AudioValidation`` with ``error`` set so the caller can log it.
    """
    if not ffprobe_path:
        return AudioValidation(has_audio=False, error="ffprobe not available")
    if not Path(mp4_path).exists():
        return AudioValidation(has_audio=False, error=f"file missing: {mp4_path}")
    import subprocess  # local import: only needed at validation time

    try:
        proc = subprocess.run(
            [
                ffprobe_path,
                "-hide_banner",
                "-loglevel", "error",
                "-show_streams",
                "-show_entries",
                "stream=codec_type,codec_name,duration,bit_rate,channels,sample_rate",
                str(mp4_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except Exception as exc:  # noqa: BLE001
        return AudioValidation(has_audio=False, error=str(exc))
    if proc.returncode != 0:
        return AudioValidation(
            has_audio=False,
            error=(proc.stderr or "ffprobe failed").strip()[:200],
        )
    return parse_ffprobe_audio(proc.stdout or "")


# blackdetect only reports a run once it closes (EOF counts), and it logs at
# INFO — a probe therefore has to raise ffmpeg's log level above the recorder's
# usual ``warning`` or it would never see a single line.
_BLACKDETECT_RE = re.compile(
    r"black_start:([\d.]+)\s+black_end:([\d.]+)\s+black_duration:([\d.]+)"
)
# Fraction of the observed footage that must be black before we call it black.
_BLACK_RATIO_THRESHOLD = 0.9
_BLACKDETECT_FILTER = "blackdetect=d=0.1:pic_th=0.98:pix_th=0.10"


def parse_blackdetect(stderr_text: str) -> Tuple[float, int]:
    """Return ``(total black seconds, interval count)`` from ffmpeg's stderr."""
    total = 0.0
    count = 0
    for match in _BLACKDETECT_RE.finditer(stderr_text or ""):
        try:
            total += float(match.group(3))
        except ValueError:
            continue
        count += 1
    return (total, count)


@dataclass
class BlackProbe:
    """Result of a short pre-flight capture used to detect a black grabber."""

    ok: bool                      # ffmpeg ran and produced frames
    is_black: bool
    backend: str = "gdigrab"
    black_seconds: float = 0.0
    observed_seconds: float = 0.0
    error: Optional[str] = None

    def summary(self) -> str:
        if not self.ok:
            return (
                f"[Video] Pre-flight {self.backend} capture failed: "
                f"{self.error or 'no frames produced'}"
            )
        if self.is_black:
            return (
                f"[Video] Pre-flight {self.backend} capture is BLACK "
                f"({self.black_seconds:.1f}s of {self.observed_seconds:.1f}s)"
            )
        return (
            f"[Video] Pre-flight {self.backend} capture OK "
            f"({self.observed_seconds:.1f}s, {self.black_seconds:.1f}s black)"
        )


def evaluate_black_probe(
    stderr_text: str,
    *,
    backend: str,
    probe_seconds: float,
    exit_code: int,
) -> BlackProbe:
    """Turn a probe's stderr + exit code into a :class:`BlackProbe`."""
    black_seconds, _ = parse_blackdetect(stderr_text)
    produced_frames = "frame=" in (stderr_text or "")
    if exit_code != 0 or not produced_frames:
        tail = (stderr_text or "").strip().splitlines()
        return BlackProbe(
            ok=False,
            is_black=False,
            backend=backend,
            error=tail[-1][:200] if tail else None,
        )
    observed = max(probe_seconds, 0.001)
    return BlackProbe(
        ok=True,
        is_black=(black_seconds / observed) >= _BLACK_RATIO_THRESHOLD,
        backend=backend,
        black_seconds=black_seconds,
        observed_seconds=observed,
    )


def build_preflight_args(
    platform: str,
    options: RecordingOptions,
    region: Optional[CaptureRegion],
    *,
    backend: str,
    probe_seconds: float = 1.0,
    probe_height: int = 240,
) -> List[str]:
    """Build a short capture that only reports whether the picture is black.

    Uses the same grabber configuration as the real recording but decodes to
    ``-f null -`` at a tiny resolution, so it costs about a second and writes
    nothing to disk.
    """
    probe_options = RecordingOptions(**vars(options))
    probe_options.capture_cursor = False
    probe_options.meter_audio = False
    probe_options.windows_backend = backend

    args = ["-hide_banner", "-loglevel", "info", "-y"]
    if is_windows(platform) and (
        resolve_windows_backend(backend, region) is WindowsCaptureBackend.DDAGRAB
    ):
        adapter = ddagrab_adapter_index(region, options.windows_dxgi_adapter)
        graph = build_ddagrab_graph(probe_options, region, probe_height, "[probe]")
        args += [
            "-init_hw_device", f"d3d11va:{adapter}",
            "-filter_complex", f"{graph};[probe]{_BLACKDETECT_FILTER}[vout]",
            "-map", "[vout]",
        ]
    elif is_windows(platform):
        draw = ["-f", "gdigrab", "-draw_mouse", "0", "-framerate", str(options.fps)]
        if region is not None and region.window_title:
            args += draw + ["-i", f"title={region.window_title}"]
        elif region is not None and region.width and region.height:
            args += draw + [
                "-offset_x", str(region.offset_x or 0),
                "-offset_y", str(region.offset_y or 0),
                "-video_size", f"{region.width}x{region.height}",
                "-i", "desktop",
            ]
        else:
            args += draw + ["-i", "desktop"]
        args += ["-vf", f"scale=-2:{probe_height},{_BLACKDETECT_FILTER}"]
    else:
        screen = (
            region.screen_index
            if region is not None and region.screen_index
            else "0"
        )
        args += [
            "-f", "avfoundation",
            "-capture_cursor", "0",
            "-framerate", str(options.fps),
            "-i", str(screen),
            "-vf", f"scale=-2:{probe_height},{_BLACKDETECT_FILTER}",
        ]
    args += ["-an", "-t", str(probe_seconds), "-f", "null", "-"]
    return args


def build_black_scan_args(
    mp4_path: Path, sample_seconds: Optional[int] = 30
) -> List[str]:
    """Build an ffmpeg pass that reports the black intervals of *mp4_path*.

    Only the head of the file is scanned by default: a grabber that produces no
    picture does so from the first frame, and an unbounded decode would stall
    the (synchronous) stop path on a long recording.
    """
    args = ["-hide_banner", "-loglevel", "info", "-y"]
    if sample_seconds:
        args += ["-t", str(sample_seconds)]
    args += [
        "-i", str(mp4_path),
        "-vf", "blackdetect=d=0.5:pic_th=0.98:pix_th=0.10",
        "-an", "-f", "null", "-",
    ]
    return args


@dataclass
class VideoValidation:
    """Result of inspecting the final recording for an actual picture."""

    is_black: bool
    black_seconds: float = 0.0
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    codec: Optional[str] = None
    error: Optional[str] = None

    def summary(self) -> str:
        if self.error:
            return f"[Video] Final mux validation failed: {self.error}"
        size = (
            f"{self.width}x{self.height}"
            if self.width and self.height
            else "unknown size"
        )
        if self.is_black:
            return (
                f"[Video] Final mux is ALL BLACK ({self.black_seconds:.1f}s black, "
                f"{size}) — the capture produced no picture"
            )
        return (
            f"[Video] Final mux picture OK ({self.codec or '?'} {size}, "
            f"{(self.duration or 0.0):.1f}s, {self.black_seconds:.1f}s black)"
        )


def parse_ffprobe_video(stdout_text: str) -> VideoValidation:
    """Parse ``ffprobe -show_streams`` flat output for the video track."""
    current: Optional[Dict[str, str]] = None
    streams: List[Dict[str, str]] = []
    for raw in (stdout_text or "").splitlines():
        line = raw.strip()
        if line == "[STREAM]":
            current = {}
        elif line == "[/STREAM]":
            if current is not None:
                streams.append(current)
            current = None
        elif current is not None and "=" in line:
            key, _, value = line.partition("=")
            current[key.strip()] = value.strip()

    def _num(value, cast):
        try:
            return cast(value) if value not in (None, "", "N/A") else None
        except (TypeError, ValueError):
            return None

    for stream in streams:
        if stream.get("codec_type") != "video":
            continue
        return VideoValidation(
            is_black=False,
            codec=stream.get("codec_name"),
            duration=_num(stream.get("duration"), float),
            width=_num(stream.get("width"), int),
            height=_num(stream.get("height"), int),
        )
    return VideoValidation(is_black=False, error="no video stream")


def validate_recording_video(
    ffmpeg_path: Optional[str],
    ffprobe_path: Optional[str],
    mp4_path: Path,
    sample_seconds: Optional[int] = 30,
    timeout: int = 60,
) -> VideoValidation:
    """Report whether *mp4_path* carries an actual picture rather than black.

    Never raises — a missing binary or a failed pass comes back with ``error``
    set so the caller can log it without a recording ever being lost to it.
    """
    if not Path(mp4_path).exists():
        return VideoValidation(is_black=False, error=f"file missing: {mp4_path}")
    import subprocess  # local import: only needed at validation time

    result = VideoValidation(is_black=False)
    if ffprobe_path:
        try:
            probe = subprocess.run(
                [
                    ffprobe_path,
                    "-hide_banner", "-loglevel", "error",
                    "-show_streams",
                    "-show_entries",
                    "stream=codec_type,codec_name,duration,width,height",
                    str(mp4_path),
                ],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=15,
            )
            if probe.returncode == 0:
                result = parse_ffprobe_video(probe.stdout or "")
        except Exception as exc:  # noqa: BLE001
            result = VideoValidation(is_black=False, error=str(exc))

    if not ffmpeg_path:
        result.error = result.error or "ffmpeg not available"
        return result
    try:
        scan = subprocess.run(
            [ffmpeg_path, *build_black_scan_args(Path(mp4_path), sample_seconds)],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)
        return result
    result.black_seconds, _ = parse_blackdetect(scan.stderr or "")
    span = result.duration or 0.0
    if sample_seconds:
        span = min(span, float(sample_seconds)) if span else float(sample_seconds)
    if span > 0:
        result.is_black = (result.black_seconds / span) >= _BLACK_RATIO_THRESHOLD
    return result


# ---------------------------------------------------------------------------
# Qt-backed recorder service
# ---------------------------------------------------------------------------

class RecorderState(Enum):
    IDLE = "idle"
    PREFLIGHT = "preflight"   # short black-frame probe before the real capture
    RECORDING = "recording"
    PAUSED = "paused"
    FINALIZING = "finalizing"
    ERROR = "error"


def resolve_ffmpeg(explicit_path: Optional[str] = None) -> Optional[str]:
    """Return a usable ffmpeg path, or ``None`` if it cannot be found."""
    if explicit_path:
        p = Path(explicit_path).expanduser()
        if p.exists():
            return str(p)
    found = shutil.which("ffmpeg")
    return found


def probe_devices(
    ffmpeg_path: str,
    platform: str = sys.platform,
    *,
    want_system_audio: bool = True,
    mic_name: Optional[str] = None,
    system_audio_name: Optional[str] = None,
) -> CaptureDevices:
    """Probe ffmpeg for the screen + microphone (+ best-effort loopback).

    Runs ``ffmpeg -list_devices true`` and parses stderr.  Falls back to
    sensible defaults when probing yields nothing.  Never raises — capture
    can still be attempted with the defaults.

    ``mic_name`` / ``system_audio_name`` request a specific device by name (or
    name substring); when given and matched they override the auto-pick.  The
    returned :class:`CaptureDevices` carries device names + a ``system_audio_note``
    for diagnostics.
    """
    import subprocess  # local import: only needed at probe time

    def _list(args: List[str]) -> str:
        try:
            proc = subprocess.run(
                [ffmpeg_path, *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            return (proc.stderr or "") + (proc.stdout or "")
        except Exception as exc:  # noqa: BLE001
            log.warning("device probe failed: %s", exc)
            return ""

    if is_macos(platform):
        text = _list(["-f", "avfoundation", "-list_devices", "true", "-i", ""])
        parsed = parse_avfoundation_devices(text)
        screen = next(
            (idx for idx, name in parsed["video"] if "screen" in name.lower()),
            "1",  # avfoundation screen is commonly index 1
        )
        audio = parsed["audio"]
        # Explicit override by name → its index; else the auto-pick.
        mic = _match_av_index(audio, mic_name) or pick_microphone(audio) or "0"
        mic_label = next((n for i, n in audio if i == mic), mic_name)
        sys_audio, sys_label, note = _resolve_system_audio_macos(
            audio, want_system_audio, system_audio_name
        )
        return CaptureDevices(
            screen=screen,
            microphone=mic,
            system_audio=sys_audio,
            microphone_name=mic_label,
            system_audio_name=sys_label,
            system_audio_note=note,
        )

    if is_windows(platform):
        text = _list(["-f", "dshow", "-list_devices", "true", "-i", "dummy"])
        names = parse_dshow_audio_devices(text)
        mic = _match_name(names, mic_name) or next(
            (n for n in names if not _is_loopback(n)),
            names[0] if names else None,
        )
        sys_audio: Optional[str] = None
        note: Optional[str] = None
        if want_system_audio:
            sys_audio = _match_name(names, system_audio_name) or pick_system_audio(
                names
            )
            if sys_audio is None:
                note = (
                    "no WASAPI loopback / 'Stereo Mix' / 'virtual-audio-capturer' "
                    "device found"
                )
        else:
            note = "disabled in settings"
        return CaptureDevices(
            screen="desktop",
            microphone=mic,
            system_audio=sys_audio,
            microphone_name=mic,
            system_audio_name=sys_audio,
            system_audio_note=note,
        )

    return CaptureDevices(screen="0")


def _match_name(names: List[str], wanted: Optional[str]) -> Optional[str]:
    """Return the device name matching *wanted* (exact, then substring)."""
    if not wanted:
        return None
    if wanted in names:
        return wanted
    low = wanted.lower()
    return next((n for n in names if low in n.lower()), None)


def _match_av_index(
    audio: List[tuple], wanted: Optional[str]
) -> Optional[str]:
    """Return the avfoundation index whose name matches *wanted*."""
    if not wanted:
        return None
    low = wanted.lower()
    for idx, name in audio:
        if name == wanted or low in name.lower():
            return idx
    return None


def _resolve_system_audio_macos(
    audio: List[tuple],
    want_system_audio: bool,
    system_audio_name: Optional[str],
) -> tuple:
    """Resolve (index, name, note) for the macOS system-audio loopback."""
    if not want_system_audio:
        return None, None, "disabled in settings"
    if system_audio_name:
        idx = _match_av_index(audio, system_audio_name)
        if idx is not None:
            name = next((n for i, n in audio if i == idx), system_audio_name)
            return idx, name, None
    loop_name = pick_system_audio([name for _, name in audio])
    if loop_name is not None:
        idx = next((i for i, n in audio if n == loop_name), None)
        return idx, loop_name, None
    return None, None, (
        "no loopback device (install BlackHole or Loopback and route output "
        "through it)"
    )


def probe_screen_indices(
    ffmpeg_path: str, platform: str = sys.platform
) -> List[str]:
    """Return the avfoundation "Capture screen N" device indices, in order.

    Empty on non-macOS or when probing fails.  Used to map monitor ordinals to
    real avfoundation video device indices (which sit after the cameras).
    """
    if not is_macos(platform):
        return []
    import subprocess  # local import: only needed at probe time

    try:
        proc = subprocess.run(
            [ffmpeg_path, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=10,
        )
        text = (proc.stderr or "") + (proc.stdout or "")
    except Exception as exc:  # noqa: BLE001
        log.warning("screen-index probe failed: %s", exc)
        return []
    return capture_screen_indices(parse_avfoundation_devices(text)["video"])


def _is_loopback(name: str) -> bool:
    low = name.lower()
    return any(hint in low for hint in _LOOPBACK_HINTS)


def list_audio_devices(
    ffmpeg_path: Optional[str], platform: str = sys.platform
) -> List[str]:
    """Return the audio capture device names for the settings UI.

    Best-effort: returns ``[]`` when ffmpeg is unavailable or probing fails so
    the caller can fall back to the "Automatic" option.
    """
    if not ffmpeg_path:
        return []
    import subprocess  # local import: only needed when populating settings

    def _list(args: List[str]) -> str:
        try:
            proc = subprocess.run(
                [ffmpeg_path, *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            return (proc.stderr or "") + (proc.stdout or "")
        except Exception as exc:  # noqa: BLE001
            log.warning("audio device list failed: %s", exc)
            return ""

    if is_macos(platform):
        text = _list(["-f", "avfoundation", "-list_devices", "true", "-i", ""])
        return [name for _, name in parse_avfoundation_devices(text)["audio"]]
    if is_windows(platform):
        text = _list(["-f", "dshow", "-list_devices", "true", "-i", "dummy"])
        return parse_dshow_audio_devices(text)
    return []


try:  # Qt is optional for importing the pure helpers (e.g. in unit tests).
    from PySide6.QtCore import QObject, QProcess, QTimer, Signal

    class ScreenRecorderService(QObject):
        """Lifecycle controller for a segmented ffmpeg screen recording."""

        state_changed = Signal(object)   # RecorderState
        elapsed_changed = Signal(int)    # whole seconds, excluding pauses
        error = Signal(str)
        audio_level = Signal(float)      # live mix peak level in dBFS
        audio_validated = Signal(object) # AudioValidation for the final mp4
        video_validated = Signal(object) # VideoValidation for the final mp4
        preflight_result = Signal(object)  # BlackProbe for the chosen backend
        backend_changed = Signal(str)    # a black grabber forced a switch

        def __init__(self, ffmpeg_path: Optional[str], parent=None) -> None:
            super().__init__(parent)
            self._ffmpeg = ffmpeg_path
            self._ffprobe = resolve_ffprobe(ffmpeg_path)
            self._meter_buf = ""             # partial metering stdout line
            self._last_validation: Optional[AudioValidation] = None
            self._last_video_validation: Optional[VideoValidation] = None
            # Pre-flight bookkeeping: which grabber we are trying, which ones we
            # already ruled out, and the probe's own process/stderr.
            self._active_backend: Optional[str] = None
            self._tried_backends: List[str] = []
            self._probe_proc: Optional[QProcess] = None
            self._probe_stderr = ""
            self._probe_seconds = 1.0
            self._preflight = True
            self._gdigrab_scale_probed = False
            self._state = RecorderState.IDLE
            self._proc: Optional[QProcess] = None
            self._output_dir: Optional[Path] = None
            self._devices: Optional[CaptureDevices] = None
            self._options = RecordingOptions()
            self._region: Optional[CaptureRegion] = None
            self._concat_on_stop = True
            self._segment_index = 0
            self._elapsed_seconds = 0
            self._timer = QTimer(self)
            self._timer.setInterval(1000)
            self._timer.timeout.connect(self._on_tick)
            self._stopping = False
            # Tail of the current ffmpeg's stderr, kept so a failure can report
            # the real reason instead of a generic "exited unexpectedly".
            self._stderr_tail = ""
            self._requested_backend = "auto"

        # -- public API -------------------------------------------------

        @property
        def state(self) -> RecorderState:
            return self._state

        @property
        def elapsed_seconds(self) -> int:
            return self._elapsed_seconds

        @property
        def output_dir(self) -> Optional[Path]:
            return self._output_dir

        @property
        def last_validation(self) -> Optional["AudioValidation"]:
            return self._last_validation

        @property
        def last_video_validation(self) -> Optional["VideoValidation"]:
            return self._last_video_validation

        @property
        def active_backend(self) -> Optional[str]:
            """The Windows grabber actually in use, once resolved."""
            return self._active_backend

        def start(
            self,
            output_dir: Path,
            devices: CaptureDevices,
            options: RecordingOptions,
            concat_on_stop: bool = True,
            region: Optional[CaptureRegion] = None,
            preflight: bool = True,
        ) -> None:
            if self._state in (
                RecorderState.PREFLIGHT,
                RecorderState.RECORDING,
                RecorderState.PAUSED,
            ):
                return
            self._output_dir = Path(output_dir)
            self._output_dir.mkdir(parents=True, exist_ok=True)
            self._devices = devices
            self._options = options
            self._requested_backend = options.windows_backend
            self._region = region
            self._concat_on_stop = concat_on_stop
            self._segment_index = 0
            self._elapsed_seconds = 0
            self._stopping = False
            self._meter_buf = ""
            self._last_validation = None
            self._last_video_validation = None
            self._tried_backends = []
            self._preflight = preflight
            self._active_backend = (
                resolve_windows_backend(options.windows_backend, region).value
                if is_windows(sys.platform)
                else None
            )
            # Surface exactly which audio sources were resolved (and why any are
            # missing) so a silent recording is never a mystery.
            for line in audio_diagnostics(devices, options, sys.platform):
                log.info(line)
            # On Windows, prove the grabber actually produces a picture before
            # committing to it — a black gdigrab/ddagrab is the #177 failure and
            # is invisible until the recording is over.
            if preflight and is_windows(sys.platform) and self._active_backend:
                self._set_state(RecorderState.PREFLIGHT)
                self._run_preflight(self._active_backend)
                return
            self._begin_capture()

        def _begin_capture(self) -> None:
            """Spawn the real capture and start the elapsed clock."""
            self._spawn_ffmpeg()
            if self._state is RecorderState.RECORDING:
                self._timer.start()

        # -- pre-flight black-frame probe -------------------------------

        def _run_preflight(self, backend: str) -> None:
            """Capture ~1 s through ``blackdetect`` with *backend*."""
            if not self._ffmpeg:
                self._begin_capture()
                return
            self._tried_backends.append(backend)
            self._probe_stderr = ""
            options = RecordingOptions(**vars(self._options))
            options.windows_backend = backend
            args = build_preflight_args(
                sys.platform,
                options,
                self._region,
                backend=backend,
                probe_seconds=self._probe_seconds,
            )
            proc = QProcess(self)
            proc.setProgram(self._ffmpeg)
            proc.setArguments(args)
            proc.setProcessChannelMode(QProcess.SeparateChannels)
            proc.readyReadStandardError.connect(self._on_probe_stderr)
            proc.finished.connect(self._on_preflight_done)
            log.info("recorder: pre-flight %s probe: %s", backend, " ".join(args))
            proc.start()
            if not proc.waitForStarted(3000):
                log.warning("recorder: pre-flight probe failed to start")
                self._probe_proc = None
                self._begin_capture()
                return
            self._probe_proc = proc
            # Watchdog: a wedged grabber must not hang the start button.
            QTimer.singleShot(10000, self._kill_stale_probe)

        def _kill_stale_probe(self) -> None:
            proc = self._probe_proc
            if proc is None or self._state is not RecorderState.PREFLIGHT:
                return
            log.warning("recorder: pre-flight probe timed out; killing it")
            proc.kill()

        def _on_probe_stderr(self) -> None:
            proc = self.sender()
            if proc is None:
                return
            try:
                self._probe_stderr += bytes(proc.readAllStandardError()).decode(
                    "utf-8", errors="replace"
                )
            except Exception:  # noqa: BLE001
                pass

        def _on_preflight_done(self, code, _status) -> None:
            if self._state is not RecorderState.PREFLIGHT:
                return  # stopped while probing
            self._probe_proc = None
            backend = self._tried_backends[-1]
            result = evaluate_black_probe(
                self._probe_stderr,
                backend=backend,
                probe_seconds=self._probe_seconds,
                exit_code=code,
            )
            if result.ok and not result.is_black:
                log.info("recorder: %s", result.summary())
                self._active_backend = backend
                self._options.windows_backend = backend
                self.preflight_result.emit(result)
                self._begin_capture()
                return

            log.warning("recorder: %s", result.summary())
            other = self._other_backend(backend)
            auto = (
                WindowsCaptureBackend.from_value(self._config_backend())
                is WindowsCaptureBackend.AUTO
            )
            if auto and other is not None and other not in self._tried_backends:
                log.warning("recorder: falling back to the %s grabber", other)
                self.backend_changed.emit(other)
                self._run_preflight(other)
                return
            # Both grabbers look bad (or the user pinned one).  Record anyway —
            # a false positive here must never block the user — but say so.
            self._active_backend = backend
            self._options.windows_backend = backend
            self.preflight_result.emit(result)
            self._begin_capture()

        def _config_backend(self) -> str:
            """The backend the user configured, before any fallback."""
            return self._requested_backend or "auto"

        def _other_backend(self, backend: str) -> Optional[str]:
            if not self._region or not self._region.dda_targets:
                return None  # ddagrab cannot address this region at all
            if backend == WindowsCaptureBackend.DDAGRAB.value:
                return WindowsCaptureBackend.GDIGRAB.value
            if backend == WindowsCaptureBackend.GDIGRAB.value:
                return WindowsCaptureBackend.DDAGRAB.value
            return None

        def pause(self) -> None:
            if self._state is not RecorderState.RECORDING:
                return
            self._timer.stop()
            self._set_state(RecorderState.PAUSED)
            self._terminate_proc()

        def resume(self) -> None:
            if self._state is not RecorderState.PAUSED:
                return
            self._spawn_ffmpeg()
            if self._state is RecorderState.RECORDING:
                self._timer.start()

        def stop(self) -> Optional[Path]:
            """Stop recording, optionally concatenate, return final mp4 path."""
            if self._state in (RecorderState.IDLE, RecorderState.FINALIZING):
                return None
            if self._state is RecorderState.PREFLIGHT:
                # Nothing was captured yet — abandon the probe and go quiet.
                self._set_state(RecorderState.IDLE)
                if self._probe_proc is not None:
                    self._probe_proc.kill()
                    self._probe_proc = None
                return None
            self._timer.stop()
            self._stopping = True
            self._set_state(RecorderState.FINALIZING)
            self._terminate_proc()
            final = None
            if self._concat_on_stop:
                final = self._concatenate_segments()
            # Validate that the produced file actually carries audio; if not,
            # log at ERROR so a silent recording is loud in the logs/UI.
            self._validate_audio(final)
            self._validate_video(final)
            self._set_state(RecorderState.IDLE)
            self._stopping = False
            return final

        def _validate_audio(self, final: Optional[Path]) -> None:
            """Probe the final mp4 (or first segment) for a usable audio track."""
            target = final
            if target is None:
                segs = self.segment_paths()
                target = segs[0] if segs else None
            if target is None:
                return
            # No audio sources were requested → nothing to validate.
            opts = self._options
            dev = self._devices
            wanted_audio = dev is not None and (
                (dev.microphone is not None and not opts.mute_microphone)
                or (dev.system_audio is not None and not opts.mute_system_audio)
            )
            result = validate_recording_audio(self._ffprobe, target)
            self._last_validation = result
            if result.error:
                log.warning("recorder: %s", result.summary())
            elif not result.has_audio and wanted_audio:
                log.error("recorder: %s", result.summary())
            else:
                log.info("recorder: %s", result.summary())
            self.audio_validated.emit(result)

        def _validate_video(self, final: Optional[Path]) -> None:
            """Probe the final mp4 (or first segment) for an all-black picture."""
            target = final
            if target is None:
                segs = self.segment_paths()
                target = segs[0] if segs else None
            if target is None:
                return
            result = validate_recording_video(self._ffmpeg, self._ffprobe, target)
            self._last_video_validation = result
            if result.error:
                log.warning("recorder: %s", result.summary())
            elif result.is_black:
                log.error("recorder: %s", result.summary())
            else:
                log.info("recorder: %s", result.summary())
            self.video_validated.emit(result)

        def segment_paths(self) -> List[Path]:
            if self._output_dir is None:
                return []
            return sorted(self._output_dir.glob("seg_*.mp4"))

        # -- internals --------------------------------------------------

        def _segment_pattern(self) -> str:
            assert self._output_dir is not None
            # Continue numbering across pause/resume so concat order is right.
            return str(self._output_dir / f"seg_{self._segment_index:05d}_%05d.mp4")

        def _spawn_ffmpeg(self) -> None:
            if not self._ffmpeg:
                self._fail("ffmpeg not available")
                return
            assert self._devices is not None and self._output_dir is not None
            try:
                args = build_ffmpeg_args(
                    sys.platform,
                    self._devices,
                    self._options,
                    self._segment_pattern(),
                    self._region,
                )
            except ValueError as exc:
                self._fail(str(exc))
                return

            self._segment_index += 1
            self._stderr_tail = ""
            proc = QProcess(self)
            proc.setProgram(self._ffmpeg)
            proc.setArguments(args)
            proc.setProcessChannelMode(QProcess.SeparateChannels)
            proc.readyReadStandardError.connect(self._on_proc_stderr)
            if self._options.meter_audio:
                proc.readyReadStandardOutput.connect(self._on_proc_stdout)
            proc.errorOccurred.connect(self._on_proc_error)
            proc.finished.connect(self._on_proc_finished)
            proc.start()
            if not proc.waitForStarted(3000):
                self._fail("ffmpeg failed to start")
                return
            self._proc = proc
            self._set_state(RecorderState.RECORDING)
            log.info("recording started: %s %s", self._ffmpeg, " ".join(args))

        def _on_proc_stderr(self) -> None:
            proc = self.sender()
            if proc is None:
                return
            try:
                chunk = bytes(proc.readAllStandardError()).decode(
                    "utf-8", errors="replace"
                )
            except Exception:  # noqa: BLE001
                return
            if not chunk:
                return
            # Keep only the last ~4 KB so a long-running capture stays bounded.
            self._stderr_tail = (self._stderr_tail + chunk)[-4096:]
            text = chunk.rstrip()
            # Grabber complaints are the difference between "black video" and a
            # diagnosable failure, so they must not hide at DEBUG level.
            if any(word in text for word in ("gdigrab", "ddagrab", "d3d11")):
                log.warning("ffmpeg: %s", text)
            else:
                log.debug("ffmpeg: %s", text)

        def _on_proc_stdout(self) -> None:
            """Parse the metering tap on ffmpeg's stdout → ``audio_level``."""
            proc = self.sender()
            if proc is None:
                return
            try:
                chunk = bytes(proc.readAllStandardOutput()).decode(
                    "utf-8", errors="replace"
                )
            except Exception:  # noqa: BLE001
                return
            if not chunk:
                return
            # Buffer until we have whole lines (the peak prints ~10×/s).
            buf = self._meter_buf + chunk
            buf, _, tail = buf.rpartition("\n")
            self._meter_buf = tail[-512:]
            if not buf:
                return
            peak = parse_meter_peak_db(buf)
            if peak is not None:
                self.audio_level.emit(peak)

        def _terminate_proc(self) -> None:
            proc = self._proc
            if proc is None:
                return
            self._proc = None
            try:
                # Ask ffmpeg to quit cleanly so it finalizes the open segment.
                proc.write(b"q")
                proc.closeWriteChannel()
                if not proc.waitForFinished(5000):
                    proc.terminate()
                    if not proc.waitForFinished(3000):
                        proc.kill()
                        proc.waitForFinished(2000)
            except Exception:  # noqa: BLE001
                proc.kill()

        def _concatenate_segments(self) -> Optional[Path]:
            segs = self.segment_paths()
            if not segs or self._output_dir is None or not self._ffmpeg:
                return None
            list_file = self._output_dir / "segments.txt"
            out_file = self._output_dir / "recording.mp4"
            try:
                list_file.write_text(build_concat_list(segs), encoding="utf-8")
            except OSError as exc:
                log.warning("could not write concat list: %s", exc)
                return None
            proc = QProcess(self)
            proc.setProgram(self._ffmpeg)
            proc.setArguments(build_concat_args(list_file, out_file))
            proc.start()
            if not proc.waitForFinished(60000):
                log.warning("concat timed out; segments are preserved")
                return None
            if proc.exitCode() != 0 or not out_file.exists():
                log.warning("concat failed (code %s); segments preserved", proc.exitCode())
                return None
            return out_file

        def _on_tick(self) -> None:
            self._elapsed_seconds += 1
            self.elapsed_changed.emit(self._elapsed_seconds)

        def _on_proc_error(self, _err) -> None:
            if self._stopping or self._state is RecorderState.PAUSED:
                return
            self._fail(self._with_stderr("ffmpeg process error"))

        def _on_proc_finished(self, code, _status) -> None:
            # Unexpected exit while we believed we were recording.
            if (
                self._state is RecorderState.RECORDING
                and not self._stopping
                and self._proc is not None
            ):
                self._fail(
                    self._with_stderr(f"ffmpeg exited unexpectedly (code {code})")
                )

        def _with_stderr(self, message: str) -> str:
            """Append the captured ffmpeg stderr tail to *message* if any.

            Without this the UI only ever sees a generic message; the appended
            tail reveals the real cause (e.g. a Screen-Recording permission
            denial or an unavailable capture device).
            """
            tail = self._stderr_tail.strip()
            if not tail:
                return message
            return f"{message}\n\n{tail}"

        def _fail(self, message: str) -> None:
            self._timer.stop()
            log.error("recorder error: %s", message)
            self._set_state(RecorderState.ERROR)
            self.error.emit(message)

        def _set_state(self, state: RecorderState) -> None:
            if state is self._state:
                return
            self._state = state
            self.state_changed.emit(state)

except ImportError:  # pragma: no cover - Qt missing (pure-helper unit tests)
    ScreenRecorderService = None  # type: ignore
