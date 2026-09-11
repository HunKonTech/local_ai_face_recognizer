"""Tests for the black-frame detection and gdigrab coordinate calibration.

These cover the #177 failure mode: a Windows capture that runs happily and
produces an all-black video.  Everything here is pure-function, no ffmpeg.
"""

from __future__ import annotations

from app.services.screen_recorder_service import (
    CaptureRegion,
    DdaTarget,
    RecordingOptions,
    VideoValidation,
    build_black_scan_args,
    build_preflight_args,
    evaluate_black_probe,
    gdigrab_coordinate_scale,
    parse_blackdetect,
    parse_ffprobe_video,
    parse_gdigrab_desktop_rect,
    scale_region_for_gdigrab,
)

GDIGRAB_BANNER = (
    "[gdigrab @ 0000026f6de46d80] Capturing whole desktop as 1680x1050x32 at (0,0)"
)
BLACK_LINE = (
    "[blackdetect @ 000001db] black_start:0 black_end:0.9 black_duration:0.9"
)


# --------------------------------------------------------------- blackdetect

def test_parse_blackdetect_sums_intervals() -> None:
    text = BLACK_LINE + "\n[blackdetect @ 1] black_start:2 black_end:2.5 black_duration:0.5"
    assert parse_blackdetect(text) == (1.4, 2)


def test_parse_blackdetect_without_intervals_is_zero() -> None:
    assert parse_blackdetect("frame=  5 fps=4.0 q=-0.0 Lsize=N/A") == (0.0, 0)


def test_evaluate_black_probe_all_black() -> None:
    probe = evaluate_black_probe(
        BLACK_LINE + "\nframe=   10 fps=10",
        backend="gdigrab", probe_seconds=1.0, exit_code=0,
    )
    assert probe.ok and probe.is_black
    assert "BLACK" in probe.summary()


def test_evaluate_black_probe_mostly_lit_is_not_black() -> None:
    text = "[blackdetect @ 1] black_start:0 black_end:0.2 black_duration:0.2\nframe=10"
    probe = evaluate_black_probe(
        text, backend="ddagrab", probe_seconds=1.0, exit_code=0
    )
    assert probe.ok and not probe.is_black


def test_evaluate_black_probe_without_frames_is_not_ok() -> None:
    probe = evaluate_black_probe(
        "Could not initialize DDA", backend="ddagrab",
        probe_seconds=1.0, exit_code=1,
    )
    assert not probe.ok
    assert probe.error == "Could not initialize DDA"
    assert "failed" in probe.summary()


# ------------------------------------------------------- gdigrab calibration

def test_parse_gdigrab_desktop_rect() -> None:
    assert parse_gdigrab_desktop_rect(GDIGRAB_BANNER) == (0, 0, 1680, 1050)


def test_parse_gdigrab_desktop_rect_without_banner() -> None:
    assert parse_gdigrab_desktop_rect("no banner here") is None


def test_gdigrab_coordinate_scale_identity_when_unscaled() -> None:
    assert gdigrab_coordinate_scale((0, 0, 2560, 1440), (0, 0, 2560, 1440)) == 1.0


def test_gdigrab_coordinate_scale_detects_system_scaling() -> None:
    # A 150 % system scale shrinks ffmpeg's virtualized desktop to 2/3.
    scale = gdigrab_coordinate_scale((0, 0, 1680, 1050), (0, 0, 2520, 1575))
    assert abs(scale - (2 / 3)) < 1e-6


def test_gdigrab_coordinate_scale_rejects_inconsistent_axes() -> None:
    assert gdigrab_coordinate_scale((0, 0, 1680, 1400), (0, 0, 2520, 1575)) == 1.0


def test_gdigrab_coordinate_scale_without_data_is_identity() -> None:
    assert gdigrab_coordinate_scale(None, (0, 0, 2520, 1575)) == 1.0
    assert gdigrab_coordinate_scale((0, 0, 1680, 1050), None) == 1.0


def test_scale_region_for_gdigrab_rounds_to_even() -> None:
    region = CaptureRegion(offset_x=2520, offset_y=0, width=2521, height=1575)
    scaled = scale_region_for_gdigrab(region, 2 / 3)
    assert (scaled.offset_x, scaled.offset_y) == (1680, 0)
    assert scaled.width % 2 == 0 and scaled.height % 2 == 0
    assert scaled.width == 1680 and scaled.height == 1050


def test_scale_region_for_gdigrab_is_a_noop_at_unity() -> None:
    region = CaptureRegion(offset_x=10, offset_y=20, width=100, height=200)
    assert scale_region_for_gdigrab(region, 1.0) is region


def test_scale_region_for_gdigrab_leaves_dda_targets_alone() -> None:
    region = CaptureRegion(
        offset_x=2520, offset_y=0, width=2520, height=1574,
        dda_targets=[DdaTarget(output_idx=1, canvas_x=2520)],
    )
    scaled = scale_region_for_gdigrab(region, 0.5)
    assert scaled.dda_targets[0].canvas_x == 2520


# ------------------------------------------------------------ probe arg build

def test_build_preflight_args_ddagrab() -> None:
    region = CaptureRegion(dda_targets=[DdaTarget(output_idx=0)])
    args = build_preflight_args(
        "win32", RecordingOptions(fps=10), region, backend="ddagrab"
    )
    # blackdetect logs at INFO, so a "warning" level probe would see nothing.
    assert args[args.index("-loglevel") + 1] == "info"
    assert "-init_hw_device" in args
    graph = args[args.index("-filter_complex") + 1]
    assert "ddagrab=output_idx=0" in graph
    assert "blackdetect" in graph
    assert args[-3:] == ["-f", "null", "-"]
    assert "-an" in args


def test_build_preflight_args_gdigrab() -> None:
    region = CaptureRegion(offset_x=0, offset_y=0, width=1680, height=1050)
    args = build_preflight_args(
        "win32", RecordingOptions(fps=10), region, backend="gdigrab"
    )
    assert "gdigrab" in args
    assert args[args.index("-video_size") + 1] == "1680x1050"
    assert "blackdetect" in args[args.index("-vf") + 1]
    # The probe never draws the cursor: a moving pointer would be the only lit
    # pixel on an otherwise black grab.
    assert args[args.index("-draw_mouse") + 1] == "0"


def test_build_black_scan_args_samples_the_head() -> None:
    args = build_black_scan_args("C:/out/recording.mp4", 30)
    assert args[args.index("-t") + 1] == "30"
    assert args.index("-t") < args.index("-i")
    assert "blackdetect" in args[args.index("-vf") + 1]


def test_build_black_scan_args_without_sample_limit() -> None:
    assert "-t" not in build_black_scan_args("C:/out/recording.mp4", None)


# ---------------------------------------------------------- video validation

FFPROBE_VIDEO = """[STREAM]
codec_type=video
codec_name=h264
duration=12.300000
width=1920
height=1080
[/STREAM]
"""


def test_parse_ffprobe_video() -> None:
    result = parse_ffprobe_video(FFPROBE_VIDEO)
    assert result.codec == "h264"
    assert (result.width, result.height) == (1920, 1080)
    assert result.duration == 12.3


def test_parse_ffprobe_video_without_video_stream() -> None:
    result = parse_ffprobe_video("[STREAM]\ncodec_type=audio\n[/STREAM]\n")
    assert result.error == "no video stream"


def test_video_validation_summary_flags_black() -> None:
    summary = VideoValidation(
        is_black=True, black_seconds=12.0, duration=12.3, width=1920, height=1080
    ).summary()
    assert "ALL BLACK" in summary
    assert "1920x1080" in summary


def test_video_validation_summary_reports_ok() -> None:
    summary = VideoValidation(
        is_black=False, duration=12.3, width=1920, height=1080, codec="h264"
    ).summary()
    assert "OK" in summary
