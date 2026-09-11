"""Tests for the pure half of the Win32/DXGI monitor probe.

The ctypes calls themselves are not exercised here (they need a real desktop);
what matters for #177 is the join that decides which DXGI output ``ddagrab``
should capture, and that a failed join degrades to "unknown" rather than to a
wrong index.
"""

from __future__ import annotations

from app.services.windows_display_probe import (
    DxgiOutput,
    PhysicalMonitor,
    merge_monitors_with_dxgi,
    virtual_desktop_rect,
)


def monitor(name, x=0, y=0, w=2560, h=1440, primary=False) -> PhysicalMonitor:
    return PhysicalMonitor(
        device_name=name, x=x, y=y, width=w, height=h, is_primary=primary
    )


def output(adapter, index, name, x=0, y=0, w=2560, h=1440) -> DxgiOutput:
    return DxgiOutput(
        adapter_index=adapter, output_index=index, device_name=name,
        x=x, y=y, width=w, height=h,
    )


def test_merge_joins_on_device_name() -> None:
    monitors = [
        monitor(r"\\.\DISPLAY1", primary=True),
        monitor(r"\\.\DISPLAY2", x=2560, w=1920, h=1080),
    ]
    # Deliberately out of order: the name, not the ordinal, must decide.
    outputs = [
        output(0, 0, r"\\.\DISPLAY2", x=2560, w=1920, h=1080),
        output(0, 1, r"\\.\DISPLAY1"),
    ]
    merged = merge_monitors_with_dxgi(monitors, outputs)
    assert [m.dxgi_output_index for m in merged] == [1, 0]
    assert all(m.dxgi_adapter_index == 0 for m in merged)


def test_merge_falls_back_to_matching_rectangle() -> None:
    monitors = [monitor(r"\\.\DISPLAY1")]
    outputs = [output(1, 2, "")]  # DXGI gave us no usable name
    merged = merge_monitors_with_dxgi(monitors, outputs)
    assert (merged[0].dxgi_adapter_index, merged[0].dxgi_output_index) == (1, 2)


def test_merge_falls_back_to_ordinal_when_counts_match() -> None:
    monitors = [monitor(r"\\.\DISPLAY1"), monitor(r"\\.\DISPLAY2", x=2560)]
    outputs = [
        output(0, 0, "other-a", x=100, w=800, h=600),
        output(0, 1, "other-b", x=900, w=800, h=600),
    ]
    merged = merge_monitors_with_dxgi(monitors, outputs)
    assert [m.dxgi_output_index for m in merged] == [0, 1]


def test_merge_leaves_unmatched_monitor_unknown() -> None:
    monitors = [monitor(r"\\.\DISPLAY1"), monitor(r"\\.\DISPLAY2", x=2560)]
    outputs = [output(0, 0, "other", x=100, w=800, h=600)]  # count mismatch
    merged = merge_monitors_with_dxgi(monitors, outputs)
    assert all(m.dxgi_output_index is None for m in merged)


def test_override_wins_over_automatic_match() -> None:
    monitors = [monitor(r"\\.\DISPLAY1")]
    outputs = [output(0, 0, r"\\.\DISPLAY1")]
    merged = merge_monitors_with_dxgi(
        monitors, outputs, {r"\\.\DISPLAY1": "1:3"}
    )
    assert (merged[0].dxgi_adapter_index, merged[0].dxgi_output_index) == (1, 3)


def test_malformed_override_is_ignored() -> None:
    monitors = [monitor(r"\\.\DISPLAY1")]
    outputs = [output(0, 0, r"\\.\DISPLAY1")]
    merged = merge_monitors_with_dxgi(
        monitors, outputs, {r"\\.\DISPLAY1": "not-a-number"}
    )
    assert merged[0].dxgi_output_index == 0


def test_merge_does_not_mutate_its_input() -> None:
    monitors = [monitor(r"\\.\DISPLAY1")]
    merge_monitors_with_dxgi(monitors, [output(0, 4, r"\\.\DISPLAY1")])
    assert monitors[0].dxgi_output_index is None


def test_virtual_desktop_rect_spans_every_monitor() -> None:
    monitors = [
        monitor(r"\\.\DISPLAY1"),
        monitor(r"\\.\DISPLAY2", x=-1920, y=-200, w=1920, h=1080),
    ]
    assert virtual_desktop_rect(monitors) == (-1920, -200, 4480, 1640)


def test_virtual_desktop_rect_without_monitors() -> None:
    assert virtual_desktop_rect([]) is None
