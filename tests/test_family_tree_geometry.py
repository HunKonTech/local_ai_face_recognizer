"""Qt-free tests for the family-tree geometry helpers (no QApplication)."""

from __future__ import annotations

from types import SimpleNamespace

from app.ui.widgets.family_tree.geometry import (
    _bus_tracks,
    _elide,
    _marriage_label,
    _years,
    lineage_sets,
)


def test_bus_tracks_separate_overlapping_groups_in_same_gap():
    ys = _bus_tracks([(100.0, [0.0, 200.0]), (100.0, [50.0, 300.0])], 64)
    assert ys[0] != ys[1]
    assert all(100 < y < 164 for y in ys)


def test_bus_tracks_reuse_track_for_disjoint_groups():
    ys = _bus_tracks([(100.0, [0.0, 50.0]), (100.0, [200.0, 300.0])], 64)
    assert ys[0] == ys[1] == 132.0


def test_bus_tracks_different_gaps_are_independent():
    ys = _bus_tracks([(100.0, [0.0, 200.0]), (250.0, [0.0, 200.0])], 64)
    assert ys == [132.0, 282.0]


def test_marriage_label_formats():
    assert _marriage_label("1984", "1990", "Budapest") == "1984–1990 · Budapest"
    assert _marriage_label("1984", None) == "1984"
    assert _marriage_label(None, "1990") == "–1990"
    assert _marriage_label(None, None, "Pécs") == "Pécs"
    assert _marriage_label(" ", "", "") == ""


def test_years_and_elide():
    assert _years(None, None) == ""
    assert _years("1900", None) == "1900–"
    assert _years("1900", "1980") == "1900–1980"
    assert _elide("short", 22) == "short"
    assert _elide("x" * 30, 22) == "x" * 21 + "…"


def _graph(nodes):
    return SimpleNamespace(nodes=nodes, node=nodes.get)


def test_lineage_sets_walks_ancestors_descendants_and_spouses():
    n = lambda pid, parents=(), children=(), spouses=(): SimpleNamespace(  # noqa: E731
        person_id=pid, parent_ids=list(parents), child_ids=list(children), spouse_ids=list(spouses)
    )
    nodes = {
        1: n(1, children=[3]),
        2: n(2, children=[3]),
        3: n(3, parents=[1, 2], children=[5], spouses=[4]),
        4: n(4, children=[5], spouses=[3]),
        5: n(5, parents=[3, 4]),
    }
    anc, desc, sp = lineage_sets(_graph(nodes), 3)
    assert anc == {1, 2}
    assert desc == {5}
    assert sp == {4}
    assert lineage_sets(_graph(nodes), None) == (set(), set(), set())
    assert lineage_sets(_graph(nodes), 99) == (set(), set(), set())
