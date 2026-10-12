"""Tests for lazy (viewport-windowed) crop loading in ClusterPanel."""

from __future__ import annotations

import pytest

from app.ui.panels.cluster_panel import ClusterPanel


class _Loader:
    """Stand-in tile that records whether its crop was decoded."""

    def __init__(self) -> None:
        self.loaded = False

    def ensure_loaded(self) -> None:
        self.loaded = True


@pytest.fixture
def panel(qtbot):
    p = ClusterPanel()
    qtbot.addWidget(p)
    p.resize(420, 300)
    p.show()
    qtbot.waitExposed(p)
    return p


def test_only_visible_rows_load(panel):
    panel._cur_cols = 4
    loaders = [_Loader() for _ in range(200)]
    panel._ordered_thumbs = loaders
    panel._scroll.verticalScrollBar().setValue(0)
    panel._load_visible_thumbs()

    # The very first tile is on screen → loaded.
    assert loaders[0].loaded is True
    # Far-down tiles are well outside the viewport + buffer → still lazy.
    assert loaders[-1].loaded is False
    assert not all(ldr.loaded for ldr in loaders)


def test_widgets_without_loader_are_skipped(panel):
    # A mix of loaders and plain objects must not raise.
    panel._cur_cols = 4
    panel._ordered_thumbs = [_Loader(), object(), _Loader()]
    panel._load_visible_thumbs()        # should not raise
    assert panel._ordered_thumbs[0].loaded is True


def _fake_face(i, person):
    from types import SimpleNamespace as NS

    return NS(
        id=i, crop_path=None, is_low_quality=False, auto_merge_review_status=None,
        is_uncertain_identification=False, person=person, person_id=1,
        identification_note=None, confidence=0.9, detector_backend="x",
        image=NS(file_path=f"/x/{i}.jpg", id=i), quality_score=0.5,
        bbox_x=0, bbox_y=0, bbox_w=10, bbox_h=10,
    )


def test_reflow_keeps_grid_and_tiles_visible(panel, qtbot):
    """#186: reflow runs with the container hidden — it must come back shown."""
    from types import SimpleNamespace as NS

    person = NS(name="P")
    panel.show_person("P", [_fake_face(i, person) for i in range(300)])
    assert panel._grid_widget.isVisible()
    assert all(not w.isHidden() for w in panel._ordered_thumbs)

    panel.resize(900, 300)        # different column count → reflow
    qtbot.wait(10)
    assert panel._grid_widget.isVisible()
    assert all(not w.isHidden() for w in panel._ordered_thumbs)
