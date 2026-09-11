"""Deoldified on-image compare divider in the image browser preview."""

from __future__ import annotations

import numpy as np
import pytest

from app.db.database import init_db
from app.services.deoldified_pairing_service import ComparisonMember
from app.ui.panels.image_browser_panel import ImageBrowserPanel
from app.utils.image_utils import save_image_bgr


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "browser.db")


def test_compare_composite_left_right(db, qtbot):
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)

    h, w = 10, 20
    panel._deol_left_bgr = np.zeros((h, w, 3), dtype=np.uint8)        # black
    panel._deol_right_bgr = np.full((h, w, 3), 255, dtype=np.uint8)   # white

    panel._deol_split = 50
    out = panel._deol_composite()
    assert out is not None
    assert (out[:, :10] == 0).all(), "left half must be the left side"
    assert (out[:, 10:] == 255).all(), "right half must be the right side"

    panel._deol_split = 0
    assert (panel._deol_composite() == 255).all(), "split=0 → all right side"

    panel._deol_split = 100
    assert (panel._deol_composite() == 0).all(), "split=100 → all left side"


def test_ensure_compare_bgr_resizes_right_to_left_shape(db, qtbot, tmp_path):
    left_path = tmp_path / "bw.jpg"
    right_path = tmp_path / "color.jpg"
    save_image_bgr(left_path, np.zeros((10, 20, 3), dtype=np.uint8))
    # Different resolution on the right side — must be resized to match left.
    save_image_bgr(right_path, np.full((20, 40, 3), 200, dtype=np.uint8))

    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    panel._deol_group = [
        ComparisonMember(1, str(left_path), "", True),
        ComparisonMember(2, str(right_path), "(artistic)", False),
    ]
    panel._deol_left_idx = 0
    panel._deol_right_idx = 1

    assert panel._deol_ensure_compare_bgr() is True
    assert panel._deol_left_bgr.shape[:2] == (10, 20)
    assert panel._deol_right_bgr.shape[:2] == (10, 20)


def test_picker_change_recomposes_with_chosen_variant(db, qtbot, tmp_path):
    """Selecting another variant on a side recomposes from that variant's pixels."""
    bw_path = tmp_path / "bw.jpg"
    artistic = tmp_path / "art.jpg"
    stable = tmp_path / "stable.jpg"
    save_image_bgr(bw_path, np.zeros((10, 20, 3), dtype=np.uint8))          # black
    save_image_bgr(artistic, np.full((10, 20, 3), 128, dtype=np.uint8))    # grey
    save_image_bgr(stable, np.full((10, 20, 3), 255, dtype=np.uint8))      # white

    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    panel._deol_group = [
        ComparisonMember(1, str(bw_path), "", True),
        ComparisonMember(2, str(artistic), "(artistic)", False),
        ComparisonMember(3, str(stable), "(stable)", False),
    ]
    panel._deol_left_idx, panel._deol_right_idx = 0, 1
    panel._deol_compare = True
    panel._deol_split = 0  # show the whole right side

    assert panel._deol_ensure_compare_bgr() is True
    panel._orig_img_bgr = panel._deol_composite()
    assert (panel._orig_img_bgr == 128).all(), "right side = artistic (grey)"

    # Switch the right picker to the 'stable' (white) variant.
    panel._on_deol_right_changed(2)
    assert panel._deol_right_idx == 2
    assert (panel._orig_img_bgr == 255).all(), "right side now = stable (white)"

    # Two colorized variants can be compared with each other.
    panel._deol_split = 100  # show the whole left side
    panel._on_deol_left_changed(1)  # left = artistic (grey)
    assert panel._deol_left_idx == 1
    assert (panel._orig_img_bgr == 128).all(), "left side = artistic (grey)"


def test_combos_visible_only_in_compare_with_three_members(db, qtbot, tmp_path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    c = tmp_path / "c.jpg"
    for p in (a, b, c):
        save_image_bgr(p, np.zeros((8, 8, 3), dtype=np.uint8))

    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    panel._deol_group = [
        ComparisonMember(1, str(a), "", True),
        ComparisonMember(2, str(b), "(artistic)", False),
        ComparisonMember(3, str(c), "(stable)", False),
    ]
    panel._populate_deol_combos()
    bar = panel._deoldified_bar  # combos live inside the (otherwise hidden) bar

    # Hidden outside compare mode.
    panel._deol_compare = False
    panel._deol_update_combo_visibility()
    assert panel._deol_left_combo.isVisibleTo(bar) is False

    # Visible in compare mode with 3 members.
    panel._deol_compare = True
    panel._deol_update_combo_visibility()
    assert panel._deol_left_combo.isVisibleTo(bar) is True
    assert panel._deol_left_combo.count() == 3

    # Two members → still hidden even in compare mode (simple 2-image case).
    panel._deol_group = panel._deol_group[:2]
    panel._deol_update_combo_visibility()
    assert panel._deol_left_combo.isVisibleTo(bar) is False


def test_clear_for_new_image_keeps_remembered_mode_and_split(db, qtbot):
    """Switching images drops cached pixels but remembers the chosen mode."""
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)

    panel._deol_mode = "compare"
    panel._deol_split = 30
    panel._deol_compare = True
    panel._deol_left_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    panel._deol_right_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    panel._deol_group = [ComparisonMember(1, "x", "", True)]
    panel._btn_view_compare.setChecked(True)

    panel._deol_clear_for_new_image()

    # Per-image state is cleared …
    assert panel._deol_compare is False
    assert panel._deol_left_bgr is None
    assert panel._deol_right_bgr is None
    assert panel._deol_group == []
    assert panel._btn_view_compare.isChecked() is False
    # … but the remembered choice persists for the next image.
    assert panel._deol_mode == "compare"
    assert panel._deol_split == 30


def test_compare_dragged_updates_split_and_recomposites(db, qtbot):
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)

    panel._deol_left_bgr = np.zeros((10, 20, 3), dtype=np.uint8)
    panel._deol_right_bgr = np.full((10, 20, 3), 255, dtype=np.uint8)
    panel._deol_compare = True
    panel._orig_img_bgr = panel._deol_right_bgr.copy()

    panel._on_compare_dragged(25)
    assert panel._deol_split == 25
    # left quarter (5 px) is the left side, the rest right side
    assert (panel._orig_img_bgr[:, :5] == 0).all()
    assert (panel._orig_img_bgr[:, 5:] == 255).all()


def test_drag_while_not_in_compare_only_records_split(db, qtbot):
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    panel._deol_compare = False
    panel._on_compare_dragged(72)
    assert panel._deol_split == 72


def test_single_colorized_view_falls_back_to_bw_when_variant_missing(db, qtbot, tmp_path):
    """A dead colorized path must not freeze the panel — show B&W instead."""
    bw_path = tmp_path / "bw.jpg"
    save_image_bgr(bw_path, np.zeros((10, 20, 3), dtype=np.uint8))

    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    panel._current_path = str(bw_path)
    panel._deol_pair_orig_id = None  # current tree image IS the B&W original
    panel._deol_pair_color_path = str(tmp_path / "gone-deoldified (artistic).jpg")
    panel._deol_viewing_color = True

    panel._apply_single_view(True, reset_zoom=False)

    assert panel._deol_viewing_color is False
    assert panel._deol_mode == "bw"
    assert panel._btn_view_bw.isChecked() is True
    assert panel._btn_view_color.isChecked() is False
    assert panel._orig_img_bgr is not None


def test_label_compare_divider_x_requires_pixmap(db, qtbot):
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    label = panel._image_label
    label.set_compare_mode(True, 50)
    # No source pixmap yet → no divider position.
    assert label._compare_divider_x() is None
    assert label._compare is True
    label.set_compare_mode(False)
    assert label._compare is False


def test_opened_colorized_file_wins_over_remembered_bw(db, qtbot):
    """Clicking a '-deoldified' file shows it in colour even after choosing B&W."""
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    panel._deol_group = [ComparisonMember(1, "bw.jpg", "", True),
                         ComparisonMember(2, "bw-deoldified.jpg", "deoldified", False)]
    panel._deol_mode = "bw"          # user last looked at black and white
    panel._deol_opened_is_color = True  # but the tree selection is colorized

    panel._deol_apply_remembered_mode()

    assert panel._deol_mode == "color"


def test_opened_bw_file_wins_over_remembered_color(db, qtbot):
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    panel._deol_group = [ComparisonMember(1, "bw.jpg", "", True),
                         ComparisonMember(2, "bw-deoldified.jpg", "deoldified", False)]
    panel._deol_mode = "color"
    panel._deol_opened_is_color = False

    panel._deol_apply_remembered_mode()

    assert panel._deol_mode == "bw"


def test_compare_mode_survives_opening_another_image(db, qtbot, tmp_path):
    """Compare is a property of the pair, so it stays on across navigation."""
    bw_path = tmp_path / "bw.jpg"
    color_path = tmp_path / "bw-deoldified.jpg"
    save_image_bgr(bw_path, np.zeros((10, 20, 3), dtype=np.uint8))
    save_image_bgr(color_path, np.full((10, 20, 3), 255, dtype=np.uint8))

    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    panel._deol_mode = "compare"
    panel._deol_opened_is_color = True
    panel._deol_group = [
        ComparisonMember(1, str(bw_path), "", True),
        ComparisonMember(2, str(color_path), "deoldified", False),
    ]
    panel._deol_left_idx, panel._deol_right_idx = 0, 1

    panel._deol_apply_remembered_mode()

    assert panel._deol_mode == "compare"
    assert panel._deol_compare is True


# ──────────────────────────────────────────────────────────────────────────────
# #182 — the view switcher must never be hidden behind the pairing setting
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def pair_on_disk(db, tmp_path):
    """A real B&W + colorized pair, both on disk and in the database."""
    from app.db.database import session_scope
    from app.db.models import Image

    bw_path = tmp_path / "photo.jpg"
    color_path = tmp_path / "photo-deoldified (stable).jpg"
    save_image_bgr(bw_path, np.zeros((10, 20, 3), dtype=np.uint8))
    save_image_bgr(color_path, np.full((10, 20, 3), 255, dtype=np.uint8))

    with session_scope() as s:
        s.add(Image(file_path=str(bw_path), file_hash="bw", file_mtime=0.0))
        s.add(Image(file_path=str(color_path), file_hash="color", file_mtime=0.0))
    with session_scope() as s:
        bw_id = s.query(Image).filter(Image.file_hash == "bw").first().id
        color_id = s.query(Image).filter(Image.file_hash == "color").first().id
    return {
        "bw_id": bw_id, "bw_path": bw_path,
        "color_id": color_id, "color_path": color_path,
    }


def _set_pairing(monkeypatch, enabled: bool) -> None:
    """Force the 'deoldified/auto_pair' setting for one test."""
    import app.app_settings as app_settings

    real = app_settings.app_qsettings()

    class _Fake:
        def value(self, key, default=None, type=None):  # noqa: A002
            if key == "deoldified/auto_pair":
                return enabled
            return real.value(key, default, type=type)

    monkeypatch.setattr(app_settings, "app_qsettings", lambda: _Fake())


def test_view_bar_visible_with_pairing_setting_off(
    pair_on_disk, qtbot, monkeypatch
):
    """The regression guard for #182: the bar appears even with sharing off."""
    _set_pairing(monkeypatch, False)
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)

    panel._setup_deoldified_pair(
        pair_on_disk["bw_id"], str(pair_on_disk["bw_path"])
    )

    assert len(panel._deol_group) == 2
    assert panel._deoldified_bar.isVisibleTo(panel) is True
    assert panel._btn_view_bw.isVisibleTo(panel._deoldified_bar) is True
    assert panel._btn_view_color.isVisibleTo(panel._deoldified_bar) is True


def test_sync_button_hidden_when_pairing_off(pair_on_disk, qtbot, monkeypatch):
    _set_pairing(monkeypatch, False)
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)

    panel._setup_deoldified_pair(
        pair_on_disk["bw_id"], str(pair_on_disk["bw_path"])
    )

    assert panel._btn_deol_sync.isVisibleTo(panel._deoldified_bar) is False
    assert panel._deol_pair_partner_id is None


def test_sync_button_shown_when_pairing_on(pair_on_disk, qtbot, monkeypatch):
    _set_pairing(monkeypatch, True)
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)

    panel._setup_deoldified_pair(
        pair_on_disk["bw_id"], str(pair_on_disk["bw_path"])
    )

    assert panel._btn_deol_sync.isVisibleTo(panel._deoldified_bar) is True
    assert panel._deol_pair_partner_id == pair_on_disk["color_id"]


def test_face_data_not_rehomed_when_pairing_off(
    pair_on_disk, qtbot, monkeypatch
):
    """Annotations written on the colorized side must stay on that image."""
    _set_pairing(monkeypatch, False)
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    panel._current_image_id = pair_on_disk["color_id"]

    panel._setup_deoldified_pair(
        pair_on_disk["color_id"], str(pair_on_disk["color_path"])
    )

    assert panel._deol_pair_orig_id is None
    assert panel._object_image_id() == pair_on_disk["color_id"]


def test_face_data_rehomed_when_pairing_on(pair_on_disk, qtbot, monkeypatch):
    _set_pairing(monkeypatch, True)
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    panel._current_image_id = pair_on_disk["color_id"]

    panel._setup_deoldified_pair(
        pair_on_disk["color_id"], str(pair_on_disk["color_path"])
    )

    assert panel._deol_pair_orig_id == pair_on_disk["bw_id"]
    assert panel._object_image_id() == pair_on_disk["bw_id"]


def test_bw_toggle_loads_original_pixels_with_pairing_off(
    pair_on_disk, qtbot, monkeypatch
):
    """Switching to B&W works off the cached group path, not a DB lookup."""
    _set_pairing(monkeypatch, False)
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    panel._current_path = str(pair_on_disk["color_path"])
    panel._current_image_id = pair_on_disk["color_id"]

    panel._setup_deoldified_pair(
        pair_on_disk["color_id"], str(pair_on_disk["color_path"])
    )
    assert panel._deol_bw_path == str(pair_on_disk["bw_path"])

    panel._on_deol_view_toggle(False)
    assert panel._deol_viewing_color is False
    assert panel._orig_img_bgr is not None
    assert int(panel._orig_img_bgr.mean()) == 0  # the black B&W image


def test_no_bar_when_sibling_file_is_missing(pair_on_disk, qtbot, monkeypatch):
    _set_pairing(monkeypatch, False)
    pair_on_disk["color_path"].unlink()
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)

    panel._setup_deoldified_pair(
        pair_on_disk["bw_id"], str(pair_on_disk["bw_path"])
    )

    assert panel._deol_group == []
    assert panel._deoldified_bar.isVisibleTo(panel) is False


def test_missing_side_reports_and_offers_repair(db, qtbot, tmp_path):
    """Issue #179: a side that cannot be opened must say so, not sit silent."""
    bw_path = tmp_path / "bw.jpg"
    save_image_bgr(bw_path, np.zeros((10, 20, 3), dtype=np.uint8))
    gone = tmp_path / "missing-deoldified (artistic).jpg"

    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    panel._deol_group = [
        ComparisonMember(1, str(bw_path), "", True),
        ComparisonMember(2, str(gone), "(artistic)", False),
    ]
    panel._deol_bw_path = str(bw_path)
    panel._deol_pair_color_path = str(gone)

    panel._apply_single_view(True, reset_zoom=False)

    # Degrades to the readable side instead of freezing on the old pixels.
    assert panel._deol_viewing_color is False
    assert panel._orig_img_bgr is not None
    assert not panel._btn_deol_fix_paths.isHidden()
    assert panel._deol_lbl.text()


def test_missing_bw_side_reports_instead_of_silent_return(db, qtbot, tmp_path):
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    gone = tmp_path / "nothing.jpg"
    panel._deol_bw_path = str(gone)
    panel._deol_group = [
        ComparisonMember(1, str(gone), "", True),
        ComparisonMember(2, str(gone), "(artistic)", False),
    ]

    panel._apply_single_view(False, reset_zoom=False)

    assert not panel._btn_deol_fix_paths.isHidden()


def test_repair_button_emits_signal(db, qtbot, tmp_path):
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    with qtbot.waitSignal(panel.path_repair_requested, timeout=1000):
        panel._btn_deol_fix_paths.click()


def test_successful_load_clears_the_error_state(db, qtbot, tmp_path):
    bw_path = tmp_path / "bw.jpg"
    color_path = tmp_path / "color.jpg"
    save_image_bgr(bw_path, np.zeros((10, 20, 3), dtype=np.uint8))
    save_image_bgr(color_path, np.full((10, 20, 3), 200, dtype=np.uint8))

    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    panel._deol_bw_path = str(bw_path)
    panel._deol_pair_color_path = str(color_path)
    panel._deol_show_path_error("boom")
    assert not panel._btn_deol_fix_paths.isHidden()

    panel._apply_single_view(True, reset_zoom=False)
    assert panel._btn_deol_fix_paths.isHidden()
    assert panel._deol_viewing_color is True
