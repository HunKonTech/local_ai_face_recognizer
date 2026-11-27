"""FamilyTreeView extensions for Tree Studio: line provider, row heights,
collapse badges, person drops and scene export (offscreen)."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QDropEvent, QImage, QMouseEvent, QPixmap

from app.services.family_tree_service import FamilyTreeGraph, FamilyTreeNode
from app.ui.widgets.family_tree import FamilyTreeView
from app.ui.widgets.family_tree.interaction import PERSON_MIME
from app.ui.widgets.family_tree.scene_export import export_scene
from app.ui.widgets.family_tree.style import _SELECTED_BORDER


def _graph(photo=None):
    n = FamilyTreeNode
    nodes = {
        1: n(1, "Grandpa", 0, (), (3,), (2,), thumbnail_path=photo),
        2: n(2, "Grandma", 0, (), (3,), (1,)),
        3: n(3, "Father", 1, (1, 2), (5,), (4,)),
        4: n(4, "Mother", 1, (), (5,), (3,)),
        5: n(5, "Kid", 2, (3, 4)),
    }
    return FamilyTreeGraph(root_id=1, nodes=nodes)


def _view(qtbot, **kw):
    v = FamilyTreeView()
    qtbot.addWidget(v)
    v.resize(800, 600)
    return v


def test_tall_box_sets_band_height_and_next_band_below(qtbot):
    v = _view(qtbot)

    def provider(node):
        lines = [(node.name, "name")]
        if node.person_id == 3:
            lines += [(f"line {i}", "line") for i in range(6)]
        return lines

    v.set_line_provider(provider)
    v.set_graph(_graph())
    tall = v.box_height(3)
    assert tall > v.BOX_H
    assert v.box_height(4) == tall  # same band
    assert v.box_height(1) == v.BOX_H
    gen1_bottom = v._node_items[3].y() + tall
    assert v._node_items[5].y() >= gen1_bottom + v.V_GAP - 0.5
    assert v._rect_items[4].rect().height() == tall


def test_badge_click_emits_collapse_toggled(qtbot):
    v = _view(qtbot)
    v.set_collapse_state({3}, set())
    v.set_graph(_graph())
    v.show()
    badge = v._badge_items[3]
    center = badge.sceneBoundingRect().center()
    view_pt = v.mapFromScene(center)
    with qtbot.waitSignal(v.collapse_toggled, timeout=1000) as sig:
        ev = QMouseEvent(QMouseEvent.MouseButtonPress, QPointF(view_pt), QPointF(view_pt),
                         Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        v.mousePressEvent(ev)
    assert sig.args == [3]


def _drop(v, pid, view_pt):
    mime = QMimeData()
    mime.setData(PERSON_MIME, str(pid).encode("ascii"))
    ev = QDropEvent(QPointF(view_pt), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    v.dropEvent(ev)


def test_drop_on_box_and_on_empty_space(qtbot):
    v = _view(qtbot)
    v.set_graph(_graph())
    v.show()
    received = []
    v.person_dropped.connect(lambda pid, target, pos: received.append((pid, target)))
    box_pt = v.mapFromScene(v._node_items[4].sceneBoundingRect().center())
    _drop(v, 77, box_pt)
    _drop(v, 78, QPoint(2, 2))
    assert received[0] == (77, 4)
    assert received[1] == (78, None)


@pytest.fixture()
def photo(tmp_path):
    path = tmp_path / "p.png"
    pm = QPixmap(40, 40)
    pm.fill(QColor("#3366cc"))
    pm.save(str(path))
    return str(path)


def test_export_png_scale_jpg_white_svg_and_no_selection(qtbot, tmp_path, photo):
    v = _view(qtbot)
    v.set_collapse_state({3}, set())
    v.set_graph(_graph(photo), selected_id=3)
    rect = v.scene().sceneRect()

    png1 = export_scene(v, tmp_path / "a.png", scale=1)
    png2 = export_scene(v, tmp_path / "b.png", scale=2)
    i1, i2 = QImage(str(png1)), QImage(str(png2))
    assert (i2.width(), i2.height()) == (i1.width() * 2, i1.height() * 2)
    assert abs(i1.width() - rect.width()) <= 1

    jpg = QImage(str(export_scene(v, tmp_path / "c.jpg", quality=95)))
    c = jpg.pixelColor(0, 0)
    assert min(c.red(), c.green(), c.blue()) > 240

    svg = (export_scene(v, tmp_path / "d.svg")).read_text(encoding="utf-8")
    assert "Grandpa" in svg and "<text" in svg
    assert "<image" in svg and "base64" in svg

    # Selection restored on screen but absent from the export.
    assert v._selected_id == 3
    sel = _SELECTED_BORDER.name().lower()
    assert sel not in svg.lower()
    # Markers hidden in export by default, still visible on screen.
    assert v._badge_items[3].isVisible()
    assert not v._pending_photos


def test_friend_link_is_dashed_and_not_recoloured(qtbot):
    from app.services.tree_view_service import apply_friends
    from app.ui.widgets.family_tree.style import _FRIEND_LINE

    g = _graph()
    g.friends = {3: (50,)}
    g.friend_nodes = {50: FamilyTreeNode(50, "Pal", 1)}
    apply_friends(g, g, "all")
    v = _view(qtbot)
    v.set_graph(g)
    assert 50 in v._node_items
    (item,) = v._friend_items
    assert item.pen().style() == Qt.DashLine
    v.set_selected(3)
    assert item.pen().style() == Qt.DashLine
    assert item.pen().color() == _FRIEND_LINE
