"""Person box drawing for :class:`FamilyTreeView`."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItemGroup,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
)

from app.ui.widgets.family_tree.geometry import _elide, _years
from app.ui.widgets.family_tree.style import (
    _ANCESTOR_BORDER,
    _DESCENDANT_BORDER,
    _FEMALE_BORDER,
    _FEMALE_FILL,
    _FOCUS_BORDER,
    _MALE_BORDER,
    _MALE_FILL,
    _ROLE_PID,
    _SELECTED_BORDER,
    _SPOUSE_BORDER,
    _TEXT,
    _UNKNOWN_BORDER,
    _UNKNOWN_FILL,
)


class _NodeBoxMixin:
    """Builds the person boxes and their selection pens."""

    def _pen_for(self, pid: int, border: QColor) -> QPen:
        if pid == self._selected_id:
            pen = QPen(_SELECTED_BORDER)
            pen.setWidth(3)
        elif pid == self._focus_id:
            pen = QPen(_FOCUS_BORDER)
            pen.setWidth(3)
        elif pid in self._ancestors:
            pen = QPen(_ANCESTOR_BORDER)
            pen.setWidth(3)
        elif pid in self._descendants:
            pen = QPen(_DESCENDANT_BORDER)
            pen.setWidth(3)
        elif pid in self._spouses:
            pen = QPen(_SPOUSE_BORDER)
            pen.setWidth(3)
        else:
            pen = QPen(border)
            pen.setWidth(1)
        return pen

    def _add_box(self, node, x, y) -> None:
        if node.gender == "male":
            fill, border = _MALE_FILL, _MALE_BORDER
        elif node.gender == "female":
            fill, border = _FEMALE_FILL, _FEMALE_BORDER
        else:
            fill, border = _UNKNOWN_FILL, _UNKNOWN_BORDER

        group = QGraphicsItemGroup()
        group.setData(_ROLE_PID, node.person_id)

        pid = node.person_id
        box_h = self._box_h.get(pid, self.BOX_H)
        rect = QGraphicsRectItem(0, 0, self.BOX_W, box_h)
        rect.setBrush(QBrush(fill))
        rect.setPen(self._pen_for(pid, border))
        if not node.is_member:
            rect.setOpacity(0.45)
        group.addToGroup(rect)
        self._rect_items[pid] = rect
        self._node_borders[pid] = border

        text_x = 6.0
        if node.thumbnail_path:
            # Reserve the photo slot now; the image itself is loaded lazily when
            # the box scrolls into view (keeps large trees fast to open).
            slot = QGraphicsRectItem(6, (box_h - self.PHOTO) / 2, self.PHOTO, self.PHOTO)
            slot.setBrush(QBrush(QColor("#cfcfcf")))
            slot.setPen(QPen(Qt.NoPen))
            group.addToGroup(slot)
            text_x = 6 + self.PHOTO + 6
            self._pending_photos[pid] = (group, (6.0, (box_h - self.PHOTO) / 2), node.thumbnail_path)

        ty = 8.0
        for text_line, role in self._lines_for(node):
            is_name = role == "name"
            text = QGraphicsSimpleTextItem(_elide(text_line, 22))
            font = text.font()
            font.setPointSize(9 if is_name else 8)
            font.setBold(is_name)
            text.setFont(font)
            text.setBrush(QBrush(_TEXT))
            text.setPos(text_x, ty)
            group.addToGroup(text)
            ty += 18 if is_name else 15

        group.setPos(x, y)
        group.setZValue(1)
        self._scene.addItem(group)
        self._node_items[node.person_id] = group
        if pid in self._collapsible:
            self._add_badge(pid, x, y + box_h)

    # Badge geometry (scene units), centred under the box's bottom edge.
    BADGE = 16

    def _add_badge(self, pid: int, x: float, bottom: float) -> None:
        collapsed = pid in self._collapsed
        badge = QGraphicsItemGroup()
        bx = x + self.BOX_W / 2 - self.BADGE / 2
        circle = QGraphicsEllipseItem(0, 0, self.BADGE, self.BADGE)
        circle.setBrush(QBrush(QColor("#ffffff")))
        circle.setPen(QPen(QColor("#555555"), 1))
        badge.addToGroup(circle)
        sign = QGraphicsSimpleTextItem("+" if collapsed else "\u2212")
        font = sign.font()
        font.setPointSize(9)
        font.setBold(True)
        sign.setFont(font)
        r = sign.boundingRect()
        badge.addToGroup(sign)
        sign.setPos((self.BADGE - r.width()) / 2, (self.BADGE - r.height()) / 2)
        badge.setPos(bx, bottom - self.BADGE / 2)
        badge.setZValue(4)
        badge.setVisible(self._show_markers)
        self._scene.addItem(badge)
        self._badge_items[pid] = badge

    def _lines_for(self, node) -> list[tuple[str, str]]:
        """Box text lines: ``[(text, "name" | "line")]``."""
        if self._line_provider is not None:
            try:
                lines = list(self._line_provider(node))
                if lines:
                    return lines
            except Exception:  # noqa: BLE001 - never break drawing
                import logging

                logging.getLogger(__name__).exception("line provider failed")
        lines = [(node.name, "name")]
        years = _years(node.birth_date, node.death_date)
        if years:
            lines.append((years, "line"))
        if node.family_code:
            lines.append((node.family_code, "line"))
        return lines

    def _content_height(self, node) -> float:
        """Height the node's text lines need (the view enforces ``BOX_H`` min)."""
        lines = self._lines_for(node)
        names = sum(1 for _, role in lines if role == "name")
        return 8 + 18 * names + 15 * (len(lines) - names) + 8
