"""Person box drawing for :class:`FamilyTreeView`."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import QGraphicsItemGroup, QGraphicsRectItem, QGraphicsSimpleTextItem

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
        rect = QGraphicsRectItem(0, 0, self.BOX_W, self.BOX_H)
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
            slot = QGraphicsRectItem(6, (self.BOX_H - self.PHOTO) / 2, self.PHOTO, self.PHOTO)
            slot.setBrush(QBrush(QColor("#cfcfcf")))
            slot.setPen(QPen(Qt.NoPen))
            group.addToGroup(slot)
            text_x = 6 + self.PHOTO + 6
            self._pending_photos[pid] = (group, (6.0, (self.BOX_H - self.PHOTO) / 2), node.thumbnail_path)

        lines = [node.name]
        years = _years(node.birth_date, node.death_date)
        if years:
            lines.append(years)
        if node.family_code:
            lines.append(node.family_code)

        ty = 8.0
        for i, line in enumerate(lines):
            text = QGraphicsSimpleTextItem(_elide(line, 22))
            font = text.font()
            font.setPointSize(9 if i == 0 else 8)
            font.setBold(i == 0)
            text.setFont(font)
            text.setBrush(QBrush(_TEXT))
            text.setPos(text_x, ty)
            group.addToGroup(text)
            ty += 18 if i == 0 else 15

        group.setPos(x, y)
        group.setZValue(1)
        self._scene.addItem(group)
        self._node_items[node.person_id] = group
