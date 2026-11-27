"""Mouse, wheel, keyboard and context-menu handling for :class:`FamilyTreeView`."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt


class _InteractionMixin:
    """Hit-testing and input handlers; listed before ``QGraphicsView`` in the MRO."""

    def _pid_at(self, view_pos) -> Optional[int]:
        # Explicit bounding-box check is more reliable than itemAt() with
        # BspTreeIndex + QGraphicsItemGroup, which can return a neighbour's item.
        scene_pos = self.mapToScene(view_pos)
        sx, sy = scene_pos.x(), scene_pos.y()
        for pid, group in self._node_items.items():
            gx, gy = group.x(), group.y()
            if gx <= sx <= gx + self.BOX_W and gy <= sy <= gy + self.BOX_H:
                return pid
        return None

    def mousePressEvent(self, event) -> None:
        pid = self._pid_at(event.position().toPoint())
        if pid is not None:
            self.person_selected.emit(pid)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        pid = self._pid_at(event.position().toPoint())
        if pid is not None:
            self.person_activated.emit(pid)
            return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event) -> None:
        pid = self._pid_at(event.pos())
        if pid is not None:
            self.person_context_menu.emit(pid, event.globalPos())
            event.accept()
            return
        super().contextMenuEvent(event)

    def wheelEvent(self, event) -> None:
        # Wheel zooms toward the cursor; drag (or scrollbars) pans. Shift+wheel
        # scrolls horizontally for users who prefer that.
        if event.modifiers() & Qt.ShiftModifier:
            super().wheelEvent(event)
            return
        self._zoom(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_in()
        elif key in (Qt.Key_Minus, Qt.Key_Underscore):
            self.zoom_out()
        elif key == Qt.Key_0:
            self.reset_zoom()
        else:
            super().keyPressEvent(event)
