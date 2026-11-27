"""Mouse, wheel, keyboard and context-menu handling for :class:`FamilyTreeView`."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt

# Drag payload: the person id as ASCII digits.
PERSON_MIME = "application/x-facelocal-person-id"


class _InteractionMixin:
    """Hit-testing and input handlers; listed before ``QGraphicsView`` in the MRO."""

    def _pid_at(self, view_pos) -> Optional[int]:
        # Explicit bounding-box check is more reliable than itemAt() with
        # BspTreeIndex + QGraphicsItemGroup, which can return a neighbour's item.
        scene_pos = self.mapToScene(view_pos)
        sx, sy = scene_pos.x(), scene_pos.y()
        for pid, group in self._node_items.items():
            gx, gy = group.x(), group.y()
            if gx <= sx <= gx + self.BOX_W and gy <= sy <= gy + self._box_h.get(pid, self.BOX_H):
                return pid
        return None

    def _badge_at(self, view_pos) -> Optional[int]:
        """Person whose collapse badge is under *view_pos*, if any."""
        if not self._badge_items or not self._show_markers:
            return None
        scene_pos = self.mapToScene(view_pos)
        sx, sy = scene_pos.x(), scene_pos.y()
        size = self.BADGE
        for pid, badge in self._badge_items.items():
            bx, by = badge.x(), badge.y()
            if bx - 2 <= sx <= bx + size + 2 and by - 2 <= sy <= by + size + 2:
                return pid
        return None

    # --- drag and drop of persons (MIME PERSON_MIME) -------------------

    def _dropped_person(self, event) -> Optional[int]:
        mime = event.mimeData()
        if mime is None or not mime.hasFormat(PERSON_MIME):
            return None
        try:
            return int(bytes(mime.data(PERSON_MIME)).decode("ascii"))
        except (ValueError, UnicodeDecodeError):
            return None

    def dragEnterEvent(self, event) -> None:
        if self._dropped_person(event) is not None:
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if self._dropped_person(event) is not None:
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        pid = self._dropped_person(event)
        if pid is None:
            super().dropEvent(event)
            return
        view_pos = event.position().toPoint()
        self.person_dropped.emit(pid, self._pid_at(view_pos), self.mapToScene(view_pos))
        event.acceptProposedAction()

    def mousePressEvent(self, event) -> None:
        badge = self._badge_at(event.position().toPoint())
        if badge is not None and event.button() == Qt.LeftButton:
            self.collapse_toggled.emit(badge)
            event.accept()
            return
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
