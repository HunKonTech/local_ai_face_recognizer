"""Overlay controls and previous/next image navigation."""

from __future__ import annotations

import logging

from .helpers import (
    _ROLE_TYPE,
)

log = logging.getLogger(__name__)


class NavigationMixin:
    """Overlay controls and previous/next image navigation."""

    # ── Overlay control handlers ──────────────────────────────────────────

    def _on_ov_bbox_check(self, checked: bool) -> None:
        self._show_bboxes = checked
        self._bbox_slider.setEnabled(checked)
        if checked:
            self._bbox_opacity = self._prev_bbox_opacity
        self._redraw_faces()

    def _on_ov_bbox_slider(self, value: int) -> None:
        pct = value / 100.0
        self._bbox_opacity = pct
        if self._show_bboxes:
            self._prev_bbox_opacity = pct
        self._bbox_pct_label.setText(f"{value}%")
        self._redraw_faces()

    def _on_ov_label_check(self, checked: bool) -> None:
        self._show_labels = checked
        self._label_slider.setEnabled(checked)
        if checked:
            self._label_opacity = self._prev_label_opacity
        self._redraw_faces()

    def _on_ov_label_slider(self, value: int) -> None:
        pct = value / 100.0
        self._label_opacity = pct
        if self._show_labels:
            self._prev_label_opacity = pct
        self._label_pct_label.setText(f"{value}%")
        self._redraw_faces()

    def _on_ov_object_check(self, checked: bool) -> None:
        self._show_objects = checked
        self._object_slider.setEnabled(checked)
        if checked:
            self._object_opacity = self._prev_object_opacity
        self._image_label.set_object_overlay(self._show_objects, self._object_opacity)

    def _on_ov_object_slider(self, value: int) -> None:
        pct = value / 100.0
        self._object_opacity = pct
        if self._show_objects:
            self._prev_object_opacity = pct
        self._object_pct_label.setText(f"{value}%")
        self._image_label.set_object_overlay(self._show_objects, self._object_opacity)

    # ── Navigation ───────────────────────────────────────────────────────

    def _navigate_prev(self) -> None:
        self._navigate(-1)

    def _navigate_next(self) -> None:
        self._navigate(+1)

    def _cycle_face_next(self) -> None:
        """Cycle to the next face on the current image (Ctrl+A). Wraps around."""
        if not self._face_data:
            return
        face_ids = [f[0] for f in self._face_data]
        if self._selected_face_id in face_ids:
            current_idx = face_ids.index(self._selected_face_id)
            next_idx = (current_idx + 1) % len(face_ids)
        else:
            next_idx = 0
        next_face_id = face_ids[next_idx]
        if self._selected_face_id == next_face_id:
            return
        self._hide_inline_editor()
        self._selected_face_id = next_face_id
        self._redraw_faces()
        self._show_face_info(next_face_id)

    def _navigate(self, direction: int) -> None:
        current = self._current_tree_item
        if current is None:
            return

        parent = current.parent()
        if parent is None:
            return

        idx = parent.indexOfChild(current)
        next_idx = idx + direction

        # Stay within the same folder if possible
        if 0 <= next_idx < parent.childCount():
            candidate = parent.child(next_idx)
            if candidate.data(0, _ROLE_TYPE) == "image":
                self._select_tree_item(candidate)
                return

        # Cross to next / previous data folder
        data_folders = list(self._iter_data_folders())
        parent_idx = next((i for i, f in enumerate(data_folders) if f is parent), -1)
        next_folder_idx = parent_idx + direction

        if not (0 <= next_folder_idx < len(data_folders)):
            return  # Already at first/last data folder

        next_folder = data_folders[next_folder_idx]

        # Collapse current folder, expand the target folder
        parent.setExpanded(False)
        self._expand_ancestors(next_folder)
        if not next_folder.isExpanded():
            next_folder.setExpanded(True)
            # _on_tree_item_expanded runs synchronously → images loaded now

        # Collect image children (subfolder items may appear first)
        image_children = [
            next_folder.child(i)
            for i in range(next_folder.childCount())
            if next_folder.child(i).data(0, _ROLE_TYPE) == "image"
        ]
        if not image_children:
            return

        candidate = image_children[0] if direction > 0 else image_children[-1]
        self._select_tree_item(candidate)
