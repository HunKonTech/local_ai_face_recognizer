"""Mouse interaction on the image: clicks, context menu, object markers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import (
    QPoint,
    QRect,
)
from PySide6.QtWidgets import (
    QDialog,
    QMenu,
)

from app.db.database import session_scope
from app.ui.i18n import t

log = logging.getLogger(__name__)


class InteractionMixin:
    """Mouse interaction on the image: clicks, context menu, object markers."""

    # ──────────────────────────────────────────────────────────────────
    # Face click / interaction
    # ──────────────────────────────────────────────────────────────────

    def _on_image_clicked(self, lx: int, ly: int) -> None:
        self._rename_edit.clearFocus()

        if self._full_pixmap is None:
            return

        # Object markers take precedence over faces — they are small, deliberate
        # targets the user explicitly placed.
        obj_hit = self._object_occ_at_label(lx, ly)
        if obj_hit is not None:
            self._select_object_marker(obj_hit)
            return

        orig_x, orig_y = self._label_to_image(lx, ly)
        if orig_x < 0 or orig_y < 0:
            return

        clicked_id = self._hit_test(orig_x, orig_y)
        if clicked_id is None:
            log.debug(
                "Face selection: click at image=(%d,%d) — no face hit, clearing selection",
                orig_x, orig_y,
            )
            self._hide_inline_editor()
            self._selected_face_id = None
            self._redraw_faces()
            self._clear_face_panel()
            return

        log.debug(
            "Face selection: click at image=(%d,%d) → face_id=%d "
            "(prev_selected=%s image_id=%s)",
            orig_x, orig_y, clicked_id, self._selected_face_id, self._current_image_id,
        )
        if self._selected_face_id != clicked_id:
            self._selected_face_id = clicked_id
            self._redraw_faces()
            self._show_face_info(clicked_id)

        entry = next((f for f in self._face_data if f[0] == clicked_id), None)
        person_name = entry[5] if entry else None
        if not person_name:
            self._show_inline_editor(clicked_id)

    def _on_image_right_clicked(self, lx: int, ly: int) -> None:
        orig_x, orig_y = self._label_to_image(lx, ly)
        face_id: Optional[int] = None
        if orig_x >= 0 and orig_y >= 0:
            face_id = self._hit_test(orig_x, orig_y)

        log.debug(
            "Right-click: label=(%d,%d) image=(%d,%d) image_id=%s faces_loaded=%d hit_face=%s",
            lx, ly, orig_x, orig_y,
            self._current_image_id,
            len(self._face_data),
            face_id,
        )

        obj_hit = self._object_occ_at_label(lx, ly)

        menu = QMenu(self)
        assign_action: Optional[object] = None
        edit_action: Optional[object] = None
        delete_action: Optional[object] = None
        ignore_action: Optional[object] = None
        object_open_action: Optional[object] = None
        object_delete_action: Optional[object] = None

        # Object marker actions take precedence (small deliberate targets).
        if obj_hit is not None:
            self._select_object_marker(obj_hit)
            _occ_id, _object_id, obj_name = obj_hit[0], obj_hit[1], obj_hit[4]
            _bbox = obj_hit[5] if len(obj_hit) > 5 else None
            title = menu.addAction(f"📌 {obj_name or '?'}")
            title.setEnabled(False)
            menu.addSeparator()
            object_edit_action = menu.addAction(t("object_ctx_edit"))
            object_frame_action = None
            if _bbox is not None:
                object_frame_action = menu.addAction(t("object_ctx_edit_frame"))
            object_note_action = menu.addAction(t("object_ctx_edit_note"))
            menu.addSeparator()
            object_open_action = menu.addAction(t("object_ctx_open"))
            object_find_action = menu.addAction(t("object_ctx_find_similar"))
            object_delete_action = menu.addAction(t("object_ctx_delete_occurrence"))
            menu.addSeparator()
            object_mark_action = menu.addAction(t("object_ctx_mark_here"))
            global_pos = self._image_label.mapToGlobal(QPoint(lx, ly))
            chosen = menu.exec(global_pos)
            if chosen is object_edit_action:
                self._edit_object_data(_object_id)
            elif object_frame_action is not None and chosen is object_frame_action:
                self._start_interactive_object_edit(_occ_id, _bbox)
            elif chosen is object_note_action:
                self._edit_object_note(_occ_id)
            elif chosen is object_open_action:
                self.object_open_requested.emit(_object_id)
            elif chosen is object_find_action:
                self.object_search_requested.emit(int(_object_id), "library")
            elif chosen is object_delete_action:
                self._delete_object_occurrence(_occ_id)
            elif chosen is object_mark_action:
                self._object_mode_btn.setChecked(True)
            return

        if face_id is not None:
            if self._selected_face_id != face_id:
                self._selected_face_id = face_id
                self._redraw_faces()
                self._show_face_info(face_id)
            self._hide_inline_editor()

            entry = next((f for f in self._face_data if f[0] == face_id), None)
            person_name = entry[5] if entry else None

            title_text = person_name if person_name else t("ibp_ctx_unknown_face")
            title = menu.addAction(f"[ {title_text} ]")
            title.setEnabled(False)
            menu.addSeparator()
            assign_action = menu.addAction(t("ibp_ctx_assign_person"))
            menu.addSeparator()
            edit_action = menu.addAction(t("ibp_ctx_edit_frame"))
            menu.addSeparator()
            delete_action = menu.addAction(t("ibp_ctx_delete"))
            # Permanent ignore only applies to faces not assigned to a named
            # person — the service refuses named-person faces anyway, but we
            # keep the menu honest by hiding the option for them.
            if self._face_is_ignorable(face_id):
                ignore_action = menu.addAction(t("ibp_ctx_ignore_forever"))
            menu.addSeparator()

        manual_mark_action = menu.addAction(t("ibp_ctx_manual_mark"))
        object_mark_action = menu.addAction(t("object_ctx_mark_here"))

        export_face_action: Optional[object] = None
        export_all_action: Optional[object] = None
        if self._current_image_id is not None:
            menu.addSeparator()
            if face_id is not None:
                export_face_action = menu.addAction(t("ibp_ctx_export_this_face"))
            export_all_action = menu.addAction(t("ibp_ctx_export_faces_one"))

        global_pos = self._image_label.mapToGlobal(QPoint(lx, ly))
        chosen = menu.exec(global_pos)
        log.debug(
            "Context menu result: chosen=%s face_id=%s",
            chosen.text() if chosen else None,
            face_id,
        )

        if chosen is assign_action and face_id is not None:
            self._show_inline_editor(face_id)
        elif chosen is edit_action and face_id is not None:
            self._start_interactive_face_edit(face_id)
        elif chosen is delete_action and face_id is not None:
            self._delete_face(face_id)
        elif ignore_action is not None and chosen is ignore_action and face_id is not None:
            self._ignore_face_forever(face_id)
        elif chosen is manual_mark_action:
            self._draw_mode_btn.setChecked(True)
        elif chosen is object_mark_action:
            self._object_mode_btn.setChecked(True)
        elif export_face_action is not None and chosen is export_face_action:
            self._export_faces_as_images(
                [self._current_image_id], face_ids=[face_id]
            )
        elif export_all_action is not None and chosen is export_all_action:
            self._export_faces_as_images([self._current_image_id])

    def _export_faces_as_images(
        self,
        image_ids: List[int],
        face_ids: Optional[List[int]] = None,
    ) -> None:
        """Open the face-image export dialog for *image_ids* (see #175)."""
        from app.ui.helpers.face_image_export import run_face_image_export

        run_face_image_export(image_ids, parent=self, face_ids=face_ids)

    def _delete_object_occurrence(self, occurrence_id: int) -> None:
        """Remove a single object occurrence marker from the current image."""
        from app.services.object_service import ObjectService

        try:
            with session_scope() as session:
                ObjectService(session).remove_occurrence(occurrence_id)
        except Exception:
            log.exception("Failed to delete object occurrence %d", occurrence_id)
            return
        if self._selected_occurrence_id == occurrence_id:
            self._selected_occurrence_id = None
            self._draw_hint.setVisible(False)
        self._refresh_object_markers()

    def _commit_object_bbox(
        self, occurrence_id: int, x: int, y: int, w: int, h: int
    ) -> None:
        """Persist a resized/moved object bounding box."""
        from app.services.object_feature_service import ObjectFeatureService
        from app.services.object_service import ObjectService

        try:
            with session_scope() as session:
                ObjectService(session).update_occurrence_bbox(occurrence_id, x, y, w, h)
                # The frame now covers different pixels, so its cached match
                # features describe the old crop; drop them to be re-extracted.
                ObjectFeatureService(session).invalidate_patch(occurrence_id)
        except Exception:
            log.exception("Failed to update object bbox for occurrence %d", occurrence_id)

    def _edit_object_data(self, object_id: int) -> None:
        """Open the object data sheet (name/description/notes) for editing."""
        from app.ui.dialogs.object_info_dialog import ObjectInfoDialog

        if ObjectInfoDialog(object_id, self).exec() == QDialog.Accepted:
            self._refresh_object_markers()
            self.object_data_changed.emit()

    def _edit_object_note(self, occurrence_id: int) -> None:
        """Edit the per-image note attached to an object occurrence."""
        from PySide6.QtWidgets import QInputDialog

        from app.services.object_service import ObjectService

        current = ""
        try:
            with session_scope() as session:
                for info in ObjectService(session).get_occurrences_for_image(
                    self._current_image_id
                ):
                    if info.occurrence_id == occurrence_id:
                        current = info.note or ""
                        break
        except Exception:
            log.exception("Failed to read occurrence note %d", occurrence_id)
        text, ok = QInputDialog.getMultiLineText(
            self, t("object_ctx_edit_note"), t("object_picker_occ_note"), current
        )
        if not ok:
            return
        try:
            with session_scope() as session:
                ObjectService(session).update_occurrence_note(occurrence_id, text)
        except Exception:
            log.exception("Failed to update occurrence note %d", occurrence_id)
            return
        self._refresh_object_markers()
        self.object_data_changed.emit()

    def _start_interactive_object_edit(
        self, occurrence_id: int, bbox: Tuple[int, int, int, int]
    ) -> None:
        """Enter handle-drag edit mode for an object occurrence's bounding box."""
        if self._full_pixmap is None:
            return
        self._hide_inline_editor()
        self._draw_mode_btn.setChecked(False)
        self._object_mode_btn.setChecked(False)
        self._interactive_edit_occ_id = occurrence_id
        self._interactive_edit_face_id = None

        # Hide the cyan marker for the box being edited so only the handles show.
        self._image_label.set_object_markers(
            [
                (oid, x, y, name, bb)
                for oid, _o, x, y, name, bb in self._object_marker_data
                if oid != occurrence_id
            ]
        )

        if self._orig_img_bgr is not None:
            ih, iw = self._orig_img_bgr.shape[:2]
        else:
            iw = self._full_pixmap.width()
            ih = self._full_pixmap.height()
        self._image_label.set_interactive_edit(tuple(bbox), iw, ih)
        self._image_label.setFocus()
        self._draw_hint.setText(t("ibp_bbox_edit_hint"))
        self._draw_hint.setVisible(True)

    # ──────────────────────────────────────────────────────────────────
    # Draw mode
    # ──────────────────────────────────────────────────────────────────

    def _on_draw_mode_toggled(self, active: bool) -> None:
        if active and self._interactive_edit_face_id is not None:
            # Switching to draw mode cancels interactive edit without restoring
            self._interactive_edit_face_id = None
            self._interactive_edit_orig_bbox = None
        self._image_label.set_draw_mode(active)
        self._draw_hint.setVisible(active)
        if active:
            self._hide_inline_editor()
            self._draw_hint.setText(t("ibp_draw_hint_add"))
        else:
            self._editing_face_id = None
            self._draw_hint.setText(t("ibp_draw_hint_add"))
        if active and self._object_mode_btn.isChecked():
            self._object_mode_btn.setChecked(False)

    def _on_object_mode_toggled(self, active: bool) -> None:
        if active and self._draw_mode_btn.isChecked():
            self._draw_mode_btn.setChecked(False)
        self._image_label.set_object_mode(active)
        self._draw_hint.setVisible(active)
        if active:
            self._hide_inline_editor()
            self._draw_hint.setText(t("object_rect_hint"))

    def _object_image_id(self) -> Optional[int]:
        """Image that owns the object tags — the B&W original when paired.

        Mirrors the rule faces already follow (`_fetch_face_data`), so a tag
        drawn on either side of a deoldified pair shows up in both views and is
        stored only once.
        """
        return self._deol_pair_orig_id or self._current_image_id

    def _on_object_rect_drawn(self, label_rect: QRect) -> None:
        """A rectangle was drawn in object mode → pick object, store bbox."""
        if self._current_image_id is None or self._full_pixmap is None:
            return
        x1, y1 = self._label_to_image(label_rect.left(), label_rect.top())
        if x1 < 0 or y1 < 0:
            return
        x2, y2 = self._label_to_image_clamped(label_rect.right(), label_rect.bottom())
        if x2 < 0 or y2 < 0:
            return
        bx, by = min(x1, x2), min(y1, y2)
        bw, bh = max(1, abs(x2 - x1)), max(1, abs(y2 - y1))

        from app.services.object_service import ObjectService
        from app.ui.dialogs.object_picker_dialog import ObjectPickerDialog

        dlg = ObjectPickerDialog(self)
        if dlg.exec() != QDialog.Accepted or dlg.chosen_object_id is None:
            return
        try:
            with session_scope() as session:
                ObjectService(session).add_occurrence_bbox(
                    dlg.chosen_object_id,
                    self._object_image_id(),
                    bx, by, bw, bh,
                    note=dlg.occurrence_note,
                )
        except Exception:
            log.exception("Failed to add object bbox occurrence in image browser")
            return
        self._refresh_object_markers()
        self._maybe_auto_search_object(dlg.chosen_object_id)

    def _maybe_auto_search_object(self, object_id: int) -> None:
        """Offer to look for the freshly framed object on other images.

        Opt-in, and it only asks the search to start — the run happens in the
        background, so tagging is never interrupted by it.
        """
        from app.app_settings import app_qsettings

        enabled = str(
            app_qsettings().value("object_matching/auto_search_after_tag", "false")
        ).lower() in ("1", "true", "yes")
        if not enabled:
            return
        # "auto" → whole library, low priority, queued behind a running search.
        self.object_search_requested.emit(int(object_id), "auto")

    def refresh_object_markers(self) -> None:
        """Public redraw of the object overlay (after an external change)."""
        self._refresh_object_markers()

    def current_folder_image_ids(self) -> List[int]:
        """Image ids in the folder currently open in the browser."""
        folder = self._current_folder
        if not folder:
            return []
        from sqlalchemy import select

        from app.db.models import Image as _Image

        try:
            with session_scope() as session:
                rows = session.execute(select(_Image.id, _Image.file_path)).all()
        except Exception:
            log.exception("Failed to list the current folder's images")
            return []
        wanted = str(Path(folder))
        return [
            int(image_id)
            for image_id, file_path in rows
            if file_path and str(Path(file_path).parent) == wanted
        ]

    def _refresh_object_markers(self) -> None:
        """Load object occurrences for the current image and overlay them."""
        image_id = self._object_image_id()
        if image_id is None:
            self._object_marker_data = []
            self._selected_occurrence_id = None
            self._image_label.set_object_markers([])
            self._image_label.set_selected_object(None)
            return
        from app.db.models import TaggedObject
        from app.services.object_service import ObjectService

        data: List[tuple] = []
        try:
            with session_scope() as session:
                svc = ObjectService(session)
                for occ in svc.get_occurrences_for_image(image_id):
                    if occ.point_x is None or occ.point_y is None:
                        continue
                    obj = session.get(TaggedObject, occ.object_id)
                    name = obj.name if obj is not None else None
                    bbox = None
                    if (
                        occ.bbox_x is not None and occ.bbox_y is not None
                        and occ.bbox_w is not None and occ.bbox_h is not None
                    ):
                        bbox = (occ.bbox_x, occ.bbox_y, occ.bbox_w, occ.bbox_h)
                    data.append(
                        (occ.occurrence_id, occ.object_id, occ.point_x, occ.point_y, name, bbox)
                    )
        except Exception:
            log.exception("Failed to load object markers")
            data = []
        self._object_marker_data = data
        # Drop a stale selection that no longer exists on this image.
        if self._selected_occurrence_id not in {d[0] for d in data}:
            self._selected_occurrence_id = None
        self._image_label.set_object_markers(
            [(occ_id, x, y, name, bbox) for occ_id, _oid, x, y, name, bbox in data]
        )
        self._image_label.set_selected_object(self._selected_occurrence_id)

    def _object_occ_at_label(self, lx: int, ly: int) -> Optional[tuple]:
        """Return the marker tuple under the label point, or None.

        Bounding-box markers match when the click is inside the box; legacy
        point markers match within a small display-pixel radius.
        """
        if self._full_pixmap is None or not self._object_marker_data:
            return None
        cr = self._image_label.contentsRect()
        lw, lh = cr.width(), cr.height()
        pw, ph = self._full_pixmap.width(), self._full_pixmap.height()
        if pw == 0 or ph == 0:
            return None
        base = min(lw / pw, lh / ph)
        eff = base * self._zoom
        ox = cr.x() + (lw - pw * eff) / 2 + self._pan_x
        oy = cr.y() + (lh - ph * eff) / 2 + self._pan_y
        best = None
        best_metric = None
        for occ_id, object_id, ix, iy, name, bbox in self._object_marker_data:
            if bbox is not None:
                bxx, byy, bww, bhh = bbox
                dx = ox + bxx * eff
                dy = oy + byy * eff
                dw = bww * eff
                dh = bhh * eff
                if dx <= lx <= dx + dw and dy <= ly <= dy + dh:
                    area = dw * dh
                    # Prefer the smallest enclosing box (topmost).
                    if best_metric is None or area < best_metric:
                        best_metric = area
                        best = (occ_id, object_id, ix, iy, name, bbox)
            else:
                dx = ox + ix * eff
                dy = oy + iy * eff
                d = ((dx - lx) ** 2 + (dy - ly) ** 2) ** 0.5
                if d <= 14.0 and (best_metric is None or d < best_metric):
                    best_metric = d
                    best = (occ_id, object_id, ix, iy, name, bbox)
        return best

    def _select_object_marker(self, marker: tuple) -> None:
        """Highlight a clicked object marker and surface its name."""
        occ_id = marker[0]
        name = marker[4]
        self._selected_occurrence_id = occ_id
        self._image_label.set_selected_object(occ_id)
        self._draw_hint.setVisible(True)
        self._draw_hint.setText(f"📌 {name or '?'}")

    def _on_rect_drawn(self, label_rect: QRect) -> None:
        log.debug(
            "pointer up: label_rect=(%d,%d,%d,%d) image_id=%s editing=%s",
            label_rect.x(), label_rect.y(), label_rect.width(), label_rect.height(),
            self._current_image_id, self._editing_face_id,
        )
        coords = self._label_rect_to_image(label_rect)
        if coords is None:
            log.debug("pointer up: coords outside image — draw ignored")
            return
        ix, iy, iw, ih = coords
        log.debug("bbox created in image coords: (%d,%d,%d,%d)", ix, iy, iw, ih)

        if self._current_image_id is None:
            log.debug("pointer up: _current_image_id is None — draw ignored")
            return

        if self._editing_face_id is not None:
            edited_id = self._editing_face_id
            self._editing_face_id = None
            self._draw_hint.setText(t("ibp_draw_hint_add"))
            self._update_face_bbox(edited_id, ix, iy, iw, ih)
            # Reload data, select the edited face, redraw once
            self._fetch_face_data()
            self._selected_face_id = edited_id
            self._redraw_faces()
            self._show_face_info(edited_id)
        else:
            # Always save manual faces to the canonical (non-deoldified) image
            face_img_id = self._deol_pair_orig_id or self._current_image_id
            log.debug("face create requested: image_id=%d bbox=(%d,%d,%d,%d)", face_img_id, ix, iy, iw, ih)
            new_face_id = self._save_manual_face(
                face_img_id, ix, iy, iw, ih
            )
            log.debug("bbox added to state: face_id=%s", new_face_id)
            # Reload the full face list (includes the just-saved face),
            # select it and redraw once — draw mode intentionally stays ON
            # so the user can immediately add more faces without re-clicking.
            self._fetch_face_data()
            log.debug("render annotations count: %d", len(self._face_data))
            if new_face_id is not None:
                self._selected_face_id = new_face_id
                self._redraw_faces()
                self._show_face_info(new_face_id)

    def _start_face_edit(self, face_id: int) -> None:
        self._hide_inline_editor()
        self._editing_face_id = face_id
        self._selected_face_id = face_id
        self._redraw_faces()
        self._show_face_info(face_id)
        self._draw_hint.setText(t("ibp_draw_hint_edit"))
        self._draw_mode_btn.setChecked(True)
