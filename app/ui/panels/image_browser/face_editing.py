"""Face bbox drawing/editing, manual faces, ignore/delete."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from PySide6.QtCore import (
    QThreadPool,
)
from PySide6.QtWidgets import (
    QMessageBox,
)

from app.db.database import session_scope
from app.db.models import Face
from app.ui.i18n import t

from .helpers import (
    _BboxEdit,
)

log = logging.getLogger(__name__)


class FaceEditingMixin:
    """Face bbox drawing/editing, manual faces, ignore/delete."""

    # ──────────────────────────────────────────────────────────────────
    # Interactive bbox editor (handle-based move / resize)
    # ──────────────────────────────────────────────────────────────────

    def _start_interactive_face_edit(self, face_id: int) -> None:
        """Enter interactive handle-drag edit mode for *face_id*."""
        entry = next((f for f in self._face_data if f[0] == face_id), None)
        if entry is None:
            return
        _, bx, by, bw, bh, *_ = entry
        self._hide_inline_editor()
        # Remember whether the user was in manual-mark mode so we can return to
        # it after the edit instead of silently dropping out of it.
        self._draw_mode_before_edit = self._draw_mode_btn.isChecked()
        self._draw_mode_btn.setChecked(False)
        self._interactive_edit_face_id = face_id
        self._interactive_edit_orig_bbox = (bx, by, bw, bh)
        self._selected_face_id = face_id

        # Redraw without baking in the editing face's bbox so only handles show it
        self._redraw_faces(exclude_edit_id=face_id)
        self._show_face_info(face_id)

        if self._orig_img_bgr is not None:
            ih, iw = self._orig_img_bgr.shape[:2]
        else:
            iw = self._full_pixmap.width() if self._full_pixmap else 1
            ih = self._full_pixmap.height() if self._full_pixmap else 1

        self._image_label.set_interactive_edit((bx, by, bw, bh), iw, ih)
        self._image_label.setFocus()   # ensure Esc / Enter key events reach the label
        self._draw_hint.setText(t("ibp_bbox_edit_hint"))
        self._draw_hint.setVisible(True)

    def _on_interactive_bbox_committed(
        self, ix: int, iy: int, iw: int, ih: int
    ) -> None:
        """Called each time the user releases a drag handle with a changed bbox."""
        if self._interactive_edit_occ_id is not None:
            self._commit_object_bbox(self._interactive_edit_occ_id, ix, iy, iw, ih)
            # Keep editing: refresh the handle overlay to the new box.
            if self._orig_img_bgr is not None:
                img_h, img_w = self._orig_img_bgr.shape[:2]
            else:
                img_w = self._full_pixmap.width() if self._full_pixmap else 1
                img_h = self._full_pixmap.height() if self._full_pixmap else 1
            self._image_label.set_interactive_edit((ix, iy, iw, ih), img_w, img_h)
            return
        face_id = self._interactive_edit_face_id
        if face_id is None:
            return
        log.debug(
            "BBox intermediate commit: face_id=%d new_bbox=(%d,%d,%d,%d)",
            face_id, ix, iy, iw, ih,
        )

        # Find old bbox for undo record
        entry = next((f for f in self._face_data if f[0] == face_id), None)
        old_bbox: Tuple[int, int, int, int]
        if entry is not None:
            old_bbox = (entry[1], entry[2], entry[3], entry[4])
        else:
            old_bbox = (ix, iy, iw, ih)

        new_bbox = (ix, iy, iw, ih)
        op_type = getattr(self._image_label, "_last_op_type", "resize")

        # Save to DB
        self._update_face_bbox(face_id, ix, iy, iw, ih)

        # Push to undo stack, clear redo
        if old_bbox != new_bbox:
            self._undo_stack.append(
                _BboxEdit(face_id=face_id, old_bbox=old_bbox, new_bbox=new_bbox, op_type=op_type)
            )
            self._redo_stack.clear()

        # Reload face data and update the handle overlay with the latest bbox
        self._fetch_face_data()
        self._selected_face_id = face_id

        if self._orig_img_bgr is not None:
            img_h, img_w = self._orig_img_bgr.shape[:2]
        else:
            img_w = self._full_pixmap.width() if self._full_pixmap else 1
            img_h = self._full_pixmap.height() if self._full_pixmap else 1

        self._redraw_faces(exclude_edit_id=face_id)
        self._image_label.set_interactive_edit((ix, iy, iw, ih), img_w, img_h)
        self._show_face_info(face_id)

    def _on_interactive_bbox_finalized(
        self, ix: int, iy: int, iw: int, ih: int
    ) -> None:
        """Called when the user clicks outside the box or presses Enter — save and exit."""
        if self._interactive_edit_occ_id is not None:
            occ_id = self._interactive_edit_occ_id
            self._interactive_edit_occ_id = None
            self._draw_hint.setVisible(False)
            self._commit_object_bbox(occ_id, ix, iy, iw, ih)
            self._refresh_object_markers()
            return
        face_id = self._interactive_edit_face_id
        if face_id is None:
            return
        log.info(
            "BBox finalized: face_id=%d final_bbox=(%d,%d,%d,%d)",
            face_id, ix, iy, iw, ih,
        )

        # Grab old bbox before clearing state
        entry = next((f for f in self._face_data if f[0] == face_id), None)
        old_bbox: Tuple[int, int, int, int]
        if entry is not None:
            old_bbox = (entry[1], entry[2], entry[3], entry[4])
        else:
            old_bbox = (ix, iy, iw, ih)

        new_bbox = (ix, iy, iw, ih)
        op_type = getattr(self._image_label, "_last_op_type", "resize")

        # Exit edit state first
        self._interactive_edit_face_id = None
        self._interactive_edit_orig_bbox = None
        self._draw_hint.setVisible(False)

        # Save to DB
        self._update_face_bbox(face_id, ix, iy, iw, ih)

        # Push to undo stack only if bbox actually changed
        if old_bbox != new_bbox:
            self._undo_stack.append(
                _BboxEdit(face_id=face_id, old_bbox=old_bbox, new_bbox=new_bbox, op_type=op_type)
            )
            self._redo_stack.clear()

        # Normal redraw (all faces baked in, no handle overlay)
        self._fetch_face_data()
        self._selected_face_id = face_id
        self._redraw_faces()
        self._show_face_info(face_id)
        self._restore_draw_mode_after_edit()

    def _on_interactive_bbox_cancelled(self) -> None:
        """Called when the user presses Esc or otherwise cancels the edit."""
        if self._interactive_edit_occ_id is not None:
            self._interactive_edit_occ_id = None
            self._draw_hint.setVisible(False)
            self._refresh_object_markers()
            return
        face_id = self._interactive_edit_face_id
        orig_bbox = self._interactive_edit_orig_bbox
        self._interactive_edit_face_id = None
        self._interactive_edit_orig_bbox = None
        self._draw_hint.setVisible(False)

        if face_id is not None and orig_bbox is not None:
            # Restore original bbox
            self._update_face_bbox(face_id, *orig_bbox)
            # Remove any undo entries added during this edit session for this face
            # (since the cancel should restore everything to the start-of-session state)
            self._undo_stack = [e for e in self._undo_stack if e.face_id != face_id]
            self._redo_stack = [e for e in self._redo_stack if e.face_id != face_id]
            self._fetch_face_data()
            self._selected_face_id = face_id
            self._redraw_faces()
            self._show_face_info(face_id)
        self._restore_draw_mode_after_edit()

    def _restore_draw_mode_after_edit(self) -> None:
        """Re-enter manual-mark mode if it was active before the interactive edit.

        Editing a frame temporarily leaves draw mode; without this the user is
        silently dropped out of "Kézi kijelölés" after every edit.
        """
        if not self._draw_mode_before_edit:
            return
        self._draw_mode_before_edit = False
        if not self._draw_mode_btn.isChecked():
            self._draw_mode_btn.setChecked(True)

    def _undo_bbox_edit(self) -> None:
        """Undo the last bbox modification (Ctrl+Z)."""
        if not self._undo_stack:
            return
        entry = self._undo_stack.pop()
        self._redo_stack.append(entry)
        self._apply_bbox_edit(entry.face_id, entry.old_bbox)

    def _redo_bbox_edit(self) -> None:
        """Redo the last undone bbox modification (Ctrl+Y)."""
        if not self._redo_stack:
            return
        entry = self._redo_stack.pop()
        self._undo_stack.append(entry)
        self._apply_bbox_edit(entry.face_id, entry.new_bbox)

    def _apply_bbox_edit(
        self, face_id: int, bbox: Tuple[int, int, int, int]
    ) -> None:
        """Apply *bbox* to *face_id* in DB and refresh the view."""
        self._update_face_bbox(face_id, *bbox)
        self._fetch_face_data()
        self._selected_face_id = face_id

        # If we're in interactive edit mode for this face, update the handle overlay
        if self._interactive_edit_face_id == face_id and self._orig_img_bgr is not None:
            img_h, img_w = self._orig_img_bgr.shape[:2]
            self._redraw_faces(exclude_edit_id=face_id)
            self._image_label.set_interactive_edit(bbox, img_w, img_h)
        else:
            self._redraw_faces()
        self._show_face_info(face_id)

    def _save_manual_face(
        self, img_id: int, x: int, y: int, w: int, h: int
    ) -> Optional[int]:
        from app.detectors.base import Detection
        from app.utils.image_utils import save_face_crop

        detection = Detection(x=x, y=y, w=w, h=h, confidence=1.0)
        crops_dir = None
        thumbnail_size = (128, 128)
        if self._config is not None:
            crops_dir = self._config.crops_dir_resolved
            crops_dir.mkdir(parents=True, exist_ok=True)
            thumbnail_size = self._config.scan.thumbnail_size

        def _persist() -> int:
            with session_scope() as session:
                face = Face(
                    image_id=img_id,
                    bbox_x=x, bbox_y=y, bbox_w=w, bbox_h=h,
                    confidence=1.0,
                    detector_backend="manual",
                    crop_path=None,
                )
                session.add(face)
                session.flush()
                new_id = face.id
                if crops_dir is not None and self._orig_img_bgr is not None:
                    crop_path = save_face_crop(
                        img_bgr=self._orig_img_bgr,
                        detection=detection,
                        crops_dir=crops_dir,
                        image_id=img_id,
                        thumbnail_size=thumbnail_size,
                        face_index=new_id,
                    )
                    if crop_path is not None:
                        face.crop_path = str(crop_path)
                # The embedding is NOT computed here: that needs a model load +
                # inference pass (seconds), which used to freeze the UI on every
                # manual mark.  A ManualFaceEmbedRunnable is started after the
                # save instead, and the person picker's scoring path computes a
                # missing embedding on the fly, so nothing is lost.
                return new_id

        # A background writer (e.g. the startup crop-repair task) can hold
        # SQLite's single write lock; the INSERT then waits out busy_timeout and
        # raises OperationalError("database is locked").  This runs in a Qt slot,
        # so an uncaught error crashes the app — retry a few times, then surface
        # a warning instead of dying.
        from sqlalchemy.exc import OperationalError

        new_id: Optional[int] = None
        last_exc: Optional[OperationalError] = None
        for attempt in range(3):
            try:
                new_id = _persist()
                break
            except OperationalError as exc:
                last_exc = exc
                log.warning(
                    "Manual face save attempt %d/3 hit a locked database: %s",
                    attempt + 1, exc,
                )
        if new_id is None:
            log.error("Manual face save failed after retries: %s", last_exc)
            QMessageBox.warning(
                self,
                t("error"),
                t("ibp_manual_face_save_failed"),
            )
            return None

        # Fill in the embedding off-thread so the face becomes comparable in
        # the person-assign panel without blocking the mark itself.
        if self._config is not None:
            from app.workers.manual_face_worker import ManualFaceEmbedRunnable

            QThreadPool.globalInstance().start(
                ManualFaceEmbedRunnable(new_id, self._config)
            )

        log.info(
            "Manual face saved: image_id=%d face_id=%d bbox=(%d,%d,%d,%d)",
            img_id, new_id, x, y, w, h,
        )
        return new_id

    def _face_is_ignorable(self, face_id: int) -> bool:
        """True when the face may go on the permanent ignore list.

        Unassigned faces and faces of auto-named ("Unknown N") or protected
        persons qualify; faces of manually named people never do.
        """
        try:
            with session_scope() as session:
                face = session.get(Face, face_id)
                if face is None or face.get_embedding() is None:
                    return False
                if face.person is None:
                    return True
                return face.person.is_auto_named or face.person.is_protected
        except Exception:  # noqa: BLE001
            log.exception("Failed to check ignorability of face %d", face_id)
            return False

    def _ignore_face_forever(self, face_id: int) -> None:
        """Add a single face to the permanent ignore list and hide it."""
        reply = QMessageBox.question(
            self,
            t("ignore_person_title"),
            t("ibp_ctx_ignore_confirm"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        if self._inline_editor_face_id == face_id:
            self._hide_inline_editor()
        try:
            with session_scope() as session:
                from app.services.ignored_face_service import IgnoredFaceService
                IgnoredFaceService(session).ignore_face_forever(face_id)
        except ValueError as exc:
            QMessageBox.warning(self, t("ignore_person_title"), str(exc))
            return
        log.info("Face %d ignored forever from image browser", face_id)
        if self._selected_face_id == face_id:
            self._selected_face_id = None
            self._clear_face_panel()
        if self._interactive_edit_face_id == face_id:
            self._interactive_edit_face_id = None
            self._interactive_edit_orig_bbox = None
            self._image_label.cancel_interactive_edit()
            self._draw_hint.setVisible(False)
        self._undo_stack = [e for e in self._undo_stack if e.face_id != face_id]
        self._redo_stack = [e for e in self._redo_stack if e.face_id != face_id]
        self._reload_current_face_data()

    def _delete_face(self, face_id: int) -> None:
        if self._inline_editor_face_id == face_id:
            self._hide_inline_editor()
        with session_scope() as session:
            face = session.get(Face, face_id)
            if face is None:
                return
            session.delete(face)
        log.info("Face %d deleted by user", face_id)
        if self._selected_face_id == face_id:
            self._selected_face_id = None
            self._clear_face_panel()
        if self._editing_face_id == face_id:
            self._editing_face_id = None
            self._draw_mode_btn.setChecked(False)
        if self._interactive_edit_face_id == face_id:
            self._interactive_edit_face_id = None
            self._interactive_edit_orig_bbox = None
            self._image_label.cancel_interactive_edit()
            self._draw_hint.setVisible(False)
        # Remove any undo/redo entries for the deleted face
        self._undo_stack = [e for e in self._undo_stack if e.face_id != face_id]
        self._redo_stack = [e for e in self._redo_stack if e.face_id != face_id]
        self._reload_current_face_data()

    def _update_face_bbox(
        self, face_id: int, x: int, y: int, w: int, h: int
    ) -> None:
        log.info(
            "BBox save: face_id=%d bbox=(%d,%d,%d,%d) "
            "caller_edit_face=%s selected=%s",
            face_id, x, y, w, h,
            self._interactive_edit_face_id, self._selected_face_id,
        )
        old_crop_path: Optional[str] = None
        new_crop_path: Optional[str] = None
        with session_scope() as session:
            from app.detectors.base import Detection
            from app.services.face_crop_service import (
                crop_path_is_shared,
                face_debug_state,
                save_crop_for_face,
            )
            from app.utils.image_utils import load_image_bgr_normalized

            face = session.get(Face, face_id)
            if face is None or face.image is None:
                return
            img_bgr = load_image_bgr_normalized(face.image.file_path)
            if img_bgr is None:
                return
            detection = Detection(x=x, y=y, w=w, h=h, confidence=1.0).clamp(
                img_bgr.shape[1], img_bgr.shape[0]
            )
            face.bbox_x = detection.x
            face.bbox_y = detection.y
            face.bbox_w = detection.w
            face.bbox_h = detection.h
            if face.person:
                _ = face.person.name
            old_crop_path = face.crop_path
            if crop_path_is_shared(session, face):
                log.warning(
                    "Shared crop path before image-browser bbox update: %s",
                    face_debug_state(face, old_crop_path),
                )
            if self._config is not None:
                crops_dir = self._config.crops_dir_resolved
                crops_dir.mkdir(parents=True, exist_ok=True)
                crop_path = save_crop_for_face(
                    face,
                    crops_dir=crops_dir,
                    thumbnail_size=self._config.scan.thumbnail_size,
                    img_bgr=img_bgr,
                )
                if crop_path is not None:
                    new_crop_path = str(crop_path)

        if old_crop_path or new_crop_path:
            from PySide6.QtGui import QPixmapCache
            if old_crop_path:
                QPixmapCache.remove(old_crop_path)
            if new_crop_path:
                QPixmapCache.remove(new_crop_path)
        log.info(
            "Face %d bbox updated to (%d,%d,%d,%d) crop=%s",
            face_id, x, y, w, h, new_crop_path,
        )
