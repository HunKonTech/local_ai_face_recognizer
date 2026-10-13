"""Face/person selection and face edit actions from preview and cluster panels."""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QMessageBox,
)

from app.db.database import session_scope
from app.db.models import Face, Person
from app.services.identity_service import IdentityService
from app.services.unknown_merge_service import UnknownMergeService
from app.ui.dialogs.rename_dialog import RenameDialog
from app.ui.i18n import t

log = logging.getLogger(__name__)


class FaceActionsMixin:
    """Face/person selection and face edit actions from preview and cluster panels."""

    @Slot(int)
    def _on_face_selected(self, face_id: int) -> None:
        self._current_face_id = face_id

        self._show_face_in_preview(face_id)

        self._remove_face_btn.setEnabled(True)
        self._reassign_btn.setEnabled(True)
        self._notify_recording_context()

    def _show_face_in_preview(self, face_id: int) -> None:
        """Reload the preview image and select *face_id*."""
        with session_scope() as session:
            face = session.get(Face, face_id)
            if face is None:
                log.warning("_show_face_in_preview: face_id=%d not found in DB", face_id)
                return
            _ = face.image
            if face.image:
                for f in face.image.faces:
                    _ = f.person
                log.debug(
                    "_show_face_in_preview: face_id=%d image_id=%d annotations=%d",
                    face_id, face.image.id, len([f for f in face.image.faces if not f.is_excluded]),
                )
            self._preview_panel.show_face(face)
        self._refresh_preview_object_markers()

    @Slot(int)
    def _on_face_accept_auto_merge(self, face_id: int) -> None:
        """Confirm a pending auto-merged face from the face-view context menu."""
        with session_scope() as session:
            UnknownMergeService(session).confirm_auto_merge(face_id)
        if self._current_person_id:
            self._on_person_selected(self._current_person_id)
        self._show_face_in_preview(face_id)
        self._image_browser._reload_current_face_data()

    @Slot(int)
    def _on_face_reject_auto_merge(self, face_id: int) -> None:
        """Send a pending auto-merged face back to an Unknown cluster."""
        with session_scope() as session:
            UnknownMergeService(session).reject_to_unknown(face_id)
        if self._current_person_id:
            self._on_person_selected(self._current_person_id)
        self._show_face_in_preview(face_id)
        self._image_browser._reload_current_face_data()

    @Slot(int)
    def _on_face_move_auto_merge(self, face_id: int) -> None:
        """Re-assign a pending face to another person (clears the pending flag)."""
        self._current_face_id = face_id
        # Reuse the standard reassign flow; reassign_face clears the markers.
        self._on_reassign_face()

    @Slot(int)
    def _on_preview_face_selected(self, face_id: int) -> None:
        """Handle a face click directly in the preview image (not from the thumbnail grid)."""
        self._current_face_id = face_id

        with session_scope() as session:
            face = session.get(Face, face_id)
            if face and face.person_id:
                self._current_person_id = face.person_id

        self._remove_face_btn.setEnabled(True)
        self._reassign_btn.setEnabled(True)
        self._notify_recording_context()

    @Slot(int)
    def _on_preview_face_assign(self, face_id: int) -> None:
        """Handle 'Személyhez adás' from the preview panel."""
        self._on_preview_face_selected(face_id)
        self._on_reassign_face()

    @Slot(int)
    def _on_preview_face_diagnostics(self, face_id: int) -> None:
        """Show the 'why this identity?' developer diagnostics for a face."""
        from app.app_settings import app_qsettings
        from app.services.face_diagnostics_service import FaceDiagnosticsService
        from app.ui.dialogs.face_diagnostics_dialog import FaceDiagnosticsDialog

        exclude_low_quality = app_qsettings().value(
            "face_quality/exclude_low_quality", True, type=bool
        )
        try:
            with session_scope() as session:
                svc = FaceDiagnosticsService(
                    session,
                    config=self._config.recognition,
                    exclude_low_quality=exclude_low_quality,
                )
                diag = svc.explain(face_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("Face diagnostics failed")
            QMessageBox.critical(self, t("error"), t("diag_error", error=exc))
            return

        if diag is None:
            return
        FaceDiagnosticsDialog(diag, parent=self).exec()

    @Slot(int)
    def _on_face_set_thumbnail(self, face_id: int) -> None:
        """Set a face crop as the manual thumbnail for its person."""
        with session_scope() as session:
            face = session.get(Face, face_id)
            if face is None or face.person_id is None:
                return
            try:
                IdentityService(session).set_person_thumbnail(
                    face.person_id, face_id
                )
            except ValueError as exc:
                QMessageBox.warning(
                    self,
                    t("thumbnail_set_error", error="", default="Thumbnail"),
                    t("thumbnail_set_error", error=str(exc)),
                )
                return
        self._refresh_persons()

    @Slot(int)
    def _on_face_clear_thumbnail(self, face_id: int) -> None:
        """Reset a person's thumbnail to automatic selection."""
        with session_scope() as session:
            face = session.get(Face, face_id)
            if face is None or face.person_id is None:
                return
            IdentityService(session).clear_manual_person_thumbnail(face.person_id)
        self._refresh_persons()

    @Slot(int, bool, str)
    def _on_face_uncertainty_change(
        self, face_id: int, is_uncertain: bool, note: str
    ) -> None:
        """Toggle uncertain-identification flag and/or update the note for *face_id*."""
        with session_scope() as session:
            face = session.get(Face, face_id)
            if face is None:
                return
            # Pass note=None to leave unchanged when the signal was emitted by
            # the toggle action (which does not touch the existing note).
            note_arg: str | None = note if note != "" else None
            IdentityService(session).set_face_uncertainty(face_id, is_uncertain, note_arg)
        self._show_face_in_preview(face_id)
        # Refresh the persons panel face grid so the badge updates there too.
        if self._current_person_id is not None:
            self._on_person_selected(self._current_person_id)

    def _on_preview_face_delete(self, face_id: int) -> None:
        """Handle 'Arc törlése' from the preview panel context menu (hard delete)."""
        reply = QMessageBox.question(
            self,
            t("remove_face_title"),
            t("remove_face_msg"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        with session_scope() as session:
            face = session.get(Face, face_id)
            if face is None:
                return
            session.delete(face)

        log.info("Face %d deleted from preview panel.", face_id)
        self._preview_panel.clear()
        if self._current_person_id is not None:
            self._on_person_selected(self._current_person_id)
        self._image_browser._reload_current_face_data()

    def _on_preview_face_not_a_face(self, face_id: int) -> None:
        """'Not a face' from the preview / cluster menu — applies to the whole
        cluster multi-selection when the right-clicked face is part of it."""
        from app.ui.helpers.not_a_face import mark_not_a_face

        selected = self._cluster_panel.selected_face_ids()
        face_ids = selected if face_id in selected else [face_id]
        marked = mark_not_a_face(face_ids, parent=self)
        if not marked:
            return
        self._status_label.setText(t("not_a_face_status", n=marked))
        self._preview_panel.clear()
        self._refresh_persons()
        if self._current_person_id is not None:
            self._on_person_selected(self._current_person_id)
        self._image_browser._reload_current_face_data()

    @Slot(int, int, int, int, int)
    def _on_preview_face_create(
        self, image_id: int, x: int, y: int, w: int, h: int
    ) -> None:
        """Save a manually marked face from the face-recognition preview."""
        from app.db.models import Image
        from app.detectors.base import Detection
        from app.utils.image_utils import load_image_bgr_normalized as load_image_bgr
        from app.utils.image_utils import save_face_crop

        log.debug(
            "face create: image_id=%d bbox=(%d,%d,%d,%d)", image_id, x, y, w, h
        )

        # Load image for clamping bbox to actual dimensions.
        # If the file is temporarily inaccessible we still create the face
        # (unclamped) so the annotation is not silently lost.
        img_path: Optional[str] = None
        img_bgr = None
        with session_scope() as session:
            image = session.get(Image, image_id)
            if image is None:
                log.warning("face create: image_id=%d not found in DB", image_id)
                return
            img_path = image.file_path

        img_bgr = load_image_bgr(img_path) if img_path else None
        if img_bgr is None:
            log.warning(
                "face create: cannot load image file %r — creating face with unclamped bbox",
                img_path,
            )

        # Clamp bbox to image dimensions when the image is available.
        if img_bgr is not None:
            detection = Detection(x=x, y=y, w=w, h=h, confidence=1.0).clamp(
                img_bgr.shape[1], img_bgr.shape[0]
            )
        else:
            detection = Detection(x=x, y=y, w=w, h=h, confidence=1.0)

        # Commit face row first (atomically) so it always lands in the DB.
        new_face_id: Optional[int] = None
        with session_scope() as session:
            face = Face(
                image_id=image_id,
                bbox_x=detection.x,
                bbox_y=detection.y,
                bbox_w=detection.w,
                bbox_h=detection.h,
                confidence=1.0,
                detector_backend="manual",
                crop_path=None,
            )
            session.add(face)
            session.flush()
            new_face_id = face.id
            log.debug("bbox added to state: face_id=%d", new_face_id)

            # Save crop in the same transaction so face_id is available for naming.
            if img_bgr is not None:
                crops_dir = self._config.crops_dir_resolved
                crops_dir.mkdir(parents=True, exist_ok=True)
                try:
                    crop_path = save_face_crop(
                        img_bgr=img_bgr,
                        detection=detection,
                        crops_dir=crops_dir,
                        image_id=image_id,
                        thumbnail_size=self._config.scan.thumbnail_size,
                        face_index=new_face_id,
                    )
                    if crop_path is not None:
                        face.crop_path = str(crop_path)
                except Exception:
                    log.exception(
                        "face create: crop save failed for face_id=%d — face will have no thumbnail",
                        new_face_id,
                    )

        if new_face_id is None:
            log.warning("face create: face_id is None after flush — aborting")
            return

        log.info(
            "Manual face added from preview: image_id=%d face_id=%d bbox=(%d,%d,%d,%d)",
            image_id, new_face_id, detection.x, detection.y, detection.w, detection.h,
        )
        self._current_face_id = new_face_id
        self._show_face_in_preview(new_face_id)
        self._remove_face_btn.setEnabled(True)
        self._reassign_btn.setEnabled(True)
        self._refresh_persons()
        self._image_browser._reload_current_face_data()

    @Slot(int, int, int, int, int)
    def _on_preview_face_bbox_update(
        self, face_id: int, x: int, y: int, w: int, h: int
    ) -> None:
        """Update a face bounding box redrawn in the face-recognition preview."""
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
            if face.image:
                _ = face.image.file_path
            if face.person:
                _ = face.person.name

            crops_dir = self._config.crops_dir_resolved
            crops_dir.mkdir(parents=True, exist_ok=True)

            old_crop_path = face.crop_path
            shared_before = crop_path_is_shared(session, face)
            if shared_before:
                log.warning(
                    "Shared crop path before bbox update: %s",
                    face_debug_state(face, old_crop_path),
                )
            crop_path = save_crop_for_face(
                face,
                crops_dir=crops_dir,
                thumbnail_size=self._config.scan.thumbnail_size,
                img_bgr=img_bgr,
            )
            if crop_path is not None:
                new_crop_path = str(crop_path)

        # Invalidate Qt's pixmap cache so the updated thumbnail is shown immediately.
        if old_crop_path or new_crop_path:
            from PySide6.QtGui import QPixmapCache
            if old_crop_path:
                QPixmapCache.remove(old_crop_path)
            if new_crop_path:
                QPixmapCache.remove(new_crop_path)

        log.info(
            "Face %d bbox updated from preview: bbox=(%d,%d,%d,%d) crop=%s",
            face_id, x, y, w, h, new_crop_path,
        )
        self._current_face_id = face_id
        self._show_face_in_preview(face_id)
        if self._current_person_id is not None:
            self._on_person_selected(self._current_person_id)
            self._show_face_in_preview(face_id)
        self._image_browser._reload_current_face_data()

    # ------------------------------------------------------------------
    # Identity actions
    # ------------------------------------------------------------------

    @Slot()
    def _on_rename(self) -> None:
        if self._current_person_id is None:
            return

        with session_scope() as session:
            person = session.get(Person, self._current_person_id)
            if person is None:
                return
            if person.is_protected:
                QMessageBox.warning(self, t("protected_rename_title"), t("protected_rename_msg"))
                return
            dlg = RenameDialog(person.name, parent=self)
            if dlg.exec() != RenameDialog.Accepted:
                return
            new_name = dlg.new_name()
            if not new_name:
                QMessageBox.warning(self, t("empty_name_title"), t("empty_name_msg"))
                return
            IdentityService(session).rename_person(self._current_person_id, new_name)

        self._refresh_persons()
        self._image_browser._reload_current_face_data()
