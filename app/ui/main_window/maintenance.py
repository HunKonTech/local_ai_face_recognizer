"""Unknown-person cleanup, duplicate/overlap review and identity repair."""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QMessageBox,
)

from app.db.database import session_scope
from app.db.models import Face, Image
from app.services.duplicate_unknown_face_finder import (
    DEFAULT_OVERLAP_SENSITIVITY,
    DuplicateUnknownFaceFinder,
    overlap_sensitivity,
)
from app.services.identity_service import IdentityService
from app.services.unknown_merge_service import UnknownMergeService
from app.ui.dialogs.overlapping_unknown_faces_dialog import (
    OverlappingUnknownFacesDialog,
)
from app.ui.i18n import t
from app.workers.deep_pipeline_worker import (
    MODE_RESCAN,
    MODE_TRAIN,
)

log = logging.getLogger(__name__)


class MaintenanceMixin:
    """Unknown-person cleanup, duplicate/overlap review and identity repair."""

    @Slot()
    def _on_cleanup_empty_unknown_persons(self) -> None:
        if self._pipeline_busy():
            QMessageBox.information(self, t("busy_title"), t("busy_msg"))
            return

        reply = QMessageBox.question(
            self,
            t("cleanup_empty_unknowns_title"),
            t("cleanup_empty_unknowns_msg"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            with session_scope() as session:
                deleted = IdentityService(session).cleanup_empty_unknown_persons()
        except Exception as exc:  # noqa: BLE001
            log.exception("Cleanup of empty Unknown persons failed")
            QMessageBox.critical(self, t("error"), str(exc))
            return

        if deleted:
            self._current_person_id = None
            self._current_face_id = None
            self._refresh_persons()
            self._image_browser._reload_current_face_data()
            self._status_label.setText(
                t("cleanup_empty_unknowns_status", persons=deleted)
            )
        else:
            QMessageBox.information(
                self,
                t("cleanup_empty_unknowns_title"),
                t("cleanup_empty_unknowns_none"),
            )

    def _on_revalidate_face_geometry(self) -> None:
        """Re-validate stored faces against landmark geometry and remove the
        geometrically-impossible, non-manual ones (cards/hands/feet/knots
        detected before the geometry gate existed).  Preview-first: the scan is
        read-only and nothing is deleted without an explicit confirmation."""
        if self._pipeline_busy():
            QMessageBox.information(self, t("busy_title"), t("busy_msg"))
            return

        from app.services.face_geometry_cleanup_service import (
            FaceGeometryCleanupService,
        )

        try:
            with session_scope() as session:
                report = FaceGeometryCleanupService(
                    session, getattr(self._config, "detection", None)
                ).scan()
                scanned = report.scanned
                flagged = report.flagged_count
                droppable_ids = report.droppable_ids
        except Exception as exc:  # noqa: BLE001
            log.exception("Geometry re-validation scan failed")
            QMessageBox.critical(self, t("error"), str(exc))
            return

        if not droppable_ids:
            QMessageBox.information(
                self,
                t("geom_cleanup_title"),
                t("geom_cleanup_none", scanned=scanned, flagged=flagged),
            )
            return

        reply = QMessageBox.question(
            self,
            t("geom_cleanup_title"),
            t(
                "geom_cleanup_confirm",
                n=len(droppable_ids),
                scanned=scanned,
                flagged=flagged,
            ),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            with session_scope() as session:
                deleted = FaceGeometryCleanupService(session).delete_faces(
                    droppable_ids
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("Geometry re-validation delete failed")
            QMessageBox.critical(self, t("error"), str(exc))
            return

        self._current_person_id = None
        self._current_face_id = None
        self._refresh_persons()
        self._image_browser._reload_current_face_data()
        self._status_label.setText(t("geom_cleanup_status", deleted=deleted))

    def _on_retroactive_verify(self) -> None:
        """Re-verify all stored (non-manual) faces via crop-level re-detection.

        Loads each original image and runs the same FaceVerifier gate used
        during normal detection.  Non-confirmed non-faces are deleted after
        the user approves a preview.
        """
        if self._pipeline_busy():
            QMessageBox.information(self, t("busy_title"), t("busy_msg"))
            return

        from app.services.retroactive_verification_service import (
            RetroactiveVerificationService,
        )

        progress_dialog = None
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QProgressDialog

            progress_dialog = QProgressDialog(
                t("retro_verify_title"), None, 0, 0, self
            )
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(500)
            progress_dialog.setValue(0)

            def _progress(current: int, total: int, path: str) -> None:
                if progress_dialog:
                    progress_dialog.setMaximum(total)
                    progress_dialog.setValue(current)
                    progress_dialog.setLabelText(
                        f"{t('retro_verify_title')} ({current}/{total})"
                    )
                from PySide6.QtWidgets import QApplication
                QApplication.processEvents()

            with session_scope() as session:
                svc = RetroactiveVerificationService(
                    session, self._config, progress_cb=_progress
                )
                report = svc.scan()
        except Exception as exc:  # noqa: BLE001
            log.exception("Retroactive face verification scan failed")
            QMessageBox.critical(self, t("error"), str(exc))
            return
        finally:
            if progress_dialog is not None:
                progress_dialog.close()

        if self._config.detection.verification_enabled is False or report.scanned == 0 and report.images_loaded == 0:
            QMessageBox.warning(
                self, t("retro_verify_title"), t("retro_verify_unavailable")
            )
            return

        droppable_ids = report.droppable_ids
        if not droppable_ids:
            QMessageBox.information(
                self,
                t("retro_verify_title"),
                t(
                    "retro_verify_none",
                    scanned=report.scanned,
                    images=report.images_loaded,
                    flagged=report.flagged_count,
                    missing=report.images_missing,
                ),
            )
            return

        reply = QMessageBox.question(
            self,
            t("retro_verify_title"),
            t(
                "retro_verify_confirm",
                scanned=report.scanned,
                images=report.images_loaded,
                n=len(droppable_ids),
                flagged=report.flagged_count,
                missing=report.images_missing,
            ),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            with session_scope() as session:
                deleted = RetroactiveVerificationService(
                    session, self._config
                ).delete_faces(droppable_ids)
        except Exception as exc:  # noqa: BLE001
            log.exception("Retroactive verification delete failed")
            QMessageBox.critical(self, t("error"), str(exc))
            return

        self._current_person_id = None
        self._current_face_id = None
        self._refresh_persons()
        self._image_browser._reload_current_face_data()
        self._status_label.setText(t("retro_verify_status", deleted=deleted))

    @Slot()
    def _on_scan(self) -> None:
        from app.gdrive import preferences as _gprefs
        _prefs = _gprefs.load()

        # Drive mode is ON but the session hasn't connected yet.
        if _prefs.enabled and self._gdrive_session is None:
            QMessageBox.information(
                self, t("gdrive_chip_opening"), t("gdrive_scan_no_session")
            )
            return

        if self._gdrive_session is None and not getattr(self, "_root_folders", None):
            QMessageBox.warning(self, t("no_folder_title"), t("no_folder_msg"))
            return

        if self._pipeline_busy():
            QMessageBox.information(self, t("busy_title"), t("busy_msg"))
            return

        # Folder re-read now runs through the AI (deep) pipeline: scan for new
        # images, detect + embed faces, then deep-recognize.
        self._start_deep_pipeline(MODE_RESCAN)

    def _retrain_after_review(self) -> None:
        """Kick off a background model retrain after a name-suggestion review.

        Silent (no folder/Drive prompts): only the deep model is retrained from
        the now-updated trusted training data, no scan or detection runs.  If a
        pipeline is already busy we skip — the next manual train will pick the
        decisions up anyway.
        """
        if self._pipeline_busy():
            log.info("Skipping post-review retrain: a pipeline is already running")
            return
        drive_active = self._gdrive_session is not None
        if not drive_active and not getattr(self, "_root_folders", None):
            # No corpus configured — nothing to train against yet.
            return
        log.info("Name-suggestion review changed data — retraining the model")
        self._start_deep_pipeline(MODE_TRAIN)

    def _refresh_amerge_btn(self) -> None:
        """Label the auto-merge review action with the pending count."""
        if not hasattr(self, "_amerge_action"):
            return
        try:
            with session_scope() as session:
                n = UnknownMergeService(session).count_pending()
        except Exception:  # noqa: BLE001
            n = 0
        self._amerge_action.setText(
            f"{t('amerge_review_menu')} ({n})" if n else t("amerge_review_menu")
        )

    @Slot()
    def _on_open_amerge_review(self) -> None:
        from app.ui.dialogs.auto_merge_review_dialog import AutoMergeReviewDialog

        dlg = AutoMergeReviewDialog(
            recognition_config=self._config.recognition,
            config=self._config,
            parent=self,
        )
        dlg.applied.connect(self._refresh_persons)
        dlg.applied.connect(self._image_browser._reload_current_face_data)
        dlg.exec()
        self._refresh_persons()
        self._image_browser._reload_current_face_data()
        self._refresh_amerge_btn()

    @Slot()
    def _on_find_overlapping_unknown_faces(self) -> None:
        if self._pipeline_busy():
            QMessageBox.information(self, t("busy_title"), t("busy_msg"))
            return

        preset = self._overlap_sensitivity()
        if preset.key == DEFAULT_OVERLAP_SENSITIVITY:
            # Strict level keeps honouring the configured detection thresholds.
            threshold = self._config.detection.duplicate_unknown_iou_threshold
            containment = self._config.detection.duplicate_unknown_containment_threshold
        else:
            threshold = preset.iou
            containment = preset.containment
        try:
            with session_scope() as session:
                finder = DuplicateUnknownFaceFinder(
                    session,
                    iou_threshold=threshold,
                    containment_threshold=containment,
                    cross_identity=preset.cross_identity,
                )
                matches = finder.find()
                images_examined = finder.images_examined
                same_person_ids = finder.same_person_duplicate_ids
        except Exception as exc:  # noqa: BLE001
            log.exception("Overlapping unknown face search failed")
            QMessageBox.critical(self, t("error"), t("overlap_search_error", error=exc))
            return

        log.info(
            "Átfedő/metsző arckeretek keresése (%s): %d kép vizsgálva, %d találat.",
            preset.key,
            images_examined,
            len(matches),
        )
        self._review_and_delete_overlap_matches(
            matches=matches,
            images_examined=images_examined,
            iou_threshold=threshold,
            containment_threshold=containment,
            same_person_duplicate_ids=same_person_ids,
        )

    def _overlap_sensitivity(self):
        """Sensitivity preset chosen in the maintenance dialog (persisted)."""
        from app.app_settings import app_qsettings

        key = app_qsettings().value(
            "overlap_cleanup/sensitivity", DEFAULT_OVERLAP_SENSITIVITY
        )
        return overlap_sensitivity(str(key))

    @Slot()
    def _on_find_embedding_duplicate_faces(self) -> None:
        """Embedding-based duplicate finder — catches the same face split across
        two different Unknown clusters, which the geometric search misses."""
        if self._pipeline_busy():
            QMessageBox.information(self, t("busy_title"), t("busy_msg"))
            return

        containment = self._config.detection.duplicate_unknown_containment_threshold
        dup_cfg = self._config.intra_image_duplicate
        try:
            with session_scope() as session:
                finder = DuplicateUnknownFaceFinder(
                    session,
                    containment_threshold=containment,
                )
                matches = finder.find_embedding_duplicates(
                    similarity_threshold=dup_cfg.duplicate_similarity,
                    min_overlap=dup_cfg.min_overlap,
                )
                images_examined = finder.images_examined
                same_person_ids = finder.same_person_duplicate_ids
        except Exception as exc:  # noqa: BLE001
            log.exception("Embedding duplicate search failed")
            QMessageBox.critical(self, t("error"), t("overlap_search_error", error=exc))
            return

        log.info(
            "Beágyazás-alapú duplikátum-keresés: %d kép vizsgálva, %d találat.",
            images_examined,
            len(matches),
        )
        self._review_and_delete_overlap_matches(
            matches=matches,
            images_examined=images_examined,
            iou_threshold=self._config.detection.duplicate_unknown_iou_threshold,
            containment_threshold=containment,
            same_person_duplicate_ids=same_person_ids,
        )

    def _review_and_delete_overlap_matches(
        self,
        matches: list,
        images_examined: int,
        iou_threshold: float,
        containment_threshold: float,
        same_person_duplicate_ids: frozenset[int] | set[int] | None = None,
    ) -> None:
        """Shared review-dialog + confirm + delete flow for both the geometric
        and embedding-based overlapping-face finders."""
        self._status_label.setText(
            t("overlap_status_found", images=images_examined, matches=len(matches))
        )

        if not matches:
            QMessageBox.information(
                self,
                t("overlap_no_matches_title"),
                t("overlap_no_matches_msg", images=images_examined),
            )
            return

        dlg = OverlappingUnknownFacesDialog(
            matches=matches,
            images_examined=images_examined,
            parent=self,
        )
        dlg.open_face_requested.connect(self._open_overlap_match_face)
        dlg.exec()
        if not dlg.delete_requested():
            return

        selected_ids = dlg.selected_unknown_face_ids()
        if not selected_ids:
            QMessageBox.information(
                self,
                t("overlap_no_selection_title"),
                t("overlap_no_selection_msg"),
            )
            return

        reply = QMessageBox.question(
            self,
            t("overlap_confirm_title"),
            t("overlap_confirm_msg", n=len(selected_ids)),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        preview_image_id = self._preview_panel.current_image_id
        try:
            with session_scope() as session:
                finder = DuplicateUnknownFaceFinder(
                    session,
                    iou_threshold=iou_threshold,
                    containment_threshold=containment_threshold,
                )
                result = finder.delete_unknown_faces(
                    selected_ids,
                    extra_deletable_ids=same_person_duplicate_ids or (),
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("Overlapping unknown face cleanup failed")
            QMessageBox.critical(self, t("error"), t("overlap_delete_error", error=exc))
            return

        log.info(
            "Átfedő arckeretek törlése: kijelölt=%d, törölt=%d, kihagyott=%d.",
            result.requested,
            result.deleted,
            len(result.missing_or_changed),
        )
        self._status_label.setText(t("overlap_deleted_status", n=result.deleted))
        if result.missing_or_changed:
            QMessageBox.warning(
                self,
                t("warning"),
                t(
                    "overlap_delete_skipped_msg",
                    n=len(result.missing_or_changed),
                ),
            )

        self._refresh_after_overlapping_unknown_delete(
            deleted_image_ids=set(result.image_ids),
            previous_preview_image_id=preview_image_id,
            deleted_face_ids=set(selected_ids),
        )

    def _on_identity_repair_scan(self) -> None:
        if self._pipeline_busy():
            QMessageBox.information(self, t("busy_title"), t("busy_msg"))
            return

        from app.app_settings import app_qsettings
        from app.services.identity_repair_service import IdentityRepairService
        from app.ui.dialogs.identity_repair_dialog import IdentityRepairDialog

        exclude_low_quality = app_qsettings().value(
            "face_quality/exclude_low_quality", True, type=bool
        )
        try:
            with session_scope() as session:
                svc = IdentityRepairService(
                    session,
                    config=self._config.identity_repair,
                    exclude_low_quality=exclude_low_quality,
                )
                candidates = svc.scan()
        except Exception as exc:  # noqa: BLE001
            log.exception("Identity repair scan failed")
            QMessageBox.critical(self, t("error"), t("repair_error", error=exc))
            return

        if not candidates:
            QMessageBox.information(
                self, t("repair_no_matches_title"), t("repair_no_matches_msg")
            )
            return

        dlg = IdentityRepairDialog(candidates=candidates, parent=self)
        dlg.exec()
        if not dlg.merge_requested():
            return

        pairs = dlg.selected_pairs()
        if not pairs:
            return

        reply = QMessageBox.question(
            self,
            t("repair_confirm_title"),
            t("repair_confirm_msg", n=len(pairs)),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            with session_scope() as session:
                svc = IdentityRepairService(
                    session,
                    config=self._config.identity_repair,
                    exclude_low_quality=exclude_low_quality,
                )
                result = svc.apply(pairs)
                # Consolidating fragmented identities can leave face-less
                # Unknown placeholders behind — sweep them in the same session.
                IdentityService(session).cleanup_empty_unknown_persons()
        except Exception as exc:  # noqa: BLE001
            log.exception("Identity repair apply failed")
            QMessageBox.critical(self, t("error"), t("repair_error", error=exc))
            return

        self._status_label.setText(
            t(
                "repair_done_status",
                groups=result.groups_consolidated,
                merged=result.persons_merged_away,
            )
        )
        self._refresh_persons()
        if self._current_person_id is not None:
            self._on_person_selected(self._current_person_id)
        self._image_browser._reload_current_face_data()

    @Slot(int)
    def _open_overlap_match_face(self, known_face_id: int) -> None:
        self._tabs.setCurrentIndex(0)
        with session_scope() as session:
            face = session.get(Face, known_face_id)
            if face is None:
                return
            person_id = face.person_id

        if person_id is not None:
            self._on_person_selected(person_id)
        self._current_face_id = known_face_id
        self._show_face_in_preview(known_face_id)
        self._remove_face_btn.setEnabled(True)
        self._reassign_btn.setEnabled(True)

    def _refresh_after_overlapping_unknown_delete(
        self,
        deleted_image_ids: set[int],
        previous_preview_image_id: Optional[int],
        deleted_face_ids: set[int],
    ) -> None:
        face_to_restore = (
            self._current_face_id
            if self._current_face_id not in deleted_face_ids
            else None
        )
        self._current_face_id = face_to_restore

        self._refresh_persons()
        if self._current_person_id is not None:
            self._on_person_selected(self._current_person_id)
            self._current_face_id = face_to_restore
        self._image_browser._reload_current_face_data()

        if (
            previous_preview_image_id is None
            or previous_preview_image_id not in deleted_image_ids
        ):
            if face_to_restore is not None:
                self._show_face_in_preview(face_to_restore)
            return

        replacement_face_id: Optional[int] = None
        with session_scope() as session:
            image = session.get(Image, previous_preview_image_id)
            if image is not None:
                named = [f for f in image.faces if f.person_id is not None and not f.is_excluded]
                visible = [f for f in image.faces if not f.is_excluded]
                replacement = (named or visible or [None])[0]
                replacement_face_id = replacement.id if replacement is not None else None

        if replacement_face_id is not None:
            self._current_face_id = replacement_face_id
            self._show_face_in_preview(replacement_face_id)
        else:
            self._preview_panel.clear()
