"""Object markers, sheets and cross-image object search."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import List, Optional

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QDialog,
    QMessageBox,
)

from app.db.database import session_scope
from app.db.models import Face
from app.ui.i18n import t

log = logging.getLogger(__name__)


class ObjectsMixin:
    """Object markers, sheets and cross-image object search."""

    def _refresh_preview_object_markers(self) -> None:
        """Load object occurrences for the current preview image and render them."""
        image_id = self._preview_panel.current_image_id
        if image_id is None:
            self._preview_panel.set_object_occurrences([])
            return
        from app.db.models import TaggedObject
        from app.services.deoldified_pairing_service import DeoldifiedPairingService
        from app.services.object_service import ObjectService
        markers = []
        try:
            with session_scope() as session:
                # Object tags live on the B&W original of a deoldified pair.
                image_id = DeoldifiedPairingService.canonical_image_id(
                    session, image_id
                )
                for occ in ObjectService(session).get_occurrences_for_image(image_id):
                    if occ.point_x is None or occ.point_y is None:
                        continue
                    obj = session.get(TaggedObject, occ.object_id)
                    name = obj.name if obj is not None else None
                    markers.append((occ.occurrence_id, occ.point_x, occ.point_y, name))
        except Exception:
            log.exception("Failed to load object markers for image %s", image_id)
            markers = []
        self._preview_panel.set_object_occurrences(markers)

    @Slot(int, int, int)
    def _on_preview_object_create(self, image_id: int, x: int, y: int) -> None:
        """Open the object picker for a clicked point and record the occurrence."""
        from app.services.deoldified_pairing_service import DeoldifiedPairingService
        from app.services.object_service import ObjectService
        from app.ui.dialogs.object_picker_dialog import ObjectPickerDialog

        dlg = ObjectPickerDialog(self)
        if dlg.exec() != QDialog.Accepted or dlg.chosen_object_id is None:
            return
        try:
            with session_scope() as session:
                # Store the tag on the B&W original so both sides show it.
                target_id = DeoldifiedPairingService.canonical_image_id(
                    session, image_id
                )
                ObjectService(session).add_occurrence(
                    dlg.chosen_object_id, target_id, x, y, note=dlg.occurrence_note
                )
        except Exception:
            log.exception("Failed to add object occurrence")
            return
        self._refresh_preview_object_markers()
        if hasattr(self, "_objects_panel"):
            self._objects_panel.refresh()
        self.statusBar().showMessage(t("object_tagged_ok"), 3000)

    @Slot(int)
    def _open_object_sheet(self, object_id: int) -> None:
        """Switch to the Objects tab and show the given object's data sheet."""
        if not hasattr(self, "_objects_panel"):
            return
        self._tabs.setCurrentWidget(self._objects_panel)
        self._objects_panel.open_object(object_id)

    # ------------------------------------------------------------------
    # Object matching — find the same region on other images (#164)
    # ------------------------------------------------------------------

    @Slot(int, str)
    def _on_object_search_requested(self, object_id: int, scope: str) -> None:
        """Start a search from the Objects tab, resolving its scope choice."""
        image_ids = None
        if scope == "folder":
            image_ids = self._current_folder_image_ids()
            if not image_ids:
                image_ids = None
        self._start_object_search(object_id=object_id, image_ids=image_ids)

    def _current_folder_image_ids(self) -> Optional[List[int]]:
        """Image ids of the folder open in the browser, when it can tell us."""
        browser = getattr(self, "_image_browser", None)
        getter = getattr(browser, "current_folder_image_ids", None)
        if getter is None:
            return None
        try:
            return [int(i) for i in getter()]
        except Exception:
            log.debug("Could not resolve the current folder's images", exc_info=True)
            return None

    def _maybe_search_objects_after_scan(self, image_ids: List[int]) -> None:
        """Look for every tagged object on the images a scan just added (#164).

        Runs quietly in the background at low priority; nothing happens when
        the scan found nothing new, the feature is switched off, or no object
        has a framed reference marking yet.
        """
        if not image_ids:
            return
        config = self._object_matching_config()
        if not (config.enabled and config.search_new_images):
            return
        try:
            from app.services.object_matching_service import ObjectMatchingService

            with session_scope() as session:
                has_reference = bool(
                    ObjectMatchingService(session, config).matchable_object_ids()
                )
        except Exception:
            log.exception("Could not check for object reference markings")
            return
        if not has_reference:
            return
        self._start_object_search(
            object_id=None, image_ids=image_ids, post_scan=True
        )

    def _start_object_search(
        self,
        object_id: Optional[int],
        image_ids: Optional[List[int]] = None,
        post_scan: bool = False,
    ) -> None:
        """Run an object search in the background and offer the hits for review.

        Never blocks: progress shows in the status bar and the review dialog is
        only offered once the run finishes, so tagging can continue meanwhile.
        A *post_scan* run covers the images a scan just added: it runs at low
        priority, re-reads their features (the files may have changed), and
        stays silent when another search is already running.
        """
        from app.tasks import TaskPriority, get_task_manager
        from app.workers.object_match_worker import ObjectMatchWorker

        config = self._object_matching_config()
        if not config.enabled:
            if not post_scan:
                QMessageBox.information(
                    self, t("object_match_review_title"), t("object_match_disabled_tip")
                )
            return

        if self._object_search_task is not None and not (
            self._object_search_task.state.is_final
        ):
            if post_scan:
                log.info("Object search already running; post-scan search skipped")
            else:
                QMessageBox.information(self, t("busy_title"), t("busy_msg"))
            return

        worker = ObjectMatchWorker(
            object_id=object_id,
            image_ids=image_ids,
            config=config,
            invalidate_image_ids=image_ids if post_scan else None,
        )

        def work(ctx):
            return worker.run_in_task(ctx)

        task = get_task_manager().submit(
            t("object_match_running"),
            work,
            supports_pause=True,
            priority=TaskPriority.LOW if post_scan else TaskPriority.NORMAL,
            on_done=lambda result: self._on_object_search_done(
                object_id, result, post_scan=post_scan
            ),
            on_error=self._on_object_search_error,
            on_cancelled=self._on_object_search_cancelled,
        )
        self._object_search_task = task
        task.progress_changed.connect(self._on_task_progress)
        self._status_label.setText(t("object_match_running"))

    def _object_matching_config(self):
        """Matching parameters, with the Objects tab's choices applied.

        The sensitivity slider moves the two gates that actually decide how
        permissive a run is — the score floor and the required inlier count —
        between a strict and a loose end of the range.  The auto-tag and
        post-scan checkboxes override the ``config.yaml`` defaults.
        """
        from app.app_settings import app_qsettings
        from app.config import ObjectMatchingConfig
        from app.ui.panels.objects_panel import matching_preferences

        base = matching_preferences(
            getattr(self._config, "object_matching", None) or ObjectMatchingConfig()
        )
        sensitivity = 50
        try:
            sensitivity = int(
                app_qsettings().value("object_matching/sensitivity", 50)
            )
        except (TypeError, ValueError):
            pass
        fraction = max(0.0, min(1.0, sensitivity / 100.0))
        return replace(
            base,
            min_score=round(0.55 - 0.30 * fraction, 3),
            min_inliers=int(round(20 - 12 * fraction)),
        )

    def _on_object_search_done(
        self, object_id: Optional[int], result: object, post_scan: bool = False
    ) -> None:
        self._object_search_task = None
        found = int(getattr(result, "suggestions_created", 0) or 0)
        auto = int(getattr(result, "auto_accepted", 0) or 0)
        skipped = int(getattr(result, "skipped_no_reference", 0) or 0)
        searched = int(getattr(result, "objects_searched", 0) or 0)
        run_id = getattr(result, "run_id", None) or None
        self._status_label.setText(t("ready"))

        if auto:
            # The new markings already exist; show them right away.
            self._refresh_object_views()

        if post_scan:
            if not (found or auto):
                return
            message = t("object_match_auto_done").format(auto=auto, pending=found)
            self.statusBar().showMessage(message, 8000)
            reply = QMessageBox.question(
                self,
                t("object_match_review_title"),
                f"{message}\n\n{t('object_match_review_question')}",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._open_object_match_review(None, run_id=run_id)
            return

        if skipped and skipped == searched:
            QMessageBox.information(
                self, t("object_match_review_title"), t("object_match_no_reference")
            )
            return
        if not (found or auto):
            self.statusBar().showMessage(t("object_match_none_found"), 5000)
            return
        self._open_object_match_review(object_id)

    def _refresh_object_views(self) -> None:
        """Redraw object markers everywhere after markings changed."""
        self._refresh_preview_object_markers()
        if hasattr(self, "_objects_panel"):
            self._objects_panel.refresh()
        self._image_browser.refresh_object_markers()

    def _open_object_match_review(
        self, object_id: Optional[int], run_id: Optional[str] = None
    ) -> None:
        from app.ui.dialogs.object_match_review_dialog import ObjectMatchReviewDialog

        dialog = ObjectMatchReviewDialog(self, object_id=object_id, run_id=run_id)
        dialog.exec()
        if dialog.accepted_count or dialog.rejected_count:
            self._refresh_object_views()
            self.statusBar().showMessage(
                t("object_match_done").format(
                    accepted=dialog.accepted_count, rejected=dialog.rejected_count
                ),
                5000,
            )

    def _on_object_search_error(self, message: str) -> None:
        self._object_search_task = None
        self._status_label.setText(t("ready"))
        log.error("Object matching failed: %s", message)
        QMessageBox.warning(self, t("object_match_review_title"), str(message))

    def _on_object_search_cancelled(self) -> None:
        self._object_search_task = None
        self._status_label.setText(t("ready"))

    @Slot(int, int, int)
    def _on_cluster_face_right_clicked(self, face_id: int, gx: int, gy: int) -> None:
        """Show the preview context menu when a cluster thumbnail is right-clicked."""
        self._current_face_id = face_id
        self._remove_face_btn.setEnabled(True)
        self._reassign_btn.setEnabled(True)
        # Ensure the preview panel has the image loaded so "Edit bbox" works.
        self._show_face_in_preview(face_id)
        person_name = self._cluster_panel.get_face_person_name(face_id)
        with session_scope() as session:
            f = session.get(Face, face_id)
            is_pending = f is not None and f.auto_merge_review_status == "pending"
        self._preview_panel.show_face_context_menu(
            face_id, gx, gy, person_name=person_name, is_pending=is_pending
        )
