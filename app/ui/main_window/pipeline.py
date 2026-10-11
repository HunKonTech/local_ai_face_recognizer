"""Scan, deep-learning pipeline runs and their task callbacks."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QApplication,
    QMessageBox,
)

from app.db.database import session_scope
from app.db.models import Person
from app.services.identity_service import IdentityService
from app.ui.dialogs.manual_face_dialog import NoFaceImagesDialog
from app.ui.dialogs.suggestion_dialog import SuggestionDialog
from app.ui.i18n import t
from app.workers.deep_pipeline_worker import (
    MODE_CLUSTER,
    MODE_DETECT_FACES,
    MODE_REBUILD,
    MODE_REBUILD_MODEL,
    MODE_REEMBED,
    MODE_RESCAN,
    MODE_TRAIN,
    DeepPipelineWorker,
)

log = logging.getLogger(__name__)


class PipelineMixin:
    """Scan, deep-learning pipeline runs and their task callbacks."""

    @Slot(str)
    def _on_scan_workflow_started(self, workflow_name: str) -> None:
        if workflow_name == "face_detection":
            self._on_scan()
        elif workflow_name == "full_rescan":
            # Full Rescan = re-detect EVERY image from scratch (MODE_REBUILD), not
            # just rebuild the neural model — the card promises "clears all
            # detections and rebuilds from scratch", and only a full re-detection
            # re-runs the multi-technology gate over already-scanned images.
            self._on_deep_rebuild()
        elif workflow_name == "train_model":
            self._on_deep_train()
        elif workflow_name == "object_matching":
            self._start_object_search(object_id=None, image_ids=None)

    @Slot(str)
    def _on_scan_maintenance_action(self, action: str) -> None:
        """Dispatch a maintenance action triggered from the Scan & Maintenance
        dialog's developer tab. These tools also run automatically as part of
        the redesigned detection pipeline; the tab keeps them manually
        launchable for debugging and one-off fixes."""
        handlers = {
            "overlap_cleanup": self._on_find_overlapping_unknown_faces,
            "embedding_duplicates": self._on_find_embedding_duplicate_faces,
            "identity_repair": self._on_identity_repair_scan,
            "cleanup_empty_unknowns": self._on_cleanup_empty_unknown_persons,
            "manage_ignored_faces": self._on_manage_ignored_faces,
            "reembed_faces": self._on_deep_reembed,
        }
        handler = handlers.get(action)
        if handler is None:
            log.warning("Unknown maintenance action: %s", action)
            return
        handler()

    # ------------------------------------------------------------------
    # Deep-learning (AI) pipeline
    # ------------------------------------------------------------------

    def _deep_pipeline_guard(self) -> bool:
        """Shared pre-flight checks for the deep pipeline modes."""
        from app.gdrive import preferences as _gprefs

        prefs = _gprefs.load()
        if prefs.enabled and self._gdrive_session is None:
            QMessageBox.information(
                self, t("gdrive_chip_opening"), t("gdrive_scan_no_session")
            )
            return False
        if self._gdrive_session is None and not getattr(self, "_root_folders", None):
            QMessageBox.warning(self, t("no_folder_title"), t("no_folder_msg"))
            return False
        if self._pipeline_busy():
            QMessageBox.information(self, t("busy_title"), t("busy_msg"))
            return False
        return True

    @Slot()
    def _on_deep_rescan(self) -> None:
        if not self._deep_pipeline_guard():
            return
        self._start_deep_pipeline(MODE_RESCAN)

    @Slot()
    def _on_deep_train(self) -> None:
        if not self._deep_pipeline_guard():
            return
        self._start_deep_pipeline(MODE_TRAIN)

    @Slot()
    def _on_deep_face_detect(self) -> None:
        if not self._deep_pipeline_guard():
            return
        self._start_deep_pipeline(MODE_DETECT_FACES)

    @Slot()
    def _on_deep_rebuild(self) -> None:
        if not self._deep_pipeline_guard():
            return
        reply = QMessageBox.question(
            self,
            t("deep_rebuild_confirm_title"),
            t("deep_rebuild_confirm_msg"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._start_deep_pipeline(MODE_REBUILD)

    @Slot()
    def _on_deep_rebuild_model(self) -> None:
        if not self._deep_pipeline_guard():
            return
        reply = QMessageBox.question(
            self,
            t("rebuild_model_confirm_title"),
            t("rebuild_model_confirm_msg"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._start_deep_pipeline(MODE_REBUILD_MODEL)

    @Slot()
    def _on_deep_reembed(self) -> None:
        if not self._deep_pipeline_guard():
            return
        reply = QMessageBox.question(
            self,
            t("reembed_confirm_title"),
            t("reembed_confirm_msg"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._start_deep_pipeline(MODE_REEMBED)

    def _start_deep_pipeline(self, mode: str) -> None:
        """Launch the deep-learning (AI) pipeline as a Task Manager task."""
        drive_active = self._gdrive_session is not None
        if not drive_active and not getattr(self, "_root_folders", None):
            return

        from app.app_settings import app_qsettings
        _qs = app_qsettings()
        ai_viz = _qs.value("debug/ai_visualization", False, type=bool)
        ai_log = _qs.value("debug/ai_log_enabled", False, type=bool)

        if drive_active and self._gdrive_session is not None:
            from app.paths import drive_mirror_dir
            drive_folders = self._gdrive_session.folders
            root_id = drive_folders.root_id if drive_folders else ""
            worker = DeepPipelineWorker(
                root_folders=[],         # unused in Drive mode
                config=self._config,
                mode=mode,
                db_path_override=self._db_path,
                drive_client=self._gdrive_session._client,
                drive_root_folder_id=root_id,
                drive_mirror_dir=drive_mirror_dir(root_id),
                ai_visualization=ai_viz,
                ai_debug_log=ai_log,
            )
        else:
            worker = DeepPipelineWorker(
                root_folders=list(self._root_folders),
                config=self._config,
                mode=mode,
                db_path_override=self._db_path,
                ai_visualization=ai_viz,
                ai_debug_log=ai_log,
            )
        worker.log_message.connect(self._log_panel.append_plain)
        if ai_viz:
            self._open_ai_viz_window()
            worker.face_debug.connect(self._on_face_debug)

        name = {
            MODE_RESCAN: t("task_deep_rescan"),
            MODE_REBUILD: t("task_deep_rebuild"),
            MODE_REBUILD_MODEL: t("task_deep_rebuild_model"),
            MODE_TRAIN: t("task_deep_train"),
            MODE_DETECT_FACES: t("task_deep_detect"),
            MODE_CLUSTER: t("task_deep_cluster"),
            MODE_REEMBED: t("task_deep_reembed"),
        }.get(mode, t("task_deep_rescan"))
        self._run_pipeline_task(worker, name)

    def _run_pipeline_task(self, worker, name: str) -> None:
        """Submit a scan/AI *worker* to the Task Manager at HIGH priority.

        The pipeline is the app's most important work, so it runs at HIGH
        priority — it preempts a lower-priority task (e.g. an export) when no
        slot is free, and that task resumes automatically afterwards.  Nothing
        blocks the UI: the toolbar progress bar and status label are passive.
        """
        from app.tasks import TaskPriority, get_task_manager

        self._pending_auto_assignment_count = 0
        self._set_scanning_state(True)
        self._active_pipeline_worker = worker

        def work(ctx):  # noqa: ANN001 — runs on the task thread
            return worker.run_in_task(ctx)

        task = get_task_manager().submit(
            name,
            work,
            supports_pause=True,
            priority=TaskPriority.HIGH,
            on_done=self._on_pipeline_task_done,
            on_error=self._on_pipeline_task_error,
            on_cancelled=self._on_pipeline_task_cancelled,
        )
        self._active_pipeline_task = task
        task.progress_changed.connect(self._on_task_progress)

    def _pipeline_busy(self) -> bool:
        """True while a scan/AI pipeline task is queued, running or paused."""
        task = self._active_pipeline_task
        return task is not None and not task.state.is_final

    @Slot(int, str)
    def _on_task_progress(self, percent: int, message: str) -> None:
        """Mirror the active pipeline task's progress onto the toolbar (passive)."""
        self._progress_bar.setValue(percent)
        if message:
            self._status_label.setText(message)

    def _on_pipeline_task_done(self, result: object) -> None:
        self._active_pipeline_task = None
        self._active_pipeline_worker = None
        self._pending_auto_assignment_count = getattr(result, "n_auto_assignments", 0)
        self._pending_object_scan_ids = list(getattr(result, "new_image_ids", None) or [])
        self._on_pipeline_finished(
            getattr(result, "success", True), getattr(result, "summary", "")
        )

    def _on_pipeline_task_error(self, message: str) -> None:
        self._active_pipeline_task = None
        self._active_pipeline_worker = None
        self._on_pipeline_error(message)

    def _on_pipeline_task_cancelled(self) -> None:
        self._active_pipeline_task = None
        self._active_pipeline_worker = None
        self._set_scanning_state(False)
        self._status_label.setText(t("ready"))

    @Slot(object)
    def _on_reset_unknown_persons(self, options) -> None:
        """Run an Unknown-identity reset with the options chosen inline in the
        Scan & Maintenance dialog.  The checkboxes there are the confirmation,
        so no extra prompt is shown."""
        from app.gdrive import preferences as _gprefs

        prefs = _gprefs.load()
        if prefs.enabled and self._gdrive_session is None:
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

        # Hard-deleting faces walks the rows one by one on the UI thread.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            with session_scope() as session:
                from app.services.unknown_person_reset_service import (
                    UnknownPersonResetService,
                )

                result = UnknownPersonResetService(session).reset(options)
        finally:
            QApplication.restoreOverrideCursor()

        self._current_person_id = None
        self._current_face_id = None
        self._cluster_panel.clear()
        self._preview_panel.clear()
        self._refresh_persons()
        self._image_browser._reload_current_face_data()
        self._status_label.setText(
            t(
                "reset_unknowns_status",
                persons=result.deleted_persons,
                faces=result.unassigned_faces,
                deleted=result.deleted_faces,
            )
        )

        # Re-cluster unassigned faces into new Unknown groups if selected
        if options.rebuild_clusters:
            self._start_deep_pipeline(MODE_CLUSTER)

    @Slot()
    def _on_stop(self) -> None:
        if self._active_pipeline_task is not None:
            self._active_pipeline_task.cancel()

    # ------------------------------------------------------------------
    # Background dependency installer callbacks
    # ------------------------------------------------------------------

    @Slot(str)
    def on_dep_installing(self, display_name: str) -> None:
        self._status_label.setText(t("dep_installing", name=display_name))

    @Slot(str)
    def on_dep_installed(self, display_name: str) -> None:
        self._status_label.setText(t("dep_installed", name=display_name))
        # onnxruntime's C-extension must be imported on the main thread once
        # before any background worker touches it — otherwise GC during the
        # import can reclaim a Qt object on the worker thread and crash AppKit.
        # We re-run the warm-up here so a freshly installed onnxruntime/
        # insightface is safely primed even though the startup warm-up ran
        # before the install completed.
        if display_name in ("onnxruntime", "insightface"):
            from app.detectors.factory import warm_up_onnxruntime
            warm_up_onnxruntime(self._config.detection)

    @Slot(str, str)
    def on_dep_install_failed(self, display_name: str, reason: str) -> None:
        log.warning("Background install failed for %s: %s", display_name, reason)

    @Slot(bool)
    def on_dep_install_done(self, any_installed: bool) -> None:
        if not any_installed:
            return
        self._status_label.setText(t("dep_install_done"))

    @Slot()
    def _on_no_face_images(self) -> None:
        dlg = NoFaceImagesDialog(config=self._config, parent=self)
        dlg.changed.connect(self._refresh_persons)
        dlg.exec()

    @Slot()
    def _on_show_suggestions(self, open_auto_tab: bool = False) -> None:
        dlg = SuggestionDialog(self._config, parent=self)
        dlg.data_changed.connect(self._refresh_persons)
        dlg.data_changed.connect(self._image_browser._reload_current_face_data)
        if open_auto_tab:
            dlg.show_auto_assignments_tab()
        dlg.exec()
        self._refresh_persons()
        self._image_browser._reload_current_face_data()
        # The user's confirm/correct/revert decisions are positive/negative
        # training signals — retrain the neural network in the background so it
        # actually learns from this review session.
        if dlg.made_decisions:
            self._retrain_after_review()

    @Slot(str)
    def _on_pipeline_error(self, message: str) -> None:
        self._set_scanning_state(False)
        QMessageBox.critical(self, t("error"), message)

    # ------------------------------------------------------------------
    # Person / face interaction
    # ------------------------------------------------------------------

    @Slot(int)
    def _on_person_selected(self, person_id: int) -> None:
        self._current_person_id = person_id
        self._current_face_id = None

        is_protected = False
        is_auto_named = False
        with session_scope() as session:
            svc = IdentityService(session)
            person = session.get(Person, person_id)
            faces = svc.get_faces_for_person(person_id)
            if person is None:
                return
            # Crop-path self-heal is NOT run here: it decodes images and writes
            # UPDATE faces SET crop_path on the UI thread, so doing it on every
            # person selection blocked the UI and contended with the startup
            # repair task's write lock ("database is locked").  Repair runs once
            # at startup only (_start_crop_repair_task); selecting a person reads.
            is_protected = person.is_protected
            is_auto_named = person.is_auto_named
            self._cluster_panel.show_person(
                person.name,
                faces,
                birth_date=person.birth_date,
                death_date=person.death_date,
            )
            self._preview_panel.clear()

        self._rename_btn.setEnabled(not is_protected)
        self._merge_btn.setEnabled(True)
        self._delete_person_btn.setEnabled(not is_protected)
        self._remove_face_btn.setEnabled(False)
        self._reassign_btn.setEnabled(False)
        self._person_info_btn.setEnabled(not is_protected)

        self._notify_recording_context()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_scanning_state(self, scanning: bool) -> None:
        self._progress_bar.setVisible(scanning)
        if scanning:
            self._progress_bar.setValue(0)
