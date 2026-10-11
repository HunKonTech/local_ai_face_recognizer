"""Re-recognition actions started from the tree context menu."""

from __future__ import annotations

import logging
from typing import List, Optional

from PySide6.QtWidgets import (
    QMessageBox,
)

from app.db.database import session_scope
from app.ui.i18n import t

log = logging.getLogger(__name__)


class RerecognitionMixin:
    """Re-recognition actions started from the tree context menu."""

    def _rerecognition_thresholds(self) -> "tuple[float, float]":
        """Read (auto, suggest) thresholds from QSettings, config as fallback."""
        from app.app_settings import app_qsettings

        rec = getattr(self._config, "recognition", None)
        auto_default = rec.rerecognition_auto_threshold if rec else 0.72
        suggest_default = rec.rerecognition_suggest_threshold if rec else 0.55
        qs = app_qsettings()
        auto = float(qs.value("rerecognition/auto_threshold", auto_default, type=float))
        suggest = float(
            qs.value("rerecognition/suggest_threshold", suggest_default, type=float)
        )
        return auto, suggest

    def _rerecognition_enabled(self) -> bool:
        from app.app_settings import app_qsettings

        rec = getattr(self._config, "recognition", None)
        default = rec.rerecognition_enabled if rec else True
        return bool(
            app_qsettings().value("rerecognition/enabled", default, type=bool)
        )

    def _start_rerecognition(self, image_ids: List[int], engine: str = "classic") -> None:
        if not image_ids or self._config is None:
            return
        if engine == "classic" and not self._rerecognition_enabled():
            QMessageBox.information(
                self, t("rerec_progress_title"), t("rerec_disabled")
            )
            return
        task = getattr(self, "_rerec_task", None)
        if task is not None and not task.state.is_final:
            return  # a run is already in flight

        auto_thr, suggest_thr = self._rerecognition_thresholds()

        from app.tasks import TaskPriority, get_task_manager
        from app.workers.rerecognition_worker import ReRecognitionWorker

        # Runs as a Task Manager task (pause/resume/stop there) — no modal
        # dialog, so the UI is never blocked.  NORMAL priority: more important
        # than exports, below the full scan/AI pipeline.
        worker = ReRecognitionWorker(
            image_ids,
            self._config,
            auto_threshold=auto_thr,
            suggest_threshold=suggest_thr,
            engine=engine,
        )
        self._rerec_worker = worker  # keep a reference so it isn't GC'd

        def work(ctx):  # noqa: ANN001 — runs on the task thread
            return worker.run_in_task(ctx)

        self._rerec_task = get_task_manager().submit(
            t("task_rerecognition"),
            work,
            supports_pause=True,
            priority=TaskPriority.NORMAL,
            on_done=self._on_rerecognition_finished,
            on_error=self._on_rerecognition_failed,
            on_cancelled=self._on_rerecognition_cancelled,
        )

    def _cleanup_rerec_worker(self) -> None:
        self._rerec_task = None
        self._rerec_worker = None

    def _on_rerecognition_cancelled(self) -> None:
        self._cleanup_rerec_worker()
        QMessageBox.information(
            self, t("rerec_progress_title"), t("rerec_cancelled")
        )
        self._refresh_after_rerecognition()

    def _on_rerecognition_failed(self, message: str) -> None:
        self._cleanup_rerec_worker()
        QMessageBox.critical(
            self, t("rerec_error_title"), t("rerec_error_body", error=message)
        )

    def _on_rerecognition_finished(self, result) -> None:
        self._cleanup_rerec_worker()

        if getattr(result, "cancelled", False):
            QMessageBox.information(
                self, t("rerec_progress_title"), t("rerec_cancelled")
            )
            self._refresh_after_rerecognition()
            return

        if result.n_examined == 0 and result.n_auto == 0:
            QMessageBox.information(
                self, t("rerec_summary_title"), t("rerec_no_profiles")
            )
            return

        self._refresh_after_rerecognition()

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle(t("rerec_summary_title"))
        summary_text = t(
            "rerec_summary_body",
            examined=result.n_examined,
            auto=result.n_auto,
            suggest=result.n_suggest,
            none=result.n_none,
        )
        # The optional AI face-detection pass (analysis only) reports its
        # counts here; older results without the fields are shown unchanged.
        if getattr(result, "ai_faces_found", None) is not None:
            summary_text += "\n" + t(
                "rerec_summary_ai",
                faces=result.ai_faces_found,
                images=result.ai_images_scanned or 0,
            )
        box.setText(summary_text)
        review_btn = None
        if result.suggest_items:
            review_btn = box.addButton(
                t("rerec_summary_review"), QMessageBox.AcceptRole
            )
        box.addButton(t("rerec_summary_close"), QMessageBox.RejectRole)
        box.exec()

        if review_btn is not None and box.clickedButton() is review_btn:
            self._open_rerecognition_review(result)

    def _open_rerecognition_review(self, result) -> None:
        from app.ui.dialogs.rerecognition_review_dialog import (
            ReRecognitionReviewDialog,
        )

        rec_cfg = getattr(self._config, "recognition", None)
        dialog = ReRecognitionReviewDialog(
            result.suggest_items,
            result.batch_id or "",
            self._all_persons,
            recognition_config=rec_cfg,
            parent=self,
        )
        dialog.applied.connect(self._refresh_after_rerecognition)
        dialog.exec()

    def _latest_undoable_batch(self) -> Optional[str]:
        try:
            from app.services.rerecognition_service import ReRecognitionService

            with session_scope() as session:
                return ReRecognitionService(session).latest_undoable_batch()
        except Exception:  # noqa: BLE001
            log.exception("Could not query re-recognition history")
            return None

    def _undo_last_rerecognition(self) -> None:
        from app.services.rerecognition_service import ReRecognitionService

        batch_id = self._latest_undoable_batch()
        if batch_id is None:
            QMessageBox.information(
                self, t("rerec_progress_title"), t("rerec_nothing_to_undo")
            )
            return
        try:
            with session_scope() as session:
                n = ReRecognitionService(session).undo_batch(batch_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("Undo of last re-recognition failed")
            QMessageBox.critical(
                self, t("rerec_error_title"), t("rerec_undo_error", error=str(exc))
            )
            return
        self._refresh_after_rerecognition()
        QMessageBox.information(
            self, t("rerec_summary_title"), t("rerec_undo_done", n=n)
        )

    def _open_rerecognition_history(self) -> None:
        from app.ui.dialogs.rerecognition_history_dialog import (
            ReRecognitionHistoryDialog,
        )

        dialog = ReRecognitionHistoryDialog(self)
        dialog.changed.connect(self._refresh_after_rerecognition)
        dialog.exec()

    def _refresh_after_rerecognition(self) -> None:
        """Refresh person list + current image overlay after a merge/undo."""
        self._reload_persons_combo()
        if self._current_image_id is not None:
            self._reload_current_face_data()
        self.person_data_changed.emit()
