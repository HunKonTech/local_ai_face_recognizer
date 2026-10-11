"""Task manager button, crop repair task and startup library check."""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QTimer, Slot

from app.db.database import session_scope
from app.db.models import Face
from app.ui.i18n import t

log = logging.getLogger(__name__)


class TasksMixin:
    """Task manager button, crop repair task and startup library check."""

    @Slot(int, int, int)
    def _on_task_counts_changed(
        self, running: int, queued: int, paused: int
    ) -> None:
        total = running + queued + paused
        self._update_tasks_button_text(total)

    def _update_tasks_button_text(self, total: Optional[int] = None) -> None:
        """Set the bottom Task Manager button label to show the live count."""
        if not hasattr(self, "_tasks_status_btn"):
            return
        if total is None:
            from app.tasks import get_task_manager
            total = get_task_manager().active_count
        self._tasks_status_btn.setText(
            t("tasks_btn_count", n=total) if total else t("tasks_btn")
        )

    # Delay before the startup crop-repair pass begins.  Starting it with the
    # window leaves it competing with first paint / thumbnail loading for CPU,
    # IO and the SQLite write lock exactly when the user starts working — on a
    # large library that reads as "the whole app is slow right after launch".
    _CROP_REPAIR_DELAY_MS = 30_000

    def _start_crop_repair_task(self) -> None:
        """Repair missing/duplicated crop paths once, in the background.

        This used to run inside ``_refresh_persons`` — i.e. on the UI thread,
        on every refresh, after every face assignment.  It is a consistency
        safety net, so running it once per app start (off the UI thread, at LOW
        priority, half a minute after launch) is enough.
        """
        QTimer.singleShot(self._CROP_REPAIR_DELAY_MS, self._submit_crop_repair_task)

    def _submit_crop_repair_task(self) -> None:
        crops_dir = self._config.crops_dir_resolved
        thumb_size = self._config.scan.thumbnail_size

        def work(ctx) -> int:  # noqa: ANN001
            from sqlalchemy.orm import lazyload

            from app.services.face_crop_service import ensure_unique_face_crops

            with session_scope() as session:
                faces = (
                    session.query(Face)
                    # Crop repair never touches embeddings — skip the default
                    # selectin blob load for this full-table query.
                    .options(lazyload(Face.blob))
                    .filter(Face.is_excluded == False)  # noqa: E712
                    .all()
                )
                ctx.checkpoint()
                ctx.report(50, "")
                return ensure_unique_face_crops(
                    session, faces, crops_dir, thumb_size
                )

        from app.tasks import get_task_manager
        from app.tasks.manager import TaskPriority

        def on_done(repaired: object) -> None:
            if repaired:
                self._refresh_persons()

        # LOW priority: this is a maintenance safety net, so scans/recognition
        # runs and anything the user starts must preempt it.
        get_task_manager().submit(
            t("task_crop_repair"),
            work,
            transient=True,
            priority=TaskPriority.LOW,
            on_done=on_done,
        )

    # ------------------------------------------------------------------
    # Image library startup check
    # ------------------------------------------------------------------

    def _check_image_library_on_startup(self) -> None:
        """Show a dialog if the library root is configured but not found."""
        from app.services.image_library_service import get_image_library_optional

        svc = get_image_library_optional()
        if svc is None:
            return

        root = svc.library_root
        if root is None:
            return  # Not configured — no warning needed.

        if svc.is_available():
            log.debug("Image library root available: %s", root)
            return

        # Root is configured but not reachable — prompt the user.
        from app.ui.dialogs.image_library_dialog import ImageLibraryMissingDialog

        dlg = ImageLibraryMissingDialog(missing_path=str(root), parent=self)
        if dlg.exec() == ImageLibraryMissingDialog.Accepted:
            new_root = dlg.new_root()
            if new_root:
                try:
                    svc.set_library_root(new_root)
                    log.info("Image library root updated to: %s", new_root)
                except (NotADirectoryError, RuntimeError) as exc:
                    from PySide6.QtWidgets import QMessageBox

                    from app.ui.i18n import t
                    QMessageBox.warning(self, t("error"), str(exc))
