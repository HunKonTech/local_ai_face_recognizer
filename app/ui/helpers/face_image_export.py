"""Shared entry point for the "export faces into separate images" flow (#175).

All three call sites (image tree context menu, big-viewer canvas context menu,
Export dialog) funnel through :func:`run_face_image_export`, so the dialog, the
background task and the summary message stay identical everywhere.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from PySide6.QtWidgets import QDialog, QMessageBox, QWidget

from app.db.database import session_scope
from app.ui.i18n import t

log = logging.getLogger(__name__)


def run_face_image_export(
    image_ids: Sequence[int],
    parent: Optional[QWidget] = None,
    face_ids: Optional[Sequence[int]] = None,
) -> None:
    """Ask for export settings, then run the export as a background task.

    Args:
        image_ids: Images whose faces should be exported.
        parent:    Parent widget for the dialog and message boxes.
        face_ids:  Optional restriction to individual faces.
    """
    ids = [int(i) for i in dict.fromkeys(image_ids)]
    if not ids:
        return

    from app.services.face_image_export_service import FaceImageExportService
    from app.ui.dialogs.face_image_export_dialog import FaceImageExportDialog

    dialog = FaceImageExportDialog(ids, face_ids=face_ids, parent=parent)
    if dialog.exec() != QDialog.Accepted:
        return
    options = dialog.options()

    def work(ctx):  # noqa: ANN001 — runs on the task thread
        with session_scope() as session:
            return FaceImageExportService(session).export_faces_of_images(
                ids, options, ctx=ctx
            )

    def on_done(result) -> None:  # noqa: ANN001 — FaceImageExportResult
        written = getattr(result, "written", 0)
        skipped = getattr(result, "skipped", 0)
        QMessageBox.information(
            parent,
            t("fexp_done_title"),
            t(
                "fexp_done_body",
                written=written,
                skipped=skipped,
                folder=options.target_dir,
            ),
        )

    def on_error(message: str) -> None:
        QMessageBox.critical(parent, t("export_error"), message)

    from app.tasks import TaskPriority, get_task_manager

    get_task_manager().submit(
        t("task_face_image_export"),
        work,
        supports_pause=True,
        priority=TaskPriority.LOW,
        on_done=on_done,
        on_error=on_error,
    )
