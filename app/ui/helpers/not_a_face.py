"""Shared entry point for the "Not a face" action.

The image browser canvas and the cluster / preview face menus both funnel
through :func:`mark_not_a_face`, so the confirmation, the learning call and the
messages stay identical everywhere.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from PySide6.QtWidgets import QMessageBox, QWidget

from app.db.database import session_scope
from app.ui.i18n import t

log = logging.getLogger(__name__)


def mark_not_a_face(face_ids: Sequence[int], parent: Optional[QWidget] = None) -> int:
    """Confirm, then mark *face_ids* as "not a face".  Returns how many were marked."""
    from app.services.non_face_service import NonFaceService

    ids = [int(f) for f in face_ids]
    if not ids:
        return 0
    reply = QMessageBox.question(
        parent,
        t("not_a_face_title"),
        t("not_a_face_confirm", n=len(ids)),
        QMessageBox.Yes | QMessageBox.No,
    )
    if reply != QMessageBox.Yes:
        return 0
    with session_scope() as session:
        marked = NonFaceService(session).mark_not_face(ids)
    if marked < len(ids):
        QMessageBox.warning(parent, t("not_a_face_title"), t("not_a_face_no_embedding"))
    log.info("Not a face: %d of %d box(es) marked", marked, len(ids))
    return marked
