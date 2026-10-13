"""Manager dialog for the permanent face ignore list.

Lists every ignored face embedding with its thumbnail (when the crop file
still exists), the source person name snapshot and the ignore date, and lets
the user revoke entries so the face can be recognised again on the next run.

"Not a face" marks are listed the same way, followed by the boxes the learned
non-face filter hid; restoring one of those teaches the filter it is a face.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig
from app.db.database import session_scope
from app.db.models import IGNORED_KIND_NOT_FACE
from app.services.ignored_face_service import IgnoredFaceService
from app.services.non_face_service import NonFaceService
from app.ui.i18n import t

log = logging.getLogger(__name__)

_THUMB_SIZE = 96

# Qt.UserRole payload: (kind, id) — an ignore-list entry or a model-hidden face.
_ENTRY = "entry"
_AUTO = "auto"


class IgnoredFacesDialog(QDialog):
    """Modal manager for permanently ignored faces."""

    def __init__(
        self,
        config: AppConfig,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._changed = False

        self.setWindowTitle(t("ignored_faces_title"))
        self.setMinimumSize(520, 420)
        self.resize(640, 560)
        self._build_ui()
        self._reload()

    def changed(self) -> bool:
        """True when at least one entry was revoked."""
        return self._changed

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._info_lbl = QLabel(t("ignored_faces_info"))
        self._info_lbl.setWordWrap(True)
        self._info_lbl.setStyleSheet("color: #A6ADC8;")
        layout.addWidget(self._info_lbl)

        self._list = QListWidget()
        self._list.setIconSize(QSize(_THUMB_SIZE, _THUMB_SIZE))
        self._list.setSelectionMode(QListWidget.ExtendedSelection)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list, stretch=1)

        self._count_lbl = QLabel()
        self._count_lbl.setStyleSheet("color: #888;")
        layout.addWidget(self._count_lbl)

        btn_row = QHBoxLayout()
        self._unignore_btn = QPushButton(t("ignored_faces_unignore"))
        self._unignore_btn.setEnabled(False)
        self._unignore_btn.clicked.connect(self._on_unignore)
        btn_row.addWidget(self._unignore_btn)
        btn_row.addStretch()

        close_btn = QPushButton(t("scanModes.close"))
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------

    def _reload(self) -> None:
        self._list.clear()
        with session_scope() as session:
            entries = IgnoredFaceService(session).list_ignored()
            for entry in entries:
                if entry.kind == IGNORED_KIND_NOT_FACE:
                    label_parts = [t("ignored_faces_kind_not_face")]
                else:
                    label_parts = [
                        entry.source_person_name or t("ignored_faces_unknown_source")
                    ]
                if entry.created_at is not None:
                    label_parts.append(entry.created_at.strftime("%Y-%m-%d %H:%M"))
                if entry.note:
                    label_parts.append(entry.note)
                self._add_item(" — ".join(label_parts), (_ENTRY, entry.id),
                               entry.thumbnail_path)
            for face in NonFaceService(session).list_model_hidden():
                pct = round(100 * (face.assignment_confidence or 0.0))
                self._add_item(t("ignored_faces_auto_hidden", p=pct),
                               (_AUTO, face.id), face.crop_path)

        n = self._list.count()
        self._count_lbl.setText(t("ignored_faces_count", n=n))
        if n == 0:
            empty = QListWidgetItem(t("ignored_faces_empty"))
            empty.setFlags(Qt.NoItemFlags)
            self._list.addItem(empty)
        self._unignore_btn.setEnabled(False)

    def _add_item(self, text: str, key: tuple, thumbnail_path: Optional[str]) -> None:
        item = QListWidgetItem(text)
        item.setData(Qt.UserRole, key)
        if thumbnail_path and Path(thumbnail_path).exists():
            pixmap = QPixmap(thumbnail_path)
            if not pixmap.isNull():
                item.setIcon(QIcon(pixmap.scaled(
                    _THUMB_SIZE, _THUMB_SIZE,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )))
        self._list.addItem(item)

    def _on_selection_changed(self) -> None:
        self._unignore_btn.setEnabled(bool(self._selected_ids()))

    def _selected_ids(self) -> list[tuple]:
        ids = []
        for item in self._list.selectedItems():
            key = item.data(Qt.UserRole)
            if key is not None:
                ids.append((key[0], int(key[1])))
        return ids

    def _on_unignore(self) -> None:
        ids = self._selected_ids()
        if not ids:
            return
        reply = QMessageBox.question(
            self,
            t("ignored_faces_title"),
            t("ignored_faces_unignore_confirm", n=len(ids)),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        with session_scope() as session:
            svc = IgnoredFaceService(session)
            non_face = NonFaceService(session)
            for kind, item_id in ids:
                done = (
                    non_face.confirm_face(item_id)
                    if kind == _AUTO
                    else svc.unignore(item_id)
                )
                self._changed = self._changed or done
        self._reload()
