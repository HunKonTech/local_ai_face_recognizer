"""Manage custom person field definitions: rename, reorder, change type, delete.

Every action is committed immediately (it edits the shared definitions, not
one person). Opened from the person editor and from the Persons menu.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.services.person_field_service import PersonFieldError, PersonFieldService
from app.ui.i18n import t

log = logging.getLogger(__name__)

_TYPES = ("text", "long_text", "date")


class PersonFieldsDialog(QDialog):
    """List of custom field definitions with management actions."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("cf_manage_title"))
        self.resize(420, 380)

        layout = QVBoxLayout(self)
        hint = QLabel(t("cf_manage_hint"))
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._list = QListWidget()
        self._list.setDragDropMode(QAbstractItemView.InternalMove)
        self._list.model().rowsMoved.connect(self._on_rows_moved)
        self._list.currentItemChanged.connect(self._on_current_changed)
        self._list.itemDoubleClicked.connect(lambda _item: self._on_rename())
        layout.addWidget(self._list, stretch=1)

        row = QHBoxLayout()
        self._rename_btn = QPushButton(t("cf_rename_btn"))
        self._rename_btn.clicked.connect(self._on_rename)
        row.addWidget(self._rename_btn)
        self._type = QComboBox()
        for ftype in _TYPES:
            self._type.addItem(t(f"cf_type_{ftype}"), ftype)
        self._type.activated.connect(self._on_type_changed)
        row.addWidget(self._type)
        self._delete_btn = QPushButton(t("cf_delete_btn"))
        self._delete_btn.clicked.connect(self._on_delete)
        row.addWidget(self._delete_btn)
        row.addStretch()
        layout.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self._reload()

    # ------------------------------------------------------------------

    def _reload(self, select_id: Optional[int] = None) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        with session_scope() as session:
            svc = PersonFieldService(session)
            counts = svc.usage_counts()
            for d in svc.list_definitions():
                item = QListWidgetItem(
                    t("cf_list_item", label=d.label, n=counts.get(d.id, 0))
                )
                item.setData(Qt.UserRole, d.id)
                item.setData(Qt.UserRole + 1, d.field_type)
                item.setData(Qt.UserRole + 2, d.label)
                item.setData(Qt.UserRole + 3, counts.get(d.id, 0))
                self._list.addItem(item)
                if d.id == select_id:
                    self._list.setCurrentItem(item)
        self._list.blockSignals(False)
        if self._list.currentItem() is None and self._list.count():
            self._list.setCurrentRow(0)
        self._on_current_changed(self._list.currentItem(), None)

    def _current(self) -> Optional[QListWidgetItem]:
        return self._list.currentItem()

    def _on_current_changed(self, item, _prev) -> None:
        enabled = item is not None
        for w in (self._rename_btn, self._type, self._delete_btn):
            w.setEnabled(enabled)
        if item is not None:
            idx = self._type.findData(item.data(Qt.UserRole + 1))
            self._type.setCurrentIndex(max(idx, 0))

    def _on_rows_moved(self, *_args) -> None:
        ids = [self._list.item(i).data(Qt.UserRole) for i in range(self._list.count())]
        try:
            with session_scope() as session:
                PersonFieldService(session).reorder(ids)
        except Exception:  # noqa: BLE001
            log.exception("Could not reorder person fields")

    def _on_rename(self) -> None:
        item = self._current()
        if item is None:
            return
        fid = item.data(Qt.UserRole)
        label, ok = QInputDialog.getText(
            self, t("cf_rename_btn"), t("cf_rename_prompt"), text=item.data(Qt.UserRole + 2)
        )
        if not ok:
            return
        try:
            with session_scope() as session:
                PersonFieldService(session).rename_definition(fid, label)
        except PersonFieldError:
            QMessageBox.warning(self, t("cf_rename_btn"), t("cf_label_taken", label=label))
            return
        self._reload(select_id=fid)

    def _on_type_changed(self, _index: int) -> None:
        item = self._current()
        if item is None:
            return
        fid = item.data(Qt.UserRole)
        new_type = self._type.currentData()
        try:
            with session_scope() as session:
                PersonFieldService(session).change_type(fid, new_type)
        except PersonFieldError:
            QMessageBox.warning(self, t("cf_manage_title"), t("cf_type_locked"))
        self._reload(select_id=fid)

    def _on_delete(self) -> None:
        item = self._current()
        if item is None:
            return
        fid = item.data(Qt.UserRole)
        n = int(item.data(Qt.UserRole + 3) or 0)
        answer = QMessageBox.question(
            self,
            t("cf_delete_btn"),
            t("cf_delete_confirm", label=item.data(Qt.UserRole + 2), n=n),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        with session_scope() as session:
            PersonFieldService(session).delete_definition(fid)
        self._reload()
