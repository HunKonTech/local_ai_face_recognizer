"""Custom person field section of the person editor (``add-dynamic-person-fields``).

Shows one row per custom field that has a value for the person, an
"add field" picker listing unused definitions plus "New field…", and an
inline form for creating a field. Nothing is written to the DB here: the
caller reads :meth:`CustomFieldsEditor.changes` after the dialog is accepted
and passes it to :meth:`PersonFieldService.apply_changes` (single save path).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.services.person_field_service import (
    NewField,
    PersonFieldService,
    normalize_label,
)
from app.ui.i18n import t

log = logging.getLogger(__name__)

_NEW = "__new__"
_TYPES = ("text", "long_text", "date")


@dataclass
class _Def:
    id: int
    label: str
    field_type: str


@dataclass
class _Row:
    ref: Union[int, NewField]
    label: str
    field_type: str
    editor: QWidget
    label_widget: QLabel
    container: QWidget
    original: str = ""

    def value(self) -> str:
        if isinstance(self.editor, QPlainTextEdit):
            return self.editor.toPlainText().strip()
        return self.editor.text().strip()


class CustomFieldsEditor(QWidget):
    """Editable custom-field rows for one person."""

    def __init__(self, person_id: Optional[int], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._person_id = person_id
        self._defs: dict[int, _Def] = {}
        self._rows: list[_Row] = []
        self._cleared: set[int] = set()
        self._suggestions: dict[int, list[str]] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._form = QFormLayout()
        self._form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        layout.addLayout(self._form)

        picker_row = QHBoxLayout()
        self._picker = QComboBox()
        self._picker.activated.connect(self._on_pick)
        picker_row.addWidget(self._picker, stretch=1)
        self._manage_btn = QPushButton(t("cf_manage_btn"))
        self._manage_btn.clicked.connect(self._on_manage)
        picker_row.addWidget(self._manage_btn)
        layout.addLayout(picker_row)

        # Inline "new field" form (hidden until "New field…" is picked).
        self._new_box = QWidget()
        new_row = QHBoxLayout(self._new_box)
        new_row.setContentsMargins(0, 0, 0, 0)
        self._new_label = QLineEdit()
        self._new_label.setPlaceholderText(t("cf_new_label_placeholder"))
        self._new_label.returnPressed.connect(self._on_new_ok)
        new_row.addWidget(self._new_label, stretch=1)
        self._new_type = QComboBox()
        for ftype in _TYPES:
            self._new_type.addItem(t(f"cf_type_{ftype}"), ftype)
        new_row.addWidget(self._new_type)
        ok = QPushButton(t("cf_new_ok"))
        ok.clicked.connect(self._on_new_ok)
        new_row.addWidget(ok)
        cancel = QToolButton()
        cancel.setText("✕")
        cancel.clicked.connect(lambda: self._new_box.setVisible(False))
        new_row.addWidget(cancel)
        self._new_box.setVisible(False)
        layout.addWidget(self._new_box)

        self._hint = QLabel()
        self._hint.setStyleSheet("color: #c08000; font-size: 11px;")
        self._hint.setVisible(False)
        layout.addWidget(self._hint)

        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        values: dict[int, str] = {}
        try:
            with session_scope() as session:
                svc = PersonFieldService(session)
                self._defs = {
                    d.id: _Def(d.id, d.label, d.field_type)
                    for d in svc.list_definitions()
                }
                if self._person_id is not None:
                    values = svc.values_of(self._person_id)
        except Exception:  # noqa: BLE001
            log.exception("Could not load custom person fields")
        for fid, d in self._defs.items():
            if fid in values:
                self._add_row(fid, d.label, d.field_type, values[fid])
        self._rebuild_picker()

    def set_suggestions(self, suggestions: dict[int, list[str]]) -> None:
        """Apply per-field value completers (loaded off the UI thread)."""
        self._suggestions = dict(suggestions)
        for row in self._rows:
            self._apply_completer(row)

    def _apply_completer(self, row: _Row) -> None:
        if not isinstance(row.ref, int) or not isinstance(row.editor, QLineEdit):
            return
        values = self._suggestions.get(row.ref)
        if not values:
            return
        completer = QCompleter(values, row.editor)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        row.editor.setCompleter(completer)

    # ------------------------------------------------------------------
    # Rows
    # ------------------------------------------------------------------

    def _add_row(self, ref, label: str, field_type: str, value: str = "") -> _Row:
        if field_type == "long_text":
            editor: QWidget = QPlainTextEdit(value)
            editor.setFixedHeight(60)
        else:
            editor = QLineEdit(value)
            if field_type == "date":
                editor.setPlaceholderText(t("example_birth_date"))
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(editor, stretch=1)
        clear = QToolButton()
        clear.setText("✕")
        clear.setToolTip(t("cf_clear_tip"))
        h.addWidget(clear)
        label_widget = QLabel(label)
        self._form.addRow(label_widget, container)
        row = _Row(ref, label, field_type, editor, label_widget, container, value)
        clear.clicked.connect(lambda: self._remove_row(row))
        self._rows.append(row)
        if isinstance(ref, int):
            self._cleared.discard(ref)
        self._apply_completer(row)
        return row

    def _remove_row(self, row: _Row) -> None:
        if isinstance(row.ref, int) and row.original:
            self._cleared.add(row.ref)
        self._rows.remove(row)
        self._form.removeRow(row.container)
        self._rebuild_picker()

    def _row_for(self, ref) -> Optional[_Row]:
        for row in self._rows:
            if row.ref == ref:
                return row
        return None

    # ------------------------------------------------------------------
    # Picker / new field
    # ------------------------------------------------------------------

    def _rebuild_picker(self) -> None:
        self._picker.blockSignals(True)
        self._picker.clear()
        self._picker.addItem(t("cf_add_field"), None)
        used = {r.ref for r in self._rows}
        for fid, d in self._defs.items():
            if fid not in used:
                self._picker.addItem(d.label, fid)
        self._picker.addItem(t("cf_new_field"), _NEW)
        self._picker.setCurrentIndex(0)
        self._picker.blockSignals(False)

    def _on_pick(self, index: int) -> None:
        data = self._picker.itemData(index)
        self._picker.setCurrentIndex(0)
        if data is None:
            return
        if data == _NEW:
            self._hint.setVisible(False)
            self._new_label.clear()
            self._new_box.setVisible(True)
            self._new_label.setFocus()
            return
        d = self._defs.get(int(data))
        if d is not None:
            row = self._add_row(d.id, d.label, d.field_type)
            row.editor.setFocus()
            self._rebuild_picker()

    def _on_new_ok(self) -> None:
        label = " ".join(self._new_label.text().split())
        if not label:
            return
        key = normalize_label(label)
        self._new_box.setVisible(False)
        existing = next(
            (d for d in self._defs.values() if normalize_label(d.label) == key), None
        )
        if existing is not None:
            # Duplicate label: swap to the existing definition.
            row = self._row_for(existing.id) or self._add_row(
                existing.id, existing.label, existing.field_type
            )
            self._hint.setText(t("cf_existing_used", label=existing.label))
            self._hint.setVisible(True)
            row.editor.setFocus()
            self._rebuild_picker()
            return
        pending = next(
            (r for r in self._rows
             if isinstance(r.ref, NewField) and normalize_label(r.ref.label) == key),
            None,
        )
        if pending is None:
            ftype = self._new_type.currentData() or "text"
            pending = self._add_row(NewField(label, ftype), label, ftype)
        pending.editor.setFocus()

    def _on_manage(self) -> None:
        from app.ui.dialogs.person_fields_dialog import PersonFieldsDialog

        PersonFieldsDialog(parent=self).exec()
        self._reload_definitions()

    def _reload_definitions(self) -> None:
        """Pick up renames / deletes / reorders made in the manage dialog."""
        try:
            with session_scope() as session:
                self._defs = {
                    d.id: _Def(d.id, d.label, d.field_type)
                    for d in PersonFieldService(session).list_definitions()
                }
        except Exception:  # noqa: BLE001
            log.exception("Could not reload custom person fields")
            return
        for row in list(self._rows):
            if not isinstance(row.ref, int):
                continue
            d = self._defs.get(row.ref)
            if d is None:
                self._rows.remove(row)
                self._form.removeRow(row.container)
                self._cleared.discard(row.ref)
            else:
                row.label = d.label
                row.label_widget.setText(d.label)
        self._rebuild_picker()

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def changes(self) -> dict:
        """``{field id | NewField: value | None}`` for everything that changed."""
        out: dict = {fid: None for fid in self._cleared}
        for row in self._rows:
            value = row.value()
            if isinstance(row.ref, NewField):
                if value:
                    out[row.ref] = value
            elif value != row.original:
                out[row.ref] = value or None
        return out
