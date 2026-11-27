"""Offscreen tests for the custom-field section of the person editor."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QLineEdit

from app.db.database import init_db, session_scope
from app.db.models import Person, PersonFieldDefinition
from app.services.person_field_service import NewField, PersonFieldService
from app.ui.dialogs.person_info_dialog import PersonInfoDialog, save_person_dialog


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "ui.db")


def _make(name="A") -> int:
    with session_scope() as s:
        p = Person(name=name, is_auto_named=False)
        s.add(p)
        s.flush()
        return p.id


def _dialog(qtbot, pid):
    with session_scope() as s:
        dlg = PersonInfoDialog(s.get(Person, pid))
    qtbot.addWidget(dlg)
    return dlg


def _pick(editor, label):
    idx = editor._picker.findText(label)
    assert idx >= 0, label
    editor._on_pick(idx)


def test_add_new_field_then_save(db, qtbot):
    pid = _make()
    dlg = _dialog(qtbot, pid)
    ed = dlg._custom_fields
    assert ed._rows == []  # unfilled fields hidden
    _pick(ed, ed._picker.itemText(ed._picker.count() - 1))  # "New field…"
    ed._new_label.setText("Foglalkozás")
    ed._on_new_ok()
    row = ed._rows[0]
    row.editor.setText("tanító")
    changes = dlg.custom_field_changes()
    assert changes == {NewField("Foglalkozás", "text"): "tanító"}
    with session_scope() as s:
        save_person_dialog(pid, dlg, s)
    with session_scope() as s:
        assert PersonFieldService(s).labelled_values(pid) == [("Foglalkozás", "tanító")]


def test_cancel_stores_nothing(db, qtbot):
    pid = _make()
    dlg = _dialog(qtbot, pid)
    ed = dlg._custom_fields
    _pick(ed, ed._picker.itemText(ed._picker.count() - 1))
    ed._new_label.setText("Hobbi")
    ed._on_new_ok()
    ed._rows[0].editor.setText("horgászat")
    dlg.reject()
    with session_scope() as s:
        assert s.query(PersonFieldDefinition).count() == 0


def test_existing_field_reused_and_cleared(db, qtbot):
    a, b = _make("A"), _make("B")
    with session_scope() as s:
        svc = PersonFieldService(s)
        fid = svc.create_definition("Foglalkozás").id
        svc.set_value(a, fid, "tanító")

    # Person B: pick existing field, value suggestion available.
    dlg = _dialog(qtbot, b)
    ed = dlg._custom_fields
    _pick(ed, "Foglalkozás")
    ed.set_suggestions({fid: ["tanító"]})
    editor = ed._rows[0].editor
    assert isinstance(editor, QLineEdit)
    assert editor.completer() is not None
    editor.setText("orvos")
    assert dlg.custom_field_changes() == {fid: "orvos"}

    # Duplicate label in "New field…" swaps to the existing definition.
    dlg2 = _dialog(qtbot, b)
    ed2 = dlg2._custom_fields
    _pick(ed2, ed2._picker.itemText(ed2._picker.count() - 1))
    ed2._new_label.setText(" foglalkozás ")
    ed2._on_new_ok()
    assert [r.ref for r in ed2._rows] == [fid]

    # Person A: clear the existing value.
    dlg3 = _dialog(qtbot, a)
    ed3 = dlg3._custom_fields
    assert [r.value() for r in ed3._rows] == ["tanító"]
    ed3._remove_row(ed3._rows[0])
    assert dlg3.custom_field_changes() == {fid: None}
    with session_scope() as s:
        save_person_dialog(a, dlg3, s)
    with session_scope() as s:
        svc = PersonFieldService(s)
        assert svc.values_of(a) == {}
        assert len(svc.list_definitions()) == 1
