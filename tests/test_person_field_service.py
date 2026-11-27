"""Custom person fields: definitions, values, catalog, merge, delete, export."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person, PersonFieldDefinition, PersonFieldValue
from app.services.identity_service import IdentityService
from app.services.person_field_service import (
    NewField,
    PersonFieldError,
    PersonFieldService,
)


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "fields.db")
    return tmp_path / "fields.db"


def _person(session, name="A"):
    p = Person(name=name, is_auto_named=False)
    session.add(p)
    session.flush()
    return p.id


def test_new_field_with_value(db):
    with session_scope() as s:
        a = _person(s)
        PersonFieldService(s).apply_changes(a, {NewField("Foglalkozás", "text"): "tanító"})
    with session_scope() as s:
        svc = PersonFieldService(s)
        (d,) = svc.list_definitions()
        assert d.label == "Foglalkozás"
        assert svc.values_of(a) == {d.id: "tanító"}


def test_duplicate_label_reuses_existing(db):
    with session_scope() as s:
        svc = PersonFieldService(s)
        d = svc.create_definition("Foglalkozás")
        with pytest.raises(PersonFieldError):
            svc.create_definition(" foglalkozás ")
        assert svc.find_by_label("  FOGLALKOZÁS ").id == d.id
        a = _person(s)
        svc.apply_changes(a, {NewField(" foglalkozás "): "orvos"})
        assert len(svc.list_definitions()) == 1
        assert svc.values_of(a) == {d.id: "orvos"}


def test_reuse_and_value_suggestions(db):
    with session_scope() as s:
        svc = PersonFieldService(s)
        d = svc.create_definition("Foglalkozás")
        a, b = _person(s, "A"), _person(s, "B")
        svc.apply_changes(a, {d.id: "tanító"})
        svc.apply_changes(b, {d.id: "orvos"})
        assert svc.values_of(b) == {d.id: "orvos"}
        assert svc.distinct_values(d.id) == ["orvos", "tanító"]


def test_clear_value_keeps_definition(db):
    with session_scope() as s:
        svc = PersonFieldService(s)
        a = _person(s)
        svc.apply_changes(a, {NewField("Foglalkozás"): "tanító"})
        fid = svc.list_definitions()[0].id
        svc.apply_changes(a, {fid: None})
        assert svc.values_of(a) == {}
        assert len(svc.list_definitions()) == 1


def test_empty_new_field_creates_nothing(db):
    with session_scope() as s:
        svc = PersonFieldService(s)
        a = _person(s)
        svc.apply_changes(a, {NewField("Hobbi"): "  "})
        assert svc.list_definitions() == []


def test_rename_keeps_values_and_key(db):
    with session_scope() as s:
        svc = PersonFieldService(s)
        a = _person(s)
        d = svc.create_definition("Foglalkozás")
        svc.set_value(a, d.id, "tanító")
        key_before = [f.key for f in svc.catalog(lambda k: k) if not f.builtin]
        svc.rename_definition(d.id, "Szakma")
        assert svc.labelled_values(a) == [("Szakma", "tanító")]
        key_after = [f.key for f in svc.catalog(lambda k: k) if not f.builtin]
        assert key_before == key_after == [f"c:{d.id}"]


def test_delete_definition_reports_usage(db):
    with session_scope() as s:
        svc = PersonFieldService(s)
        d = svc.create_definition("Szakma")
        for name in ("A", "B"):
            svc.set_value(_person(s, name), d.id, "x")
        assert svc.usage_count(d.id) == 2
        assert svc.delete_definition(d.id) == 2
        assert svc.list_definitions() == []
        assert s.query(PersonFieldValue).count() == 0


def test_reorder_and_type_change_rules(db):
    with session_scope() as s:
        svc = PersonFieldService(s)
        a = svc.create_definition("A")
        b = svc.create_definition("B", "date")
        svc.reorder([b.id, a.id])
        assert [d.label for d in svc.list_definitions()] == ["B", "A"]
        svc.set_value(_person(s), a.id, "v")
        assert svc.can_change_type(a.id, "long_text")
        assert not svc.can_change_type(a.id, "date")
        assert svc.can_change_type(b.id, "text")  # unused date field
        with pytest.raises(PersonFieldError):
            svc.change_type(a.id, "date")


def test_catalog_and_values_for(db):
    with session_scope() as s:
        svc = PersonFieldService(s)
        a = _person(s)
        s.get(Person, a).nickname = "Pisti"
        d = svc.create_definition("Foglalkozás")
        svc.set_value(a, d.id, "tanító")
        keys = {f.key for f in svc.catalog(lambda k: k)}
        assert {"b:nickname", f"c:{d.id}"} <= keys
        vals = svc.values_for([a])[a]
        assert vals["b:nickname"] == "Pisti"
        assert vals[f"c:{d.id}"] == "tanító"
        assert "b:birth_date" not in vals


def _face(session, pid):
    img = Image(file_path=f"/x/{pid}.jpg", file_hash=str(pid), file_mtime=0.0,
                width=10, height=10, detection_done=True)
    session.add(img)
    session.flush()
    session.add(Face(image_id=img.id, person_id=pid, bbox_x=0, bbox_y=0, bbox_w=5,
                     bbox_h=5, confidence=0.9, detector_backend="cpu"))
    session.flush()


def test_merge_fills_gaps_target_wins(db):
    with session_scope() as s:
        svc = PersonFieldService(s)
        src, tgt = _person(s, "S"), _person(s, "T")
        _face(s, src)
        job = svc.create_definition("Foglalkozás")
        hobby = svc.create_definition("Hobbi")
        svc.set_value(src, job.id, "tanító")
        svc.set_value(src, hobby.id, "horgászat")
        svc.set_value(tgt, job.id, "igazgató")
        s.commit()
        IdentityService(s).merge_persons(source_id=src, target_id=tgt)
    with session_scope() as s:
        svc = PersonFieldService(s)
        assert s.get(Person, src) is None
        assert dict(svc.labelled_values(tgt)) == {
            "Foglalkozás": "igazgató", "Hobbi": "horgászat"}
        assert s.query(PersonFieldValue).filter_by(person_id=src).count() == 0


def test_bulk_move_undo_restores_source_values(db):
    with session_scope() as s:
        src = Person(name="Unknown 1", is_auto_named=True)
        s.add(src)
        s.flush()
        tgt = _person(s, "T")
        _face(s, src.id)
        svc = PersonFieldService(s)
        job = svc.create_definition("Foglalkozás")
        hobby = svc.create_definition("Hobbi")
        svc.set_value(src.id, job.id, "tanító")
        svc.set_value(src.id, hobby.id, "horgászat")
        s.commit()
        src_id = src.id
        face_ids = [f.id for f in s.query(Face).filter_by(person_id=src_id)]
        result = IdentityService(s).reassign_faces_bulk(face_ids, tgt)
    with session_scope() as s:
        assert s.get(Person, src_id) is None
        IdentityService(s).restore_face_assignments(result.snapshots, result.removed_persons)
    with session_scope() as s:
        assert dict(PersonFieldService(s).labelled_values(src_id)) == {
            "Foglalkozás": "tanító", "Hobbi": "horgászat"}


def test_delete_person_removes_values(db):
    with session_scope() as s:
        a = _person(s)
        svc = PersonFieldService(s)
        svc.set_value(a, svc.create_definition("X").id, "v")
        s.commit()
        IdentityService(s).delete_person(a)
    with session_scope() as s:
        assert s.query(PersonFieldValue).count() == 0


def test_old_database_gets_tables(tmp_path):
    path = tmp_path / "old.db"
    init_db(path)
    with session_scope() as s:
        s.add(Person(name="Old", is_auto_named=False, nickname="N"))
    con = sqlite3.connect(path)
    con.execute("DROP TABLE person_field_values")
    con.execute("DROP TABLE person_field_definitions")
    con.commit()
    con.close()
    init_db(path)
    with session_scope() as s:
        p = s.query(Person).filter_by(name="Old").one()
        assert p.nickname == "N"
        PersonFieldService(s).apply_changes(p.id, {NewField("Új"): "x"})
        assert s.query(PersonFieldDefinition).count() == 1


def test_export_includes_custom_fields(db, tmp_path):
    from app.services.export_service import ExportService

    with session_scope() as s:
        a = _person(s, "Alice")
        PersonFieldService(s).apply_changes(a, {NewField("Foglalkozás"): "tanító"})
    project = tmp_path / "astro"
    (project / "src" / "data").mkdir(parents=True)
    with session_scope() as s:
        data_dir = ExportService(s).export_astro(
            str(tmp_path / "out"), run_build=False, astro_project_dir=str(project))
    persons = json.loads((Path(data_dir) / "persons.json").read_text(encoding="utf-8"))
    rec = next(p for p in persons if p["name"] == "Alice")
    assert rec["customFields"] == [{"label": "Foglalkozás", "value": "tanító"}]
