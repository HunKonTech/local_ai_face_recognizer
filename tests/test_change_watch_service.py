"""Cross-process change log: triggers, origin filtering, cursor, pruning."""

from __future__ import annotations

import time

import pytest
from sqlalchemy import create_engine, event, text

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person, Relationship
from app.services.change_watch_service import (
    PROCESS_ORIGIN,
    ChangeWatchService,
    get_change_watch_service,
    install_change_triggers,
)
from app.services.person_field_service import PersonFieldService

OTHER = "0ther0rigin"


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "cw.db"
    init_db(path)
    return path


def _other_engine(path):
    """Second "process": own engine whose triggers carry another origin."""
    eng = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})

    @event.listens_for(eng, "connect")
    def _on_connect(dbapi_conn, _rec):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()
        install_change_triggers(dbapi_conn, origin=OTHER)

    return eng


def _log_rows():
    with session_scope() as s:
        return s.execute(
            text("SELECT table_name, row_id, op, origin FROM change_log ORDER BY id")
        ).all()


def _person(name="A") -> int:
    with session_scope() as s:
        p = Person(name=name, is_auto_named=False)
        s.add(p)
        s.flush()
        return p.id


def test_writes_are_logged_with_own_origin(db):
    pid = _person()
    other = _person("B")
    with session_scope() as s:
        s.execute(text("DELETE FROM change_log"))
    with session_scope() as s:
        s.get(Person, pid).name = "Renamed"
    with session_scope() as s:
        s.add(Relationship(relationship_type="Spouse", person_a_id=pid, person_b_id=other))
    with session_scope() as s:
        svc = PersonFieldService(s)
        svc.set_value(pid, svc.create_definition("Foglalkozás").id, "tanító")
    rows = _log_rows()
    tables = [r[0] for r in rows]
    assert ("persons", pid, "U", PROCESS_ORIGIN) in rows
    assert "relationships" in tables
    assert ("person_field_values", pid, "I", PROCESS_ORIGIN) in rows
    assert {r[3] for r in rows} == {PROCESS_ORIGIN}


def test_face_reassignment_is_not_logged(db):
    pid = _person()
    with session_scope() as s:
        img = Image(file_path="/x.jpg", file_hash="h", file_mtime=0.0)
        s.add(img)
        s.flush()
        s.add(Face(image_id=img.id, bbox_x=0, bbox_y=0, bbox_w=5, bbox_h=5,
                   confidence=0.9, detector_backend="cpu"))
    with session_scope() as s:
        s.execute(text("DELETE FROM change_log"))
    with session_scope() as s:
        s.query(Face).update({Face.person_id: pid})
    assert _log_rows() == []


def test_triggers_on_every_pooled_connection(db):
    from app.db import database

    eng = database._engine
    with eng.connect() as c1, eng.connect() as c2:
        for c in (c1, c2):
            n = c.execute(text(
                "SELECT COUNT(*) FROM sqlite_temp_master WHERE type='trigger'")).scalar()
            assert n >= 12


def test_other_process_changes_detected_grouped_and_own_ignored(db):
    pid = _person("A")
    pid2 = _person("B")
    mine = get_change_watch_service()
    assert mine is not None
    mine.reset_cursor()

    # Own write: never reported.
    with session_scope() as s:
        s.get(Person, pid).nickname = "own"
    assert not mine.changes_since_cursor()

    # Other process: 2 persons (one twice) + 1 relationship.
    other = _other_engine(db)
    with other.begin() as conn:
        conn.execute(text("UPDATE persons SET nickname='x' WHERE id=:i"), {"i": pid})
        conn.execute(text("UPDATE persons SET nickname='y' WHERE id=:i"), {"i": pid})
        conn.execute(text("UPDATE persons SET nickname='z' WHERE id=:i"), {"i": pid2})
        conn.execute(text(
            "INSERT INTO relationships (relationship_type, person_a_id, person_b_id, created_at, updated_at) "
            "VALUES ('Spouse', :a, :b, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"), {"a": pid, "b": pid2})
    changes = mine.changes_since_cursor()
    assert changes.counts() == {"person": 2, "relationship": 1}
    # Cursor advanced: nothing new the second time.
    assert not mine.changes_since_cursor()

    # The other process never sees its own writes, but sees ours.
    theirs = ChangeWatchService(other, origin=OTHER)
    with session_scope() as s:
        s.get(Person, pid2).nickname = "from A"
    assert theirs.changes_since_cursor().counts() == {"person": 1}
    other.dispose()


def test_prune_removes_entries_older_than_30_days(db):
    with session_scope() as s:
        s.execute(text("DELETE FROM change_log"))
        old = time.time() - 31 * 24 * 3600
        s.execute(text(
            "INSERT INTO change_log (table_name, row_id, op, origin, changed_at) "
            "VALUES ('persons', 1, 'U', 'x', :t), ('persons', 2, 'U', 'x', :n)"),
            {"t": old, "n": time.time()})
    svc = get_change_watch_service()
    assert svc.prune() == 1
    assert len(_log_rows()) == 1


def test_old_database_gets_change_log(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    init_db(path)
    con = sqlite3.connect(path)
    con.execute("DROP TABLE change_log")
    con.commit()
    con.close()
    init_db(path)
    pid = _person()
    assert ("persons", pid, "I", PROCESS_ORIGIN) in _log_rows()


def _fake_window(qtbot):
    from PySide6.QtWidgets import QMainWindow, QTabWidget

    from app.ui.main_window.external_changes import ExternalChangesMixin

    class _Win(ExternalChangesMixin, QMainWindow):
        def __init__(self):
            super().__init__()
            self._tabs = QTabWidget()
            self.refreshed = 0
            self._setup_external_change_watch()

        def _refresh_persons(self):
            self.refreshed += 1

    win = _Win()
    qtbot.addWidget(win)
    return win


def test_window_shows_bar_for_foreign_change_only(db, qtbot, monkeypatch):
    import app.ui.main_window.external_changes as ext

    monkeypatch.setattr(ext, "auto_refresh_enabled", lambda: False)
    pid = _person()
    get_change_watch_service().reset_cursor()
    win = _fake_window(qtbot)

    with session_scope() as s:  # own edit
        s.get(Person, pid).nickname = "own"
    win._check_external_changes()
    assert win._external_change_bar.isHidden()

    other = _other_engine(db)
    with other.begin() as conn:
        conn.execute(text("UPDATE persons SET name='Ext' WHERE id=:i"), {"i": pid})
    win._check_external_changes()
    assert not win._external_change_bar.isHidden()
    win._external_change_bar._refresh_btn.click()
    assert win.refreshed == 1
    assert win._external_change_bar.isHidden()
    other.dispose()


def test_window_auto_refresh_skips_bar(db, qtbot, monkeypatch):
    import app.ui.main_window.external_changes as ext

    monkeypatch.setattr(ext, "auto_refresh_enabled", lambda: True)
    pid = _person()
    get_change_watch_service().reset_cursor()
    win = _fake_window(qtbot)
    other = _other_engine(db)
    with other.begin() as conn:
        conn.execute(text("UPDATE persons SET name='Ext' WHERE id=:i"), {"i": pid})
    win._check_external_changes()
    assert win._external_change_bar.isHidden()
    assert win.refreshed == 1
    other.dispose()


def test_real_second_process_is_detected(db, tmp_path):
    import os
    import subprocess
    import sys

    pid = _person()
    mine = get_change_watch_service()
    mine.reset_cursor()
    script = tmp_path / "child.py"
    script.write_text(
        "import sys\n"
        "from app.db.database import init_db, session_scope\n"
        "from app.db.models import Person\n"
        "init_db(sys.argv[1])\n"
        "with session_scope() as s:\n"
        "    s.get(Person, int(sys.argv[2])).name = 'P2'\n",
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    subprocess.run([sys.executable, str(script), str(db), str(pid)],
                   check=True, timeout=120, env=env)
    assert mine.changes_since_cursor().counts() == {"person": 1}
