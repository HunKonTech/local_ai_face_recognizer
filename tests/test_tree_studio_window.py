"""Tree Studio window end-to-end (offscreen, temp DB, no modal dialogs)."""

from __future__ import annotations

import json

import pytest

from app.db.database import init_db, session_scope
from app.db.models import Person
from app.services.family_service import FamilyService
from app.services.person_field_service import PersonFieldService
from app.services.tree_view_service import TreeViewService


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "studio.db")
    return tmp_path / "studio.db"


def _seed():
    with session_scope() as s:
        def mk(name, ext=None, gender=None):
            p = Person(name=name, is_auto_named=False, external_family_code=ext, gender=gender)
            s.add(p)
            s.flush()
            return p.id

        ids = {k: mk(k, ext) for k, ext in (
            ("Root", "#Kiss#"), ("RootWife", None), ("Son", None), ("SonWife", None),
            ("Grandkid", None), ("Daughter", None), ("Stranger", None), ("Kiss Béla", None),
        )}
        fam = FamilyService(s)
        fam.add_spouse(ids["Root"], ids["RootWife"])
        for parent in ("Root", "RootWife"):
            fam.add_parent_child(ids[parent], ids["Son"])
            fam.add_parent_child(ids[parent], ids["Daughter"])
        fam.add_spouse(ids["Son"], ids["SonWife"])
        fam.add_parent_child(ids["Son"], ids["Grandkid"])
        fam.add_parent_child(ids["SonWife"], ids["Grandkid"])
        svc = PersonFieldService(s)
        job = svc.create_definition("Foglalkozás")
        svc.set_value(ids["Son"], job.id, "tanító")
        s.get(Person, ids["Son"]).nickname = "Pisti"
        return ids, job.id


@pytest.fixture()
def win(db, qtbot):
    from app.tree_studio.window import TreeStudioWindow

    w = TreeStudioWindow()
    qtbot.addWidget(w)
    return w


def _box_text(win, pid):
    from PySide6.QtWidgets import QGraphicsSimpleTextItem

    group = win.tree._node_items[pid]
    return [c.text() for c in group.childItems() if isinstance(c, QGraphicsSimpleTextItem)]


def test_root_by_code_collapse_hide_fields(win):
    ids, job = _seed()
    win.reload_persons()
    win.new_view("#Kiss#")
    assert win.root_id == ids["Root"]
    assert win.visible == set(ids.values()) - {ids["Stranger"], ids["Kiss Béla"]}

    # Collapse Son: spouse stays, grandkid hidden; badge marks it.
    win.toggle_collapse(ids["Son"])
    assert ids["SonWife"] in win.visible and ids["Grandkid"] not in win.visible
    assert ids["Son"] in win.tree._badge_items
    win.toggle_collapse(ids["Son"])
    assert ids["Grandkid"] in win.visible

    # Hide a spouse and show again.
    win.hide_person(ids["SonWife"])
    assert ids["SonWife"] not in win.visible and ids["Son"] in win.visible
    win.show_person(ids["SonWife"])
    assert ids["SonWife"] in win.visible

    # Fields per view + per-person override.
    win.state.fields = ["b:nickname", f"c:{job}"]
    win._changed()
    assert _box_text(win, ids["Son"]) == ["Son", "Pisti", "tanító"]
    assert _box_text(win, ids["Daughter"]) == ["Daughter"]
    win.set_person_field(ids["Son"], f"c:{job}", False)
    assert _box_text(win, ids["Son"]) == ["Son", "Pisti"]

    # Person list search + on-chart marker.
    win._person_search.setText("kiss")
    assert win.listed_names() == ["Kiss Béla"]
    win._person_search.setText("")
    bold = {win._person_list.item(i).text() for i in range(win._person_list.count())
            if win._person_list.item(i).font().bold()}
    assert "Son" in bold and "Stranger" not in bold


def test_drops_reveal_relate_float_and_connect(win):
    ids, _job = _seed()
    win.new_view(str(ids["Root"]))

    win.toggle_collapse(ids["Son"])
    win.hide_person(ids["Grandkid"])
    assert win.handle_drop(ids["Grandkid"], None) == "revealed"
    assert ids["Grandkid"] in win.visible

    assert "sibling" not in win.available_relations(ids["Root"])
    assert "sibling" in win.available_relations(ids["Son"])

    assert win.handle_drop(ids["Stranger"], None) == "floating"
    assert ids["Stranger"] in win.visible and ids["Stranger"] in win.state.floating

    assert win.handle_drop(ids["Stranger"], ids["Daughter"], "spouse") == "spouse"
    assert ids["Stranger"] not in win.state.floating
    with session_scope() as s:
        assert FamilyService(s).are_spouses(ids["Daughter"], ids["Stranger"])

    assert win.handle_drop(ids["Kiss Béla"], ids["Son"], "sibling") == "sibling"
    with session_scope() as s:
        fam = FamilyService(s)
        assert fam.is_parent_child(ids["Root"], ids["Kiss Béla"])
    assert ids["Kiss Béla"] in win.visible


def test_save_reload_and_json_round_trip(win, qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QInputDialog, QMessageBox

    ids, job = _seed()
    win.new_view("#Kiss#")
    win.toggle_collapse(ids["Son"])
    win.hide_person(ids["Daughter"])
    win.state.fields = ["b:nickname"]
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Nagyszülők", True))
    assert win.save_view_as()
    saved = win.state.to_json()

    from app.tree_studio.window import TreeStudioWindow

    other = TreeStudioWindow()
    qtbot.addWidget(other)
    other.open_view(win.view_id)
    assert other.state.to_json() == saved
    assert other.visible == win.visible

    path = tmp_path / "v.json"
    monkeypatch.setattr("app.tree_studio.views.QFileDialog.getSaveFileName",
                        lambda *a, **k: (str(path), ""))
    win.export_view_json()
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["format"] == "facelocal.treeview.v1"
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    other.import_view_json(str(path))
    assert other.view_name == "Nagyszülők (2)"
    assert other.state.to_json() == saved
    with session_scope() as s:
        assert len(TreeViewService(s).list_views()) == 2


def test_export_image_from_window(win, tmp_path):
    from app.ui.widgets.family_tree.scene_export import export_scene

    ids, _ = _seed()
    win.new_view("#Kiss#")
    out = export_scene(win.tree, tmp_path / "tree.svg")
    text = out.read_text(encoding="utf-8")
    assert "Grandkid" in text


def test_external_change_detected(win, db, monkeypatch):
    from sqlalchemy import create_engine, event, text

    import app.tree_studio.external as ext
    from app.services.change_watch_service import get_change_watch_service, install_change_triggers

    ids, _ = _seed()
    win.new_view("#Kiss#")
    get_change_watch_service().reset_cursor()
    monkeypatch.setattr(ext, "auto_refresh_enabled", lambda: True)
    eng = create_engine(f"sqlite:///{db}")
    event.listen(eng, "connect", lambda c, _r: install_change_triggers(c, origin="faceLocal"))
    with eng.begin() as conn:
        conn.execute(text("UPDATE persons SET name='Renamed' WHERE id=:i"), {"i": ids["Son"]})
    eng.dispose()
    win.check_external_changes()
    assert _box_text(win, ids["Son"])[0] == "Renamed"


def test_launch_command_and_main_flag():
    from app.tree_studio.main import launch_command, parse_args

    cmd = launch_command(db_path="x.db")
    assert cmd[-2:] == ["--db", "x.db"]
    assert "app.tree_studio" in cmd or "--tree-studio" in cmd
    args = parse_args(["--db", "y.db", "--view", "V"])
    assert args.db == "y.db" and args.view == "V"


def test_collapse_badge_is_couple_aware(win):
    ids, _ = _seed()
    win.new_view("#Kiss#")
    win.toggle_collapse(ids["Son"])
    assert win.tree._collapsed >= {ids["Son"], ids["SonWife"]}
    # Expanding via the spouse's badge expands the couple's children.
    win.toggle_collapse(ids["SonWife"])
    assert ids["Grandkid"] in win.visible
    assert not win.state.collapsed


def test_language_switch_retranslates(win, monkeypatch):
    import app.ui.i18n as i18n

    monkeypatch.setattr(i18n, "set_language", lambda lang: setattr(i18n, "_lang", lang))
    monkeypatch.setattr(i18n, "_lang", "en")
    win._set_language("hu")
    assert win._act_new.text() == "Új nézet…"
    assert win._persons_dock.windowTitle() == "Személyek"
    assert win._view_menu.title() == "&Nézet"
    win._set_language("en")
    assert win._act_new.text() == "New view…"
