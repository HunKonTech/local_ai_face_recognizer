"""Smoke tests for the main application window."""

from __future__ import annotations

import pytest

from app.config import AppConfig
from app.db.database import init_db
from app.ui.main_window import MainWindow


@pytest.fixture()
def config(tmp_path):
    db_path = tmp_path / "main_window.db"
    init_db(db_path)
    cfg = AppConfig()
    cfg.storage.db_path = str(db_path)
    cfg.storage.crops_dir = str(tmp_path / "crops")
    cfg.crops_dir_resolved.mkdir(parents=True, exist_ok=True)
    return cfg


def _stub_heavy_startup(monkeypatch) -> None:
    """Prevent network / background workers from starting during smoke tests."""
    for name in (
        "_start_update_check",
        "_start_crop_repair_task",
        "_check_image_library_on_startup",
        "_restore_last_folder",
        "_setup_gdrive_chip",
    ):
        monkeypatch.setattr(MainWindow, name, lambda self, _n=name: None)


def test_main_window_construct(config, qtbot, monkeypatch):
    _stub_heavy_startup(monkeypatch)
    window = MainWindow(config=config)
    qtbot.addWidget(window)
    assert window._config is config
    assert window._db_path == str(config.db_path_resolved)
    window.close()


def test_path_repair_dialog_opens_with_the_current_folders(
    config, qtbot, monkeypatch, tmp_path
):
    """The missing-image review window prefills the library / scan folders."""
    _stub_heavy_startup(monkeypatch)
    window = MainWindow(config=config)
    qtbot.addWidget(window)

    folder = tmp_path / "photos"
    folder.mkdir()
    window._root_folders = [str(folder)]

    window._open_path_repair_dialog()
    dialog = window._path_repair_dialog
    qtbot.addWidget(dialog)

    assert str(folder) in dialog._search_roots()
    dialog.close()
    window.close()


def test_scan_triggers_object_search_on_new_images(config, qtbot, monkeypatch):
    """After a scan, tagged objects are searched on the new images only (#164)."""
    from app.db.database import session_scope
    from app.db.models import Image
    from app.services.object_service import ObjectService

    _stub_heavy_startup(monkeypatch)
    # Keep the user's real Objects-tab checkboxes out of the test.
    monkeypatch.setattr(
        "app.ui.panels.objects_panel.matching_preferences", lambda base: base
    )
    window = MainWindow(config=config)
    qtbot.addWidget(window)
    calls = []
    monkeypatch.setattr(
        window, "_start_object_search", lambda **kwargs: calls.append(kwargs)
    )

    # No object has a frame yet: nothing to look for.
    window._maybe_search_objects_after_scan([1])
    assert calls == []

    with session_scope() as session:
        image = Image(file_path="/a.jpg", file_hash="a", file_mtime=0.0)
        session.add(image)
        session.flush()
        objects = ObjectService(session)
        obj = objects.create_object(name="Asztal")
        objects.add_occurrence_bbox(obj.id, image.id, 1, 1, 50, 40)

    window._maybe_search_objects_after_scan([])
    assert calls == []

    window._maybe_search_objects_after_scan([4, 5])
    assert calls == [{"object_id": None, "image_ids": [4, 5], "post_scan": True}]

    config.object_matching.search_new_images = False
    window._maybe_search_objects_after_scan([6])
    config.object_matching.search_new_images = True
    config.object_matching.enabled = False
    window._maybe_search_objects_after_scan([6])
    assert len(calls) == 1
    window.close()
