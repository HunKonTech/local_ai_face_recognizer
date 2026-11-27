"""Image browser "save metadata" action: one write per image, locked files reported."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from PIL import Image as PilImage

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.ui.panels.image_browser import image_loading
from app.ui.panels.image_browser.image_loading import ImageLoadingMixin
from app.utils import image_metadata as meta
from app.utils.exif import read_exif_gps


@pytest.fixture()
def setup(tmp_path, monkeypatch):
    init_db(tmp_path / "db.sqlite")
    jpg = tmp_path / "photo.jpg"
    PilImage.new("RGB", (160, 120)).save(jpg, format="JPEG")
    with session_scope() as session:
        person = Person(name="Szilvay Géza", is_auto_named=False)
        session.add(person)
        img = Image(file_path=str(jpg), file_hash="h", file_mtime=0.0, detection_done=True,
                    image_latitude=47.5, image_longitude=19.05, photo_date="1985.07.20")
        session.add(img)
        session.flush()
        session.add(Face(image_id=img.id, person_id=person.id, bbox_x=10, bbox_y=10,
                         bbox_w=40, bbox_h=40, confidence=0.9, detector_backend="cpu",
                         is_excluded=False))
        image_id = img.id

    shown: list[str] = []
    monkeypatch.setattr(image_loading.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: image_loading.QMessageBox.Ok))
    monkeypatch.setattr(image_loading.QMessageBox, "information",
                        staticmethod(lambda _parent, _title, msg: shown.append(msg)))
    panel = SimpleNamespace(
        _current_image_id=image_id,
        _deol_pair_partner_id=None,
        _effective_gps=ImageLoadingMixin._effective_gps,
    )
    return jpg, panel, shown


def test_save_writes_everything_in_one_replace(setup, monkeypatch):
    jpg, panel, shown = setup
    replaces: list[str] = []
    real = os.replace
    monkeypatch.setattr(meta.os, "replace",
                        lambda s, d: (replaces.append(str(d)), real(s, d))[1])

    ImageLoadingMixin._on_save_face_metadata(panel)

    assert replaces == [str(jpg)]
    assert read_exif_gps(jpg) is not None
    assert meta.read_face_metadata(jpg)["faces"][0]["person_name"] == "Szilvay Géza"
    assert shown and "1" in shown[0]


def test_save_reports_locked_file(setup, monkeypatch):
    jpg, panel, shown = setup
    real = os.replace

    def locked(src, dst):
        if str(dst) == str(jpg):
            raise PermissionError("held by viewer")
        return real(src, dst)

    monkeypatch.setattr(meta.os, "replace", locked)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    ImageLoadingMixin._on_save_face_metadata(panel)

    assert meta.sidecar_path_for(jpg).exists()
    assert "photo.jpg" in shown[0]
