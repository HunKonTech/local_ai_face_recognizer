"""Image-data dialog opened from the face-recognition preview panel."""

from __future__ import annotations

import pytest

from app.db.database import init_db, session_scope
from app.db.models import Image, Place
from app.ui.dialogs.image_metadata_dialog import ImageMetadataDialog


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "meta.db")


def _add_image(session, path="/tmp/meta.jpg") -> Image:
    img = Image(file_path=path, file_hash=f"h_{path}", file_mtime=0.0)
    session.add(img)
    session.flush()
    return img


def test_typed_place_and_dates_are_saved(db, qtbot):
    with session_scope() as session:
        image_id = _add_image(session).id

    dlg = ImageMetadataDialog(image_id)
    qtbot.addWidget(dlg)

    dlg._place_search._search.setText("Visegrád")
    dlg._photo_date.setText("1954")
    dlg._note.setPlainText("nagymama háza")
    dlg.accept()

    with session_scope() as session:
        place = session.query(Place).filter(Place.name == "Visegrád").one_or_none()
        assert place is not None, "typed place was not created"
        img = session.get(Image, image_id)
        assert img.place_id == place.id
        assert img.photo_date == "1954"
        assert img.note == "nagymama háza"


def test_existing_place_is_preselected_and_kept(db, qtbot):
    with session_scope() as session:
        place = Place(name="Eger")
        session.add(place)
        session.flush()
        img = _add_image(session, "/tmp/meta2.jpg")
        img.place_id = place.id
        image_id, place_id = img.id, place.id

    dlg = ImageMetadataDialog(image_id)
    qtbot.addWidget(dlg)
    assert dlg._place_search.current_place_id() == place_id

    dlg._photo_date.setText("1930-as évek")
    dlg.accept()

    with session_scope() as session:
        img = session.get(Image, image_id)
        assert img.place_id == place_id, "existing place link was lost"
        assert img.photo_date == "1930-as évek"


def test_remove_place_clears_the_link(db, qtbot):
    with session_scope() as session:
        place = Place(name="Pécs")
        session.add(place)
        session.flush()
        img = _add_image(session, "/tmp/meta3.jpg")
        img.place_id = place.id
        image_id = img.id

    dlg = ImageMetadataDialog(image_id)
    qtbot.addWidget(dlg)
    dlg._clear_place()
    dlg.accept()

    with session_scope() as session:
        assert session.get(Image, image_id).place_id is None
