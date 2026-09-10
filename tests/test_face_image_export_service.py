"""Tests for exporting the faces of an image into separate files (#175)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.services import face_image_export_service as mod
from app.services.face_image_export_service import (
    MODE_ORIGINAL,
    MODE_SQUARE,
    FaceImageExportOptions,
    FaceImageExportService,
)


def _make_test_image(path: Path, width: int = 400, height: int = 300) -> None:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, : width // 2] = (0, 0, 255)
    img[:, width // 2:] = (255, 0, 0)
    assert cv2.imwrite(str(path), img)


def _seed(tmp_path: Path, *, second_person: bool = True) -> Path:
    """Create a DB with one image holding two faces; return the image path."""
    init_db(tmp_path / "face_export.db")
    image_path = tmp_path / "csalad.jpg"
    _make_test_image(image_path)

    with session_scope() as session:
        img = Image(
            file_path=str(image_path),
            file_hash="hash",
            file_mtime=image_path.stat().st_mtime,
            photo_date="2023.06.17",
        )
        merse = Person(
            name="Horváth Merse",
            is_auto_named=False,
            last_name="Horváth",
            first_name="Merse",
            family_code="C44",
        )
        session.add_all([img, merse])
        if second_person:
            geza = Person(
                name="Szilvay Géza (1920-1992)",
                is_auto_named=False,
                last_name="Szilvay",
                first_name="Géza",
            )
            session.add(geza)
        session.flush()

        faces = [
            Face(
                image_id=img.id,
                person=merse,
                bbox_x=20, bbox_y=20, bbox_w=60, bbox_h=80,
                confidence=0.9, detector_backend="cpu",
            ),
            Face(
                image_id=img.id,
                person=geza if second_person else None,
                bbox_x=250, bbox_y=40, bbox_w=50, bbox_h=50,
                confidence=0.8, detector_backend="cpu",
            ),
        ]
        session.add_all(faces)
        session.flush()
    return image_path


def _options(tmp_path: Path, **overrides) -> FaceImageExportOptions:
    opts = FaceImageExportOptions(
        pattern="portré-#CSID#-#Vezetéknév# #Keresztnév#-#Dátum#.jpg",
        target_dir=str(tmp_path / "out"),
    )
    for key, value in overrides.items():
        setattr(opts, key, value)
    return opts


def test_exports_one_file_per_face_with_pattern_names(tmp_path):
    _seed(tmp_path)
    options = _options(tmp_path)

    with session_scope() as session:
        result = FaceImageExportService(session).export_faces_of_images(
            _image_ids(session), options
        )

    assert result.written == 2
    assert result.skipped == 0
    assert result.errors == []
    names = sorted(p.name for p in Path(options.target_dir).iterdir())
    assert names == [
        "portré-C44-Horváth Merse-2023.06.17.jpg",
        "portré-Szilvay Géza-2023.06.17.jpg",
    ]


def test_missing_family_code_collapses_the_separator(tmp_path):
    _seed(tmp_path)
    options = _options(tmp_path)
    with session_scope() as session:
        FaceImageExportService(session).export_faces_of_images(
            _image_ids(session), options
        )
    # The second person has no family code, so no "portré--Szilvay…" name.
    assert (Path(options.target_dir) / "portré-Szilvay Géza-2023.06.17.jpg").exists()


def test_original_mode_keeps_full_resolution_with_margin(tmp_path):
    _seed(tmp_path, second_person=False)
    options = _options(
        tmp_path,
        pattern="#Arc ID#.jpg",
        mode=MODE_ORIGINAL,
        margin_percent=20,
        include_unknown=False,
    )
    with session_scope() as session:
        result = FaceImageExportService(session).export_faces_of_images(
            _image_ids(session), options
        )

    assert result.written == 1
    crop = cv2.imread(result.files[0])
    # bbox 60×80 at (20, 20) with a 20% margin → 12 px / 16 px per side, all
    # of it inside the 400×300 photo, and no resizing.
    assert crop.shape[1] == 60 + 2 * 12
    assert crop.shape[0] == 80 + 2 * 16


def test_original_mode_clamps_the_margin_at_the_image_edge(tmp_path):
    _seed(tmp_path, second_person=False)
    options = _options(
        tmp_path,
        pattern="#Arc ID#.jpg",
        mode=MODE_ORIGINAL,
        margin_percent=50,  # 30 px left of a box starting at x=20 → clipped
        include_unknown=False,
    )
    with session_scope() as session:
        result = FaceImageExportService(session).export_faces_of_images(
            _image_ids(session), options
        )
    crop = cv2.imread(result.files[0])
    assert crop.shape[1] == 20 + 60 + 30  # left edge clamped to 0


def test_square_mode_uses_the_requested_edge_length(tmp_path):
    _seed(tmp_path, second_person=False)
    options = _options(
        tmp_path,
        pattern="#Arc ID#.jpg",
        mode=MODE_SQUARE,
        square_size=256,
        include_unknown=False,
    )
    with session_scope() as session:
        result = FaceImageExportService(session).export_faces_of_images(
            _image_ids(session), options
        )
    crop = cv2.imread(result.files[0])
    assert crop.shape[:2] == (256, 256)


def test_name_collisions_get_a_numeric_suffix(tmp_path):
    _seed(tmp_path)
    options = _options(tmp_path, pattern="portré-#Dátum#.jpg")
    with session_scope() as session:
        result = FaceImageExportService(session).export_faces_of_images(
            _image_ids(session), options
        )
    assert result.written == 2
    names = sorted(p.name for p in Path(options.target_dir).iterdir())
    assert names == ["portré-2023.06.17-2.jpg", "portré-2023.06.17.jpg"]


def test_unknown_faces_can_be_excluded(tmp_path):
    _seed(tmp_path, second_person=False)  # second face has no person
    options = _options(tmp_path, pattern="#Arc ID#.jpg", include_unknown=False)
    with session_scope() as session:
        result = FaceImageExportService(session).export_faces_of_images(
            _image_ids(session), options
        )
    assert result.written == 1


def test_only_face_ids_restricts_the_export_to_one_face(tmp_path):
    _seed(tmp_path)
    with session_scope() as session:
        image_ids = _image_ids(session)
        face_id = session.query(Face.id).order_by(Face.id).first()[0]
    options = _options(tmp_path, pattern="#Arc ID#.jpg", only_face_ids=(face_id,))
    with session_scope() as session:
        result = FaceImageExportService(session).export_faces_of_images(
            image_ids, options
        )
    assert result.written == 1
    assert Path(result.files[0]).name == f"{face_id}.jpg"


def test_excluded_and_low_quality_faces_are_skipped(tmp_path):
    _seed(tmp_path)
    with session_scope() as session:
        first = session.query(Face).order_by(Face.id).first()
        first.is_excluded = True
    options = _options(tmp_path, pattern="#Arc ID#.jpg")
    with session_scope() as session:
        result = FaceImageExportService(session).export_faces_of_images(
            _image_ids(session), options
        )
    assert result.written == 1


def test_each_image_is_decoded_only_once(tmp_path, monkeypatch):
    _seed(tmp_path)
    calls: list[str] = []
    original = mod.load_image_bgr_normalized

    def counting_loader(path):  # noqa: ANN001, ANN202
        calls.append(path)
        return original(path)

    monkeypatch.setattr(mod, "load_image_bgr_normalized", counting_loader)

    options = _options(tmp_path, pattern="#Arc ID#.jpg")
    with session_scope() as session:
        result = FaceImageExportService(session).export_faces_of_images(
            _image_ids(session), options
        )
    assert result.written == 2
    assert len(calls) == 1  # two faces, one decode


def test_missing_source_file_is_reported_not_raised(tmp_path):
    image_path = _seed(tmp_path)
    image_path.unlink()
    options = _options(tmp_path, pattern="#Arc ID#.jpg")
    with session_scope() as session:
        result = FaceImageExportService(session).export_faces_of_images(
            _image_ids(session), options
        )
    assert result.written == 0
    assert result.skipped == 2
    assert result.errors and "not found" in result.errors[0]


def test_preview_names_and_count_match_the_export(tmp_path):
    _seed(tmp_path)
    options = _options(tmp_path)
    with session_scope() as session:
        service = FaceImageExportService(session)
        assert service.count_faces(_image_ids(session)) == 2
        names = service.preview_names(_image_ids(session), options)
    assert names == [
        "portré-C44-Horváth Merse-2023.06.17.jpg",
        "portré-Szilvay Géza-2023.06.17.jpg",
    ]


def test_estimated_date_renders_as_an_approximate_value(tmp_path):
    _seed(tmp_path, second_person=False)
    with session_scope() as session:
        img = session.query(Image).one()
        img.photo_date = None
        img.estimated_date = "1920 körül"
    options = _options(tmp_path, pattern="portré-#Név#-#Dátum#.jpg", include_unknown=False)
    with session_scope() as session:
        result = FaceImageExportService(session).export_faces_of_images(
            _image_ids(session), options
        )
    assert Path(result.files[0]).name == "portré-Horváth Merse-kb. 1920.jpg"


def test_no_faces_yields_an_empty_result(tmp_path):
    init_db(tmp_path / "empty.db")
    options = _options(tmp_path, pattern="#Arc ID#.jpg")
    with session_scope() as session:
        result = FaceImageExportService(session).export_faces_of_images([999], options)
    assert result.written == 0
    assert result.files == []


def _image_ids(session) -> list[int]:
    return [row[0] for row in session.query(Image.id).all()]
