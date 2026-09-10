"""Image metadata dialog — set where and when a photo was taken.

Opened from the face-recognition preview panel (button or right-click on the
image) so metadata can be edited without cluttering that panel with extra
input fields.  Edits the same ``Image`` columns as the image browser: linked
place, photo date, estimated date and free-text note.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.db.models import Image, Place
from app.services.place_service import PlaceService
from app.ui.i18n import t
from app.ui.widgets.place_search_select import PlaceSearchSelect

log = logging.getLogger(__name__)


class ImageMetadataDialog(QDialog):
    """Edit place / date / note metadata of a single image."""

    def __init__(self, image_id: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._image_id = image_id
        self._place_id: Optional[int] = None
        self.setWindowTitle(t("imeta_title"))
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._file_label = QLabel("")
        self._file_label.setWordWrap(True)
        self._file_label.setStyleSheet("QLabel { color: #aaa; font-size: 11px; }")
        layout.addWidget(self._file_label)

        # ── Place ─────────────────────────────────────────────────────────
        place_hdr = QLabel(t("imeta_place_hdr"))
        place_hdr.setStyleSheet("font-weight: bold;")
        layout.addWidget(place_hdr)

        self._place_label = QLabel("")
        self._place_label.setWordWrap(True)
        layout.addWidget(self._place_label)

        self._place_search = PlaceSearchSelect()
        self._place_search.place_selected.connect(self._on_place_selected)
        self._place_search.create_requested.connect(self._on_place_create_requested)
        layout.addWidget(self._place_search)

        hint = QLabel(t("imeta_place_hint"))
        hint.setWordWrap(True)
        hint.setStyleSheet("QLabel { color: #888; font-size: 10px; }")
        layout.addWidget(hint)

        place_btns = QHBoxLayout()
        place_btns.setContentsMargins(0, 0, 0, 0)
        place_btns.addStretch()
        self._clear_place_btn = QPushButton(t("imeta_place_clear"))
        self._clear_place_btn.clicked.connect(self._clear_place)
        place_btns.addWidget(self._clear_place_btn)
        layout.addLayout(place_btns)

        # ── Dates ─────────────────────────────────────────────────────────
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._photo_date = QLineEdit()
        self._photo_date.setPlaceholderText(t("ibp_date_placeholder"))
        form.addRow(t("ibp_date_hdr"), self._photo_date)

        self._estimated_date = QLineEdit()
        self._estimated_date.setPlaceholderText(t("ibp_estimated_date_placeholder"))
        self._estimated_date.setToolTip(t("ibp_estimated_date_tooltip"))
        form.addRow(t("ibp_estimated_date_hdr"), self._estimated_date)
        layout.addLayout(form)

        # ── Note ──────────────────────────────────────────────────────────
        layout.addWidget(QLabel(t("ibp_note_hdr")))
        self._note = QTextEdit()
        self._note.setPlaceholderText(t("ibp_note_placeholder"))
        self._note.setFixedHeight(70)
        layout.addWidget(self._note)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load()

    # ── Loading ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            with session_scope() as session:
                img = session.get(Image, self._image_id)
                if img is None:
                    return
                self._file_label.setText(Path(img.file_path).name)
                self._photo_date.setText(img.photo_date or "")
                self._estimated_date.setText(img.estimated_date or "")
                self._note.setPlainText(img.note or "")
                self._place_id = img.place_id
                places = session.query(Place).order_by(Place.name).all()
                self._place_search.set_places(places)
                place_name = ""
                if img.place_id is not None:
                    place = session.get(Place, img.place_id)
                    place_name = place.name if place else ""
        except Exception:
            log.exception("Failed to load metadata of image %d", self._image_id)
            return

        if self._place_id is not None:
            self._place_search.set_current_by_id(self._place_id)
        self._update_place_label(place_name if self._place_id is not None else None)

    def _update_place_label(self, name: Optional[str]) -> None:
        if name:
            self._place_label.setText(t("imeta_place_current", name=name))
        else:
            self._place_label.setText(t("imeta_place_none"))

    # ── Place editing ─────────────────────────────────────────────────────

    def _on_place_selected(self, place_id: int) -> None:
        self._place_id = place_id
        with session_scope() as session:
            place = session.get(Place, place_id)
            name = place.name if place else ""
        self._update_place_label(name)

    def _on_place_create_requested(self, name: str) -> None:
        """Enter pressed on a name that matches no existing place — create it."""
        name = name.strip()
        if not name:
            return
        try:
            with session_scope() as session:
                place = PlaceService(session).get_or_create_by_name(name)
                place_id = place.id
                place_name = place.name
                places = session.query(Place).order_by(Place.name).all()
                self._place_search.set_places(places)
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to create place %r", name)
            QMessageBox.warning(self, t("imeta_title"), t("imeta_save_error", error=str(exc)))
            return
        self._place_id = place_id
        self._place_search.set_current_by_id(place_id)
        self._update_place_label(place_name)

    def _clear_place(self) -> None:
        self._place_id = None
        self._place_search.clear_selection()
        self._place_search.clear_query()
        self._update_place_label(None)

    # ── Saving ────────────────────────────────────────────────────────────

    def accept(self) -> None:
        photo_date = self._photo_date.text().strip() or None
        estimated_date = self._estimated_date.text().strip() or None
        note = self._note.toPlainText().strip() or None

        # A place highlighted in the list wins; otherwise a name left in the
        # search box is created on the fly; otherwise the current link stays.
        selected_id = self._place_search.current_place_id()
        typed_name = self._place_search.current_query()
        if selected_id is not None:
            place_id: Optional[int] = selected_id
            pending_name: Optional[str] = None
        elif typed_name:
            place_id, pending_name = None, typed_name
        else:
            place_id, pending_name = self._place_id, None

        try:
            with session_scope() as session:
                img = session.get(Image, self._image_id)
                if img is None:
                    super().accept()
                    return
                svc = PlaceService(session)
                if place_id is None and pending_name:
                    place_id = svc.get_or_create_by_name(pending_name).id
                svc.assign_place_to_image(self._image_id, place_id)
                img.photo_date = photo_date
                img.estimated_date = estimated_date
                img.note = note
                file_path = img.file_path
                exif_lat = img.exif_latitude
                image_lat = img.image_latitude
                place = session.get(Place, place_id) if place_id is not None else None
                place_lat = place.latitude if place else None
                place_lon = place.longitude if place else None
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to save metadata of image %d", self._image_id)
            QMessageBox.critical(self, t("imeta_title"), t("imeta_save_error", error=str(exc)))
            return

        # Same rule as the image browser: the place's GPS is written into the
        # file only when the image has no coordinates of its own.
        if (
            exif_lat is None
            and image_lat is None
            and place_lat is not None
            and place_lon is not None
        ):
            try:
                from app.utils.exif import write_exif_gps
                write_exif_gps(file_path, place_lat, place_lon)
            except Exception:  # noqa: BLE001
                log.warning("Could not write place GPS into %r", file_path, exc_info=True)

        log.info(
            "Image metadata saved: image=%d place=%s date=%r",
            self._image_id, place_id, photo_date,
        )
        super().accept()
