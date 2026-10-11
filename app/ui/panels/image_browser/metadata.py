"""Photo date, note, place and GPS editing."""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtWidgets import (
    QMessageBox,
)

from app.db.database import session_scope
from app.db.models import Image, Place
from app.services.place_service import ANONYMOUS_GPS_PLACE_NAME, PlaceService
from app.ui.i18n import t

from .helpers import (
    _read_exif_date,
)

log = logging.getLogger(__name__)


class MetadataMixin:
    """Photo date, note, place and GPS editing."""

    # ──────────────────────────────────────────────────────────────────
    # Photo date
    # ──────────────────────────────────────────────────────────────────

    def _save_photo_date(self) -> None:
        # Save to the original image when viewing a deoldified pair
        target_id = self._deol_pair_orig_id or self._current_image_id
        if target_id is None:
            return
        value = self._photo_date_edit.text().strip() or None
        with session_scope() as session:
            img = session.get(Image, target_id)
            if img is None:
                return
            img.photo_date = value
        log.debug("photo_date saved for image %d: %r", target_id, value)

    def _save_estimated_date(self) -> None:
        target_id = self._deol_pair_orig_id or self._current_image_id
        if target_id is None:
            return
        value = self._estimated_date_edit.text().strip() or None
        with session_scope() as session:
            img = session.get(Image, target_id)
            if img is None:
                return
            img.estimated_date = value
        log.debug("estimated_date saved for image %d: %r", target_id, value)

    def _save_note(self) -> None:
        target_id = self._deol_pair_orig_id or self._current_image_id
        if target_id is None:
            return
        value = self._note_edit.toPlainText().strip() or None
        with session_scope() as session:
            img = session.get(Image, target_id)
            if img is None:
                return
            img.note = value
        log.debug("note saved for image %d: %r", target_id, value)

    def _load_exif_suggestion(self, path: str) -> None:
        """Read EXIF DateTimeOriginal and show as a suggestion (not auto-fill)."""
        suggestion = _read_exif_date(path)
        if suggestion:
            self._exif_suggestion_label.setText(suggestion)
            self._exif_row_widget.setVisible(True)
        else:
            self._exif_suggestion_label.setText("")
            self._exif_row_widget.setVisible(False)

    def _accept_exif_suggestion(self) -> None:
        suggested = self._exif_suggestion_label.text()
        if not suggested:
            return
        self._photo_date_edit.blockSignals(True)
        self._photo_date_edit.setText(suggested)
        self._photo_date_edit.blockSignals(False)
        self._save_photo_date()
        self._exif_row_widget.setVisible(False)

    # ──────────────────────────────────────────────────────────────────
    # Place assignment
    # ──────────────────────────────────────────────────────────────────

    def _reload_places(self) -> None:
        with session_scope() as session:
            self._all_places = session.query(Place).order_by(Place.name).all()
        self._place_search.set_places(self._all_places)

    def _update_place_panel(
        self,
        place_id: Optional[int],
        place_name: str,
        is_anonymous: bool,
        source: Optional[str],
        place_lat: Optional[float] = None,
        place_lon: Optional[float] = None,
    ) -> None:
        self._place_search.set_current_by_id(place_id)
        self._place_rename_edit.setEnabled(place_id is not None)
        self._place_rename_btn.setEnabled(place_id is not None)
        if place_id is None:
            self._place_status_label.setText(t("ibp_place_none"))
            self._place_rename_edit.clear()
            self._place_coords_label.setText(t("ibp_place_coords_none"))
            return
        label = place_name
        if is_anonymous and source == "exif":
            label = f"{ANONYMOUS_GPS_PLACE_NAME} #{place_id}"
        self._place_status_label.setText(label)
        self._place_rename_edit.setText("" if is_anonymous else place_name)
        if place_lat is not None and place_lon is not None:
            self._place_coords_label.setText(
                t("ibp_place_coords_label", lat=place_lat, lon=place_lon)
            )
        else:
            self._place_coords_label.setText(t("ibp_place_coords_none"))

    def _on_place_selected(self, _place_id: int) -> None:
        pass

    def _assign_existing_place(self) -> None:
        if self._current_image_id is None:
            return
        place_id = self._place_search.current_place_id()
        if place_id is None:
            return
        with session_scope() as session:
            PlaceService(session).assign_place_to_image(self._current_image_id, place_id)
            place = session.get(Place, place_id)
            name = place.name if place else ""
            anon = bool(place and place.is_anonymous)
            source = place.source if place else None
            place_lat = place.latitude if place else None
            place_lon = place.longitude if place else None
            img = session.get(Image, self._current_image_id)
            img_exif_lat = img.exif_latitude if img else None
            img_image_lat = img.image_latitude if img else None
        log.info("Place assigned to image %d: %s", self._current_image_id, place_id)
        self._reload_places()
        self._update_place_panel(place_id, name, anon, source, place_lat, place_lon)
        # Requirement #4: write place GPS to EXIF if no EXIF GPS and no image-level coords
        if (
            img_exif_lat is None
            and img_image_lat is None
            and place_lat is not None
            and place_lon is not None
            and self._current_path
        ):
            from app.utils.exif import write_exif_gps
            write_exif_gps(self._current_path, place_lat, place_lon)
        self._reload_gps_panel()

    def _create_or_assign_place_by_text(self) -> None:
        # Read the name from the search box, falling back to the rename field so
        # a name typed in either input works.  Empty input used to fail silently
        # (the button "did nothing"); now we tell the user instead.
        name = self._place_search.current_query()
        if not name:
            name = self._place_rename_edit.text().strip()
        if not name:
            QMessageBox.information(
                self, t("ibp_place_create_btn"), t("ibp_place_need_name")
            )
            return
        with session_scope() as session:
            svc = PlaceService(session)
            if self._current_image_id is not None:
                # Image open → create (if new) and assign to it.
                place = svc.assign_place_to_image_by_name(self._current_image_id, name)
            else:
                # No image selected → still create the place so it exists and
                # shows up in the list, rather than silently doing nothing.
                place = svc.get_or_create_by_name(name)
            place_id = place.id
            place_name = place.name
            anon = place.is_anonymous
            source = place.source
            place_lat = place.latitude
            place_lon = place.longitude
            img = (
                session.get(Image, self._current_image_id)
                if self._current_image_id is not None
                else None
            )
            img_exif_lat = img.exif_latitude if img else None
            img_image_lat = img.image_latitude if img else None
        log.info(
            "Place created/assigned: image=%s name=%r", self._current_image_id, name
        )
        self._reload_places()
        self._update_place_panel(place_id, place_name, anon, source, place_lat, place_lon)
        # Requirement #4: write place GPS to EXIF if no EXIF GPS and no image-level coords
        if (
            img is not None
            and img_exif_lat is None
            and img_image_lat is None
            and place_lat is not None
            and place_lon is not None
            and self._current_path
        ):
            from app.utils.exif import write_exif_gps
            write_exif_gps(self._current_path, place_lat, place_lon)
        self._reload_gps_panel()

    def _rename_current_place(self) -> None:
        if self._current_image_id is None:
            return
        new_name = self._place_rename_edit.text().strip()
        if not new_name:
            return
        try:
            with session_scope() as session:
                img = session.get(Image, self._current_image_id)
                if img is None or img.place_id is None:
                    return
                place = PlaceService(session).name_place(img.place_id, new_name)
                place_id = place.id
                place_name = place.name
                anon = place.is_anonymous
                source = place.source
                place_lat = place.latitude
                place_lon = place.longitude
        except Exception as exc:  # noqa: BLE001
            from PySide6.QtWidgets import QMessageBox

            from app.ui.i18n import t
            QMessageBox.critical(
                self,
                t("ibp_place_rename_btn"),
                t("places_merge_error", error=str(exc)),
            )
            return
        log.info("Place renamed: %d -> %r", place_id, new_name)
        self._reload_places()
        self._update_place_panel(place_id, place_name, anon, source, place_lat, place_lon)

    # ── Image GPS coordinates ─────────────────────────────────────────

    def _load_image_coords(
        self,
        image_lat: Optional[float],
        image_lon: Optional[float],
        exif_lat: Optional[float],
        exif_lon: Optional[float],
        place_lat: Optional[float],
        place_lon: Optional[float],
        place_id: Optional[int],
    ) -> None:
        """Populate the GPS coordinate widgets according to priority order."""
        self._gps_edit.blockSignals(True)
        if image_lat is not None and image_lon is not None:
            self._gps_edit.setText(f"{image_lat:.6f}, {image_lon:.6f}")
            self._gps_source_label.setText(t("ibp_gps_source_manual"))
            self._gps_source_label.setStyleSheet("color: #88ee88; font-size: 10px; font-style: italic;")
        elif exif_lat is not None and exif_lon is not None:
            self._gps_edit.setText(f"{exif_lat:.6f}, {exif_lon:.6f}")
            self._gps_source_label.setText(t("ibp_gps_source_exif"))
            self._gps_source_label.setStyleSheet("color: #aaddaa; font-size: 10px; font-style: italic;")
        elif place_lat is not None and place_lon is not None:
            self._gps_edit.setText(f"{place_lat:.6f}, {place_lon:.6f}")
            self._gps_source_label.setText(t("ibp_gps_source_place"))
            self._gps_source_label.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        else:
            self._gps_edit.clear()
            self._gps_source_label.setText(t("ibp_gps_source_none"))
            self._gps_source_label.setStyleSheet("color: #555; font-size: 10px; font-style: italic;")
        self._gps_edit.blockSignals(False)

    def _reload_gps_panel(self) -> None:
        """Reload GPS panel from DB for the current image."""
        if self._current_image_id is None:
            return
        with session_scope() as session:
            img = session.get(Image, self._current_image_id)
            if img is None:
                return
            self._load_image_coords(
                img.image_latitude,
                img.image_longitude,
                img.exif_latitude,
                img.exif_longitude,
                img.place.latitude if img.place else None,
                img.place.longitude if img.place else None,
                img.place_id,
            )

    def _save_image_coords(self) -> None:
        """Validate and save the image-level GPS coordinate to DB and optionally EXIF."""
        if self._current_image_id is None:
            return
        from app.utils.exif import validate_coords, write_exif_gps

        text = self._gps_edit.text()
        parsed = validate_coords(text)
        if parsed is None:
            if text.strip():
                log.warning(
                    "Image %d: invalid GPS coordinate input %r — not saved",
                    self._current_image_id, text,
                )
                self._gps_source_label.setText(t("ibp_gps_invalid"))
                self._gps_source_label.setStyleSheet(
                    "color: #ee8888; font-size: 10px; font-style: italic;"
                )
            else:
                # Empty field → clear stored coords
                self._clear_image_coords()
            return

        lat, lon = parsed
        needs_exif_write = False
        with session_scope() as session:
            img = session.get(Image, self._current_image_id)
            if img is None:
                return
            img.image_latitude = lat
            img.image_longitude = lon
            needs_exif_write = img.exif_latitude is None

        log.info("Image %d: image_latitude=%.6f image_longitude=%.6f saved", self._current_image_id, lat, lon)
        self._gps_source_label.setText(t("ibp_gps_source_manual"))
        self._gps_source_label.setStyleSheet("color: #88ee88; font-size: 10px; font-style: italic;")

        # Requirement #3: write to EXIF if no EXIF GPS exists
        if needs_exif_write and self._current_path:
            write_exif_gps(self._current_path, lat, lon)

    def _clear_image_coords(self) -> None:
        """Remove the image-level GPS coordinate."""
        if self._current_image_id is None:
            return
        with session_scope() as session:
            img = session.get(Image, self._current_image_id)
            if img is None:
                return
            img.image_latitude = None
            img.image_longitude = None
        log.info("Image %d: image_latitude/longitude cleared", self._current_image_id)
        self._reload_gps_panel()

    def _update_exif_date(self) -> None:
        """Write the photo date field into the image EXIF DateTimeOriginal."""
        if not self._current_path:
            return
        from app.utils.exif import parse_flexible_date, write_exif_date

        date_str = self._photo_date_edit.text().strip()
        if not date_str:
            # Also try the DB value in case the field is stale
            if self._current_image_id is not None:
                with session_scope() as session:
                    img = session.get(Image, self._current_image_id)
                    date_str = (img.photo_date or "") if img else ""

        dt = parse_flexible_date(date_str)
        if dt is None:
            log.info(
                "EXIF date update skipped — unparseable date for image %r: %r",
                self._current_path, date_str,
            )
            return

        write_exif_date(self._current_path, dt)
