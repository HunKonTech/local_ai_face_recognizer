"""Thumbnail/Drive fetch, prefetch and image loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import (
    QThreadPool,
    Slot,
)
from PySide6.QtWidgets import (
    QMessageBox,
    QTreeWidgetItem,
)

from app.db.database import session_scope
from app.db.models import Image
from app.ui.i18n import t

from .helpers import (
    _DRIVE_PREFETCH_AHEAD,
    _ROLE_FILE_PATH,
    _ROLE_PAYLOAD,
    _ROLE_TYPE,
    _TREE_THUMB_SIZE,
    _parse_date_from_filename,
)

log = logging.getLogger(__name__)


class ImageLoadingMixin:
    """Thumbnail/Drive fetch, prefetch and image loading."""

    @Slot(str)
    def _on_thumbnail_failed(self, cache_key: str) -> None:
        pass  # placeholder icon stays

    # ──────────────────────────────────────────────────────────────────
    # Drive on-demand download callbacks
    # ──────────────────────────────────────────────────────────────────

    @Slot(int, str)
    def _on_drive_fetch_ready(self, image_id: int, local_path: str) -> None:
        """Called when a DriveFetchRunnable completes.

        If this fetch was for the currently requested image, display it.
        Also update the thumbnail in the tree.
        """
        # Remove from prefetch-in-progress set.
        self._prefetch_set.discard(image_id)

        # Refresh the thumbnail in the tree if we have a cache key.
        cache_key = f"thumb_{image_id}"
        if cache_key not in self._thumb_cache:
            tree_item = self._tree_items.get(cache_key)
            if tree_item is not None:
                from app.workers.thumbnail_worker import ThumbnailRunnable
                # File is now in mirror — generate thumbnail normally.
                worker = ThumbnailRunnable(
                    file_path=local_path,
                    cache_key=cache_key,
                    size=_TREE_THUMB_SIZE,
                )
                worker.signals.ready.connect(self._on_thumbnail_ready)
                worker.signals.failed.connect(self._on_thumbnail_failed)
                QThreadPool.globalInstance().start(worker)

        # If this is the image the user is waiting to see, load it now.
        if image_id == self._drive_pending_image_id:
            self._drive_pending_image_id = None
            self._load_image(image_id)

    @Slot(int, str)
    def _on_drive_fetch_failed(self, image_id: int, error: str) -> None:
        self._prefetch_set.discard(image_id)
        if image_id == self._drive_pending_image_id:
            self._drive_pending_image_id = None
            self._image_label.set_source_pixmap(None)
            self._image_label.setText(
                t("ibp_drive_download_failed", error=error)
            )

    # ──────────────────────────────────────────────────────────────────
    # Drive smart prefetch
    # ──────────────────────────────────────────────────────────────────

    def _prefetch_adjacent(self, current_item: QTreeWidgetItem) -> None:
        """Queue Drive downloads for the next *_DRIVE_PREFETCH_AHEAD* siblings.

        Skips images whose mirror file already exists and ones already
        in-flight.
        """
        if self._drive_client is None:
            return
        parent = current_item.parent()
        if parent is None:
            return

        idx = parent.indexOfChild(current_item)
        pool = QThreadPool.globalInstance()

        for delta in range(1, _DRIVE_PREFETCH_AHEAD + 1):
            si = idx + delta
            if si >= parent.childCount():
                break
            sibling = parent.child(si)
            if sibling.data(0, _ROLE_TYPE) != "image":
                continue
            img_id: int = sibling.data(0, _ROLE_PAYLOAD)
            file_path: str = sibling.data(0, _ROLE_FILE_PATH) or ""
            if not file_path:
                continue
            if Path(file_path).exists():
                continue  # already in mirror
            if img_id in self._prefetch_set:
                continue  # already downloading

            self._prefetch_set.add(img_id)
            from app.workers.drive_image_worker import DriveFetchRunnable
            fetcher = DriveFetchRunnable(
                drive_client=self._drive_client,
                image_id=img_id,
                local_path=file_path,
                silent=True,
            )
            fetcher.signals.ready.connect(self._on_drive_fetch_ready)
            pool.start(fetcher)
            log.debug("Drive prefetch queued: image_id=%d", img_id)

    # ──────────────────────────────────────────────────────────────────
    # Image loading
    # ──────────────────────────────────────────────────────────────────

    def _load_image(self, image_id: int) -> None:
        log.info("Image selected: image_id=%d", image_id)

        # CRITICAL: clear face selection state before loading new image
        self._selected_face_id = None
        self._editing_face_id = None
        self._hide_inline_editor()
        self._cancel_rename()
        # Exit interactive edit mode without restoring (different image)
        if self._interactive_edit_face_id is not None:
            self._interactive_edit_face_id = None
            self._interactive_edit_orig_bbox = None
            self._image_label.cancel_interactive_edit()
            self._draw_hint.setVisible(False)

        # Reset deoldified pairing state
        self._deol_pair_orig_id = None
        self._deol_pair_partner_id = None
        self._deol_pair_color_path = ""
        self._deol_bw_path = ""
        self._deol_pairing_enabled = False
        self._deol_viewing_color = False
        self._deol_clear_for_new_image()
        self._deoldified_bar.setVisible(False)

        with session_scope() as session:
            img = session.get(Image, image_id)
            if img is None:
                log.warning("Image %d not found in DB", image_id)
                return
            for f in img.faces:
                _ = f.person
            self._current_image_id = image_id
            # Existence-checked: img.file_path is the absolute path recorded on
            # the indexing machine, so on a moved or re-rooted library it can
            # point nowhere.  Everything downstream (pixels, EXIF writes) opens
            # this path, so it must be the one that is actually on disk.
            from app.services.image_library_service import (
                resolve_existing_image_path,
            )
            _resolved = resolve_existing_image_path(img)
            self._current_path = str(_resolved) if _resolved else img.file_path
            self._save_meta_btn.setEnabled(True)
            self.image_displayed.emit(image_id, img.file_path)
            self._detection_done = img.detection_done
            photo_date = img.photo_date or ""
            estimated_date = img.estimated_date or ""
            note = img.note or ""
            current_place_id = img.place_id
            current_place_name = img.place.name if img.place else ""
            current_place_is_anonymous = bool(img.place and img.place.is_anonymous)
            current_place_source = img.place.source if img.place else None
            current_place_lat = img.place.latitude if img.place else None
            current_place_lon = img.place.longitude if img.place else None
            current_image_lat = img.image_latitude
            current_image_lon = img.image_longitude
            current_exif_lat = img.exif_latitude
            current_exif_lon = img.exif_longitude
            # Auto-fill date from filename when not yet set
            if not photo_date:
                from_name = _parse_date_from_filename(img.file_path)
                if from_name:
                    img.photo_date = from_name
                    photo_date = from_name
                    log.debug(
                        "photo_date auto-set from filename for image %d: %r",
                        image_id, from_name,
                    )
            # Track current folder
            self._current_folder = str(Path(img.file_path).parent)
            self._face_data = [
                (
                    f.id,
                    f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                    f.person.name if f.person else None,
                    f.person_id,
                    bool(f.is_uncertain_identification),
                )
                for f in img.faces
                if not f.is_excluded
            ]

        # Detect deoldified pair and override state if pairing is enabled
        self._setup_deoldified_pair(image_id, self._current_path)

        # When current image is deoldified: load face data from the original
        if self._deol_pair_orig_id is not None:
            with session_scope() as session:
                orig = session.get(Image, self._deol_pair_orig_id)
                if orig is not None:
                    for f in orig.faces:
                        _ = f.person
                    self._face_data = [
                        (
                            f.id,
                            f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                            f.person.name if f.person else None,
                            f.person_id,
                            bool(f.is_uncertain_identification),
                        )
                        for f in orig.faces
                        if not f.is_excluded
                    ]
                    if not photo_date:
                        photo_date = orig.photo_date or ""
                    if not estimated_date:
                        estimated_date = orig.estimated_date or ""

        # Re-read through the service so duplicate/missing crop paths are
        # repaired before any preview state is rendered.
        self._fetch_face_data()

        self._photo_date_edit.blockSignals(True)
        self._photo_date_edit.setText(photo_date)
        self._photo_date_edit.blockSignals(False)
        self._estimated_date_edit.blockSignals(True)
        self._estimated_date_edit.setText(estimated_date)
        self._estimated_date_edit.blockSignals(False)
        self._note_edit.blockSignals(True)
        self._note_edit.setPlainText(note)
        self._note_edit.blockSignals(False)
        self._update_place_panel(
            current_place_id,
            current_place_name,
            current_place_is_anonymous,
            current_place_source,
            current_place_lat,
            current_place_lon,
        )
        self._load_image_coords(
            current_image_lat,
            current_image_lon,
            current_exif_lat,
            current_exif_lon,
            current_place_lat,
            current_place_lon,
            current_place_id,
        )

        p = Path(self._current_path)
        self._folder_label.setText(str(p.parent))
        self._filename_label.setText(p.name)
        self._load_exif_suggestion(self._current_path)

        from app.utils.image_utils import load_image_bgr_normalized
        display_path = (
            self._deol_pair_color_path
            if (self._deol_viewing_color and self._deol_pair_color_path)
            else self._current_path
        )

        # Drive mode: if mirror file is missing, download it first.
        if not Path(display_path).exists() and self._drive_client is not None:
            self._drive_pending_image_id = image_id
            self._image_label.set_source_pixmap(None)
            self._image_label.setText(t("ibp_drive_downloading"))
            from app.workers.drive_image_worker import DriveFetchRunnable
            fetcher = DriveFetchRunnable(
                drive_client=self._drive_client,
                image_id=image_id,
                local_path=display_path,
                silent=False,
            )
            fetcher.signals.ready.connect(self._on_drive_fetch_ready)
            fetcher.signals.failed.connect(self._on_drive_fetch_failed)
            QThreadPool.globalInstance().start(fetcher)
            return

        img_bgr = load_image_bgr_normalized(display_path)
        if img_bgr is None:
            self._orig_img_bgr = None
            self._full_pixmap = None
            self._image_label.set_source_pixmap(None)
            self._image_label.setText(t("ibp_load_error", path=display_path))
            self._clear_face_panel()
            log.warning("Cannot load image file: %s", display_path)
            return

        self._orig_img_bgr = img_bgr
        self._reset_zoom()
        self._redraw_faces()
        self._deol_apply_remembered_mode()
        self._clear_face_panel()
        self._reload_persons_combo()
        self._reload_places()
        self._refresh_object_markers()

    def _on_save_face_metadata(self) -> None:
        """Write everything about the current image into its metadata.

        Persists, into the image file (with sidecar-JSON fallback for persons):

        * recognised persons + face bounding boxes (``facelocal`` XMP/EXIF),
        * the effective GPS coordinate (image-level → EXIF → linked place), and
        * the photo date (EXIF DateTimeOriginal).

        When the displayed image is part of a deoldified (colour ↔ B&W) pair,
        both variants receive the same data so the annotation survives whichever
        copy is later shared or re-imported.
        """
        if self._current_image_id is None:
            return

        # Current image + its deoldified partner (either direction), de-duplicated.
        image_ids: list[int] = [self._current_image_id]
        if (
            self._deol_pair_partner_id is not None
            and self._deol_pair_partner_id != self._current_image_id
        ):
            image_ids.append(self._deol_pair_partner_id)

        reply = QMessageBox.warning(
            self,
            t("fmeta_confirm_title"),
            t("fmeta_warning"),
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply != QMessageBox.Ok:
            return

        from app.services.face_metadata_export_service import FaceMetadataExportService
        from app.services.image_library_service import resolve_existing_image_path
        from app.utils.exif import parse_flexible_date, write_exif_date, write_exif_gps

        gps_written = 0
        date_written = 0

        with session_scope() as session:
            # 1) GPS + date go in FIRST — they only touch EXIF. The face-metadata
            #    XMP write below re-attaches the existing EXIF, so writing it last
            #    keeps GPS/date intact while adding the persons block.
            for image_id in image_ids:
                img = session.get(Image, image_id)
                if img is None:
                    continue
                abs_path = resolve_existing_image_path(img)
                if abs_path is None or not Path(abs_path).exists():
                    continue
                path_str = str(abs_path)

                lat, lon = self._effective_gps(img)
                if lat is not None and lon is not None and write_exif_gps(path_str, lat, lon):
                    gps_written += 1

                dt = parse_flexible_date((img.photo_date or "").strip())
                if dt is not None and write_exif_date(path_str, dt):
                    date_written += 1

            # 2) Persons + face boxes (XMP / EXIF UserComment / sidecar JSON).
            summary = FaceMetadataExportService(session).export_images(image_ids)

        msg = t(
            "ib3_save_meta_summary",
            total=summary.total,
            embedded=summary.embedded_count,
            sidecar=summary.sidecar_count,
            gps=gps_written,
            date=date_written,
            failed=summary.failed_count,
        )
        errors = summary.errors
        if errors:
            msg += "\n\n" + "\n".join(errors[:10])
        QMessageBox.information(self, t("fmeta_done_title"), msg)

    @staticmethod
    def _effective_gps(img: "Image") -> tuple[Optional[float], Optional[float]]:
        """Resolve the GPS coordinate to embed: image-level → EXIF → place."""
        if img.image_latitude is not None and img.image_longitude is not None:
            return img.image_latitude, img.image_longitude
        if img.exif_latitude is not None and img.exif_longitude is not None:
            return img.exif_latitude, img.exif_longitude
        if img.place is not None and img.place.latitude is not None:
            return img.place.latitude, img.place.longitude
        return None, None

    def _clear_current_image(self) -> None:
        """Reset preview when switching folders."""
        self._current_image_id = None
        self._current_path = ""
        if hasattr(self, "_save_meta_btn"):
            self._save_meta_btn.setEnabled(False)
        self._face_data = []
        self._selected_face_id = None
        self._editing_face_id = None
        self._interactive_edit_face_id = None
        self._interactive_edit_orig_bbox = None
        self._orig_img_bgr = None
        self._full_pixmap = None
        self._deol_pair_orig_id = None
        self._deol_pair_color_path = ""
        self._deol_bw_path = ""
        self._deol_pairing_enabled = False
        self._deol_viewing_color = False
        self._deol_clear_for_new_image()
        self._deoldified_bar.setVisible(False)
        self._image_label.set_source_pixmap(None)
        self._image_label.setText(t("ib3_select_image_hint"))
        self._folder_label.setText("")
        self._filename_label.setText("")
        self._note_edit.blockSignals(True)
        self._note_edit.clear()
        self._note_edit.blockSignals(False)
        self._exif_row_widget.setVisible(False)
        self._update_place_panel(None, "", False, None)
        self._load_image_coords(None, None, None, None, None, None, None)
        self._clear_face_panel()
        self._hide_inline_editor()
        self._draw_mode_btn.setChecked(False)
        self._reset_zoom()
