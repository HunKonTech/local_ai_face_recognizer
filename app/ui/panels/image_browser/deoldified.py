"""Colorized / black-and-white pair handling and comparison view."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QMessageBox,
)

from app.db.database import session_scope
from app.db.models import Image
from app.ui.i18n import t

log = logging.getLogger(__name__)


class DeoldifiedMixin:
    """Colorized / black-and-white pair handling and comparison view."""

    # ──────────────────────────────────────────────────────────────────
    # Deoldified pairing
    # ──────────────────────────────────────────────────────────────────

    def _setup_deoldified_pair(self, image_id: int, image_path: str) -> None:
        """Detect the comparison group and configure the toggle bar.

        A group is the B&W original plus every colorized variant that shares its
        filename (folder-independent).  Two members behave exactly like the old
        single pair; three or more enable the left/right pickers in compare mode.

        The bar itself is shown whenever a group exists, so a colorized pair is
        never invisible.  The 'deoldified/auto_pair' setting only decides
        whether the two sides *share data* — annotations written here landing on
        the B&W original, and the sync button that copies missing fields over.
        """
        from app.app_settings import app_qsettings
        settings = app_qsettings()
        pairing_enabled = settings.value("deoldified/auto_pair", False, type=bool)

        from app.services.deoldified_pairing_service import (
            DeoldifiedPairingService,
            is_deoldified_path,
        )

        with session_scope() as session:
            img = session.get(Image, image_id)
            if img is None:
                return
            group = DeoldifiedPairingService(session).get_comparison_group(img)
        if len(group) < 2:
            return

        self._deol_group = group
        self._deol_pairing_enabled = pairing_enabled
        bw_member = group[0]  # is_bw=True by construction
        self._deol_bw_path = bw_member.file_path
        current_is_color = is_deoldified_path(image_path)
        self._deol_opened_is_color = current_is_color

        # Default selection: left = B&W original, right = colorized. When the
        # tree image is itself a colorized variant, prefer it on the right so the
        # opened image stays visible.
        self._deol_left_idx = 0
        self._deol_right_idx = 1
        if current_is_color:
            for i, m in enumerate(group):
                if m.image_id == image_id and not m.is_bw:
                    self._deol_right_idx = i
                    break
            # Only redirect annotation writes onto the original when the two
            # sides are actually meant to share data.
            self._deol_pair_orig_id = bw_member.image_id if pairing_enabled else None
            self._deol_viewing_color = True
            self._btn_view_bw.setChecked(False)
            self._btn_view_color.setChecked(True)
        else:
            self._deol_pair_orig_id = None  # current IS the B&W original
            self._deol_viewing_color = False
            self._btn_view_bw.setChecked(True)
            self._btn_view_color.setChecked(False)

        self._deol_pair_color_path = group[self._deol_right_idx].file_path
        self._update_deol_sync_partner(image_id)
        self._populate_deol_combos()
        self._btn_deol_sync.setVisible(pairing_enabled)
        self._deoldified_bar.setVisible(True)
        log.debug(
            "Comparison group of %d image(s): %s (data sharing: %s)",
            len(group), [m.label or "bw" for m in group], pairing_enabled,
        )

        # Auto-sync annotations from the filled side into the empty side.
        if pairing_enabled and settings.value("deoldified/auto_sync", True, type=bool):
            self._run_deoldified_sync(image_id, announce=False)

    def _update_deol_sync_partner(self, image_id: int) -> None:
        """Point the sync partner at the other side of the active comparison.

        When the tree image is the B&W original, the partner is the selected
        colorized variant; when it is colorized, the partner is the original.
        """
        if not self._deol_group or not self._deol_pairing_enabled:
            self._deol_pair_partner_id = None
            return
        if self._deol_pair_orig_id is not None:
            self._deol_pair_partner_id = self._deol_pair_orig_id
        else:
            self._deol_pair_partner_id = self._deol_group[self._deol_right_idx].image_id

    def _deol_member_text(self, member) -> str:
        """Combo label for one comparison member."""
        if member.is_bw:
            return t("ibp_deol_bw_label")
        if member.label and member.label != "deoldified":
            return f"{t('ibp_view_colorized')} {member.label}"
        return t("ibp_view_colorized")

    def _populate_deol_combos(self) -> None:
        """Fill the left/right pickers from the current group (signals blocked)."""
        for combo, idx in (
            (self._deol_left_combo, self._deol_left_idx),
            (self._deol_right_combo, self._deol_right_idx),
        ):
            combo.blockSignals(True)
            combo.clear()
            for m in self._deol_group:
                combo.addItem(self._deol_member_text(m))
            if 0 <= idx < combo.count():
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        self._deol_update_combo_visibility()

    def _deol_update_combo_visibility(self) -> None:
        """Show the left/right pickers only in compare mode with 3+ members."""
        show = self._deol_compare and len(self._deol_group) >= 3
        for w in (
            self._deol_left_lbl, self._deol_left_combo,
            self._deol_right_lbl, self._deol_right_combo,
        ):
            w.setVisible(show)

    def _on_deol_left_changed(self, idx: int) -> None:
        """Left picker changed → reload both cached sides and recompose."""
        if idx < 0 or idx >= len(self._deol_group) or idx == self._deol_left_idx:
            return
        self._deol_left_idx = idx
        # Right pixels are resized to the left shape, so invalidate both.
        self._deol_left_bgr = None
        self._deol_right_bgr = None
        self._deol_recompose_compare()

    def _on_deol_right_changed(self, idx: int) -> None:
        """Right picker changed → update single-view color path and recompose."""
        if idx < 0 or idx >= len(self._deol_group) or idx == self._deol_right_idx:
            return
        self._deol_right_idx = idx
        self._deol_right_bgr = None
        member = self._deol_group[idx]
        if not member.is_bw:
            self._deol_pair_color_path = member.file_path
        self._update_deol_sync_partner(self._current_image_id)
        self._deol_recompose_compare()

    def _deol_recompose_compare(self) -> None:
        """Re-render the compare composite after a picker change."""
        if not self._deol_compare:
            return
        if not self._deol_ensure_compare_bgr():
            return
        comp = self._deol_composite()
        if comp is not None:
            self._orig_img_bgr = comp
            self._redraw_faces()

    def _on_deol_view_toggle(self, show_colorized: bool) -> None:
        """User picked the single B&W or colorized view (remembered choice)."""
        self._deol_mode = "color" if show_colorized else "bw"
        self._apply_single_view(show_colorized, reset_zoom=True)

    def _apply_single_view(
        self, show_colorized: bool, *, reset_zoom: bool
    ) -> None:
        """Display one side of the pair, with the compare divider off."""
        from app.utils.image_utils import load_image_bgr_normalized as load_image_bgr

        # The group already carries resolved, on-disk paths, so switching sides
        # never touches the database.
        if show_colorized:
            path = self._deol_pair_color_path
        else:
            path = self._deol_bw_path or self._current_path

        img_bgr = load_image_bgr(path)
        if img_bgr is None:
            log.warning("Cannot load image for deoldified view: %s", path)
            if show_colorized:
                # Colorized variant file is gone — degrade to the B&W side
                # instead of leaving the panel frozen on the previous view.
                bw_path = self._deol_bw_path or self._current_path
                bw_bgr = load_image_bgr(bw_path) if bw_path else None
                if bw_bgr is not None:
                    self._deol_mode = "bw"
                    img_bgr = bw_bgr
                    show_colorized = False
                    self._deol_show_path_error(t("ibp_deol_variant_missing"))
                else:
                    self._deol_show_path_error(t("ibp_deol_side_missing"))
                    return
            else:
                # The B&W side used to fail silently, which looked exactly like
                # a dead button; say what happened and offer the repair tool.
                self._deol_show_path_error(t("ibp_deol_side_missing"))
                return
        else:
            self._deol_clear_path_error()

        self._deol_compare = False
        self._image_label.set_compare_mode(False)
        self._btn_view_compare.setChecked(False)
        self._deol_update_combo_visibility()
        self._deol_viewing_color = show_colorized
        self._btn_view_bw.setChecked(not show_colorized)
        self._btn_view_color.setChecked(show_colorized)
        self._orig_img_bgr = img_bgr
        if reset_zoom:
            self._reset_zoom()
        self._redraw_faces()

    def _deol_show_path_error(self, message: str) -> None:
        """Explain a failed load in the bar and offer the path-repair tool."""
        self._deol_lbl.setText(message)
        self._btn_deol_fix_paths.setVisible(True)

    def _deol_clear_path_error(self) -> None:
        """Restore the normal bar caption after a successful load."""
        # isHidden(), not isVisible(): the bar may be laid out but not yet
        # shown, and the caption still has to be restored.
        if not self._btn_deol_fix_paths.isHidden():
            self._btn_deol_fix_paths.setVisible(False)
            self._deol_lbl.setText(t("ibp_deol_pair_lbl"))

    def _deol_ensure_compare_bgr(self) -> bool:
        """Load and cache the two selected sides, right resized to left shape."""
        if self._deol_left_bgr is not None and self._deol_right_bgr is not None:
            return True
        if not self._deol_group:
            return False
        from app.utils.image_utils import load_image_bgr_normalized as load_image_bgr

        left_m = self._deol_group[self._deol_left_idx]
        right_m = self._deol_group[self._deol_right_idx]
        left = load_image_bgr(left_m.file_path)
        right = load_image_bgr(right_m.file_path)
        if left is None or right is None:
            log.warning(
                "Cannot load image for deoldified compare: %s",
                left_m.file_path if left is None else right_m.file_path,
            )
            self._deol_show_path_error(t("ibp_deol_side_missing"))
            return False
        self._deol_clear_path_error()
        if right.shape[:2] != left.shape[:2]:
            right = cv2.resize(
                right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA
            )
        self._deol_left_bgr = left
        self._deol_right_bgr = right
        return True

    def _deol_composite(self) -> Optional[np.ndarray]:
        """Compose left side and right side split at the current slider position."""
        left = self._deol_left_bgr
        right = self._deol_right_bgr
        if left is None or right is None:
            return None
        h, w = left.shape[:2]
        split = max(0, min(w, int(round(w * self._deol_split / 100.0))))
        out = right.copy()
        out[:, :split] = left[:, :split]
        return out

    def _on_deol_compare_toggle(self) -> None:
        """User picked compare mode (remembered choice)."""
        self._deol_mode = "compare"
        if not self._enter_compare(reset_zoom=True):
            # Pixels unavailable → fall back to the previous single view.
            self._deol_mode = "color" if self._deol_viewing_color else "bw"
            self._btn_view_compare.setChecked(False)

    def _enter_compare(self, *, reset_zoom: bool) -> bool:
        """Activate the on-image compare divider; False if pixels unavailable."""
        if not self._deol_ensure_compare_bgr():
            return False
        comp = self._deol_composite()
        if comp is None:
            return False
        self._deol_compare = True
        self._deol_viewing_color = False
        self._btn_view_bw.setChecked(False)
        self._btn_view_color.setChecked(False)
        self._btn_view_compare.setChecked(True)
        self._orig_img_bgr = comp
        if reset_zoom:
            self._reset_zoom()
        self._redraw_faces()
        self._image_label.set_compare_mode(True, self._deol_split)
        self._deol_update_combo_visibility()
        return True

    def _on_compare_dragged(self, percent: int) -> None:
        """Recompose the view as the on-image compare divider is dragged."""
        self._deol_split = percent
        if not self._deol_compare:
            return
        comp = self._deol_composite()
        if comp is not None:
            self._orig_img_bgr = comp
            self._redraw_faces()

    def _deol_apply_remembered_mode(self) -> None:
        """Re-apply the remembered view choice to a newly opened image.

        Compare mode is a property of the pair, so it survives navigation.  The
        B&W/colorized choice does not: the file picked in the tree decides which
        side is shown, otherwise clicking a '-deoldified' file would keep
        showing black and white.  ``_load_image`` has already loaded that side,
        so this only records the matching mode.
        """
        if not self._deol_group:
            return
        if self._deol_mode == "compare":
            if not self._enter_compare(reset_zoom=False):
                self._deol_mode = None
            return
        self._deol_mode = "color" if self._deol_opened_is_color else "bw"

    def _deol_clear_for_new_image(self) -> None:
        """Per-image reset: drop cached pixels and the compare divider.

        Keeps the remembered mode/split so the choice persists across images.
        """
        self._deol_compare = False
        self._deol_opened_is_color = False
        self._deol_left_bgr = None
        self._deol_right_bgr = None
        self._deol_group = []
        self._deol_left_idx = 0
        self._deol_right_idx = 1
        if hasattr(self, "_image_label"):
            self._image_label.set_compare_mode(False)
        if hasattr(self, "_btn_view_compare"):
            self._btn_view_compare.setChecked(False)
        if hasattr(self, "_deol_left_combo"):
            self._deol_update_combo_visibility()
        if hasattr(self, "_btn_deol_fix_paths"):
            self._btn_deol_fix_paths.setVisible(False)
            self._deol_lbl.setText(t("ibp_deol_pair_lbl"))

    def _run_deoldified_sync(self, image_id: int, *, announce: bool) -> Optional[dict]:
        """Copy annotations between the current image and its deoldified pair.

        Data flows one way, from the side that has annotations into the side
        that is empty; nothing happens when both sides are empty or both
        already hold data.  When ``announce`` is True a result popup is shown.
        Returns the sync summary dict, or None when nothing was copied.
        """
        if self._deol_pair_partner_id is None:
            if announce:
                QMessageBox.information(
                    self, t("ibp_deol_sync"), t("ibp_deol_sync_no_pair")
                )
            return None

        from app.services.deoldified_pairing_service import DeoldifiedPairingService

        crops_dir = self._config.crops_dir_resolved if self._config else None
        thumbnail_size = self._config.scan.thumbnail_size if self._config else None
        crop_mode = (
            self._config.embedding.crop_mode if self._config else "legacy"
        )

        with session_scope() as session:
            current = session.get(Image, image_id)
            partner = session.get(Image, self._deol_pair_partner_id)
            if current is None or partner is None:
                return None
            svc = DeoldifiedPairingService(session)
            result = svc.sync_pair_data(
                current,
                partner,
                crops_dir=crops_dir,
                thumbnail_size=thumbnail_size,
                crop_mode=crop_mode,
            )

        if result is not None:
            log.info(
                "Deoldified sync: %d new face(s), %d updated, %d object(s), "
                "metadata=%s (%d → %d)",
                result["faces_copied"],
                result["faces_updated"],
                result["objects_moved"],
                result["metadata_fields"],
                result["source_id"],
                result["target_id"],
            )
            self._reload_current_face_data()
            self._refresh_object_markers()

        if announce:
            if result is None:
                QMessageBox.information(
                    self, t("ibp_deol_sync"), t("ibp_deol_sync_skipped")
                )
            else:
                QMessageBox.information(
                    self,
                    t("ibp_deol_sync"),
                    t("ibp_deol_sync_done").format(
                        faces=result["faces_copied"],
                        updated=result["faces_updated"],
                        objects=result["objects_moved"],
                        fields=len(result["metadata_fields"]),
                    ),
                )
        return result

    def _on_deol_sync_clicked(self) -> None:
        """Manual trigger for copying annotations between the deoldified pair."""
        if self._current_image_id is None:
            return
        self._run_deoldified_sync(self._current_image_id, announce=True)
