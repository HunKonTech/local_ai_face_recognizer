"""Label/image coordinate transforms, face redraw and zoom/pan."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from PySide6.QtCore import (
    QRect,
)

from .helpers import (
    _bgr_to_qpixmap,
    _draw_faces,
)

log = logging.getLogger(__name__)


class CanvasMixin:
    """Label/image coordinate transforms, face redraw and zoom/pan."""

    # ──────────────────────────────────────────────────────────────────
    # Coordinate mapping
    # ──────────────────────────────────────────────────────────────────

    def _label_to_image(self, lx: int, ly: int) -> Tuple[int, int]:
        if self._full_pixmap is None:
            return -1, -1
        cr = self._image_label.contentsRect()
        lw, lh = cr.width(), cr.height()
        pw, ph = self._full_pixmap.width(), self._full_pixmap.height()
        if pw == 0 or ph == 0:
            return -1, -1
        base_scale = min(lw / pw, lh / ph)
        eff = base_scale * self._zoom
        disp_w = pw * eff
        disp_h = ph * eff
        ox = cr.x() + (lw - disp_w) / 2 + self._pan_x
        oy = cr.y() + (lh - disp_h) / 2 + self._pan_y
        rx = lx - ox
        ry = ly - oy
        if rx < 0 or ry < 0 or rx >= disp_w or ry >= disp_h:
            return -1, -1
        return int(rx / eff), int(ry / eff)

    def _label_to_image_clamped(self, lx: int, ly: int) -> Tuple[int, int]:
        """Like _label_to_image but clamps out-of-bounds to the image edge."""
        if self._full_pixmap is None:
            return -1, -1
        cr = self._image_label.contentsRect()
        lw, lh = cr.width(), cr.height()
        pw, ph = self._full_pixmap.width(), self._full_pixmap.height()
        if pw == 0 or ph == 0:
            return -1, -1
        base_scale = min(lw / pw, lh / ph)
        eff = base_scale * self._zoom
        disp_w = pw * eff
        disp_h = ph * eff
        ox = cr.x() + (lw - disp_w) / 2 + self._pan_x
        oy = cr.y() + (lh - disp_h) / 2 + self._pan_y
        rx = max(0.0, min(lx - ox, disp_w - 1))
        ry = max(0.0, min(ly - oy, disp_h - 1))
        return int(rx / eff), int(ry / eff)

    def _label_rect_to_image(
        self, rect: QRect
    ) -> Optional[Tuple[int, int, int, int]]:
        x1, y1 = self._label_to_image(rect.left(), rect.top())
        if x1 < 0 or y1 < 0:
            return None
        # Use clamped conversion for bottom-right so rects that extend into
        # the margin area are clipped to the image edge instead of becoming 1×1.
        x2, y2 = self._label_to_image_clamped(rect.right(), rect.bottom())
        if x2 < 0 or y2 < 0:
            return None
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    def _image_to_label(self, ix: int, iy: int) -> Tuple[int, int]:
        if self._full_pixmap is None:
            return -1, -1
        cr = self._image_label.contentsRect()
        lw, lh = cr.width(), cr.height()
        pw, ph = self._full_pixmap.width(), self._full_pixmap.height()
        if pw == 0 or ph == 0:
            return -1, -1
        base_scale = min(lw / pw, lh / ph)
        eff = base_scale * self._zoom
        ox = cr.x() + (lw - pw * eff) / 2 + self._pan_x
        oy = cr.y() + (lh - ph * eff) / 2 + self._pan_y
        return int(ox + ix * eff), int(oy + iy * eff)

    def _hit_test(self, ox: int, oy: int) -> Optional[int]:
        for face_id, x, y, w, h, _, _, _uncertain in self._face_data:
            if x <= ox <= x + w and y <= oy <= y + h:
                return face_id
        return None

    def _redraw_faces(
        self, *, exclude_edit_id: Optional[int] = None
    ) -> None:
        """Render the annotated image and push it to the image label.

        When *exclude_edit_id* is given, that face's bounding box is omitted
        from the baked-in PIL rendering so the handle overlay is the only
        visual representation of the editing face.
        """
        if self._orig_img_bgr is None:
            return
        log.debug(
            "Overlay re-render: image_id=%s selected_face_id=%s "
            "faces=%s exclude_edit_id=%s imode=%s imode_face=%s",
            self._current_image_id,
            self._selected_face_id,
            [(fd[0], fd[5]) for fd in self._face_data],
            exclude_edit_id,
            self._image_label._imode,
            self._interactive_edit_face_id,
        )
        render_faces = (
            [f for f in self._face_data if f[0] != exclude_edit_id]
            if exclude_edit_id is not None
            else self._face_data
        )
        annotated = _draw_faces(
            self._orig_img_bgr, render_faces, self._selected_face_id,
            show_bboxes=self._show_bboxes,
            bbox_opacity=self._bbox_opacity if self._show_bboxes else 0.0,
            show_labels=self._show_labels,
            label_opacity=self._label_opacity if self._show_labels else 0.0,
            pending_ids=self._pending_face_ids,
        )
        self._full_pixmap = _bgr_to_qpixmap(annotated)
        self._image_label.set_source_pixmap(self._full_pixmap)
        self._position_pending_action_bar()

    # ──────────────────────────────────────────────────────────────────
    # Zoom / pan
    # ──────────────────────────────────────────────────────────────────

    def _on_wheel_zoom(self, delta: int, lx: int, ly: int) -> None:
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        new_zoom = max(1.0, min(10.0, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        if self._full_pixmap is not None:
            cr = self._image_label.contentsRect()
            lw, lh = cr.width(), cr.height()
            pw, ph = self._full_pixmap.width(), self._full_pixmap.height()
            base_scale = min(lw / pw, lh / ph)
            old_eff = base_scale * self._zoom
            new_eff = base_scale * new_zoom
            old_ox = cr.x() + (lw - pw * old_eff) / 2 + self._pan_x
            old_oy = cr.y() + (lh - ph * old_eff) / 2 + self._pan_y
            ix = (lx - old_ox) / old_eff
            iy = (ly - old_oy) / old_eff
            self._pan_x = lx - (cr.x() + (lw - pw * new_eff) / 2) - ix * new_eff
            self._pan_y = ly - (cr.y() + (lh - ph * new_eff) / 2) - iy * new_eff
        self._zoom = new_zoom
        self._clamp_pan()
        self._image_label.set_zoom_pan(self._zoom, self._pan_x, self._pan_y)
        self._hide_inline_editor()
        self._position_pending_action_bar()

    def _on_pan_moved(self, dx: int, dy: int) -> None:
        self._pan_x += dx
        self._pan_y += dy
        self._clamp_pan()
        self._image_label.set_zoom_pan(self._zoom, self._pan_x, self._pan_y)
        self._hide_inline_editor()
        self._position_pending_action_bar()

    def _clamp_pan(self) -> None:
        if self._full_pixmap is None:
            self._pan_x = self._pan_y = 0.0
            return
        cr = self._image_label.contentsRect()
        lw, lh = cr.width(), cr.height()
        pw, ph = self._full_pixmap.width(), self._full_pixmap.height()
        base_scale = min(lw / pw, lh / ph)
        eff = base_scale * self._zoom
        max_x = max(0.0, (pw * eff - lw) / 2)
        max_y = max(0.0, (ph * eff - lh) / 2)
        self._pan_x = max(-max_x, min(max_x, self._pan_x))
        self._pan_y = max(-max_y, min(max_y, self._pan_y))

    def _reset_zoom(self) -> None:
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._image_label.set_zoom_pan(1.0, 0.0, 0.0)
