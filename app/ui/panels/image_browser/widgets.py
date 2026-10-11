"""Small Qt widgets used by the image browser (inline editor, drawable image label)."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import (
    QPoint,
    QPointF,
    QRect,
    QRectF,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from app.db.models import Person
from app.ui.i18n import t
from app.ui.widgets.person_search_select import PersonSearchSelect

from .helpers import (
    _ROLE_TYPE,
)

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Inline face editor overlay
# ──────────────────────────────────────────────────────────────────────────────

class _FocusLineEdit(QLineEdit):
    focus_lost = Signal()
    escape_pressed = Signal()

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        self.focus_lost.emit()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.escape_pressed.emit()
        else:
            super().keyPressEvent(event)


class _PendingActionBar(QFrame):
    """Floating quick-action bar shown over an auto-merged (pending) face box.

    Three actions, mirroring the review dialog: confirm (✓), move to another
    person (⤴) and delete the face + its box (🗑).
    """

    accept_requested = Signal()
    move_requested = Signal()
    delete_requested = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "_PendingActionBar { background: #2a2118; border: 1px solid #ffb020;"
            " border-radius: 4px; }"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(4)

        def _btn(text: str, tip: str, color: str) -> QPushButton:
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setFixedHeight(24)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(
                f"QPushButton {{ color: {color}; font-size: 13px; font-weight: bold;"
                " background: #3a3127; border: 1px solid #5a4a30; border-radius: 3px;"
                " padding: 1px 8px; }"
                "QPushButton:hover { background: #4a3d2a; }"
            )
            return b

        self._accept_btn = _btn("✓", t("amerge_accept"), "#7CFC7C")
        self._move_btn = _btn("⤴", t("amerge_move"), "#FFD27A")
        self._delete_btn = _btn("🗑", t("amerge_delete"), "#FF8A8A")
        self._accept_btn.clicked.connect(self.accept_requested)
        self._move_btn.clicked.connect(self.move_requested)
        self._delete_btn.clicked.connect(self.delete_requested)
        layout.addWidget(self._accept_btn)
        layout.addWidget(self._move_btn)
        layout.addWidget(self._delete_btn)
        self.adjustSize()


class _InlineFaceEditor(QFrame):
    assign_requested = Signal(int)   # person_id
    create_requested = Signal(str)   # new person name
    closed = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "_InlineFaceEditor { background: #252525; border: 1px solid #666;"
            " border-radius: 4px; }"
        )
        self.setFixedWidth(230)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(4)

        hdr = QHBoxLayout()
        self._name_lbl = QLabel("?")
        self._name_lbl.setStyleSheet(
            "color: #88ee88; font-weight: bold; font-size: 11px;"
        )
        # The header doubles as a drag handle — hint it with a move cursor.
        self._name_lbl.setCursor(Qt.SizeAllCursor)
        hdr.addWidget(self._name_lbl, stretch=1)
        close_btn = QPushButton("×")
        close_btn.setFixedSize(16, 16)
        close_btn.setFlat(True)
        close_btn.setStyleSheet(
            "QPushButton { color: #888; font-size: 14px; padding: 0; border: none; }"
            "QPushButton:hover { color: #fff; }"
        )
        close_btn.clicked.connect(self.closed)
        hdr.addWidget(close_btn)
        layout.addLayout(hdr)

        self._person_search = PersonSearchSelect()
        self._person_search.person_selected.connect(self._on_assign)
        layout.addWidget(self._person_search)

        self._assign_btn = QPushButton()
        self._assign_btn.clicked.connect(self._on_assign)
        layout.addWidget(self._assign_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3a3a;")
        layout.addWidget(sep)

        self._new_edit = QLineEdit()
        self._new_edit.returnPressed.connect(self._on_create)
        layout.addWidget(self._new_edit)

        self._create_btn = QPushButton()
        self._create_btn.clicked.connect(self._on_create)
        layout.addWidget(self._create_btn)

        self.adjustSize()
        self.hide()

        # Drag-to-move state. The popup can be repositioned by grabbing its
        # header strip (anything above the person search) so it never has to
        # cover the face the user is trying to look at.
        self._drag_origin: Optional[QPoint] = None

    def retranslate(self) -> None:
        self._assign_btn.setText(t("ibp_assign_btn"))
        self._new_edit.setPlaceholderText(t("ibp_new_placeholder"))
        self._create_btn.setText(t("ibp_create_btn"))
        self._person_search.retranslate()

    def populate(
        self,
        person_name: Optional[str],
        person_id: Optional[int],
        persons: List[Person],
        priority_ids: Optional[List[int]] = None,
        match_scores: Optional[dict[int, float]] = None,
        scores_pending: bool = False,
    ) -> None:
        self._name_lbl.setText(person_name or "?")
        # Forget the previously edited face's selection so a new face never
        # inherits it (e.g. "Apa" staying highlighted while editing "Jerne").
        self._person_search.clear_selection()
        self._person_search.set_persons(persons, priority_ids=priority_ids)
        if scores_pending and not match_scores:
            # Background scoring is running — show "computing…" instead of
            # silently missing percentages (apply_late_match_scores clears it).
            self._person_search.set_scores_pending()
        else:
            self._person_search.set_match_scores(match_scores)
        if person_id is not None:
            self._person_search.set_current_by_id(person_id)
        else:
            # Unassigned face: start at the top so the strongest matches are
            # visible, and preselect the best match when match-sorting is on.
            self._person_search.preselect_best_match()
        self._new_edit.clear()

    def apply_late_match_scores(
        self, match_scores: Optional[dict[int, float]], *, preselect_best: bool
    ) -> None:
        """Apply face-match scores computed after the editor already opened.

        The popup opens instantly in name order (no blocking scoring on the GUI
        thread); this re-ranks the list once background scoring finishes.  When
        *preselect_best* (an unassigned face), the top match is re-selected.
        """
        self._person_search.set_match_scores(match_scores)
        if preselect_best:
            self._person_search.preselect_best_match()

    def reset_search(self) -> None:
        """Clear the person search filter after a successful assignment."""
        self._person_search.reset_filter()

    def focus_search(self) -> None:
        """Set keyboard focus to the person search input."""
        self._person_search.focus_search()

    def _on_assign(self, _person_id: int = 0) -> None:
        pid = self._person_search.current_person_id()
        if pid is not None:
            self.assign_requested.emit(pid)

    def _on_create(self) -> None:
        name = self._new_edit.text().strip()
        if name:
            self.create_requested.emit(name)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.closed.emit()
        else:
            super().keyPressEvent(event)

    # ── Drag-to-move (grab the header strip) ─────────────────────────────
    def _in_drag_handle(self, pos: QPoint) -> bool:
        # The header is everything above the person search widget. The close
        # button lives there too, but it consumes its own clicks before they
        # reach the frame, so grabbing the header never closes the popup.
        return pos.y() < self._person_search.y()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._in_drag_handle(
            event.position().toPoint()
        ):
            self._drag_origin = event.position().toPoint()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_origin is not None:
            new_pos = self.mapToParent(event.position().toPoint() - self._drag_origin)
            parent = self.parentWidget()
            if parent is not None:
                new_pos.setX(max(0, min(new_pos.x(), parent.width() - self.width())))
                new_pos.setY(max(0, min(new_pos.y(), parent.height() - self.height())))
            self.move(new_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_origin is not None and event.button() == Qt.LeftButton:
            self._drag_origin = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        # The inner list scrolls itself; when it reaches its top/bottom Qt
        # propagates the leftover wheel event up to this frame.  Swallow it here
        # so it can never reach the image label behind us and zoom the picture
        # (which would close the popup mid-selection).
        event.accept()


# ──────────────────────────────────────────────────────────────────────────────
# Drawable image label (zoom + pan + draw mode)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Tree widget subclass — intercepts Left/Right on image items for navigation
# ──────────────────────────────────────────────────────────────────────────────

class _BrowserTree(QTreeWidget):
    go_prev = Signal()
    go_next = Signal()

    def keyPressEvent(self, event) -> None:
        current = self.currentItem()
        if current is not None and current.data(0, _ROLE_TYPE) == "image":
            from app.services.shortcut_service import get_shortcut_service, normalize_key
            svc = get_shortcut_service()
            key_str = normalize_key(event)
            if svc.is_enabled():
                prev_sc = svc.get("image.previous")
                next_sc = svc.get("image.next")
                prev_key = (prev_sc.current_key if prev_sc and prev_sc.current_key else "Left")
                next_key = (next_sc.current_key if next_sc and next_sc.current_key else "Right")
            else:
                prev_key, next_key = "Left", "Right"
            if key_str == prev_key:
                self.go_prev.emit()
                event.accept()
                return
            elif key_str == next_key:
                self.go_next.emit()
                event.accept()
                return
        super().keyPressEvent(event)


class _DrawableImageLabel(QLabel):
    clicked              = Signal(int, int)
    rect_drawn           = Signal(QRect)
    right_clicked        = Signal(int, int)
    wheel_zoomed         = Signal(int, int, int)
    pan_moved            = Signal(int, int)
    go_prev              = Signal()
    go_next              = Signal()
    # Interactive bbox-edit mode signals
    bbox_edit_committed  = Signal(int, int, int, int)   # drag-release save, stay in edit mode
    bbox_edit_finalized  = Signal(int, int, int, int)   # click-outside / Enter → save AND exit
    bbox_edit_cancelled  = Signal()
    undo_requested       = Signal()
    redo_requested       = Signal()
    # Deoldified compare divider dragged → emits split position as percent (0..100)
    compare_dragged      = Signal(int)
    # Object-tagging: a rectangle was drawn in object mode (label-space QRect)
    object_rect_drawn    = Signal(QRect)

    _BORDER_NORMAL = "QLabel { background: #1a1a1a; }"
    _BORDER_DRAW   = "QLabel { background: #1a1a1a; border: 2px solid #ffcc00; }"
    _BORDER_EDIT   = "QLabel { background: #1a1a1a; border: 2px solid #32dc32; }"
    _BORDER_OBJECT = "QLabel { background: #1a1a1a; border: 2px solid #50c8ff; }"

    # Handle rendering constants
    _HANDLE_HALF = 6    # half-side of handle squares in display pixels
    _HIT_RADIUS  = 10   # hit-test radius around each handle
    _MIN_DIM     = 12   # minimum bbox dimension in image pixels

    # Handle identifiers (corners, edge-midpoints, centre)
    _HANDLES = ("tl", "tc", "tr", "ml", "mc", "mr", "bl", "bc", "br")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.CrossCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)   # needed for hover cursor changes without button held
        self._draw_mode = False
        self._object_mode = False
        # Object occurrence markers in image coords:
        # (occ_id, x, y, name, bbox_or_None) where bbox = (bx, by, bw, bh)
        self._object_markers: list[tuple] = []
        self._selected_occ_id: Optional[int] = None
        self._show_objects: bool = True
        self._object_opacity: float = 0.8
        self._start: Optional[QPoint] = None
        self._end: Optional[QPoint] = None
        self._mid_start: Optional[QPoint] = None
        self._source_pix: Optional[QPixmap] = None
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        # Interactive edit mode
        self._imode: bool = False
        self._imode_bbox: Optional[Tuple[int, int, int, int]] = None   # image coords
        self._imode_img_w: int = 0
        self._imode_img_h: int = 0
        self._imode_handle: Optional[str] = None   # which handle is being dragged
        self._imode_drag_start: Optional[QPointF] = None
        self._imode_drag_start_bbox: Optional[Tuple[int, int, int, int]] = None
        self._last_op_type: str = "resize"   # set by mouseReleaseEvent
        # Deoldified compare divider (drawn directly on the image)
        self._compare: bool = False
        self._compare_split: float = 0.5      # fraction 0..1 (left=B&W, right=color)
        self._compare_dragging: bool = False
        self.setStyleSheet(self._BORDER_NORMAL)

    _COMPARE_HIT = 16   # px hit radius around the compare divider line

    # ── Public setters ─────────────────────────────────────────────────

    def set_source_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        self._source_pix = pixmap
        self.update()

    def set_zoom_pan(self, zoom: float, pan_x: float, pan_y: float) -> None:
        self._zoom = zoom
        self._pan_x = pan_x
        self._pan_y = pan_y
        self.update()

    def set_draw_mode(self, enabled: bool) -> None:
        self._draw_mode = enabled
        self._start = None
        self._end = None
        if enabled:
            self._exit_interactive_edit(commit=False, silent=True)
        self.setStyleSheet(self._BORDER_DRAW if enabled else self._BORDER_NORMAL)
        self.update()

    def set_object_mode(self, enabled: bool) -> None:
        """Toggle object-tagging mode (click places an object marker)."""
        self._object_mode = enabled
        if enabled:
            self._draw_mode = False
            self._start = None
            self._end = None
            self.setStyleSheet(self._BORDER_OBJECT)
        else:
            self.setStyleSheet(self._BORDER_NORMAL)
        self.update()

    def set_object_markers(self, markers: list[tuple]) -> None:
        """Set object occurrence markers (image coords) to overlay.

        Each marker is ``(occ_id, x, y, name, bbox_or_None)`` where bbox is
        ``(bx, by, bw, bh)`` in image coords.
        """
        self._object_markers = markers
        self.update()

    def set_object_overlay(self, show: bool, opacity: float) -> None:
        """Control object overlay visibility and opacity (0.0–1.0)."""
        self._show_objects = show
        self._object_opacity = max(0.0, min(1.0, opacity))
        self.update()

    def set_selected_object(self, occ_id: Optional[int]) -> None:
        """Highlight the marker for *occ_id* (or clear when None)."""
        self._selected_occ_id = occ_id
        self.update()

    def set_compare_mode(self, enabled: bool, split_percent: int = 50) -> None:
        """Enable/disable the on-image compare divider at *split_percent*."""
        self._compare = enabled
        self._compare_split = max(0.0, min(1.0, split_percent / 100.0))
        self._compare_dragging = False
        if not enabled:
            self.setCursor(Qt.CrossCursor)
        self.update()

    def set_compare_split(self, split_percent: int) -> None:
        """Move the compare divider to *split_percent* (0..100)."""
        self._compare_split = max(0.0, min(1.0, split_percent / 100.0))
        self.update()

    def _compare_divider_x(self) -> Optional[float]:
        """Display-x of the divider line, or None when not applicable."""
        if not self._compare or self._source_pix is None:
            return None
        t = self._imode_transform()
        if t is None:
            return None
        eff, ox, _oy = t
        disp_w = self._source_pix.width() * eff
        return ox + self._compare_split * disp_w

    def _set_compare_from_x(self, lx: float) -> None:
        if self._source_pix is None:
            return
        t = self._imode_transform()
        if t is None:
            return
        eff, ox, _oy = t
        disp_w = self._source_pix.width() * eff
        if disp_w <= 0:
            return
        frac = max(0.0, min(1.0, (lx - ox) / disp_w))
        self._compare_split = frac
        self.update()
        self.compare_dragged.emit(int(round(frac * 100)))

    def set_interactive_edit(
        self,
        bbox: Tuple[int, int, int, int],
        img_w: int,
        img_h: int,
    ) -> None:
        """Enter interactive-edit mode showing resize/move handles for *bbox*."""
        self._imode = True
        self._imode_bbox = bbox
        self._imode_img_w = img_w
        self._imode_img_h = img_h
        self._imode_handle = None
        self._imode_drag_start = None
        self._imode_drag_start_bbox = None
        self._draw_mode = False
        self._start = None
        self._end = None
        self.setStyleSheet(self._BORDER_EDIT)
        # Show the correct cursor for the current pointer position right away
        # (over a handle → resize/move, otherwise cross) instead of waiting for
        # the next move event.
        self._refresh_hover_cursor()
        self.update()

    def cancel_interactive_edit(self) -> None:
        """Exit interactive-edit mode; emit bbox_edit_cancelled."""
        self._exit_interactive_edit(commit=False, silent=False)

    def commit_interactive_edit(self) -> None:
        """Commit the current bbox and exit interactive-edit mode."""
        self._exit_interactive_edit(commit=True, silent=False)

    def dismiss_interactive_edit(self) -> None:
        """Silently exit interactive-edit mode without emitting any signal.

        Use this when the bbox was already persisted by intermediate commits
        and you just want to clear the imode UI state without triggering any
        save or restore logic.
        """
        self._exit_interactive_edit(commit=False, silent=True)

    def _exit_interactive_edit(
        self, *, commit: bool, silent: bool
    ) -> None:
        if not self._imode:
            return
        bbox = self._imode_bbox
        self._imode = False
        self._imode_bbox = None
        self._imode_handle = None
        self._imode_drag_start = None
        self._imode_drag_start_bbox = None
        self.setStyleSheet(self._BORDER_NORMAL)
        # Reset the cursor: during edit mode it may have been changed to a
        # resize/move shape by _update_edit_cursor(); restore the panel default
        # and drain any override cursor. _imode is already False above so this
        # lands on the plain cross.
        self._reset_cursor_state()
        self.update()
        if not silent:
            if commit and bbox is not None:
                # finalized = save AND exit (click-outside or Enter)
                self.bbox_edit_finalized.emit(*bbox)
            else:
                self.bbox_edit_cancelled.emit()

    # ── Coordinate helpers ──────────────────────────────────────────────

    def _imode_transform(self) -> Optional[Tuple[float, float, float]]:
        """Return (eff, ox, oy) for image→display mapping, or None."""
        if self._source_pix is None:
            return None
        cr = self.contentsRect()
        lw, lh = cr.width(), cr.height()
        pw, ph = self._source_pix.width(), self._source_pix.height()
        if pw <= 0 or ph <= 0 or lw <= 0 or lh <= 0:
            return None
        base_scale = min(lw / pw, lh / ph)
        eff = base_scale * self._zoom
        disp_w = pw * eff
        disp_h = ph * eff
        ox = cr.x() + (lw - disp_w) / 2 + self._pan_x
        oy = cr.y() + (lh - disp_h) / 2 + self._pan_y
        return eff, ox, oy

    def _img_to_disp(
        self, ix: float, iy: float,
        eff: float, ox: float, oy: float,
    ) -> Tuple[float, float]:
        return ox + ix * eff, oy + iy * eff

    def _disp_to_img(
        self, lx: float, ly: float,
        eff: float, ox: float, oy: float,
    ) -> Tuple[float, float]:
        return (lx - ox) / eff, (ly - oy) / eff

    def _handle_positions(
        self,
        bbox: Tuple[int, int, int, int],
        eff: float,
        ox: float,
        oy: float,
    ) -> Dict[str, Tuple[float, float]]:
        """Return display-coord centre of each handle for *bbox*."""
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        img_pts = {
            "tl": (x,      y),
            "tc": (cx,     y),
            "tr": (x + w,  y),
            "ml": (x,      cy),
            "mc": (cx,     cy),
            "mr": (x + w,  cy),
            "bl": (x,      y + h),
            "bc": (cx,     y + h),
            "br": (x + w,  y + h),
        }
        return {
            name: self._img_to_disp(ix, iy, eff, ox, oy)
            for name, (ix, iy) in img_pts.items()
        }

    def _hit_test_handle(
        self, lx: float, ly: float
    ) -> Optional[str]:
        """Return the name of the nearest handle if within hit radius."""
        if not self._imode or self._imode_bbox is None:
            return None
        t = self._imode_transform()
        if t is None:
            return None
        eff, ox, oy = t
        positions = self._handle_positions(self._imode_bbox, eff, ox, oy)
        best_name: Optional[str] = None
        best_dist = self._HIT_RADIUS + 1.0
        for name, (hx, hy) in positions.items():
            dist = ((lx - hx) ** 2 + (ly - hy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_name = name
        return best_name

    # ── Drag update ─────────────────────────────────────────────────────

    def _apply_drag(
        self,
        handle: str,
        start: QPointF,
        current: QPointF,
    ) -> None:
        """Update _imode_bbox according to the current drag position."""
        if self._imode_drag_start_bbox is None:
            return
        t = self._imode_transform()
        if t is None:
            return
        eff, ox, oy = t

        sx, sy = self._disp_to_img(start.x(), start.y(), eff, ox, oy)
        cx, cy = self._disp_to_img(current.x(), current.y(), eff, ox, oy)
        dx, dy = cx - sx, cy - sy

        x0, y0, w0, h0 = self._imode_drag_start_bbox
        iw, ih = self._imode_img_w, self._imode_img_h
        _m = self._MIN_DIM

        if handle == "mc":
            # Move the whole box
            nx = max(0, min(iw - w0, x0 + dx))
            ny = max(0, min(ih - h0, y0 + dy))
            self._imode_bbox = (int(nx), int(ny), w0, h0)
            return

        # Resize: determine which edges move
        nx, ny, nw, nh = x0, y0, w0, h0

        if handle in ("tl", "ml", "bl"):
            new_x = max(0, min(x0 + w0 - _m, x0 + dx))
            nw = w0 - (int(new_x) - x0)
            nx = int(new_x)
        if handle in ("tr", "mr", "br"):
            new_w = max(_m, min(iw - x0, w0 + dx))
            nw = int(new_w)
        if handle in ("tl", "tc", "tr"):
            new_y = max(0, min(y0 + h0 - _m, y0 + dy))
            nh = h0 - (int(new_y) - y0)
            ny = int(new_y)
        if handle in ("bl", "bc", "br"):
            new_h = max(_m, min(ih - y0, h0 + dy))
            nh = int(new_h)

        # Final clamp: ensure box stays inside image
        nx = max(0, min(iw - nw, nx))
        ny = max(0, min(ih - nh, ny))
        nw = max(_m, min(iw - nx, nw))
        nh = max(_m, min(ih - ny, nh))
        self._imode_bbox = (nx, ny, nw, nh)

    # ── Mouse events ────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._mid_start = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() != Qt.LeftButton:
            return
        pos = event.position()
        lx, ly = pos.x(), pos.y()

        if self._compare:
            dx = self._compare_divider_x()
            if dx is not None and abs(lx - dx) <= self._COMPARE_HIT:
                self._compare_dragging = True
                self._set_compare_from_x(lx)
                return
            # Not grabbing the divider → fall through (face select / pan).

        if self._imode:
            handle = self._hit_test_handle(lx, ly)
            if handle is not None:
                self._imode_handle = handle
                self._imode_drag_start = pos
                self._imode_drag_start_bbox = self._imode_bbox
            # Click outside any handle exits edit mode
            elif self._imode_bbox is not None:
                self._exit_interactive_edit(commit=True, silent=False)
            return

        if self._object_mode:
            # Draw a rectangle just like face draw mode.
            self._start = event.position().toPoint()
            self._end = self._start
            self.update()
            return

        if self._draw_mode:
            self._start = event.position().toPoint()
            self._end = self._start
            self.update()
        else:
            self.clicked.emit(int(lx), int(ly))

    def mouseMoveEvent(self, event) -> None:
        if self._mid_start is not None:
            cur = event.position().toPoint()
            self.pan_moved.emit(
                cur.x() - self._mid_start.x(),
                cur.y() - self._mid_start.y(),
            )
            self._mid_start = cur
            return

        pos = event.position()
        lx, ly = pos.x(), pos.y()

        if self._compare_dragging:
            self._set_compare_from_x(lx)
            return
        if self._compare:
            dx = self._compare_divider_x()
            near = dx is not None and abs(lx - dx) <= self._COMPARE_HIT
            self.setCursor(Qt.SplitHCursor if near else Qt.CrossCursor)
            # fall through so hover/other behaviour still works when not near

        if self._imode and self._imode_handle is not None:
            self._apply_drag(self._imode_handle, self._imode_drag_start, pos)
            # Update cursor based on which handle is being dragged
            self._update_edit_cursor(self._imode_handle)
            self.update()
            return

        if self._imode:
            # Hover: update cursor to show which handle would be grabbed
            handle = self._hit_test_handle(lx, ly)
            self._update_edit_cursor(handle)
            return

        if (self._draw_mode or self._object_mode) and self._start is not None:
            self._end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._mid_start = None
            self._reset_cursor_state()
            return
        if event.button() != Qt.LeftButton:
            return

        if self._compare_dragging:
            self._compare_dragging = False
            return

        if self._imode and self._imode_handle is not None:
            was_handle = self._imode_handle
            old_bbox = self._imode_drag_start_bbox
            new_bbox = self._imode_bbox
            self._imode_handle = None
            self._imode_drag_start = None
            self._imode_drag_start_bbox = None
            # Emit a committed signal if the bbox actually changed
            if new_bbox is not None and old_bbox != new_bbox:
                # Determine op type and tag BEFORE emitting (slot runs synchronously)
                ow, oh = old_bbox[2], old_bbox[3]
                nw, nh = new_bbox[2], new_bbox[3]
                self._last_op_type = "move" if (ow == nw and oh == nh) else "resize"
                self.bbox_edit_committed.emit(*new_bbox)
            # Drag finished: the resize/move cursor set during the drag must
            # not stay frozen. Re-evaluate hover for the (possibly moved) box.
            self._reset_cursor_state()
            self.update()
            return

        if self._object_mode and self._start is not None:
            end = event.position().toPoint()
            rect = QRect(self._start, end).normalized()
            self._start = None
            self._end = None
            self.update()
            if rect.width() >= 8 and rect.height() >= 8:
                self.object_rect_drawn.emit(rect)
            return

        if not self._draw_mode or self._start is None:
            return
        end = event.position().toPoint()
        rect = QRect(self._start, end).normalized()
        self._start = None
        self._end = None
        self.update()
        if rect.width() >= 8 and rect.height() >= 8:
            self.rect_drawn.emit(rect)

    def _update_edit_cursor(self, handle: Optional[str]) -> None:
        cursors = {
            "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
            "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
            "tc": Qt.SizeVerCursor,   "bc": Qt.SizeVerCursor,
            "ml": Qt.SizeHorCursor,   "mr": Qt.SizeHorCursor,
            "mc": Qt.SizeAllCursor,
        }
        if handle and handle in cursors:
            self.setCursor(cursors[handle])
        else:
            self.setCursor(Qt.CrossCursor)

    # ── Cursor reset ────────────────────────────────────────────────────

    def _refresh_hover_cursor(self) -> None:
        """Set the widget cursor based on what is under the pointer *now*.

        After a drag/move/resize ends the active interaction cursor (e.g. a
        diagonal resize arrow) must not stay frozen on screen. This recomputes
        the correct cursor for the current mouse position and mode instead of
        waiting for the next ``mouseMoveEvent``.
        """
        # While in interactive-edit mode, show a handle cursor if the pointer
        # is over a handle, otherwise the plain cross.
        if self._imode and self._imode_handle is None:
            local = self.mapFromGlobal(QCursor.pos())
            handle = self._hit_test_handle(local.x(), local.y())
            self._update_edit_cursor(handle)
            return
        if self._imode:
            # An active drag is in progress — leave the drag cursor as-is.
            return
        # Compare-divider hover is handled live in mouseMoveEvent; fall back
        # to the panel default cross cursor everywhere else.
        self.setCursor(Qt.CrossCursor)

    def _reset_cursor_state(self) -> None:
        """Robustly restore the cursor after any interaction finishes.

        Clears stale drag/move/resize handle state, drains any application
        override cursors that may have been left active, and re-evaluates the
        widget cursor for the current pointer position. Safe to call from any
        exit path (release, commit, cancel, ESC, leave, view refresh).
        """
        # Clear interaction flags so nothing stays "grabbed".
        self._imode_handle = None
        self._imode_drag_start = None
        self._imode_drag_start_bbox = None
        self._mid_start = None
        self._compare_dragging = False

        # Balance any QApplication.setOverrideCursor() calls (e.g. a busy
        # cursor pushed during a long save). Drain until the stack is empty so
        # a missed restore elsewhere cannot leave the cursor stuck.
        app = QApplication.instance()
        if app is not None:
            guard = 0
            while app.overrideCursor() is not None and guard < 16:
                QApplication.restoreOverrideCursor()
                guard += 1

        self._refresh_hover_cursor()

    def leaveEvent(self, event) -> None:
        """Reset the cursor when the pointer leaves the image view.

        Without this a resize/move cursor set mid-drag could remain visible if
        the pointer slipped out of the widget. Skips reset while a drag is
        genuinely in progress so the active gesture is not disturbed.
        """
        if self._imode_handle is None and not self._compare_dragging \
                and self._mid_start is None:
            self._reset_cursor_state()
        super().leaveEvent(event)

    # ── Key events ──────────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        from app.services.shortcut_service import get_shortcut_service, normalize_key
        svc = get_shortcut_service()
        key_str = normalize_key(event)

        if self._imode:
            if key_str == "Esc":
                self._exit_interactive_edit(commit=False, silent=False)
                event.accept()
                return
            if key_str in ("Return", "Enter"):
                self._exit_interactive_edit(commit=True, silent=False)
                event.accept()
                return

        if svc.is_enabled():
            undo_sc = svc.get("bbox.undo")
            redo_sc = svc.get("bbox.redo")
            undo_key = undo_sc.current_key if undo_sc and undo_sc.current_key else "Ctrl+Z"
            redo_key = redo_sc.current_key if redo_sc and redo_sc.current_key else "Ctrl+Y"
            prev_sc = svc.get("image.previous")
            next_sc = svc.get("image.next")
            prev_key = prev_sc.current_key if prev_sc and prev_sc.current_key else "Left"
            next_key = next_sc.current_key if next_sc and next_sc.current_key else "Right"
        else:
            undo_key, redo_key = "Ctrl+Z", "Ctrl+Y"
            prev_key, next_key = "Left", "Right"

        if key_str == undo_key:
            self.undo_requested.emit()
            event.accept()
        elif key_str == redo_key:
            self.redo_requested.emit()
            event.accept()
        elif key_str == prev_key and not self._imode:
            self.go_prev.emit()
            event.accept()
        elif key_str == next_key and not self._imode:
            self.go_next.emit()
            event.accept()
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:
        if self._imode:
            # Cancel edit before opening context menu
            self._exit_interactive_edit(commit=True, silent=False)
        self.right_clicked.emit(event.pos().x(), event.pos().y())

    def wheelEvent(self, event) -> None:
        pos = event.position().toPoint()
        # Defensive guard: if an overlay child (the inline person editor) sits
        # under the cursor, never zoom — the overlay owns the wheel there.  The
        # overlay normally swallows the event itself; this is the belt to that
        # suspenders so a stray propagation can never zoom/close the popup.
        if self.childAt(pos) is not None:
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta != 0:
            self.wheel_zoomed.emit(delta, pos.x(), pos.y())
        event.accept()

    # ── Painting ────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        cr = self.contentsRect()
        painter.fillRect(cr, Qt.black)

        if self._source_pix is not None:
            lw, lh = cr.width(), cr.height()
            pw, ph = self._source_pix.width(), self._source_pix.height()
            if pw > 0 and ph > 0:
                base_scale = min(lw / pw, lh / ph)
                eff = base_scale * self._zoom
                disp_w = pw * eff
                disp_h = ph * eff
                ox = cr.x() + (lw - disp_w) / 2 + self._pan_x
                oy = cr.y() + (lh - disp_h) / 2 + self._pan_y
                painter.drawPixmap(
                    QRect(int(ox), int(oy), int(disp_w), int(disp_h)),
                    self._source_pix,
                )
                if self._compare:
                    dx = ox + self._compare_split * disp_w
                    dx = max(ox, min(ox + disp_w, dx))
                    cyc = oy + disp_h / 2
                    painter.setRenderHint(QPainter.Antialiasing, True)
                    painter.setPen(QPen(QColor(255, 255, 255), 2))
                    painter.drawLine(
                        QPointF(dx, oy), QPointF(dx, oy + disp_h)
                    )
                    painter.setPen(QPen(QColor(0, 0, 0, 150), 1))
                    painter.setBrush(QBrush(QColor(255, 255, 255)))
                    painter.drawEllipse(QPointF(dx, cyc), 10.0, 10.0)
                    painter.setPen(QPen(QColor(20, 20, 20), 1))
                    painter.drawText(
                        QRectF(dx - 10, cyc - 10, 20, 20),
                        Qt.AlignCenter, "⇔",
                    )
        else:
            painter.setPen(Qt.gray)
            painter.drawText(cr, Qt.AlignCenter | Qt.TextWordWrap, self.text())

        if self._object_markers and self._show_objects:
            self._paint_object_markers(painter)

        if self._draw_mode and self._start is not None and self._end is not None:
            painter.setPen(QPen(Qt.yellow, 2, Qt.DashLine))
            painter.drawRect(QRect(self._start, self._end).normalized())

        if self._object_mode and self._start is not None and self._end is not None:
            painter.setPen(QPen(QColor(80, 200, 255), 2, Qt.DashLine))
            painter.drawRect(QRect(self._start, self._end).normalized())

        if self._imode and self._imode_bbox is not None:
            self._paint_edit_handles(painter)

    def _paint_object_markers(self, painter: QPainter) -> None:
        """Draw cyan object boxes (rectangle + name) using the current transform.

        Bounding-box occurrences render as a rectangle (like a face box); legacy
        point-only occurrences fall back to a small dot.  The selected marker is
        always shown at full opacity with a white highlight; the rest honour the
        object-overlay opacity slider.
        """
        t = self._imode_transform()
        if t is None:
            return
        eff, ox, oy = t
        painter.setRenderHint(QPainter.Antialiasing, True)
        font = painter.font()
        font.setPixelSize(12)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        base_op = self._object_opacity
        for marker in self._object_markers:
            occ_id, ix, iy, name = marker[0], marker[1], marker[2], marker[3]
            bbox = marker[4] if len(marker) > 4 else None
            selected = occ_id == self._selected_occ_id
            painter.setOpacity(1.0 if selected else base_op)

            if bbox is not None:
                bx, by, bw, bh = bbox
                dx = ox + bx * eff
                dy = oy + by * eff
                dw = bw * eff
                dh = bh * eff
                if selected:
                    painter.setPen(QPen(QColor(255, 255, 255), 3))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(QRectF(dx - 2, dy - 2, dw + 4, dh + 4))
                painter.setPen(QPen(QColor(80, 200, 255), 3 if selected else 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(QRectF(dx, dy, dw, dh))
                label_x = dx
                label_y = dy - 4
            else:
                dx = ox + ix * eff
                dy = oy + iy * eff
                if selected:
                    painter.setPen(QPen(QColor(255, 255, 255), 2))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawEllipse(QPointF(dx, dy), 11.0, 11.0)
                painter.setPen(QPen(QColor(80, 200, 255), 3 if selected else 2))
                painter.setBrush(QBrush(QColor(80, 200, 255, 220 if selected else 170)))
                r = 8.0 if selected else 7.0
                painter.drawEllipse(QPointF(dx, dy), r, r)
                painter.setBrush(QBrush(QColor(255, 255, 255, 230)))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(dx, dy), 2.4, 2.4)
                label_x = dx
                label_y = dy - 9

            if name:
                tw = metrics.horizontalAdvance(name)
                th = metrics.height()
                lx = label_x
                ly = label_y - th
                painter.setPen(QPen(QColor(80, 200, 255, 200), 1))
                painter.setBrush(QBrush(QColor(0, 35, 50, 210)))
                painter.drawRoundedRect(QRectF(lx, ly, tw + 10, th + 4), 3, 3)
                painter.setPen(QColor(80, 200, 255))
                painter.setBrush(Qt.NoBrush)
                painter.drawText(
                    QRectF(lx + 5, ly + 2, tw, th),
                    Qt.AlignCenter,
                    name,
                )
        painter.setOpacity(1.0)

    def _paint_edit_handles(self, painter: QPainter) -> None:
        t = self._imode_transform()
        if t is None:
            return
        eff, ox, oy = t
        bbox = self._imode_bbox
        x, y, w, h = bbox
        dx = ox + x * eff
        dy = oy + y * eff
        dw = w * eff
        dh = h * eff

        painter.setRenderHint(QPainter.Antialiasing, True)

        # Bbox outline (bright green dashed)
        painter.setPen(QPen(QColor(50, 220, 50), 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRectF(dx, dy, dw, dh))

        # Handles
        positions = self._handle_positions(bbox, eff, ox, oy)
        hs = self._HANDLE_HALF
        for name, (hx, hy) in positions.items():
            if name == "mc":
                # Centre handle: filled circle
                painter.setPen(QPen(QColor(50, 220, 50), 1))
                painter.setBrush(QBrush(QColor(50, 220, 50, 200)))
                painter.drawEllipse(QPointF(hx, hy), float(hs), float(hs))
            else:
                painter.setPen(QPen(QColor(50, 220, 50), 1))
                painter.setBrush(QBrush(QColor(20, 40, 20, 220)))
                painter.drawRect(QRectF(hx - hs, hy - hs, hs * 2, hs * 2))


# ──────────────────────────────────────────────────────────────────────────────
