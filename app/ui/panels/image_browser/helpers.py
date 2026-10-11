"""Module-level helpers for the image browser (tree model, face drawing, date parsing)."""

from __future__ import annotations

import functools
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import (
    Qt,
)
from PySide6.QtGui import (
    QImage,
    QPixmap,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QWidget,
)

from app.services.image_browser_service import (
    FolderSummary,
)

log = logging.getLogger(__name__)

# (face_id, bbox_x, bbox_y, bbox_w, bbox_h, person_name_or_None, person_id_or_None, is_uncertain)
_FaceData = Tuple[int, int, int, int, int, Optional[str], Optional[int], bool]


@dataclass
class _BboxEdit:
    """One undo/redo entry for a face bounding-box modification."""
    face_id: int
    old_bbox: Tuple[int, int, int, int]  # x, y, w, h in original image coords
    new_bbox: Tuple[int, int, int, int]
    op_type: str  # "move" | "resize"


_TREE_THUMB_SIZE = 56   # max side of thumbnail icon in tree

# Data roles stored on QTreeWidgetItem
_ROLE_TYPE      = Qt.UserRole        # "folder" | "folder_node" | "image" | "dummy"
_ROLE_PAYLOAD   = Qt.UserRole + 1    # folder_path (str) | image_id (int)
_ROLE_CACHE     = Qt.UserRole + 2    # thumbnail_cache_key (str, images only)
_ROLE_FILE_PATH = Qt.UserRole + 3    # file_path (str, images only)
_ROLE_TREE_NODE = Qt.UserRole + 4    # _TreeNode (folder/folder_node items only)

# How many images ahead to prefetch in Drive mode.
_DRIVE_PREFETCH_AHEAD = 4

_UF_HIDDEN = 0x8000   # macOS BSD hidden flag (chflags hidden)


def _is_visible_dir(folder_path: str) -> bool:
    """Return True only if the folder exists and is not hidden on this platform.

    Checks:
      - directory must exist (is_dir)
      - no path component starts with '.' (Unix/macOS dot-hidden)
      - macOS: neither the folder nor any ancestor up to 6 levels has UF_HIDDEN
      - Windows: FILE_ATTRIBUTE_HIDDEN not set on the folder
    """
    p = Path(folder_path)
    if not p.is_dir():
        return False

    # Dot-prefix convention (Unix / macOS)
    if any(part.startswith(".") for part in p.parts):
        return False

    if sys.platform == "darwin":
        check = p
        for _ in range(6):
            try:
                if os.stat(check).st_flags & _UF_HIDDEN:
                    return False
            except OSError:
                return False
            parent = check.parent
            if parent == check:
                break
            check = parent

    elif sys.platform == "win32":
        try:
            import ctypes
            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(p))
            if attrs != -1 and (attrs & 0x2):   # FILE_ATTRIBUTE_HIDDEN
                return False
        except Exception:
            pass

    return True


# ──────────────────────────────────────────────────────────────────────────────
# EXIF helper
# ──────────────────────────────────────────────────────────────────────────────

def _read_exif_date(path: str) -> Optional[str]:
    """Return DateTimeOriginal from EXIF as 'YYYY.MM.DD', or None if unavailable."""
    try:
        from PIL import Image as PilImage
        with PilImage.open(path) as img:
            exif = img._getexif()  # type: ignore[attr-defined]
            if not exif:
                return None
        # Tag 36867 = DateTimeOriginal, 36868 = DateTimeDigitized, 306 = DateTime
        for tag_id in (36867, 36868, 306):
            val = exif.get(tag_id)
            if val:
                # Format is "YYYY:MM:DD HH:MM:SS"
                date_part = val.split(" ")[0]
                return date_part.replace(":", ".")
    except Exception:
        pass
    return None


_RE_FILENAME_DATETIME = re.compile(
    r"^(\d{4})\.(\d{2})\.(\d{2})-(\d{2})\.(\d{2})\.(\d{2})"
)
_RE_FILENAME_DATE = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})")


def _parse_date_from_filename(path: str) -> Optional[str]:
    """Return 'YYYY.MM.DD HH:MM:SS' or 'YYYY.MM.DD' if the filename starts with a
    date/datetime pattern, else None."""
    stem = Path(path).stem
    m = _RE_FILENAME_DATETIME.match(stem)
    if m:
        yyyy, mm, dd, hh, mi, ss = m.groups()
        return f"{yyyy}.{mm}.{dd} {hh}:{mi}:{ss}"
    m = _RE_FILENAME_DATE.match(stem)
    if m:
        yyyy, mm, dd = m.groups()
        return f"{yyyy}.{mm}.{dd}"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Rendering helpers (shared with image display)
# ──────────────────────────────────────────────────────────────────────────────

def _bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


@functools.lru_cache(maxsize=64)
def _get_pil_font(size: int):
    """Load (and cache) the PIL font used for face-name labels.

    ``_draw_faces`` calls this once per face on every overlay redraw, and
    redraws happen on every face selection/highlight change. Without caching,
    each call re-reads and re-parses a TrueType font file from disk, which
    dominates redraw time once a photo has more than a few faces.
    """
    import sys

    from PIL import ImageFont
    if sys.platform == "darwin":
        candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Arial.ttf",
        ]
    elif sys.platform == "win32":
        candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# Amber highlight for faces auto-merged from an Unknown cluster, awaiting
# review.  RGB for PIL labels; the BGR sibling is derived where boxes are drawn.
_PENDING_RGB = (255, 176, 32)


def _draw_faces(
    img_bgr: np.ndarray,
    faces: List[_FaceData],
    selected_id: Optional[int],
    show_bboxes: bool = True,
    bbox_opacity: float = 1.0,
    show_labels: bool = True,
    label_opacity: float = 1.0,
    pending_ids: Optional[set] = None,
) -> np.ndarray:
    from PIL import Image as PILImage
    from PIL import ImageDraw

    from app.ui.helpers.label_placement import FaceLabel, place_labels

    pending_ids = pending_ids or set()
    img = img_bgr.copy()
    image_h, image_w = img.shape[:2]

    # Label measurement/layout/rendering all funnel into Phase 3, which is a
    # no-op unless labels are actually visible. Skip the whole PIL round-trip
    # (BGR->RGB->PIL, per-face font loads/measurements, RGBA compositing) when
    # there is nothing to draw — this is on the hot path since every face
    # selection/highlight change re-runs this function on the full-res image.
    do_labels = show_labels and label_opacity > 0.0 and bool(faces)

    if do_labels:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        font_size = max(34, min(96, int(min(image_w, image_h) * 0.028)))

        # --- Phase 1: measure all labels ---
        _UNCERTAIN_RGB = (255, 160, 50)
        face_meta = []  # (name, font, pad_x, pad_y, label_w, label_h, color_rgb)
        for face_id, x, y, w, h, person_name, _, is_uncertain in faces:
            selected = face_id == selected_id
            if selected:
                color_rgb = (50, 220, 50)
            elif face_id in pending_ids:
                color_rgb = _PENDING_RGB
            elif is_uncertain:
                color_rgb = _UNCERTAIN_RGB
            else:
                color_rgb = (180, 180, 180)
            if face_id in pending_ids and person_name:
                name = "⚠ " + person_name
            elif is_uncertain and person_name:
                name = person_name + " (?)"
            else:
                name = person_name or "?"
            label_font_size = font_size
            font = _get_pil_font(label_font_size)

            bbox = draw.textbbox((0, 0), name, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            pad_x = max(8, label_font_size // 4)
            pad_y = max(5, label_font_size // 7)
            max_label_w = max(80, image_w - 8)
            while tw + 2 * pad_x > max_label_w and label_font_size > 26:
                label_font_size -= 2
                font = _get_pil_font(label_font_size)
                bbox = draw.textbbox((0, 0), name, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                pad_x = max(8, label_font_size // 4)
                pad_y = max(5, label_font_size // 7)

            label_w = min(tw + 2 * pad_x, image_w)
            label_h = th + 2 * pad_y
            face_meta.append((name, font, pad_x, pad_y, label_w, label_h, color_rgb))

        # --- Phase 2: compute collision-free positions ---
        face_labels = [
            FaceLabel(
                face_id=faces[i][0],
                x=faces[i][1], y=faces[i][2],
                w=faces[i][3], h=faces[i][4],
                selected=faces[i][0] == selected_id,
                label_w=face_meta[i][4],
                label_h=face_meta[i][5],
            )
            for i in range(len(faces))
        ]
        layouts = place_labels(image_w, image_h, face_labels)

        # --- Phase 3: render labels with opacity via RGBA compositing ---
        lbl_alpha = int(label_opacity * 255)
        overlay = PILImage.new("RGBA", pil_img.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)

        for i, layout in enumerate(layouts):
            name, font, pad_x, pad_y, label_w, label_h, color_rgb = face_meta[i]

            if layout.leader_start and layout.leader_end:
                odraw.line([layout.leader_start, layout.leader_end],
                           fill=(120, 120, 120, lbl_alpha), width=2)

            odraw.rectangle(
                [layout.label_x, layout.label_y,
                 layout.label_x + label_w, layout.label_y + label_h],
                fill=(20, 20, 20, lbl_alpha),
            )
            odraw.text(
                (layout.label_x + pad_x, layout.label_y + pad_y),
                name,
                font=font,
                fill=(*color_rgb, lbl_alpha),
            )

        base_rgba = pil_img.convert("RGBA")
        pil_img = PILImage.alpha_composite(base_rgba, overlay).convert("RGB")

        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # --- Phase 4: render bounding boxes ---
    # BGR amber for pending boxes (reversed _PENDING_RGB).
    pending_bgr = (_PENDING_RGB[2], _PENDING_RGB[1], _PENDING_RGB[0])

    def _box_color(face_id: int, selected: bool):
        if selected:
            return (50, 220, 50)
        if face_id in pending_ids:
            return pending_bgr
        return (180, 180, 180)

    def _draw_pending_badge(canvas, x, y):
        """Small '?' marker at the top-left corner of a pending face box."""
        r = max(10, int(min(image_w, image_h) * 0.012))
        cx, cy = x + r, y + r
        cv2.circle(canvas, (cx, cy), r, pending_bgr, -1)
        cv2.circle(canvas, (cx, cy), r, (30, 30, 30), 1)
        scale = r / 11.0
        cv2.putText(
            canvas, "?", (cx - int(5 * scale), cy + int(6 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 30), max(1, int(scale * 2)),
            cv2.LINE_AA,
        )

    if show_bboxes and bbox_opacity > 0.0:
        if bbox_opacity >= 0.99:
            for face_id, x, y, w, h, _, _, _uncertain in faces:
                selected = face_id == selected_id
                color = _box_color(face_id, selected)
                thickness = 3 if selected else 2
                cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
                if face_id in pending_ids and not selected:
                    _draw_pending_badge(img, x, y)
        else:
            overlay_cv = img.copy()
            for face_id, x, y, w, h, _, _, _uncertain in faces:
                selected = face_id == selected_id
                color = _box_color(face_id, selected)
                thickness = 3 if selected else 2
                cv2.rectangle(overlay_cv, (x, y), (x + w, y + h), color, thickness)
                if face_id in pending_ids and not selected:
                    _draw_pending_badge(overlay_cv, x, y)
            cv2.addWeighted(overlay_cv, bbox_opacity, img, 1.0 - bbox_opacity, 0, img)

    return img


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color: #3a3a3a;")
    return line


def _overlay_group() -> QWidget:
    """One overlay control trio (checkbox + slider + percentage).

    Grouping them keeps the three parts together when the toolbar above the
    image wraps onto a second row on a narrow window.
    """
    box = QWidget()
    row = QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    return box


# ──────────────────────────────────────────────────────────────────────────────
# Folder hierarchy helpers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _TreeNode:
    """Node in the pre-computed folder hierarchy."""
    path: str
    name: str
    summary: Optional[FolderSummary] = None   # None → intermediate structural node
    children: Dict[str, "_TreeNode"] = None   # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = {}

    @property
    def has_images(self) -> bool:
        return self.summary is not None

    @property
    def has_children(self) -> bool:
        return bool(self.children)


def _build_folder_tree(folders: List[FolderSummary]) -> List[_TreeNode]:
    """Build a hierarchical tree from a flat list of FolderSummary objects.

    Finds the longest common path prefix shared by all folder paths and strips it
    so only the diverging part of each path is shown in the tree.  Intermediate
    directories (containing no images directly) are created as structural nodes.
    Returns the sorted list of top-level tree nodes.
    """
    if not folders:
        return []

    summary_map = {f.folder_path: f for f in folders}
    paths = [f.folder_path for f in folders]
    parts_list = [Path(p).parts for p in paths]

    # Find common-ancestor depth.  For a single path we strip all but the last
    # component so the folder itself is shown as the root.
    if len(parts_list) == 1:
        common_len = len(parts_list[0]) - 1
    else:
        min_len = min(len(p) for p in parts_list)
        common_len = 0
        for i in range(min_len):
            if all(p[i] == parts_list[0][i] for p in parts_list):
                common_len = i + 1
            else:
                break

    # If the common ancestor path itself is one of the data folders (i.e., a
    # folder is an ancestor of another folder), go one level up so that folder
    # appears as a tree node rather than being silently skipped.
    if common_len > 0:
        common_parts = parts_list[0][:common_len]
        common_str = common_parts[0] if len(common_parts) == 1 else str(Path(*common_parts))
        if common_str in summary_map:
            common_len -= 1

    root: Dict[str, _TreeNode] = {}

    for path_str, parts in zip(paths, parts_list):
        rel_parts = parts[common_len:]
        if not rel_parts:
            continue

        current = root
        for i, part in enumerate(rel_parts):
            # Reconstruct the full path up to this depth
            full_parts = parts[:common_len + i + 1]
            full_str = full_parts[0] if len(full_parts) == 1 else str(Path(*full_parts))
            is_leaf = (i == len(rel_parts) - 1)

            if part not in current:
                current[part] = _TreeNode(
                    path=full_str,
                    name=part,
                    summary=summary_map.get(full_str) if is_leaf else None,
                )
            elif is_leaf and full_str in summary_map:
                current[part].summary = summary_map[full_str]

            current = current[part].children

    return [root[k] for k in sorted(root.keys())]
