"""Preview panel — shows the original image with face bounding boxes overlaid
when a face thumbnail is clicked in the cluster panel.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.db.models import Face

log = logging.getLogger(__name__)

_MAX_PREVIEW_W = 600
_MAX_PREVIEW_H = 500


def _bgr_to_qpixmap(img_bgr: np.ndarray, max_w: int, max_h: int) -> QPixmap:
    """Convert a BGR numpy array to a QPixmap scaled to fit *max_w*×*max_h*."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)
    return pixmap.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class PreviewPanel(QWidget):
    """Shows a full image preview with the selected face highlighted."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_image_path: Optional[str] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._image_label = QLabel("Click a face thumbnail to preview")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(300, 200)
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_label.setStyleSheet("QLabel { background: #222; border: 1px solid #444; }")
        layout.addWidget(self._image_label)

        self._path_label = QLabel("")
        self._path_label.setWordWrap(True)
        self._path_label.setStyleSheet("QLabel { color: #aaa; font-size: 10px; }")
        layout.addWidget(self._path_label)

        btn_row = QHBoxLayout()
        self._open_btn = QPushButton("Open in File Manager")
        self._open_btn.setEnabled(False)
        self._open_btn.clicked.connect(self._open_in_filemanager)
        btn_row.addWidget(self._open_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_face(self, face: Face) -> None:
        """Display the original image with the given face's bbox highlighted.

        Args:
            face: :class:`~app.db.models.Face` ORM object (with loaded image).
        """
        if face.image is None:
            self._image_label.setText("(no image reference)")
            return

        img_path = face.image.file_path
        self._current_image_path = img_path

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            self._image_label.setText(f"Cannot load:\n{img_path}")
            return

        # Draw a rectangle around the selected face
        x, y, w, h = face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (50, 200, 50), 3)

        # Draw all other faces from the same image in a muted colour
        for other_face in face.image.faces:
            if other_face.id == face.id:
                continue
            ox, oy, ow, oh = (
                other_face.bbox_x, other_face.bbox_y,
                other_face.bbox_w, other_face.bbox_h,
            )
            cv2.rectangle(img_bgr, (ox, oy), (ox + ow, oy + oh), (100, 100, 100), 1)

        pixmap = _bgr_to_qpixmap(img_bgr, _MAX_PREVIEW_W, _MAX_PREVIEW_H)
        self._image_label.setPixmap(pixmap)
        self._path_label.setText(img_path)
        self._open_btn.setEnabled(True)

    def clear(self) -> None:
        self._image_label.clear()
        self._image_label.setText("Click a face thumbnail to preview")
        self._path_label.setText("")
        self._open_btn.setEnabled(False)
        self._current_image_path = None

    # ------------------------------------------------------------------

    def _open_in_filemanager(self) -> None:
        if not self._current_image_path:
            return
        path = Path(self._current_image_path)
        if not path.exists():
            log.warning("File not found: %s", path)
            return

        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(path.parent)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(path)])
            elif sys.platform == "win32":
                subprocess.Popen(["explorer", "/select,", str(path)])
        except OSError as exc:
            log.warning("Cannot open file manager: %s", exc)
