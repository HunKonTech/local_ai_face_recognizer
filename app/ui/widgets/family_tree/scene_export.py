"""Export a :class:`FamilyTreeView` scene to PNG, JPG or SVG.

The whole scene is rendered (not just the viewport), every pending photo is
loaded first, the interactive selection highlight is removed, and collapse
badges are shown only when asked for (``family-tree-image-export`` spec).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QImage, QPainter

log = logging.getLogger(__name__)

FORMATS = ("png", "jpg", "svg")


def export_scene(
    view,
    path: str | Path,
    fmt: Optional[str] = None,
    *,
    scale: float = 1.0,
    quality: int = 90,
    show_markers: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Render *view*'s whole scene to *path*; returns the written path.

    ``fmt`` defaults to the file suffix. ``scale`` multiplies the pixel size of
    PNG/JPG output (SVG is vector, but its photos are embedded at box size).
    """
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt not in FORMATS:
        raise ValueError(f"unsupported export format: {fmt!r}")

    _load_all_photos(view, progress)

    selected = view._selected_id
    markers = view._show_markers
    try:
        if selected is not None:
            view.set_selected(None)
        view.set_markers_visible(show_markers)
        scene = view.scene()
        rect = scene.sceneRect()
        if fmt == "svg":
            _render_svg(scene, rect, path)
        else:
            _render_raster(scene, rect, path, fmt, scale, quality)
    finally:
        view.set_markers_visible(markers)
        if selected is not None:
            view.set_selected(selected)
    log.info("Family tree exported to %s (%s, scale %.1f)", path, fmt, scale)
    return path


def _load_all_photos(view, progress) -> None:
    total = len(view._pending_photos)
    done = 0
    while view._pending_photos:
        before = len(view._pending_photos)
        view._load_visible_photos(budget=50, everything=True)
        done += before - len(view._pending_photos)
        if progress is not None:
            progress(done, total)
        if len(view._pending_photos) >= before:  # defensive: no progress
            break
    view._photo_timer.stop()


def _render_raster(scene, rect: QRectF, path: Path, fmt: str, scale: float, quality: int) -> None:
    size = QSize(max(1, round(rect.width() * scale)), max(1, round(rect.height() * scale)))
    image = QImage(size, QImage.Format_ARGB32)
    image.fill(QColor("#ffffff") if fmt == "jpg" else Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    scene.render(painter, QRectF(0, 0, size.width(), size.height()), rect)
    painter.end()
    if fmt == "jpg":
        image = image.convertToFormat(QImage.Format_RGB32)
        ok = image.save(str(path), "JPG", int(quality))
    else:
        ok = image.save(str(path), "PNG")
    if not ok:
        raise OSError(f"could not write {path}")


def _render_svg(scene, rect: QRectF, path: Path) -> None:
    from PySide6.QtSvg import QSvgGenerator

    gen = QSvgGenerator()
    gen.setFileName(str(path))
    gen.setSize(QSize(round(rect.width()), round(rect.height())))
    gen.setViewBox(QRectF(0, 0, rect.width(), rect.height()))
    gen.setTitle("Family tree")
    painter = QPainter(gen)
    scene.render(painter, QRectF(0, 0, rect.width(), rect.height()), rect)
    painter.end()
