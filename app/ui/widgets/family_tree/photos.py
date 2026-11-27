"""Lazy photo loading for :class:`FamilyTreeView`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem

from app.ui.widgets.family_tree.style import _ROLE_PID


class _PhotosMixin:
    """Loads box photos only when they scroll into (or near) the viewport."""

    def _schedule_photo_load(self) -> None:
        if self._pending_photos:
            self._photo_timer.start()

    # Below this many photos, load them all at once (small/medium trees);
    # above it, load only what is in/near the viewport so huge trees stay fast.
    EAGER_PHOTO_LIMIT = 250

    def _load_visible_photos(self, *, budget: int = 120, everything: bool = False) -> None:
        """Load pending photos — all of them when small, else viewport-near.

        ``everything`` (image export) takes every pending photo regardless of
        the viewport, still in ``budget``-sized steps.
        """
        if not self._pending_photos:
            return

        if everything:
            targets = list(self._pending_photos.keys())
        elif len(self._pending_photos) <= self.EAGER_PHOTO_LIMIT:
            targets = list(self._pending_photos.keys())
            budget = len(targets)  # no partial pass for small trees
        else:
            # Pad the viewport so photos are ready slightly before they scroll in.
            view_rect = self.viewport().rect().adjusted(-200, -200, 200, 200)
            scene_rect = self.mapToScene(view_rect).boundingRect()
            targets = [
                int(pid)
                for item in self._scene.items(scene_rect)
                if (pid := item.data(_ROLE_PID)) is not None
                and int(pid) in self._pending_photos
            ]

        loaded = 0
        for pid in targets:
            pending = self._pending_photos.pop(pid, None)
            if pending is None:
                continue
            group, pos, path = pending
            pixmap = self._load_photo(path)
            if pixmap is not None:
                pix = QGraphicsPixmapItem(pixmap)
                # addToGroup first so setPos is in group-local coords, not scene coords.
                group.addToGroup(pix)
                pix.setPos(pos[0], pos[1] + (self.PHOTO - pixmap.height()) / 2)
            loaded += 1
            if loaded >= budget:
                if self._pending_photos:
                    self._photo_timer.start()  # finish remaining next tick
                break

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._schedule_photo_load()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._schedule_photo_load()

    def _load_photo(self, path: Optional[str]) -> Optional[QPixmap]:
        if not path or not Path(path).exists():
            return None
        from app.utils.image_utils import load_pixmap_exif

        pixmap = load_pixmap_exif(path)
        if pixmap.isNull():
            return None
        return pixmap.scaled(
            self.PHOTO, self.PHOTO, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
