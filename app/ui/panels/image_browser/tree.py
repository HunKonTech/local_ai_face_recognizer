"""Folder/image tree population, selection and context menu."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import (
    QPoint,
    Qt,
    QThreadPool,
)
from PySide6.QtGui import (
    QColor,
    QIcon,
    QPixmap,
)
from PySide6.QtWidgets import (
    QMenu,
    QTreeWidgetItem,
)

from app.db.database import session_scope
from app.services.image_browser_service import (
    ImageBrowserService,
)
from app.ui.i18n import t
from app.workers.thumbnail_worker import ThumbnailRunnable

from .helpers import (
    _ROLE_CACHE,
    _ROLE_FILE_PATH,
    _ROLE_PAYLOAD,
    _ROLE_TREE_NODE,
    _ROLE_TYPE,
    _TREE_THUMB_SIZE,
    _TreeNode,
)

log = logging.getLogger(__name__)


class TreeMixin:
    """Folder/image tree population, selection and context menu."""

    # ──────────────────────────────────────────────────────────────────
    # Tree construction helpers
    # ──────────────────────────────────────────────────────────────────

    def _make_tree_node_item(self, node: _TreeNode) -> QTreeWidgetItem:
        """Create a QTreeWidgetItem for a hierarchy node (data folder or intermediate).

        Children are loaded lazily on first expand — a dummy child is added for
        any node that has either subfolders or images.
        """
        if node.has_images:
            summary = node.summary
            name_color = "#fab387" if summary.unknown_faces_count > 0 else "#cdd6f4"

            # In Drive mode show "☁ <project name>" instead of the mirror path.
            display = node.name
            if (
                self._drive_client is not None
                and self._drive_mirror_dir is not None
                and node.path == str(self._drive_mirror_dir)
            ):
                display = f"☁ {self._drive_project_name}" if self._drive_project_name else "☁ Drive"

            label = f"{display}  ({summary.total_images})"
            item = QTreeWidgetItem([label])
            item.setData(0, _ROLE_TYPE, "folder")
            item.setData(0, _ROLE_PAYLOAD, node.path)
            item.setForeground(0, QColor(name_color))

            tip_lines = [
                f"Összes: {summary.total_images}",
                f"Feldolgozva: {summary.processed_images}",
            ]
            if summary.detected_faces_count > 0:
                tip_lines += [
                    f"Arcok: {summary.detected_faces_count}",
                    f"Ismeretlen: {summary.unknown_faces_count}",
                ]
            item.setToolTip(0, "\n".join(tip_lines))
        else:
            # Intermediate structural node (no images directly in this directory)
            item = QTreeWidgetItem([node.name])
            item.setData(0, _ROLE_TYPE, "folder_node")
            item.setData(0, _ROLE_PAYLOAD, node.path)
            item.setForeground(0, QColor("#89b4fa"))

        # Store the tree node so _on_tree_item_expanded can access children
        item.setData(0, _ROLE_TREE_NODE, node)

        # Add a dummy child for any node that can be expanded (has subfolders or images)
        if node.has_children or node.has_images:
            dummy = QTreeWidgetItem(["…"])
            dummy.setData(0, _ROLE_TYPE, "dummy")
            dummy.setFlags(Qt.NoItemFlags)
            item.addChild(dummy)

        return item

    # ── Tree search helpers ───────────────────────────────────────────

    def _find_folder_item_by_path(self, folder_path: str) -> Optional[QTreeWidgetItem]:
        """DFS search for the 'folder' item whose payload matches *folder_path*.

        Intermediate 'folder_node' items are expanded on-the-fly as needed so
        that their lazy-loaded subfolder children become available to search.
        Only paths that are proper ancestors of *folder_path* are expanded —
        unrelated branches are never touched.
        """
        target = Path(folder_path)

        def _could_contain(item_path: str) -> bool:
            """True if *folder_path* equals or is a descendant of *item_path*."""
            try:
                target.relative_to(item_path)
                return True
            except ValueError:
                return False

        def _search(item: QTreeWidgetItem) -> Optional[QTreeWidgetItem]:
            item_type = item.data(0, _ROLE_TYPE)
            item_payload = item.data(0, _ROLE_PAYLOAD) or ""

            if item_type == "folder" and item_payload == folder_path:
                return item

            if item_type in ("folder", "folder_node") and _could_contain(item_payload):
                # Expand the item (lazy-load its children) if not already done
                if not item.isExpanded():
                    item.setExpanded(True)
                for i in range(item.childCount()):
                    result = _search(item.child(i))
                    if result is not None:
                        return result

            return None

        for i in range(self._file_tree.topLevelItemCount()):
            result = _search(self._file_tree.topLevelItem(i))
            if result is not None:
                return result
        return None

    def _expand_ancestors(self, item: QTreeWidgetItem) -> None:
        """Expand all ancestor items so *item* is visible in the tree."""
        parent = item.parent()
        while parent is not None:
            parent.setExpanded(True)
            parent = parent.parent()

    def _iter_data_folders(self):
        """Yield all 'folder' type items in DFS (display) order."""
        def _recurse(item: QTreeWidgetItem):
            if item.data(0, _ROLE_TYPE) == "folder":
                yield item
            for i in range(item.childCount()):
                yield from _recurse(item.child(i))

        for i in range(self._file_tree.topLevelItemCount()):
            yield from _recurse(self._file_tree.topLevelItem(i))

    def _make_placeholder_icon(self) -> QIcon:
        px = QPixmap(_TREE_THUMB_SIZE, _TREE_THUMB_SIZE)
        px.fill(QColor("#11111b"))
        return QIcon(px)

    # ──────────────────────────────────────────────────────────────────
    # Tree signals
    # ──────────────────────────────────────────────────────────────────

    def _on_tree_item_expanded(self, item: QTreeWidgetItem) -> None:
        item_type = item.data(0, _ROLE_TYPE)
        if item_type not in ("folder", "folder_node"):
            return
        # Only act while a dummy placeholder is still present
        has_dummy = any(
            item.child(i).data(0, _ROLE_TYPE) == "dummy"
            for i in range(item.childCount())
        )
        if not has_dummy:
            return

        node: Optional[_TreeNode] = item.data(0, _ROLE_TREE_NODE)

        # Remove the dummy
        for i in range(item.childCount() - 1, -1, -1):
            if item.child(i).data(0, _ROLE_TYPE) == "dummy":
                item.removeChild(item.child(i))

        # Add subfolder/folder_node children first (sorted by name)
        if node is not None:
            for child_name in sorted(node.children.keys()):
                item.addChild(self._make_tree_node_item(node.children[child_name]))

        # For data folders, also lazy-load the images inside that folder
        if item_type == "folder":
            self._load_folder_children(item)

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        if item.data(0, _ROLE_TYPE) != "image":
            return
        self._select_tree_item(item)

    # ──────────────────────────────────────────────────────────────────
    # Re-recognition (context menu → background worker)
    # ──────────────────────────────────────────────────────────────────

    def _selected_image_ids(self, fallback_pos: Optional[QPoint] = None) -> List[int]:
        """Collect image ids from the tree selection (or the item under cursor)."""
        ids: List[int] = []
        for item in self._file_tree.selectedItems():
            if item.data(0, _ROLE_TYPE) == "image":
                pid = item.data(0, _ROLE_PAYLOAD)
                if isinstance(pid, int):
                    ids.append(pid)
        if not ids and fallback_pos is not None:
            item = self._file_tree.itemAt(fallback_pos)
            if item is not None and item.data(0, _ROLE_TYPE) == "image":
                pid = item.data(0, _ROLE_PAYLOAD)
                if isinstance(pid, int):
                    ids.append(pid)
        # Preserve order, drop duplicates.
        seen: set = set()
        return [i for i in ids if not (i in seen or seen.add(i))]

    def _on_tree_context_menu(self, pos: QPoint) -> None:
        image_ids = self._selected_image_ids(fallback_pos=pos)
        menu = QMenu(self)

        if image_ids:
            n = len(image_ids)
            rerec_menu = menu.addMenu(
                t("rerec_ctx_one") if n == 1 else t("rerec_ctx_many", n=n)
            )
            classic_action = rerec_menu.addAction(t("rerec_ctx_engine_classic"))
            classic_action.triggered.connect(
                lambda: self._start_rerecognition(image_ids, engine="classic")
            )
            deep_action = rerec_menu.addAction(t("rerec_ctx_engine_deep"))
            deep_action.triggered.connect(
                lambda: self._start_rerecognition(image_ids, engine="deep")
            )
            menu.addSeparator()

            export_faces = menu.addAction(
                t("ibp_ctx_export_faces_one") if n == 1
                else t("ibp_ctx_export_faces_many", n=n)
            )
            export_faces.triggered.connect(
                lambda: self._export_faces_as_images(image_ids)
            )
            menu.addSeparator()

        undo_action = menu.addAction(t("rerec_ctx_undo_last"))
        undo_action.setEnabled(self._latest_undoable_batch() is not None)
        undo_action.triggered.connect(self._undo_last_rerecognition)

        history_action = menu.addAction(t("rerec_ctx_history"))
        history_action.triggered.connect(self._open_rerecognition_history)

        menu.exec(self._file_tree.viewport().mapToGlobal(pos))

    def _load_folder_children(self, folder_item: QTreeWidgetItem) -> None:
        """Append image children to a data folder; subfolder children remain untouched.

        Called after the dummy placeholder has already been removed by
        _on_tree_item_expanded (or by the restore path).  Safe to call
        multiple times: stale image/placeholder children are removed first.
        """
        folder_path: str = folder_item.data(0, _ROLE_PAYLOAD)

        # Remove any stale image/placeholder children (dummy was already removed
        # by _on_tree_item_expanded; subfolder nodes are always preserved)
        to_remove = [
            folder_item.child(i)
            for i in range(folder_item.childCount())
            if folder_item.child(i).data(0, _ROLE_TYPE) in ("dummy", "image", "placeholder")
        ]
        for child in to_remove:
            folder_item.removeChild(child)

        # Clear stale tree_items entries that belonged to this folder
        # (identified by checking which image items are still parented here)
        self._tree_items = {
            k: v for k, v in self._tree_items.items()
            if v.parent() is not folder_item
        }

        with session_scope() as session:
            summaries = ImageBrowserService(session).list_images(
                folder_path,
                place_id=self._place_filter_id,
            )

        if not summaries:
            empty = QTreeWidgetItem([t("ib3_no_images_in_folder")])
            empty.setData(0, _ROLE_TYPE, "placeholder")
            empty.setFlags(Qt.NoItemFlags)
            folder_item.addChild(empty)
            return

        pool = QThreadPool.globalInstance()
        placeholder_icon = self._make_placeholder_icon()

        for summary in summaries:
            child = QTreeWidgetItem([summary.filename])
            child.setData(0, _ROLE_TYPE, "image")
            child.setData(0, _ROLE_PAYLOAD, summary.image_id)
            child.setData(0, _ROLE_CACHE, summary.thumbnail_cache_key)
            child.setData(0, _ROLE_FILE_PATH, summary.file_path)
            child.setIcon(0, placeholder_icon)

            # Face-count tooltip
            if summary.face_count > 0:
                tip = f"{summary.face_count} arc"
                if summary.unknown_face_count:
                    tip += f"  ({summary.unknown_face_count} ismeretlen)"
                child.setToolTip(0, tip)

            folder_item.addChild(child)
            self._tree_items[summary.thumbnail_cache_key] = child

            # Use cached thumbnail if available
            if summary.thumbnail_cache_key in self._thumb_cache:
                pixmap = self._thumb_cache[summary.thumbnail_cache_key]
                child.setIcon(0, QIcon(pixmap))
            elif Path(summary.file_path).exists():
                worker = ThumbnailRunnable(
                    file_path=summary.file_path,
                    cache_key=summary.thumbnail_cache_key,
                    size=_TREE_THUMB_SIZE,
                )
                worker.signals.ready.connect(self._on_thumbnail_ready)
                worker.signals.failed.connect(self._on_thumbnail_failed)
                pool.start(worker)
            elif self._drive_client is not None:
                # Mirror file missing — download from Drive and generate thumb.
                from app.workers.drive_image_worker import DriveThumbRunnable
                dw = DriveThumbRunnable(
                    drive_client=self._drive_client,
                    image_id=summary.image_id,
                    local_path=summary.file_path,
                    cache_key=summary.thumbnail_cache_key,
                    size=_TREE_THUMB_SIZE,
                )
                dw.signals.ready.connect(self._on_thumbnail_ready)
                dw.signals.failed.connect(self._on_thumbnail_failed)
                pool.start(dw)
            else:
                log.warning("Missing file for thumbnail: %s", summary.file_path)

        log.info(
            "Loaded %d image items for folder %s", len(summaries), folder_path
        )

    # ──────────────────────────────────────────────────────────────────
    # Keyboard navigation
    # ──────────────────────────────────────────────────────────────────

    def _on_tree_current_item_changed(self, current: QTreeWidgetItem, _previous: QTreeWidgetItem) -> None:
        """Handle keyboard Up/Down navigation in the image tree.

        Fires for every current-item change (click, keyboard, programmatic).
        Programmatic calls from _select_tree_item block signals so they never
        reach here; the refresh() restore path is guarded by the image-id check.
        """
        if current is None or current.data(0, _ROLE_TYPE) != "image":
            return
        image_id = current.data(0, _ROLE_PAYLOAD)
        if not isinstance(image_id, int):
            return
        if image_id == self._current_image_id:
            return  # already showing this image (e.g. tree refresh restoring selection)
        self._select_tree_item(current)

    def _select_tree_item(self, item: QTreeWidgetItem) -> None:
        """Select item visually in the tree and load the image."""
        self._current_tree_item = item
        # Block signals so setCurrentItem does not re-trigger _on_tree_current_item_changed.
        self._file_tree.blockSignals(True)
        self._file_tree.setCurrentItem(item)
        self._file_tree.blockSignals(False)
        self._file_tree.scrollToItem(item)
        image_id = item.data(0, _ROLE_PAYLOAD)
        self._load_image(image_id)
        self._update_nav_buttons()
        # Smart prefetch: queue downloads for adjacent images in Drive mode.
        if self._drive_client is not None:
            self._prefetch_adjacent(item)

    def _update_nav_buttons(self) -> None:
        """Grey out ◀ / ▶ when at the very first / last image across all folders."""
        current = self._current_tree_item
        if current is None:
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
            return

        parent = current.parent()
        if parent is None:
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
            return

        idx = parent.indexOfChild(current)
        data_folders = list(self._iter_data_folders())
        parent_idx = next((i for i, f in enumerate(data_folders) if f is parent), -1)

        has_prev = (idx > 0) or (parent_idx > 0)
        has_next = (idx < parent.childCount() - 1) or (
            0 <= parent_idx < len(data_folders) - 1
        )

        self._prev_btn.setEnabled(has_prev)
        self._next_btn.setEnabled(has_next)

    def _open_first_image(self) -> None:
        """Expand the path to the first data folder and select its first image.

        Walks down the first branch of the tree, expanding folder_node items
        lazily until the first 'folder' (data) item is found and opened.
        """
        item: Optional[QTreeWidgetItem] = (
            self._file_tree.topLevelItem(0)
            if self._file_tree.topLevelItemCount() > 0
            else None
        )
        while item is not None:
            item_type = item.data(0, _ROLE_TYPE)
            if item_type == "folder":
                item.setExpanded(True)  # fires _on_tree_item_expanded → loads images
                for i in range(item.childCount()):
                    if item.child(i).data(0, _ROLE_TYPE) == "image":
                        self._select_tree_item(item.child(i))
                        return
                return
            elif item_type == "folder_node":
                item.setExpanded(True)  # fires lazy loading of subfolder items
                # Find the first non-dummy child after expansion
                next_item = None
                for i in range(item.childCount()):
                    child = item.child(i)
                    if child.data(0, _ROLE_TYPE) in ("folder", "folder_node"):
                        next_item = child
                        break
                item = next_item
            else:
                break
