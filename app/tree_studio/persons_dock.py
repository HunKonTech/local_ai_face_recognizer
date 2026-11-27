"""Tree Studio: searchable person list (drag source, on-chart marker)."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QMimeData, QSize, Qt
from PySide6.QtGui import QDrag, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDockWidget,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.ui.i18n import t
from app.ui.widgets.family_tree.interaction import PERSON_MIME
from app.utils.person_search import normalize

log = logging.getLogger(__name__)

_ROLE_PID = Qt.UserRole
_MAX_ROWS = 400  # list is filtered by search; keeps huge DBs responsive
_ICON = 32


class _PersonList(QListWidget):
    """List whose items drag as ``PERSON_MIME`` (the person id)."""

    def __init__(self) -> None:
        super().__init__()
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.setIconSize(QSize(_ICON, _ICON))
        self.setUniformItemSizes(True)

    def startDrag(self, _actions) -> None:  # noqa: N802
        item = self.currentItem()
        if item is None:
            return
        mime = QMimeData()
        mime.setData(PERSON_MIME, str(int(item.data(_ROLE_PID))).encode("ascii"))
        mime.setText(item.text())
        drag = QDrag(self)
        drag.setMimeData(mime)
        if not item.icon().isNull():
            drag.setPixmap(item.icon().pixmap(_ICON, _ICON))
        drag.exec(Qt.CopyAction)


class PersonsDockMixin:
    """Persons dock: name search, thumbnails, bold = on the chart."""

    def _build_persons_dock(self) -> None:
        self._persons: list[tuple[int, str, str, str]] = []  # id, name, norm, thumb
        self._icon_cache: dict[str, QIcon] = {}
        dock = QDockWidget(self)
        dock.setObjectName("ts_persons_dock")
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(6, 6, 6, 6)
        self._person_search = QLineEdit()
        self._person_search.setClearButtonEnabled(True)
        self._person_search.textChanged.connect(self._fill_person_list)
        layout.addWidget(self._person_search)
        self._person_list = _PersonList()
        self._person_list.itemDoubleClicked.connect(
            lambda item: self.edit_person(int(item.data(_ROLE_PID)))
        )
        layout.addWidget(self._person_list, stretch=1)
        self._person_hint = QLabel()
        self._person_hint.setWordWrap(True)
        self._person_hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._person_hint)
        dock.setWidget(body)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self._persons_dock = dock
        self.reload_persons()

    def _retranslate_persons_dock(self) -> None:
        self._persons_dock.setWindowTitle(t("ts_persons_dock"))
        self._person_search.setPlaceholderText(t("ts_search_placeholder"))
        self._person_hint.setText(t("ts_on_chart_tip"))

    def reload_persons(self) -> None:
        from sqlalchemy import text

        try:
            with session_scope() as session:
                rows = session.execute(text(
                    "SELECT p.id, p.name, COALESCE(p.thumbnail_path, ("
                    "  SELECT f.crop_path FROM faces f WHERE f.person_id = p.id"
                    "  AND f.crop_path IS NOT NULL ORDER BY f.confidence DESC LIMIT 1))"
                    " FROM persons p WHERE p.is_auto_named = 0 ORDER BY p.name"
                )).all()
        except Exception:  # noqa: BLE001
            log.exception("Could not load persons for Tree Studio")
            rows = []
        self._persons = [(pid, name, normalize(name or ""), thumb or "") for pid, name, thumb in rows]
        self._fill_person_list()

    def _fill_person_list(self) -> None:
        query = normalize(self._person_search.text() or "")
        self._person_list.clear()
        shown = 0
        for pid, name, norm, thumb in self._persons:
            if query and query not in norm:
                continue
            item = QListWidgetItem(name)
            item.setData(_ROLE_PID, pid)
            icon = self._person_icon(thumb)
            if icon is not None:
                item.setIcon(icon)
            self._person_list.addItem(item)
            shown += 1
            if shown >= _MAX_ROWS:
                break
        self._refresh_person_markers()

    def _person_icon(self, path: str):  # noqa: ANN202
        if not path:
            return None
        icon = self._icon_cache.get(path)
        if icon is None:
            if not Path(path).exists():
                return None
            pm = QPixmap(path)
            if pm.isNull():
                return None
            icon = QIcon(pm.scaled(_ICON, _ICON, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self._icon_cache[path] = icon
        return icon

    def _refresh_person_markers(self) -> None:
        """Bold = the person is drawn in the current view."""
        if not hasattr(self, "_person_list"):
            return
        visible = getattr(self, "visible", set())
        for i in range(self._person_list.count()):
            item = self._person_list.item(i)
            font = item.font()
            font.setBold(int(item.data(_ROLE_PID)) in visible)
            item.setFont(font)

    def listed_names(self) -> list[str]:
        return [self._person_list.item(i).text() for i in range(self._person_list.count())]
