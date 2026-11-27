"""Tree Studio: notice edits made in Face-Local (shared change log + setting)."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication

from app.ui.main_window.external_changes import auto_refresh_enabled
from app.ui.widgets.external_change_bar import ExternalChangeBar

log = logging.getLogger(__name__)


class StudioExternalChangesMixin:
    def _setup_external_changes(self, layout) -> None:  # noqa: ANN001
        self._ext_pending: dict[str, set[int]] = {}
        self._external_change_bar = ExternalChangeBar()
        self._external_change_bar.refresh_requested.connect(self._on_external_refresh)
        self._external_change_bar.dismissed.connect(self._ext_pending.clear)
        layout.insertWidget(0, self._external_change_bar)
        app = QGuiApplication.instance()
        if app is not None:
            app.applicationStateChanged.connect(self._on_app_state)

    def _retranslate_external(self) -> None:
        if hasattr(self, "_external_change_bar"):
            self._external_change_bar.retranslate()

    def _on_app_state(self, state) -> None:  # noqa: ANN001
        if state == Qt.ApplicationActive:
            self.check_external_changes()

    def check_external_changes(self) -> None:
        from app.services.change_watch_service import get_change_watch_service

        svc = get_change_watch_service()
        if svc is None:
            return
        try:
            changes = svc.changes_since_cursor()
        except Exception:  # noqa: BLE001
            log.exception("External change check failed")
            return
        if not changes:
            return
        for kind, ids in changes.by_kind.items():
            self._ext_pending.setdefault(kind, set()).update(ids)
        if auto_refresh_enabled():
            self._on_external_refresh()
        else:
            self._external_change_bar.show_changes(
                {k: len(v) for k, v in self._ext_pending.items()}
            )

    def _on_external_refresh(self) -> None:
        self._ext_pending.clear()
        self._external_change_bar.setVisible(False)
        self.reload_persons()
        self._refresh_root_from_code()
        self.rebuild(keep_scroll=True)
