"""Detect family-data edits made by another app process on window activation.

See :mod:`app.services.change_watch_service`. When the application becomes
active, other processes' change-log entries since the last check are read;
the affected views refresh automatically (setting ``ui/auto_refresh_on_activate``)
or an :class:`ExternalChangeBar` offers a Refresh button.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QVBoxLayout, QWidget

from app.ui.widgets.external_change_bar import ExternalChangeBar

log = logging.getLogger(__name__)

AUTO_REFRESH_KEY = "ui/auto_refresh_on_activate"


def auto_refresh_enabled() -> bool:
    from app.app_settings import app_qsettings

    return bool(app_qsettings().value(AUTO_REFRESH_KEY, False, type=bool))


class ExternalChangesMixin:
    """Activation hook, notice bar and refresh routing for MainWindow."""

    def _setup_external_change_watch(self) -> None:
        """Wrap the central tabs with the notice bar and hook activation."""
        self._ext_pending: dict[str, set[int]] = {}
        self._external_change_bar = ExternalChangeBar()
        self._external_change_bar.refresh_requested.connect(self._on_external_refresh)
        self._external_change_bar.dismissed.connect(self._ext_pending.clear)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._external_change_bar)
        layout.addWidget(self._tabs, stretch=1)
        self.setCentralWidget(container)

        app = QGuiApplication.instance()
        if app is not None:
            app.applicationStateChanged.connect(self._on_application_state_changed)

    def _on_application_state_changed(self, state) -> None:
        if state == Qt.ApplicationActive:
            self._check_external_changes()

    def _check_external_changes(self) -> None:
        from app.services.change_watch_service import get_change_watch_service

        svc = get_change_watch_service()
        if svc is None:
            return
        try:
            changes = svc.changes_since_cursor()
        except Exception:  # noqa: BLE001 - never disturb activation
            log.exception("External change check failed")
            return
        if not changes:
            return
        for kind, ids in changes.by_kind.items():
            self._ext_pending.setdefault(kind, set()).update(ids)
        log.info("External changes detected: %s", changes.counts())
        if auto_refresh_enabled():
            self._on_external_refresh()
        else:
            self._external_change_bar.show_changes(
                {k: len(v) for k, v in self._ext_pending.items()}
            )

    def _on_external_refresh(self) -> None:
        kinds = set(self._ext_pending)
        self._ext_pending.clear()
        self._external_change_bar.setVisible(False)
        self._refresh_after_external_change(kinds)

    def _refresh_after_external_change(self, kinds: set[str]) -> None:
        """Reload the views affected by the given change kinds."""
        if kinds & {"person", "field"}:
            self._refresh_persons()
            for name in ("_persons_panel", "_family_tree_panel"):
                panel = getattr(self, name, None)
                if panel is not None:
                    panel.refresh()
        elif "relationship" in kinds or "view" in kinds:
            panel = getattr(self, "_family_tree_panel", None)
            if panel is not None:
                panel.refresh()
