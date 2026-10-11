"""System tray, notifications and update checks."""

from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QSystemTrayIcon,
)

from app.ui.dialogs.update_dialog import UpdateDialog
from app.ui.i18n import t

log = logging.getLogger(__name__)


class TrayUpdatesMixin:
    """System tray, notifications and update checks."""

    # ------------------------------------------------------------------
    # System tray
    # ------------------------------------------------------------------

    def _setup_tray(self) -> None:
        self._tray = QSystemTrayIcon(self)
        icon = self.windowIcon()
        if icon.isNull():
            icon = QIcon.fromTheme(
                "dialog-information",
                self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon),
            )
        self._tray.setIcon(icon)
        self._tray.setToolTip("Face-Local")
        if QSystemTrayIcon.isSystemTrayAvailable():
            self._tray.show()

    def _notify(self, title: str, message: str) -> None:
        """Send a system tray notification if enabled in settings."""
        from app.app_settings import app_qsettings
        enabled = app_qsettings().value("updates/notify", True, type=bool)
        if not enabled:
            return
        if QSystemTrayIcon.isSystemTrayAvailable() and QSystemTrayIcon.supportsMessages():
            self._tray.showMessage(title, message, QSystemTrayIcon.MessageIcon.Information, 6000)
        else:
            # Fallback: status bar
            self._status_label.setText(f"{title}: {message}")

    # ------------------------------------------------------------------
    # Update check
    # ------------------------------------------------------------------

    def _start_update_check(self) -> None:
        """Background update check at startup.

        On macOS, when SparkleHelper is bundled inside the .app, delegates to
        Sparkle for a native background check and returns early.  Sparkle will
        show its own update sheet if a new version is found; the Qt dialog is
        not used in that path.

        On all other platforms (and on macOS outside a frozen bundle, e.g. when
        running from source), falls back to the GitHub-API-based check.
        """
        import sys
        if sys.platform == "darwin":
            from app.updater import start_background_update_check
            if start_background_update_check():
                # SparkleHelper is running; it handles everything from here.
                return

        from app import __version__
        from app.services.update_service import fetch_latest_release, is_newer

        signal = self._update_ready

        class _CheckThread(QThread):
            def run(self_inner) -> None:  # noqa: N805
                release = fetch_latest_release()
                if release and is_newer(release.version, __version__):
                    signal.emit(release)

        self._check_thread = _CheckThread(self)
        self._check_thread.start()

    @Slot(object)
    def _on_update_found(self, release) -> None:
        self._pending_release = release
        self._update_notify_btn.setText(
            t("status_update_available", version=release.version)
        )
        self._update_notify_btn.setVisible(True)
        self._status_label.setText(t("update_status_found", version=release.version))
        self._notify(
            t("update_notification_title"),
            t("update_notification_msg", version=release.version),
        )

    @Slot()
    def _on_update_notify_clicked(self) -> None:
        if self._pending_release:
            dlg = UpdateDialog(self._pending_release, parent=self, auto_start=True)
            dlg.exec()
