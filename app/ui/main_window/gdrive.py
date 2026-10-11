"""Google Drive project chip, open/close and preferences."""

from __future__ import annotations

import logging

from PySide6.QtCore import QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QMessageBox,
)

from app.config import save_db_path
from app.db.database import ensure_unknown_person, init_db
from app.ui.dialogs.settings_dialog import SettingsDialog
from app.ui.i18n import t

log = logging.getLogger(__name__)


class GDriveMixin:
    """Google Drive project chip, open/close and preferences."""

    # ------------------------------------------------------------------
    # Google Drive workflow
    # ------------------------------------------------------------------

    def _setup_gdrive_chip(self) -> None:
        """Initialise the Drive chip visibility and start the refresh timer."""
        self._update_gdrive_toolbar_btn()
        self._update_gdrive_chip()

        # Poll session status every 15 seconds to keep the chip fresh.
        self._gdrive_status_timer = QTimer(self)
        self._gdrive_status_timer.setInterval(15_000)
        self._gdrive_status_timer.timeout.connect(self._update_gdrive_chip)
        self._gdrive_status_timer.start()

        # Auto-open Drive session on startup if mode is enabled and ready.
        from app.gdrive import preferences as _gprefs
        _p = _gprefs.load()
        if _p.enabled and _p.is_ready and self._gdrive_session is None:
            self._status_label.setText(t("gdrive_mode_enabled_opening"))
            QTimer.singleShot(400, self._on_open_drive_project)

    def _update_gdrive_toolbar_btn(self) -> None:
        """Show/hide and label the Drive action in the System menu."""
        from app.gdrive import preferences
        prefs = preferences.load()
        is_open = self._gdrive_session is not None

        if is_open:
            self._gdrive_action.setText(t("gdrive_close_project_btn"))
            self._gdrive_action.setToolTip(t("gdrive_close_project_tip"))
            self._gdrive_action.setVisible(True)
        elif prefs.is_ready:
            self._gdrive_action.setText(t("gdrive_open_project_btn"))
            self._gdrive_action.setToolTip(t("gdrive_open_project_tip"))
            self._gdrive_action.setVisible(True)
        else:
            self._gdrive_action.setVisible(False)

    def _update_gdrive_chip(self) -> None:
        """Update the status bar Drive chip from the current session state."""
        from app.gdrive import preferences
        prefs = preferences.load()

        if self._gdrive_session is None:
            if not prefs.is_ready:
                self._gdrive_chip_btn.setVisible(False)
                return
            self._gdrive_chip_btn.setText(t("gdrive_chip_idle"))
            self._gdrive_chip_btn.setToolTip(t("gdrive_open_project_tip"))
            self._gdrive_chip_btn.setVisible(True)
            return

        # Session is open — show sync status.
        status = self._gdrive_session.status
        if status.current_op:
            label = t("gdrive_chip_syncing")
        elif not status.last_sync_succeeded:
            label = t("gdrive_chip_error")
        else:
            folders = self._gdrive_session.folders
            name = folders.root_name if folders else ""
            label = t("gdrive_chip_open", name=name) if name else t("gdrive_chip_open", name="Drive")
        self._gdrive_chip_btn.setText(label)
        self._gdrive_chip_btn.setToolTip(t("gdrive_close_project_tip"))
        self._gdrive_chip_btn.setVisible(True)

    @Slot()
    def _on_gdrive_chip_clicked(self) -> None:
        """Clicking the chip opens the project if closed, closes if open."""
        self._on_toggle_drive_project()

    @Slot()
    def _on_toggle_drive_project(self) -> None:
        """Open or close the Drive project depending on current state."""
        if self._gdrive_session is not None:
            self._on_close_drive_project()
        else:
            self._on_open_drive_project()

    @Slot()
    def _on_open_drive_project(self) -> None:
        """Start opening the Drive project session in a background thread."""
        from app.gdrive import preferences
        from app.gdrive.connectivity import is_online

        prefs = preferences.load()
        if not prefs.is_ready:
            QMessageBox.information(
                self, t("gdrive_not_configured_title"), t("gdrive_not_configured_msg")
            )
            return

        if not is_online():
            QMessageBox.warning(self, t("gdrive_error_title"), t("gdrive_offline_error"))
            return

        if self._gdrive_open_thread and self._gdrive_open_thread.isRunning():
            return  # already opening

        self._gdrive_action.setEnabled(False)
        self._gdrive_chip_btn.setText(t("gdrive_chip_opening"))
        self._gdrive_chip_btn.setVisible(True)

        account_email = prefs.account_email
        folder_id = prefs.folder_id

        class _OpenThread(QThread):
            succeeded = Signal(object, str)   # (session, local_db_path)
            failed = Signal(str)

            def run(self_inner) -> None:  # noqa: N805
                try:
                    from app.gdrive.drive_client import build_drive_client
                    from app.gdrive.project_session import GDriveProjectSession
                    client = build_drive_client(account_email)
                    session = GDriveProjectSession(client, folder_id)
                    local_db = session.open()
                    self_inner.succeeded.emit(session, str(local_db))
                except Exception as exc:  # noqa: BLE001
                    log.exception("Drive project open failed")
                    self_inner.failed.emit(str(exc))

        thread = _OpenThread(self)
        thread.succeeded.connect(self._on_drive_open_succeeded)
        thread.failed.connect(self._on_drive_open_failed)
        thread.finished.connect(lambda: self._gdrive_action.setEnabled(True))
        self._gdrive_open_thread = thread
        thread.start()

    @Slot(object, str)
    def _on_drive_open_succeeded(self, session, local_db_path: str) -> None:
        """Called on the main thread after Drive session opens successfully."""
        self._gdrive_session = session

        # Switch the database to the Drive-downloaded local copy.
        init_db(local_db_path)
        ensure_unknown_person()
        self._db_path = local_db_path
        self._current_person_id = None
        self._current_face_id = None
        self._cluster_panel.clear()
        self._preview_panel.clear()

        folders = session.folders
        name = folders.root_name if folders else "Drive"

        # Tell the image browser to use Drive-mode lazy loading.
        from app.paths import drive_mirror_dir
        root_id = folders.root_id if folders else ""
        self._image_browser.set_drive_mode(
            client=session._client,
            mirror_dir=drive_mirror_dir(root_id),
            project_name=name,
        )

        self._refresh_persons()
        self._image_browser.refresh()

        log.info("Drive project opened: %s (local DB: %s)", name, local_db_path)
        self._status_label.setText(t("gdrive_project_opened", name=name))
        self._update_gdrive_toolbar_btn()
        self._update_gdrive_chip()

    @Slot(str)
    def _on_drive_open_failed(self, error: str) -> None:
        log.error("Drive project open failed: %s", error)
        QMessageBox.critical(
            self, t("gdrive_error_title"), t("gdrive_open_failed", error=error)
        )
        self._gdrive_chip_btn.setText(t("gdrive_chip_idle"))
        self._update_gdrive_toolbar_btn()
        self._update_gdrive_chip()

    @Slot()
    def _on_close_drive_project(self) -> None:
        """Ask for confirmation and then close the Drive session."""
        if self._gdrive_session is None:
            return

        reply = QMessageBox.question(
            self,
            t("gdrive_confirm_close_title"),
            t("gdrive_confirm_close_msg"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._start_drive_close(after_quit=False)

    def _start_drive_close(self, *, after_quit: bool = False) -> None:
        """Kick off Drive session shutdown in a background thread.

        Args:
            after_quit: When ``True``, call ``QApplication.quit()`` after the
                        thread finishes so the app window actually closes.
        """
        if self._gdrive_session is None:
            if after_quit:
                QApplication.quit()
            return

        session = self._gdrive_session
        self._gdrive_session = None  # mark as closing
        self._gdrive_chip_btn.setText(t("gdrive_chip_closing"))
        self._gdrive_action.setEnabled(False)

        class _CloseThread(QThread):
            done = Signal()

            def run(self_inner) -> None:  # noqa: N805
                try:
                    session.close(upload_pending=True)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Drive session close error: %s", exc)
                self_inner.done.emit()

        thread = _CloseThread(self)
        if after_quit:
            thread.done.connect(QApplication.quit)
        else:
            thread.done.connect(self._on_drive_close_done)
        self._gdrive_close_thread = thread
        thread.start()

    @Slot()
    def _on_drive_close_done(self) -> None:
        """Called after the Drive session has been fully closed."""
        log.info("Drive session closed.")
        self._gdrive_closing = False

        # Switch back to the local database and refresh all views.
        local_db = str(self._config.db_path_resolved)
        self._db_path = local_db
        init_db(local_db)
        ensure_unknown_person()
        self._current_person_id = None
        self._current_face_id = None
        self._cluster_panel.clear()
        self._preview_panel.clear()
        self._image_browser.clear_drive_mode()
        self._refresh_persons()
        self._image_browser.refresh()

        self._update_gdrive_toolbar_btn()
        self._update_gdrive_chip()
        self._gdrive_action.setEnabled(True)
        self._status_label.setText(t("ready"))

    @Slot()
    def _on_drive_prefs_changed(self) -> None:
        """Refresh Drive UI and auto-open/close session when mode toggle changes."""
        from app.gdrive import preferences as _gprefs
        prefs = _gprefs.load()

        if prefs.enabled and prefs.is_ready and self._gdrive_session is None:
            # User just turned Drive ON — open the session automatically.
            self._status_label.setText(t("gdrive_mode_enabled_opening"))
            self._on_open_drive_project()
        elif not prefs.enabled and self._gdrive_session is not None:
            # User just turned Drive OFF — close the session gracefully.
            self._start_drive_close(after_quit=False)

        self._update_gdrive_toolbar_btn()
        self._update_gdrive_chip()

    # ------------------------------------------------------------------
    # Settings slot
    # ------------------------------------------------------------------

    @Slot()
    def _on_settings(self, initial_tab=None) -> None:
        # QAction.triggered hands us a bool; only a real tab name counts.
        if not isinstance(initial_tab, str):
            initial_tab = None
        dlg = SettingsDialog(
            current_db_path=self._db_path,
            parent=self,
            app_config=self._config,
            initial_tab=initial_tab,
        )
        # Wire up the Drive prefs-changed signal so the chip/button refresh.
        if hasattr(dlg, "_gdrive_tab"):
            dlg._gdrive_tab.prefs_changed.connect(self._on_drive_prefs_changed)
        if dlg.exec() != SettingsDialog.Accepted:
            return

        # Neural graph deferred open — triggered by the Debug-tab button.
        # The modal exec() above blocks any new window; open it now that it's closed.
        if getattr(self, "_pending_open_nn_graph", False):
            self._pending_open_nn_graph = False
            self._open_nn_graph_window()

        # Database change
        new_db = dlg.selected_db_path()
        if new_db and new_db != self._db_path:
            self._db_path = new_db
            self._config.storage.db_path = new_db
            save_db_path(new_db)
            init_db(new_db)  # also re-initialises the ImageLibraryService
            ensure_unknown_person()
            self._check_image_library_on_startup()
            self._current_person_id = None
            self._current_face_id = None
            self._cluster_panel.clear()
            self._preview_panel.clear()
            self._refresh_persons()
            self._image_browser.refresh()
            self._locations_panel.refresh()
            self._persons_panel.refresh()
            self._objects_panel.refresh()
            QMessageBox.information(self, t("settings_title"), t("db_switched"))
            log.info("Database switched to: %s", new_db)

        # Language change
        if dlg.language_changed():
            self._retranslate()
            self._refresh_persons()

        # A changed pairing setting only takes effect when the image is re-read.
        self._image_browser.reload_current_image()

        # Pick up any recording-setting changes for the next recording.
        self._load_recording_prefs()
