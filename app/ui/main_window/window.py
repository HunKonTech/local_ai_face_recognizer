"""Primary application window — construction, settings and shutdown."""

from __future__ import annotations

import logging
from typing import List, Optional

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QMainWindow,
    QMessageBox,
)

from app.config import AppConfig
from app.db.database import ensure_unknown_person, init_db
from app.ui.i18n import t

from .deep_model import DeepModelMixin
from .export import ExportMixin
from .face_actions import FaceActionsMixin
from .folders import FoldersMixin
from .gdrive import GDriveMixin
from .maintenance import MaintenanceMixin
from .objects import ObjectsMixin
from .persons import PersonsMixin
from .pipeline import PipelineMixin
from .recording import RecordingMixin
from .tasks import TasksMixin
from .tray_updates import TrayUpdatesMixin
from .ui_build import UiBuildMixin

log = logging.getLogger(__name__)


class MainWindow(UiBuildMixin, FoldersMixin, PipelineMixin, MaintenanceMixin, GDriveMixin, RecordingMixin, ObjectsMixin, FaceActionsMixin, PersonsMixin, ExportMixin, DeepModelMixin, TrayUpdatesMixin, TasksMixin, QMainWindow):
    """Primary application window."""

    log_signal = Signal(str, int)
    _update_ready = Signal(object)   # ReleaseInfo
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        # The scan / AI pipeline runs as a Task Manager task; these hold the
        # live task handle and its worker (kept alive while it runs).
        self._active_pipeline_task = None
        self._object_search_task = None
        self._active_pipeline_worker = None
        self._current_person_id: Optional[int] = None
        self._current_face_id: Optional[int] = None
        self._recorder = None  # ScreenRecorderService, lazily created on start
        self._recording_log = None  # RecordingTimelineLog while recording
        self._recording_metadata = None  # RecordingMetadataWriter while recording
        self._last_audio_validation = None  # AudioValidation of the last stop
        self._last_video_validation = None  # VideoValidation of the last stop
        # Active image/person context shown in the image-browser tab (for the
        # recording timeline log).
        self._browser_image_name: Optional[str] = None
        self._browser_person_name: Optional[str] = None
        self._load_recording_prefs()
        self._connect_display_events()
        self._pending_auto_assignment_count: int = 0
        # New / changed images of the last scan, searched for tagged objects
        # once the pipeline finishes (#164).
        self._pending_object_scan_ids: List[int] = []
        self._db_path: str = str(config.db_path_resolved)
        self._pending_release = None

        # Google Drive project session — None when local mode is active
        from app.gdrive.project_session import GDriveProjectSession
        self._gdrive_session: Optional[GDriveProjectSession] = None
        self._gdrive_open_thread: Optional[QThread] = None
        self._gdrive_close_thread: Optional[QThread] = None
        self._gdrive_closing: bool = False  # True while shutdown in progress

        self._apply_window_icon()

        init_db(config.db_path_resolved)
        ensure_unknown_person()

        self._build_ui()
        self._connect_log_handler()
        self._refresh_persons()
        self._retranslate()
        self._restore_last_folder()
        self._setup_tray()
        self._image_browser.refresh()
        self._check_image_library_on_startup()
        self._setup_gdrive_chip()

        self.resize(1280, 780)
        self._update_ready.connect(self._on_update_found)
        self._start_update_check()
        self._setup_shortcuts()
        self._task_manager_dialog = None
        self._start_crop_repair_task()

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def _setup_shortcuts(self) -> None:
        """Register global shortcut handlers and bind them to this window.

        Shortcuts are dispatched via per-window QShortcut objects owned by this
        MainWindow (see ShortcutService).  We deliberately do NOT install an
        application-wide event filter: that crashes PySide6 once a
        QWebEngineView (the Places map) is alive.
        """
        from app.services.shortcut_service import get_shortcut_service
        svc = get_shortcut_service()
        svc.register("general.settings",  self._on_settings)
        svc.register("general.log_panel", self._toggle_log_panel)
        svc.set_host(self)
        # Scope shortcuts to the active page: switching tabs swaps which
        # page-specific shortcuts are live, so the same key can mean different
        # things on different pages. Global ("general") shortcuts stay active
        # everywhere.
        self._tabs.currentChanged.connect(self._on_tab_context_changed)
        self._on_tab_context_changed(self._tabs.currentIndex())

    # Maps each outer tab to the shortcut context (page) it activates. Tabs not
    # listed here fall back to "other" — only global shortcuts stay live there.
    _TAB_CONTEXT = {
        0: "faces",    # Arcfelismerés
        1: "image",    # Képböngésző
        4: "faces",    # Személyek
        # 6: Társaságok — no page-specific shortcuts ("other")
        7: "collage",  # Kollázs
    }

    def _on_tab_context_changed(self, index: int) -> None:
        from app.services.shortcut_service import get_shortcut_service
        context = self._TAB_CONTEXT.get(index, "other")
        get_shortcut_service().set_active_context(context)

        # Refresh the groups tab so groups created elsewhere (e.g. the person
        # dialog) show up without restarting.
        if (
            hasattr(self, "_groups_panel")
            and self._tabs.widget(index) is self._groups_panel
        ):
            self._groups_panel.reload()

    def _toggle_log_panel(self) -> None:
        if hasattr(self, "_log_dock"):
            if self._log_dock.isVisible():
                self._log_dock.hide()
            else:
                self._log_dock.show()

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        """Intercept close to shut down an open Drive session gracefully."""
        # Warn when background tasks are still running; cancel them on confirm.
        from app.tasks import get_task_manager
        manager = get_task_manager()
        if manager.has_active_tasks():
            n = manager.active_count
            reply = QMessageBox.question(
                self,
                t("tasks_title"),
                t("tasks_close_warning", n=n),
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            manager.cancel_all()
            # Give the worker threads a brief, bounded window to observe the
            # cancel and exit cleanly, so we don't tear down QThreads that are
            # still running (which logs "QThread destroyed while running" and
            # could interrupt a mid-flight DB write).  The UI stays responsive
            # via processEvents; we never block longer than the grace period.
            manager.wait_for_idle(timeout_s=5.0)
        # Finalize any in-progress recording so segments + timeline are saved.
        if self._recorder is not None:
            from app.services.screen_recorder_service import RecorderState
            if self._recorder.state in (
                RecorderState.RECORDING,
                RecorderState.PAUSED,
            ):
                self._on_record_stop()
        if self._gdrive_session is not None and not self._gdrive_closing:
            # Don't actually close yet — let the Drive thread finish, then quit.
            event.ignore()
            self._gdrive_closing = True
            self._gdrive_chip_btn.setText(t("gdrive_chip_closing"))
            self._status_label.setText(t("gdrive_closing_wait"))
            self._start_drive_close(after_quit=True)
            return
        event.accept()
