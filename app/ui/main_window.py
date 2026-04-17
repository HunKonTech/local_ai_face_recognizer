"""Main application window."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QLabel,
    QHBoxLayout,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, Person
from app.logging_setup import QLogHandler
from app.services.clustering_service import ClusteringService
from app.services.export_service import ExportService
from app.services.identity_service import IdentityService
from app.ui.dialogs.merge_dialog import MergeDialog
from app.ui.dialogs.rename_dialog import RenameDialog
from app.ui.dialogs.settings_dialog import SettingsDialog
from app.ui.dialogs.tpu_status_dialog import TpuStatusDialog
from app.ui.i18n import t
from app.ui.panels.cluster_panel import ClusterPanel
from app.ui.panels.log_panel import LogPanel
from app.ui.panels.preview_panel import PreviewPanel
from app.ui.panels.sidebar_panel import SidebarPanel
from app.workers.pipeline_worker import PipelineWorker

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Primary application window."""

    log_signal = Signal(str, int)

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._worker: Optional[PipelineWorker] = None
        self._current_person_id: Optional[int] = None
        self._current_face_id: Optional[int] = None
        self._db_path: str = str(config.db_path_resolved)

        init_db(config.db_path_resolved)

        self._build_ui()
        self._connect_log_handler()
        self._refresh_persons()
        self._retranslate()

        self.resize(1280, 780)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_toolbar()
        self._build_central()
        self._build_log_dock()
        self._build_status_bar()

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        self._folder_btn = QPushButton()
        self._folder_btn.clicked.connect(self._on_select_folder)
        tb.addWidget(self._folder_btn)

        self._folder_label = QLabel()
        self._folder_label.setStyleSheet("color: #888;")
        tb.addWidget(self._folder_label)

        tb.addSeparator()

        self._scan_btn = QPushButton()
        self._scan_btn.setEnabled(False)
        self._scan_btn.clicked.connect(self._on_scan)
        tb.addWidget(self._scan_btn)

        self._stop_btn = QPushButton()
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        tb.addWidget(self._stop_btn)

        tb.addSeparator()

        self._export_csv_btn = QPushButton()
        self._export_csv_btn.clicked.connect(self._on_export_csv)
        tb.addWidget(self._export_csv_btn)

        self._export_images_btn = QPushButton()
        self._export_images_btn.clicked.connect(self._on_export_images)
        tb.addWidget(self._export_images_btn)

        tb.addSeparator()

        self._tpu_btn = QPushButton()
        self._tpu_btn.clicked.connect(self._on_tpu_status)
        tb.addWidget(self._tpu_btn)

        self._settings_btn = QPushButton()
        self._settings_btn.clicked.connect(self._on_settings)
        tb.addWidget(self._settings_btn)

    def _build_central(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        self._sidebar = SidebarPanel()
        self._sidebar.person_selected.connect(self._on_person_selected)
        self._sidebar.set_recluster_callback(self._on_recluster)
        self._sidebar.setMinimumWidth(200)
        self._sidebar.setMaximumWidth(280)
        splitter.addWidget(self._sidebar)

        centre = QWidget()
        centre_layout = QVBoxLayout(centre)
        centre_layout.setContentsMargins(0, 0, 0, 0)

        self._cluster_panel = ClusterPanel()
        self._cluster_panel.face_selected.connect(self._on_face_selected)
        centre_layout.addWidget(self._cluster_panel)

        actions = self._build_action_row()
        centre_layout.addLayout(actions)
        splitter.addWidget(centre)

        self._preview_panel = PreviewPanel()
        self._preview_panel.setMinimumWidth(280)
        splitter.addWidget(self._preview_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)

        self.setCentralWidget(splitter)

    def _build_action_row(self) -> QHBoxLayout:
        layout = QHBoxLayout()

        self._rename_btn = QPushButton()
        self._rename_btn.setEnabled(False)
        self._rename_btn.clicked.connect(self._on_rename)
        layout.addWidget(self._rename_btn)

        self._merge_btn = QPushButton()
        self._merge_btn.setEnabled(False)
        self._merge_btn.clicked.connect(self._on_merge)
        layout.addWidget(self._merge_btn)

        self._delete_person_btn = QPushButton()
        self._delete_person_btn.setEnabled(False)
        self._delete_person_btn.setStyleSheet("color: #e57373;")
        self._delete_person_btn.clicked.connect(self._on_delete_person)
        layout.addWidget(self._delete_person_btn)

        self._remove_face_btn = QPushButton()
        self._remove_face_btn.setEnabled(False)
        self._remove_face_btn.clicked.connect(self._on_remove_face)
        layout.addWidget(self._remove_face_btn)

        self._reassign_btn = QPushButton()
        self._reassign_btn.setEnabled(False)
        self._reassign_btn.clicked.connect(self._on_reassign_face)
        layout.addWidget(self._reassign_btn)

        layout.addStretch()
        return layout

    def _build_log_dock(self) -> None:
        self._log_panel = LogPanel()
        self._log_dock = QDockWidget(self)
        self._log_dock.setWidget(self._log_panel)
        self._log_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._log_dock)
        self._log_dock.setMinimumHeight(120)

    def _build_status_bar(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._progress_bar.setMaximumWidth(300)
        status.addPermanentWidget(self._progress_bar)

        self._status_label = QLabel()
        status.addWidget(self._status_label)

    # ------------------------------------------------------------------
    # Retranslate — call after language change
    # ------------------------------------------------------------------

    def _retranslate(self) -> None:
        self.setWindowTitle(t("window_title"))
        self._folder_btn.setText(t("select_folder"))
        if not hasattr(self, "_root_folder"):
            self._folder_label.setText(f"  {t('no_folder')}")
        self._scan_btn.setText(t("scan_index"))
        self._stop_btn.setText(t("stop"))
        self._export_csv_btn.setText(t("export_csv"))
        self._export_images_btn.setText(t("export_images"))
        self._tpu_btn.setText(t("tpu_status"))
        self._settings_btn.setText(t("settings"))
        self._rename_btn.setText(t("rename_person"))
        self._merge_btn.setText(t("merge_into"))
        self._delete_person_btn.setText(t("delete_person"))
        self._remove_face_btn.setText(t("remove_face"))
        self._reassign_btn.setText(t("reassign_face"))
        self._log_dock.setWindowTitle(t("activity_log"))
        self._status_label.setText(t("ready"))

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _connect_log_handler(self) -> None:
        handler = QLogHandler(signal=self.log_signal)
        handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(handler)
        self.log_signal.connect(self._log_panel.append_log)

    # ------------------------------------------------------------------
    # Toolbar slots
    # ------------------------------------------------------------------

    @Slot()
    def _on_select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, t("select_folder"), str(Path.home())
        )
        if folder:
            self._root_folder = folder
            self._folder_label.setText(f"  {folder}")
            self._scan_btn.setEnabled(True)
            log.info("Root folder selected: %s", folder)

    @Slot()
    def _on_scan(self) -> None:
        if not hasattr(self, "_root_folder"):
            QMessageBox.warning(self, t("no_folder_title"), t("no_folder_msg"))
            return

        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, t("busy_title"), t("busy_msg"))
            return

        self._set_scanning_state(True)
        self._worker = PipelineWorker(
            root_folder=self._root_folder,
            config=self._config,
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.log_message.connect(self._log_panel.append_plain)
        self._worker.finished.connect(self._on_pipeline_finished)
        self._worker.error.connect(self._on_pipeline_error)
        self._worker.start()

    @Slot()
    def _on_stop(self) -> None:
        if self._worker:
            self._worker.abort()
            self._stop_btn.setEnabled(False)

    @Slot()
    def _on_tpu_status(self) -> None:
        dlg = TpuStatusDialog(parent=self)
        dlg.exec()

    @Slot()
    def _on_settings(self) -> None:
        dlg = SettingsDialog(current_db_path=self._db_path, parent=self)
        if dlg.exec() != SettingsDialog.Accepted:
            return

        # Database change
        new_db = dlg.selected_db_path()
        if new_db and new_db != self._db_path:
            self._db_path = new_db
            self._config.storage.db_path = new_db
            init_db(new_db)
            self._current_person_id = None
            self._current_face_id = None
            self._cluster_panel.clear()
            self._preview_panel.clear()
            self._refresh_persons()
            QMessageBox.information(self, t("settings_title"), t("db_switched"))
            log.info("Database switched to: %s", new_db)

        # Language change
        if dlg.language_changed():
            self._retranslate()
            self._refresh_persons()

    # ------------------------------------------------------------------
    # Pipeline slots
    # ------------------------------------------------------------------

    @Slot(int, int, str, str)
    def _on_progress(self, current: int, total: int, stage: str, detail: str) -> None:
        if total > 0:
            self._progress_bar.setValue(int(current / total * 100))
        self._status_label.setText(f"{stage}: {detail}")

    @Slot(bool, str)
    def _on_pipeline_finished(self, success: bool, summary: str) -> None:
        self._set_scanning_state(False)
        self._status_label.setText(summary)
        self._progress_bar.setValue(100)
        self._refresh_persons()
        if not success:
            QMessageBox.warning(self, t("warning"), summary)

    @Slot(str)
    def _on_pipeline_error(self, message: str) -> None:
        self._set_scanning_state(False)
        QMessageBox.critical(self, t("error"), message)

    # ------------------------------------------------------------------
    # Person / face interaction
    # ------------------------------------------------------------------

    @Slot(int)
    def _on_person_selected(self, person_id: int) -> None:
        self._current_person_id = person_id
        self._current_face_id = None

        with session_scope() as session:
            svc = IdentityService(session)
            person = session.get(Person, person_id)
            faces = svc.get_faces_for_person(person_id)
            if person is None:
                return
            for f in faces:
                _ = f.image  # noqa: F841
            self._cluster_panel.show_person(person.name, faces)
            self._preview_panel.clear()

        self._rename_btn.setEnabled(True)
        self._merge_btn.setEnabled(True)
        self._delete_person_btn.setEnabled(True)
        self._remove_face_btn.setEnabled(False)
        self._reassign_btn.setEnabled(False)

    @Slot(int)
    def _on_face_selected(self, face_id: int) -> None:
        self._current_face_id = face_id

        with session_scope() as session:
            face = session.get(Face, face_id)
            if face:
                _ = face.image
                if face.image:
                    _ = face.image.faces
                self._preview_panel.show_face(face)

        self._remove_face_btn.setEnabled(True)
        self._reassign_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Identity actions
    # ------------------------------------------------------------------

    @Slot()
    def _on_rename(self) -> None:
        if self._current_person_id is None:
            return

        with session_scope() as session:
            person = session.get(Person, self._current_person_id)
            if person is None:
                return
            dlg = RenameDialog(person.name, parent=self)
            if dlg.exec() != RenameDialog.Accepted:
                return
            new_name = dlg.new_name()
            if not new_name:
                QMessageBox.warning(self, t("empty_name_title"), t("empty_name_msg"))
                return
            IdentityService(session).rename_person(self._current_person_id, new_name)

        self._refresh_persons()

    @Slot()
    def _on_merge(self) -> None:
        if self._current_person_id is None:
            return

        with session_scope() as session:
            persons = session.query(Person).order_by(Person.name).all()
            for p in persons:
                _ = p.faces  # noqa: F841
            source = session.get(Person, self._current_person_id)
            if source is None or len(persons) < 2:
                return

            dlg = MergeDialog(source, persons, parent=self)
            if dlg.exec() != MergeDialog.Accepted:
                return
            target_id = dlg.target_person_id()
            if target_id is None:
                return

            try:
                IdentityService(session).merge_persons(
                    source_id=self._current_person_id, target_id=target_id
                )
            except ValueError as exc:
                QMessageBox.warning(self, t("merge_error_title"), str(exc))
                return

        self._current_person_id = None
        self._cluster_panel.clear()
        self._preview_panel.clear()
        self._refresh_persons()

    @Slot()
    def _on_delete_person(self) -> None:
        if self._current_person_id is None:
            return

        with session_scope() as session:
            person = session.get(Person, self._current_person_id)
            if person is None:
                return
            name = person.name

        reply = QMessageBox.question(
            self,
            t("delete_person_title"),
            t("delete_person_confirm", name=name),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        with session_scope() as session:
            IdentityService(session).delete_person(self._current_person_id)

        self._current_person_id = None
        self._current_face_id = None
        self._cluster_panel.clear()
        self._preview_panel.clear()
        self._delete_person_btn.setEnabled(False)
        self._rename_btn.setEnabled(False)
        self._merge_btn.setEnabled(False)
        self._refresh_persons()
        log.info("Person '%s' deleted.", name)

    @Slot()
    def _on_remove_face(self) -> None:
        if self._current_face_id is None or self._current_person_id is None:
            return

        reply = QMessageBox.question(
            self,
            t("remove_face_title"),
            t("remove_face_msg"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        with session_scope() as session:
            IdentityService(session).remove_face_from_cluster(self._current_face_id)

        self._on_person_selected(self._current_person_id)

    @Slot()
    def _on_reassign_face(self) -> None:
        if self._current_face_id is None:
            return

        with session_scope() as session:
            persons = session.query(Person).order_by(Person.name).all()
            for p in persons:
                _ = p.faces  # noqa: F841

            class _FakePerson:
                name = f"Face #{self._current_face_id}"
                id = -1
                faces: list = []

            dlg = MergeDialog(_FakePerson(), persons, parent=self)
            dlg.setWindowTitle(t("reassign_title"))
            if dlg.exec() != MergeDialog.Accepted:
                return
            target_id = dlg.target_person_id()
            if target_id is None:
                return
            IdentityService(session).reassign_face(self._current_face_id, target_id)

        if self._current_person_id:
            self._on_person_selected(self._current_person_id)

    @Slot()
    def _on_recluster(self) -> None:
        reply = QMessageBox.question(
            self,
            t("recluster_title"),
            t("recluster_msg"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._status_label.setText(t("reclustering"))
        QApplication.processEvents()

        with session_scope() as session:
            n = ClusteringService(session, self._config.clustering).recluster()

        self._status_label.setText(t("recluster_done", n=n))
        self._refresh_persons()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @Slot()
    def _on_export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, t("export_csv"), "export.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        with session_scope() as session:
            out = ExportService(session).export_csv(
                target_path=path, person_id=self._current_person_id
            )
        QMessageBox.information(self, t("export_done"), t("export_csv_saved", path=out))

    @Slot()
    def _on_export_images(self) -> None:
        if self._current_person_id is None:
            QMessageBox.information(self, t("no_person_title"), t("no_person_msg"))
            return

        folder = QFileDialog.getExistingDirectory(
            self, t("export_images"), str(Path.home())
        )
        if not folder:
            return

        with session_scope() as session:
            n = ExportService(session).export_person_images(
                person_id=self._current_person_id, target_dir=folder
            )

        QMessageBox.information(
            self, t("export_done"), t("exported_n", n=n, folder=folder)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_scanning_state(self, scanning: bool) -> None:
        self._scan_btn.setEnabled(not scanning)
        self._stop_btn.setEnabled(scanning)
        self._progress_bar.setVisible(scanning)
        if scanning:
            self._progress_bar.setValue(0)

    def _refresh_persons(self) -> None:
        with session_scope() as session:
            persons: List[Person] = (
                session.query(Person).order_by(Person.name).all()
            )
            for p in persons:
                _ = p.faces  # noqa: F841
            self._sidebar.populate(persons)
        log.debug("Sidebar refreshed: %d person(s)", len(persons))
