"""Menu, toolbar, central tabs, status bar and retranslation."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDialog,
    QDockWidget,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.logging_setup import QLogHandler
from app.paths import app_icon_path
from app.ui.i18n import t
from app.ui.panels.cluster_panel import ClusterPanel
from app.ui.panels.collage_panel import CollagePanel
from app.ui.panels.family_search_panel import FamilySearchPanel
from app.ui.panels.family_tree_panel import FamilyTreePanel
from app.ui.panels.groups_panel import GroupsPanel
from app.ui.panels.image_browser_panel import ImageBrowserPanel
from app.ui.panels.locations_panel import LocationsPanel
from app.ui.panels.log_panel import LogPanel
from app.ui.panels.objects_panel import ObjectsPanel
from app.ui.panels.persons_panel import PersonsPanel
from app.ui.panels.preview_panel import PreviewPanel
from app.ui.panels.sidebar_panel import SidebarPanel
from app.ui.widgets.eliding_label import ElidingLabel
from app.ui.widgets.flow_layout import FlowContainer

log = logging.getLogger(__name__)


class UiBuildMixin:
    """Menu, toolbar, central tabs, status bar and retranslation."""

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_menu_bar()
        self._build_toolbar()
        self._build_central()
        self._build_log_dock()
        self._build_status_bar()

    def _apply_window_icon(self) -> None:
        icon_path = app_icon_path()
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def _build_menu_bar(self) -> None:
        mb = self.menuBar()

        # ── Fájl ──────────────────────────────────────────────────────────
        self._mb_file_menu = mb.addMenu("")
        self._select_folder_action = self._mb_file_menu.addAction(
            "", self._on_select_folder
        )
        self._mb_scan_menu_action = self._mb_file_menu.addAction(
            "", self._on_open_scan_modes
        )

        # ── Eszközök ──────────────────────────────────────────────────────
        self._mb_tools_menu = mb.addMenu("")
        self._export_action = self._mb_tools_menu.addAction("", self._on_open_export)
        self._no_face_action = self._mb_tools_menu.addAction(
            "", self._on_no_face_images
        )

        # ── Összevonás ────────────────────────────────────────────────────
        self._mb_merge_menu = mb.addMenu("")
        self._suggestions_action = self._mb_merge_menu.addAction(
            "", self._on_show_suggestions
        )
        self._suggestions_action.setToolTip(t("suggestions_tip"))
        self._amerge_action = self._mb_merge_menu.addAction(
            "", self._on_open_amerge_review
        )

        # ── Rendszer ──────────────────────────────────────────────────────
        self._mb_system_menu = mb.addMenu("")
        self._settings_action = self._mb_system_menu.addAction("", self._on_settings)
        self._tasks_action = self._mb_system_menu.addAction(
            "", self._on_open_task_manager
        )
        self._mb_system_menu.addSeparator()
        self._gdrive_action = self._mb_system_menu.addAction(
            "", self._on_toggle_drive_project
        )
        self._gdrive_action.setVisible(False)
        self._mb_system_menu.addSeparator()
        self._log_action = self._mb_system_menu.addAction(
            "", self._on_log_action_toggled
        )
        self._log_action.setCheckable(True)
        self._log_action.setChecked(True)

        # ── Debug ─────────────────────────────────────────────────────────
        self._mb_debug_menu = mb.addMenu("")
        self._ai_viz_action = self._mb_debug_menu.addAction(
            "", self._open_ai_viz_window
        )
        self._nn_graph_action = self._mb_debug_menu.addAction(
            "", self._open_nn_graph_window
        )
        self._mb_debug_menu.addSeparator()
        self._tasks_debug_action = self._mb_debug_menu.addAction(
            "", self._on_open_task_manager
        )

    def _build_toolbar(self) -> None:
        tb = QToolBar(t("main_toolbar"))
        tb.setMovable(False)
        self._main_toolbar = tb
        self.addToolBar(tb)

        self._folder_btn = QPushButton()
        self._folder_btn.clicked.connect(self._on_select_folder)
        tb.addWidget(self._folder_btn)

        self._folder_label = QLabel()
        self._folder_label.setStyleSheet("color: #A6ADC8;")
        self._folder_label.setMaximumWidth(380)
        tb.addWidget(self._folder_label)

        tb.addSeparator()

        self._scan_modes_btn = QPushButton()
        self._scan_modes_btn.setEnabled(True)
        self._scan_modes_btn.clicked.connect(self._on_open_scan_modes)
        tb.addWidget(self._scan_modes_btn)

        tb.addSeparator()

        # ── Képernyőfelvétel — marad a sorban ─────────────────────────────
        from app.ui.widgets.recording_controls import RecordingControls
        self._recording_controls = RecordingControls()
        self._recording_controls.start_requested.connect(self._on_record_start)
        self._recording_controls.pause_toggle_requested.connect(
            self._on_record_pause_toggle
        )
        self._recording_controls.stop_requested.connect(self._on_record_stop)
        tb.addWidget(self._recording_controls)

    def _on_log_action_toggled(self, checked: bool = False) -> None:
        if self._log_dock.isVisible():
            self._log_dock.hide()
        else:
            self._log_dock.show()

    def _build_central(self) -> None:
        # Outer tabs: Arcok | Kollázs
        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.North)
        self._tabs.setDocumentMode(True)

        # --- Tab 0: Arcfelismerés (existing layout) ---
        face_widget = QWidget()
        face_layout = QVBoxLayout(face_widget)
        face_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        # A wide, visible handle that is easy to grab, and panels that can be
        # dragged down to their real minimums without snapping shut.
        splitter.setHandleWidth(8)
        splitter.setChildrenCollapsible(False)
        splitter.setOpaqueResize(True)
        self._face_splitter = splitter

        self._sidebar = SidebarPanel()
        self._sidebar.person_selected.connect(self._on_person_selected)
        self._sidebar.setMinimumWidth(260)
        self._sidebar.setMaximumWidth(400)
        splitter.addWidget(self._sidebar)

        centre = QWidget()
        centre_layout = QVBoxLayout(centre)
        centre_layout.setContentsMargins(0, 0, 0, 0)

        self._cluster_panel = ClusterPanel()
        self._cluster_panel.face_selected.connect(self._on_face_selected)
        self._cluster_panel.face_right_clicked.connect(self._on_cluster_face_right_clicked)
        self._cluster_panel.selection_changed.connect(self._on_face_selection_changed)
        centre_layout.addWidget(self._cluster_panel)

        actions = self._build_action_row()
        centre_layout.addWidget(actions)
        # The centre column must be able to shrink so the preview splitter
        # can be dragged wider; only the sidebar keeps a hard minimum.
        centre.setMinimumWidth(160)
        splitter.addWidget(centre)

        self._preview_panel = PreviewPanel()
        self._preview_panel.setMinimumWidth(200)
        self._preview_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._preview_panel.face_selected.connect(self._on_preview_face_selected)
        self._preview_panel.face_assign_requested.connect(self._on_preview_face_assign)
        self._preview_panel.face_delete_requested.connect(self._on_preview_face_delete)
        self._preview_panel.face_not_a_face_requested.connect(self._on_preview_face_not_a_face)
        self._preview_panel.face_create_requested.connect(self._on_preview_face_create)
        self._preview_panel.face_bbox_update_requested.connect(
            self._on_preview_face_bbox_update
        )
        self._preview_panel.face_diagnostics_requested.connect(
            self._on_preview_face_diagnostics
        )
        self._preview_panel.face_set_thumbnail_requested.connect(
            self._on_face_set_thumbnail
        )
        self._preview_panel.face_clear_thumbnail_requested.connect(
            self._on_face_clear_thumbnail
        )
        self._preview_panel.face_accept_auto_merge.connect(
            self._on_face_accept_auto_merge
        )
        self._preview_panel.face_move_auto_merge.connect(
            self._on_face_move_auto_merge
        )
        self._preview_panel.face_reject_auto_merge.connect(
            self._on_face_reject_auto_merge
        )
        self._preview_panel.face_uncertainty_change_requested.connect(
            self._on_face_uncertainty_change
        )
        self._preview_panel.object_create_requested.connect(
            self._on_preview_object_create
        )
        splitter.addWidget(self._preview_panel)

        # Sidebar keeps its size; the centre grid and the preview share the
        # remaining width and follow the handle as it is dragged.
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([320, 480, 480])
        splitter.setStyleSheet(
            "QSplitter::handle:horizontal { background: #45475a; "
            "margin: 2px 0; border-radius: 2px; }"
            "QSplitter::handle:horizontal:hover { background: #89b4fa; }"
        )

        face_layout.addWidget(splitter)
        self._tabs.addTab(face_widget, t("tab_face_recognition"))

        # --- Tab 1: Képböngésző ---
        self._image_browser = ImageBrowserPanel(config=self._config)
        self._image_browser.person_data_changed.connect(self._refresh_persons)
        self._image_browser.image_displayed.connect(self._on_browser_image_displayed)
        self._image_browser.active_person_changed.connect(
            self._on_browser_person_changed
        )
        self._image_browser.object_open_requested.connect(self._open_object_sheet)
        self._image_browser.object_search_requested.connect(
            self._on_object_search_requested
        )
        self._image_browser.pairing_settings_requested.connect(
            lambda: self._on_settings("pairing")
        )
        self._image_browser.path_repair_requested.connect(
            self._open_path_repair_dialog
        )
        self._tabs.addTab(self._image_browser, t("tab_image_browser"))

        # --- Tab 2: Családi kereső ---
        self._family_search = FamilySearchPanel()
        self._family_search.image_open_requested.connect(self._open_image_from_family_search)
        self._tabs.addTab(self._family_search, t("tab_family_search"))

        # --- Tab 3: Helyek ---
        self._locations_panel = LocationsPanel()
        self._tabs.addTab(self._locations_panel, t("tab_locations"))

        # --- Tab 4: Személyek ---
        self._persons_panel = PersonsPanel()
        self._persons_panel.person_data_changed.connect(self._refresh_persons)
        self._tabs.addTab(self._persons_panel, t("tab_persons"))

        # --- Tab 5: Objektumok ---
        self._objects_panel = ObjectsPanel(
            matching_config=getattr(self._config, "object_matching", None)
        )
        self._objects_panel.object_data_changed.connect(
            self._refresh_preview_object_markers
        )
        self._objects_panel.object_search_requested.connect(
            self._on_object_search_requested
        )
        self._tabs.addTab(self._objects_panel, t("tab_objects"))

        # --- Tab 6: Társaságok / közösségek ---
        self._groups_panel = GroupsPanel()
        self._tabs.addTab(self._groups_panel, t("tab_groups"))

        # --- Tab 7: Kollázs nézet ---
        self._collage_panel = CollagePanel()
        self._tabs.addTab(self._collage_panel, t("tab_collage"))

        # --- Tab 8: Családfa ---
        self._family_tree_panel = FamilyTreePanel()
        self._family_tree_panel.person_data_changed.connect(self._refresh_persons)
        self._tabs.addTab(self._family_tree_panel, t("tab_family_tree"))

        self.setCentralWidget(self._tabs)

    def _build_action_row(self) -> QWidget:
        # A wrapping flow layout: when the centre column is too narrow to fit
        # every button on one line, the buttons reflow onto additional rows
        # instead of clipping or forcing the column wide. This keeps the
        # preview splitter freely draggable at any width.
        container = FlowContainer(h_spacing=4, v_spacing=4)
        layout = container.layout()

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
        self._delete_person_btn.setStyleSheet(
            "QPushButton { color: #F38BA8; border-color: #6B3040; }"
            "QPushButton:hover { background-color: #3D2030; border-color: #F38BA8; }"
            "QPushButton:disabled { color: #6C7086; border-color: #313244; }"
        )
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

        self._move_faces_btn = QPushButton()
        self._move_faces_btn.setVisible(False)
        self._move_faces_btn.clicked.connect(self._on_move_faces_batch)
        layout.addWidget(self._move_faces_btn)

        self._sel_count_lbl = QLabel()
        self._sel_count_lbl.setStyleSheet("color: #888;")
        layout.addWidget(self._sel_count_lbl)

        self._person_info_btn = QPushButton()
        self._person_info_btn.setEnabled(False)
        self._person_info_btn.setToolTip(t("person_info_tip"))
        self._person_info_btn.clicked.connect(self._on_person_info)
        layout.addWidget(self._person_info_btn)

        return container

    def _build_log_dock(self) -> None:
        self._log_panel = LogPanel()
        self._log_dock = QDockWidget(self)
        self._log_dock.setWidget(self._log_panel)
        self._log_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self._log_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._log_dock)
        self._log_dock.setMinimumHeight(60)
        self._log_dock.visibilityChanged.connect(self._on_log_dock_visibility_changed)

    def _on_log_dock_visibility_changed(self, visible: bool) -> None:
        if hasattr(self, "_log_action"):
            self._log_action.blockSignals(True)
            self._log_action.setChecked(visible)
            self._log_action.blockSignals(False)

    def _build_status_bar(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._progress_bar.setMaximumWidth(300)
        status.addPermanentWidget(self._progress_bar)

        self._update_notify_btn = QPushButton()
        self._update_notify_btn.setVisible(False)
        self._update_notify_btn.setStyleSheet(
            "QPushButton { color: #F9E2AF; font-weight: bold; padding: 1px 6px; "
            "background-color: #3D3020; border: 1px solid #F9E2AF; border-radius: 3px; }"
            "QPushButton:hover { background-color: #5A4530; }"
        )
        self._update_notify_btn.clicked.connect(self._on_update_notify_clicked)
        status.addPermanentWidget(self._update_notify_btn)

        # Drive status chip — permanent widget on the right side of the bar.
        self._gdrive_chip_btn = QPushButton()
        self._gdrive_chip_btn.setFlat(True)
        self._gdrive_chip_btn.setVisible(False)
        self._gdrive_chip_btn.setStyleSheet(
            "QPushButton { color: #89DCEB; font-size: 11px; padding: 1px 6px; "
            "border: 1px solid #313244; border-radius: 3px; background: #1E1E2E; }"
            "QPushButton:hover { background: #313244; }"
        )
        self._gdrive_chip_btn.clicked.connect(self._on_gdrive_chip_clicked)

        from app.tasks import get_task_manager
        get_task_manager().counts_changed.connect(self._on_task_counts_changed)
        status.addPermanentWidget(self._gdrive_chip_btn)

        # Always-visible Task Manager button anchored at the bottom-left of the
        # status bar (addWidget → left side). Unlike the right-side chip, which
        # only appears while tasks run, this is always available.
        self._tasks_status_btn = QPushButton()
        self._tasks_status_btn.setFlat(True)
        self._tasks_status_btn.setStyleSheet(
            "QPushButton { color: #A6E3A1; font-size: 11px; padding: 1px 8px; "
            "border: 1px solid #313244; border-radius: 3px; background: #1E1E2E; }"
            "QPushButton:hover { background: #313244; }"
        )
        self._tasks_status_btn.clicked.connect(self._on_open_task_manager)
        status.addWidget(self._tasks_status_btn)

        # Eliding: a long status line must not stop the window from being
        # resized smaller (#166).
        self._status_label = ElidingLabel()
        status.addWidget(self._status_label, 1)

    # ------------------------------------------------------------------
    # Retranslate — call after language change
    # ------------------------------------------------------------------

    def _retranslate(self) -> None:
        self.setWindowTitle(t("window_title"))
        # Menu bar titles
        self._mb_file_menu.setTitle(t("mb_file"))
        self._select_folder_action.setText(t("manage_folders"))
        self._mb_scan_menu_action.setText(t("scanModes.openButton"))
        self._mb_tools_menu.setTitle(t("mb_tools"))
        self._mb_merge_menu.setTitle(t("mb_merge"))
        self._mb_system_menu.setTitle(t("mb_system"))
        self._mb_debug_menu.setTitle(t("mb_debug"))
        self._ai_viz_action.setText(t("mb_ai_viz"))
        self._nn_graph_action.setText(t("mb_nn_graph"))
        self._tasks_debug_action.setText(t("tasks_title"))
        # Toolbar
        self._folder_btn.setText(t("manage_folders"))
        if not hasattr(self, "_root_folders") or not self._root_folders:
            self._folder_label.setText(f"  {t('no_folder')}")
        self._scan_modes_btn.setText(t("scanModes.openButton"))
        self._export_action.setText(t("tb_export"))
        self._no_face_action.setText(t("view_no_face"))
        self._suggestions_action.setText(t("suggestions_btn"))
        self._suggestions_action.setToolTip(t("suggestions_tip"))
        self._refresh_amerge_btn()
        self._settings_action.setText(t("settings"))
        self._tasks_action.setText(t("tasks_btn"))
        if hasattr(self, "_tasks_status_btn"):
            self._update_tasks_button_text()
            self._tasks_status_btn.setToolTip(t("tasks_title"))
        if hasattr(self, "_recording_controls"):
            self._recording_controls.retranslate()
        self._update_gdrive_toolbar_btn()
        self._rename_btn.setText(t("rename_person"))
        self._merge_btn.setText(t("merge_into"))
        self._delete_person_btn.setText(t("delete_person"))
        self._remove_face_btn.setText(t("remove_face"))
        self._reassign_btn.setText(t("reassign_face"))
        self._move_faces_btn.setText(t("persons_move_faces_btn"))
        self._person_info_btn.setText(t("person_info"))
        self._person_info_btn.setToolTip(t("person_info_tip"))
        self._tabs.setTabText(0, t("tab_face_recognition"))
        self._tabs.setTabText(1, t("tab_image_browser"))
        self._tabs.setTabText(2, t("tab_family_search"))
        self._tabs.setTabText(3, t("tab_locations"))
        self._tabs.setTabText(4, t("tab_persons"))
        self._tabs.setTabText(5, t("tab_objects"))
        self._tabs.setTabText(6, t("tab_groups"))
        self._tabs.setTabText(7, t("tab_collage"))
        self._tabs.setTabText(8, t("tab_family_tree"))
        self._log_dock.setWindowTitle(t("activity_log"))
        if hasattr(self, "_log_action"):
            self._log_action.setText(t("toggle_log"))
            self._log_action.setToolTip(t("activity_log"))
        self._status_label.setText(t("ready"))
        if hasattr(self, "_cluster_panel"):
            self._cluster_panel.retranslate()
        if hasattr(self, "_image_browser"):
            self._image_browser.retranslate()
        if hasattr(self, "_family_search"):
            self._family_search.retranslate()
        if hasattr(self, "_locations_panel"):
            self._locations_panel.retranslate()
        if hasattr(self, "_persons_panel"):
            self._persons_panel.retranslate()
        if hasattr(self, "_objects_panel"):
            self._objects_panel.retranslate()
        if hasattr(self, "_family_tree_panel"):
            self._family_tree_panel.retranslate()

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
        from app.ui.dialogs.folder_list_dialog import FolderListDialog

        current = getattr(self, "_root_folders", [])
        dlg = FolderListDialog(list(current), parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        folders = [f for f in dlg.folders() if Path(f).is_dir()]
        if not folders:
            return
        self._root_folders = folders
        self._save_folders(folders)
        self._set_folder_label(folders)
        self._scan_modes_btn.setEnabled(True)
        log.info("Source folders updated: %s", folders)
