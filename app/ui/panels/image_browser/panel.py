"""Image Browser Panel — core widget wiring (UI build, refresh, fullscreen)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import (
    QSize,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QIcon,
    QImage,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStyle,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.db.models import Image, Person, Place
from app.services.image_browser_service import (
    ImageBrowserService,
)
from app.ui.i18n import t
from app.ui.widgets.flow_layout import FlowContainer
from app.ui.widgets.place_search_select import PlaceSearchSelect
from app.ui.widgets.universal_search_bar import UniversalSearchBar

from .canvas import CanvasMixin
from .deoldified import DeoldifiedMixin
from .face_editing import FaceEditingMixin
from .helpers import (
    _ROLE_TYPE,
    _TREE_THUMB_SIZE,
    _BboxEdit,
    _build_folder_tree,
    _FaceData,
    _hline,
    _is_visible_dir,
    _overlay_group,
)
from .image_loading import ImageLoadingMixin
from .inline_editor import InlineEditorMixin
from .interaction import InteractionMixin
from .metadata import MetadataMixin
from .navigation import NavigationMixin
from .person_actions import PersonActionsMixin
from .rerecognition import RerecognitionMixin
from .search import SearchMixin
from .tree import TreeMixin
from .widgets import (
    _BrowserTree,
    _DrawableImageLabel,
    _FocusLineEdit,
    _InlineFaceEditor,
    _PendingActionBar,
)

log = logging.getLogger(__name__)


class ImageBrowserPanel(DeoldifiedMixin, TreeMixin, RerecognitionMixin, NavigationMixin, ImageLoadingMixin, InteractionMixin, FaceEditingMixin, CanvasMixin, InlineEditorMixin, MetadataMixin, SearchMixin, PersonActionsMixin, QWidget):
    """Finder-style image browser: folder/image tree | image preview."""

    person_data_changed = Signal()
    # Emitted with (image_id, file_path) whenever a new image is displayed.
    image_displayed = Signal(int, str)
    # Emitted with the selected face's person name (str) or None whenever the
    # active face selection / assignment changes — drives the recording log.
    active_person_changed = Signal(object)
    # Emitted when the user asks to open a tagged object's data sheet (object_id)
    object_open_requested = Signal(int)
    # Emitted when the user asks to look for an object on other images:
    # (object_id, scope) where scope is "library" or "folder".
    object_search_requested = Signal(int, str)
    # Emitted from the black-and-white / colorized bar to open the pairing settings
    pairing_settings_requested = Signal()
    # Emitted when an image file could not be opened and the user asks for the
    # missing-path repair tool.
    path_repair_requested = Signal()
    def __init__(self, config=None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config

        # State
        self._current_folder: Optional[str] = None
        self._current_image_id: Optional[int] = None
        self._current_path: str = ""
        self._detection_done: bool = False
        self._face_data: List[_FaceData] = []
        # Face ids on the current image that are auto-merged-from-Unknown and
        # awaiting review (drives the amber overlay + the floating action bar).
        self._pending_face_ids: set[int] = set()
        # Object occurrence markers for the current image, with object_id:
        # (occ_id, object_id, x, y, name, bbox_or_None)
        self._object_marker_data: List[tuple] = []
        self._selected_occurrence_id: Optional[int] = None
        self._show_objects: bool = True
        self._object_opacity: float = 0.8
        self._prev_object_opacity: float = 0.8
        self._selected_face_id: Optional[int] = None
        self._editing_face_id: Optional[int] = None
        self._renaming: bool = False
        self._inline_editor_face_id: Optional[int] = None
        self._orig_img_bgr: Optional[np.ndarray] = None
        self._full_pixmap: Optional[QPixmap] = None
        self._recent_assignment_person_ids: List[int] = []
        self._all_persons: List[Person] = []
        self._all_places: List[Place] = []
        self._place_filter_id: Optional[int] = None

        # Interactive bbox editor state
        self._interactive_edit_face_id: Optional[int] = None
        self._interactive_edit_orig_bbox: Optional[Tuple[int, int, int, int]] = None
        # When set, the interactive bbox editor is editing an object occurrence
        # (not a face); the bbox commit/finalize/cancel signals route to it.
        self._interactive_edit_occ_id: Optional[int] = None
        # Whether manual-mark (draw) mode was active when the interactive edit
        # started, so it can be restored after the edit finishes/cancels.
        self._draw_mode_before_edit: bool = False
        self._undo_stack: List[_BboxEdit] = []
        self._redo_stack: List[_BboxEdit] = []

        # Overlay state
        self._show_bboxes: bool = True
        self._bbox_opacity: float = 0.7
        self._prev_bbox_opacity: float = 0.7
        self._show_labels: bool = True
        self._label_opacity: float = 0.4
        self._prev_label_opacity: float = 0.4

        # Zoom / pan
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0

        # Thumbnail / tree state
        self._thumb_cache: Dict[str, QPixmap] = {}
        self._tree_items: Dict[str, QTreeWidgetItem] = {}  # cache_key → image tree item
        self._search_result_items: Dict[str, QListWidgetItem] = {}  # cache_key → search list item
        self._current_tree_item: Optional[QTreeWidgetItem] = None

        # Drive mode — set by main window when a Drive project is open
        self._drive_client = None          # GDriveClient | None
        self._drive_mirror_dir: Optional[Path] = None
        self._drive_project_name: str = ""
        self._prefetch_set: set = set()    # image_ids whose fetch is in-flight
        # image_id → requested load: when a fetch completes AND this id is
        # the current image, we display it.
        self._drive_pending_image_id: Optional[int] = None

        # Fullscreen state
        self._fs_hidden_widgets: List[QWidget] = []
        self._fs_was_maximized: bool = False

        # Deoldified pairing state (reset on each image load)
        self._deol_pair_orig_id: Optional[int] = None  # original image ID when current is deoldified
        self._deol_pair_partner_id: Optional[int] = None  # the paired image's ID (either direction)
        self._deol_pair_color_path: str = ""           # path to colorized variant
        self._deol_bw_path: str = ""                   # path to the B&W original (view only)
        self._deol_pairing_enabled: bool = False       # data sharing allowed for this image
        self._deol_viewing_color: bool = False         # True = showing colorized pixels
        self._deol_compare: bool = False               # True = compare divider active now
        self._deol_mode: Optional[str] = None          # remembered choice: 'bw'|'color'|'compare'
        self._deol_opened_is_color: bool = False       # tree selection is a colorized variant
        self._deol_split: int = 50                     # remembered split position (percent)
        # Comparison group: B&W original first, then colorized variants. The
        # compare view composes any two of them, chosen by left/right index.
        self._deol_group: list = []                    # list[ComparisonMember]
        self._deol_left_idx: int = 0                   # group index shown on the left (B&W by default)
        self._deol_right_idx: int = 1                  # group index shown on the right (1st colorized)
        self._deol_left_bgr: Optional[np.ndarray] = None   # cached left-side pixels for compare
        self._deol_right_bgr: Optional[np.ndarray] = None  # cached right-side pixels (resized to left)

        self._build_ui()
        self._setup_shortcuts()
        self._prev_btn.setEnabled(False)
        self._next_btn.setEnabled(False)

    # ──────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 0, 8, 0)
        root.setSpacing(0)

        main_splitter = QSplitter(Qt.Horizontal)

        # ── Column 1: Finder-style folder/image tree ──────────────────
        self._tree_panel = QWidget()
        self._tree_panel.setMinimumWidth(200)
        self._tree_panel.setMaximumWidth(420)
        tp_layout = QVBoxLayout(self._tree_panel)
        tp_layout.setContentsMargins(4, 4, 4, 4)
        tp_layout.setSpacing(4)

        self._tree_hdr_lbl = QLabel()
        self._tree_hdr_lbl.setStyleSheet(
            "font-weight: bold; font-size: 11px; color: #888;"
        )
        tp_layout.addWidget(self._tree_hdr_lbl)

        # Universal search bar (replaces the old place-filter row)
        self._tree_search_bar = UniversalSearchBar(
            suggest_fn=self._tree_suggest,
            parent=self._tree_panel,
        )
        # NB: only wire search_requested — the search bar emits both
        # tokens_changed AND search_requested on every chip add/remove, so
        # connecting both would run the (expensive) search twice per change.
        self._tree_search_bar.search_requested.connect(self._on_tree_search_changed)
        tp_layout.addWidget(self._tree_search_bar)

        # Flat search results list (shown instead of tree when search is active)
        self._search_results_list = QListWidget()
        self._search_results_list.setIconSize(QSize(_TREE_THUMB_SIZE, _TREE_THUMB_SIZE))
        self._search_results_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._search_results_list.setStyleSheet(
            "QListWidget {"
            "  border: none;"
            "  background: #181825;"
            "  color: #cdd6f4;"
            "  outline: 0;"
            "}"
            "QListWidget::item {"
            "  padding: 3px 4px;"
            "  border-bottom: 1px solid #1e1e2e;"
            "}"
            "QListWidget::item:selected { background: #2d2d50; }"
            "QListWidget::item:hover:!selected { background: #22223a; }"
        )
        self._search_results_list.itemDoubleClicked.connect(
            self._on_search_result_double_clicked
        )
        self._search_results_list.hide()
        tp_layout.addWidget(self._search_results_list, stretch=1)

        self._file_tree = _BrowserTree()
        self._file_tree.setHeaderHidden(True)
        self._file_tree.setIconSize(QSize(_TREE_THUMB_SIZE, _TREE_THUMB_SIZE))
        self._file_tree.setIndentation(16)
        self._file_tree.setAnimated(True)
        self._file_tree.setStyleSheet(
            "QTreeWidget {"
            "  border: none;"
            "  background: #181825;"
            "  color: #cdd6f4;"
            "  outline: 0;"
            "}"
            "QTreeWidget::item {"
            "  padding: 2px 0;"
            "  border-bottom: 1px solid #1e1e2e;"
            "}"
            "QTreeWidget::item:selected {"
            "  background: #2d2d50;"
            "  border-bottom: 1px solid #1e1e2e;"
            "}"
            "QTreeWidget::item:hover:!selected {"
            "  background: #22223a;"
            "}"
        )
        self._file_tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self._file_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._file_tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        self._file_tree.itemExpanded.connect(self._on_tree_item_expanded)
        self._file_tree.currentItemChanged.connect(self._on_tree_current_item_changed)
        self._file_tree.go_prev.connect(self._navigate_prev)
        self._file_tree.go_next.connect(self._navigate_next)
        tp_layout.addWidget(self._file_tree, stretch=1)

        main_splitter.addWidget(self._tree_panel)

        # ── Column 2: Preview + info sub-splitter ────────────────────
        preview_splitter = QSplitter(Qt.Horizontal)

        # Image area
        image_widget = QWidget()
        image_widget.setMinimumWidth(320)
        im_layout = QVBoxLayout(image_widget)
        im_layout.setContentsMargins(4, 4, 4, 4)
        im_layout.setSpacing(4)

        # Draw mode toggle + nav buttons + fullscreen row
        # Flow layout: the controls wrap onto further rows when the image
        # column is narrow, so the whole window stays resizable down to the
        # size of the picture itself.
        top_bar_widget = FlowContainer(h_spacing=6, v_spacing=4)
        top_bar = top_bar_widget.layout()

        _nav_style = (
            "QPushButton {"
            "  font-size: 16px; padding: 0 10px; min-width: 32px; min-height: 26px;"
            "  background: #313244; border: 1px solid #45475a; border-radius: 4px;"
            "  color: #cdd6f4;"
            "}"
            "QPushButton:hover  { background: #45475a; }"
            "QPushButton:pressed { background: #585b70; }"
            "QPushButton:disabled { color: #45475a; border-color: #313244; }"
            "QPushButton:checked { background: #585b70; border-color: #88aaff; }"
        )

        # Toggle for the left (folder) panel
        self._toggle_left_btn = QPushButton("◧")
        self._toggle_left_btn.setCheckable(True)
        self._toggle_left_btn.setChecked(True)
        self._toggle_left_btn.setStyleSheet(_nav_style)
        self._toggle_left_btn.toggled.connect(self._on_toggle_left_panel)
        top_bar.addWidget(self._toggle_left_btn)

        self._draw_mode_btn = QPushButton()
        self._draw_mode_btn.setCheckable(True)
        self._draw_mode_btn.toggled.connect(self._on_draw_mode_toggled)
        top_bar.addWidget(self._draw_mode_btn)

        self._object_mode_btn = QPushButton()
        self._object_mode_btn.setCheckable(True)
        self._object_mode_btn.toggled.connect(self._on_object_mode_toggled)
        top_bar.addWidget(self._object_mode_btn)

        # ── Overlay controls ─────────────────────────────────────────────
        _ov_lbl_style = "QLabel { color: #aaa; font-size: 11px; }"
        _ov_pct_style = "QLabel { color: #888; font-size: 11px; min-width: 30px; }"
        _ov_chk_style = (
            "QCheckBox { color: #cdd6f4; font-size: 11px; spacing: 3px; }"
            "QCheckBox::indicator { width: 13px; height: 13px; }"
        )
        _ov_sld_style = (
            "QSlider::groove:horizontal { height: 4px; background: #45475a; border-radius: 2px; }"
            "QSlider::handle:horizontal { width: 10px; height: 10px; margin: -3px 0;"
            " background: #89b4fa; border-radius: 5px; }"
            "QSlider::sub-page:horizontal { background: #89b4fa; border-radius: 2px; }"
            "QSlider:disabled { opacity: 0.4; }"
        )

        self._bbox_check = QCheckBox(t("overlay_bboxes"))
        self._bbox_check.setChecked(True)
        self._bbox_check.setToolTip(t("overlay_bbox_tip"))
        self._bbox_check.setStyleSheet(_ov_chk_style)
        _bbox_group = _overlay_group()
        _bbox_group.layout().addWidget(self._bbox_check)

        self._bbox_slider = QSlider(Qt.Horizontal)
        self._bbox_slider.setRange(0, 100)
        self._bbox_slider.setValue(70)
        self._bbox_slider.setToolTip(t("overlay_bbox_tip"))
        self._bbox_slider.setMinimumWidth(60)
        self._bbox_slider.setMaximumWidth(110)
        self._bbox_slider.setStyleSheet(_ov_sld_style)
        _bbox_group.layout().addWidget(self._bbox_slider)

        self._bbox_pct_label = QLabel("70%")
        self._bbox_pct_label.setStyleSheet(_ov_pct_style)
        self._bbox_pct_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        _bbox_group.layout().addWidget(self._bbox_pct_label)
        top_bar.addWidget(_bbox_group)

        self._label_check = QCheckBox(t("overlay_labels"))
        self._label_check.setChecked(True)
        self._label_check.setToolTip(t("overlay_label_tip"))
        self._label_check.setStyleSheet(_ov_chk_style)
        _label_group = _overlay_group()
        _label_group.layout().addWidget(self._label_check)

        self._label_slider = QSlider(Qt.Horizontal)
        self._label_slider.setRange(0, 100)
        self._label_slider.setValue(40)
        self._label_slider.setToolTip(t("overlay_label_tip"))
        self._label_slider.setMinimumWidth(60)
        self._label_slider.setMaximumWidth(110)
        self._label_slider.setStyleSheet(_ov_sld_style)
        _label_group.layout().addWidget(self._label_slider)

        self._label_pct_label = QLabel("40%")
        self._label_pct_label.setStyleSheet(_ov_pct_style)
        self._label_pct_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        _label_group.layout().addWidget(self._label_pct_label)
        top_bar.addWidget(_label_group)

        # Object overlay controls (independent show/opacity, cyan colour)
        self._object_check = QCheckBox(t("overlay_objects"))
        self._object_check.setChecked(True)
        self._object_check.setToolTip(t("overlay_objects_tip"))
        self._object_check.setStyleSheet(_ov_chk_style)
        _object_group = _overlay_group()
        _object_group.layout().addWidget(self._object_check)

        self._object_slider = QSlider(Qt.Horizontal)
        self._object_slider.setRange(0, 100)
        self._object_slider.setValue(80)
        self._object_slider.setToolTip(t("overlay_objects_tip"))
        self._object_slider.setMinimumWidth(60)
        self._object_slider.setMaximumWidth(110)
        self._object_slider.setStyleSheet(_ov_sld_style)
        _object_group.layout().addWidget(self._object_slider)

        self._object_pct_label = QLabel("80%")
        self._object_pct_label.setStyleSheet(_ov_pct_style)
        self._object_pct_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        _object_group.layout().addWidget(self._object_pct_label)
        top_bar.addWidget(_object_group)

        # Wire overlay signals
        self._bbox_check.toggled.connect(self._on_ov_bbox_check)
        self._bbox_slider.valueChanged.connect(self._on_ov_bbox_slider)
        self._label_check.toggled.connect(self._on_ov_label_check)
        self._label_slider.valueChanged.connect(self._on_ov_label_slider)
        self._object_check.toggled.connect(self._on_ov_object_check)
        self._object_slider.valueChanged.connect(self._on_ov_object_slider)
        # ─────────────────────────────────────────────────────────────────

        self._prev_btn = QPushButton("◀")
        self._prev_btn.setToolTip(t("img_nav_prev_tip"))
        self._prev_btn.setStyleSheet(_nav_style)
        self._prev_btn.clicked.connect(self._navigate_prev)
        top_bar.addWidget(self._prev_btn)

        self._next_btn = QPushButton("▶")
        self._next_btn.setToolTip(t("img_nav_next_tip"))
        self._next_btn.setStyleSheet(_nav_style)
        self._next_btn.clicked.connect(self._navigate_next)
        top_bar.addWidget(self._next_btn)

        self._fs_btn = QPushButton()
        self._fs_btn.clicked.connect(self._enter_fullscreen)
        top_bar.addWidget(self._fs_btn)

        self._exit_fs_btn = QPushButton()
        self._exit_fs_btn.setStyleSheet(
            "QPushButton { color: #ff8888; font-weight: bold; }"
        )
        self._exit_fs_btn.clicked.connect(self._exit_fullscreen)
        self._exit_fs_btn.setVisible(False)
        top_bar.addWidget(self._exit_fs_btn)

        # Toggle for the right (info) panel
        self._toggle_right_btn = QPushButton("◨")
        self._toggle_right_btn.setCheckable(True)
        self._toggle_right_btn.setChecked(True)
        self._toggle_right_btn.setStyleSheet(_nav_style)
        self._toggle_right_btn.toggled.connect(self._on_toggle_right_panel)
        top_bar.addWidget(self._toggle_right_btn)

        im_layout.addWidget(top_bar_widget)

        self._draw_hint = QLabel()
        self._draw_hint.setAlignment(Qt.AlignCenter)
        self._draw_hint.setStyleSheet(
            "color: #ffcc00; font-size: 11px; background: #2a2000; padding: 3px;"
        )
        self._draw_hint.setVisible(False)
        im_layout.addWidget(self._draw_hint)

        # ── Deoldified pair toggle bar ────────────────────────────────────
        self._deoldified_bar = QWidget()
        _deol_row = QHBoxLayout(self._deoldified_bar)
        _deol_row.setContentsMargins(0, 2, 0, 2)
        _deol_row.setSpacing(6)
        self._deol_lbl = QLabel()
        self._deol_lbl.setStyleSheet("color: #888; font-size: 11px;")
        _deol_row.addWidget(self._deol_lbl)
        self._btn_view_bw = QPushButton()
        self._btn_view_bw.setCheckable(True)
        self._btn_view_bw.setStyleSheet(_nav_style)
        self._btn_view_bw.clicked.connect(lambda: self._on_deol_view_toggle(False))
        _deol_row.addWidget(self._btn_view_bw)
        self._btn_view_color = QPushButton()
        self._btn_view_color.setCheckable(True)
        self._btn_view_color.setStyleSheet(_nav_style)
        self._btn_view_color.clicked.connect(lambda: self._on_deol_view_toggle(True))
        _deol_row.addWidget(self._btn_view_color)
        self._btn_view_compare = QPushButton()
        self._btn_view_compare.setCheckable(True)
        self._btn_view_compare.setStyleSheet(_nav_style)
        self._btn_view_compare.clicked.connect(self._on_deol_compare_toggle)
        _deol_row.addWidget(self._btn_view_compare)
        # Left/right pickers for the comparison group — only shown in compare
        # mode when 3+ related images exist (B&W original + 2+ colorized).
        self._deol_left_lbl = QLabel()
        self._deol_left_lbl.setStyleSheet("color: #888; font-size: 11px;")
        _deol_row.addWidget(self._deol_left_lbl)
        self._deol_left_combo = QComboBox()
        self._deol_left_combo.currentIndexChanged.connect(self._on_deol_left_changed)
        _deol_row.addWidget(self._deol_left_combo)
        self._deol_right_lbl = QLabel()
        self._deol_right_lbl.setStyleSheet("color: #888; font-size: 11px;")
        _deol_row.addWidget(self._deol_right_lbl)
        self._deol_right_combo = QComboBox()
        self._deol_right_combo.currentIndexChanged.connect(self._on_deol_right_changed)
        _deol_row.addWidget(self._deol_right_combo)
        _deol_row.addStretch()
        self._btn_deol_sync = QPushButton()
        self._btn_deol_sync.setStyleSheet(_nav_style)
        self._btn_deol_sync.clicked.connect(self._on_deol_sync_clicked)
        _deol_row.addWidget(self._btn_deol_sync)
        # Shown only after a side failed to load, so a stale path offers its
        # own way out instead of leaving the toggle apparently dead.
        self._btn_deol_fix_paths = QPushButton()
        self._btn_deol_fix_paths.setStyleSheet(_nav_style)
        self._btn_deol_fix_paths.clicked.connect(self.path_repair_requested.emit)
        self._btn_deol_fix_paths.setVisible(False)
        _deol_row.addWidget(self._btn_deol_fix_paths)
        self._btn_deol_settings = QPushButton("⚙")
        self._btn_deol_settings.setStyleSheet(_nav_style)
        self._btn_deol_settings.clicked.connect(self.pairing_settings_requested.emit)
        _deol_row.addWidget(self._btn_deol_settings)
        self._deoldified_bar.setVisible(False)
        im_layout.addWidget(self._deoldified_bar)

        self._image_label = _DrawableImageLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(360, 260)
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_label.clicked.connect(self._on_image_clicked)
        self._image_label.rect_drawn.connect(self._on_rect_drawn)
        self._image_label.right_clicked.connect(self._on_image_right_clicked)
        self._image_label.wheel_zoomed.connect(self._on_wheel_zoom)
        self._image_label.pan_moved.connect(self._on_pan_moved)
        self._image_label.go_prev.connect(self._navigate_prev)
        self._image_label.go_next.connect(self._navigate_next)
        self._image_label.bbox_edit_committed.connect(self._on_interactive_bbox_committed)
        self._image_label.bbox_edit_finalized.connect(self._on_interactive_bbox_finalized)
        self._image_label.bbox_edit_cancelled.connect(self._on_interactive_bbox_cancelled)
        self._image_label.undo_requested.connect(self._undo_bbox_edit)
        self._image_label.redo_requested.connect(self._redo_bbox_edit)
        self._image_label.compare_dragged.connect(self._on_compare_dragged)
        self._image_label.object_rect_drawn.connect(self._on_object_rect_drawn)
        im_layout.addWidget(self._image_label, stretch=1)

        # Register undo/redo with the global ShortcutService so Ctrl+Z/Y
        # fires even when the image label doesn't have keyboard focus.
        from app.services.shortcut_service import get_shortcut_service
        _svc = get_shortcut_service()
        _svc.register("bbox.undo", self._undo_bbox_edit)
        _svc.register("bbox.redo", self._redo_bbox_edit)
        _svc.register("face.cycle_next", self._cycle_face_next)

        # Inline editor (floats over image label)
        self._inline_editor = _InlineFaceEditor(self._image_label)
        self._inline_editor.assign_requested.connect(self._on_inline_assign)
        self._inline_editor.create_requested.connect(self._on_inline_create)
        self._inline_editor.closed.connect(self._hide_inline_editor)
        # Whether background match scoring should preselect the top match when it
        # arrives (set per open in _start_match_scoring; true for unassigned faces).
        self._match_score_preselect: bool = False
        # Face id currently being scored in the background (dedupes concurrent
        # requests when the info panel and the inline editor show the same face).
        self._match_score_inflight: Optional[int] = None

        # Floating quick-action bar for pending (auto-merged) faces.
        self._pending_bar = _PendingActionBar(self._image_label)
        self._pending_bar.hide()
        self._pending_bar.accept_requested.connect(self._on_pending_accept)
        self._pending_bar.move_requested.connect(self._on_pending_move)
        self._pending_bar.delete_requested.connect(self._on_pending_delete)

        preview_splitter.addWidget(image_widget)

        # Info panel (right side)
        info_content = QWidget()
        info_layout = QVBoxLayout(info_content)
        info_layout.setContentsMargins(8, 8, 8, 8)
        info_layout.setSpacing(6)

        # Save everything about this image (persons + face boxes, GPS, date)
        # into its metadata. Disabled until an image is displayed.
        self._save_meta_btn = QPushButton()
        self._save_meta_btn.setEnabled(False)
        self._save_meta_btn.clicked.connect(self._on_save_face_metadata)
        info_layout.addWidget(self._save_meta_btn)

        info_layout.addWidget(_hline())

        self._folder_hdr = QLabel()
        self._folder_hdr.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        info_layout.addWidget(self._folder_hdr)

        self._folder_label = QLabel("")
        self._folder_label.setWordWrap(True)
        self._folder_label.setMinimumWidth(0)
        self._folder_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._folder_label.setStyleSheet("color: #88aaff; font-size: 11px;")
        info_layout.addWidget(self._folder_label)

        self._filename_hdr = QLabel()
        self._filename_hdr.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        info_layout.addWidget(self._filename_hdr)

        self._filename_label = QLabel("")
        self._filename_label.setWordWrap(True)
        self._filename_label.setMinimumWidth(0)
        self._filename_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._filename_label.setStyleSheet("color: #ddaa55; font-size: 11px;")
        info_layout.addWidget(self._filename_label)

        info_layout.addWidget(_hline())

        self._date_hdr = QLabel()
        self._date_hdr.setWordWrap(True)
        self._date_hdr.setMinimumWidth(0)
        self._date_hdr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._date_hdr.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        info_layout.addWidget(self._date_hdr)

        self._photo_date_edit = QLineEdit()
        self._photo_date_edit.returnPressed.connect(self._save_photo_date)
        self._photo_date_edit.editingFinished.connect(self._save_photo_date)
        info_layout.addWidget(self._photo_date_edit)

        # Estimated date — fallback used only when no photo date is known.
        self._estimated_date_hdr = QLabel(t("ibp_estimated_date_hdr"))
        self._estimated_date_hdr.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        info_layout.addWidget(self._estimated_date_hdr)
        self._estimated_date_edit = QLineEdit()
        self._estimated_date_edit.returnPressed.connect(self._save_estimated_date)
        self._estimated_date_edit.editingFinished.connect(self._save_estimated_date)
        info_layout.addWidget(self._estimated_date_edit)

        # EXIF suggestion section — header on its own line, value + button below
        exif_row = QWidget()
        exif_row_layout = QVBoxLayout(exif_row)
        exif_row_layout.setContentsMargins(0, 0, 0, 0)
        exif_row_layout.setSpacing(2)
        self._exif_suggestion_hdr = QLabel()
        self._exif_suggestion_hdr.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        exif_row_layout.addWidget(self._exif_suggestion_hdr)
        exif_val_row = QHBoxLayout()
        exif_val_row.setContentsMargins(0, 0, 0, 0)
        exif_val_row.setSpacing(4)
        self._exif_suggestion_label = QLabel("")
        self._exif_suggestion_label.setWordWrap(True)
        self._exif_suggestion_label.setMinimumWidth(0)
        self._exif_suggestion_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._exif_suggestion_label.setStyleSheet("color: #aaddaa; font-size: 11px;")
        exif_val_row.addWidget(self._exif_suggestion_label, 1)
        self._exif_accept_btn = QPushButton()
        self._exif_accept_btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self._exif_accept_btn.clicked.connect(self._accept_exif_suggestion)
        exif_val_row.addWidget(self._exif_accept_btn, 0)
        exif_row_layout.addLayout(exif_val_row)
        self._exif_row_widget = exif_row
        self._exif_row_widget.setVisible(False)
        info_layout.addWidget(self._exif_row_widget)

        self._update_exif_date_btn = QPushButton()
        self._update_exif_date_btn.clicked.connect(self._update_exif_date)
        info_layout.addWidget(self._update_exif_date_btn)

        info_layout.addWidget(_hline())

        self._place_hdr = QLabel()
        self._place_hdr.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        info_layout.addWidget(self._place_hdr)

        self._place_status_label = QLabel("")
        self._place_status_label.setWordWrap(True)
        self._place_status_label.setMinimumWidth(0)
        self._place_status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        info_layout.addWidget(self._place_status_label)

        self._place_search = PlaceSearchSelect()
        self._place_search.place_selected.connect(self._on_place_selected)
        # Double-clicking a result accepts it, mirroring the "Assign" button.
        self._place_search.place_double_clicked.connect(
            lambda _place_id: self._assign_existing_place()
        )
        # Enter on a non-matching query creates the place (same as the button).
        self._place_search.create_requested.connect(
            lambda _name: self._create_or_assign_place_by_text()
        )
        info_layout.addWidget(self._place_search)

        place_btns = QHBoxLayout()
        place_btns.setContentsMargins(0, 0, 0, 0)
        place_btns.setSpacing(4)
        self._place_assign_btn = QPushButton()
        self._place_assign_btn.clicked.connect(self._assign_existing_place)
        place_btns.addWidget(self._place_assign_btn)
        self._place_create_btn = QPushButton()
        self._place_create_btn.clicked.connect(self._create_or_assign_place_by_text)
        place_btns.addWidget(self._place_create_btn)
        info_layout.addLayout(place_btns)

        self._place_rename_edit = QLineEdit()
        self._place_rename_edit.returnPressed.connect(self._rename_current_place)
        info_layout.addWidget(self._place_rename_edit)

        self._place_rename_btn = QPushButton()
        self._place_rename_btn.clicked.connect(self._rename_current_place)
        info_layout.addWidget(self._place_rename_btn)

        self._place_coords_label = QLabel()
        self._place_coords_label.setStyleSheet("color: #888; font-size: 10px;")
        self._place_coords_label.setWordWrap(True)
        info_layout.addWidget(self._place_coords_label)

        info_layout.addWidget(_hline())

        # ── Image-level GPS coordinates ───────────────────────────────
        self._gps_hdr = QLabel()
        self._gps_hdr.setStyleSheet("font-weight: bold; color: #888; font-size: 11px;")
        info_layout.addWidget(self._gps_hdr)

        self._gps_source_label = QLabel()
        self._gps_source_label.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        info_layout.addWidget(self._gps_source_label)

        self._gps_edit = QLineEdit()
        self._gps_edit.returnPressed.connect(self._save_image_coords)
        info_layout.addWidget(self._gps_edit)

        gps_btns = QHBoxLayout()
        gps_btns.setContentsMargins(0, 0, 0, 0)
        gps_btns.setSpacing(4)
        self._gps_save_btn = QPushButton()
        self._gps_save_btn.clicked.connect(self._save_image_coords)
        gps_btns.addWidget(self._gps_save_btn)
        self._gps_clear_btn = QPushButton()
        self._gps_clear_btn.clicked.connect(self._clear_image_coords)
        gps_btns.addWidget(self._gps_clear_btn)
        info_layout.addLayout(gps_btns)

        info_layout.addWidget(_hline())

        # ── Image note ────────────────────────────────────────────────
        self._note_hdr = QLabel()
        self._note_hdr.setStyleSheet("font-weight: bold; color: #888; font-size: 11px;")
        info_layout.addWidget(self._note_hdr)

        self._note_edit = QTextEdit()
        self._note_edit.setFixedHeight(72)
        self._note_edit.setAcceptRichText(False)
        info_layout.addWidget(self._note_edit)

        self._note_save_timer = QTimer(self)
        self._note_save_timer.setSingleShot(True)
        self._note_save_timer.setInterval(600)
        self._note_save_timer.timeout.connect(self._save_note)
        self._note_edit.textChanged.connect(self._note_save_timer.start)

        info_layout.addWidget(_hline())

        self._face_hdr = QLabel()
        self._face_hdr.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        info_layout.addWidget(self._face_hdr)

        self._face_status_label = QLabel()
        self._face_status_label.setWordWrap(True)
        self._face_status_label.setMinimumWidth(0)
        self._face_status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._face_status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        info_layout.addWidget(self._face_status_label)

        self._person_name_label = QLabel("")
        self._person_name_label.setWordWrap(True)
        self._person_name_label.setMinimumWidth(0)
        self._person_name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._person_name_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #88ee88; padding: 4px 0px;"
        )
        self._person_name_label.setVisible(False)
        info_layout.addWidget(self._person_name_label)

        self._rename_btn = QPushButton()
        self._rename_btn.setVisible(False)
        self._rename_btn.clicked.connect(self._start_rename)
        info_layout.addWidget(self._rename_btn)

        self._person_info_btn = QPushButton()
        self._person_info_btn.setVisible(False)
        self._person_info_btn.clicked.connect(self._on_person_info)
        info_layout.addWidget(self._person_info_btn)

        self._rename_edit = _FocusLineEdit()
        self._rename_edit.setVisible(False)
        self._rename_edit.returnPressed.connect(self._commit_rename)
        self._rename_edit.focus_lost.connect(self._commit_rename)
        self._rename_edit.escape_pressed.connect(self._cancel_rename)
        info_layout.addWidget(self._rename_edit)

        info_layout.addWidget(_hline())

        # SINGLE assignment surface: this button opens the same context-menu
        # style inline editor as right-clicking the face (search + match
        # percentages + create-new-person in one popup).  The panel used to
        # duplicate all of that inline, which meant two divergent assign UIs.
        self._assign_open_btn = QPushButton()
        self._assign_open_btn.clicked.connect(self._open_assign_editor)
        self._assign_open_btn.setEnabled(False)
        info_layout.addWidget(self._assign_open_btn)

        info_layout.addStretch()

        self._info_scroll = QScrollArea()
        self._info_scroll.setWidget(info_content)
        self._info_scroll.setWidgetResizable(True)
        self._info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._info_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._info_scroll.setFrameShape(QFrame.NoFrame)
        self._info_scroll.setMinimumWidth(160)
        # Room to drag the splitter well past the old 340 px ceiling: the
        # place/person rows need it on a wide screen.
        self._info_scroll.setMaximumWidth(560)
        # Reserve the scrollbar's width on the right so the vertical scrollbar
        # never covers the frames of the widgets underneath it.
        _sb = self.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        info_layout.setContentsMargins(8, 8, 8 + _sb, 8)
        preview_splitter.addWidget(self._info_scroll)

        preview_splitter.setStretchFactor(0, 3)
        preview_splitter.setStretchFactor(1, 1)

        main_splitter.addWidget(preview_splitter)

        main_splitter.setSizes([280, 880])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 2)

        root.addWidget(main_splitter, stretch=1)
        self.retranslate()

    def _setup_shortcuts(self) -> None:
        from app.services.shortcut_service import get_shortcut_service
        get_shortcut_service().register("general.fullscreen", self._toggle_fullscreen)

    # ──────────────────────────────────────────────────────────────────
    # Public API  (consumed by MainWindow)
    # ──────────────────────────────────────────────────────────────────

    def retranslate(self) -> None:
        self._tree_hdr_lbl.setText(t("ib3_folders_hdr"))
        self._tree_search_bar.set_placeholder(t("ibp_universal_placeholder"))
        self._tree_search_bar.retranslate()
        self._toggle_left_btn.setToolTip(t("ib3_toggle_left_tip"))
        self._toggle_right_btn.setToolTip(t("ib3_toggle_right_tip"))
        self._save_meta_btn.setText(t("ib3_save_meta_btn"))
        self._save_meta_btn.setToolTip(t("ib3_save_meta_tip"))
        self._draw_mode_btn.setText(t("ibp_manual_mark"))
        self._draw_mode_btn.setToolTip(t("ibp_manual_mark_tooltip"))
        self._object_mode_btn.setText(t("object_mode"))
        self._object_mode_btn.setToolTip(t("object_mode_tip"))
        self._object_check.setText(t("overlay_objects"))
        self._object_check.setToolTip(t("overlay_objects_tip"))
        self._fs_btn.setText(t("ibp_fullscreen"))
        self._exit_fs_btn.setText(t("ibp_exit_fullscreen"))
        self._draw_hint.setText(t("ibp_draw_hint_add"))
        self._image_label.setText(t("ib3_select_image_hint"))
        self._folder_hdr.setText(t("ibp_folder_hdr"))
        self._filename_hdr.setText(t("ibp_filename_hdr"))
        self._date_hdr.setText(t("ibp_date_hdr"))
        self._photo_date_edit.setPlaceholderText(t("ibp_date_placeholder"))
        self._photo_date_edit.setToolTip(t("ibp_date_tooltip"))
        self._estimated_date_hdr.setText(t("ibp_estimated_date_hdr"))
        self._estimated_date_edit.setPlaceholderText(t("ibp_estimated_date_placeholder"))
        self._estimated_date_edit.setToolTip(t("ibp_estimated_date_tooltip"))
        self._exif_suggestion_hdr.setText(t("ibp_exif_suggestion"))
        self._exif_accept_btn.setText(t("ibp_accept_exif"))
        self._exif_accept_btn.setToolTip(t("ibp_accept_exif_tooltip"))
        self._place_hdr.setText(t("ibp_place_hdr"))
        self._place_assign_btn.setText(t("ibp_place_assign_btn"))
        self._place_create_btn.setText(t("ibp_place_create_btn"))
        self._place_rename_edit.setPlaceholderText(t("ibp_place_rename_placeholder"))
        self._place_rename_btn.setText(t("ibp_place_rename_btn"))
        self._place_search.retranslate()
        self._gps_hdr.setText(t("ibp_gps_hdr"))
        self._gps_edit.setPlaceholderText(t("ibp_gps_placeholder"))
        self._gps_save_btn.setText(t("ibp_gps_save_btn"))
        self._gps_clear_btn.setText(t("ibp_gps_clear_btn"))
        self._update_exif_date_btn.setText(t("ibp_update_exif_date_btn"))
        self._update_exif_date_btn.setToolTip(t("ibp_update_exif_date_tip"))
        self._note_hdr.setText(t("ibp_note_hdr"))
        self._note_edit.setPlaceholderText(t("ibp_note_placeholder"))
        self._note_edit.setToolTip(t("ibp_note_tooltip"))
        self._face_hdr.setText(t("ibp_face_hdr"))
        self._face_status_label.setText(t("ibp_click_face"))
        self._rename_btn.setText(t("ibp_rename_btn"))
        self._rename_btn.setToolTip(t("ibp_rename_tooltip"))
        self._person_info_btn.setText(t("ibp_person_info_btn"))
        self._person_info_btn.setToolTip(t("ibp_person_info_tooltip"))
        self._rename_edit.setPlaceholderText(t("ibp_rename_placeholder"))
        self._assign_open_btn.setText(t("ibp_assign_open_btn"))
        self._inline_editor.retranslate()
        self._deol_lbl.setText(t("ibp_deol_pair_lbl"))
        self._btn_view_bw.setText(t("ibp_view_original_bw"))
        self._btn_view_color.setText(t("ibp_view_colorized"))
        self._btn_view_compare.setText(t("ibp_view_compare"))
        self._btn_view_compare.setToolTip(t("ibp_view_compare_tip"))
        self._deol_left_lbl.setText(t("ibp_deol_left"))
        self._deol_right_lbl.setText(t("ibp_deol_right"))
        if self._deol_group:
            self._populate_deol_combos()  # refresh translated member labels
        self._btn_deol_sync.setText(t("ibp_deol_sync"))
        self._btn_deol_sync.setToolTip(t("ibp_deol_sync_tip"))
        self._btn_deol_fix_paths.setText(t("ibp_deol_fix_paths"))
        self._btn_deol_fix_paths.setToolTip(t("ibp_deol_fix_paths_tip"))
        self._btn_deol_settings.setToolTip(t("ibp_deol_settings_tip"))

    def reload_current_image(self) -> None:
        """Re-apply settings-dependent state to the image already on screen."""
        if self._current_image_id is not None:
            self._load_image(self._current_image_id)

    def refresh(self) -> None:
        """Reload folder list from DB; keep the selected folder expanded if still present."""
        log.debug("ImageBrowserPanel.refresh() — reloading folder tree")
        # New or re-pointed rows may have created (or broken) pairs
        from app.services.deoldified_pairing_service import (
            invalidate_deoldified_index,
        )
        invalidate_deoldified_index()
        # A refresh follows a scan, an import or a path repair — files may have
        # appeared or moved, so drop the cached existence answers too.
        from app.services.image_library_service import (
            invalidate_path_existence_cache,
        )
        invalidate_path_existence_cache()
        self._reload_persons_combo()
        self._reload_places()
        prev_folder = self._current_folder

        self._file_tree.blockSignals(True)
        self._file_tree.clear()
        self._tree_items.clear()
        self._file_tree.blockSignals(False)

        with session_scope() as session:
            folders = ImageBrowserService(session).list_folders()

        folders = [f for f in folders if _is_visible_dir(f.folder_path)]

        if not folders:
            self._tree_hdr_lbl.setText(t("ib3_folders_hdr"))
            placeholder = QTreeWidgetItem([t("ib3_no_folders")])
            placeholder.setData(0, _ROLE_TYPE, "placeholder")
            placeholder.setFlags(Qt.NoItemFlags)
            self._file_tree.addTopLevelItem(placeholder)
            return

        self._tree_hdr_lbl.setText(t("ib3_folders_hdr_n", n=len(folders)))

        # Build and populate the hierarchical folder tree
        tree_roots = _build_folder_tree(folders)
        for root_node in tree_roots:
            self._file_tree.addTopLevelItem(self._make_tree_node_item(root_node))

        # Restore the previously open folder if still present
        restore_item: Optional[QTreeWidgetItem] = None
        if prev_folder:
            restore_item = self._find_folder_item_by_path(prev_folder)

        if restore_item is not None:
            self._expand_ancestors(restore_item)
            # setExpanded fires itemExpanded → _on_tree_item_expanded handles
            # subfolder creation and image loading automatically
            restore_item.setExpanded(True)
            # Re-select the previously open image if still present
            if self._current_image_id is not None:
                cache_key = f"thumb_{self._current_image_id}"
                img_item = self._tree_items.get(cache_key)
                if img_item is not None:
                    self._file_tree.setCurrentItem(img_item)
                    self._file_tree.scrollToItem(img_item)
                    self._current_tree_item = img_item
                    self._update_nav_buttons()
        else:
            # No previously open folder — auto-open first image in first folder
            if self._current_image_id is None:
                self._open_first_image()

    # ──────────────────────────────────────────────────────────────────
    # Drive mode
    # ──────────────────────────────────────────────────────────────────

    def set_drive_mode(self, client, mirror_dir: Path, project_name: str) -> None:
        """Activate Drive-aware lazy loading.

        Called by the main window after a Drive project is opened.  The
        panel uses *client* to download images on demand and *mirror_dir*
        to write them.

        Args:
            client:       Authenticated :class:`~app.gdrive.protocols.GDriveClient`.
            mirror_dir:   Persistent local directory for downloaded images.
            project_name: Human-readable Drive folder name shown in the tree.
        """
        self._drive_client = client
        self._drive_mirror_dir = mirror_dir
        self._drive_project_name = project_name
        self._prefetch_set.clear()
        log.info("ImageBrowser: Drive mode ON — project=%r", project_name)

    def clear_drive_mode(self) -> None:
        """Deactivate Drive mode; revert to local-only file access."""
        self._drive_client = None
        self._drive_mirror_dir = None
        self._drive_project_name = ""
        self._prefetch_set.clear()
        self._drive_pending_image_id = None
        log.info("ImageBrowser: Drive mode OFF")

    def open_image_by_id(self, image_id: int) -> None:
        """Open an image in the existing viewer, expanding its folder when possible."""
        self._current_tree_item = None
        with session_scope() as session:
            img = session.get(Image, image_id)
            if img is None:
                return
            folder_path = str(Path(img.file_path).parent)

        folder_item = self._find_folder_item_by_path(folder_path)
        if folder_item is not None:
            self._expand_ancestors(folder_item)
            # setExpanded fires _on_tree_item_expanded → loads subfolders + images
            folder_item.setExpanded(True)
            cache_key = f"thumb_{image_id}"
            item = self._tree_items.get(cache_key)
            if item is not None:
                self._select_tree_item(item)
                return

        self._load_image(image_id)
        self._prev_btn.setEnabled(False)
        self._next_btn.setEnabled(False)

    def _fetch_face_data(self) -> None:
        """Reload face list from DB without triggering any UI update."""
        # When pairing is active and current is deoldified, reload from the original
        source_id = self._deol_pair_orig_id or self._current_image_id
        if source_id is None:
            return
        with session_scope() as session:
            # NOTE: crop-path self-heal (ensure_unique_face_crops) intentionally
            # does NOT run here.  It is a maintenance/repair pass that decodes the
            # source image and writes UPDATE faces SET crop_path on the UI thread;
            # running it on every image selection made navigation slow and, while
            # the one-time startup repair task holds the SQLite write lock, caused
            # "database is locked" crashes.  Repair now runs once at startup only
            # (MainWindow._start_crop_repair_task); this path is read-only.
            svc = ImageBrowserService(session)
            self._face_data = svc.get_image_faces(source_id)
            from app.services.unknown_merge_service import UnknownMergeService
            self._pending_face_ids = set(
                UnknownMergeService(session).pending_face_ids(source_id)
            )

    def _reload_current_face_data(self) -> None:
        """Reload face assignments for the current image and refresh the UI."""
        if self._current_image_id is None:
            return
        self._fetch_face_data()
        if self._selected_face_id is not None:
            self._show_face_info(self._selected_face_id)
        self._redraw_faces()
        log.debug("Face data reloaded for image_id=%d", self._current_image_id)

    # ──────────────────────────────────────────────────────────────────
    # Thumbnail callbacks
    # ──────────────────────────────────────────────────────────────────

    @Slot(str, QImage)
    def _on_thumbnail_ready(self, cache_key: str, qimage: QImage) -> None:
        pixmap = QPixmap.fromImage(qimage)
        self._thumb_cache[cache_key] = pixmap
        icon = QIcon(pixmap)
        tree_item = self._tree_items.get(cache_key)
        if tree_item is not None:
            tree_item.setIcon(0, icon)
        search_item = self._search_result_items.get(cache_key)
        if search_item is not None:
            search_item.setIcon(icon)

    # ──────────────────────────────────────────────────────────────────
    # Fullscreen
    # ──────────────────────────────────────────────────────────────────

    def _toggle_fullscreen(self) -> None:
        if self._exit_fs_btn.isVisible():
            self._exit_fullscreen()
        else:
            self._enter_fullscreen()

    def _enter_fullscreen(self) -> None:
        main_win = self.window()
        if main_win is self:
            return
        self._fs_was_maximized = main_win.isMaximized()
        self._fs_hidden_widgets = []
        for w in main_win.findChildren(QToolBar):
            if w.isVisible():
                self._fs_hidden_widgets.append(w)
                w.hide()
        for w in main_win.findChildren(QDockWidget):
            if w.isVisible():
                self._fs_hidden_widgets.append(w)
                w.hide()
        sb = main_win.statusBar()
        if sb and sb.isVisible():
            self._fs_hidden_widgets.append(sb)
            sb.hide()
        tabs = main_win.centralWidget()
        if isinstance(tabs, QTabWidget):
            tabs.tabBar().hide()
        self._fs_btn.setVisible(False)
        self._exit_fs_btn.setVisible(True)
        main_win.showFullScreen()

    def _exit_fullscreen(self) -> None:
        main_win = self.window()
        if main_win is self:
            return
        for w in self._fs_hidden_widgets:
            w.show()
        self._fs_hidden_widgets = []
        tabs = main_win.centralWidget()
        if isinstance(tabs, QTabWidget):
            tabs.tabBar().show()
        self._fs_btn.setVisible(True)
        self._exit_fs_btn.setVisible(False)
        if self._fs_was_maximized:
            main_win.showMaximized()
        else:
            main_win.showNormal()

    # ──────────────────────────────────────────────────────────────────
    # Side panel visibility toggles
    # ──────────────────────────────────────────────────────────────────

    def _on_toggle_left_panel(self, checked: bool) -> None:
        """Show / hide the folder tree column. Image area grows to fill the space."""
        self._tree_panel.setVisible(checked)
        self._hide_inline_editor()
        self._image_label.update()

    def _on_toggle_right_panel(self, checked: bool) -> None:
        """Show / hide the info column. Image area grows to fill the space."""
        self._info_scroll.setVisible(checked)
        self._hide_inline_editor()
        self._image_label.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._hide_inline_editor()
        self._image_label.update()
