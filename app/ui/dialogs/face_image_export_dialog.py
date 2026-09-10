"""Dialog for exporting the faces of selected images into separate files (#175).

Everything lives in this one window — pattern, crop settings and destination —
so the user never faces a stack of modal prompts.  Settings are remembered in
``QSettings`` under the ``face_export/`` prefix.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.app_settings import app_qsettings
from app.db.database import session_scope
from app.services.face_image_export_service import (
    DEFAULT_PATTERN,
    MODE_ORIGINAL,
    MODE_SQUARE,
    FaceImageExportOptions,
    FaceImageExportService,
)
from app.services.filename_pattern import available_tokens, unknown_tokens
from app.ui.i18n import t

log = logging.getLogger(__name__)

_S_PATTERN = "face_export/pattern"
_S_DIR = "face_export/target_dir"
_S_MODE = "face_export/mode"
_S_MARGIN = "face_export/margin_percent"
_S_SQUARE = "face_export/square_size"
_S_QUALITY = "face_export/jpeg_quality"
_S_UNKNOWN = "face_export/include_unknown"
_S_EXCLUDED = "face_export/skip_excluded"


class FaceImageExportDialog(QDialog):
    """Collects :class:`FaceImageExportOptions` for a set of images.

    Args:
        image_ids:  Images whose faces will be exported.
        face_ids:   Optional restriction to individual faces.
        parent:     Parent widget.
    """

    def __init__(
        self,
        image_ids: Sequence[int],
        face_ids: Optional[Sequence[int]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._image_ids = [int(i) for i in dict.fromkeys(image_ids)]
        self._face_ids = tuple(int(f) for f in face_ids) if face_ids else None
        self.setWindowTitle(t("fexp_title"))
        self.setMinimumWidth(560)
        self._build_ui()
        self._load_settings()
        self._refresh_preview()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._header = QLabel("")
        self._header.setWordWrap(True)
        layout.addWidget(self._header)

        # --- Filename pattern -----------------------------------------
        pat_box = QGroupBox(t("fexp_pattern_group"))
        pat_layout = QVBoxLayout(pat_box)

        self._pattern_edit = QLineEdit()
        self._pattern_edit.setPlaceholderText(DEFAULT_PATTERN)
        self._pattern_edit.textChanged.connect(self._refresh_preview)
        pat_layout.addWidget(self._pattern_edit)

        self._preview = QLabel("")
        self._preview.setWordWrap(True)
        self._preview.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._preview.setStyleSheet("color: #aaa; font-size: 11px;")
        pat_layout.addWidget(self._preview)

        self._warning = QLabel("")
        self._warning.setWordWrap(True)
        self._warning.setStyleSheet("color: #d9534f; font-size: 11px;")
        self._warning.setVisible(False)
        pat_layout.addWidget(self._warning)

        hint = QLabel(t("fexp_token_hint"))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #aaa; font-size: 11px;")
        pat_layout.addWidget(hint)

        self._token_table = self._build_token_table()
        pat_layout.addWidget(self._token_table)
        layout.addWidget(pat_box)

        # --- Crop geometry --------------------------------------------
        crop_box = QGroupBox(t("fexp_crop_group"))
        crop_layout = QVBoxLayout(crop_box)

        self._mode_original = QRadioButton(t("fexp_mode_original"))
        self._mode_original.setToolTip(t("fexp_mode_original_tip"))
        self._mode_square = QRadioButton(t("fexp_mode_square"))
        self._mode_square.setToolTip(t("fexp_mode_square_tip"))
        self._mode_original.setChecked(True)
        crop_layout.addWidget(self._mode_original)
        crop_layout.addWidget(self._mode_square)

        form = QFormLayout()
        self._square_spin = QSpinBox()
        self._square_spin.setRange(64, 4096)
        self._square_spin.setSingleStep(64)
        self._square_spin.setSuffix(" px")
        self._square_spin.setValue(512)
        form.addRow(t("fexp_square_size"), self._square_spin)

        self._margin_spin = QSpinBox()
        self._margin_spin.setRange(0, 200)
        self._margin_spin.setSuffix(" %")
        self._margin_spin.setValue(30)
        form.addRow(t("fexp_margin"), self._margin_spin)

        self._quality_spin = QSpinBox()
        self._quality_spin.setRange(50, 100)
        self._quality_spin.setValue(92)
        form.addRow(t("fexp_quality"), self._quality_spin)
        crop_layout.addLayout(form)

        self._mode_square.toggled.connect(self._square_spin.setEnabled)
        self._square_spin.setEnabled(False)
        layout.addWidget(crop_box)

        # --- Which faces ----------------------------------------------
        sel_box = QGroupBox(t("fexp_faces_group"))
        sel_layout = QVBoxLayout(sel_box)
        self._include_unknown = QCheckBox(t("fexp_include_unknown"))
        self._include_unknown.setChecked(True)
        self._include_unknown.toggled.connect(self._refresh_preview)
        self._skip_excluded = QCheckBox(t("fexp_skip_excluded"))
        self._skip_excluded.setChecked(True)
        self._skip_excluded.toggled.connect(self._refresh_preview)
        sel_layout.addWidget(self._include_unknown)
        sel_layout.addWidget(self._skip_excluded)
        if self._face_ids:
            note = QLabel(t("fexp_single_face_note"))
            note.setStyleSheet("color: #aaa; font-size: 11px;")
            sel_layout.addWidget(note)
        layout.addWidget(sel_box)

        # --- Destination ----------------------------------------------
        dest_box = QGroupBox(t("fexp_dest_group"))
        dest_layout = QHBoxLayout(dest_box)
        self._dir_label = QLineEdit()
        self._dir_label.setReadOnly(True)
        self._dir_label.setPlaceholderText(t("fexp_dest_placeholder"))
        browse = QPushButton(f"📁  {t('export_choose_folder')}")
        browse.clicked.connect(self._on_browse)
        dest_layout.addWidget(self._dir_label, 1)
        dest_layout.addWidget(browse)
        layout.addWidget(dest_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self
        )
        buttons.button(QDialogButtonBox.Ok).setText(t("fexp_start"))
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_token_table(self) -> QTableWidget:
        specs = available_tokens()
        table = QTableWidget(len(specs), 2, self)
        table.setHorizontalHeaderLabels(
            [t("fexp_col_token"), t("fexp_col_meaning")]
        )
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.setMaximumHeight(190)
        table.setToolTip(t("fexp_token_table_tip"))
        for row, spec in enumerate(specs):
            token_item = QTableWidgetItem(spec.display)
            token_item.setData(Qt.UserRole, spec.display)
            table.setItem(row, 0, token_item)
            table.setItem(row, 1, QTableWidgetItem(t(spec.i18n_key)))
        table.doubleClicked.connect(self._on_token_double_clicked)
        return table

    # ------------------------------------------------------------------
    # Behaviour
    # ------------------------------------------------------------------

    def _on_token_double_clicked(self, index) -> None:  # noqa: ANN001 — QModelIndex
        item = self._token_table.item(index.row(), 0)
        if item is None:
            return
        self._pattern_edit.insert(item.text())
        self._pattern_edit.setFocus()

    def _on_browse(self) -> None:
        start = self._dir_label.text() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self, t("export_choose_folder"), start
        )
        if folder:
            self._dir_label.setText(folder)

    def _refresh_preview(self) -> None:
        pattern = self._pattern_edit.text().strip() or DEFAULT_PATTERN

        bad = unknown_tokens(pattern)
        if bad:
            self._warning.setText(
                t("fexp_unknown_tokens", tokens=", ".join(f"#{b}#" for b in bad))
            )
            self._warning.setVisible(True)
        else:
            self._warning.setVisible(False)

        options = self._options(pattern=pattern, target_dir="preview")
        try:
            with session_scope() as session:
                service = FaceImageExportService(session)
                self._face_count = service.count_faces(
                    self._image_ids,
                    include_unknown=options.include_unknown,
                    skip_excluded=options.skip_excluded,
                    only_face_ids=options.only_face_ids,
                )
                names: List[str] = service.preview_names(
                    self._image_ids, options, limit=3
                )
        except Exception:  # noqa: BLE001 — preview must never break the dialog
            log.exception("Face export preview failed")
            self._face_count = 0
            names = []

        self._header.setText(
            t(
                "fexp_header",
                images=len(self._image_ids),
                faces=self._face_count,
            )
        )
        if names:
            self._preview.setText(
                t("fexp_preview") + "\n" + "\n".join(f"• {n}" for n in names)
            )
        else:
            self._preview.setText(t("fexp_preview_empty"))

    def _options(
        self,
        pattern: Optional[str] = None,
        target_dir: Optional[str] = None,
    ) -> FaceImageExportOptions:
        return FaceImageExportOptions(
            pattern=pattern if pattern is not None else self._pattern_edit.text().strip(),
            target_dir=target_dir if target_dir is not None else self._dir_label.text(),
            margin_percent=self._margin_spin.value(),
            mode=MODE_SQUARE if self._mode_square.isChecked() else MODE_ORIGINAL,
            square_size=self._square_spin.value(),
            jpeg_quality=self._quality_spin.value(),
            include_unknown=self._include_unknown.isChecked(),
            skip_excluded=self._skip_excluded.isChecked(),
            only_face_ids=self._face_ids,
        )

    def options(self) -> FaceImageExportOptions:
        """The settings the user confirmed."""
        opts = self._options()
        if not opts.pattern:
            opts.pattern = DEFAULT_PATTERN
        return opts

    def _on_accept(self) -> None:
        if not self._dir_label.text():
            QMessageBox.warning(self, t("fexp_title"), t("fexp_need_folder"))
            return
        if getattr(self, "_face_count", 0) <= 0:
            QMessageBox.warning(self, t("fexp_title"), t("fexp_no_faces"))
            return
        self._save_settings()
        self.accept()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        qs = app_qsettings()
        self._pattern_edit.setText(str(qs.value(_S_PATTERN, DEFAULT_PATTERN)))
        self._dir_label.setText(str(qs.value(_S_DIR, "") or ""))
        square = str(qs.value(_S_MODE, MODE_ORIGINAL)) == MODE_SQUARE
        self._mode_square.setChecked(square)
        self._mode_original.setChecked(not square)
        self._square_spin.setEnabled(square)
        self._margin_spin.setValue(int(qs.value(_S_MARGIN, 30, type=int)))
        self._square_spin.setValue(int(qs.value(_S_SQUARE, 512, type=int)))
        self._quality_spin.setValue(int(qs.value(_S_QUALITY, 92, type=int)))
        self._include_unknown.setChecked(bool(qs.value(_S_UNKNOWN, True, type=bool)))
        self._skip_excluded.setChecked(bool(qs.value(_S_EXCLUDED, True, type=bool)))

    def _save_settings(self) -> None:
        qs = app_qsettings()
        opts = self._options()
        qs.setValue(_S_PATTERN, opts.pattern or DEFAULT_PATTERN)
        qs.setValue(_S_DIR, opts.target_dir)
        qs.setValue(_S_MODE, opts.mode)
        qs.setValue(_S_MARGIN, opts.margin_percent)
        qs.setValue(_S_SQUARE, opts.square_size)
        qs.setValue(_S_QUALITY, opts.jpeg_quality)
        qs.setValue(_S_UNKNOWN, opts.include_unknown)
        qs.setValue(_S_EXCLUDED, opts.skip_excluded)
