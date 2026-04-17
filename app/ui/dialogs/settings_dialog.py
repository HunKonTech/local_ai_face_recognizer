"""Settings dialog — language and database management."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from app.ui.i18n import SUPPORTED, current_language, set_language, t


class SettingsDialog(QDialog):
    """Settings dialog with language and database management tabs."""

    def __init__(self, current_db_path: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("settings_title"))
        self.setMinimumWidth(500)
        self._current_db_path = current_db_path
        self._new_db_path: Optional[str] = None
        self._language_changed = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        # ── Language ──────────────────────────────────────────────────────
        lang_group = QGroupBox(t("lang_label").rstrip(":"))
        lang_layout = QFormLayout(lang_group)

        self._lang_combo = QComboBox()
        for code, name in SUPPORTED.items():
            self._lang_combo.addItem(name, userData=code)

        # Select current
        idx = self._lang_combo.findData(current_language())
        if idx >= 0:
            self._lang_combo.setCurrentIndex(idx)

        lang_layout.addRow(t("lang_label"), self._lang_combo)
        layout.addWidget(lang_group)

        # ── Database ──────────────────────────────────────────────────────
        db_group = QGroupBox(t("db_group"))
        db_layout = QVBoxLayout(db_group)

        current_row = QHBoxLayout()
        current_row.addWidget(QLabel(t("current_db")))
        self._db_label = QLineEdit(self._current_db_path)
        self._db_label.setReadOnly(True)
        self._db_label.setStyleSheet("color: #aaa;")
        current_row.addWidget(self._db_label)
        db_layout.addLayout(current_row)

        btn_row = QHBoxLayout()
        self._new_db_btn = QPushButton(t("new_db"))
        self._new_db_btn.clicked.connect(self._on_new_db)
        btn_row.addWidget(self._new_db_btn)

        self._open_db_btn = QPushButton(t("open_db"))
        self._open_db_btn.clicked.connect(self._on_open_db)
        btn_row.addWidget(self._open_db_btn)
        btn_row.addStretch()
        db_layout.addLayout(btn_row)

        layout.addWidget(db_group)

        # ── Buttons ───────────────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    # ------------------------------------------------------------------

    def _on_new_db(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            t("db_new_title"),
            str(Path.home() / "faces.db"),
            "SQLite (*.db *.sqlite)",
        )
        if path:
            self._new_db_path = path
            self._db_label.setText(path)

    def _on_open_db(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            t("db_open_title"),
            str(Path.home()),
            "SQLite (*.db *.sqlite);;All files (*)",
        )
        if path:
            self._new_db_path = path
            self._db_label.setText(path)

    def _on_accept(self) -> None:
        selected_lang = self._lang_combo.currentData()
        if selected_lang != current_language():
            set_language(selected_lang)
            self._language_changed = True
        self.accept()

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------

    def selected_db_path(self) -> Optional[str]:
        """Return new DB path if the user changed it, else None."""
        return self._new_db_path

    def language_changed(self) -> bool:
        return self._language_changed
