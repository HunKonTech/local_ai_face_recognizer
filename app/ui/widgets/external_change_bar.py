"""Non-modal bar announcing family-data changes made by another app window.

Reusable by Face-Local and Tree Studio (``add-db-change-notification``).
"""

from __future__ import annotations

from typing import Mapping, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QToolButton, QWidget

from app.ui.i18n import t

# Display order of change kinds in the summary.
_KIND_ORDER = ("person", "relationship", "field", "view")


def summarize_changes(counts: Mapping[str, int]) -> str:
    """``{"person": 3, "relationship": 1}`` -> "3 persons and 1 relationship"."""
    parts = []
    for kind in sorted(counts, key=lambda k: (_KIND_ORDER.index(k) if k in _KIND_ORDER else 99, k)):
        n = int(counts[kind])
        if n <= 0:
            continue
        form = "one" if n == 1 else "many"
        key = f"ext_kind_{kind}_{form}"
        text = t(key, n=n)
        if text == key:  # unknown kind: generic wording
            text = t(f"ext_kind_other_{form}", n=n)
        parts.append(text)
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return t("ext_list_join", head=", ".join(parts[:-1]), last=parts[-1])


class ExternalChangeBar(QFrame):
    """Summary text + Refresh + dismiss. Hidden until :meth:`show_changes`."""

    refresh_requested = Signal()
    dismissed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ExternalChangeBar")
        self.setStyleSheet(
            "#ExternalChangeBar { background: #fff4d6; border-bottom: 1px solid #e0c070; }"
            "#ExternalChangeBar QLabel { color: #5a4300; }"
        )
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 4, 6, 4)
        self._label = QLabel()
        self._label.setWordWrap(True)
        row.addWidget(self._label, stretch=1)
        self._refresh_btn = QPushButton()
        self._refresh_btn.clicked.connect(self._on_refresh)
        row.addWidget(self._refresh_btn)
        self._close_btn = QToolButton()
        self._close_btn.setText("✕")
        self._close_btn.clicked.connect(self._on_dismiss)
        row.addWidget(self._close_btn)
        self._counts: dict[str, int] = {}
        self.retranslate()
        self.setVisible(False)

    def show_changes(self, counts: Mapping[str, int]) -> None:
        self._counts = {k: int(v) for k, v in counts.items() if int(v) > 0}
        self._label.setText(t("ext_changes_msg", items=summarize_changes(self._counts)))
        self.setVisible(bool(self._counts))

    def text(self) -> str:
        return self._label.text()

    def retranslate(self) -> None:
        self._refresh_btn.setText(t("ext_refresh_btn"))
        self._close_btn.setToolTip(t("ext_dismiss_tip"))
        if self._counts:
            self._label.setText(t("ext_changes_msg", items=summarize_changes(self._counts)))

    def _on_refresh(self) -> None:
        self.setVisible(False)
        self.refresh_requested.emit()

    def _on_dismiss(self) -> None:
        self.setVisible(False)
        self._counts = {}
        self.dismissed.emit()
