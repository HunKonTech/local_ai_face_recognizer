"""A QLabel that elides its text instead of forcing its parent window wide.

A plain QLabel reports the full text width as its minimum size, so one long
status line is enough to stop the whole window from being resized smaller.
This label shrinks to nothing, elides what does not fit and keeps the full
text in its tooltip.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget


class ElidingLabel(QLabel):
    """Label whose text is elided to the width it actually gets."""

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        mode: Qt.TextElideMode = Qt.ElideRight,
    ) -> None:
        super().__init__(text, parent)
        self._full_text = text
        self._elide_mode = mode
        policy = self.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Ignored)
        self.setSizePolicy(policy)

    # -- QLabel overrides ------------------------------------------------

    def setText(self, text: str) -> None:  # noqa: N802
        self._full_text = text or ""
        self.setToolTip(self._full_text)
        self._apply_elide()

    def text(self) -> str:
        return self._full_text

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_elide()

    # -- internals -------------------------------------------------------

    def _apply_elide(self) -> None:
        metrics = QFontMetrics(self.font())
        width = max(0, self.width())
        super().setText(metrics.elidedText(self._full_text, self._elide_mode, width))
