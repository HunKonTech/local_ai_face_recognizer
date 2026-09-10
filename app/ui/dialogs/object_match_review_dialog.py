"""Review dialog for object-matching hits (#164).

A hit is never turned into an object marking on its own — the user sees the
target image with the proposed frame drawn on it and decides.  Accepting marks
the object there and gives the matcher another reference sample; rejecting is
remembered so the same pairing is never proposed again.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.services.object_matching_service import ObjectMatchingService, SuggestionInfo
from app.ui.i18n import t
from app.ui.widgets.object_gallery_widget import crop_pixmap_full_with_frame

log = logging.getLogger(__name__)

_CARD_W = 340
_CARD_H = 240


class _SuggestionCard(QFrame):
    """One hit: the target image with the proposed frame, plus a verdict."""

    def __init__(self, info: SuggestionInfo, parent: "ObjectMatchReviewDialog") -> None:
        super().__init__(parent)
        self._info = info
        self._dialog = parent
        self.setFrameShape(QFrame.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumHeight(_CARD_H)
        pixmap = crop_pixmap_full_with_frame(
            info.image_path or "", info.bbox, _CARD_W, _CARD_H
        )
        if pixmap is not None:
            self._preview.setPixmap(pixmap)
        layout.addWidget(self._preview)

        self._caption = QLabel()
        self._caption.setWordWrap(True)
        layout.addWidget(self._caption)

        buttons = QHBoxLayout()
        self._accept_btn = QPushButton()
        self._accept_btn.clicked.connect(self._on_accept)
        self._reject_btn = QPushButton()
        self._reject_btn.clicked.connect(self._on_reject)
        buttons.addWidget(self._accept_btn)
        buttons.addWidget(self._reject_btn)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.retranslate()

    @property
    def info(self) -> SuggestionInfo:
        return self._info

    def retranslate(self) -> None:
        self._accept_btn.setText(t("object_match_accept"))
        self._reject_btn.setText(t("object_match_reject"))
        self._caption.setText(
            f"<b>{self._info.object_name}</b><br>"
            f"{t('object_match_score')}: {round(self._info.score * 100)}% &nbsp;·&nbsp; "
            f"{t('object_match_size')}: {round(self._info.scale * 100)}%"
        )

    def _on_accept(self) -> None:
        self._dialog.decide(self._info.suggestion_id, accept=True)

    def _on_reject(self) -> None:
        self._dialog.decide(self._info.suggestion_id, accept=False)


class ObjectMatchReviewDialog(QDialog):
    """Grid of pending hits with per-item and bulk verdicts.

    Args:
        object_id: Limit the review to one object; ``None`` reviews everything
            pending, which is what the library-wide batch mode produces.
        run_id: Limit the review to one search run.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        object_id: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self._object_id = object_id
        self._run_id = run_id
        self.accepted_count = 0
        self.rejected_count = 0
        self._cards: List[_SuggestionCard] = []

        self.setMinimumSize(820, 600)
        self._build_ui()
        self.reload()

    # -- construction ----------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._intro = QLabel()
        self._intro.setWordWrap(True)
        layout.addWidget(self._intro)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._grid_host = QWidget()
        self._grid = QVBoxLayout(self._grid_host)
        self._grid.setAlignment(Qt.AlignTop)
        self._scroll.setWidget(self._grid_host)
        layout.addWidget(self._scroll, 1)

        self._empty = QLabel()
        self._empty.setAlignment(Qt.AlignCenter)
        self._empty.setVisible(False)
        layout.addWidget(self._empty)

        # Bulk row: the threshold and its action sit together, so the slider is
        # the confirmation and no extra prompt is needed.
        bulk = QHBoxLayout()
        self._threshold_label = QLabel()
        self._threshold = QSlider(Qt.Horizontal)
        self._threshold.setRange(0, 100)
        self._threshold.setValue(80)
        self._threshold.setFixedWidth(180)
        self._threshold.valueChanged.connect(self._retranslate_threshold)
        self._accept_above_btn = QPushButton()
        self._accept_above_btn.clicked.connect(self._on_accept_above)
        bulk.addWidget(self._threshold_label)
        bulk.addWidget(self._threshold)
        bulk.addWidget(self._accept_above_btn)
        bulk.addStretch(1)
        layout.addLayout(bulk)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._retranslate()

    def _retranslate(self) -> None:
        self.setWindowTitle(t("object_match_review_title"))
        self._intro.setText(t("object_match_review_intro"))
        self._empty.setText(t("object_match_empty"))
        close = self._buttons.button(QDialogButtonBox.Close)
        if close is not None:
            close.setText(t("close"))
        self._retranslate_threshold()
        for card in self._cards:
            card.retranslate()

    def _retranslate_threshold(self) -> None:
        self._threshold_label.setText(f"{t('object_match_score')} ≥")
        self._accept_above_btn.setText(
            f"{t('object_match_accept_above')} {self._threshold.value()}%"
        )

    # -- data ------------------------------------------------------------

    def reload(self) -> None:
        """Rebuild the card list from the pending suggestions."""
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards = []

        with session_scope() as session:
            pending = ObjectMatchingService(session).list_pending(
                object_id=self._object_id, run_id=self._run_id
            )
        for info in pending:
            card = _SuggestionCard(info, self)
            self._grid.addWidget(card)
            self._cards.append(card)

        has_items = bool(self._cards)
        self._scroll.setVisible(has_items)
        self._empty.setVisible(not has_items)
        self._accept_above_btn.setEnabled(has_items)

    def decide(self, suggestion_id: int, accept: bool) -> None:
        """Record one verdict and drop its card."""
        try:
            with session_scope() as session:
                service = ObjectMatchingService(session)
                if accept:
                    service.accept_suggestion(suggestion_id)
                else:
                    service.reject_suggestion(suggestion_id)
        except ValueError:
            log.warning("Object match suggestion %s vanished", suggestion_id)
            self.reload()
            return

        if accept:
            self.accepted_count += 1
        else:
            self.rejected_count += 1
        self._remove_card(suggestion_id)

    def _remove_card(self, suggestion_id: int) -> None:
        for card in list(self._cards):
            if card.info.suggestion_id != suggestion_id:
                continue
            self._cards.remove(card)
            card.setParent(None)
            card.deleteLater()
        has_items = bool(self._cards)
        self._scroll.setVisible(has_items)
        self._empty.setVisible(not has_items)
        self._accept_above_btn.setEnabled(has_items)

    def _on_accept_above(self) -> None:
        threshold = self._threshold.value() / 100.0
        with session_scope() as session:
            accepted = ObjectMatchingService(session).accept_above(
                threshold, object_id=self._object_id
            )
        self.accepted_count += accepted
        self.reload()

    # -- Qt --------------------------------------------------------------

    def changeEvent(self, event) -> None:  # noqa: N802 — Qt naming
        super().changeEvent(event)
        if event.type() == event.Type.LanguageChange:
            self._retranslate()
