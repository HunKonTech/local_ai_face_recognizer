"""Universal tree search and place filter."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import (
    Qt,
    QThreadPool,
)
from PySide6.QtGui import (
    QIcon,
)
from PySide6.QtWidgets import (
    QListWidgetItem,
)

from app.db.database import session_scope
from app.services.family_service import FamilyImageSearchCriteria, FamilyService
from app.ui.i18n import t
from app.workers.thumbnail_worker import ThumbnailRunnable

from .helpers import (
    _TREE_THUMB_SIZE,
)

log = logging.getLogger(__name__)


class SearchMixin:
    """Universal tree search and place filter."""

    # ── Universal tree search ─────────────────────────────────────────

    def _tree_suggest(self, query: str) -> list[tuple[str, str, str]]:
        """Provide autocomplete suggestions for the image-browser search bar."""
        try:
            with session_scope() as session:
                return FamilyService(session).get_search_suggestions(query)
        except Exception:
            log.exception("ImageBrowserPanel: tree search autocomplete error")
            return []

    def _on_tree_search_changed(self) -> None:
        tokens = self._tree_search_bar.get_tokens()
        if not tokens:
            self._search_results_list.hide()
            self._file_tree.show()
            self._tree_hdr_lbl.setText(t("ib3_folders_hdr"))
            return
        self._run_tree_search(tokens)

    def _run_tree_search(self, tokens) -> None:
        from app.ui.widgets.universal_search_bar import (
            TOKEN_DATE,
            TOKEN_FAMILY_CODE,
            TOKEN_IMAGE,
            TOKEN_NICKNAME,
            TOKEN_PERSON,
            TOKEN_PLACE,
        )
        name_terms: list[str] = []
        any_terms: list[str] = []
        place_text = ""
        photo_date = ""
        image_text = ""

        for tok in tokens:
            if tok.token_type in (TOKEN_PERSON, TOKEN_NICKNAME, TOKEN_FAMILY_CODE):
                name_terms.append(tok.value)
            elif tok.token_type == TOKEN_PLACE:
                if not place_text:
                    place_text = tok.value
                else:
                    any_terms.append(tok.value)
            elif tok.token_type == TOKEN_DATE:
                if not photo_date:
                    photo_date = tok.value
                else:
                    any_terms.append(tok.value)
            elif tok.token_type == TOKEN_IMAGE:
                if not image_text:
                    image_text = tok.value
                else:
                    any_terms.append(tok.value)
            else:
                any_terms.append(tok.value)

        criteria = FamilyImageSearchCriteria(
            name_terms=tuple(name_terms),
            allow_other_people=True,
            image_text=image_text,
            photo_date=photo_date,
            place_text=place_text,
            any_terms=tuple(any_terms),
        )
        log.debug("ImageBrowserPanel tree search criteria: %r", criteria)

        try:
            with session_scope() as session:
                results, total = FamilyService(session).search_images_by_criteria(
                    criteria, limit=200
                )
        except Exception:
            log.exception("ImageBrowserPanel: tree search error")
            results, total = [], 0

        self._search_results_list.clear()
        self._search_result_items.clear()
        pool = QThreadPool.globalInstance()
        placeholder_icon = self._make_placeholder_icon()
        for result in results:
            item = QListWidgetItem(result.filename)
            item.setData(Qt.UserRole, result.image_id)
            item.setToolTip(result.file_path)
            item.setIcon(placeholder_icon)
            self._search_results_list.addItem(item)

            # Async thumbnail — never decode the full-resolution image on the
            # GUI thread (that froze the UI with many large photos).  Reuse the
            # shared thumbnail cache and the same worker pattern as the tree.
            cache_key = f"thumb_{result.image_id}"
            self._search_result_items[cache_key] = item
            cached = self._thumb_cache.get(cache_key)
            if cached is not None:
                item.setIcon(QIcon(cached))
            elif Path(result.file_path).exists():
                worker = ThumbnailRunnable(
                    file_path=result.file_path,
                    cache_key=cache_key,
                    size=_TREE_THUMB_SIZE,
                )
                worker.signals.ready.connect(self._on_thumbnail_ready)
                worker.signals.failed.connect(self._on_thumbnail_failed)
                pool.start(worker)

        if not results:
            placeholder = QListWidgetItem(t("ibp_search_no_results"))
            placeholder.setFlags(Qt.NoItemFlags)
            self._search_results_list.addItem(placeholder)

        self._tree_hdr_lbl.setText(t("ibp_search_results_hdr", n=total))
        self._file_tree.hide()
        self._search_results_list.show()

    def _on_search_result_double_clicked(self, item: QListWidgetItem) -> None:
        image_id = item.data(Qt.UserRole)
        if image_id is not None:
            self._load_image(int(image_id))

    # ── (legacy placeholder — no longer exposed in UI) ────────────────

    def _apply_place_filter_by_name(self) -> None:
        pass

    def _clear_place_filter(self) -> None:
        self._place_filter_id = None
        self._current_tree_item = None
        self.refresh()
