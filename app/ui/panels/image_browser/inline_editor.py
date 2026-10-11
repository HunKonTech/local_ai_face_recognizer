"""Inline person assign editor, pending-action bar and face info panel."""

from __future__ import annotations

import logging

from PySide6.QtCore import (
    QThreadPool,
    Slot,
)
from PySide6.QtWidgets import (
    QMessageBox,
)

from app.db.database import session_scope
from app.db.models import Person
from app.services.unknown_merge_service import UnknownMergeService
from app.ui.i18n import t

log = logging.getLogger(__name__)


class InlineEditorMixin:
    """Inline person assign editor, pending-action bar and face info panel."""

    # ──────────────────────────────────────────────────────────────────
    # Inline face editor
    # ──────────────────────────────────────────────────────────────────

    def _start_match_scoring(self, face_id: int, *, had_person: bool) -> None:
        """Score *face_id* against known people in the background.

        Scoring deserialises many embeddings and is slow on a large database, so
        it runs in a thread-pool worker instead of the GUI thread — otherwise
        selecting a face freezes the app.  The inline editor opens instantly in
        name order; :meth:`_on_match_scores_ready` re-ranks it when the worker
        finishes, but only if the editor still shows this same face.
        """
        from app.workers.match_score_worker import MatchScoreRunnable

        recognition_cfg = (
            getattr(self._config, "recognition", None) if self._config else None
        )
        # Newest request wins; a stale worker result is discarded on arrival.
        self._match_score_preselect = not had_person
        # One worker per face is enough: its result is applied to every surface
        # showing the face (info panel + inline editor) by _on_match_scores_ready.
        if self._match_score_inflight == face_id:
            return
        self._match_score_inflight = face_id
        worker = MatchScoreRunnable(face_id, recognition_cfg, self._config)
        worker.signals.ready.connect(self._on_match_scores_ready)
        # On failure deliver an empty score set so the "computing similarity…"
        # hint in the picker is cleared instead of sticking forever.
        worker.signals.failed.connect(self._on_match_scores_failed)
        QThreadPool.globalInstance().start(worker)

    @Slot(int, object)
    def _on_match_scores_ready(self, face_id: int, scores: object) -> None:
        """Apply background-computed match scores if the inline editor still
        shows this face; stale results are discarded."""
        if self._match_score_inflight == face_id:
            self._match_score_inflight = None
        scores = scores or {}
        # Inline editor popup — the single assignment surface: apply only while
        # it still shows this same face.
        if (
            self._inline_editor_face_id == face_id
            and self._inline_editor.isVisible()
        ):
            self._inline_editor.apply_late_match_scores(
                scores, preselect_best=self._match_score_preselect
            )

    @Slot(int)
    def _on_match_scores_failed(self, face_id: int) -> None:
        """Clear the picker's "computing…" state when background scoring fails."""
        self._on_match_scores_ready(face_id, {})

    def _show_inline_editor(self, face_id: int) -> None:
        entry = next((f for f in self._face_data if f[0] == face_id), None)
        if entry is None:
            log.debug("_show_inline_editor: face_id=%d not found in face_data", face_id)
            return
        # The inline person picker and the pending action bar must not overlap.
        self._pending_bar.hide()
        _, bx, by, bw, bh, person_name, person_id, *_ = entry

        # Open instantly in name order; match scores are computed off-thread and
        # applied by _on_match_scores_ready so selecting a face never blocks the
        # GUI (heavy on a large database).
        self._inline_editor.populate(
            person_name,
            person_id,
            self._all_persons,
            priority_ids=list(self._recent_assignment_person_ids),
            match_scores=None,
            scores_pending=True,
        )
        self._inline_editor.adjustSize()

        lx_top, ly_top = self._image_to_label(bx, by)
        _, ly_bot = self._image_to_label(bx, by + bh)

        # _image_to_label returns (-1, -1) only when _full_pixmap is None.
        # For off-screen faces (pan/zoom) the values can be negative — clamp
        # rather than bailing out, so the editor still appears after a right-click.
        if self._full_pixmap is None:
            log.debug("_show_inline_editor: no pixmap, aborting")
            return
        lx_top = max(0, lx_top)
        ly_top = max(0, ly_top)
        if ly_bot < 0:
            ly_bot = ly_top + int(bh * min(
                self._image_label.contentsRect().width() / self._full_pixmap.width(),
                self._image_label.contentsRect().height() / self._full_pixmap.height(),
            ) * self._zoom)

        log.debug(
            "_show_inline_editor: face_id=%d person=%r label_top=(%d,%d) label_bot_y=%d",
            face_id, person_name, lx_top, ly_top, ly_bot,
        )

        editor_w = self._inline_editor.width()
        editor_h = self._inline_editor.sizeHint().height()
        cr_w = self._image_label.contentsRect().width()
        cr_h = self._image_label.contentsRect().height()

        if ly_top - editor_h - 8 >= 0:
            pos_y = ly_top - editor_h - 8
        elif ly_bot >= 0 and ly_bot + editor_h + 8 <= cr_h:
            pos_y = ly_bot + 8
        else:
            pos_y = max(4, ly_top - editor_h // 2)

        pos_x = max(4, min(lx_top, cr_w - editor_w - 4))
        pos_y = max(4, min(pos_y, cr_h - editor_h - 4))

        self._inline_editor.move(pos_x, pos_y)
        self._inline_editor.show()
        self._inline_editor.raise_()
        self._inline_editor_face_id = face_id
        # Set the face id first, then kick off background scoring: its result is
        # delivered via the event loop and _on_match_scores_ready matches it
        # against this id, so the ordering only updates while this face is shown.
        self._start_match_scoring(face_id, had_person=person_id is not None)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._inline_editor.focus_search)

    def _hide_inline_editor(self) -> None:
        self._inline_editor.hide()
        self._inline_editor_face_id = None

    # ──────────────────────────────────────────────────────────────────
    # Pending (auto-merged-from-Unknown) review actions
    # ──────────────────────────────────────────────────────────────────

    def _position_pending_action_bar(self) -> None:
        """Show the quick-action bar over the selected pending face, else hide it."""
        fid = self._selected_face_id
        if (
            fid is None
            or fid not in self._pending_face_ids
            or self._full_pixmap is None
        ):
            self._pending_bar.hide()
            return
        entry = next((f for f in self._face_data if f[0] == fid), None)
        if entry is None:
            self._pending_bar.hide()
            return
        _, bx, by, bw, bh, *_ = entry
        self._pending_bar.adjustSize()
        bar_w = self._pending_bar.width()
        bar_h = self._pending_bar.height()
        cr = self._image_label.contentsRect()

        lx, ly_top = self._image_to_label(bx, by)
        _, ly_bot = self._image_to_label(bx, by + bh)
        # Prefer just above the box; fall back to just below, then clamp.
        if ly_top - bar_h - 6 >= 0:
            pos_y = ly_top - bar_h - 6
        elif ly_bot + bar_h + 6 <= cr.height():
            pos_y = ly_bot + 6
        else:
            pos_y = max(2, ly_top + 2)
        pos_x = max(2, min(lx, cr.width() - bar_w - 2))
        self._pending_bar.move(pos_x, pos_y)
        self._pending_bar.show()
        self._pending_bar.raise_()

    def _notify_unknown_merge(self, result) -> None:
        """Inform the user when an assignment scattered an Unknown cluster."""
        if result is None or not getattr(result, "was_unknown_cluster", False):
            return
        moved = result.n_pending + result.n_auto_confirmed
        if moved <= 0:
            return
        QMessageBox.information(
            self,
            t("amerge_notice_title"),
            t(
                "amerge_notice_msg",
                source=result.source_person_name or "?",
                target=result.target_name,
                pending=result.n_pending,
                auto=result.n_auto_confirmed,
            ),
        )

    def _on_pending_accept(self) -> None:
        fid = self._selected_face_id
        if fid is None:
            return
        with session_scope() as session:
            UnknownMergeService(session).confirm_auto_merge(fid)
        self._reload_current_face_data()
        self.person_data_changed.emit()

    def _on_pending_move(self) -> None:
        fid = self._selected_face_id
        if fid is None:
            return
        # Reuse the inline editor: the face is already on a named person, so
        # picking a target goes through reassign_face, which clears the markers.
        self._show_inline_editor(fid)

    def _on_pending_delete(self) -> None:
        fid = self._selected_face_id
        if fid is None:
            return
        reply = QMessageBox.question(
            self,
            t("remove_face_title"),
            t("remove_face_msg"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._pending_bar.hide()
        self._delete_face(fid)
        self.person_data_changed.emit()

    def _on_inline_assign(self, person_id: int) -> None:
        face_id = self._inline_editor_face_id
        if face_id is None:
            return
        log.info(
            "Face assignment: face_id=%d → person_id=%d (inline editor)",
            face_id, person_id,
        )
        self._inline_editor.reset_search()
        self._hide_inline_editor()
        with session_scope() as session:
            result = UnknownMergeService(
                session, self._config.recognition if self._config else None
            ).assign_unknown_face(face_id, person_id)
        self._remember_recent_person(person_id)
        self._reload_current_face_data()
        self._reload_persons_combo()
        self.person_data_changed.emit()
        self._image_label.setFocus()
        self._notify_unknown_merge(result)

    def _on_inline_create(self, name: str) -> None:
        face_id = self._inline_editor_face_id
        if face_id is None:
            return
        self._hide_inline_editor()

        # Safety: exit any stale interactive edit state.  Normally the inline
        # editor can only be opened when _imode is False, but defensive cleanup
        # prevents _imode from leaking into the post-creation state.
        if self._interactive_edit_face_id is not None:
            log.warning(
                "New person (inline): stale interactive-edit detected for "
                "face_id=%d — dismissing silently",
                self._interactive_edit_face_id,
            )
            self._interactive_edit_face_id = None
            self._interactive_edit_orig_bbox = None
            self._draw_hint.setVisible(False)
            self._image_label.dismiss_interactive_edit()

        with session_scope() as session:
            person = Person(name=name, is_auto_named=False)
            session.add(person)
            session.flush()
            person_id = person.id
            log.info(
                "New person creation: face_id=%d new_person='%s' person_id=%d",
                face_id, name, person_id,
            )
            result = UnknownMergeService(
                session, self._config.recognition if self._config else None
            ).assign_unknown_face(face_id, person_id)
            log.info(
                "Face assignment: face_id=%d → person_id=%d (new person via inline)",
                face_id, person_id,
            )
        self._remember_recent_person(person_id)
        self._reload_current_face_data()
        self._reload_persons_combo()
        self.person_data_changed.emit()
        self._image_label.setFocus()
        self._notify_unknown_merge(result)
        self._open_person_info_dialog(person_id)

    # ──────────────────────────────────────────────────────────────────
    # Face info panel
    # ──────────────────────────────────────────────────────────────────

    def _show_face_info(self, face_id: int) -> None:
        entry = next((f for f in self._face_data if f[0] == face_id), None)
        if entry is None:
            return
        _, _, _, _, _, person_name, person_id, *_ = entry
        self.active_person_changed.emit(person_name)
        self._assign_open_btn.setEnabled(True)
        if person_name:
            self._face_status_label.setText(t("ibp_identified"))
            self._person_name_label.setText(person_name)
            self._person_name_label.setVisible(True)
            self._rename_btn.setVisible(True)
            self._person_info_btn.setVisible(True)
        else:
            self._face_status_label.setText(t("ibp_not_identified"))
            self._person_name_label.setVisible(False)
            self._rename_btn.setVisible(False)
            self._person_info_btn.setVisible(False)
        self._rename_edit.setVisible(False)

    def _clear_face_panel(self) -> None:
        self._cancel_rename()
        if self._detection_done and not self._face_data:
            self._face_status_label.setText(t("ibp_no_face_detected"))
            self._face_status_label.setStyleSheet(
                "color: #ffaa44; font-size: 11px; font-style: italic;"
            )
        else:
            self._face_status_label.setText(t("ibp_click_face"))
            self._face_status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._person_name_label.setVisible(False)
        self._rename_btn.setVisible(False)
        self._person_info_btn.setVisible(False)
        self._assign_open_btn.setEnabled(False)
        self.active_person_changed.emit(None)
