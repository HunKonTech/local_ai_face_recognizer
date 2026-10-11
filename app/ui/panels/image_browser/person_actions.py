"""Rename, recent persons, assign editor and person info."""

from __future__ import annotations

import logging
from typing import Dict

from app.db.database import session_scope
from app.db.models import Person
from app.services.family_service import FamilyService
from app.services.identity_service import IdentityService
from app.ui.dialogs.person_info_dialog import PersonInfoDialog

log = logging.getLogger(__name__)


class PersonActionsMixin:
    """Rename, recent persons, assign editor and person info."""

    # ──────────────────────────────────────────────────────────────────
    # Inline rename
    # ──────────────────────────────────────────────────────────────────

    def _start_rename(self) -> None:
        if self._selected_face_id is None:
            return
        entry = next(
            (f for f in self._face_data if f[0] == self._selected_face_id), None
        )
        if entry is None or not entry[5]:
            return
        self._renaming = True
        self._person_name_label.setVisible(False)
        self._rename_btn.setVisible(False)
        self._rename_edit.setText(entry[5])
        self._rename_edit.setVisible(True)
        self._rename_edit.setFocus()
        self._rename_edit.selectAll()

    def _commit_rename(self) -> None:
        if not self._renaming:
            return
        self._renaming = False
        new_name = self._rename_edit.text().strip()
        self._rename_edit.setVisible(False)
        if new_name:
            self._save_person_rename(new_name)
        if self._selected_face_id is not None:
            entry = next(
                (f for f in self._face_data if f[0] == self._selected_face_id), None
            )
            if entry and entry[5]:
                self._person_name_label.setVisible(True)
                self._rename_btn.setVisible(True)
                self._person_info_btn.setVisible(True)

    def _cancel_rename(self) -> None:
        if not self._renaming:
            return
        self._renaming = False
        self._rename_edit.setVisible(False)
        self._rename_edit.clear()
        if self._selected_face_id is not None:
            entry = next(
                (f for f in self._face_data if f[0] == self._selected_face_id), None
            )
            if entry and entry[5]:
                self._person_name_label.setVisible(True)
                self._rename_btn.setVisible(True)
                self._person_info_btn.setVisible(True)

    def _save_person_rename(self, new_name: str) -> None:
        if self._selected_face_id is None:
            return
        entry = next(
            (f for f in self._face_data if f[0] == self._selected_face_id), None
        )
        if entry is None:
            return
        person_id = entry[6]
        if person_id is None:
            return
        with session_scope() as session:
            IdentityService(session).rename_person(person_id, new_name)
        log.info("Person %d renamed to %r", person_id, new_name)
        self._reload_current_face_data()
        self._reload_persons_combo()
        self.person_data_changed.emit()

    # ──────────────────────────────────────────────────────────────────
    # Person combo
    # ──────────────────────────────────────────────────────────────────

    def _reload_persons_combo(self) -> None:
        # Refreshes the person list consumed by the inline assign editor (the
        # single assignment surface — it is (re)populated on every open).
        with session_scope() as session:
            self._all_persons = session.query(Person).order_by(Person.name).all()

    def _recent_person_rank(self, session) -> Dict[int, int]:
        rank: Dict[int, int] = {}
        for person_id in self._recent_assignment_person_ids:
            if person_id not in rank:
                rank[person_id] = len(rank)
        return rank

    def _remember_recent_person(self, person_id: int) -> None:
        self._recent_assignment_person_ids = [
            person_id,
            *(pid for pid in self._recent_assignment_person_ids if pid != person_id),
        ][:20]

    # ──────────────────────────────────────────────────────────────────
    # Assign / create actions (info panel buttons)
    # ──────────────────────────────────────────────────────────────────

    def _open_assign_editor(self) -> None:
        """Open the single shared assign popup (inline editor) for the selected face.

        The info panel used to embed its own person search + create-new-person
        controls; assignment now always goes through the same context-menu
        style editor as right-clicking a face, so there is exactly one UI (with
        match percentages and background scoring) everywhere.

        The panel button can be pressed while the image label is still in
        interactive-edit mode; dismiss that first so the popup anchors cleanly
        and the next face click is not consumed by the stale edit-exit handler.
        """
        if self._selected_face_id is None:
            return
        if self._interactive_edit_face_id is not None:
            self._interactive_edit_face_id = None
            self._interactive_edit_orig_bbox = None
            self._draw_hint.setVisible(False)
            self._image_label.dismiss_interactive_edit()
        self._show_inline_editor(self._selected_face_id)

    def _open_person_info_dialog(self, person_id: int) -> None:
        with session_scope() as session:
            person = session.get(Person, person_id)
            if person is None:
                return
            dlg = PersonInfoDialog(person, parent=self)
            if dlg.exec() != PersonInfoDialog.Accepted:
                return
            person.gender = dlg.gender()
            person.family_code = dlg.family_code() or None
            person.external_family_code = dlg.external_family_code() or None
            person.name_prefix = dlg.name_prefix() or None
            person.last_name = dlg.last_name() or None
            person.first_name = dlg.first_name() or None
            person.second_name = dlg.second_name() or None
            person.nickname = dlg.nickname() or None
            person.married_name = dlg.married_name() or None
            person.birth_place = dlg.birth_place() or None
            person.birth_date = dlg.birth_date() or None
            person.death_date = dlg.death_date() or None
            person.death_place = dlg.death_place() or None
            person.notes = dlg.notes() or None
            session.flush()
            try:
                FamilyService(session).link_derived_parents(person.id)
            except Exception:
                log.exception("Could not link derived parents for person %d", person.id)
        log.info("Person details saved: person_id=%d", person_id)
        self._reload_persons_combo()
        self.person_data_changed.emit()

    def _on_person_info(self) -> None:
        if self._selected_face_id is None:
            return
        entry = next(
            (f for f in self._face_data if f[0] == self._selected_face_id), None
        )
        if entry is None or entry[6] is None:
            return
        self._open_person_info_dialog(entry[6])
