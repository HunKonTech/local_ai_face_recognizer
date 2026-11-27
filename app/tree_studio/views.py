"""Tree Studio: new / open / save / rename / duplicate / delete views, JSON files."""

from __future__ import annotations

import json
import logging
from typing import Optional

from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from app.db.database import session_scope
from app.db.models import Person
from app.services.family_tree_service import FamilyTreeService
from app.services.tree_view_service import (
    RootRef,
    TreeViewError,
    TreeViewService,
    TreeViewState,
)
from app.ui.i18n import t

log = logging.getLogger(__name__)


class ViewsMixin:
    """View lifecycle; ``self.state`` is the view being edited."""

    _dirty = False

    def _build_views_menu(self, menu) -> None:  # noqa: ANN001
        self._act_new = menu.addAction("", self.new_view)
        self._act_open = menu.addAction("", self.open_view)
        menu.addSeparator()
        self._act_save = menu.addAction("", self.save_view)
        self._act_save.setShortcut("Ctrl+S")
        self._act_save_as = menu.addAction("", self.save_view_as)
        self._act_rename = menu.addAction("", self.rename_view)
        self._act_duplicate = menu.addAction("", self.duplicate_view)
        self._act_delete = menu.addAction("", self.delete_view)
        menu.addSeparator()
        self._act_export_json = menu.addAction("", self.export_view_json)
        self._act_import_json = menu.addAction("", self.import_view_json)
        menu.addSeparator()
        self._act_export_image = menu.addAction("", self.export_image)

    def _retranslate_views(self) -> None:
        for act, key in (
            (self._act_new, "ts_new_view"), (self._act_open, "ts_open_view"),
            (self._act_save, "ts_save_view"), (self._act_save_as, "ts_save_view_as"),
            (self._act_rename, "ts_rename_view"), (self._act_duplicate, "ts_duplicate_view"),
            (self._act_delete, "ts_delete_view"), (self._act_export_json, "ts_export_json"),
            (self._act_import_json, "ts_import_json"), (self._act_export_image, "ts_export_image"),
        ):
            act.setText(t(key))
        has_view = self.view_id is not None
        for act in (self._act_rename, self._act_duplicate, self._act_delete):
            act.setEnabled(has_view)

    def mark_dirty(self) -> None:
        self._dirty = True

    # ------------------------------------------------------------------
    # Root resolution
    # ------------------------------------------------------------------

    def resolve_root(self, text: str) -> Optional[RootRef]:
        """Person (by name / id) or ``#code#`` → :class:`RootRef`; may ask."""
        text = (text or "").strip()
        if not text:
            return None
        with session_scope() as session:
            if text.startswith("#"):
                pid, candidates = FamilyTreeService(session).find_root_by_external_code(text)
                if pid is not None:
                    return RootRef(pid, text)
                if not candidates:
                    QMessageBox.information(self, t("ts_title"), t("ts_root_missing", text=text))
                    return None
                labels = [f"{name}  ({code})" for _pid, name, code in candidates]
                choice, ok = QInputDialog.getItem(
                    self, t("ts_title"), t("ts_root_choose", code=text), labels, 0, False
                )
                if not ok:
                    return None
                return RootRef(candidates[labels.index(choice)][0], text)
            if text.isdigit() and session.get(Person, int(text)) is not None:
                return RootRef(int(text))
            matches = session.query(Person.id, Person.name).filter(Person.name == text).all()
            if not matches:
                matches = (
                    session.query(Person.id, Person.name)
                    .filter(Person.name.ilike(f"%{text}%"))
                    .order_by(Person.name)
                    .limit(50)
                    .all()
                )
        if not matches:
            QMessageBox.information(self, t("ts_title"), t("ts_root_missing", text=text))
            return None
        if len(matches) == 1:
            return RootRef(matches[0][0])
        labels = [f"{name}  (#{pid})" for pid, name in matches]
        choice, ok = QInputDialog.getItem(self, t("ts_title"), t("ts_new_view_prompt"), labels, 0, False)
        return RootRef(matches[labels.index(choice)][0]) if ok else None

    def _refresh_root_from_code(self) -> None:
        """A view rooted by ``#code#`` follows whoever has that code now."""
        code = self.state.root.external_code
        if not code:
            return
        with session_scope() as session:
            pid, _c = FamilyTreeService(session).find_root_by_external_code(code)
        if pid is not None:
            self.state.root.person_id = pid

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _confirm_discard(self) -> bool:
        if not self._dirty:
            return True
        answer = QMessageBox.question(
            self, t("ts_title"), t("ts_save_view") + "?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save,
        )
        if answer == QMessageBox.Cancel:
            return False
        if answer == QMessageBox.Save:
            return self.save_view()
        return True

    def new_view(self, root_text: Optional[str] = None) -> None:
        if not self._confirm_discard():
            return
        if root_text is None:
            root_text, ok = QInputDialog.getText(self, t("ts_new_view"), t("ts_new_view_prompt"))
            if not ok:
                return
        root = self.resolve_root(root_text)
        if root is None:
            return
        self.set_state(TreeViewState(root=root), view_id=None, name=None)
        self.mark_dirty()

    def set_state(self, state: TreeViewState, *, view_id: Optional[int], name: Optional[str]) -> None:
        self.state = state
        self.view_id = view_id
        self.view_name = name
        self._dirty = False
        self._refresh_root_from_code()
        self._retranslate_views()
        self.rebuild()

    def open_view(self, view_id: Optional[int] = None) -> None:
        if not self._confirm_discard():
            return
        with session_scope() as session:
            svc = TreeViewService(session)
            views = [(v.id, v.name) for v in svc.list_views()]
            if view_id is None:
                if not views:
                    QMessageBox.information(self, t("ts_title"), t("ts_no_views"))
                    return
                names = [n for _i, n in views]
                choice, ok = QInputDialog.getItem(
                    self, t("ts_open_view"), t("ts_choose_view"), names, 0, False
                )
                if not ok:
                    return
                view_id = views[names.index(choice)][0]
            state = svc.load(view_id)
            name = svc.get(view_id).name
        self.set_state(state, view_id=view_id, name=name)

    def save_view(self) -> bool:
        if self.view_id is None:
            return self.save_view_as()
        try:
            with session_scope() as session:
                TreeViewService(session).save(self.view_name, self.state, self.view_id)
        except TreeViewError as exc:
            QMessageBox.warning(self, t("ts_save_view"), str(exc))
            return False
        self._dirty = False
        self._update_title()
        return True

    def save_view_as(self) -> bool:
        name, ok = QInputDialog.getText(
            self, t("ts_save_view_as"), t("ts_view_name_prompt"), text=self.view_name or ""
        )
        if not ok or not name.strip():
            return False
        try:
            with session_scope() as session:
                view = TreeViewService(session).save(name, self.state)
                self.view_id, self.view_name = view.id, view.name
        except TreeViewError:
            QMessageBox.warning(self, t("ts_save_view_as"), t("ts_view_name_taken", name=name))
            return False
        self._dirty = False
        self._retranslate_views()
        self._update_title()
        return True

    def rename_view(self) -> None:
        if self.view_id is None:
            return
        name, ok = QInputDialog.getText(
            self, t("ts_rename_view"), t("ts_view_name_prompt"), text=self.view_name or ""
        )
        if not ok or not name.strip():
            return
        try:
            with session_scope() as session:
                self.view_name = TreeViewService(session).rename(self.view_id, name).name
        except TreeViewError:
            QMessageBox.warning(self, t("ts_rename_view"), t("ts_view_name_taken", name=name))
            return
        self._update_title()

    def duplicate_view(self) -> None:
        if self.view_id is None:
            return
        with session_scope() as session:
            svc = TreeViewService(session)
            svc.save(self.view_name, self.state, self.view_id)
            dup = svc.duplicate(self.view_id)
            dup_id, dup_name = dup.id, dup.name
        self._dirty = False
        self.open_view(dup_id)
        self.view_name = dup_name
        self._update_title()

    def delete_view(self) -> None:
        if self.view_id is None:
            return
        answer = QMessageBox.question(
            self, t("ts_delete_view"), t("ts_delete_view_confirm", name=self.view_name),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        with session_scope() as session:
            TreeViewService(session).delete(self.view_id)
        self.view_id, self.view_name = None, None
        self.mark_dirty()
        self._retranslate_views()
        self._update_title()

    # ------------------------------------------------------------------
    # JSON files
    # ------------------------------------------------------------------

    def export_view_json(self) -> None:
        path, _f = QFileDialog.getSaveFileName(
            self, t("ts_export_json"), f"{self.view_name or 'tree-view'}.json", t("ts_json_filter")
        )
        if not path:
            return
        with session_scope() as session:
            doc = TreeViewService(session).export_document(self.view_name or "View", self.state)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2)

    def import_view_json(self, path: Optional[str] = None) -> None:
        if path is None:
            path, _f = QFileDialog.getOpenFileName(self, t("ts_import_json"), "", t("ts_json_filter"))
            if not path:
                return
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
            with session_scope() as session:
                result = TreeViewService(session).import_document(doc)
        except Exception as exc:  # noqa: BLE001
            log.exception("View import failed")
            QMessageBox.warning(self, t("ts_import_json"), t("ts_import_failed", error=str(exc)))
            return
        msg = t("ts_import_done", name=result.name)
        if result.unmatched:
            names = "\n".join(
                f"• {u.get('name') or '?'} (#{u['id']})" for u in result.unmatched
            )
            msg += "\n\n" + t("ts_import_unmatched", n=len(result.unmatched), names=names)
        QMessageBox.information(self, t("ts_import_json"), msg)
        self._dirty = False
        self.open_view(result.view_id)
