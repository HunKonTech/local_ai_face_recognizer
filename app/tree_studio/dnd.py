"""Tree Studio: dropping persons onto the tree (design D7).

* related to the tree (in the unfiltered rooted graph) → reveal in place;
* unrelated, dropped on a box → choose a relationship, create it;
* unrelated, dropped on empty space → floating box with "Connect…".
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QInputDialog, QMenu, QMessageBox

from app.db.database import session_scope
from app.db.models import Person
from app.services.family_service import FamilyService
from app.services.tree_view_service import RootRef, reveal
from app.ui.i18n import t

log = logging.getLogger(__name__)

RELATIONS = ("parent", "child", "spouse", "sibling")


class DropMixin:
    def _on_person_dropped(self, pid: int, target: Optional[int], _scene_pos) -> None:  # noqa: ANN001
        self.handle_drop(pid, target)

    def handle_drop(self, pid: int, target: Optional[int], relation: Optional[str] = None) -> str:
        """Apply a drop; returns what happened (for tests / logging).

        ``relation`` skips the menu (tests); ``None`` asks the user.
        """
        if self.root_id is None and not self.visible:
            # Empty view: the dropped person becomes the root.
            self.state.root = RootRef(pid)
            self._changed()
            return "root"
        if self.full_graph is not None and pid in self.full_graph.nodes:
            reveal(self.full_graph, self.state, pid, self.root_id)
            self._selected = pid
            self._changed()
            return "revealed"
        if target is None or target == pid:
            if pid not in self.state.floating:
                self.state.floating.append(pid)
            self.state.hidden.discard(pid)
            self._changed()
            return "floating"
        relation = relation or self._ask_relation(pid, target)
        if relation is None:
            return "cancelled"
        if not self.create_relationship(pid, target, relation):
            return "failed"
        if pid in self.state.floating:
            self.state.floating.remove(pid)
        self.state.hidden.discard(pid)
        self._changed()
        return relation

    def target_has_parents(self, target: int) -> bool:
        node = self.full_graph.node(target) if self.full_graph is not None else None
        return bool(node and node.parent_ids)

    def available_relations(self, target: int) -> list[str]:
        return [r for r in RELATIONS if r != "sibling" or self.target_has_parents(target)]

    def _ask_relation(self, pid: int, target: int) -> Optional[str]:
        names = self._names([pid, target])
        menu = QMenu(self)
        chosen: dict[str, str] = {}
        for rel in self.available_relations(target):
            act = menu.addAction(
                t(f"ts_rel_{rel}", u=names.get(pid, "?"), a=names.get(target, "?"))
            )
            act.triggered.connect(lambda _c=False, r=rel: chosen.setdefault("r", r))
        menu.exec(QCursor.pos())
        return chosen.get("r")

    def create_relationship(self, pid: int, target: int, relation: str) -> bool:
        """Write *pid* ↔ *target* in the shared DB (``pid`` is the new person)."""
        try:
            with session_scope() as session:
                fam = FamilyService(session)
                if relation == "parent":
                    fam.add_parent_child(pid, target)
                elif relation == "child":
                    fam.add_parent_child(target, pid)
                elif relation == "spouse":
                    fam.add_spouse(target, pid)
                elif relation == "sibling":
                    parents = self.full_graph.node(target).parent_ids if self.full_graph else ()
                    if not parents:
                        raise ValueError("target has no parents")
                    for parent in parents:
                        fam.add_parent_child(parent, pid)
                else:
                    raise ValueError(relation)
        except Exception as exc:  # noqa: BLE001
            log.exception("Tree Studio relationship failed")
            QMessageBox.warning(self, t("ts_title"), t("ts_rel_failed", error=str(exc)))
            return False
        log.info("Tree Studio: %s %s of %s", pid, relation, target)
        return True

    def connect_floating(self, pid: int) -> None:
        """Link a floating person to a chosen visible person."""
        candidates = [p for p in self.visible if p != pid and p not in self.state.floating]
        if not candidates:
            return
        names = self._names(candidates + [pid])
        ordered = sorted(candidates, key=lambda p: names.get(p, ""))
        labels = [f"{names.get(p, '?')}  (#{p})" for p in ordered]
        choice, ok = QInputDialog.getItem(
            self, t("ts_connect"), t("ts_connect_choose", u=names.get(pid, "?")), labels, 0, False
        )
        if not ok:
            return
        target = ordered[labels.index(choice)]
        rel_labels = {
            t(f"ts_rel_{r}", u=names.get(pid, "?"), a=names.get(target, "?")): r
            for r in self.available_relations(target)
        }
        rel_label, ok = QInputDialog.getItem(
            self, t("ts_connect"), t("ts_connect"), list(rel_labels), 0, False
        )
        if ok:
            self.handle_drop(pid, target, rel_labels[rel_label])

    @staticmethod
    def person_exists(pid: int) -> bool:
        with session_scope() as session:
            return session.get(Person, pid) is not None
