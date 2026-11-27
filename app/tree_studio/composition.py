"""Tree Studio: collapse / expand, hide / show, box fields, person context menu."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
)

from app.db.database import session_scope
from app.db.models import Person
from app.services.tree_view_service import RootRef
from app.ui.i18n import t

log = logging.getLogger(__name__)


class CompositionMixin:
    """Edits ``self.state`` and redraws."""

    def _build_compose_menu(self, menu) -> None:  # noqa: ANN001
        self._act_fields = menu.addAction("", self.edit_view_fields)
        self._act_hidden = menu.addAction("", self.edit_hidden_persons)
        self._act_expand_all = menu.addAction("", self.expand_all)
        self._act_members = menu.addAction("", self._toggle_non_members)
        self._act_members.setCheckable(True)
        from PySide6.QtGui import QActionGroup

        self._friends_menu = menu.addMenu("")
        group = QActionGroup(self)
        self._friend_mode_acts = {}
        for mode in ("off", "all", "selected"):
            act = self._friends_menu.addAction("")
            act.setCheckable(True)
            act.setActionGroup(group)
            act.triggered.connect(lambda _c=False, m=mode: self.set_friend_mode(m))
            self._friend_mode_acts[mode] = act
        menu.addSeparator()
        self._act_zoom_in = menu.addAction("", self.tree.zoom_in)
        self._act_zoom_in.setShortcut("Ctrl++")
        self._act_zoom_out = menu.addAction("", self.tree.zoom_out)
        self._act_zoom_out.setShortcut("Ctrl+-")
        self._act_zoom_reset = menu.addAction("", self.tree.reset_zoom)
        self._act_zoom_reset.setShortcut("Ctrl+0")

    def _retranslate_compose(self) -> None:
        for act, key in (
            (self._act_fields, "ts_fields"), (self._act_hidden, "ts_hidden_persons"),
            (self._act_expand_all, "ts_expand_all"), (self._act_members, "ts_include_non_members"),
            (self._act_zoom_in, "ts_zoom_in"), (self._act_zoom_out, "ts_zoom_out"),
            (self._act_zoom_reset, "ts_zoom_reset"),
        ):
            act.setText(t(key))
        self._act_members.setChecked(self.state.include_non_members)
        self._friends_menu.setTitle(t("ts_friends_menu"))
        for mode, act in self._friend_mode_acts.items():
            act.setText(t(f"ts_friends_{mode}"))
            act.setChecked(self.state.friend_mode == mode)

    def _changed(self) -> None:
        self.mark_dirty()
        self.rebuild(keep_scroll=True)

    # ------------------------------------------------------------------
    # Collapse / hide
    # ------------------------------------------------------------------

    def _coparents(self, pid: int) -> set[int]:
        """Spouses of *pid* who share at least one child with them."""
        node = self.full_graph.node(pid) if self.full_graph is not None else None
        if node is None:
            return set()
        kids = set(node.child_ids)
        out = set()
        for sp in node.spouse_ids:
            snode = self.full_graph.node(sp)
            if snode is not None and kids & set(snode.child_ids):
                out.add(sp)
        return out

    def is_effectively_collapsed(self, pid: int) -> bool:
        """Collapsed itself, or a co-parent spouse is (shared children hidden)."""
        return pid in self.state.collapsed or bool(self._coparents(pid) & self.state.collapsed)

    def toggle_collapse(self, pid: int) -> None:
        if self.is_effectively_collapsed(pid):
            self.state.collapsed.discard(pid)
            self.state.collapsed -= self._coparents(pid)
        else:
            self.state.collapsed.add(pid)
        self._changed()

    def expand_all(self) -> None:
        self.state.collapsed.clear()
        self._changed()

    def hide_person(self, pid: int) -> None:
        self.state.hidden.add(pid)
        self._changed()

    def show_person(self, pid: int) -> None:
        self.state.hidden.discard(pid)
        self._changed()

    def set_friend_mode(self, mode: str) -> None:
        self.state.friend_mode = mode
        for m, act in self._friend_mode_acts.items():
            act.setChecked(m == mode)
        self._changed()

    def friends_shown_for(self, pid: int) -> bool:
        mode = self.state.friend_mode
        return mode == "all" or (mode == "selected" and pid in self.state.friend_anchors)

    def toggle_person_friends(self, pid: int) -> None:
        """Context menu "Show friends" for one anchor (switches to selected mode)."""
        st = self.state
        if self.friends_shown_for(pid):
            if st.friend_mode == "all":
                anchors = set((self.full_graph.friends if self.full_graph else {}) or {})
                st.friend_anchors = (anchors & self.visible) - {pid}
                st.friend_mode = "selected"
            else:
                st.friend_anchors.discard(pid)
        else:
            if st.friend_mode == "off":
                st.friend_anchors = set()
            st.friend_mode = "selected" if st.friend_mode != "all" else "all"
            st.friend_anchors.add(pid)
        self._retranslate_compose()
        self._changed()

    def _toggle_non_members(self) -> None:
        self.state.include_non_members = self._act_members.isChecked()
        self._changed()

    def _names(self, ids) -> dict[int, str]:  # noqa: ANN001
        ids = list(ids)
        if not ids:
            return {}
        with session_scope() as session:
            return dict(session.query(Person.id, Person.name).filter(Person.id.in_(ids)).all())

    def edit_hidden_persons(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(t("ts_hidden_title"))
        layout = QVBoxLayout(dlg)
        lst = QListWidget()
        names = self._names(self.state.hidden)
        for pid in sorted(self.state.hidden, key=lambda p: names.get(p, "")):
            item = QListWidgetItem(names.get(pid, f"#{pid}"))
            item.setData(Qt.UserRole, pid)
            lst.addItem(item)
        layout.addWidget(lst)
        show_btn = QPushButton(t("ts_show_again"))

        def _show() -> None:
            item = lst.currentItem()
            if item is None:
                return
            self.show_person(int(item.data(Qt.UserRole)))
            lst.takeItem(lst.row(item))

        show_btn.clicked.connect(_show)
        lst.itemDoubleClicked.connect(lambda _i: _show())
        layout.addWidget(show_btn)
        box = QDialogButtonBox(QDialogButtonBox.Close)
        box.rejected.connect(dlg.reject)
        layout.addWidget(box)
        dlg.resize(320, 360)
        dlg.exec()

    # ------------------------------------------------------------------
    # Fields
    # ------------------------------------------------------------------

    def edit_view_fields(self) -> None:
        """Pick and order the fields shown under the name in every box."""
        dlg = QDialog(self)
        dlg.setWindowTitle(t("ts_fields_title"))
        layout = QVBoxLayout(dlg)
        hint = QLabel(t("ts_fields_hint"))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        lst = QListWidget()
        lst.setDragDropMode(QAbstractItemView.InternalMove)
        by_key = {f.key: f for f in self._catalog}
        ordered = [k for k in self.state.fields if k in by_key]
        ordered += [f.key for f in self._catalog if f.key not in ordered]
        for key in ordered:
            item = QListWidgetItem(by_key[key].label)
            item.setData(Qt.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if key in self.state.fields else Qt.Unchecked)
            lst.addItem(item)
        layout.addWidget(lst)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(dlg.accept)
        box.rejected.connect(dlg.reject)
        layout.addWidget(box)
        dlg.resize(340, 460)
        if dlg.exec() != QDialog.Accepted:
            return
        self.state.fields = [
            lst.item(i).data(Qt.UserRole)
            for i in range(lst.count())
            if lst.item(i).checkState() == Qt.Checked
        ]
        self._changed()

    def set_person_field(self, pid: int, key: str, visible: bool) -> None:
        self.state.set_person_field(pid, key, visible)
        self._changed()

    # ------------------------------------------------------------------
    # Context menu on a box
    # ------------------------------------------------------------------

    def _on_box_context_menu(self, pid: int, global_pos) -> None:  # noqa: ANN001
        menu = self.build_box_menu(pid)
        menu.exec(global_pos)

    def build_box_menu(self, pid: int) -> QMenu:
        menu = QMenu(self)
        menu.addAction(t("ts_edit_person"), lambda: self.edit_person(pid))
        menu.addAction(t("ts_edit_relationships"), lambda: self.edit_relationships(pid))
        menu.addSeparator()
        node = self.full_graph.node(pid) if self.full_graph is not None else None
        if node is not None and node.child_ids:
            if self.is_effectively_collapsed(pid):
                menu.addAction(t("ts_expand"), lambda: self.toggle_collapse(pid))
            else:
                menu.addAction(t("ts_collapse"), lambda: self.toggle_collapse(pid))
        if pid != self.root_id:
            menu.addAction(t("ts_hide_person"), lambda: self.hide_person(pid))
            menu.addAction(t("ts_set_root"), lambda: self.set_root(pid))
        if pid in self.state.floating:
            menu.addAction(t("ts_connect"), lambda: self.connect_floating(pid))
            menu.addAction(t("ts_remove_floating"), lambda: self.remove_floating(pid))
        friends = (self.full_graph.friends if self.full_graph is not None else {}) or {}
        if friends.get(pid):
            act = menu.addAction(t("ts_show_friends"))
            act.setCheckable(True)
            act.setChecked(self.friends_shown_for(pid))
            act.toggled.connect(lambda _c: self.toggle_person_friends(pid))
        fields_menu = menu.addMenu(t("ts_person_fields"))
        order = [f.key for f in self._catalog]
        current = set(self.state.fields_for(pid, order))
        for spec in self._catalog:
            act = fields_menu.addAction(spec.label)
            act.setCheckable(True)
            act.setChecked(spec.key in current)
            act.toggled.connect(
                lambda checked, k=spec.key: self.set_person_field(pid, k, checked)
            )
        return menu

    def set_root(self, pid: int) -> None:
        self.state.root = RootRef(pid)
        self.state.hidden.discard(pid)
        if pid in self.state.floating:
            self.state.floating.remove(pid)
        self._changed()

    def remove_floating(self, pid: int) -> None:
        if pid in self.state.floating:
            self.state.floating.remove(pid)
            self._changed()

    # ------------------------------------------------------------------
    # Editing persons / relationships (same dialogs as Face-Local)
    # ------------------------------------------------------------------

    def edit_person(self, pid: int) -> None:
        from app.ui.dialogs.person_info_dialog import edit_person_dialog

        if edit_person_dialog(pid, parent=self):
            self.reload_persons()
            self.rebuild(keep_scroll=True)

    def edit_relationships(self, pid: int) -> None:
        from app.ui.dialogs.family_tree_editor_dialog import FamilyTreeEditorDialog

        dialog = FamilyTreeEditorDialog(pid, parent=self)
        dialog.exec()
        self.rebuild(keep_scroll=True)
