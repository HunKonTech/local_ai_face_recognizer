"""Tree Studio main window: a composed family-tree view on the shared DB.

``TreeStudioWindow(…Mixins, QMainWindow)``; mixins share state via ``self``:
``views`` (save / load / JSON), ``persons_dock`` (search list, drag source),
``composition`` (collapse, hide, fields, context menu), ``dnd`` (drops,
relationships), ``export`` (images), ``external`` (change notice).
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from app.db.database import session_scope
from app.services.family_tree_service import FamilyTreeNode, FamilyTreeService
from app.services.tree_view_service import (
    TreeViewState,
    apply_friends,
    collapsible_ids,
    filter_graph,
    visible_ids,
)
from app.tree_studio.composition import CompositionMixin
from app.tree_studio.dnd import DropMixin
from app.tree_studio.export import ExportMixin
from app.tree_studio.external import StudioExternalChangesMixin
from app.tree_studio.persons_dock import PersonsDockMixin
from app.tree_studio.views import ViewsMixin
from app.ui.i18n import t
from app.ui.widgets.family_tree import FamilyTreeView

log = logging.getLogger(__name__)


class TreeStudioWindow(
    ViewsMixin,
    PersonsDockMixin,
    CompositionMixin,
    DropMixin,
    ExportMixin,
    StudioExternalChangesMixin,
    QMainWindow,
):
    """Compose, save and export family-tree views."""

    def __init__(self, config=None) -> None:  # noqa: ANN001
        super().__init__()
        self._config = config
        self.state = TreeViewState()
        self.view_id: Optional[int] = None
        self.view_name: Optional[str] = None
        self.root_id: Optional[int] = None
        self.full_graph = None  # unfiltered rooted graph (relatedness, reveal)
        self.visible: set[int] = set()
        self._catalog = []  # FieldSpec list
        self._values: dict[int, dict[str, str]] = {}
        self._selected: Optional[int] = None

        self.tree = FamilyTreeView()
        self.tree.set_line_provider(self._lines_for)
        self.tree.person_selected.connect(self._on_tree_selected)
        self.tree.person_activated.connect(self.edit_person)
        self.tree.person_context_menu.connect(self._on_box_context_menu)
        self.tree.collapse_toggled.connect(self.toggle_collapse)
        self.tree.person_dropped.connect(self._on_person_dropped)

        self._empty_hint = QLabel()
        self._empty_hint.setWordWrap(True)
        self._empty_hint.setStyleSheet("color: #888; padding: 8px;")

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._central_layout = layout
        layout.addWidget(self._empty_hint)
        layout.addWidget(self.tree, stretch=1)
        self.setCentralWidget(central)

        self._build_menus()
        self._build_persons_dock()
        self._setup_external_changes(layout)
        self.resize(1280, 820)
        self._retranslate()
        self.rebuild()

    # ------------------------------------------------------------------
    # Menus + translation
    # ------------------------------------------------------------------

    def _build_menus(self) -> None:
        mb = self.menuBar()
        self._view_menu = mb.addMenu("")
        self._build_views_menu(self._view_menu)
        self._compose_menu = mb.addMenu("")
        self._build_compose_menu(self._compose_menu)
        self._lang_menu = mb.addMenu("")
        self._lang_en = self._lang_menu.addAction("English", lambda: self._set_language("en"))
        self._lang_hu = self._lang_menu.addAction("Magyar", lambda: self._set_language("hu"))

    def _set_language(self, lang: str) -> None:
        from app.ui.i18n import set_language

        set_language(lang)
        self._retranslate()
        self.rebuild(keep_scroll=True)

    def _retranslate(self) -> None:
        self._view_menu.setTitle(t("ts_menu_view"))
        self._compose_menu.setTitle(t("ts_menu_compose"))
        self._lang_menu.setTitle(t("ts_menu_language"))
        self._retranslate_views()
        self._retranslate_compose()
        self._retranslate_persons_dock()
        self._retranslate_external()
        self._empty_hint.setText(t("ts_no_root"))
        self._update_title()

    def _update_title(self) -> None:
        name = self.view_name or t("ts_unsaved")
        self.setWindowTitle(t("ts_title_view", name=name))

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------

    def rebuild(self, *, keep_scroll: bool = False) -> None:
        """Re-read the DB and redraw the composed view."""
        hbar, vbar = self.tree.horizontalScrollBar(), self.tree.verticalScrollBar()
        scroll = (hbar.value(), vbar.value())
        from app.services.family_tree_service import FamilyTreeGraph
        from app.services.person_field_service import PersonFieldService

        graph = FamilyTreeGraph(root_id=None)
        try:
            with session_scope() as session:
                svc = FamilyTreeService(session)
                root = self.state.root.person_id
                if root is not None:
                    try:
                        graph = svc.build_tree(
                            root, include_non_members=self.state.include_non_members
                        )
                    except Exception:  # noqa: BLE001 - root deleted meanwhile
                        log.warning("Tree Studio root %s no longer exists", root)
                        root = None
                self.root_id = root
                self.full_graph = graph
                visible = visible_ids(graph, self.state, root) if root is not None else set()
                shown = filter_graph(graph, {p for p in visible if p in graph.nodes})
                self._add_floating_nodes(session, shown, graph)
                apply_friends(
                    shown, graph, self.state.friend_mode,
                    self.state.friend_anchors, self.state.hidden,
                )
                field_svc = PersonFieldService(session)
                self._catalog = field_svc.catalog()
                self._values = field_svc.values_for(shown.nodes)
        except Exception:  # noqa: BLE001
            log.exception("Tree Studio rebuild failed")
            shown = FamilyTreeGraph(root_id=None)
        self.visible = set(shown.nodes)
        collapsible = collapsible_ids(graph) & self.visible
        shown_collapsed = {p for p in collapsible if self.is_effectively_collapsed(p)}
        self.tree.set_collapse_state(collapsible, shown_collapsed)
        selected = self._selected if self._selected in self.visible else None
        self.tree.set_graph(shown, focus_id=self.root_id, selected_id=selected)
        if keep_scroll:
            hbar.setValue(scroll[0])
            vbar.setValue(scroll[1])
        self._empty_hint.setVisible(not self.visible)
        self._refresh_person_markers()
        self._update_title()

    def _add_floating_nodes(self, session, shown, graph) -> None:  # noqa: ANN001
        """Floating persons not in the rooted graph get a stand-alone node."""
        from app.db.models import Person

        root_gen = 0
        for pid in self.state.floating:
            if pid in self.state.hidden or pid in shown.nodes:
                continue
            node = graph.node(pid)
            if node is None:
                person = session.get(Person, pid)
                if person is None:
                    continue
                node = FamilyTreeNode(
                    person_id=pid,
                    name=person.name,
                    generation=root_gen,
                    is_member=bool(person.is_family_tree_member),
                    gender=person.gender,
                    family_code=person.family_code,
                    birth_date=person.birth_date,
                    death_date=person.death_date,
                    thumbnail_path=person.thumbnail_path,
                )
            shown.nodes[pid] = node

    def _lines_for(self, node) -> list[tuple[str, str]]:  # noqa: ANN001
        """Name plus the view's fields resolved for this person (design D5)."""
        lines = [(node.name, "name")]
        values = self._values.get(node.person_id, {})
        order = [f.key for f in self._catalog]
        for key in self.state.fields_for(node.person_id, order):
            value = values.get(key)
            if value:
                lines.append((value, "line"))
        return lines

    def _on_tree_selected(self, pid: int) -> None:
        self._selected = pid
        self.tree.set_selected(pid)

    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: ANN001, N802
        if not self._confirm_discard():
            event.ignore()
            return
        super().closeEvent(event)
