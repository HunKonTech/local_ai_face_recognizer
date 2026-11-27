"""Composed family-tree views for Tree Studio (``add-tree-studio-app``).

A view is a :class:`TreeViewState` (root, collapsed / hidden / floating
persons, displayed fields, per-person field overrides, options). The pure
:func:`visible_ids` decides who is drawn; :class:`TreeViewService` stores
views in ``family_tree_views`` and exchanges them as JSON files.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Person, SavedTreeView

log = logging.getLogger(__name__)

FORMAT = "facelocal.treeview.v1"

# Default box lines in Tree Studio: dates + family code (name is always shown).
DEFAULT_FIELDS = ["b:birth_date", "b:death_date", "b:family_code"]


class TreeViewError(ValueError):
    """Invalid view operation (empty or duplicate name, unknown view)."""


@dataclass
class RootRef:
    """A view root: a person id or an external family code like ``#Zoli#``."""

    person_id: Optional[int] = None
    external_code: Optional[str] = None

    def to_json(self) -> dict:
        return {"personId": self.person_id, "externalCode": self.external_code}

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "RootRef":
        data = data or {}
        pid = data.get("personId")
        return cls(int(pid) if pid is not None else None, data.get("externalCode") or None)


@dataclass
class TreeViewState:
    root: RootRef = field(default_factory=RootRef)
    collapsed: set[int] = field(default_factory=set)
    hidden: set[int] = field(default_factory=set)
    floating: list[int] = field(default_factory=list)
    fields: list[str] = field(default_factory=lambda: list(DEFAULT_FIELDS))
    # pid -> {"show": set[key], "hide": set[key]}
    person_fields: dict[int, dict[str, set[str]]] = field(default_factory=dict)
    include_non_members: bool = True
    options: dict[str, Any] = field(default_factory=dict)
    # Friend links from family codes: "off" | "all" | "selected" (anchors).
    friend_mode: str = "off"
    friend_anchors: set[int] = field(default_factory=set)

    # -- per-person field resolution ----------------------------------

    def fields_for(self, pid: int, catalog_order: Iterable[str]) -> list[str]:
        """View fields minus the person's ``hide`` plus their ``show``.

        Order: the view's field order, then extra shown fields in catalog order.
        """
        override = self.person_fields.get(pid, {})
        hide = override.get("hide", set())
        show = override.get("show", set())
        out = [k for k in self.fields if k not in hide]
        extra = [k for k in catalog_order if k in show and k not in out]
        return out + extra

    def set_person_field(self, pid: int, key: str, visible: bool) -> None:
        """Override one field for one person (removes a redundant override)."""
        ov = self.person_fields.setdefault(pid, {"show": set(), "hide": set()})
        ov.setdefault("show", set())
        ov.setdefault("hide", set())
        in_view = key in self.fields
        ov["show"].discard(key)
        ov["hide"].discard(key)
        if visible and not in_view:
            ov["show"].add(key)
        elif not visible and in_view:
            ov["hide"].add(key)
        if not ov["show"] and not ov["hide"]:
            self.person_fields.pop(pid, None)

    # -- serialisation --------------------------------------------------

    def to_json(self) -> dict:
        return {
            "format": FORMAT,
            "root": self.root.to_json(),
            "collapsed": sorted(self.collapsed),
            "hidden": sorted(self.hidden),
            "floating": list(self.floating),
            "fields": list(self.fields),
            "personFields": {
                str(pid): {"show": sorted(ov.get("show", ())), "hide": sorted(ov.get("hide", ()))}
                for pid, ov in sorted(self.person_fields.items())
            },
            "includeNonMembers": self.include_non_members,
            "options": dict(self.options),
            "friendMode": self.friend_mode,
            "friendAnchors": sorted(self.friend_anchors),
        }

    @classmethod
    def from_json(cls, data: dict) -> "TreeViewState":
        if data.get("format", FORMAT) != FORMAT:
            raise TreeViewError(f"unsupported view format {data.get('format')!r}")
        return cls(
            root=RootRef.from_json(data.get("root")),
            collapsed={int(x) for x in data.get("collapsed", [])},
            hidden={int(x) for x in data.get("hidden", [])},
            floating=[int(x) for x in data.get("floating", [])],
            fields=[str(k) for k in data.get("fields", DEFAULT_FIELDS)],
            person_fields={
                int(pid): {"show": set(ov.get("show", [])), "hide": set(ov.get("hide", []))}
                for pid, ov in (data.get("personFields") or {}).items()
            },
            include_non_members=bool(data.get("includeNonMembers", True)),
            options=dict(data.get("options") or {}),
            friend_mode=_friend_mode(data.get("friendMode")),
            friend_anchors={int(x) for x in data.get("friendAnchors", [])},
        )

    def referenced_ids(self) -> set[int]:
        ids = (
            set(self.collapsed) | set(self.hidden) | set(self.floating)
            | set(self.person_fields) | set(self.friend_anchors)
        )
        if self.root.person_id is not None:
            ids.add(self.root.person_id)
        return ids

    def remap(self, mapping: dict[int, Optional[int]]) -> "TreeViewState":
        """Copy with person ids replaced; ids mapped to ``None`` are dropped."""

        def m(pid: int) -> Optional[int]:
            return mapping.get(pid, pid)

        root = RootRef(
            m(self.root.person_id) if self.root.person_id is not None else None,
            self.root.external_code,
        )
        floating: list[int] = []
        for pid in self.floating:
            new = m(pid)
            if new is not None and new not in floating:
                floating.append(new)
        return TreeViewState(
            root=root,
            collapsed={x for x in (m(p) for p in self.collapsed) if x is not None},
            hidden={x for x in (m(p) for p in self.hidden) if x is not None},
            floating=floating,
            fields=list(self.fields),
            person_fields={
                new: {k: set(v) for k, v in ov.items()}
                for pid, ov in self.person_fields.items()
                if (new := m(pid)) is not None
            },
            include_non_members=self.include_non_members,
            options=dict(self.options),
            friend_mode=self.friend_mode,
            friend_anchors={x for x in (m(p) for p in self.friend_anchors) if x is not None},
        )

    def drop_unknown(self, existing: set[int]) -> "TreeViewState":
        """Copy without persons that no longer exist (deleted since the save)."""
        return self.remap({pid: None for pid in self.referenced_ids() if pid not in existing})


# ---------------------------------------------------------------------------
# Visibility (pure, unit-tested with hand-built graphs)
# ---------------------------------------------------------------------------


def visible_ids(graph, state: TreeViewState, root_id: Optional[int] = None) -> set[int]:
    """Persons drawn for *state* on the unfiltered rooted *graph*.

    BFS from the root that never enters ``hidden`` persons and does not follow
    child edges out of ``collapsed`` persons (nor to children shared with a
    collapsed spouse); parent and spouse edges are always followed (spouses of
    a collapsed person stay). Persons reachable
    along another path stay visible. Floating persons are added on top.
    """
    root = root_id if root_id is not None else getattr(graph, "root_id", None)
    seen: set[int] = set()
    if root is not None and root in graph.nodes and root not in state.hidden:
        seen.add(root)
        queue: deque[int] = deque([root])
        while queue:
            node = graph.node(queue.popleft())
            if node is None:
                continue
            nexts = list(node.parent_ids) + list(node.spouse_ids)
            if node.person_id not in state.collapsed:
                # A couple's shared children are collapsed when either partner
                # is: do not reach them through the other (co-parent) spouse.
                couple = {node.person_id, *node.spouse_ids}
                for child in node.child_ids:
                    cnode = graph.node(child)
                    cparents = set(cnode.parent_ids) if cnode is not None else set()
                    if not (cparents & couple & state.collapsed):
                        nexts.append(child)
            for nxt in nexts:
                if nxt in seen or nxt in state.hidden or nxt not in graph.nodes:
                    continue
                seen.add(nxt)
                queue.append(nxt)
    seen.update(pid for pid in state.floating if pid not in state.hidden)
    return seen


def collapsible_ids(graph) -> set[int]:
    """Persons who have children in the full graph (they get a +/- badge)."""
    return {pid for pid, node in graph.nodes.items() if node.child_ids}


def reveal(graph, state: TreeViewState, pid: int, root_id: Optional[int] = None) -> bool:
    """Make *pid* visible: un-hide it and expand collapsed / hidden ancestors.

    Walks the shortest path from the root to *pid* in the unfiltered graph and
    clears every collapse / hide on it. Returns ``False`` when *pid* is not
    reachable from the root (unrelated).
    """
    root = root_id if root_id is not None else getattr(graph, "root_id", None)
    if root is None or pid not in graph.nodes or root not in graph.nodes:
        return False
    prev: dict[int, Optional[int]] = {root: None}
    queue: deque[int] = deque([root])
    while queue and pid not in prev:
        node = graph.node(queue.popleft())
        for nxt in list(node.parent_ids) + list(node.spouse_ids) + list(node.child_ids):
            if nxt in graph.nodes and nxt not in prev:
                prev[nxt] = node.person_id
                queue.append(nxt)
    if pid not in prev:
        return False
    cur: Optional[int] = pid
    while cur is not None:
        state.hidden.discard(cur)
        parent = prev[cur]
        if parent is not None:
            pnode = graph.node(parent)
            if pnode is not None and cur in pnode.child_ids:
                state.collapsed.discard(parent)
        cur = parent
    if pid in state.floating:
        state.floating.remove(pid)
    return True


FRIEND_MODES = ("off", "all", "selected")


def _friend_mode(value) -> str:  # noqa: ANN001
    return value if value in FRIEND_MODES else "off"


def visible_friends(
    graph, mode: str, anchors_filter: Iterable[int], visible: set[int], hidden: Iterable[int] = ()
) -> dict[int, list[int]]:
    """``{friend: [anchors]}`` drawn for *mode* (design D3).

    Only visible anchors that pass the mode count; hidden friends are left out.
    """
    if mode not in ("all", "selected"):
        return {}
    allowed = set(anchors_filter)
    hidden = set(hidden)
    out: dict[int, list[int]] = {}
    for anchor in sorted(getattr(graph, "friends", {}) or {}):
        if anchor not in visible or (mode == "selected" and anchor not in allowed):
            continue
        for friend in graph.friends[anchor]:
            if friend in hidden:
                continue
            out.setdefault(friend, []).append(anchor)
    return out


def apply_friends(shown, source, mode: str, anchors_filter=(), hidden=()) -> dict[int, list[int]]:  # noqa: ANN001
    """Add friend edges and satellite boxes to *shown* (filtered graph).

    *source* is the graph the friends were derived on (``friends`` and
    ``friend_nodes``). A friend already in *shown* is only linked; others get
    one satellite box beside their first anchor (by id), in its row.
    """
    from dataclasses import replace

    links = visible_friends(source, mode, anchors_filter, set(shown.nodes), hidden)
    shown.friend_edges = []
    shown.satellites = {}
    attach_end: dict[int, int] = {}  # anchor -> last box chained beside it
    for friend in sorted(links):
        anchors = links[friend]
        if friend not in shown.nodes:
            template = source.friend_nodes.get(friend) or source.node(friend)
            if template is None:
                continue
            first = anchors[0]
            anchor_node = shown.node(first)
            shown.nodes[friend] = replace(
                template,
                generation=anchor_node.generation,
                parent_ids=(), child_ids=(), spouse_ids=(),
            )
            shown.satellites[friend] = attach_end.get(first, first)
            attach_end[first] = friend
        for anchor in anchors:
            shown.friend_edges.append((friend, anchor))
    return links


def filter_graph(graph, keep: set[int]):
    """Copy of *graph* restricted to *keep* (edge lists trimmed)."""
    from dataclasses import replace

    from app.services.family_tree_service import FamilyTreeGraph

    out = FamilyTreeGraph(
        root_id=graph.root_id,
        marriages=dict(graph.marriages),
        friends=dict(getattr(graph, "friends", {}) or {}),
        friend_nodes=dict(getattr(graph, "friend_nodes", {}) or {}),
    )
    for pid in keep:
        node = graph.node(pid)
        if node is None:
            continue
        out.nodes[pid] = replace(
            node,
            parent_ids=tuple(p for p in node.parent_ids if p in keep),
            child_ids=tuple(c for c in node.child_ids if c in keep),
            spouse_ids=tuple(s for s in node.spouse_ids if s in keep),
        )
    return out


# ---------------------------------------------------------------------------
# Storage + JSON exchange
# ---------------------------------------------------------------------------


@dataclass
class ImportResult:
    view_id: int
    name: str
    unmatched: list[dict]


class TreeViewService:
    """CRUD for saved views and JSON file exchange. Caller owns the session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # -- CRUD -------------------------------------------------------------

    def list_views(self) -> list[SavedTreeView]:
        return list(
            self._session.execute(
                select(SavedTreeView).order_by(SavedTreeView.name)
            ).scalars()
        )

    def get(self, view_id: int) -> SavedTreeView:
        view = self._session.get(SavedTreeView, view_id)
        if view is None:
            raise TreeViewError(f"no view {view_id}")
        return view

    def _find_name(self, name: str) -> Optional[SavedTreeView]:
        key = " ".join(name.split()).casefold()
        for v in self.list_views():
            if v.name.casefold() == key:
                return v
        return None

    def unique_name(self, name: str) -> str:
        base = " ".join((name or "").split()) or "View"
        if self._find_name(base) is None:
            return base
        n = 2
        while self._find_name(f"{base} ({n})") is not None:
            n += 1
        return f"{base} ({n})"

    def save(self, name: str, state: TreeViewState, view_id: Optional[int] = None) -> SavedTreeView:
        """Create (``view_id=None``) or overwrite a view."""
        clean = " ".join((name or "").split())
        if not clean:
            raise TreeViewError("empty name")
        other = self._find_name(clean)
        if other is not None and other.id != view_id:
            raise TreeViewError(f"duplicate name {clean!r}")
        payload = json.dumps(state.to_json(), ensure_ascii=False)
        if view_id is None:
            view = SavedTreeView(name=clean, state_json=payload)
            self._session.add(view)
        else:
            view = self.get(view_id)
            view.name = clean
            view.state_json = payload
        self._session.flush()
        return view

    def load(self, view_id: int) -> TreeViewState:
        """State of a saved view, minus persons deleted since the save."""
        state = TreeViewState.from_json(json.loads(self.get(view_id).state_json))
        return state.drop_unknown(self._existing(state.referenced_ids()))

    def rename(self, view_id: int, name: str) -> SavedTreeView:
        view = self.get(view_id)
        return self.save(name, TreeViewState.from_json(json.loads(view.state_json)), view_id)

    def duplicate(self, view_id: int, name: Optional[str] = None) -> SavedTreeView:
        view = self.get(view_id)
        state = TreeViewState.from_json(json.loads(view.state_json))
        return self.save(self.unique_name(name or view.name), state)

    def delete(self, view_id: int) -> None:
        self._session.delete(self.get(view_id))
        self._session.flush()

    def _existing(self, ids: Iterable[int]) -> set[int]:
        ids = list(set(ids))
        if not ids:
            return set()
        return set(
            self._session.execute(select(Person.id).where(Person.id.in_(ids))).scalars()
        )

    # -- JSON exchange ----------------------------------------------------

    def export_document(self, name: str, state: TreeViewState) -> dict:
        doc = state.to_json()
        doc["name"] = name
        persons = {}
        ids = state.referenced_ids()
        if ids:
            for p in self._session.execute(select(Person).where(Person.id.in_(ids))).scalars():
                persons[str(p.id)] = {
                    "name": p.name,
                    "familyCode": p.family_code,
                    "externalFamilyCode": p.external_family_code,
                    "birthDate": p.birth_date,
                }
        doc["persons"] = persons
        return doc

    def import_document(self, doc: dict, name: Optional[str] = None) -> ImportResult:
        """Store *doc* as a new view, matching persons to this database (D4)."""
        state = TreeViewState.from_json(doc)
        persons = {int(k): v for k, v in (doc.get("persons") or {}).items()}
        mapping: dict[int, Optional[int]] = {}
        unmatched: list[dict] = []
        for pid in state.referenced_ids():
            info = persons.get(pid, {})
            match = self._match_person(pid, info)
            mapping[pid] = match
            if match is None:
                unmatched.append({"id": pid, **info})
        state = state.remap(mapping)
        view = self.save(self.unique_name(name or doc.get("name") or "View"), state)
        return ImportResult(view.id, view.name, unmatched)

    def _match_person(self, pid: int, info: dict) -> Optional[int]:
        existing = self._session.get(Person, pid)
        if existing is not None:
            if not info:
                return pid
            if (
                (info.get("name") and info.get("name") == existing.name)
                or (info.get("familyCode") and info.get("familyCode") == existing.family_code)
                or (
                    info.get("externalFamilyCode")
                    and info.get("externalFamilyCode") == existing.external_family_code
                )
            ):
                return pid
        for column, value in (
            (Person.family_code, info.get("familyCode")),
            (Person.external_family_code, info.get("externalFamilyCode")),
        ):
            if value:
                ids = list(
                    self._session.execute(select(Person.id).where(column == value)).scalars()
                )
                if len(ids) == 1:
                    return ids[0]
        if info.get("name"):
            q = select(Person.id).where(Person.name == info["name"])
            if info.get("birthDate"):
                q = q.where(Person.birth_date == info["birthDate"])
            ids = list(self._session.execute(q).scalars())
            if len(ids) == 1:
                return ids[0]
        return None
