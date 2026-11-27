"""Spouse and parent-child connector drawing for :class:`FamilyTreeView`."""

from __future__ import annotations

from collections import defaultdict

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsRectItem, QGraphicsSimpleTextItem

from app.ui.widgets.family_tree.geometry import _bus_tracks, _marriage_label
from app.ui.widgets.family_tree.style import (
    _ANCESTOR_BORDER,
    _DESCENDANT_BORDER,
    _FRIEND_LINE,
    _LINE,
    _SPOUSE_BORDER,
    _TEXT,
)


class _ConnectorsMixin:
    """Draws and recolours connector lines; state lives on the view."""

    def _center(self, x: float, y: float, h: float | None = None) -> tuple[float, float]:
        return x + self.BOX_W / 2, y + (self.BOX_H if h is None else h) / 2

    def _center_of(self, pid: int, positions) -> tuple[float, float]:
        return self._center(*positions[pid], self._box_h.get(pid, self.BOX_H))

    def _draw_connectors(self, graph, positions, focus_id) -> None:
        # _connector_items: list of dicts describing each drawn line + enough
        # metadata to recolour it for any selection (see _update_line_colors).
        # _connector_groups: per parent→children group context (parents, the
        # junction x where the drop meets the bus, and each child's column x).
        self._connector_items = []
        self._connector_groups = []

        # --- Spouse links (horizontal) ---
        drawn_spouse: set[tuple[int, int]] = set()
        for pid, node in graph.nodes.items():
            if pid not in positions:
                continue
            x, y = positions[pid]
            for spouse in node.spouse_ids:
                if spouse not in positions:
                    continue
                key = tuple(sorted((pid, spouse)))
                if key in drawn_spouse:
                    continue
                drawn_spouse.add(key)
                sx, sy = positions[spouse]
                if x <= sx:
                    lx, ly, rx, ry = x, y, sx, sy
                else:
                    lx, ly, rx, ry = sx, sy, x, y
                half = self._box_h.get(pid, self.BOX_H) / 2
                item = self._add_line_path([
                    (lx + self.BOX_W, ly + half),
                    (rx, ry + half),
                ])
                self._connector_items.append(
                    {"item": item, "kind": "spouse", "pair": frozenset([pid, spouse])}
                )

                # Marriage date label centred ON the horizontal spouse line.
                label = _marriage_label(*graph.marriage(pid, spouse))
                if label:
                    mid_x = (lx + self.BOX_W + rx) / 2
                    line_y = (ly + ry) / 2 + half
                    self._add_edge_label(mid_x, line_y, label)

        # --- Parent → children: group siblings by their canonical couple so they
        #     all share one horizontal bus instead of each getting its own line.
        #
        # Problem: some children may only have ONE parent listed in the DB (e.g.
        # only linked to Anya, not to Apa), giving a different frozenset key than
        # siblings who have both parents. Fix: expand each child's parent set to
        # include their parents' spouses who are also parents in this graph.
        all_parents: set[int] = set()
        for _node in graph.nodes.values():
            for p in _node.parent_ids:
                if p in positions:
                    all_parents.add(p)

        children_by_parents: dict[frozenset, list[int]] = defaultdict(list)
        for pid, node in graph.nodes.items():
            if pid not in positions:
                continue
            parents_in_graph = frozenset(p for p in node.parent_ids if p in positions)
            if not parents_in_graph:
                continue
            # Expand a lone listed parent with their spouse if that spouse is
            # also a parent somewhere in the graph, so partial-link siblings
            # share a bus. Children with both parents listed keep exactly that
            # pair, so remarriages get one bus per marriage.
            couple_key: set[int] = set(parents_in_graph)
            if len(parents_in_graph) == 1:
                pnode = graph.node(next(iter(parents_in_graph)))
                if pnode:
                    for spouse in pnode.spouse_ids:
                        if spouse in all_parents:
                            couple_key.add(spouse)
            children_by_parents[frozenset(couple_key)].append(pid)

        families = []
        for parent_ids_fs, children in children_by_parents.items():
            parents_in_pos = sorted(p for p in parent_ids_fs if p in positions)
            if not parents_in_pos:
                continue
            parent_centers = [self._center_of(p, positions) for p in parents_in_pos]
            # Horizontal midpoint between the parents (or single parent).
            drop_x = sum(c[0] for c in parent_centers) / len(parent_centers)
            # Bottom edge of the parents' row.
            py = max(
                positions[p][1] + self._box_h.get(p, self.BOX_H) for p in parents_in_pos
            )
            children_sorted = sorted(children, key=lambda c: positions[c][0])
            child_x = {
                c: positions[c][0] + self.BOX_W / 2 for c in children_sorted
            }
            families.append(
                (parents_in_pos, parent_centers, drop_x, py, children_sorted, child_x)
            )
        bus_ys = _bus_tracks(
            [(py, [drop_x, *child_x.values()]) for _, _, drop_x, py, _, child_x in families],
            self.V_GAP,
        )

        for (parents_in_pos, parent_centers, drop_x, py, children_sorted, child_x), bus_y in zip(
            families, bus_ys
        ):

            gidx = len(self._connector_groups)
            self._connector_groups.append({
                "parents": frozenset(parents_in_pos),
                "children": frozenset(children_sorted),
                "drop_x": drop_x,
                "child_x": child_x,
            })

            # A married couple already has its spouse line connecting the boxes,
            # so the children hang straight from that line's midpoint — no extra
            # bridge. A non-couple pair (rare: two parents not married to each
            # other) keeps a bottom-edge bridge so the drop has something to meet.
            is_couple = False
            if len(parents_in_pos) == 2:
                first, second = parents_in_pos
                first_node = graph.node(first)
                is_couple = first_node is not None and second in first_node.spouse_ids

            if is_couple:
                # Spouse line sits at the boxes' mid-height; drop from there.
                junction_top = min(c[1] for c in parent_centers)
            else:
                if len(parent_centers) == 2:
                    left_cx = min(c[0] for c in parent_centers)
                    right_cx = max(c[0] for c in parent_centers)
                    item = self._add_line_path([(left_cx, py), (right_cx, py)])
                    self._connector_items.append(
                        {"item": item, "kind": "group", "group": gidx, "role": "bridge"}
                    )
                junction_top = py

            # Vertical drop from the parents down to the bus level.
            item = self._add_line_path([(drop_x, junction_top), (drop_x, bus_y)])
            self._connector_items.append(
                {"item": item, "kind": "group", "group": gidx, "role": "junction"}
            )

            # Horizontal bus split into per-gap segments between the connection
            # points (each child's column + the junction). Segmenting lets a
            # selection highlight only the stretch from one child to the junction.
            points = sorted(set(child_x.values()) | {drop_x})
            for a, b in zip(points, points[1:]):
                item = self._add_line_path([(a, bus_y), (b, bus_y)])
                self._connector_items.append(
                    {"item": item, "kind": "group", "group": gidx,
                     "role": "bus", "a": a, "b": b}
                )

            # Individual vertical drops from the bus to each child box.
            for child_id in children_sorted:
                cx = child_x[child_id]
                cy = positions[child_id][1]
                item = self._add_line_path([(cx, bus_y), (cx, cy)])
                self._connector_items.append(
                    {"item": item, "kind": "group", "group": gidx,
                     "role": "childdrop", "child": child_id}
                )

    def _draw_friend_lines(self, graph, positions) -> None:
        """Dashed friend -> anchor links (excluded from lineage recolouring).

        Same row: a horizontal segment at 1/3 box height (clear of the spouse
        line at 1/2). Different rows: an elbow from the friend's top/bottom.
        """
        self._friend_items = []
        for friend, anchor in getattr(graph, "friend_edges", None) or ():
            if friend not in positions or anchor not in positions:
                continue
            fx, fy = positions[friend]
            ax, ay = positions[anchor]
            fh = self._box_h.get(friend, self.BOX_H)
            ah = self._box_h.get(anchor, self.BOX_H)
            if abs(fy - ay) < 0.5:
                y = fy + fh / 3
                if fx < ax:
                    pts = [(fx + self.BOX_W, y), (ax, y)]
                else:
                    pts = [(fx, y), (ax + self.BOX_W, y)]
            else:
                fcx, acx = fx + self.BOX_W / 2, ax + self.BOX_W / 2
                if fy < ay:
                    start, end = (fcx, fy + fh), (acx, ay)
                else:
                    start, end = (fcx, fy), (acx, ay + ah)
                mid = (start[1] + end[1]) / 2
                pts = [start, (start[0], mid), (end[0], mid), end]
            item = self._add_line_path(pts)
            pen = QPen(_FRIEND_LINE)
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            item.setPen(pen)
            item.setZValue(-4)
            self._friend_items.append(item)

    def _add_edge_label(self, cx: float, cy: float, text: str) -> None:
        """Draw a small white-backed text tag centred on (cx, cy).

        Used for the marriage-date label placed on the spouse connecting line.
        The white background keeps the label legible over any line that crosses it.
        """
        label = QGraphicsSimpleTextItem(text)
        font = label.font()
        font.setPointSize(8)
        label.setFont(font)
        label.setBrush(QBrush(_TEXT))
        rect = label.boundingRect()
        pad = 2.5
        top = cy - rect.height() / 2 - pad
        bg = QGraphicsRectItem(
            cx - rect.width() / 2 - pad, top,
            rect.width() + 2 * pad, rect.height() + 2 * pad,
        )
        bg.setBrush(QBrush(QColor("#ffffff")))
        bg.setPen(QPen(_LINE, 0))
        bg.setZValue(2)
        self._scene.addItem(bg)
        label.setPos(cx - rect.width() / 2, cy - rect.height() / 2)
        label.setZValue(3)
        self._scene.addItem(label)

    def _add_line_path(self, points: list[tuple[float, float]]) -> QGraphicsPathItem:
        path = QPainterPath()
        path.moveTo(*points[0])
        for pt in points[1:]:
            path.lineTo(*pt)
        item = QGraphicsPathItem(path)
        pen = QPen(_LINE)
        pen.setWidth(2)
        item.setPen(pen)
        item.setZValue(-5)
        self._scene.addItem(item)
        return item

    def _update_line_colors(self) -> None:
        """Recolour connectors to trace the selected person's direct lineage.

        For the selected person S we light up exactly the path that leads from S
        up to its ancestors (blue) and down to its descendants (green); spouse
        links of S go red. Sibling branches that merely share a bus with S stay
        grey, so the highlight reaches only where it should.
        """
        sel = self._selected_id
        anc = self._ancestors
        desc = self._descendants
        up_or_self = (anc | {sel}) if sel is not None else set()
        down_or_self = (desc | {sel}) if sel is not None else set()

        # Per-group relevance: is this group on S's ancestor line (and through
        # which child), or on S's descendant line (whole group lights up)?
        group_state = []
        for g in self._connector_groups:
            parents = g["parents"]
            up_child = None
            if sel is not None and parents and parents <= anc:
                for c in g["children"]:
                    if c in up_or_self:
                        up_child = c
                        break
            down = sel is not None and bool(parents & down_or_self)
            group_state.append((up_child, down))

        for it in self._connector_items:
            color = _LINE
            if it["kind"] == "spouse":
                if sel is not None and sel in it["pair"]:
                    color = _SPOUSE_BORDER
            else:
                g = it["group"]
                up_child, down = group_state[g]
                if down:
                    color = _DESCENDANT_BORDER
                elif up_child is not None:
                    color = self._up_color_for(it, g, up_child)
            pen = QPen(color)
            pen.setWidth(3 if color is not _LINE else 2)
            it["item"].setPen(pen)

    def _up_color_for(self, it: dict, g: int, up_child: int) -> QColor:
        """Colour for a group connector when the group is on S's ancestor line.

        Only the single stretch from ``up_child`` to the parents' junction is
        lit; the other siblings' drops and bus segments stay grey.
        """
        role = it["role"]
        if role in ("bridge", "junction"):
            return _ANCESTOR_BORDER
        if role == "childdrop":
            return _ANCESTOR_BORDER if it["child"] == up_child else _LINE
        if role == "bus":
            grp = self._connector_groups[g]
            cx = grp["child_x"].get(up_child)
            if cx is None:
                return _LINE
            lo, hi = min(cx, grp["drop_x"]), max(cx, grp["drop_x"])
            # Segment [a, b] is on the up_child→junction stretch.
            if it["a"] >= lo - 0.5 and it["b"] <= hi + 0.5:
                return _ANCESTOR_BORDER
        return _LINE
