"""Graphical family-tree view (the ``FamilyTreeView`` class).

Renders a :class:`FamilyTreeGraph` as a classic genealogy chart: persons are
boxes laid out in horizontal generation bands, coloured by sex (blue = male,
orange = female), connected by spouse lines and parent→child elbows. The
focused / selected person is outlined in red. This replaces the old collapsible
tree-list with the picture-style layout.

Each generation is a row; horizontal placement comes from
:mod:`app.services.family_tree_layout` (children centred under their parents,
couples side by side, separate branches side by side). Sibling buses that
would overlap in the same gap between rows are spread onto separate tracks.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from PySide6.QtCore import QPoint, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QGraphicsItemGroup,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from app.services.family_tree_layout import compute_layout, place_satellites
from app.services.family_tree_service import FamilyTreeGraph
from app.ui.widgets.family_tree.connectors import _ConnectorsMixin
from app.ui.widgets.family_tree.geometry import _marriage_label, lineage_sets
from app.ui.widgets.family_tree.interaction import _InteractionMixin
from app.ui.widgets.family_tree.node_box import _NodeBoxMixin
from app.ui.widgets.family_tree.photos import _PhotosMixin
from app.ui.widgets.family_tree.style import _BAND_A, _BAND_B, _GEN_NUMBER, _LINE


class FamilyTreeView(
    _ConnectorsMixin, _NodeBoxMixin, _PhotosMixin, _InteractionMixin, QGraphicsView
):
    """Scrollable, zoomable graphical family tree."""

    person_selected = Signal(int)
    person_activated = Signal(int)
    # Right-click on a person box: (person id, global screen position).
    person_context_menu = Signal(int, QPoint)
    # "+/-" badge under a person with children was clicked.
    collapse_toggled = Signal(int)
    # A person was dropped (MIME PERSON_MIME): (person id, target pid | None,
    # scene position).
    person_dropped = Signal(int, object, QPointF)

    BOX_W = 168
    BOX_H = 86
    PHOTO = 56
    H_GAP = 26
    V_GAP = 64
    GUTTER = 44  # left column showing the generation number

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QBrush(_BAND_A))
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Large-tree friendliness: index for fast viewport queries, partial
        # repaints, cached background bands.
        self._scene.setItemIndexMethod(QGraphicsScene.BspTreeIndex)
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.setCacheMode(QGraphicsView.CacheBackground)

        self._node_items: dict[int, QGraphicsItemGroup] = {}
        self._rect_items: dict[int, QGraphicsRectItem] = {}
        self._node_borders: dict[int, QColor] = {}
        self._graph: Optional[FamilyTreeGraph] = None
        self._focus_id: Optional[int] = None
        self._selected_id: Optional[int] = None
        self._ancestors: set[int] = set()
        self._descendants: set[int] = set()
        self._spouses: set[int] = set()
        self._connector_items: list = []
        self._connector_groups: list = []

        # Box contents and per-generation row heights (Tree Studio). The
        # default provider reproduces the classic name / years / code lines.
        self._line_provider = None
        self._box_h: dict[int, float] = {}
        self._row_y: list[float] = []
        self._row_h: list[float] = []
        # Collapse badges: persons that get one, and which are collapsed.
        self._collapsible: set[int] = set()
        self._collapsed: set[int] = set()
        self._badge_items: dict[int, object] = {}
        self._friend_items: list = []
        self._show_markers = True
        self.setAcceptDrops(True)

        # Photos are loaded lazily for boxes that scroll into view, so a tree
        # with thousands of people opens instantly. pid -> (group, pos, path).
        self._pending_photos: dict[int, tuple] = {}
        self._photo_timer = QTimer(self)
        self._photo_timer.setSingleShot(True)
        self._photo_timer.setInterval(80)
        self._photo_timer.timeout.connect(self._load_visible_photos)
        self.horizontalScrollBar().valueChanged.connect(self._schedule_photo_load)
        self.verticalScrollBar().valueChanged.connect(self._schedule_photo_load)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_graph(
        self,
        graph: FamilyTreeGraph,
        *,
        focus_id: Optional[int] = None,
        selected_id: Optional[int] = None,
    ) -> None:
        self._scene.clear()
        self._node_items = {}
        self._rect_items = {}
        self._node_borders = {}
        self._pending_photos = {}
        self._connector_items = []
        self._connector_groups = []
        self._box_h = {}
        self._badge_items = {}
        self._friend_items = []
        self._graph = graph
        self._focus_id = focus_id
        self._selected_id = selected_id
        if not graph.nodes:
            self._scene.setSceneRect(0, 0, 1, 1)
            return

        # Highlight the selected person's direct line in distinct colours.
        self._ancestors, self._descendants, self._spouses = self._lineage_sets(graph, selected_id)

        h_gap = self._min_spouse_gap(graph)
        positions, total_w, gens = self._layout(graph, h_gap=h_gap)

        # Generation bands (alternating background stripes) with a left gutter
        # carrying an increasing generation number (1 at the top).
        ordered_gens = sorted(gens)
        band_left = -h_gap - self.GUTTER
        band_width = total_w + 2 * h_gap + self.GUTTER
        for band_index, gen in enumerate(ordered_gens):
            y = self._row_y[band_index] - self.V_GAP / 2
            col_h = self._row_h[band_index] + self.V_GAP
            band = self._scene.addRect(
                band_left, y, band_width, col_h,
                QPen(Qt.NoPen),
                QBrush(_BAND_A if band_index % 2 == 0 else _BAND_B),
            )
            band.setZValue(-10)

            number = QGraphicsSimpleTextItem(str(band_index + 1))
            font = number.font()
            font.setPointSize(16)
            font.setBold(True)
            number.setFont(font)
            number.setBrush(QBrush(_GEN_NUMBER))
            num_rect = number.boundingRect()
            number.setPos(
                band_left + (self.GUTTER - num_rect.width()) / 2,
                y + (col_h - num_rect.height()) / 2,
            )
            number.setZValue(-9)
            self._scene.addItem(number)

        # Connector lines first (under the boxes).
        self._draw_connectors(graph, positions, focus_id)
        self._update_line_colors()
        self._draw_friend_lines(graph, positions)

        # Boxes.
        for pid, (x, y) in positions.items():
            node = graph.node(pid)
            if node is not None:
                self._add_box(node, x, y)

        margin = 40
        self._scene.setSceneRect(
            band_left - margin,
            -self.V_GAP,
            band_width + 2 * margin,
            (self._row_y[-1] + self._row_h[-1] + self.V_GAP if self._row_y else 0)
            + margin,
        )

        # Bring the focus/selected person into view, else start at the top.
        anchor = selected_id if selected_id in graph.nodes else focus_id
        if anchor in self._node_items:
            self.centerOn(self._node_items[anchor])
        else:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().minimum())
            self.verticalScrollBar().setValue(self.verticalScrollBar().minimum())
        self._schedule_photo_load()

    # ------------------------------------------------------------------
    # Zoom + selection
    # ------------------------------------------------------------------

    def set_line_provider(self, provider) -> None:
        """``provider(node) -> [(text, role)]``; ``None`` restores the default.

        ``role`` is ``"name"`` (bold, larger) or ``"line"``. Takes effect on the
        next :meth:`set_graph`.
        """
        self._line_provider = provider

    def set_collapse_state(self, collapsible, collapsed) -> None:
        """Persons that get a +/- badge and which of them are collapsed."""
        self._collapsible = set(collapsible or ())
        self._collapsed = set(collapsed or ())

    def set_markers_visible(self, visible: bool) -> None:
        self._show_markers = bool(visible)
        for item in self._badge_items.values():
            item.setVisible(self._show_markers)

    def box_height(self, pid: int) -> float:
        return self._box_h.get(pid, self.BOX_H)

    def zoom_in(self) -> None:
        self._zoom(1.2)

    def zoom_out(self) -> None:
        self._zoom(1 / 1.2)

    def reset_zoom(self) -> None:
        self.resetTransform()
        self._schedule_photo_load()

    def _zoom(self, factor: float) -> None:
        scale = self.transform().m11() * factor
        if scale < 0.05 or scale > 4.0:
            return
        self.scale(factor, factor)
        self._schedule_photo_load()

    def set_selected(self, selected_id: Optional[int]) -> None:
        """Re-colour borders and lines to mark the selection's relatives.

        Cheap update of existing boxes — no full re-layout.
        """
        if self._graph is None:
            return
        self._selected_id = selected_id
        self._ancestors, self._descendants, self._spouses = self._lineage_sets(
            self._graph, selected_id
        )
        for pid, rect in self._rect_items.items():
            rect.setPen(self._pen_for(pid, self._node_borders.get(pid, _LINE)))
        self._update_line_colors()

    @staticmethod
    def _lineage_sets(graph, selected_id) -> tuple[set[int], set[int], set[int]]:
        """Transitive ancestors, descendants, and direct spouses of selected person."""
        return lineage_sets(graph, selected_id)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _min_spouse_gap(self, graph: FamilyTreeGraph) -> int:
        """Minimum horizontal gap so the widest marriage label fits between boxes."""
        from PySide6.QtGui import QFont, QFontMetrics
        f = QFont()
        f.setPointSize(8)
        fm = QFontMetrics(f)
        seen: set[tuple[int, int]] = set()
        max_label_w = 0
        for pid, node in graph.nodes.items():
            for spouse in node.spouse_ids:
                key = tuple(sorted((pid, spouse)))
                if key in seen:
                    continue
                seen.add(key)
                label = _marriage_label(*graph.marriage(pid, spouse))
                if label:
                    max_label_w = max(max_label_w, fm.horizontalAdvance(label) + 20)
        return max(self.H_GAP, max_label_w)

    def _layout(self, graph: FamilyTreeGraph, h_gap: Optional[int] = None):
        """Return (positions{pid:(x,y)}, total_width, gens{gen:[nodes]}).

        Horizontal placement comes from :func:`compute_layout` (children under
        their parents, couples side by side, branches never interleaved).
        """
        if h_gap is None:
            h_gap = self.H_GAP
        satellites = getattr(graph, "satellites", None) or {}
        family = [n for pid, n in graph.nodes.items() if pid not in satellites]
        placed = compute_layout(family, box_w=self.BOX_W, gap=h_gap)
        if satellites:
            placed = place_satellites(
                placed, satellites,
                {pid: n.spouse_ids for pid, n in graph.nodes.items()},
                box_w=self.BOX_W, gap=h_gap,
            )
        # Per-generation row heights: every box in a band is as tall as the
        # band's tallest content; the next band starts below it (no overlap).
        n_bands = max((band for _, band in placed.values()), default=-1) + 1
        row_h = [float(self.BOX_H)] * n_bands
        for pid, (_x, band) in placed.items():
            node = graph.node(pid)
            if node is not None:
                row_h[band] = max(row_h[band], self._content_height(node))
        row_y: list[float] = []
        y = 0.0
        for h in row_h:
            row_y.append(y)
            y += h + self.V_GAP
        self._row_h, self._row_y = row_h, row_y
        self._box_h = {pid: row_h[band] for pid, (_x, band) in placed.items()}
        positions: dict[int, tuple[float, float]] = {
            pid: (x, row_y[band]) for pid, (x, band) in placed.items()
        }
        gens: dict[int, list] = defaultdict(list)
        for node in graph.nodes.values():
            gens[node.generation].append(node)
        for nodes in gens.values():
            nodes.sort(key=lambda n: positions[n.person_id][0])
        total_w = max((x for x, _ in positions.values()), default=0) + self.BOX_W
        return positions, total_w, gens
