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

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QGraphicsItemGroup,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from app.services.family_tree_layout import compute_layout
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

        col_h = self.BOX_H + self.V_GAP
        # Generation bands (alternating background stripes) with a left gutter
        # carrying an increasing generation number (1 at the top).
        ordered_gens = sorted(gens)
        band_left = -h_gap - self.GUTTER
        band_width = total_w + 2 * h_gap + self.GUTTER
        for band_index, gen in enumerate(ordered_gens):
            y = band_index * col_h - self.V_GAP / 2
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
            len(ordered_gens) * col_h + margin,
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
        placed = compute_layout(
            graph.nodes.values(), box_w=self.BOX_W, gap=h_gap
        )
        step_y = self.BOX_H + self.V_GAP
        positions: dict[int, tuple[float, float]] = {
            pid: (x, band * step_y) for pid, (x, band) in placed.items()
        }
        gens: dict[int, list] = defaultdict(list)
        for node in graph.nodes.values():
            gens[node.generation].append(node)
        for nodes in gens.values():
            nodes.sort(key=lambda n: positions[n.person_id][0])
        total_w = max((x for x, _ in positions.values()), default=0) + self.BOX_W
        return positions, total_w, gens
