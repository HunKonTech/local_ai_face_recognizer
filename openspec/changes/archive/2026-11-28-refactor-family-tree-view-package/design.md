# Design

## Context

`FamilyTreeView` (in `app/ui/widgets/family_tree_view.py`) is the only consumer-facing class. It is used by `FamilyTreePanel` and by `tests/test_family_tree_view.py`. The tests reach into private state: `_node_items`, `_rect_items`, `_pending_photos`, `_load_visible_photos`, `_scene`, `_ancestors`, `_descendants`, `_connector_items`, `_connector_groups`, `_graph`, `_min_spouse_gap`. They also import the module-level colours `_ANCESTOR_BORDER`, `_DESCENDANT_BORDER` and `_LINE`. Horizontal layout already lives in the Qt-free `app/services/family_tree_layout.py`.

## Goals / Non-Goals

**Goals:**
- Every module stays under ~300 lines and owns one topic.
- Existing tests pass without edits. Private attribute names stay the same, because mixins share state through `self`.
- Later changes can add a renderer style, export and drag and drop as new modules or mixins.

**Non-Goals:**
- Renaming private attributes or methods.
- Changing drawing order, z-values, colours or hit-testing.

## Decisions

**D1: mixin package, not composition objects.** `class FamilyTreeView(_ConnectorsMixin, _NodeBoxMixin, _PhotosMixin, _InteractionMixin, QGraphicsView)`. This matches the `image_browser` and `main_window` conventions and keeps `self._…` state shared, so the tests keep working. Composition (separate renderer objects) would be cleaner, but it would rename the state the tests depend on. It can still come later, when the box renderer becomes configurable.

**D2: Qt-free geometry module.** `_bus_tracks`, `_marriage_label`, `_years`, `_elide` and the lineage-set walk (now a `@staticmethod`) move to `geometry.py` as plain functions. `FamilyTreeView._lineage_sets` stays as a thin static wrapper. These functions get unit tests without a `QApplication`.

**D3: the compat shim re-exports everything tests import.** `family_tree_view.py` keeps a docstring that names the new home and does `from app.ui.widgets.family_tree import FamilyTreeView` plus the colour names from `style.py`. This is the same approach as `panels/image_browser_panel.py`.

**D4: move code mechanically.** Methods are cut and pasted unchanged, then imports are fixed. No logic edits go into the same commit, so the diff can be reviewed as a move.

## Risks / Trade-offs

- [Python MRO puts `QGraphicsView` event handlers behind a mixin's override] → `_InteractionMixin` is listed before `QGraphicsView`, and the handlers keep calling `super()`. This is checked by the existing click and context-menu tests.
- [Circular imports between `view.py` and mixins] → mixins import only `style` and `geometry`, never `view`.
- [Class constants (`BOX_W`, `BOX_H`, …) read by mixins] → they stay on `FamilyTreeView`, and mixins read them through `self`, as they do today.
