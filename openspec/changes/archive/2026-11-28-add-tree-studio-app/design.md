# Design

## Context

- `FamilyTreeService.build_tree(root_id, include_non_members=...)` builds a rooted `FamilyTreeGraph`. Couples are inferred as co-parents, and every node carries generation, parent, child and spouse ids.
- `FamilyTreeView` (split into `app/ui/widgets/family_tree/` by `refactor-family-tree-view-package`) draws fixed-height boxes. Its lines are name, years and family code. Layout x comes from the Qt-free `family_tree_layout.compute_layout`, and y is `band * (BOX_H + V_GAP)`.
- `PersonFieldService.catalog()` / `values_for()` (from `add-dynamic-person-fields`) provide field keys `b:<col>` / `c:<id>`.
- `ChangeWatchService` with `ExternalChangeBar` (from `add-db-change-notification`) provides detection on activation.
- `app/main.py` parses `--config`/`--db`, initialises logging, config and DB, then builds `MainWindow`.

## Goals / Non-Goals

**Goals:** one widget for both apps, view logic testable without Qt, views stable under data changes, and a packaged build that needs no second executable.

**Non-Goals:** manual node positioning, undo/redo of view edits beyond save/load, and collaborative editing of one view.

## Decisions

**D1: launching.** `app/tree_studio/__main__.py` → `app.tree_studio.main.main()`. The bootstrap (args, logging, config, `init_db`) is factored out of `app/main.py` into a shared `app/bootstrap.py` function, and both entry points call it. The packaged exe gets a `--tree-studio` flag that dispatches to Tree Studio, so the release build is unchanged. Face-Local's menu entry starts `sys.executable` with the right arguments (frozen: the exe plus `--tree-studio`; source: `-m app.tree_studio`), passing the current `--db`/`--config`.

**D2: view state is a plain dataclass plus a pure visibility function.**

```
TreeViewState(root: RootRef(person_id | external_code),
              collapsed: set[int], hidden: set[int], floating: list[int],
              fields: list[str], person_fields: dict[int, {show:set, hide:set}],
              include_non_members: bool, options: dict)
visible_ids(graph, state) -> set[int]
```

`visible_ids` runs a BFS from the root over the unfiltered graph. It does not enter `hidden` persons, and it does not follow child edges out of `collapsed` persons. Parent and spouse edges are followed, so spouses of collapsed persons stay. Persons reachable through another path stay automatically. Floating persons are added on top. The result filters the graph before layout. The function lives in `app/services/tree_view_service.py` and is unit-tested with hand-built graphs.

**D3: storage.** Table `family_tree_views(id, name UNIQUE NOCASE, state_json TEXT, created_at, updated_at)`, registered with `ChangeWatchService` as kind `tree_view`. `state_json` is the dataclass serialised with `"format": "facelocal.treeview.v1"`. Unknown person ids are dropped silently on load, which covers persons deleted since the save. The existing `FamilyTreeSettings` singleton is not changed.

**D4: JSON file exchange.** The file format is the same document plus `persons: {id: {name, family_code, external_family_code, birth_date}}` for every referenced id. On import, the id is kept when it exists and its name or a family code agrees. Otherwise there is one unique match by `(family_code)`, then by `(external_family_code)`, then by `(name, birth_date)`. The rest are reported and dropped. The imported view gets a unique name ("X (2)").

**D5: node content renderer.** `node_box.py` takes `lines_for(node) -> list[(text, role)]` from the view. The default provider reproduces today's lines (name, years, family code), so Face-Local's tab looks unchanged. Tree Studio's provider uses `values_for()` and resolves the field list per person (the view fields, minus that person's `hide`, plus that person's `show`, in catalog order). Box height = padding + name line + n × line height, with the photo slot as the minimum.

**D6: per-generation row heights.** `_layout` computes `row_h[band] = max(box_h(node))` over the nodes in each band, and `row_y[band]` as a cumulative sum plus `V_GAP`. Every box in a band is drawn at `row_h[band]`, so connectors keep using one bottom/top y per band. `_center`, the bus y computation and `_pid_at` take the node's band height instead of `BOX_H`. Generation bands use the same `row_y`/`row_h`. `compute_layout` (x only) is unchanged.

**D7: collapse marker and drop handling in the widget.** A small "+/−" badge sits under each person who has children in the full graph. Clicking it emits `collapse_toggled(pid)`. The widget accepts drags with the MIME type `application/x-facelocal-person-id` and emits `person_dropped(pid, target_pid | None, scene_pos)`. Tree Studio decides what to do (spec: related → reveal; unrelated on a box → relationship menu → `FamilyService.add_parent_child` / `add_spouse`; sibling = `add_parent_child` from each of the target's parents; empty space → floating). "Related" means present in `build_tree(root)` with the view's member setting.

**D8: image export.** `family_tree/scene_export.py` provides `export_scene(view, path, fmt, scale, quality, show_markers)`. Before rendering it loads every pending photo synchronously (`_load_visible_photos` without budget), clears the selection highlight (it re-renders with `selected_id=None` and restores it afterwards), and hides the badges unless `show_markers` is set. PNG/JPG: `QImage(scene_rect.size() * scale)`, filled white for JPG, then `scene.render()`. SVG: `QSvgGenerator` (from PySide6's QtSvg, no new dependency) with the same render. Pixmaps are embedded by Qt as base64 images.

**D9: Tree Studio window.** `app/tree_studio/window.py`: `TreeStudioWindow(…Mixins, QMainWindow)`. Mixins: `views` (open/save/rename/duplicate/delete, import/export JSON), `persons_dock` (search list, drag source, visible marker), `composition` (collapse, hide list, floating, field picker, per-person override from the box context menu), `dnd`, `export`, `external_changes` (reuses the bar and service). The central widget is `FamilyTreeView`. Person edit uses `edit_person_dialog`, and relationships use `FamilyTreeEditorDialog`.

## Risks / Trade-offs

- [SVG output size with many photos] → photos are embedded at box resolution (56 px × scale), not at original size.
- [The panel and Tree Studio drift apart] → both use the same widget, and only the line provider and callbacks differ.
- [Matching in a JSON import can choose the wrong person when the data is ambiguous] → only unique matches are accepted. The rest are reported, never guessed.
- [Synchronous photo loading for export can be slow on big trees] → export runs with a progress dialog. Photo loading is I/O-bound and bounded by the number of visible persons.
- [A relationship created from Tree Studio by mistake] → the dialog states the exact relationship ("U lesz A gyermeke"). Undo goes through the existing relationship editor (remove).
