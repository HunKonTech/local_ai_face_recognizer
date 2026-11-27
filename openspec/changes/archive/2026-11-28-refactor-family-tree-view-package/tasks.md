# Tasks

## 1. Package skeleton and pure helpers

- [x] 1.1 Create `app/ui/widgets/family_tree/` with `__init__.py` (exports `FamilyTreeView`) and `style.py` (colour and `_ROLE_PID` constants moved verbatim); verify `python -c "import app.ui.widgets.family_tree.style"` succeeds.
- [x] 1.2 Move `_bus_tracks`, `_marriage_label`, `_years`, `_elide` and the lineage-set walk into `geometry.py` (no Qt imports); add `tests/test_family_tree_geometry.py` covering bus-track separation, marriage label formatting, year formatting and elision, and verify it runs without a `QApplication`.

## 2. Mixins

- [x] 2.1 Move connector drawing (`_draw_connectors`, `_add_edge_label`, `_add_line_path`, `_update_line_colors`, `_up_color_for`, `_center`) into `connectors.py` as `_ConnectorsMixin`; verify the connector/line-colour tests in `tests/test_family_tree_view.py` pass.
- [x] 2.2 Move `_add_box` and `_pen_for` into `node_box.py`, the photo code (`_schedule_photo_load`, `_load_visible_photos`, `_load_photo`, `showEvent`/`resizeEvent` scheduling) into `photos.py`, and `_pid_at` plus mouse/wheel/key/context handlers into `interaction.py`; verify the photo and click tests pass.
- [x] 2.3 Leave `__init__`, `set_graph`, `set_selected`, zoom, `_min_spouse_gap`, `_layout` and `_lineage_sets` in `view.py` with the mixins in the base list; verify every module is under 300 lines (`wc -l`).

## 3. Compatibility and docs

- [x] 3.1 Reduce `app/ui/widgets/family_tree_view.py` to a re-export shim (`FamilyTreeView`, `_ANCESTOR_BORDER`, `_DESCENDANT_BORDER`, `_LINE` and the other colours); verify `QT_QPA_PLATFORM=offscreen python -m pytest -q -o addopts="" tests/test_family_tree_view.py tests/test_family_tree_panel.py` passes with no test edits.
- [x] 3.2 Add the `widgets/family_tree/` package with its module topics to the UI map in `CLAUDE.md`; verify `ruff check app/ui/widgets` is clean.
- [x] 3.3 Run the app (`python -m app.main`), open the family tree tab, and verify that selection colours, marriage labels, photos, zoom and the context menu look and behave as before.
