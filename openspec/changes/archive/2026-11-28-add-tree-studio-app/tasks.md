# Tasks

## 1. View model and storage

- [x] 1.1 Add `TreeViewState`, `RootRef` and `visible_ids()` to `app/services/tree_view_service.py` (design D2); verify with `tests/test_tree_view_service.py` covering every collapse/hide/floating scenario in the `family-tree-views` spec on hand-built graphs.
- [x] 1.2 Add the `family_tree_views` model and migration, view CRUD (unique name, rename, duplicate, delete) and the `ChangeWatchService` registration (design D3); verify with service tests, including that loading a view with a deleted person id drops it.
- [x] 1.3 Add a root lookup by external code (exact `#root#`, prefix candidates) to `FamilyTreeService`; verify with tests for the exact, missing and candidate cases.
- [x] 1.4 Implement JSON export/import with person matching and the unmatched report (design D4); verify with tests: the same-DB round trip is equal, an id mismatch falls back to family code, and 2 unknown persons are reported.

## 2. Shared widget extensions

- [x] 2.1 Make the node box take a line provider (design D5) with a default that reproduces the current lines; verify that `tests/test_family_tree_view.py` passes and that Face-Local's family tree tab looks unchanged.
- [x] 2.2 Implement per-generation row heights in layout, connectors, bands and hit-testing (design D6); verify with a widget test where one tall box in a band gives that band the larger height and the next band starts below it with no overlap.
- [x] 2.3 Add the collapse badge with `collapse_toggled`, and drop acceptance with `person_dropped` (design D7); verify with offscreen tests that simulate a badge click and a drop on a box and on empty space.
- [x] 2.4 Add `scene_export.py` (design D8); verify with tests that the PNG at 2x has double the pixel size, the JPG corners are white, the SVG contains `<text>` with a person name and an embedded image, and the selected-border colour is absent from the export.

## 3. Tree Studio app

- [x] 3.1 Factor the bootstrap into `app/bootstrap.py`, add `app/tree_studio/` (`__main__`, `main`, window skeleton), the `--tree-studio` flag and a Face-Local menu entry that launches it as a process (design D1); verify that `python -m app.tree_studio --db <test db>` opens and that the menu entry starts a second process on the same DB.
- [x] 3.2 Add the persons dock: search, thumbnails, visible marker, drag source; verify by running the app: searching filters, and people already on the chart are marked.
- [x] 3.3 Add composition actions: collapse/expand, hide/show with a hidden-persons list, the view field picker from the catalog, and per-person show/hide from the box context menu; verify in the running app against the `family-tree-views` scenarios (spouse kept on collapse, override for a single person).
- [x] 3.4 Add drop handling: reveal a related person, the relationship menu for an unrelated person on a box (no sibling without parents), floating plus Connect for empty space (design D7); verify in the running app and confirm in Face-Local that the new relationship appears after refresh.
- [x] 3.5 Add views UI: new from person or external code, open, save, rename, duplicate, delete, JSON export/import with an unmatched report; verify the save, close, reopen and load round trip in the running app.
- [x] 3.6 Add export UI (format, scale, JPG quality, markers on/off, progress); verify by exporting a large tree to PNG, JPG and SVG and opening each file.
- [x] 3.7 Wire person and relationship editing (`edit_person_dialog`, `FamilyTreeEditorDialog`) and the external-change bar with the shared auto-refresh setting; verify that editing in Face-Local and activating Tree Studio shows the bar, and the reverse.
- [x] 3.8 Add EN + HU strings for all Tree Studio UI to the closest `i18n/strings/sNN_*.py` (or a new topical part), and retranslation; verify that switching the language updates every Tree Studio label.

## 4. Integration

- [x] 4.1 Run `ruff check app/` and all family, person and tree test files; run both apps side by side on one DB through one full workflow (root by `#code#`, collapse, hide a spouse, drop, save, export to SVG, reload) and verify there are no errors in either log.
- [x] 4.2 Update `CLAUDE.md` (code map: `app/tree_studio/`, `tree_view_service`, `--tree-studio`); verify the documented command starts the app.
