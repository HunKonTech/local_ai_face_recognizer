# Tasks

## 1. Derivation

- [x] 1.1 Implement `friend_links()` (design D1) with scheme-aware marker handling for multi-codes and ranges; verify with `tests/test_family_tree_service.py` cases for the single, shared, range, missing-anchor and self-link scenarios, and with a custom scheme whose friend letter differs.
- [x] 1.2 Add `FamilyTreeGraph.friends` filled by `build_tree` for in-graph anchors (design D2); verify with a test that friends who are not members are not in `nodes` and that generations are unchanged.

## 2. View state and layout

- [x] 2.1 Add `friend_mode` and `friend_anchors` to `TreeViewState` (including JSON format and DB round trip) and `visible_friends()` (design D3); verify with service tests for the off, all and selected modes, a hidden friend, and a friend who is already a visible member.
- [x] 2.2 Place satellite friend boxes in `compute_layout` (design D4); verify with `tests/test_family_tree_layout.py` that the off mode gives identical positions, and that with friends on the satellite sits next to the anchor block and family order is preserved.

## 3. Drawing and UI

- [x] 3.1 Draw dashed friend connectors (design D5) with `_FRIEND_LINE` in `style.py`; verify with an offscreen widget test that a friend link produces a path item with `Qt.DashLine`, and that selecting a person does not recolour it.
- [x] 3.2 Add the Tree Studio view option (off / everyone / selected) and the "Show friends" toggle in the person context menu, with EN + HU strings; verify in the running app that the mode is saved with the view and restored on load.
- [x] 3.3 Add the Face-Local family tree panel checkbox remembered in QSettings (design D6), with EN + HU strings and retranslation; verify by turning it on, restarting, and seeing dashed friend lines.

## 4. Integration

- [x] 4.1 Run `ruff check app/` and the family test files; in the running app, give a test person `C81B,C82B` and verify one box with two dashed lines in everyone mode, and none in off mode.
