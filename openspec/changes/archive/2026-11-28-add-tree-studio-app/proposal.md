# Proposal

## Why

Face-Local's family tree tab shows the whole family from one root, with fixed box contents. Users want to compose specific trees for printing and sharing. That means choosing who is on the chart, collapsing and expanding branches, hiding individuals, dragging people in, choosing which facts each box shows, and saving and exporting the result. The family tree tab sits next to the photo workflow and is the wrong place for this editor. A separate app on the same database and codebase keeps Face-Local focused and still edits the same persons.

## What Changes

- **Tree Studio**: a separate desktop app (`python -m app.tree_studio`, or the packaged executable with `--tree-studio`, also launchable from Face-Local's menu). It opens the same database and config as Face-Local.
- **Root**: a view starts from a chosen person, or from an external family code `#root#`. The person whose external code is exactly `#root#` is the root.
- **Composing the chart**:
  - Collapsing a person hides their descendants. Persons reachable another way stay visible, and spouses stay until they are hidden individually.
  - Expanding restores the hidden descendants.
  - Any person, including a spouse, can be hidden one by one and shown again from a list of hidden persons.
- **Drag and drop** from a searchable person list:
  - A person already related to the tree appears in their place: their branch is expanded and they are un-hidden.
  - An unrelated person dropped on a box creates a relationship (parent / child / spouse / sibling) after a choice.
  - An unrelated person dropped on empty space becomes a floating box with a "Connect…" action.
- **Box contents**: the view chooses which person fields appear in boxes (built-in and custom fields from `add-dynamic-person-fields`). Each person can override this by showing or hiding individual fields. Box height adapts per generation row.
- **Person editing** uses the same editor and save path as Face-Local. Relationship editing reuses the existing relationship editor.
- **Saved views**: views are stored in the database by name, and can be loaded, renamed, duplicated and deleted. They can also be exported to and imported from a JSON file. On import, persons are matched by id and identifying data, and unmatched persons are reported.
- **Image export**: the whole chart can be exported to PNG, JPG and SVG, with photos included.
- External changes are detected on activation, as in Face-Local (`add-db-change-notification`). Saved views are registered as a watched table.
- The shared `FamilyTreeView` widget gains configurable box lines, a height per generation, collapse markers and drop handling. With the default field set, Face-Local's tab looks as it does today.

Depends on: `refactor-family-tree-view-package`, `add-dynamic-person-fields`, `add-db-change-notification`.
Out of scope: friend lines (`add-family-tree-friend-lines`), genogram symbols, manual box positioning, printing and PDF.

## Capabilities

### New Capabilities
- `tree-studio-app`: the separate app: start-up on the shared database, the person list, drag and drop with relationship creation, and person and relationship editing.
- `family-tree-views`: the composition of a tree view (root, collapse, hide, floating persons, field display per view and per person), and saving, loading and JSON exchange of views.
- `family-tree-image-export`: exporting a composed tree to PNG, JPG and SVG.

### Modified Capabilities
<!-- none: external-change-notification is extended by registering a table, not by changing its requirements -->

## Impact

- New package `app/tree_studio/` (entry point, main window, mixins: views, person list, drag and drop, export).
- New `app/services/tree_view_service.py` (view CRUD, JSON import/export, visibility computation) and table `family_tree_views`, plus migration.
- `app/ui/widgets/family_tree/`: node box renderer driven by a field list, per-generation row heights in layout and connectors, collapse indicator, drop signals, and a `scene_export.py` module.
- `app/main.py`: `--tree-studio` flag; Face-Local menu entry that launches Tree Studio as a separate process.
- `app/services/family_tree_service.py`: lookup of a root by external code.
- i18n strings (EN + HU). Tests for the service, visibility rules, JSON round-trip and export.
