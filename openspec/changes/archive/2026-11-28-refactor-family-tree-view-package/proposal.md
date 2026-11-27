# Proposal

## Why

`app/ui/widgets/family_tree_view.py` is an 832-line `QGraphicsView`. It mixes colours, pure geometry helpers, layout, connector drawing, box drawing, lazy photo loading and mouse/keyboard handling in one class. The planned work needs to change it in several places at once: configurable box contents with a height per generation, collapse markers, drag and drop, scene export, friend lines and a later genogram style. That work would push the file well past the project's ~800-line module limit and make it hard to review. Splitting it first, with **no behaviour change**, gives each later change its own module to grow in.

## What Changes

- Turn the module into a package `app/ui/widgets/family_tree/`, built from mixins (the same pattern as `panels/image_browser/`):
  - `style.py`: colours and size constants.
  - `geometry.py`: Qt-free pure helpers (`_bus_tracks`, `_marriage_label`, `_years`, `_elide`, lineage-set computation).
  - `connectors.py`: spouse/parent-child connectors, bus tracks, edge labels, line recolouring.
  - `node_box.py`: person box drawing.
  - `photos.py`: lazy photo loading.
  - `interaction.py`: hit-testing, mouse, wheel, keys, context menu.
  - `view.py`: `FamilyTreeView` itself (`__init__`, `set_graph`, zoom, layout call, selection).
- `app/ui/widgets/family_tree_view.py` stays as a compatibility re-export. It keeps `FamilyTreeView` and the private names the tests import (`_ANCESTOR_BORDER`, `_DESCENDANT_BORDER`, `_LINE`, ...).
- The public API (signals, `set_graph`, `set_selected`, zoom methods, class constants) is unchanged.

Out of scope: any visual or behavioural change, variable box height, export, and new features. Those come in later changes (`add-dynamic-person-fields`, `add-tree-studio-app`, `add-family-tree-friend-lines`).

## Capabilities

### New Capabilities
<!-- none: pure refactor, skip_specs is set -->

### Modified Capabilities
<!-- none -->

## Impact

- `app/ui/widgets/family_tree_view.py` becomes a shim; new package `app/ui/widgets/family_tree/`.
- `app/ui/panels/family_tree_panel.py`: import stays valid; it may switch to the package import.
- `tests/test_family_tree_view.py` must pass unchanged.
- `CLAUDE.md` UI map: add the new package.
- No DB, config, i18n or dependency change.
