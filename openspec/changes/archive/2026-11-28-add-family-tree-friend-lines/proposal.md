# Proposal

## Why

Friends and acquaintances are recorded in the family code with the friend marker: `C81B` is a friend of C81. Shared friends use multi-codes (`C81B,C82B`) and ranges (`C[1-9]B`). The family tree ignores these codes completely, so friends never appear on the chart, even when they matter for the family's story or for the photos. Users want to show them connected to their person with a dashed line, switchable for the whole chart or only for chosen persons.

## What Changes

- Friend links are **derived from family codes only**. A person whose family code contains friend codes is linked to each anchor person: the person whose family code equals the friend code without its marker. Multi-codes and ranges give one link per anchor. No relationship rows are created. Explicit friend relationships are left to the later genogram work.
- The tree can draw friend links as **dashed lines** between the friend and the anchor:
  - A friend who is already on the chart is connected where they are.
  - A friend who is not on the chart is shown as an extra box next to the anchor, in the anchor's generation row.
  - A friend of several anchors gets one box and a dashed line to each visible anchor.
- Friend display mode: **off**, **everyone**, or **only selected persons** (for chosen anchors).
  - In Tree Studio the mode and the selected persons are part of the saved view. A person's context menu toggles "show friends" for that person.
  - Face-Local's family tree tab gets a simple off/everyone toggle that is remembered between runs.
- Hiding a friend box works like hiding any person in the view.

Depends on: `add-tree-studio-app` (view state and the shared widget extensions).
Out of scope: explicit friend relationships, friend-of-friend chains, and genogram emotional lines.

## Capabilities

### New Capabilities
- `family-tree-friend-links`: deriving friend links from family codes and showing them on the family tree with the off / everyone / selected-persons modes.

### Modified Capabilities
<!-- none: family-tree-views is new in add-tree-studio-app and not yet archived; this change adds its own capability -->

## Impact

- `app/services/family_tree_service.py` (or a new `family_friend_service.py`): derive the anchor→friends mapping from family codes using `family_code_interpreter` (friend suffix, `_split_multi`, `expand_range_code`), respecting the active code scheme.
- `app/services/tree_view_service.py`: friend mode and selected anchors in `TreeViewState`; friends included in visibility.
- `app/services/family_tree_layout.py`: friend boxes placed next to their anchor's block.
- `app/ui/widgets/family_tree/connectors.py`: dashed friend connectors.
- Tree Studio: view option and context-menu toggle. Face-Local `family_tree_panel.py`: off/everyone toggle.
- i18n strings (EN + HU). Tests for derivation, visibility and layout.
