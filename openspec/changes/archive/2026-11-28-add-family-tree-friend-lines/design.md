# Design

## Context

- `family_code_interpreter` already understands friend codes. `parse_extended_code` returns `suffix_type == "friend"`, `_split_multi` splits `C81B,C82B` / `C81B C82B`, `expand_range_code` expands `C[1-9]B` using the active scheme's `friend_letter`, and `code_is_relational_marker` detects these codes.
- `family_service.derive_links_from_code` explicitly ignores friend codes, which stay that way: no relationship rows are created.
- After `add-tree-studio-app`, the view is filtered by `visible_ids(graph, state)` before layout. `compute_layout` places couple blocks per band. Connectors live in `family_tree/connectors.py`.

## Goals / Non-Goals

**Goals:** a pure, scheme-aware derivation that is testable without Qt, with friend boxes that do not disturb the family layout.

**Non-Goals:** editing friend links from the tree, friend-of-friend, and explicit `Friend` relationships (they come with the genogram work).

## Decisions

**D1: derivation function.** `friend_links(persons) -> dict[anchor_id, list[friend_id]]` in `family_tree_service.py`:
1. Build an index from canonical upper-case `family_code` to person ids.
2. For each person, split the family code into multi parts, expand ranges, and keep the parts whose parsed `suffix_type == "friend"`.
3. The anchor code is the part with the trailing friend marker removed (`C81B` → `C81`).
4. Look the anchor code up in the index, ignoring self-links.

When an anchor code matches several persons (the code is not unique, which is a soft-validated data problem), all of them are linked. The result is cached per graph build.

**D2: graph carries friends separately.** `FamilyTreeGraph` gets a `friends: dict[int, tuple[int, ...]]` field (anchor → friends), filled by `build_tree` for anchors in the graph. Friends that are not family members are **not** added to `nodes`, which keeps generations and family layout untouched. The view layer adds them.

**D3: visibility.** `TreeViewState` gets `friend_mode: "off"|"all"|"selected"` and `friend_anchors: set[int]`. After `visible_ids`, `visible_friends(graph, state, visible)` returns `{friend_id: [anchor_ids]}` for the visible anchors that pass the mode, excluding hidden friends. A friend who is already visible is only linked. The others become *satellite* nodes, which get the generation of their first anchor (ordered by the anchor's x after layout, falling back to id order).

**D4: layout.** Satellites are passed to `compute_layout` as extra members of their first anchor's block, appended after the couple chain on the side away from the spouse. The alternative, a separate band or the floating area, would draw long dashed lines across the chart. That was rejected because it is unreadable on wide trees. A test asserts that the family nodes' x positions shift only by the inserted width, and that the relative order is kept.

**D5: drawing.** In `connectors.py`, friend lines are `QPen(Qt.DashLine)` in a dedicated colour (`_FRIEND_LINE`, in `style.py`). They run from the side midpoint of the friend box to the anchor box. When both are in the same row, the line is a horizontal dashed segment at 1/3 box height, which does not collide with the spouse line at 1/2. Across rows it is an elbow path. Friend lines are excluded from the lineage recolouring.

**D6: Face-Local toggle.** `FamilyTreePanel` builds a `TreeViewState` equivalent with `friend_mode` from QSettings `family_tree/show_friends` (bool → "all"/"off"), set by a checkbox in the panel toolbar and retranslated in `retranslate()`.

## Risks / Trade-offs

- [Large ranges (`C[1-99]B`) create many links] → links exist only to anchors that exist. Drawing is limited to visible anchors in the chosen mode.
- [A friend code that refers to a stale or renamed anchor code] → no link is drawn, the same as today when a code is invalid. The person editor's existing code warnings already flag this.
- [Satellite boxes widen blocks and shift layout] → only when friends are shown. The off mode is identical to the layout before this change, which a regression test checks.
