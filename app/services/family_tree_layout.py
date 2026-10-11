"""Horizontal placement of a family tree drawn in generation bands.

Pure geometry (no Qt, no DB) shared by the in-app :class:`FamilyTreeView`; the
web export's ``family-tree.astro`` carries a line-by-line JavaScript port, so
both renderers place people identically.

Every generation is a row. Within a row, spouses form one *block* (partners
side by side). Blocks are positioned by alternating sweeps:

* **down** — each block moves under the midpoint of its members' parents, so
  children hang directly beneath their parents and siblings stay together;
* **up** — each block moves over the midpoint of its children, so a couple
  ends up centred above its family.

After every sweep a row is re-ordered by the desired positions and placed
with the least total displacement that keeps the blocks from overlapping
(weighted isotonic regression, "pool adjacent violators"). Unrelated family
branches (connected components) are laid out separately and put side by side,
so their lines can never cross.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional, Protocol, Sequence


class LayoutNode(Protocol):
    person_id: int
    generation: int
    name: str
    parent_ids: Sequence[int]
    spouse_ids: Sequence[int]
    birth_date: Optional[str]


SWEEPS = 6


def compute_layout(
    nodes: Iterable[LayoutNode],
    *,
    box_w: float,
    gap: float,
    group_gap: Optional[float] = None,
) -> dict[int, tuple[float, int]]:
    """Return ``{person_id: (left_x, band_index)}`` for every node.

    ``gap`` separates spouses and siblings, ``group_gap`` (default ``2*gap``)
    separates blocks that do not share parents. Band 0 is the oldest
    generation; ``left_x`` starts at 0.
    """
    by_id = {n.person_id: n for n in nodes}
    if not by_id:
        return {}
    group_gap = 2 * gap if group_gap is None else group_gap

    gens = sorted({n.generation for n in by_id.values()})
    band_of_gen = {g: i for i, g in enumerate(gens)}
    band = {pid: band_of_gen[n.generation] for pid, n in by_id.items()}

    parents: dict[int, list[int]] = {}
    children: dict[int, list[int]] = defaultdict(list)
    for pid, n in by_id.items():
        ps = sorted(p for p in n.parent_ids if p in by_id and band[p] < band[pid])
        parents[pid] = ps
        for p in ps:
            children[p].append(pid)
    spouses = {
        pid: sorted(s for s in n.spouse_ids if s in by_id and band[s] == band[pid])
        for pid, n in by_id.items()
    }
    key = {
        pid: ((n.birth_date or "~"), (n.name or "").lower(), pid)
        for pid, n in by_id.items()
    }

    positions: dict[int, tuple[float, int]] = {}
    offset = 0.0
    for comp in _components(by_id, parents, children, spouses):
        xs = _layout_component(
            comp, band, parents, children, spouses, key,
            box_w=box_w, gap=gap, group_gap=group_gap,
        )
        left = min(xs.values())
        right = max(xs.values()) + box_w
        for pid, x in xs.items():
            positions[pid] = (x - left + offset, band[pid])
        offset += right - left + group_gap
    return positions


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _components(by_id, parents, children, spouses) -> list[list[int]]:
    """Connected family branches, largest/oldest first."""
    seen: set[int] = set()
    comps: list[list[int]] = []
    for start in sorted(by_id):
        if start in seen:
            continue
        seen.add(start)
        comp, stack = [], [start]
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in (*parents[cur], *children.get(cur, ()), *spouses[cur]):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(comp)
    comps.sort(key=lambda c: (-len(c), min(c)))
    return comps


def _blocks(members: list[int], spouses) -> list[list[int]]:
    """Spouse-connected groups within one band."""
    here = set(members)
    seen: set[int] = set()
    out: list[list[int]] = []
    for pid in members:
        if pid in seen:
            continue
        seen.add(pid)
        comp, stack = [], [pid]
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for s in spouses[cur]:
                if s in here and s not in seen:
                    seen.add(s)
                    stack.append(s)
        out.append(comp)
    return out


def _chain(block: list[int], spouses, pref: dict[int, tuple]) -> list[int]:
    """Order a spouse block as a chain so every couple stays adjacent."""
    if len(block) <= 1:
        return list(block)
    inside = set(block)
    adj = {p: [q for q in spouses[p] if q in inside] for p in block}
    ends = [p for p in block if len(adj[p]) == 1]
    cur: Optional[int] = min(ends or block, key=lambda p: pref[p])
    order: list[int] = []
    visited: set[int] = set()
    while cur is not None:
        order.append(cur)
        visited.add(cur)
        nxt = [q for q in adj[cur] if q not in visited]
        cur = min(nxt, key=lambda q: pref[q]) if nxt else None
    order.extend(sorted((p for p in block if p not in visited), key=lambda p: pref[p]))
    return order


def _isotonic(targets: list[float], weights: list[float]) -> list[float]:
    """Weighted least-squares non-decreasing fit (pool adjacent violators)."""
    pools: list[list[float]] = []  # [value, weight, count]
    for t, w in zip(targets, weights):
        pools.append([t, w, 1])
        while len(pools) > 1 and pools[-2][0] > pools[-1][0]:
            v2, w2, c2 = pools.pop()
            v1, w1, c1 = pools.pop()
            wt = w1 + w2
            pools.append([(v1 * w1 + v2 * w2) / wt, wt, c1 + c2])
    out: list[float] = []
    for v, _, c in pools:
        out.extend([v] * int(c))
    return out


def _layout_component(
    comp, band, parents, children, spouses, key, *, box_w, gap, group_gap
) -> dict[int, float]:
    """Left x per person of one connected branch."""
    step = box_w + gap
    rows: dict[int, list[int]] = defaultdict(list)
    for pid in comp:
        rows[band[pid]].append(pid)
    bands = sorted(rows)
    blocks_of = {b: _blocks(sorted(rows[b], key=lambda p: key[p]), spouses) for b in bands}

    # Block centre per person's x: x[pid] is the box centre.
    x: dict[int, float] = {}

    def family(order: list[int]) -> Optional[tuple]:
        for m in order:
            if parents[m]:
                return tuple(parents[m])
        return None

    def place(entries: list[tuple]) -> None:
        """entries: (desired_centre, tiebreak, ordered_members, weight)."""
        entries.sort(key=lambda e: (e[0], e[1]))
        targets, weights, offsets, widths = [], [], [], []
        cursor = 0.0
        prev_family: object = object()
        for i, (desired, _tb, order, weight) in enumerate(entries):
            width = len(order) * box_w + (len(order) - 1) * gap
            fam = family(order)
            if i:
                same = fam is not None and fam == prev_family
                cursor += gap if same else group_gap
            offsets.append(cursor)
            widths.append(width)
            targets.append(desired - width / 2 - cursor)
            weights.append(weight)
            cursor += width
            prev_family = fam
        fitted = _isotonic(targets, weights)
        for (_d, _tb, order, _w), y, off in zip(entries, fitted, offsets):
            left = y + off
            for i, m in enumerate(order):
                x[m] = left + i * step + box_w / 2

    def member_offsets(order: list[int]) -> dict[int, float]:
        n = len(order)
        return {m: (i - (n - 1) / 2) * step for i, m in enumerate(order)}

    def sweep_down(b: int) -> None:
        entries = []
        fallback = max(x.values(), default=0.0)
        for block in blocks_of[b]:
            wanted = {
                m: sum(x[p] for p in parents[m]) / len(parents[m])
                for m in block
                if parents[m] and all(p in x for p in parents[m])
            }
            pref = {m: (wanted.get(m, x.get(m, fallback)), key[m]) for m in block}
            order = _chain(block, spouses, pref)
            offs = member_offsets(order)
            if wanted:
                desired = sum(wanted[m] - offs[m] for m in wanted) / len(wanted)
            elif all(m in x for m in order):
                desired = sum(x[m] for m in order) / len(order)
            else:
                fallback += step * (len(order) + 1)
                desired = fallback
            entries.append((desired, (family(order) or (), key[order[0]]), order, len(order)))
        place(entries)

    def sweep_up(b: int) -> None:
        entries = []
        for block in blocks_of[b]:
            order = sorted(block, key=lambda m: x[m])
            centre = sum(x[m] for m in order) / len(order)
            inside = set(order)
            shifts = []
            for m in order:
                for c in children.get(m, ()):
                    ps = [p for p in parents[c] if p in inside]
                    if m != ps[0]:
                        continue  # count each child once per block
                    junction = sum(x[p] for p in ps) / len(ps)
                    shifts.append(x[c] - junction)
            desired = centre + (sum(shifts) / len(shifts) if shifts else 0.0)
            entries.append((desired, (family(order) or (), key[order[0]]), order, len(order)))
        place(entries)

    for b in bands:
        sweep_down(b)
    for _ in range(SWEEPS):
        for b in reversed(bands[:-1]):
            sweep_up(b)
        for b in bands[1:]:
            sweep_down(b)

    return {pid: cx - box_w / 2 for pid, cx in x.items()}
