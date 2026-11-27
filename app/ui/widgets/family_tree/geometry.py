"""Qt-free geometry and text helpers for the family-tree view."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional


def lineage_sets(graph, selected_id) -> tuple[set[int], set[int], set[int]]:
    """Transitive ancestors, descendants, and direct spouses of selected person."""
    if selected_id is None or selected_id not in graph.nodes:
        return set(), set(), set()

    def _walk(start: int, attr: str) -> set[int]:
        seen: set[int] = set()
        stack = [start]
        while stack:
            node = graph.node(stack.pop())
            if node is None:
                continue
            for nxt in getattr(node, attr):
                if nxt not in seen and nxt != selected_id:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    node = graph.node(selected_id)
    spouses = set(node.spouse_ids) if node else set()
    return _walk(selected_id, "parent_ids"), _walk(selected_id, "child_ids"), spouses


def _bus_tracks(
    families: list[tuple[float, list[float]]], v_gap: float
) -> list[float]:
    """Bus y per sibling group so buses sharing a row gap never overlap.

    ``families`` holds ``(parents_bottom_y, connection_xs)`` per group. Groups
    whose horizontal spans overlap in the same gap get separate tracks
    (greedy interval colouring), spread evenly around the gap's middle.
    """
    out = [0.0] * len(families)
    by_gap: dict[float, list[int]] = defaultdict(list)
    for i, (py, _) in enumerate(families):
        by_gap[round(py, 1)].append(i)
    for py, idxs in by_gap.items():
        spans = {i: (min(families[i][1]), max(families[i][1])) for i in idxs}
        track_end: list[float] = []
        track_of: dict[int, int] = {}
        for i in sorted(idxs, key=lambda i: spans[i]):
            left, right = spans[i]
            for t, end in enumerate(track_end):
                if end < left - 6:
                    track_of[i] = t
                    track_end[t] = right
                    break
            else:
                track_of[i] = len(track_end)
                track_end.append(right)
        k = len(track_end)
        spacing = min(10.0, (v_gap - 20) / max(k - 1, 1))
        for i in idxs:
            out[i] = py + v_gap / 2 + (track_of[i] - (k - 1) / 2) * spacing
    return out


def _marriage_label(
    start: Optional[str], end: Optional[str], place: Optional[str] = None
) -> str:
    """Compact marriage label for the spouse line: '1984–1990 · Budapest'."""
    start = (start or "").strip()
    end = (end or "").strip()
    place = (place or "").strip()
    if start and end:
        date_part = f"{start}–{end}"
    elif start:
        date_part = start
    elif end:
        date_part = f"–{end}"
    else:
        date_part = ""
    if date_part and place:
        return f"{date_part} · {place}"
    return date_part or place


def _years(birth: Optional[str], death: Optional[str]) -> str:
    if not birth and not death:
        return ""
    return f"{birth or ''}–{death or ''}"


def _elide(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
