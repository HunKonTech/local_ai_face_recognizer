"""Tests for the shared family-tree placement (#151: children overlapping)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.services.family_tree_layout import compute_layout
from app.services.family_tree_service import assign_generations

BOX_W, GAP = 100.0, 20.0


@dataclass
class N:
    person_id: int
    generation: int
    name: str = ""
    parent_ids: tuple = ()
    spouse_ids: tuple = ()
    birth_date: Optional[str] = None


@dataclass
class Family:
    nodes: dict = field(default_factory=dict)

    def add(self, pid, gen, parents=(), spouses=(), name=None):
        self.nodes[pid] = N(pid, gen, name or str(pid), tuple(parents), tuple(spouses))

    def marry(self, a, b):
        self.nodes[a].spouse_ids += (b,)
        self.nodes[b].spouse_ids += (a,)


def _two_families() -> Family:
    """Two top couples, each with four children; one child of each marries."""
    f = Family()
    f.add(1, 0); f.add(2, 0); f.marry(1, 2)
    f.add(3, 0); f.add(4, 0); f.marry(3, 4)
    for pid in (10, 11, 12, 13):
        f.add(pid, 1, parents=(1, 2))
    for pid in (20, 21, 22, 23):
        f.add(pid, 1, parents=(3, 4))
    f.marry(13, 20)
    for pid in (30, 31, 32):
        f.add(pid, 2, parents=(13, 20))
    f.add(40, 2, parents=(10,))
    return f


def _centre(pos, pid):
    return pos[pid][0] + BOX_W / 2


def test_no_boxes_overlap_in_a_row():
    pos = compute_layout(_two_families().nodes.values(), box_w=BOX_W, gap=GAP)
    rows = {}
    for pid, (x, band) in pos.items():
        rows.setdefault(band, []).append(x)
    for xs in rows.values():
        xs.sort()
        for a, b in zip(xs, xs[1:]):
            assert b - a >= BOX_W + GAP - 1e-6


def test_siblings_are_contiguous_and_under_their_parents():
    f = _two_families()
    pos = compute_layout(f.nodes.values(), box_w=BOX_W, gap=GAP)
    row1 = sorted((p for p, (_, b) in pos.items() if b == 1), key=lambda p: pos[p][0])
    left_family = [p for p in row1 if p in (10, 11, 12, 13)]
    right_family = [p for p in row1 if p in (20, 21, 22, 23)]
    # Each sibling set is one contiguous run, not interleaved with the other.
    assert row1 == left_family + right_family or row1 == right_family + left_family
    # The married siblings sit next to each other between the two families.
    assert abs(row1.index(13) - row1.index(20)) == 1
    # Parents' midpoint lies within the span of their children.
    for (a, b), kids in (((1, 2), left_family), ((3, 4), right_family)):
        mid = (_centre(pos, a) + _centre(pos, b)) / 2
        assert min(_centre(pos, k) for k in kids) <= mid <= max(_centre(pos, k) for k in kids)


def test_couple_children_centred_below_the_couple():
    f = _two_families()
    pos = compute_layout(f.nodes.values(), box_w=BOX_W, gap=GAP)
    couple_mid = (_centre(pos, 13) + _centre(pos, 20)) / 2
    kids_mid = sum(_centre(pos, k) for k in (30, 31, 32)) / 3
    assert abs(couple_mid - kids_mid) < BOX_W


def test_unrelated_branches_do_not_interleave():
    f = _two_families()
    f.add(100, 0); f.add(101, 1, parents=(100,)); f.add(102, 1, parents=(100,))
    pos = compute_layout(f.nodes.values(), box_w=BOX_W, gap=GAP)
    other = {100, 101, 102}
    main_xs = [x for p, (x, _) in pos.items() if p not in other]
    other_xs = [x for p, (x, _) in pos.items() if p in other]
    assert min(other_xs) > max(main_xs) or max(other_xs) < min(main_xs)


def test_married_in_spouse_parents_sit_directly_above():
    """A spouse's parents are pulled down next to their child's band."""
    parents = {2: {1}, 3: {2}, 5: {4}}
    spouses = {3: {5}, 5: {3}}
    #   1           grandparent
    #   2           parent
    #   3 == 5      couple; 5's own parent is 4 (no further ancestors)
    gen = assign_generations({1, 2, 3, 4, 5}, parents, spouses)
    assert gen[3] == gen[5] == 2
    assert gen[4] == 1  # not left floating in the top band


def test_married_in_spouse_parents_take_siblings_along():
    """#194: a married-in spouse's sibling must not hold their parents up."""
    #   1                 great-grandparent
    #   2                 grandparent
    #   3                 parent
    #   4 == 7            couple; 7's parents 5 == 6, 7's sibling 8
    parents = {2: {1}, 3: {2}, 4: {3}, 7: {5, 6}, 8: {5, 6}}
    spouses = {4: {7}, 7: {4}, 5: {6}, 6: {5}}
    gen = assign_generations(set(range(1, 9)), parents, spouses)
    assert gen[4] == gen[7] == gen[8] == 3
    assert gen[5] == gen[6] == 2
    assert gen[1] == 0


def test_married_in_spouse_parents_drag_sibling_family_down():
    """The sibling's own children move down with the sibling."""
    parents = {3: {9}, 4: {3}, 5: {4}, 6: {1, 2}, 7: {1, 2}, 8: {7}}
    spouses = {1: {2}, 2: {1}, 5: {6}, 6: {5}}
    #   9 -> 3 -> 4 -> 5 == 6 ; 6 and 7 are children of 1 == 2 ; 8 is 7's child
    gen = assign_generations({1, 2, 3, 4, 5, 6, 7, 8, 9}, parents, spouses)
    assert gen[5] == gen[6] == gen[7] == 3
    assert gen[1] == gen[2] == 2
    assert gen[8] == 4
