"""Tree Studio view model: visibility, storage, root lookup, JSON exchange."""

from __future__ import annotations

import pytest

from app.db.database import init_db, session_scope
from app.db.models import Person
from app.services.family_tree_service import FamilyTreeGraph, FamilyTreeNode, FamilyTreeService
from app.services.tree_view_service import (
    RootRef,
    TreeViewError,
    TreeViewService,
    TreeViewState,
    collapsible_ids,
    filter_graph,
    reveal,
    visible_ids,
)

# ---------------------------------------------------------------------------
# Hand-built graph
#
#   R (root) == RS
#      |
#      P == S          Q (child of R) -- Q is also parent of C1
#     / \\
#   C1   C2
#   |
#   G
# ---------------------------------------------------------------------------

R, RS, P, S, Q, C1, C2, G = range(1, 9)


def _graph():
    def n(pid, gen, parents=(), children=(), spouses=()):
        return FamilyTreeNode(pid, f"N{pid}", gen, tuple(parents), tuple(children), tuple(spouses))

    nodes = {
        R: n(R, 0, children=(P, Q), spouses=(RS,)),
        RS: n(RS, 0, children=(P, Q), spouses=(R,)),
        P: n(P, 1, parents=(R, RS), children=(C1, C2), spouses=(S,)),
        S: n(S, 1, children=(C1, C2), spouses=(P,)),
        Q: n(Q, 1, parents=(R, RS), children=(C1,)),
        C1: n(C1, 2, parents=(P, S, Q), children=(G,)),
        C2: n(C2, 2, parents=(P, S)),
        G: n(G, 3, parents=(C1,)),
    }
    return FamilyTreeGraph(root_id=R, nodes=nodes)


def test_everything_visible_by_default():
    assert visible_ids(_graph(), TreeViewState()) == set(range(1, 9))


def test_collapse_keeps_spouse_and_hides_descendants():
    g = _graph()
    g.nodes[Q] = FamilyTreeNode(Q, "NQ", 1, (R, RS), ())  # Q not a parent of C1 here
    g.nodes[C1] = FamilyTreeNode(C1, "N", 2, (P, S), (G,))
    vis = visible_ids(g, TreeViewState(collapsed={P}))
    assert {P, S} <= vis
    assert not vis & {C1, C2, G}


def test_collapse_keeps_child_reachable_another_way():
    vis = visible_ids(_graph(), TreeViewState(collapsed={P}))
    assert C1 in vis and G in vis  # via Q
    assert C2 not in vis


def test_expand_restores_except_hidden():
    state = TreeViewState(collapsed={P}, hidden={C2})
    state.collapsed.discard(P)
    vis = visible_ids(_graph(), state)
    assert {C1, G} <= vis and C2 not in vis


def test_hide_spouse_and_show_again():
    state = TreeViewState(hidden={S})
    vis = visible_ids(_graph(), state)
    assert S not in vis and P in vis
    state.hidden.discard(S)
    assert S in visible_ids(_graph(), state)


def test_hidden_person_hides_persons_only_reachable_through_them():
    g = _graph()
    vis = visible_ids(g, TreeViewState(hidden={C1}))
    assert G not in vis


def test_floating_added():
    g = _graph()
    g.nodes[99] = FamilyTreeNode(99, "X", 0)
    state = TreeViewState(floating=[99])
    assert 99 in visible_ids(g, state)


def test_reveal_expands_ancestors_and_unhides():
    g = _graph()
    state = TreeViewState(collapsed={C1}, hidden={G})
    assert G not in visible_ids(g, state)
    assert reveal(g, state, G)
    assert G in visible_ids(g, state)
    assert C1 not in state.collapsed
    assert not reveal(g, state, 1234)


def test_collapsible_and_filter_graph():
    g = _graph()
    assert collapsible_ids(g) == {R, RS, P, S, Q, C1}
    small = filter_graph(g, {R, RS, P})
    assert small.node(P).child_ids == ()
    assert small.node(R).child_ids == (P,)


def test_fields_for_person_override():
    state = TreeViewState(fields=["b:birth_date", "b:nickname"])
    state.set_person_field(5, "b:birth_date", False)
    state.set_person_field(5, "c:3", True)
    assert state.fields_for(5, ["b:nickname", "b:birth_date", "c:3"]) == ["b:nickname", "c:3"]
    assert state.fields_for(6, ["b:nickname", "b:birth_date", "c:3"]) == ["b:birth_date", "b:nickname"]
    state.set_person_field(5, "b:birth_date", True)
    state.set_person_field(5, "c:3", False)
    assert 5 not in state.person_fields


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "views.db")


def _people(session, *specs):
    ids = []
    for name, code, ext in specs:
        p = Person(name=name, is_auto_named=False, family_code=code, external_family_code=ext)
        session.add(p)
        session.flush()
        ids.append(p.id)
    return ids


def _state(a, b, c):
    st = TreeViewState(root=RootRef(a), collapsed={a}, hidden={b}, floating=[c],
                       fields=["b:nickname", "c:1"])
    st.set_person_field(b, "b:birth_date", True)
    return st


def test_crud_and_round_trip(db):
    with session_scope() as s:
        a, b, c = _people(s, ("A", "C", None), ("B", "C1", None), ("C", None, "#x#"))
        svc = TreeViewService(s)
        v = svc.save("Nagyszülők", _state(a, b, c))
        with pytest.raises(TreeViewError):
            svc.save("nagyszülők", TreeViewState())
        vid = v.id
    with session_scope() as s:
        svc = TreeViewService(s)
        loaded = svc.load(vid)
        assert loaded.to_json() == _state(a, b, c).to_json()
        svc.rename(vid, "Ősök")
        dup = svc.duplicate(vid)
        assert dup.name == "Ősök (2)"
        assert [v.name for v in svc.list_views()] == ["Ősök", "Ősök (2)"]
        svc.delete(dup.id)
        assert len(svc.list_views()) == 1


def test_load_drops_deleted_person(db):
    with session_scope() as s:
        a, b, c = _people(s, ("A", None, None), ("B", None, None), ("C", None, None))
        vid = TreeViewService(s).save("V", _state(a, b, c)).id
    with session_scope() as s:
        s.delete(s.get(Person, b))
    with session_scope() as s:
        st = TreeViewService(s).load(vid)
        assert b not in st.hidden and b not in st.person_fields
        assert st.collapsed == {a}


def test_root_lookup_by_external_code(db):
    with session_scope() as s:
        z, z1, z2 = _people(s, ("Z", None, "#Zoli#"), ("Z1", None, "#Zoli#1"), ("Z2", None, "#Zoli#2"))
        svc = FamilyTreeService(s)
        assert svc.find_root_by_external_code("#Zoli#") == (z, [])
        s.get(Person, z).external_family_code = "#Zolika#"
        s.flush()
        pid, cands = svc.find_root_by_external_code("#Zoli#")
        assert pid is None
        assert [c[0] for c in cands] == [z1, z2]
        assert svc.find_root_by_external_code("#Nincs#") == (None, [])


def test_json_same_db_round_trip(db):
    with session_scope() as s:
        a, b, c = _people(s, ("A", "C", None), ("B", "C1", None), ("C", None, "#x#"))
        svc = TreeViewService(s)
        doc = svc.export_document("Export", _state(a, b, c))
        res = svc.import_document(doc)
        assert res.unmatched == []
        assert res.name == "Export"
        assert svc.load(res.view_id).to_json() == _state(a, b, c).to_json()
        res2 = svc.import_document(doc)
        assert res2.name == "Export (2)"


def test_json_import_falls_back_to_family_code_and_reports_unknown(db):
    with session_scope() as s:
        (b,) = _people(s, ("B renamed", "C1", None))
        # Exported elsewhere: ids 900-902; B only matches by family code.
        doc = _state(900, 902, 901).to_json()
        doc["name"] = "X"
        doc["persons"] = {
            "900": {"name": "A", "familyCode": "Q", "externalFamilyCode": None, "birthDate": None},
            "901": {"name": "C", "familyCode": None, "externalFamilyCode": None, "birthDate": "1900"},
            "902": {"name": "B", "familyCode": "C1", "externalFamilyCode": None, "birthDate": None},
        }
        svc = TreeViewService(s)
        res = svc.import_document(doc)
        st = svc.load(res.view_id)
        assert st.hidden == {b}
        assert sorted(u["name"] for u in res.unmatched) == ["A", "C"]
        assert st.root.person_id is None and st.floating == []


def test_json_import_id_with_disagreeing_data_falls_back(db):
    with session_scope() as s:
        a, b = _people(s, ("A", "K1", None), ("B", "K2", None))
        doc = TreeViewState(hidden={a}).to_json()
        # Same id, but the data belongs to B -> matched by family code to B.
        doc["persons"] = {str(a): {"name": "B?", "familyCode": "K2"}}
        res = TreeViewService(s).import_document(doc, name="Y")
        assert TreeViewService(s).load(res.view_id).hidden == {b}


# --- friends (add-family-tree-friend-lines) ----------------------------------


def _friend_graph():
    g = _graph()
    # F (100) friend of P and Q; S is also a friend of R (already a member).
    g.friends = {P: (100,), Q: (100,), R: (S,)}
    g.friend_nodes = {100: FamilyTreeNode(100, "F", 1)}
    return g


def test_visible_friends_modes_hidden_and_member():
    from app.services.tree_view_service import visible_friends

    g = _friend_graph()
    vis = visible_ids(g, TreeViewState())
    assert visible_friends(g, "off", (), vis) == {}
    assert visible_friends(g, "all", (), vis) == {100: [P, Q], S: [R]}
    assert visible_friends(g, "selected", {P}, vis) == {100: [P]}
    assert visible_friends(g, "all", (), vis, hidden={100}) == {S: [R]}
    assert TreeViewState().friend_mode == "off"


def test_apply_friends_one_satellite_two_edges_member_linked():
    from app.services.tree_view_service import apply_friends

    g = _friend_graph()
    shown = filter_graph(g, visible_ids(g, TreeViewState()))
    apply_friends(shown, g, "all")
    assert 100 in shown.nodes and shown.satellites == {100: P}
    assert sorted(shown.friend_edges) == sorted([(100, P), (100, Q), (S, R)])
    assert S not in shown.satellites  # already a member: only linked


def test_friend_mode_json_and_db_round_trip(db):
    with session_scope() as s:
        (a,) = _people(s, ("A", "C81", None))
        st = TreeViewState(root=RootRef(a), friend_mode="selected", friend_anchors={a})
        svc = TreeViewService(s)
        vid = svc.save("Barátok", st).id
        loaded = svc.load(vid)
        assert loaded.friend_mode == "selected" and loaded.friend_anchors == {a}
        assert TreeViewState.from_json({"friendMode": "bogus"}).friend_mode == "off"
