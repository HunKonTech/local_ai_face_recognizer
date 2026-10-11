"""Tests for deriving relationships from family codes (#151).

Covers the full interpreter grammar — root person ``C0`` and its spouse
``C00``, ancestors ``F``, siblings ``T`` — so codes alone build the tree.
"""

from __future__ import annotations

import pytest

from app.db.database import init_db, session_scope
from app.db.models import Person
from app.services.family_service import FamilyService, derive_code_relations


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "code_links.db")


def _rels(code):
    return [(r.kind, r.ref.code, r.ref.is_root) for r in derive_code_relations(code)]


def test_derive_root_spouse_and_children():
    assert _rels("C0") == [("spouse", "C", False)]  # legacy: spouse of a "C"
    assert _rels("C00") == [("spouse", "C", True)]
    assert _rels("C8") == [("parent", "C", True)]
    assert _rels("C80") == [("spouse", "C8", False)]
    assert _rels("C81") == [("parent", "C8", False)]
    assert _rels("C801") == [("parent", "C80", False)]  # stepchild


def test_derive_ancestors_siblings_and_numbered_spouses():
    assert _rels("C0F1") == [("child", "C", True), ("spouse", "C0F2", False)]
    assert _rels("C0F12") == [("child", "C0F1", False), ("spouse", "C0F11", False)]
    assert _rels("C00F2") == [("child", "C00", False), ("spouse", "C00F1", False)]
    assert _rels("C00T1") == [
        ("sibling", "C00", False),
        ("parent", "C00F1", False),
        ("parent", "C00F2", False),
    ]
    assert _rels("C00T10") == [("spouse", "C00T1", False)]
    assert _rels("C00T11") == [("parent", "C00T1", False)]
    assert _rels("G1H2") == [("spouse", "G1", False)]
    assert _rels("G1H21") == [("parent", "G1H2", False)]
    assert _rels("G1{2}3") == [("parent", "G1", False), ("parent", "G1H2", False)]


def test_derive_ignores_friend_and_multi_codes():
    assert _rels("C81B") == []
    assert _rels("C8B,C80B") == []
    assert _rels("") == []


def _add(session, name, code, gender=None):
    p = Person(name=name, is_auto_named=False, family_code=code, gender=gender)
    session.add(p)
    session.flush()
    return p.id


def test_regenerate_links_whole_family_from_codes(db):
    """The screenshot case: Cikky (C0), her husband (C00) and both sets of
    parents (C0F1/C0F2, C00F1/C00F2) link up from the codes alone."""
    with session_scope() as session:
        ids = {
            code: _add(session, code, code)
            for code in ("C0", "C00", "C0F1", "C0F2", "C00F1", "C00F2", "C3", "C30",
                         "C00T1", "C00T11")
        }
        fam = FamilyService(session)
        fam.regenerate_all_links()

        assert fam.are_spouses(ids["C0"], ids["C00"])
        assert fam.is_parent_child(ids["C0F1"], ids["C0"])
        assert fam.is_parent_child(ids["C0F2"], ids["C0"])
        assert fam.are_spouses(ids["C0F1"], ids["C0F2"])
        assert fam.is_parent_child(ids["C00F1"], ids["C00"])
        assert fam.is_parent_child(ids["C00F2"], ids["C00"])
        assert fam.are_spouses(ids["C00F1"], ids["C00F2"])
        assert fam.is_parent_child(ids["C0"], ids["C3"])  # child of the root C0
        assert fam.are_spouses(ids["C3"], ids["C30"])
        # Sibling of C00 shares C00's parents; its child hangs under it.
        assert fam.is_parent_child(ids["C00F1"], ids["C00T1"])
        assert fam.is_parent_child(ids["C00F2"], ids["C00T1"])
        assert fam.is_parent_child(ids["C00T1"], ids["C00T11"])


def test_legacy_c_and_c0_codes_still_link(db):
    """Older data: C = root, C0 = root's spouse, C8 = their child."""
    with session_scope() as session:
        root = _add(session, "Root", "C")
        spouse = _add(session, "Spouse", "C0")
        kid = _add(session, "Kid", "C8")
        fam = FamilyService(session)
        fam.regenerate_all_links()
        assert fam.are_spouses(root, spouse)
        assert fam.is_parent_child(root, kid)
        assert not fam.is_parent_child(spouse, kid)  # inferred at render time


def test_saving_a_referenced_person_links_dependents(db):
    """Coding the child first and its father later still links them."""
    with session_scope() as session:
        cikky = _add(session, "Cikky", "C0")
        father = _add(session, "Father", None)
        fam = FamilyService(session)
        session.get(Person, father).family_code = "C0F1"
        session.flush()
        fam.link_derived_parents(cikky)  # e.g. Cikky re-saved, father pre-coded
        assert fam.is_parent_child(father, cikky)

        mother = _add(session, "Mother", "C0F2")
        fam.link_derived_parents(mother)
        assert fam.is_parent_child(mother, cikky)
        assert fam.are_spouses(father, mother)
