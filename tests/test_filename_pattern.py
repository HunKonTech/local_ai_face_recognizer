"""Tests for the ``#Token#`` filename pattern renderer (#175)."""

from __future__ import annotations

from app.services.filename_pattern import (
    available_tokens,
    render_pattern,
    resolve_token,
    safe_filename,
    safe_pattern_filename,
    unknown_tokens,
)


def _values(**overrides) -> dict:
    values = {
        "family_code": "C44",
        "external_family_code": "",
        "name": "Horváth Merse",
        "name_prefix": "",
        "last_name": "Horváth",
        "first_name": "Merse",
        "nickname": "",
        "date": "2023.06.17",
        "year": "2023",
        "source_name": "IMG_0042",
        "index": 1,
        "face_id": 7,
        "image_id": 3,
    }
    values.update(overrides)
    return values


# ---------------------------------------------------------------------------
# The two concrete cases from issue #175
# ---------------------------------------------------------------------------


def test_issue_example_full_pattern():
    pattern = "portré-#CSID#-#Vezeték név# #Keresztnév#-#Dátum#.jpg"
    assert (
        render_pattern(pattern, _values())
        == "portré-C44-Horváth Merse-2023.06.17.jpg"
    )


def test_issue_example_missing_family_code_collapses_separators():
    pattern = "portré-#CSID#-#Név#-#Dátum#.jpg"
    values = _values(
        family_code="",
        name="Szilvay Géza (1920-1992)",
        date="1920 körül",
    )
    assert (
        render_pattern(pattern, values)
        == "portré-Szilvay Géza (1920-1992)-1920 körül.jpg"
    )


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------


def test_aliases_are_case_and_whitespace_insensitive():
    assert resolve_token("Vezetéknév") == "last_name"
    assert resolve_token("vezeték név") == "last_name"
    assert resolve_token("VEZETÉK-NÉV") == "last_name"
    assert resolve_token("LastName") == "last_name"


def test_every_token_has_a_unique_key_and_an_i18n_key():
    specs = available_tokens()
    keys = [s.key for s in specs]
    assert len(keys) == len(set(keys))
    assert all(s.i18n_key and s.aliases for s in specs)


def test_unknown_tokens_are_reported_and_dropped():
    pattern = "portré-#Nincsilyen#-#Név#.jpg"
    assert unknown_tokens(pattern) == ["Nincsilyen"]
    assert render_pattern(pattern, _values()) == "portré-Horváth Merse.jpg"


def test_adjacent_tokens_do_not_merge():
    assert render_pattern("#CSID##Év#.jpg", _values()) == "C442023.jpg"


def test_numeric_tokens_render_as_text():
    out = render_pattern("#Kép ID#_#Arc ID#_#Sorszám#.jpg", _values())
    assert out == "3_7_1.jpg"


# ---------------------------------------------------------------------------
# Tidying and extension handling
# ---------------------------------------------------------------------------


def test_all_tokens_empty_leaves_only_the_literal_text():
    values = {k: "" for k in _values()}
    assert render_pattern("portré-#CSID#-#Név#.jpg", values) == "portré.jpg"


def test_leading_and_trailing_separators_are_trimmed():
    values = _values(family_code="")
    assert render_pattern("#CSID#-#Név#-#Becenév#.jpg", values) == "Horváth Merse.jpg"


def test_a_date_looking_tail_is_not_taken_for_an_extension():
    # No ".jpg" in the pattern: the trailing "1920" must stay in the stem.
    assert render_pattern("portré-#Dátum#", _values(date="kb. 1920")) == "portré-kb. 1920"


def test_pattern_without_tokens_is_returned_as_is():
    assert render_pattern("plain name.jpg", _values()) == "plain name.jpg"


# ---------------------------------------------------------------------------
# Filesystem safety
# ---------------------------------------------------------------------------


def test_safe_filename_strips_hostile_characters_but_keeps_dots():
    assert safe_filename('a/b:c*d?"e<f>g|h') == "a_b_c_d__e_f_g_h"
    assert safe_filename("2023.06.17") == "2023.06.17"


def test_safe_filename_falls_back_when_nothing_remains():
    assert safe_filename("", fallback="face") == "face"
    assert safe_filename("   ", fallback="face") == "face"


def test_safe_pattern_filename_keeps_the_extension_when_the_stem_is_long():
    long_name = "N" * 400
    out = safe_pattern_filename("#Név#.jpg", _values(name=long_name))
    assert out.endswith(".jpg")
    assert len(out) < 200


def test_a_value_containing_hash_is_not_re_scanned_for_tokens():
    # External family codes look like "#root#path" — a second pass would
    # mistake "root" for a token.
    out = render_pattern("p-#Külső CSID#.jpg", _values(external_family_code="#Nagy#2a"))
    assert out == "p-#Nagy#2a.jpg"
