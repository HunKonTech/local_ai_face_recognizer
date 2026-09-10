"""Filename pattern rendering with ``#Token#`` placeholders.

Used by the "export the faces of an image into separate files" feature (#175):
the user supplies a pattern such as::

    portré-#CSID#-#Vezetéknév# #Keresztnév#-#Dátum#.jpg

which renders to ``portré-C44-Horváth Merse-2023.06.17.jpg``.

Tokens whose value is missing must not leave the filename ugly, so after
substitution the leftover separator runs are collapsed and trimmed::

    portré-#CSID#-#Név#-#Dátum#.jpg   (no family code)
        →  portré-Szilvay Géza (1920-1992)-1920 körül.jpg

The module is deliberately UI-free (no Qt, no i18n lookups) so it can be unit
tested and reused by other exports.  Token *names* are matched
case-insensitively and ignore spaces/hyphens, so ``#Vezeték név#``,
``#vezetéknév#`` and ``#LastName#`` are the same token.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

# Any ``#...#`` group.  Non-greedy and it must not contain a ``#`` itself, so
# two adjacent tokens never merge into one match.
_TOKEN_RE = re.compile(r"#([^#]+)#")

# Characters we treat as "glue" between filename parts.  When a token in the
# middle renders empty, the glue on both sides collapses into one.
_SEPARATORS = "-_ "

# Characters a filesystem (Windows in particular) refuses in a name.  Note the
# dot is *not* in the list: "2023.06.17" must survive intact.
_UNSAFE_RE = re.compile(r'[\\/:*?"<>|\r\n\t]')

MAX_STEM_LENGTH = 120


@dataclass(frozen=True)
class TokenSpec:
    """One recognised placeholder.

    Attributes:
        key:      Canonical key used in the value mapping (e.g. ``"last_name"``).
        aliases:  Accepted spellings, Hungarian first (shown in the UI help).
        i18n_key: Key of the human description in :mod:`app.ui.i18n`.
    """

    key: str
    aliases: Tuple[str, ...]
    i18n_key: str

    @property
    def display(self) -> str:
        """The primary spelling, wrapped in ``#`` for display/insertion."""
        return f"#{self.aliases[0]}#"


# Order matters: this is the order of the help table in the dialog.
_TOKENS: Tuple[TokenSpec, ...] = (
    TokenSpec("family_code", ("CSID", "FamilyCode"), "fexp_tok_family_code"),
    TokenSpec(
        "external_family_code",
        ("Külső CSID", "ExtID", "ExternalFamilyCode"),
        "fexp_tok_external_family_code",
    ),
    TokenSpec("name", ("Név", "Name"), "fexp_tok_name"),
    TokenSpec("name_prefix", ("Előtag", "Prefix"), "fexp_tok_name_prefix"),
    TokenSpec("last_name", ("Vezetéknév", "Vezeték név", "LastName"), "fexp_tok_last_name"),
    TokenSpec("first_name", ("Keresztnév", "FirstName"), "fexp_tok_first_name"),
    TokenSpec("nickname", ("Becenév", "Nickname"), "fexp_tok_nickname"),
    TokenSpec("date", ("Dátum", "Date"), "fexp_tok_date"),
    TokenSpec("year", ("Év", "Year"), "fexp_tok_year"),
    TokenSpec("source_name", ("Fájlnév", "FileName"), "fexp_tok_source_name"),
    TokenSpec("index", ("Sorszám", "Index"), "fexp_tok_index"),
    TokenSpec("face_id", ("Arc ID", "FaceID"), "fexp_tok_face_id"),
    TokenSpec("image_id", ("Kép ID", "ImageID"), "fexp_tok_image_id"),
)


def _normalise(alias: str) -> str:
    """Fold an alias to its lookup form: lowercase, no spaces/hyphens."""
    return re.sub(r"[\s\-_]+", "", alias).casefold()


_ALIAS_TO_KEY: Dict[str, str] = {
    _normalise(alias): spec.key for spec in _TOKENS for alias in spec.aliases
}


def available_tokens() -> List[TokenSpec]:
    """All recognised tokens, in display order."""
    return list(_TOKENS)


def token_keys() -> List[str]:
    """Canonical keys of all recognised tokens."""
    return [spec.key for spec in _TOKENS]


def resolve_token(raw: str) -> Optional[str]:
    """Return the canonical key for the token text *raw*, or ``None``."""
    return _ALIAS_TO_KEY.get(_normalise(raw))


def unknown_tokens(pattern: str) -> List[str]:
    """Token texts in *pattern* that are not recognised (for live validation)."""
    seen: List[str] = []
    for match in _TOKEN_RE.finditer(pattern or ""):
        raw = match.group(1)
        if resolve_token(raw) is None and raw not in seen:
            seen.append(raw)
    return seen


def _tidy(text: str) -> str:
    """Collapse separator runs left behind by empty tokens and trim the edges."""
    # A run of separators (optionally with stray commas) becomes the first
    # separator of the run — "portré--Szilvay" → "portré-Szilvay",
    # "Horváth  Merse" → "Horváth Merse".
    def _collapse(match: "re.Match[str]") -> str:
        run = match.group(0)
        for ch in run:
            if ch in "-_":
                return ch
        return " "

    text = re.sub(r"[\-_ ,]{2,}", _collapse, text)
    return text.strip(_SEPARATORS + ",")


def render_pattern(pattern: str, values: Mapping[str, object]) -> str:
    """Render *pattern*, substituting recognised tokens from *values*.

    Missing or empty values render as an empty string; unknown tokens are also
    dropped (the dialog warns about them separately).  The suffix is preserved:
    only the stem is tidied, so a trailing ``.jpg`` never gets mangled.

    Substitution is single-pass — a value that itself contains ``#`` (external
    family codes look like ``#root#path``) is never re-scanned for tokens.
    """
    pattern = pattern or ""
    stem, dot, suffix = _split_suffix(pattern)

    def _sub(match: "re.Match[str]") -> str:
        key = resolve_token(match.group(1))
        if key is None:
            return ""
        value = values.get(key)
        return "" if value is None else str(value).strip()

    rendered = _TOKEN_RE.sub(_sub, stem)
    return _tidy(rendered) + dot + suffix


def _split_suffix(pattern: str) -> Tuple[str, str, str]:
    """Split *pattern* into ``(stem, ".", suffix)``; empty parts when no suffix.

    Only a short, token-free, alphanumeric tail counts as an extension, so a
    pattern ending in ``#Dátum#`` (which may render as ``2023.06.17``) is not
    mistaken for one.
    """
    idx = pattern.rfind(".")
    if idx <= 0:
        return pattern, "", ""
    suffix = pattern[idx + 1:]
    if 1 <= len(suffix) <= 5 and suffix.isalnum():
        return pattern[:idx], ".", suffix
    return pattern, "", ""


def safe_filename(name: str, fallback: str = "file", max_length: int = MAX_STEM_LENGTH) -> str:
    """Strip filesystem-hostile characters from *name*, truncating if long.

    Keeps dots and spaces (dates and multi-part names need them) and returns
    *fallback* when nothing usable remains.
    """
    cleaned = _UNSAFE_RE.sub("_", name or "").strip()
    # Windows also refuses a trailing dot or space on a path component.
    cleaned = cleaned.rstrip(". ")
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip(". ")
    return cleaned or fallback


def safe_pattern_filename(
    pattern: str,
    values: Mapping[str, object],
    fallback: str = "face",
) -> str:
    """:func:`render_pattern` + :func:`safe_filename`, keeping the extension.

    The length cap applies to the stem only, so the extension is never cut off.
    """
    rendered = render_pattern(pattern, values)
    stem, dot, suffix = _split_suffix(rendered)
    safe_stem = safe_filename(stem, fallback=fallback)
    if not dot:
        return safe_stem
    return f"{safe_stem}{dot}{safe_filename(suffix, fallback='jpg', max_length=8)}"
