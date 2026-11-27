"""Custom (user-defined) person fields and the person field catalog.

Definitions live in ``person_field_definitions`` and per-person values in
``person_field_values`` (EAV, see ``openspec`` change
``add-dynamic-person-fields``). The *catalog* lists built-in ``Person``
columns (keys ``b:<column>``) and custom fields (keys ``c:<definition id>``)
uniformly, so views like the family tree / Tree Studio can enumerate person
facts without knowing which ones are real columns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Optional, Union

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.db.models import (
    PERSON_FIELD_TYPES,
    Person,
    PersonFieldDefinition,
    PersonFieldValue,
)

log = logging.getLogger(__name__)

BUILTIN_PREFIX = "b:"
CUSTOM_PREFIX = "c:"

# (column, i18n label key, type). Order = default catalog order.
BUILTIN_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("name_prefix", "name_prefix", "text"),
    ("last_name", "last_name", "text"),
    ("first_name", "first_name", "text"),
    ("second_name", "second_name", "text"),
    ("nickname", "nickname", "text"),
    ("married_name", "married_name", "text"),
    ("birth_date", "birth_date", "date"),
    ("birth_place", "birth_place", "text"),
    ("death_date", "death_date", "date"),
    ("death_place", "death_place", "text"),
    ("family_code", "family_code", "text"),
    ("external_family_code", "external_family_code", "text"),
    ("notes", "notes", "long_text"),
)


class PersonFieldError(ValueError):
    """Invalid definition operation (empty/duplicate label, bad type change)."""


@dataclass(frozen=True)
class NewField:
    """A field created in the editor but not stored yet (design D4)."""

    label: str
    field_type: str = "text"


@dataclass(frozen=True)
class FieldSpec:
    """One catalog entry. ``label`` is translated for built-ins."""

    key: str
    label: str
    field_type: str
    builtin: bool


FieldRef = Union[int, NewField]


def normalize_label(label: str) -> str:
    """Comparison key: surrounding whitespace stripped, case folded."""
    return " ".join((label or "").split()).casefold()


def custom_key(field_id: int) -> str:
    return f"{CUSTOM_PREFIX}{field_id}"


def builtin_key(column: str) -> str:
    return f"{BUILTIN_PREFIX}{column}"


class PersonFieldService:
    """Definition CRUD, per-person values, catalog. Caller owns the session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Definitions
    # ------------------------------------------------------------------

    def list_definitions(self) -> list[PersonFieldDefinition]:
        return list(
            self._session.execute(
                select(PersonFieldDefinition).order_by(
                    PersonFieldDefinition.sort_order, PersonFieldDefinition.id
                )
            ).scalars()
        )

    def find_by_label(self, label: str) -> Optional[PersonFieldDefinition]:
        key = normalize_label(label)
        if not key:
            return None
        for d in self.list_definitions():
            if normalize_label(d.label) == key:
                return d
        return None

    def create_definition(self, label: str, field_type: str = "text") -> PersonFieldDefinition:
        clean = " ".join((label or "").split())
        if not clean:
            raise PersonFieldError("empty label")
        if field_type not in PERSON_FIELD_TYPES:
            raise PersonFieldError(f"unknown field type {field_type!r}")
        if self.find_by_label(clean) is not None:
            raise PersonFieldError(f"duplicate label {clean!r}")
        max_order = self._session.execute(
            select(func.max(PersonFieldDefinition.sort_order))
        ).scalar()
        d = PersonFieldDefinition(
            label=clean, field_type=field_type, sort_order=(max_order or 0) + 1
        )
        self._session.add(d)
        self._session.flush()
        return d

    def get_or_create(self, label: str, field_type: str = "text") -> PersonFieldDefinition:
        """Existing definition with the same (normalised) label, else a new one."""
        existing = self.find_by_label(label)
        return existing if existing is not None else self.create_definition(label, field_type)

    def rename_definition(self, field_id: int, label: str) -> PersonFieldDefinition:
        d = self._require(field_id)
        clean = " ".join((label or "").split())
        if not clean:
            raise PersonFieldError("empty label")
        other = self.find_by_label(clean)
        if other is not None and other.id != d.id:
            raise PersonFieldError(f"duplicate label {clean!r}")
        d.label = clean
        self._session.flush()
        return d

    def can_change_type(self, field_id: int, new_type: str) -> bool:
        """Text <-> long text always; to/from ``date`` only while unused."""
        d = self._require(field_id)
        if new_type not in PERSON_FIELD_TYPES:
            return False
        if new_type == d.field_type:
            return True
        if {d.field_type, new_type} <= {"text", "long_text"}:
            return True
        return self.usage_count(field_id) == 0

    def change_type(self, field_id: int, new_type: str) -> PersonFieldDefinition:
        if not self.can_change_type(field_id, new_type):
            raise PersonFieldError("type change not allowed while values exist")
        d = self._require(field_id)
        d.field_type = new_type
        self._session.flush()
        return d

    def reorder(self, ordered_ids: Iterable[int]) -> None:
        for order, fid in enumerate(ordered_ids, start=1):
            d = self._session.get(PersonFieldDefinition, fid)
            if d is not None:
                d.sort_order = order
        self._session.flush()

    def usage_count(self, field_id: int) -> int:
        return int(
            self._session.execute(
                select(func.count()).select_from(PersonFieldValue).where(
                    PersonFieldValue.field_id == field_id
                )
            ).scalar()
            or 0
        )

    def usage_counts(self) -> dict[int, int]:
        rows = self._session.execute(
            select(PersonFieldValue.field_id, func.count()).group_by(
                PersonFieldValue.field_id
            )
        ).all()
        return {fid: int(n) for fid, n in rows}

    def delete_definition(self, field_id: int) -> int:
        """Delete a definition and all its values; returns the removed value count."""
        d = self._require(field_id)
        count = self.usage_count(field_id)
        self._session.execute(
            delete(PersonFieldValue).where(PersonFieldValue.field_id == field_id)
        )
        self._session.delete(d)
        self._session.flush()
        return count

    def _require(self, field_id: int) -> PersonFieldDefinition:
        d = self._session.get(PersonFieldDefinition, field_id)
        if d is None:
            raise PersonFieldError(f"no field definition {field_id}")
        return d

    # ------------------------------------------------------------------
    # Values
    # ------------------------------------------------------------------

    def values_of(self, person_id: int) -> dict[int, str]:
        """``{field_id: value}`` for one person."""
        rows = self._session.execute(
            select(PersonFieldValue.field_id, PersonFieldValue.value).where(
                PersonFieldValue.person_id == person_id
            )
        ).all()
        return {fid: v for fid, v in rows}

    def set_value(self, person_id: int, field_id: int, value: Optional[str]) -> None:
        clean = (value or "").strip()
        row = self._session.get(PersonFieldValue, (person_id, field_id))
        if not clean:
            if row is not None:
                self._session.delete(row)
        elif row is None:
            self._session.add(
                PersonFieldValue(person_id=person_id, field_id=field_id, value=clean)
            )
        elif row.value != clean:
            row.value = clean
            row.updated_at = datetime.utcnow()
        self._session.flush()

    def apply_changes(
        self, person_id: int, changes: Mapping[FieldRef, Optional[str]]
    ) -> None:
        """Apply editor changes: ``{field id | NewField: value | None}``.

        A :class:`NewField` whose label matches an existing definition reuses
        it; an empty value for a new field creates nothing.
        """
        for ref, value in changes.items():
            if isinstance(ref, NewField):
                if not (value or "").strip():
                    continue
                field_id = self.get_or_create(ref.label, ref.field_type).id
            else:
                field_id = int(ref)
                if self._session.get(PersonFieldDefinition, field_id) is None:
                    continue
            self.set_value(person_id, field_id, value)

    def distinct_values(self, field_id: int) -> list[str]:
        rows = self._session.execute(
            select(PersonFieldValue.value)
            .where(PersonFieldValue.field_id == field_id)
            .distinct()
        ).scalars()
        return sorted(set(rows), key=str.casefold)

    def merge_values(self, source_id: int, target_id: int) -> dict[int, str]:
        """Copy source values into gaps of the target, then drop the source's.

        Returns the source's original ``{field_id: value}`` (for undo).
        """
        source_vals = self.values_of(source_id)
        target_fields = set(self.values_of(target_id))
        for fid, value in source_vals.items():
            if fid not in target_fields:
                self._session.add(
                    PersonFieldValue(person_id=target_id, field_id=fid, value=value)
                )
        self._session.execute(
            delete(PersonFieldValue).where(PersonFieldValue.person_id == source_id)
        )
        self._session.flush()
        return source_vals

    def restore_values(self, person_id: int, values: Mapping[int, str]) -> None:
        """Recreate a person's values (merge undo); unknown fields are skipped."""
        for fid, value in values.items():
            if self._session.get(PersonFieldDefinition, int(fid)) is None:
                continue
            self.set_value(person_id, int(fid), value)

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------

    def catalog(self, translate=None) -> list[FieldSpec]:
        """Built-in then custom fields with stable keys.

        ``translate`` maps an i18n key to a label (defaults to ``t``).
        """
        if translate is None:
            try:
                from app.ui.i18n import t as translate
            except Exception:  # noqa: BLE001 - headless use without UI
                translate = lambda k: k  # noqa: E731
        specs = [
            FieldSpec(builtin_key(col), translate(label_key), ftype, True)
            for col, label_key, ftype in BUILTIN_FIELDS
        ]
        specs.extend(
            FieldSpec(custom_key(d.id), d.label, d.field_type, False)
            for d in self.list_definitions()
        )
        return specs

    def values_for(self, person_ids: Iterable[int]) -> dict[int, dict[str, str]]:
        """``{pid: {catalog key: value}}`` for non-empty values (bulk)."""
        ids = list({int(p) for p in person_ids})
        out: dict[int, dict[str, str]] = {pid: {} for pid in ids}
        if not ids:
            return out
        cols = [getattr(Person, col) for col, _, _ in BUILTIN_FIELDS]
        for chunk_start in range(0, len(ids), 500):
            chunk = ids[chunk_start:chunk_start + 500]
            for row in self._session.execute(
                select(Person.id, *cols).where(Person.id.in_(chunk))
            ).all():
                pid = row[0]
                for (col, _, _), value in zip(BUILTIN_FIELDS, row[1:]):
                    if value not in (None, ""):
                        out[pid][builtin_key(col)] = str(value)
            for pid, fid, value in self._session.execute(
                select(
                    PersonFieldValue.person_id,
                    PersonFieldValue.field_id,
                    PersonFieldValue.value,
                ).where(PersonFieldValue.person_id.in_(chunk))
            ).all():
                out[pid][custom_key(fid)] = value
        return out

    def labelled_values(self, person_id: int) -> list[tuple[str, str]]:
        """``[(label, value)]`` of a person's custom fields in definition order."""
        vals = self.values_of(person_id)
        return [(d.label, vals[d.id]) for d in self.list_definitions() if d.id in vals]
