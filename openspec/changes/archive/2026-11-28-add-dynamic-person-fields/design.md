# Design

## Context

`Person` (`app/db/models.py`) has fixed columns. The schema grows through `create_all` plus idempotent helpers in `app/db/database.py` (`_migrate_add_columns`, `_migrate_person_groups`, ...). The person editor `PersonInfoDialog` returns values through getters, and five call sites copy them into `PersonService.update_person(...)` or straight onto the ORM object: `edit_person_dialog`, `main_window/persons.py`, `persons_panel.py`, `family_tree_panel.py` and `image_browser/person_actions.py`. Merge with undo lives in `IdentityService.merge_persons` with `_PERSON_SNAPSHOT_COLUMNS`.

## Goals / Non-Goals

**Goals:** a normalised definition/value model, one save path, and a catalog API that Tree Studio can use without knowing which fields are columns.

**Non-Goals:** typed validation beyond the three types, enum fields, Persons table columns, and full-text search on values.

## Decisions

**D1: EAV with two tables.**
`person_field_definitions(id, label UNIQUE NOCASE, field_type, sort_order, created_at, updated_at)` and
`person_field_values(person_id FK CASCADE, field_id FK CASCADE, value TEXT NOT NULL, updated_at, PK(person_id, field_id))`.
Rejected alternatives: a JSON column on `persons`, which is hard to query, rename and count usages of; and adding a real column per field (`ALTER TABLE`), which would mean runtime DDL and no rename.

**D2: `date` is a flexible string,** like `birth_date` ("1930-as évek", "1954.03.12"), so it is stored as text. The type only affects the input widget and later rendering (for example year extraction). Long text uses a `QPlainTextEdit`.

**D3: catalog keys.** Built-in fields use `b:<column>` (`b:nickname`, `b:married_name`, `b:birth_date`, ...), from a fixed list in the service with i18n label keys. Custom fields use `c:<definition id>`. `PersonFieldService.catalog()` returns `[FieldSpec(key, label, type, builtin)]`, and `values_for(person_ids) -> {pid: {key: value}}` fetches values in bulk with one query per table. This is what Tree Studio uses to choose which lines appear on a node.

**D4: one save helper.** Add `PersonInfoDialog.custom_field_changes()`. It returns a mapping from an existing definition id, or a pending new field (label, type), to the new value, or to `None` for a cleared one. Add a module function `save_person_dialog(person_id, dlg, session)`. The helper writes the built-in fields through `PersonService.update_person` and the custom fields through `PersonFieldService.apply_changes`. All five call sites switch to it. New definitions are created inside the same session, so a cancel never leaves an orphan definition.

**D5: editor UI.** Below the built-in form, add a "Saját mezők" section: one row per filled field (label, input, clear button), then a combo box "+ Mező hozzáadása..." that lists the unused definitions plus "Új mező...". "Új mező..." opens a small inline label + type form. A case-insensitive label match swaps to the existing field. Value completers reuse the existing `_DialogDataThread` pattern (distinct values per field, loaded off the UI thread).

**D6: merge.** In `merge_persons`, after the faces move, use `INSERT OR IGNORE` to copy the source values into the target, then delete the source's values. The undo snapshot stores the source's `{field_id: value}` and recreates it on restore.

**D7: migration.** `create_all` creates the tables on new and old databases. A `_migrate_person_fields` helper adds the NOCASE unique index if it is missing. No data migration is needed.

## Risks / Trade-offs

- [The label is user data, so it cannot be translated] → this is accepted. Labels are shown as typed, and only built-in labels go through `t()`.
- [Changing a definition's type after values exist] → a type change is allowed only between `text` and `long_text`. Changing to or from `date` is blocked when values exist.
- [Editing in two places at once (Face-Local + Tree Studio)] → last write wins per value row. Change detection comes with `add-db-change-notification`.
