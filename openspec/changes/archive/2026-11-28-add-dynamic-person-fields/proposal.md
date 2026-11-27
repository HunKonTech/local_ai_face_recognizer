# Proposal

## Why

Person data in Face-Local is a fixed set of columns (name parts, nickname, birth/death, notes, ...). Users want to record extra facts, such as occupation, that differ from family to family. Today the only place for them is the free-text notes, which the family tree cannot display field by field. The planned Tree Studio app also needs to show and hide individual person facts on the tree. That requires a field model it can enumerate.

## What Changes

- Users can add **custom person fields** from the person editor. A field is created once (for example "Foglalkozás"). After that, any other person can use it from a list of existing fields, without defining it again.
- A field definition has a user-given label, a type (`text`, `long_text`, `date`) and a display order. Values are stored per person.
- The person editor shows the custom fields that have a value for that person, a "+ Mező" picker listing the existing definitions, and a "New field..." entry. A field can be cleared for one person without touching the definition.
- A field-management dialog renames, reorders and deletes definitions. Deleting a definition asks for confirmation and states how many persons have a value for it.
- A **person field catalog** service lists the built-in fields (nickname, married name, birth/death date and place, family code, ...) and the custom fields under stable keys, together with each person's values. The family tree and Tree Studio consume this catalog.
- Merging persons carries the custom values over: the target's values win, the source fills the gaps. Merge undo restores the source's values.
- The HTML/JSON person export includes the custom fields.
- Every caller of the person editor saves custom fields through the same path.

Out of scope: choice/enum field types, per-field validation rules, showing custom fields as columns in the Persons table, searching by custom field, and showing fields on the family tree (that belongs to `add-tree-studio-app`).

## Capabilities

### New Capabilities
- `person-custom-fields`: user-defined person fields: defining, reusing, editing, deleting, merging and exporting them, and the field catalog that lists built-in and custom fields under stable keys.

### Modified Capabilities
<!-- none: no existing spec covers person data -->

## Impact

- DB: two new tables (field definitions, field values), created by `create_all` plus an idempotent migration in `app/db/database.py`. No change to existing columns.
- `app/db/models.py`: new models; a `Person` relationship with cascade delete.
- New `app/services/person_field_service.py` (definitions, values, catalog).
- `app/ui/dialogs/person_info_dialog.py` and `edit_person_dialog()`: custom-field section. The other save sites (`main_window/persons.py`, `panels/persons_panel.py`, `panels/family_tree_panel.py`, `panels/image_browser/person_actions.py`) are routed through one shared save helper.
- New dialog `app/ui/dialogs/person_fields_dialog.py`, opened from the person editor and the Persons menu.
- `app/services/identity_service.py`: merge and undo snapshot.
- `app/services/export_service.py`: `customFields` in the person JSON.
- i18n strings (EN + HU).
- Tests: new `tests/test_person_field_service.py`; extend the merge and export tests.
