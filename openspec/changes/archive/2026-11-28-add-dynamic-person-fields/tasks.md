# Tasks

## 1. Data model and service

- [x] 1.1 Add `PersonFieldDefinition` and `PersonFieldValue` models (design D1) with a `Person.field_values` cascade relationship, plus `_migrate_person_fields` in `app/db/database.py`; verify with `tests/test_database.py`: opening a pre-change DB file creates both tables and keeps existing person columns.
- [x] 1.2 Create `app/services/person_field_service.py` with definition CRUD (case-insensitive unique label, rename, reorder, delete with a usage count, type-change rule), `apply_changes(person_id, changes)`, `distinct_values(field_id)`, `catalog()` and `values_for(person_ids)` (design D3); verify with new `tests/test_person_field_service.py` covering every spec scenario at service level.

## 2. Person editor and single save path

- [x] 2.1 Add the custom-field section to `PersonInfoDialog` (design D5: filled rows, add picker, new-field inline form, duplicate-label swap, background value completers) and `custom_field_changes()`; verify with an offscreen dialog test that adding, clearing and cancelling produce the expected change set.
- [x] 2.2 Add `save_person_dialog(...)` (design D4) and switch `edit_person_dialog`, `main_window/persons.py`, `panels/persons_panel.py`, `panels/family_tree_panel.py` and `panels/image_browser/person_actions.py` to it; verify `grep -n "dlg.nickname()" app/ui` only hits the helper, and `tests/test_persons_panel_refresh.py` plus `tests/test_family_tree_panel.py` pass.
- [x] 2.3 Add `app/ui/dialogs/person_fields_dialog.py` (list, rename, drag to reorder, delete with a count confirmation), opened from the editor and the Persons menu, with retranslation in `_retranslate()`; verify by running the app: rename keeps values and delete removes them.
- [x] 2.4 Add EN + HU strings for every new label, button and confirmation to the closest `i18n/strings/sNN_*.py`; verify with `grep` that every new key has both `en` and `hu`.

## 3. Merge, delete and export

- [x] 3.1 Merge values in `IdentityService.merge_persons` and extend the undo snapshot (design D6); verify with new merge tests for gap filling, target precedence and undo restoring the source values.
- [x] 3.2 Make sure deleting a person removes its values (FK cascade + ORM cascade); verify with a test in `tests/test_person_full_delete.py`.
- [x] 3.3 Add `customFields: [{label, value}]` to the person record in `app/services/export_service.py`; verify with an export test that asserts the field appears.

## 4. Integration

- [x] 4.1 Run `ruff check app/` and the person, family and export test files; then, in the running app, add "Foglalkozás" to one person, pick it for a second one from the list, and verify the value suggestions and the save from the family tree panel.
