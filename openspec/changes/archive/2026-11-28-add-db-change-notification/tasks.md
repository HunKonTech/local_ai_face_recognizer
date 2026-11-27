# Tasks

## 1. Change log and triggers

- [x] 1.1 Add the `change_log` model and migration and create the per-process origin token in `app/services/change_watch_service.py`; verify in `tests/test_database.py` that the table exists after opening an old DB.
- [x] 1.2 Install TEMP triggers for the registered tables from an engine `connect` listener (design D1, D2) and expose `register_watched_table`; verify with `tests/test_change_watch_service.py`: a person rename, a relationship insert and a field value change each log one entry with this process's origin, and a face reassignment logs none.
- [x] 1.3 Implement `changes_since_cursor()` and startup pruning (design D3, D5); verify with a test that uses two engines on one temp DB file (two origins): engine A sees B's changes grouped by kind and never its own, the cursor advances, and entries older than 30 days are pruned.

## 2. Face-Local UI

- [x] 2.1 Add `app/ui/widgets/external_change_bar.py` (summary text, Refresh, dismiss) with EN + HU strings, including plural-safe kind/count wording; verify with an offscreen widget test that the summary text for {person: 3, relationship: 1} is correct and that Refresh emits its signal.
- [x] 2.2 Add the `main_window/external_changes.py` mixin: activation hook, auto-refresh setting check and `_refresh_after_external_change(kinds)` routing (design D4); verify by running two app processes on the same DB: renaming a person in one shows the bar in the other on activation, Refresh updates the sidebar and the family tree, and the app's own edits never show the bar.
- [x] 2.3 Add the "Refresh automatically when the window is activated" checkbox to the settings dialog (default off, stored through `app_qsettings()`, retranslated); verify that toggling it persists across restarts and that, when it is on, activation refreshes without the bar.

## 3. Integration

- [x] 3.1 Run `ruff check app/` and the database, person and family test files; time 1000 person updates with triggers on and off and verify the overhead stays below 10%.
