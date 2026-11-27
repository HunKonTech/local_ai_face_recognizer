# Proposal

## Why

The planned Tree Studio app will edit the same SQLite database as Face-Local, and both may be open at once. Today neither process notices the other's writes. Panels show stale persons, relationships and fields until something refreshes them by chance. A later save can then work from outdated data without the user knowing.

## What Changes

- Every write that one app process makes to the family-related tables is recorded in a shared change log. This covers persons, relationships, custom person fields and, once they exist, saved tree views. Each entry names which table and row changed, and which process wrote it.
- When an app window becomes active (focus change), the app checks for changes written by **other** processes since its last check.
- If there are any, a non-modal bar tells the user what changed (for example "3 persons and 1 relationship changed in another window") and offers a **Refresh** button. The app's own writes never trigger the bar.
- A new setting, **"Refresh automatically when the window is activated"**, refreshes the affected views without asking. It is off by default.
- Old log entries are pruned automatically, so the log does not grow without bound.
- The mechanism is a reusable service and bar widget. Face-Local uses it now, and Tree Studio (`add-tree-studio-app`) will use the same parts.

Out of scope: live push while the window stays focused, merging conflicting concurrent edits, and detecting writes from third-party tools (for example an SQLite browser).

## Capabilities

### New Capabilities
- `external-change-notification`: recording family-data writes per process, detecting other processes' changes when the window is activated, notifying the user and refreshing manually or automatically.

### Modified Capabilities
<!-- none -->

## Impact

- DB: new `change_log` table (created by `create_all`/migration); per-connection temporary triggers installed by an engine `connect` listener in `app/db/database.py`.
- New `app/services/change_watch_service.py`, which registers the watched tables, reads changes since a cursor and prunes the log.
- New `app/ui/widgets/external_change_bar.py`.
- `app/ui/main_window/`: activation hook and refresh routing (persons sidebar, Persons tab, family tree panel). The new setting goes in `dialogs/settings_dialog.py`, stored through `app_settings`.
- i18n strings (EN + HU).
- Tests: new `tests/test_change_watch_service.py` (two engines on one DB file simulate two processes).
