# Design

## Context

The DB is SQLite in WAL mode with `busy_timeout=5000` (`app/db/database.py`), so two processes can read and write safely. There is no change signalling at all. Panels refresh only after their own actions (for example `MainWindow._refresh_persons`). Settings persist through `app/app_settings.py` (`app_qsettings()`).

## Goals / Non-Goals

**Goals:** cheap detection on window activation, the ability to tell our own writes from other processes' writes, a per-kind summary, and reuse by Tree Studio.

**Non-Goals:** row-level conflict resolution, real-time push, and tracking writes made by tools outside the app.

## Decisions

**D1: `change_log` table + per-connection TEMP triggers that carry an origin literal.**
`change_log(id INTEGER PRIMARY KEY AUTOINCREMENT, table_name TEXT, row_id INTEGER, op TEXT, origin TEXT, changed_at REAL)`.
An engine `connect` listener runs `CREATE TEMP TRIGGER ... AFTER INSERT/UPDATE/DELETE ON main.<table>` for each watched table. The body inserts into `main.change_log` with `origin = '<process token>'`. The token is a random value made once per process and baked into the SQL.
Alternatives considered:
- Persistent triggers that call a Python-registered SQL function `app_origin()`. Any connection without the function (an SQLite browser, older app versions) would then fail on every write to persons. **Rejected.**
- Persistent triggers without an origin, plus "remember my own revision after commit". This races when local and foreign commits interleave, and the foreign change gets swallowed. **Rejected.**
- `PRAGMA data_version`. It cannot say what changed, and the connection pool makes "own versus foreign" unreliable. **Rejected.**

The accepted trade-off: writes from outside tools are not logged (see the Non-Goals).

**D2: watched-table registry.** `ChangeWatchService.WATCHED = {"persons": "person", "relationships": "relationship", "person_field_definitions": "field", "person_field_values": "person"}`. For field values the row key is `person_id`. Later changes register more tables through `register_watched_table(table, kind, row_expr)` before the engine connects; `add-tree-studio-app` adds its views table this way. The `faces` table is deliberately not watched, so scans and bulk face assignments do not flood the log.

**D3: cursor per process.** At startup the service stores `max(id)` as its cursor. `changes_since_cursor()` returns entries with `id > cursor AND origin != me`, grouped by kind as distinct row ids, and then advances the cursor. The query runs on window activation (`QGuiApplication.applicationStateChanged` → `ApplicationActive`). It hits the primary key, so it is constant-time.

**D4: UI.** `ExternalChangeBar` is a slim bar placed above the central tab widget, with the summary text, Refresh and dismiss buttons. Refresh calls a `MainWindow._refresh_after_external_change(kinds)` routing method in a new `main_window/external_changes.py` mixin. For persons it calls `_refresh_persons` plus the persons panel and family tree panel `refresh()`; for relationships, the family tree. The auto-refresh setting key is `ui/auto_refresh_on_activate` (bool), shown in the settings dialog's general tab.

**D5: pruning.** At startup, run `DELETE FROM change_log WHERE changed_at < now - 30 days`. Running processes only read entries newer than their own startup cursor, so a long-running process could only miss an entry older than 30 days, which is acceptable.

## Risks / Trade-offs

- [TEMP triggers must exist on every pooled connection] → they are installed in the `connect` event, which fires for each new DBAPI connection. A test opens two sessions from the pool and checks that both log.
- [Bulk person operations (cleanup of empty unknown persons) produce many entries] → one row per affected row is cheap in SQLite, and pruning bounds growth. The summary counts distinct rows.
- [A merge deletes a person that a stale view still shows] → Refresh rebuilds the views from the DB. Actions on a missing person already fail gracefully (`session.get` → None).
