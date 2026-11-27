"""Cross-process change detection for family data (``add-db-change-notification``).

Every app process installs *TEMP* triggers on each new SQLite connection
(see :func:`install_change_triggers`). They append one ``change_log`` row per
insert / update / delete on the watched tables, tagged with this process's
random origin token baked into the trigger SQL. Other processes (Tree Studio,
a second Face-Local window) read entries with ``origin != me`` when their
window is activated.

Writes from tools outside the app (an SQLite browser) carry no TEMP triggers
and are deliberately not logged; persistent triggers would break those tools.
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)

PRUNE_AFTER_SECONDS = 30 * 24 * 3600

# Random token identifying this OS process in change_log.origin.
PROCESS_ORIGIN = secrets.token_hex(8)


@dataclass(frozen=True)
class WatchedTable:
    table: str
    kind: str
    row_expr: str = "id"  # column of NEW/OLD that identifies the changed row


_WATCHED: dict[str, WatchedTable] = {}


def register_watched_table(table: str, kind: str, row_expr: str = "id") -> None:
    """Watch *table*; entries are summarised under *kind*.

    Register before the engine opens connections (connections opened earlier
    keep their old trigger set until the pool is recycled).
    """
    _WATCHED[table] = WatchedTable(table, kind, row_expr)


def watched_tables() -> dict[str, WatchedTable]:
    return dict(_WATCHED)


register_watched_table("persons", "person")
register_watched_table("relationships", "relationship")
register_watched_table("person_field_definitions", "field")
register_watched_table("person_field_values", "person", "person_id")


def _now_sql() -> str:
    return "((julianday('now') - 2440587.5) * 86400.0)"


def install_change_triggers(dbapi_conn, origin: str = PROCESS_ORIGIN) -> int:
    """Create this connection's TEMP triggers; returns how many were created.

    Tables that do not exist yet (fresh DB before ``create_all``) are skipped;
    :func:`app.db.database.init_db` recycles the pool after migrations so later
    connections get the full set.
    """
    if not all(c.isalnum() for c in origin):
        raise ValueError("origin token must be alphanumeric")
    cur = dbapi_conn.cursor()
    try:
        existing = {
            row[0]
            for row in cur.execute(
                "SELECT name FROM main.sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "change_log" not in existing:
            return 0
        created = 0
        for w in _WATCHED.values():
            if w.table not in existing:
                continue
            for op, ref in (("INSERT", "NEW"), ("UPDATE", "NEW"), ("DELETE", "OLD")):
                name = f"cl_{w.table}_{op.lower()}"
                cur.execute(f"DROP TRIGGER IF EXISTS temp.{name}")
                cur.execute(
                    f"CREATE TEMP TRIGGER {name} AFTER {op} ON main.{w.table} "
                    f"BEGIN INSERT INTO change_log "
                    f"(table_name, row_id, op, origin, changed_at) VALUES "
                    f"('{w.table}', {ref}.{w.row_expr}, '{op[0]}', '{origin}', {_now_sql()}); "
                    f"END"
                )
                created += 1
        return created
    finally:
        cur.close()


@dataclass
class ExternalChanges:
    """Distinct changed row ids per kind, written by other processes."""

    by_kind: dict[str, set[int]] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.by_kind.items() if v}

    def __bool__(self) -> bool:
        return any(self.by_kind.values())


class ChangeWatchService:
    """Per-process cursor over ``change_log``."""

    def __init__(self, engine: Engine, origin: str = PROCESS_ORIGIN) -> None:
        self._engine = engine
        self._origin = origin
        self._cursor = 0
        self.prune()
        self.reset_cursor()

    @property
    def origin(self) -> str:
        return self._origin

    @property
    def cursor(self) -> int:
        return self._cursor

    def reset_cursor(self) -> None:
        with self._engine.connect() as conn:
            self._cursor = int(
                conn.execute(text("SELECT COALESCE(MAX(id), 0) FROM change_log")).scalar()
                or 0
            )

    def prune(self, now: Optional[float] = None) -> int:
        cutoff = (time.time() if now is None else now) - PRUNE_AFTER_SECONDS
        try:
            with self._engine.begin() as conn:
                res = conn.execute(
                    text("DELETE FROM change_log WHERE changed_at < :c"), {"c": cutoff}
                )
                return int(res.rowcount or 0)
        except Exception:  # noqa: BLE001 - never block startup on pruning
            log.exception("change_log prune failed")
            return 0

    def changes_since_cursor(self) -> ExternalChanges:
        """Other processes' changes since the last call; advances the cursor."""
        kinds = {w.table: w.kind for w in _WATCHED.values()}
        out = ExternalChanges()
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT id, table_name, row_id, origin FROM change_log "
                    "WHERE id > :cur ORDER BY id"
                ),
                {"cur": self._cursor},
            ).all()
        for row_id_pk, table, row_id, origin in rows:
            self._cursor = max(self._cursor, int(row_id_pk))
            if origin == self._origin:
                continue
            kind = kinds.get(table, table)
            out.by_kind.setdefault(kind, set())
            if row_id is not None:
                out.by_kind[kind].add(int(row_id))
        return out


_service: Optional[ChangeWatchService] = None


def get_change_watch_service() -> Optional[ChangeWatchService]:
    """The service for the current DB (``None`` before ``init_db``)."""
    return _service


def init_change_watch(engine: Engine) -> Optional[ChangeWatchService]:
    global _service
    try:
        _service = ChangeWatchService(engine)
    except Exception:  # noqa: BLE001 - optional feature, never block startup
        log.exception("Change watching disabled")
        _service = None
    return _service
