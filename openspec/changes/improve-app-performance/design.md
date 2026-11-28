# Design

## Context

Profiled with `cProfile` on an offscreen `MainWindow` over a copy of the real database (crop repair and update check disabled), plus a read-only audit of the original DB and the `app.perf` warnings in `data/face_local.log*` since September 2026. Numbers are in the proposal. Key code paths:

- `ObjectsPanel.__init__ → refresh()` calls `crop_pixmap()` → `load_pixmap_exif()` for every object: a full decode of the original photo (often 10–20 MP, on the D: archive drive) just to draw a 48 px icon.
- `PersonsPanel._start_thumbnail_loads()` checks `Path(path).exists()` per person on the UI thread and `_on_table_thumb_ready()` loops over all rows for each thumbnail.
- `ThumbnailRunnable._generate()` decodes the whole crop with PIL + OpenCV every time; nothing is cached across runs.
- Foreign keys `auto_assignments.previous_person_id`, `auto_assignments.corrected_person_id`, `recognition_merge_log.matched_person_id`, `family_tree_settings.default_root_person_id`, `images.place_id`, `object_match_suggestions.{created,source}_occurrence_id` have no leading index. SQLite must scan the child table for every parent row deleted.
- `app/ui/main_window/face_actions.py` imports `unknown_merge_service` → `clustering_service` → `sklearn` at module import time.

## Goals / Non-Goals

**Goals:** start-up under 6 s to an interactive window on the reference DB; Persons tab under 0.5 s; deleting 400 persons under 0.5 s; no UI-thread stall over 200 ms in normal use; same results and data.

**Non-Goals:** rewriting panels in Qt models/views from scratch, multi-process recognition, changing recognition quality.

## Decisions

**D1: FK index migration + guard test.** Add `Index` declarations for the seven columns above, plus an idempotent `CREATE INDEX IF NOT EXISTS` migration. A test iterates `PRAGMA foreign_key_list` for every table of a fresh DB and fails when an FK column has no index whose first column is that column. This is the highest value change (35.6 s → 50 ms) and the safest, so it ships first and alone.

**D2: lazy panels.** Panels that load data in `__init__` (Objects, Persons, Groups, Collage, Locations) switch to the existing `mark_stale()` / `showEvent` pattern already used by the Persons panel: construct empty, load on first show, reload on show only when stale. The start-up path only builds the tab shells.

**D3: shared thumbnail cache.** `ThumbnailCache.get(path, size, bbox=None) -> QImage | None` with an in-memory LRU (e.g. 2,000 entries) in front of an on-disk cache under the app data dir (`thumbs/<sha1(path|mtime|size|bbox)>.jpg`). Misses are produced on a worker using `QImageReader` with `setScaledSize`/`setClipRect` (EXIF orientation via `setAutoTransform(True)`), so a 20 MP JPEG is decoded at 1/8 scale by libjpeg. Consumers receive the image via a signal carrying the key; they keep `key → widget/row` maps (no scans). The existing `ThumbnailRunnable` becomes a thin wrapper. Alternative rejected: storing thumbnails as DB blobs (grows the already 1 GB DB, more write contention).

**D4: Persons table.** Keep `self._row_of: dict[int, int]` rebuilt with the table (and on sort via `QTableWidget` sort signals, or by storing the person id on the item and using `QTableWidget.indexFromItem` from a `person_id → QTableWidgetItem` dict, which survives sorting). Disable sorting and updates (`setUpdatesEnabled(False)`) during bulk fill. Existence checks move into the worker.

**D5: `refresh_persons` off the UI thread and coalesced.** Edits call `request_persons_refresh()`, which restarts a 150 ms single-shot timer; the timer runs the four aggregate queries on the task manager and applies the result on the UI thread. Single-person edits update only that person's sidebar entry (`SidebarPanel.update_person(summary)`). `_crop_mtime` stats move into the thumbnail worker.

**D6: lazy heavy imports.** Module-level imports of `sklearn`, `insightface`, `onnxruntime`, `torch` and the deep model packages move into the functions that use them. A test imports `app.ui.main_window` with those modules blocked (`sys.modules[name] = None`) to keep it that way.

**D7: bulk data access.** Embedding / feature loads select only the columns they need (`face_id, embedding`) in chunks with `yield_per`, never full ORM entities with relationships. Long write jobs commit in batches of ~500 rows and run `PRAGMA wal_checkpoint(PASSIVE)` when idle, so a single COMMIT does not stall for ~1 s on the archive drive.

**D8: benchmark + budgets.** `scripts/bench_app.py --db <copy>` reproduces the measurements (start-up phases, each tab's first show, `refresh_persons`, delete 400 persons in a rolled-back transaction) and prints a table compared with the budgets in the spec. `app.perf` adds `startup.*` phase timings to its report.

## Risks / Trade-offs

- [Disk thumbnail cache grows] → size cap with LRU eviction by atime (e.g. 500 MB) and a "clear cache" action in settings.
- [Stale thumbnails after a crop is regenerated] → the cache key includes the file mtime and size.
- [Lazy panels show an empty state briefly] → show a "Loading…" placeholder; data arrives in the same event-loop turn for small panels.
- [Async refresh races with edits] → results carry a generation counter; older results are dropped.
- [New indexes cost write time] → seven small integer indexes; negligible next to the 700× delete gain.
