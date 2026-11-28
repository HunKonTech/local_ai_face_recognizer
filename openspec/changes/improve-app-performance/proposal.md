# Proposal

## Why

Face-Local has become very slow. Measurements on the real family database (1 GB, 1,556 persons, 22,899 faces, 13,341 images; Windows 10, 2026-11-28) show where the time goes:

| What | Measured | Main cause |
|---|---|---|
| App start-up (window usable) | **32.1 s** | Objects panel decodes 77 full-size originals on the UI thread at construction (14.0 s); Persons table loads 1,556 thumbnails at start-up although the tab is hidden; `sklearn` imported eagerly (3.8 s) |
| Deleting persons (merge cleanup, "delete empty Unknowns") | **35.6 s for 400 persons**, 93 s seen in the log | `auto_assignments.previous_person_id` / `corrected_person_id` have `ON DELETE SET NULL` foreign keys **without an index**: every deleted person scans the 7,184-row table twice. With the indexes the same delete takes **50 ms** (~700×) |
| Opening the Persons tab | **6.2 s** | Every finished thumbnail scans all table rows to find its row (872,126 `item()`/`data()` calls, O(n²)); 1,551 synchronous `Path.exists()` on the archive drive; table built twice (start-up + first show) |
| `ui.refresh_persons` (after almost every edit) | up to **3.6 s** in the logs, 0.4 s on a warm cache | Runs on the UI thread; 1,555 `stat()` calls per refresh; synchronous family-search refresh |
| Slow SQL warnings in the logs | 485 statements > 50 ms; COMMITs ~1 s | Bulk `face_blobs`/`images` loads of full ORM rows; WAL commits on the archive drive with `synchronous=NORMAL` checkpoints during long writes |

The app must feel instant again for everyday work (start, switch tabs, edit/merge persons) on this database size, and keep scaling to 5–10× more data.

## What Changes

- **Database**: add the missing foreign-key indexes (and a regression check that every FK column has a leading index); keep bulk deletes and merges in one transaction; read only needed columns for bulk embedding/image loads.
- **Start-up**: build heavy panels lazily (data loads on first show, not in `__init__`); import heavy optional libraries (`sklearn`, deep models, onnxruntime) on first use; keep the window responsive while background work starts.
- **Thumbnails**: one shared, persistent, size-keyed thumbnail cache on disk (keyed by path + mtime + size) used by the sidebar, Persons table, Objects panel, Tree Studio and image browser; decode with `QImageReader.setScaledSize` instead of decoding full originals; no synchronous file-system checks on the UI thread.
- **Persons tab / sidebar**: `person_id → row` map instead of row scans; one build per data change; incremental updates after a single edit instead of full rebuilds; `refresh_persons` data loading off the UI thread with a coalescing timer.
- **Measurement**: a reproducible benchmark script and budgets (see spec) so regressions are caught; the existing `app.perf` report gains start-up phase timings.

Out of scope: changing the recognition models or the clustering algorithm, GPU/TPU work, and the web export.

## Capabilities

### New Capabilities
- `app-responsiveness`: response-time budgets for start-up, tab switching, person edits/merges/deletes and thumbnail display on a reference database, plus the requirement that the UI thread never blocks on bulk image decoding or full-table rebuilds.

### Modified Capabilities
<!-- none -->

## Impact

- `app/db/models.py`, `app/db/database.py`: new indexes + idempotent migration; FK-index regression test.
- `app/ui/panels/objects_panel.py`, `app/ui/widgets/object_gallery_widget.py`, `app/ui/panels/persons_panel.py`, `app/ui/panels/sidebar_panel.py`, `app/ui/main_window/persons.py`, `app/ui/main_window/ui_build.py`: lazy loading, row maps, async refresh.
- New `app/services/thumbnail_cache.py` (or extension of `app/workers/thumbnail_worker.py`), used by all thumbnail consumers.
- `app/ui/main_window/face_actions.py`, `app/services/clustering_service.py`, `app/clustering/clusterer.py`: lazy heavy imports.
- New `scripts/bench_app.py` (profiles start-up, tab switches, refresh, delete on a copy of a DB).
- No change to data, config format or user-visible behaviour except speed.
