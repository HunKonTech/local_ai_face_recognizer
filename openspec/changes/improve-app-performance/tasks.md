# Tasks

## 1. Measure first

- [ ] 1.1 Add `scripts/bench_app.py` (design D8): copy the DB with the SQLite backup API, profile start-up phases, first show of every tab, `refresh_persons`, and a rolled-back delete of 400 persons; print a table with the spec budgets; verify it reproduces the baseline (≈32 s start-up, 6.2 s Persons tab, 35.6 s delete) on the reference DB.
- [ ] 1.2 Add `startup.*` phase timings (imports, `init_db`, `_build_ui` per panel, first paint) to `app.perf`; verify they appear in `report()` and in the log.

## 2. Database (ship first)

- [ ] 2.1 Add indexes on `auto_assignments.previous_person_id`, `auto_assignments.corrected_person_id`, `recognition_merge_log.matched_person_id`, `family_tree_settings.default_root_person_id`, `images.place_id`, `object_match_suggestions.created_occurrence_id`, `object_match_suggestions.source_occurrence_id` (models + idempotent migration); verify deleting 400 persons on a copy of the reference DB drops from ≈35 s to < 0.5 s.
- [ ] 2.2 Add `tests/test_fk_indexes.py`: every FK column in a fresh DB has a leading index; verify it fails when one index is removed.
- [ ] 2.3 Change bulk embedding / feature loads to column-only chunked queries and batch long write jobs (design D7); verify the slow-SQL warnings for `face_blobs` / `image_features` / `images` loads disappear in the benchmark log.

## 3. Start-up

- [ ] 3.1 Make Objects, Groups, Collage and Locations panels load on first show (design D2); verify start-up no longer calls `crop_pixmap` / `load_pixmap_exif`.
- [ ] 3.2 Stop the Persons panel from building its table and thumbnails while hidden at start-up; verify `_populate_row` runs once, on first show.
- [ ] 3.3 Move `sklearn` and other heavy optional imports into the functions that use them (design D6) and add an import-guard test; verify `import app.ui.main_window` does not import `sklearn`.

## 4. Thumbnails

- [ ] 4.1 Add the shared `ThumbnailCache` (memory LRU + disk cache keyed by path, mtime, size, crop; `QImageReader` scaled decode with auto-transform) (design D3); verify with tests: cache hit after first load, invalidation on mtime change, EXIF rotation kept, and a 20 MP JPEG thumbnail decodes at reduced scale.
- [ ] 4.2 Switch the sidebar, Persons table, Objects panel/gallery, Tree Studio person list and family tree photos to the cache; move per-file `exists()` / `stat()` calls into the workers; verify no file-system calls per person remain on the UI thread (profile).
- [ ] 4.3 Add a size cap with eviction and a "Clear thumbnail cache" button in settings (EN + HU); verify eviction keeps the cache under the cap.

## 5. Persons views

- [ ] 5.1 Replace row scans in `PersonsPanel` (`_on_table_thumb_ready`, `_select_person_row`) with `person_id → item` maps that survive sorting (design D4); verify the 872k `item()` calls are gone and the tab opens < 0.5 s.
- [ ] 5.2 Make `refresh_persons` coalesced and asynchronous, with single-person incremental updates and a generation counter (design D5); verify a rename updates sidebar and table within 0.2 s of UI-thread time and rapid edits trigger one reload.

## 6. Verify

- [ ] 6.1 Run the benchmark on the reference DB and record before/after in the change; verify every budget in the `app-responsiveness` spec is met, and run `ruff check app/` and the full test suite.
