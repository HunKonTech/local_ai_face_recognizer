# Face-Local — Claude Code notes

Offline PySide6 desktop app: detect → embed → cluster faces, group photos by person. Full guide: [AGENT.md](AGENT.md) (commit/git policy lives there too).

## Working rules (save tokens)
- **Don't read whole files.** Find with `grep -n "def name"` / Grep, then `Read` with `offset`/`limit`.
- Big UI classes are split into mixins — a method is in the module named for its topic (map below).
- Terse answers. Small, targeted edits. One task per session.

## Commands
```bash
python -m app.main                                   # run app
QT_QPA_PLATFORM=offscreen python -m pytest -q -o addopts="" tests/test_x.py   # one test file (drop -o addopts if pytest-timeout is installed)
ruff check app/ui/panels/image_browser               # lint (E,F,W,I; line length ignored)
```
`tests/test_gdrive_*` fail locally when `googleapiclient`/`cryptography` are not installed — not a regression.

## Code map
| Area | Where |
|---|---|
| Config | `app/config.py` (+ `config.yaml`) |
| DB (SQLAlchemy, always `session_scope()`) | `app/db/database.py`, `app/db/models.py` |
| Detect / embed / cluster | `app/detectors/`, `app/embeddings/`, `app/clustering/`, `app/deep/` |
| Business logic (68 flat modules) | `app/services/*_service.py` — name says the domain (`family_*`, `export_*`, `place_*`, `object_*`, `*_merge_*`, `*_duplicate_*`) |
| Background | `app/workers/` (QThread/runnables), `app/tasks/manager.py` (`get_task_manager().submit(...)`) |
| Google Drive | `app/gdrive/` |

## UI map (`app/ui/`)
- **i18n**: `t("key")`, API in `i18n/__init__.py`. Strings live in `i18n/strings/sNN_*.py` (`STRINGS` dict, `{"en":…, "hu":…}`); `grep -rn '"key"' app/ui/i18n/strings` to find one. Every string needs **EN and HU**. New strings: append to the topically closest part.
- **main_window/** — `MainWindow(…Mixins, QMainWindow)`; `window.py` = `__init__`, settings, `closeEvent`.
  `ui_build` menus/toolbar/tabs/retranslate · `folders` · `pipeline` scan + deep runs · `maintenance` unknown cleanup/duplicates/identity repair · `gdrive` · `recording` · `objects` · `face_actions` preview/cluster face actions · `persons` rename/merge/delete/refresh · `export` collage + packages + path repair · `deep_model` · `tray_updates` · `tasks`.
- **panels/image_browser/** — `ImageBrowserPanel(…Mixins, QWidget)`; `panel.py` = `__init__`, `_build_ui`, `refresh`, fullscreen. `helpers.py` (module functions), `widgets.py` (`_DrawableImageLabel`, inline editor) ·
  `tree` · `rerecognition` · `navigation` · `image_loading` · `interaction` (clicks, objects) · `face_editing` (bbox) · `canvas` (coords, zoom) · `inline_editor` · `metadata` (date/note/place/GPS) · `search` · `person_actions` · `deoldified`.
  `panels/image_browser_panel.py` is only a compat re-export.
- Other panels: `panels/` (cluster, persons, family_search, locations, objects, collage, groups, preview); dialogs in `dialogs/`; shared widgets in `widgets/`.

## Conventions
- Mixin modules share state via `self`; a new method goes into the module of its topic, not into `panel.py`/`window.py`. Keep modules < ~800 lines; add a new mixin rather than growing one.
- Graceful fallbacks (no TPU/model → CPU), never crash on missing optional deps.
- UI persistent text must be re-set in `_retranslate()` (`main_window/ui_build.py`).
