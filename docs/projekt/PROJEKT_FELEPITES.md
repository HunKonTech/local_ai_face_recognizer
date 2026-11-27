# Face-Local — projekt felépítése

> Állapot: 2026-11-27 · `main` @ `484ba6d` (v1.0.72) · 515 commit (2026-04-16 → 2026-10-13)

Offline, adatvédelmet előtérbe helyező PySide6 asztali alkalmazás: fényképeken arcokat **detektál → beágyaz (embedding) → klaszterez**, majd a képeket személyek szerint csoportosítja. Kiegészítők: családfa, helyszínek, tárgyak, kollázs, képernyőfelvétel, Google Drive online mód, web / Docker / szerver export.

## Technológiai stack

| Réteg | Technológia |
|---|---|
| Nyelv | Python ≥ 3.11 |
| GUI | PySide6 (6.6–6.10), mixin-alapú nagy osztályok |
| Adatbázis | SQLite + SQLAlchemy 2.x (31 tábla) |
| Arcdetektálás | YuNet (ONNX), OpenCV SSD res10, Coral Edge TPU, InsightFace SCRFD (opcionális) |
| Embedding | MobileFaceNet TFLite (`ai-edge-litert`), SFace, InsightFace, CoreML (macOS) |
| Klaszterezés | scikit-learn DBSCAN + kényszerek |
| Mély felismerés | saját MLP-ensemble (`app/deep/`) az embeddingek fölött |
| Felhő | Google Drive API (opcionális `gdrive` extra) |
| Web | Astro statikus oldal (`web/astro`) |
| Build / release | GitHub Actions (macOS DMG + notarizáció, Windows Inno Setup, Linux .deb), Sparkle auto-update |

## Könyvtárszerkezet

```
local_ai_face_recognizer/
├── app/                 Alkalmazás forráskód (≈ 105 ezer sor Python)
├── tests/               ~155 pytest fájl (≈ 34,5 ezer sor)
├── scripts/             build, csomagolás, release, benchmark, appcast
├── models/              ONNX / TFLite / Caffe modellfájlok
├── web/astro/           projekt weboldal (Astro)
├── tools/face_recognition/  külső segédeszköz
├── docs/                tervezési és teljesítmény dokumentumok
├── openspec/            OpenSpec változáskezelés (specs, changes/archive)
├── sparkle/, assets/    frissítő, ikonok
├── memory/              agent memória jegyzetek
├── .github/             CI workflow-k, Copilot agent/prompt/skill fájlok
├── config.yaml          helyi konfiguráció (config.example.yaml a minta)
├── AGENT.md, CLAUDE.md  fejlesztői / agent útmutató
└── pyproject.toml, requirements*.txt
```

## Az `app/` csomag

| Modul | Fájl | Sor | Szerep |
|---|---:|---:|---|
| `app/*.py` (gyökér) | 10 | 3 129 | `main.py` belépési pont, `config.py` (AppConfig + YAML), `paths`, `accel`, `perf`, diagnosztika, logging |
| `app/ui/` | 129 | 48 289 | teljes Qt felület (részletesen lent) |
| `app/services/` | 73 | 39 145 | üzleti logika, domainenként egy-egy `*_service.py` |
| `app/db/` | 4 | 3 100 | `models.py` (ORM), `database.py` (`session_scope()`), `query_utils.py` |
| `app/gdrive/` | 15 | 2 948 | OAuth, Drive kliens, cache, DB szinkron, online projekt session |
| `app/deep/` | 6 | 2 582 | MLP-ensemble osztályozó, dataset, értékelés, nem-arc szűrő |
| `app/workers/` | 13 | 2 363 | QThread workerek (pipeline, deep pipeline, thumbnail, geokódolás, exportok) |
| `app/utils/` | 6 | 1 696 | EXIF, képmetaadat, fuzzy dátum, személykeresés |
| `app/detectors/` | 10 | 1 651 | detektorok + `factory.py` (Coral próba, fallback lánc) |
| `app/embeddings/` | 8 | 1 067 | embedderek, arcigazítás, TTA |
| `app/tasks/` | 3 | 915 | háttér-feladatkezelő (`get_task_manager().submit`), erőforrás-szabályozó |
| `app/jobs/` | 3 | 167 | job típusok, megszakítás |
| `app/clustering/` | 2 | 133 | DBSCAN klaszterező |
| `app/updater/` | 2 | 117 | frissítés-ellenőrzés |

### Szolgáltatások (`app/services/`) domain szerint

| Domain | Szolgáltatások |
|---|---|
| Pipeline | `scan`, `detection`, `embedding`, `clustering`, `face_grouping`, `ai_face_detection`, `multi_stage_face_validator`, `face_quality`, `face_verification`, `non_face` |
| Mély felismerés | `deep_recognition`, `rerecognition`, `retroactive_verification`, `suggestion`, `match_scoring`, `vector_scoring` |
| Identitás / tisztítás | `identity`, `identity_repair`, `person`, `unknown_merge`, `unknown_person_reset`, `duplicate_unknown_face_finder`, `intra_image_duplicate`, `intra_image_consistency`, `overlap_resolution`, `face_geometry_cleanup`, `ignored_face` |
| Család | `family`, `family_tree`, `family_tree_layout`, `family_code_interpreter`, `family_code_schemes`, `person_group` |
| Helyek | `place`, `geocoding` |
| Tárgyak | `object`, `object_feature`, `object_matching` |
| Képek | `image_library`, `image_browser`, `image_path_matcher`, `deoldified_pairing`, `face_date`, `face_crop`, `filename_pattern` |
| Export / csomag | `export` (3,6 ezer sor), `local_server_export`, `docker_export`, `project_package`, `database_package`, `face_image_export`, `face_metadata_export`, `image_metadata_export` |
| Kollázs | `collage`, `collage_parser` |
| Felvétel | `screen_recorder` (2,3 ezer sor), `recording_metadata`, `recording_timeline_log`, `subtitle` |
| Egyéb | `update`, `shortcut`, `dependency_installer`, `server_settings_store`, `gmail_check`, `windows_display_probe`, `face_diagnostics`, `detection_run_logger` |

### Felhasználói felület (`app/ui/`)

| Almappa | Fájl | Sor | Tartalom |
|---|---:|---:|---|
| `dialogs/` | 45 | 18 545 | beállítások, export, összevonás, javaslatok, identitás-javítás, task manager, TPU státusz… |
| `panels/` | 29 | 14 058 | cluster, persons, preview, family_tree, family_search, locations, objects, collage, groups, log, sidebar + `image_browser/` mixinek |
| `main_window/` | 16 | 5 229 | `MainWindow` = mixinek (`ui_build`, `folders`, `pipeline`, `maintenance`, `gdrive`, `recording`, `objects`, `face_actions`, `persons`, `export`, `deep_model`, `tray_updates`, `tasks`) |
| `widgets/` | 18 | 4 807 | családfa nézet, térkép, idővonal, keresők, felvétel vezérlők |
| `i18n/` | 11 | 4 553 | `t("key")`, kétnyelvű (EN/HU) szótárak `strings/sNN_*.py` részekben |
| `helpers/` | 5 | 364 | közös segédfüggvények |

## Adatmodell (`app/db/models.py`)

Fő entitások: **Image**, **Face** (+ `FaceBlob`), **Person**, `PersonGroup` / `PersonGroupMembership`, `Relationship`, `FamilyTreeSettings`.
Felismerés-támogatás: `FaceCorrection`, `MergeSuggestion`, `MergeDecision`, `FaceVeto`, `TrainingRun`, `AutoAssignment`, `RecognitionMergeLog`, `AiFaceDetection`, `IgnoredFace`.
Helyek: `Place`, `PlaceAlias`, `GeocodingCache`, `PlaceAddressSuggestion`.
Tárgyak: `TaggedObject`, `ObjectOccurrence`, `ObjectPersonLink`, `ObjectAlias`, `ImageFeatures`, `ObjectPatchFeatures`, `ObjectMatchSuggestion`.
Egyéb: `Collage`, `CollageNode`, `RemoteImage` (Drive).

## Feldolgozási folyamat

```
Mappa(k) ─► ScanService (fájlkeresés, SHA-256)
        ─► DetectionService (YuNet / SSD / Coral / SCRFD, fallback lánc)
        ─► többlépcsős validátor (nem-arc, fél arc, beágyazott doboz szűrés)
        ─► igazítás + crop ─► EmbeddingService (MobileFaceNet / SFace / InsightFace)
        ─► ClusteringService (DBSCAN) ─► Person hozzárendelés
        ─► Deep pipeline: MLP-ensemble tanítás a címkézett arcokból
        ─► AutoAssignment javaslatok ─► felhasználói jóváhagyás (review)
```

Futtatás: `PipelineWorker` / `DeepPipelineWorker` (QThread) vagy a központi `TaskManager`.

## Tesztelés, minőség, CI

- `pytest` (pytest-qt, timeout 120 s), `ruff` (E, F, W, I), opcionális `mypy`.
- Workflow-k: `build-release.yml` (macOS / Windows / Linux build + release + appcast), `test-report-dashboard.yml`, `copilot-setup-steps.yml`.
- `tests/test_gdrive_*` helyben elbukik, ha nincs `googleapiclient` / `cryptography` (nem regresszió).

## Kapcsolódó dokumentumok

- [AGENT.md](../../AGENT.md) — fejlesztői útmutató, git policy
- [CLAUDE.md](../../CLAUDE.md) — rövid kódtérkép
- [docs/blueprint.md](../blueprint.md), [docs/ARCFELISMERES_RENDSZER.md](../ARCFELISMERES_RENDSZER.md), [docs/performance_audit.md](../performance_audit.md)
- [PROJEKT_ELEMZES.md](PROJEKT_ELEMZES.md) · [NYITOTT_JEGYEK.md](NYITOTT_JEGYEK.md)
