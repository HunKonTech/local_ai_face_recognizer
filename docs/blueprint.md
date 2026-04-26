# Face-Local Blueprint

This document describes the entire application architecture so that an LLM can reproduce it from scratch.

## 1. Overview

Face-Local is a **desktop GUI application** for offline face detection, embedding, clustering, and person labeling in photo collections. It runs entirely locally (no internet required), stores data in SQLite, and uses PySide6 for the Qt-based UI.

### Key characteristics
- **Language**: Python 3.11+
- **GUI Framework**: PySide6 (Qt bindings)
- **Database**: SQLite (via SQLAlchemy ORM)
- **Face Detection**: TensorFlow Lite TFLite models (Coral EdgeTPU or CPU fallback)
- **Face Embedding**: TFLite MobileFaceNet model (CPU only)
- **Clustering**: scikit-learn DBSCAN with cosine distance
- **Image Processing**: OpenCV (cv2)
- **Configuration**: YAML files
- **Platforms**: macOS, Windows, Linux

---

## 2. Application Entry Point

**File**: `app/main.py`

```
python -m app.main                      # default config
python -m app.main --config config.yaml # explicit config
python -m app.main --debug              # verbose logging
python -m app.main --db /tmp/test.db    # override database path
```

### Flow at startup:
1. Parse CLI arguments (`argparse`)
2. Setup logging (`app.logging_setup.setup_logging`)
3. Load configuration (`app.config.load_config`)
4. Setup i18n translations (`app.ui.i18n.load_prefs`)
5. Create QApplication with dark palette
6. Create and show `MainWindow`
7. Run event loop (`app.exec()`)

---

## 3. Configuration System

**File**: `app/config.py`

### Config hierarchy
```
AppConfig
├── detection: DetectionConfig
│   ├── confidence_threshold: float = 0.65
│   ├── min_face_size: int = 50
│   ├── coral_model_path: Optional[str] = None
│   ├── cpu_model_path: Optional[str] = None
│   └── cpu_model_input_size: tuple = (300, 300)
├── embedding: EmbeddingConfig
│   ├── model_path: Optional[str] = None
│   ├── input_size: tuple = (112, 112)
│   └── embedding_dim: int = 192
├── clustering: ClusteringConfig
│   ├── epsilon: float = 0.4
│   ├── min_samples: int = 2
│   └── metric: str = "cosine"
├── storage: StorageConfig
│   ├── crops_dir: str = "data/crops"
│   └── db_path: str = "data/faces.db"
├── scan: ScanConfig
│   ├── image_extensions: list = [".jpg", ".jpeg", ".png", ".webp"]
│   ├── worker_threads: int = 2
│   └── thumbnail_size: tuple = (128, 128)
└── base_dir: str (resolved path base)
```

### Config loading order:
1. Explicit `--config` argument
2. `FACE_LOCAL_CONFIG` environment variable
3. Auto-discovery: `config.yaml`, then `config.example.yaml`
4. Freeze mode adds user config directory paths

**Key functions**:
- `load_config(config_path: Optional[str]) -> AppConfig`
- `save_db_path(new_db_path: str, config_path: Optional[str]) -> None`

---

## 4. Database Schema

**File**: `app/db/models.py`
**File**: `app/db/database.py`

### Tables

#### `images`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| file_path | TEXT UNIQUE | Absolute path to image |
| file_hash | VARCHAR(64) | SHA-256 of content |
| file_mtime | FLOAT | os.path.getmtime() |
| width | INTEGER | Image width in pixels |
| height | INTEGER | Image height in pixels |
| detection_done | BOOLEAN | Detection completed flag |
| embedding_done | BOOLEAN | Embedding completed flag |
| created_at | DATETIME | Creation timestamp |
| updated_at | DATETIME | Last update timestamp |

**Relationships**: `faces` (one-to-many, cascade delete)

#### `faces`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| image_id | INTEGER FK | Reference to `images.id` |
| person_id | INTEGER FK | Reference to `persons.id` (nullable) |
| bbox_x, bbox_y, bbox_w, bbox_h | INTEGER | Bounding box in original image |
| confidence | FLOAT | Detection confidence [0.0-1.0] |
| detector_backend | VARCHAR(32) | "coral" or "cpu" |
| crop_path | TEXT | Path to face thumbnail |
| _embedding | BLOB | Serialized float32 numpy array |
| is_excluded | BOOLEAN | Excluded from clustering |
| created_at | DATETIME | Creation timestamp |

**Relationships**: `image`, `person`, `corrections_a`, `corrections_b`

**Helper methods**:
- `Face.get_embedding() -> Optional[np.ndarray]`
- `Face.set_embedding(vector: np.ndarray) -> None`

#### `persons`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| name | VARCHAR(255) | Person name |
| is_auto_named | BOOLEAN | True if auto-generated (e.g., "Unknown 1") |
| thumbnail_path | TEXT | Representative face crop |
| notes | TEXT | User notes |
| created_at | DATETIME | Creation timestamp |
| updated_at | DATETIME | Last update timestamp |

**Relationships**: `faces` (one-to-many)

#### `face_corrections`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| face_id_a | INTEGER FK | First face |
| face_id_b | INTEGER FK | Second face |
| same_person | BOOLEAN | True=same, False=different |
| created_at | DATETIME | Creation timestamp |

**Constraint**: UNIQUE(face_id_a, face_id_b)

#### `collages`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| collage_uid | VARCHAR(64) | Picasa album UID |
| source_file | TEXT UNIQUE | Path to .cxf/.cfx file |
| album_title | TEXT | Album title |
| album_date | VARCHAR(128) | Album date |
| format_width | INTEGER | Canvas width |
| format_height | INTEGER | Canvas height |
| orientation | VARCHAR(32) | "landscape" or "portrait" |
| bg_color | VARCHAR(16) | Background color |
| spacing | FLOAT | Node spacing |
| created_at | DATETIME | Creation timestamp |
| updated_at | DATETIME | Last update timestamp |

**Relationships**: `nodes` (one-to-many, cascade delete)

#### `collage_nodes`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| collage_id | INTEGER FK | Reference to `collages.id` |
| node_uid | VARCHAR(64) | Node UID from XML |
| rel_x, rel_y, rel_w, rel_h | FLOAT | Normalized coordinates [0,1] |
| theta | FLOAT | Rotation in radians |
| scale | FLOAT | Picasa zoom scale (100=fit) |
| theme | VARCHAR(64) | Picasa theme name |
| src_raw | TEXT | Original source path from XML |
| src_resolved | TEXT | Resolved absolute path |
| src_missing | BOOLEAN | True if file not found |
| image_id | INTEGER FK | Reference to `images.id` (nullable) |
| year | VARCHAR(16) | Year extracted from filename |
| location | VARCHAR(256) | Location |
| event_name | VARCHAR(256) | Event name |
| notes | TEXT | User notes |
| created_at | DATETIME | Creation timestamp |
| updated_at | DATETIME | Last update timestamp |

**Methods**:
- `CollageNode.pixel_bbox(collage_w, collage_h) -> (px, py, pw, ph)`

### Database initialization
**File**: `app/db/database.py`

```python
# SQLite with WAL mode
init_db(db_path: Path) -> Engine
get_engine() -> Engine
get_session() -> Session  # caller must close
session_scope() -> Generator[Session]  # auto commit/rollback
```

---

## 5. Core Services

### 5.1 ScanService

**File**: `app/services/scan_service.py`

**Responsibility**: Discover image files and index them in the database.

**Key classes**:
- `ScanService(session, config, progress_cb)`

**Methods**:
- `scan(root_folder: str) -> List[int]` — returns IDs of new/changed images

**Process**:
1. Recursively enumerate files with matching extensions (`discover_images`)
2. For each file:
   - Check if path already exists in DB
   - If exists with same mtime + `detection_done=True` → skip
   - Otherwise compute SHA-256 hash
   - If hash changed → reset processing flags
   - If new file → insert record
3. Return list of image IDs needing processing

**Functions**:
- `hash_file(path: Path) -> str` — SHA-256, 4MB chunks
- `discover_images(root: Path, extensions: List[str]) -> Generator[Path]`

### 5.2 DetectionService

**File**: `app/services/detection_service.py`

**Responsibility**: Run face detection on images, save face records and crops.

**Key classes**:
- `DetectionService(session, detector: FaceDetector, config, progress_cb)`

**Methods**:
- `process(image_ids: List[int]) -> int` — returns total faces detected

**Process**:
1. For each image ID:
   - Load image with OpenCV (`cv2.imread`)
   - Run detector (`detector.detect(image_bgr, confidence_threshold, min_face_size)`)
   - Delete existing face records for this image
   - For each detection:
     - Save face crop to disk (`save_face_crop`)
     - Create `Face` record in DB
   - Mark image as `detection_done=True`
2. Handle TPU fallback: if Coral fails, switch to CPU detector

**Used by**: `PipelineWorker._run_detection`

### 5.3 EmbeddingService

**File**: `app/services/embedding_service.py`

**Responsibility**: Generate embedding vectors for faces.

**Key classes**:
- `EmbeddingService(session, embedder: FaceEmbedder, config, progress_cb)`

**Methods**:
- `process_pending() -> int` — returns number of faces embedded

**Process**:
1. Query all faces where `_embedding IS NULL` and `is_excluded=False`
2. For each face:
   - Load crop image from disk
   - Run embedder (`embedder.embed(img_bgr)`)
   - Store embedding (`face.set_embedding(vector)`)
3. Commit every 50 faces

**Used by**: `PipelineWorker._run_embedding`

### 5.4 ClusteringService

**File**: `app/services/clustering_service.py`

**Responsibility**: Group faces into person clusters using DBSCAN.

**Key classes**:
- `ClusteringService(session, config)`

**Methods**:
- `run() -> int` — returns number of persons
- `recluster() -> int` — re-run with current corrections

**Process**:
1. Load all embedded, non-excluded faces
2. Load manual corrections (`same_pairs`, `diff_pairs`)
3. Run `cluster_embeddings()` (DBSCAN)
4. Map labels to Person records:
   - Each unique label → Person record
   - Label -1 (noise) → individual singleton Person
   - "Unknown N" naming for auto-created persons
5. Assign `face.person_id` for each face
6. Commit

**Used by**: `PipelineWorker._run_clustering`

### 5.5 IdentityService

**File**: `app/services/identity_service.py`

**Responsibility**: User-driven operations on Person clusters.

**Key classes**:
- `IdentityService(session)`

**Methods**:
- `rename_person(person_id, new_name) -> Person`
- `merge_persons(source_id, target_id) -> Person` — moves all faces, deletes source
- `delete_person(person_id) -> None` — unassigns faces and deletes person
- `reassign_face(face_id, target_person_id) -> Face`
- `remove_face_from_cluster(face_id) -> Face` — unassigns, marks `is_excluded=True`
- `exclude_face(face_id) -> Face` — marks `is_excluded=True`
- `record_same(face_id_a, face_id_b) -> FaceCorrection`
- `record_different(face_id_a, face_id_b) -> FaceCorrection`
- `list_persons(named_only, search) -> List[Person]`
- `get_faces_for_person(person_id) -> List[Face]`

### 5.6 ExportService

**File**: `app/services/export_service.py`

**Responsibility**: Export face images and metadata.

**Key classes**:
- `ExportService(session)`

**Methods**:
- `export_person_images(person_id, target_dir, copy_originals) -> int` — copy face crops/originals
- `export_csv(target_path, person_id) -> Path` — CSV report
- `export_json(target_path, person_id) -> Path` — JSON report
- `export_html(target_dir, person_id) -> Path` — static HTML gallery
- `export_collage_html(target_dir, collage_id) -> Path` — collage HTML gallery

### 5.7 CollageService

**File**: `app/services/collage_service.py`

**Responsibility**: Picasa collage import and rendering.

**Key classes**:
- `CollageService(session)`

**Methods**:
- `import_collage(file_path, search_roots, overwrite) -> Collage`
- `relink_images(collage_id) -> int` — re-link nodes to Image records
- `list_collages() -> List[Collage]`
- `get_collage(collage_id) -> Optional[Collage]`
- `get_nodes(collage_id) -> List[CollageNode]`
- `update_node_metadata(node_id, year, location, event_name, notes) -> CollageNode`
- `get_faces_for_node(node) -> List[Face]`
- `projected_faces(collage, render_w, render_h) -> List[Dict]` — face projections
- `render_collage_image(collage, render_h, draw_borders, draw_faces) -> np.ndarray`
- `export_annotated_collage(collage, output_dir, render_h) -> Path`

### 5.8 CollageParser

**File**: `app/services/collage_parser.py`

**Responsibility**: Parse Picasa collage XML files (.cxf/.cfx).

**Key classes**:
- `CollageNodeData` — dataclass for node data (no ORM)
- `CollageData` — dataclass for full collage (no ORM)

**Functions**:
- `parse_collage_file(file_path, search_roots) -> CollageData`
- `project_face_to_collage(face_bbox, img_w, img_h, node, collage_w, collage_h) -> Optional[tuple]`

**Path resolution strategy**:
1. Try exact path (Windows/Wine)
2. Strip `[X]\` drive prefix, try relative to collage directory
3. Try common POSIX mount points
4. Try search_roots
5. Filename-only fallback search

---

## 6. Detectors

**File**: `app/detectors/base.py`

### Interface
```python
class FaceDetector(ABC):
    @property
    def backend_name(self) -> str: ...

    @abstractmethod
    def detect(image_bgr: np.ndarray, confidence_threshold: float) -> List[Detection]: ...

@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    confidence: float
```

**Implementations**:
- `app/detectors/coral_detector.py` — Coral EdgeTPU (if available)
- `app/detectors/cpu_detector.py` — CPU via OpenCV DNN or TFLite

**Factory**: `app/detectors/factory.create_detector(config) -> FaceDetector`

---

## 7. Embedders

**File**: `app/embeddings/base.py`

### Interface
```python
class FaceEmbedder(ABC):
    @property
    def embedding_dim(self) -> int: ...

    @abstractmethod
    def embed(face_bgr: np.ndarray) -> np.ndarray: ...
```

**Implementation**: `app/embeddings/tflite_embedder.TFLiteEmbedder`

---

## 8. Clustering

**File**: `app/clustering/clusterer.py`

**Function**: `cluster_embeddings(face_ids, embeddings, epsilon, min_samples, metric, same_pairs, diff_pairs) -> Dict[int, int]`

**Process**:
1. Stack embeddings into matrix, L2-normalize
2. Run DBSCAN (`eps=epsilon`, `min_samples=min_samples`, `metric=metric`)
3. Apply manual constraints (`same_pairs` merge)
4. Return `{face_id: cluster_label}` mapping

**Helper functions**:
- `compute_centroid(embeddings) -> np.ndarray`
- `cosine_distance(a, b) -> float`

---

## 9. Processing Pipeline

**File**: `app/workers/pipeline_worker.py`

**Class**: `PipelineWorker(QThread)`

Runs in a background thread to keep GUI responsive.

**Signals**:
- `progress(current, total, stage, detail)`
- `log_message(message)`
- `finished(success, summary)`
- `error(message)`

**Methods**:
- `abort() -> None` — request graceful stop
- `run() -> None` — called by QThread.start()

### Pipeline stages (in order):

#### Stage 1: Scan
```
_on_scan clicked → PipelineWorker.start()
  → _run_scan()
    → ScanService(session, config.scan, progress_cb).scan(root_folder)
    → returns List[int] of new/changed image IDs
```

#### Stage 2: Detection
```
_run_detection(image_ids)
  → create_detector(config.detection) → FaceDetector
  → DetectionService(session, detector, config, progress_cb).process(image_ids)
  → returns total faces detected
```

#### Stage 3: Embedding
```
_run_embedding()
  → TFLiteEmbedder(config.embedding)
  → EmbeddingService(session, embedder, config, progress_cb).process_pending()
  → returns number of faces embedded
```

#### Stage 4: Clustering
```
_run_clustering()
  → ClusteringService(session, config.clustering).run()
  → returns number of persons
```

---

## 10. UI Architecture

### Main Window

**File**: `app/ui/main_window.py`

**Class**: `MainWindow(QMainWindow)`

**Layout**:
- **Toolbar**: Action buttons for folder selection, scan, export, settings, collage import/export
- **Central**: QTabWidget with two tabs:
  - Tab 0: "Arcfelismerés" (Face Recognition) — 3-panel layout
  - Tab 1: "Kollázs" (Collage) — CollagePanel
- **Dock**: LogPanel at bottom
- **Status bar**: Progress bar + status label

### 3-Panel Layout (Face Recognition tab)

```
┌─────────────────┬───────────────────────────┬──────────────────┐
│  SidebarPanel   │     ClusterPanel          │  PreviewPanel    │
│  (260-400px)    │     (flexible)            │  (280px min)     │
│                 │                           │                  │
│  ┌───────────┐  │  [Header: person name]    │  [Image preview  │
│  │ All Faces │  │                           │   with face      │
│  │ (grid)    │  │  [Face thumbnails grid]   │   boxes]         │
│  └───────────┘  │                           │                  │
│                 │  [Action buttons]         │  [Open/Zoom]     │
│  ┌───────────┐  │  - Átnevezés            │                  │
│  │ Search    │  │  - Összefűzés           │                  │
│  │ ────────  │  │  - Törlés              │                  │
│  │ Person 1  │  │  - Arc eltávolítása     │                  │
│  │ Person 2  │  │  - Átassignálás         │                  │
│  │ ...       │  │                           │                  │
│  └───────────┘  │                           │                  │
│                 │                           │                  │
│  [Re-cluster]   │                           │                  │
└─────────────────┴───────────────────────────┴──────────────────┘
```

**Signals flow**:
1. User clicks person in SidebarPanel → `person_selected(int person_id)`
2. MainWindow queries DB, shows faces in ClusterPanel
3. User clicks face thumbnail → `face_selected(int face_id)`
4. MainWindow loads image + face data, shows in PreviewPanel

### Panels

| Panel | File | Purpose |
|-------|------|---------|
| SidebarPanel | `app/ui/panels/sidebar_panel.py` | Person list + face thumbnail grid, search |
| ClusterPanel | `app/ui/panels/cluster_panel.py` | Face grid for selected person |
| PreviewPanel | `app/ui/panels/preview_panel.py` | Image preview with face boxes |
| CollagePanel | `app/ui/panels/collage_panel.py` | Collage list and viewer |
| LogPanel | `app/ui/panels/log_panel.py` | Activity log display |

### Dialogs

| Dialog | File | Purpose |
|--------|------|---------|
| RenameDialog | `app/ui/dialogs/rename_dialog.py` | Rename a person |
| MergeDialog | `app/ui/dialogs/merge_dialog.py` | Merge/split persons |
| SettingsDialog | `app/ui/dialogs/settings_dialog.py` | App settings, DB path, language |
| ExportDialog | `app/ui/dialogs/export_dialog.py` | Export options (images, CSV, JSON, HTML) |
| UpdateDialog | `app/ui/dialogs/update_dialog.py` | GitHub update notification |
| NoFaceImagesDialog | `app/ui/dialogs/manual_face_dialog.py` | View/manage images with no detected faces |

### i18n System

**File**: `app/ui/i18n.py`

- `t(key, **kwargs) -> str` — translate by key
- `load_prefs() -> None` — load language preference from QSettings
- Language files stored in `app/ui/i18n/` directory

---

## 11. File Structure

```
app/
├── __init__.py              # __version__ variable
├── __main__.py              # Entry point: python -m app.main
├── main.py                  # Application main() function
├── config.py                # Configuration loading
├── paths.py                 # Path helpers (is_frozen, bundle_root, etc.)
├── logging_setup.py         # Logging configuration

├── db/
│   ├── __init__.py
│   ├── database.py          # SQLAlchemy engine, session_scope
│   └── models.py           # ORM models: Image, Face, Person, Collage, CollageNode, FaceCorrection

├── detectors/
│   ├── __init__.py
│   ├── base.py             # FaceDetector ABC, Detection dataclass
│   ├── factory.py          # create_detector()
│   ├── cpu_detector.py     # CPU detector (OpenCV DNN)
│   └── coral_detector.py   # Coral EdgeTPU detector

├── embeddings/
│   ├── __init__.py
│   ├── base.py             # FaceEmbedder ABC
│   └── tflite_embedder.py  # TFLite embedder implementation

├── clustering/
│   ├── __init__.py
│   └── clusterer.py        # DBSCAN clustering with constraints

├── services/
│   ├── __init__.py
│   ├── scan_service.py         # Image discovery and indexing
│   ├── detection_service.py    # Face detection pipeline
│   ├── embedding_service.py   # Face embedding generation
│   ├── clustering_service.py   # DBSCAN person clustering
│   ├── identity_service.py     # Person/face CRUD operations
│   ├── export_service.py       # Export (images, CSV, JSON, HTML)
│   ├── collage_service.py      # Picasa collage management
│   ├── collage_parser.py      # Collage XML parser
│   └── update_service.py       # GitHub release checker

├── workers/
│   ├── __init__.py
│   └── pipeline_worker.py     # Background QThread pipeline

├── ui/
│   ├── __init__.py
│   ├── i18n.py               # Internationalization
│   ├── main_window.py         # MainWindow
│   ├── panels/
│   │   ├── __init__.py
│   │   ├── sidebar_panel.py   # Person list + face grid
│   │   ├── cluster_panel.py   # Face grid for person
│   │   ├── preview_panel.py   # Image + face preview
│   │   ├── collage_panel.py   # Collage viewer
│   │   └── log_panel.py      # Activity log
│   └── dialogs/
│       ├── __init__.py
│       ├── rename_dialog.py
│       ├── merge_dialog.py
│       ├── settings_dialog.py
│       ├── export_dialog.py
│       ├── update_dialog.py
│       └── manual_face_dialog.py

└── utils/
    ├── __init__.py
    └── image_utils.py       # save_face_crop, image processing
```

---

## 12. Processing Flow Diagram

```
User selects folder
        │
        ▼
┌─────────────────────────────────────────────────┐
│  PipelineWorker (QThread)                       │
│                                                 │
│  Stage 1: ScanService.scan()                    │
│    └─ discovers files, computes hashes          │
│    └─ inserts/updates Image records             │
│    └─ returns new/changed image IDs            │
│                                                 │
│  Stage 2: DetectionService.process()            │
│    └─ creates FaceDetector (Coral or CPU)       │
│    └─ for each image:                           │
│        └─ cv2.imread()                          │
│        └─ detector.detect() → List[Detection]  │
│        └─ save_face_crop() → crop files         │
│        └─ Face records in DB                    │
│                                                 │
│  Stage 3: EmbeddingService.process_pending()    │
│    └─ creates TFLiteEmbedder                    │
│    └─ for each unembedded face:                 │
│        └─ cv2.imread(crop)                     │
│        └─ embedder.embed() → vector             │
│        └─ face.set_embedding()                  │
│                                                 │
│  Stage 4: ClusteringService.run()               │
│    └─ loads all embeddings from DB              │
│    └─ loads FaceCorrection pairs                 │
│    └─ cluster_embeddings() → label_map          │
│    └─ creates/reuses Person records             │
│    └─ assigns face.person_id                    │
│                                                 │
└─────────────────────────────────────────────────┘
        │
        ▼
MainWindow receives finished signal
        │
        ▼
_refresh_persons() → SidebarPanel.populate()
        │
        ▼
User clicks person → _on_person_selected()
        │
        ▼
_show_person() → ClusterPanel.show_person()
```

---

## 13. Key Data Flows

### Image to Person flow
```
Image file on disk
    │
    ▼ (ScanService)
Image record in DB (file_path, file_hash, file_mtime)
    │
    ▼ (DetectionService)
Face records (bbox, confidence, crop_path)
    │
    ▼ (EmbeddingService)
Face records with _embedding (serialized numpy array)
    │
    ▼ (ClusteringService)
Person records + face.person_id assigned
```

### Face crop saving
```
Original image (cv2.imread)
    │
    ▼
detector.detect() → List[Detection] (x, y, w, h, confidence)
    │
    ▼
For each Detection:
  └─ save_face_crop(img_bgr, detection, crops_dir, image_id, thumbnail_size, face_index)
      │
      ▼
      crop = img_bgr[y:y+h, x:x+w]  (or with padding)
      │
      ▼
      Resize to thumbnail_size (128x128)
      │
      ▼
      Save as: {crops_dir}/{image_id}/{face_id}_{face_index}.jpg
```

---

## 14. Dependencies

### Core dependencies
```
PySide6              # Qt GUI framework
SQLAlchemy           # ORM
opencv-python        # Image processing
numpy                # Array operations
scikit-learn         # DBSCAN clustering
Pillow               # Image loading
```

### ML models
```
TensorFlow Lite      # TFLite runtime (via tflite-runtime or tensorflow)
EdgeTPU runtime     # Optional: Coral USB accelerator
```

### Configuration
```
PyYAML               # YAML config parsing
```

### Development
```
pytest               # Testing
```

---

## 15. Build & Distribution

### Build scripts
- `scripts/package_app.py` — pyinstaller packaging
- `scripts/build_windows_installer.iss` — Inno Setup for Windows
- `scripts/build_linux_deb.sh` — Linux DEB builder
- `.github/workflows/build-release.yml` — CI/CD release pipeline

### Frozen mode detection
`app/paths.py` provides:
- `is_frozen() -> bool` — running as packaged app
- `bundle_root() -> Path` — app bundle directory
- `user_data_dir() -> Path` — user-writable data directory
- `user_config_dir() -> Path` — user config directory

---

## 16. Key Implementation Patterns

### Database session pattern
```python
from app.db.database import session_scope

with session_scope() as session:
    # work with session
    session.add(obj)
    # auto-commits on exit
    # auto-rollbacks on exception
```

### Progress callback pattern
```python
def progress_cb(current: int, total: Optional[int], detail: str):
    pass

svc = SomeService(session, config, progress_cb)
```

### Qt signal pattern
```python
class Worker(QThread):
    progress = Signal(int, int, str, str)  # current, total, stage, detail
    finished = Signal(bool, str)             # success, summary
```

### Service instantiation pattern
```python
with session_scope() as session:
    svc = Service(session, config)
    result = svc.do_work()
# session auto-closed
```

### Face embedding storage
```python
# Store
face._embedding = vector.astype(np.float32).tobytes()

# Retrieve
vector = np.frombuffer(face._embedding, dtype=np.float32).copy()
```

### Relative path resolution
```python
config = load_config()
crop_path = config.crops_dir_resolved / "face.jpg"  # Path resolution
```