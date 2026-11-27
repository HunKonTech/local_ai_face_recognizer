# Face-Local

**Offline face grouping and person labeling with Google Coral Edge TPU acceleration.**

Scans a folder tree of images, detects faces, groups identical people into clusters, and lets you label, merge, and browse identities — entirely local, no cloud dependency.

---

## What it does

1. **Scan** — recursively indexes images (`.jpg`, `.jpeg`, `.png`, `.webp`), hashes each file, skips unchanged ones on re-runs.
2. **Detect** — runs face detection with YuNet (default, with landmarks), a Google Coral Edge TPU (if available) or the CPU fallback (OpenCV DNN SSD / Haar cascade); optional InsightFace SCRFD co-detector. A multi-stage validator drops non-faces, half faces and nested boxes.
3. **Embed** — aligns each face and generates embeddings on **CPU**: MobileFaceNet TFLite (192-dim), SFace, InsightFace ArcFace (512-dim) or CoreML on macOS (Coral is not used for this step — no practical Edge TPU ArcFace model is publicly available).
4. **Recognize** — trains an MLP-ensemble classifier from your labeled faces and proposes assignments for unknown faces (auto-assignment review).
5. **Cluster** — groups the remaining unknown faces by cosine similarity using DBSCAN.
6. **Label** — PySide6 GUI lets you rename clusters, merge wrong splits, reassign individual faces, and mark same/different pairs.
7. **Organize** — image browser, family tree, places with map and geocoding, tagged objects, person groups, collages.
8. **Export** — CSV/JSON reports, face images, EXIF/metadata, static web gallery (Astro), local server / Docker export, `.facepack` / `.facedb` project packages.
9. **Google Drive** — optional online work mode on a Drive folder (`gdrive` extra).
10. **Record** — a built-in screen recorder (toolbar ● / ⏸ / ■) captures the app
   window, cursor and microphone for documenting meetings or image
   walk-throughs, alongside a same-named `.srt` subtitle log of which image and
   selected person was active at each moment.

Everything is persisted in a local SQLite database. No network calls are made.

### Screen recording

The recorder shells out to the system **`ffmpeg`** binary (install via
`brew install ffmpeg` on macOS, or set the path in **Settings → Recording**).
Output is written as short, independently playable MP4 **segments** (default
8 s) so an interrupted recording still leaves a usable file; the segments are
concatenated into `recording.mp4` when you stop. The microphone is always
captured. **System/speaker audio** is best-effort and only included when a
virtual loopback device is present — macOS needs
[BlackHole](https://github.com/ExistentialAudio/BlackHole) or Loopback,
Windows a WASAPI `virtual-audio-capturer`; otherwise it is silently skipped.

On **Windows** the desktop is grabbed either through the DXGI Desktop
Duplication API (`ddagrab`) or through the legacy GDI grabber (`gdigrab`). GDI
records an all-black picture on many hybrid-GPU, HDR or hardware-accelerated
desktops, so the default `auto` mode runs a one-second black-frame probe before
each recording and keeps whichever grabber actually produces an image. The mode
can be pinned, and tested on demand, under **Settings → Recording**. The final
file is checked for a black video track as well, so a failed capture is reported
instead of discovered later.

---

## Architecture

```
app/
├── main.py            Entry point (Qt app init, CLI args: --config, --db, --debug, --self-test)
├── config.py          AppConfig dataclasses + YAML loader
├── app_settings.py    QSettings INI in Documents/localAIFaceRecognizer/settings/
├── db/                SQLAlchemy ORM (31 tables: Image, Face, Person, Place, TaggedObject,
│                      Relationship, AutoAssignment, TrainingRun, …) + session_scope()
├── detectors/         YuNet, Coral Edge TPU, OpenCV SSD/Haar, InsightFace SCRFD;
│                      factory.py = create_detector() with fallback chain
├── embeddings/        MobileFaceNet TFLite, SFace, InsightFace ArcFace, CoreML;
│                      alignment.py (5-point ArcFace alignment), tta.py (flip TTA)
├── deep/              MLP-ensemble classifier, dataset, evaluation, non-face filter
├── clustering/        cluster_embeddings() — DBSCAN + same-pair constraints
├── services/          73 domain services (*_service.py): scan, detection, embedding,
│                      clustering, deep_recognition, identity, family, place, object,
│                      export, project/database packages, screen_recorder, …
├── workers/           QThreads: deep_pipeline_worker (scan → detect → embed → cleanup →
│                      train → recognize → cluster → suggest), thumbnails, exports, geocoding
├── tasks/             TaskManager — prioritized background tasks with UI visibility
├── gdrive/            Google Drive online mode (OAuth, client, cache, DB sync)
├── updater/           Sparkle bridge (macOS auto-update)
├── utils/             EXIF, image metadata, fuzzy dates, person search
└── ui/
    ├── main_window/   MainWindow = window.py + topic mixins (ui_build, pipeline, persons, …)
    ├── panels/        cluster, persons, image_browser/ (mixins), preview, family_tree,
    │                  family_search, locations, objects, collage, groups, log, sidebar
    ├── dialogs/       settings, scan modes, export, merge, suggestions, identity repair, …
    ├── widgets/       shared widgets (family tree view, maps, face grids, search selects)
    └── i18n/          t("key"); EN + HU strings in strings/sNN_*.py
```

See [AGENT.md](AGENT.md) for the full file-level map.

---

## Requirements

- **Python 3.11+**
- **Linux** (primary target — tested on Ubuntu 22.04 and Raspberry Pi OS Bookworm)
- **Google Coral USB Accelerator** *(optional)* — for Edge TPU acceleration
- Display / X11 or Wayland for the GUI

macOS and Windows are secondary targets. PySide6 and OpenCV work on both, but pycoral only supports Linux officially.

### macOS: first launch (unsigned build)

The released `.app` may not be notarized, so macOS Gatekeeper can block the first launch.

After installing:

**System Settings → Privacy & Security → Open Anyway**

If it is still blocked:

```bash
xattr -dr com.apple.quarantine /Applications/Face-Local.app
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url> face-local
cd face-local
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note on tflite-runtime:** The package is available for Linux x86-64 and ARM.
> On macOS or Windows, install TensorFlow instead: `pip install tensorflow`

### 3. Download model files

#### Face detection (CPU) — Caffe SSD res10

```bash
mkdir -p models
wget -P models/ \
  https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

wget -P models/ \
  https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

Without these files, the detector falls back to OpenCV's Haar cascade (lower quality but always available).

#### Face detection (CPU) — YuNet (recommended; enables aligned crops)

YuNet returns 5 facial landmarks per face, which the `aligned` embedding crop
mode needs to warp faces onto the ArcFace template.  When present it is used in
preference to the Caffe SSD / Haar CPU path.

```bash
wget -P models/ \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

Override the location with `detection.yunet_model_path`, or set
`detection.use_yunet: false` to force the Caffe/Haar path.

> **Changing `embedding.crop_mode`** (`legacy` → `square` → `aligned`) changes
> the geometry of every crop and therefore every embedding.  Existing
> embeddings are *not* comparable to new ones, so after switching you must
> re-run a full **re-detect + re-embed** of the library.

#### Face embedding — MobileFaceNet TFLite (CPU only)

The embedding model is NOT included in the repository.  You have three options:

**Option A — Community pre-converted model** (easiest):
```bash
# Search for "mobilefacenet tflite" on GitHub or HuggingFace.
# A commonly used one is from the sirius-ai/MobileFaceNet_TF project.
# Place the downloaded .tflite file at:
cp /path/to/mobilefacenet.tflite models/mobilefacenet.tflite
```

**Option B — Convert from ONNX yourself**:
```bash
pip install onnx onnx-tf tensorflow
# Download mobilefacenet.onnx from InsightFace model zoo
# (https://github.com/deepinsight/insightface/tree/master/model_zoo)
# Then convert to TFLite (see the comments in config.example.yaml → embedding)
```

**Option C — Use the HOG stub (development only)**:  
Leave `embedding.model_path` unset in `config.yaml`.  The app will warn you and use a low-quality HOG-based fallback.  Pipeline plumbing works, face recognition quality does not.

#### Face detection (Coral Edge TPU)

```bash
wget -P models/ \
  https://raw.githubusercontent.com/google-coral/test_data/master/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite
```

Then set in `config.yaml`:
```yaml
detection:
  coral_model_path: models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite
```

### 4. Install pycoral (Coral users only)

Follow the official guide at https://coral.ai/docs/accelerator/get-started/

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install libedgetpu1-std   # or libedgetpu1-max for max clock speed
pip install pycoral
```

### 5. Configure

```bash
cp config.example.yaml config.yaml
# Edit config.yaml — set model paths, thresholds, etc.
```

---

## Running the application

```bash
# Default (auto-discovers config.yaml in cwd)
python -m app.main

# Explicit config
python -m app.main --config /path/to/config.yaml

# Debug logging
python -m app.main --debug

# Custom database location
python -m app.main --db /tmp/test.db
```

Or if installed via pip:

```bash
face-local --config config.yaml
```

---

## GUI workflow

1. **Select Folder** — choose the root directory (or several folders) containing your images.
2. **Scan & Maintenance** — pick a mode in the scan dialog:
   - **Face Detection & Assignment** — scan new images → detect → embed → clean up → train the AI model → recognize → cluster unknowns → suggestions
   - **Train AI Model** — embed missing faces and retrain only
   - **Full Rescan** — rebuild every automatic face box from scratch (manual boxes and confirmed assignments are kept)
   - Maintenance: re-verify stored faces, identity repair, duplicate / overlap cleanup
3. **Browse** — Faces, Persons, Image browser, Family tree, Places, Objects, Collage and Groups tabs.
4. **Review** — accept or reject the AI's automatic assignments and name suggestions.
5. **Rename / Merge / Reassign** — name people, merge clusters of the same person, move or remove single faces (multi-select supported).
6. **Edit in the image browser** — draw or fix face boxes, add dates, notes, places and objects.
7. **Export** — CSV/JSON, face images, metadata, web gallery, server/Docker export or a project package.

---

## Coral vs CPU — what's real, what's not

| Stage | Coral | CPU |
|-------|-------|-----|
| Face detection | ✅ Real pycoral integration via `CoralDetector` | ✅ YuNet (default) / OpenCV DNN SSD + Haar fallback / optional InsightFace SCRFD |
| Face embedding | ❌ Not used — no practical public Edge TPU ArcFace model available | ✅ MobileFaceNet TFLite via `TFLiteEmbedder`, SFace, InsightFace ArcFace, CoreML (macOS) |
| Deep recognition | ❌ Not applicable | ✅ MLP ensemble (`app/deep/classifier.py`) |
| Clustering | ❌ Not applicable | ✅ scikit-learn DBSCAN |

**Performance expectations:**
- Coral: ~50–200 ms per image for detection (USB Accelerator speed)
- CPU DNN: ~100–500 ms per image depending on hardware
- Haar cascade: fast but misses ~30–40% of faces
- MobileFaceNet embedding: ~10–50 ms per face crop on modern x86 CPU

---

## Testing Coral fallback

To verify the CPU fallback path without Coral hardware:

```bash
# 1. Don't install pycoral — factory will auto-fallback:
python -m app.main

# 2. With pycoral installed but no USB Accelerator:
python -m app.main
# You will see in the log: "pycoral available but no Edge TPU devices found"

# 3. Set coral_model_path but unplug the USB stick:
python -m app.main
# You will see: "Coral init failed: ... — falling back to CPU"

# 4. Run tests:
pytest tests/test_detectors.py -v
# TestFactoryFallback verifies the CPU fallback path with monkeypatching
```

---

## Running tests

```bash
pytest -v

# Specific modules
pytest tests/test_clustering.py -v
pytest tests/test_database.py -v
pytest tests/test_detectors.py -v
pytest tests/test_scan_service.py -v
```

Tests do NOT require a real camera, Coral hardware, or model files.  They use in-memory SQLite databases and synthetic embeddings.

---

## Configuration reference

See [`config.example.yaml`](config.example.yaml) for all options with inline documentation.

Key thresholds:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `detection.confidence_threshold` | `0.65` | Lower → more detections, more false positives |
| `detection.landmark_geometry_enabled` | `true` | Reject non-faces (cards, hands, tree knots) via landmark geometry |
| `ai_face_detection.confidence_threshold` | `0.65` | AI pass detector confidence floor |
| `ai_face_detection.min_face_size` | `36` | AI pass minimum face box width/height (px) |
| `ai_face_detection.verification_enabled` | `true` | Re-confirm AI detections in their own crop to drop false positives |
| `clustering.epsilon` | `0.4` | Lower → stricter matching, more clusters |
| `clustering.min_samples` | `2` | Higher → singletons become noise |
| `scan.thumbnail_size` | `[128, 128]` | Larger → better visual quality, more disk space |

---

## Project status

### Working
- Full pipeline: scan → detect → validate → embed → train → recognize → cluster → review
- Detectors: YuNet (with landmarks), Coral Edge TPU, OpenCV DNN SSD + Haar fallback, optional InsightFace SCRFD
- Embedders: MobileFaceNet TFLite, SFace, InsightFace ArcFace, CoreML (macOS); 5-point alignment + flip TTA
- Deep recognition: MLP ensemble trained from labeled faces, auto-assignment review, name suggestions
- Cleanup tools: non-face / half-face filter, duplicate and overlap resolution, identity repair, retroactive verification
- Image browser with face box editing, dates, notes, places (map + geocoding), objects
- Family tree, family search, person groups, collages
- Export: CSV, JSON, images, EXIF/metadata, Astro web gallery, local server / Docker, `.facepack` / `.facedb` packages
- Moved / renamed originals re-linked by content hash
- Google Drive online mode, screen recorder, Sparkle auto-update, EN/HU UI
- Releases for macOS (signed, notarized), Windows (installer) and Linux (.deb)

### Known limitations
- Embedding model files must be downloaded separately (see Setup); without them a low-quality HOG stub is used
- Recognition quality is still being tuned (false positives, extra unknowns — see the open GitHub issues)
- Clustering of unknowns is a global re-run (not incremental)
- No HEIC support by default (extension point in `ScanConfig.image_extensions`)
- Windows system-audio recording needs a loopback device (e.g. Stereo Mix)

---

## License

[KOAI Personal Use License v1.0](LICENSE)

---

# Magyar változat

# Face-Local

**Offline arccsoportosítás és személycímkézés Google Coral Edge TPU gyorsítással.**

Beolvas egy képeket tartalmazó mappastruktúrát, arcokat érzékel, azonos személyeket klaszterekbe rendez, majd lehetővé teszi az identitások címkézését, egyesítését és böngészését teljesen helyben, felhőfüggőség nélkül.

---

## Mit csinál

1. **Beolvasás** — rekurzívan indexeli a képeket (`.jpg`, `.jpeg`, `.png`, `.webp`), minden fájlt hashel, majd újrafuttatáskor kihagyja a változatlanokat.
2. **Érzékelés** — arcérzékelést futtat YuNet-tel (alapértelmezett, landmarkokkal), Google Coral Edge TPU-val (ha elérhető), vagy CPU-s tartalékmegoldással (OpenCV DNN SSD / Haar cascade); opcionálisan InsightFace SCRFD társdetektorral. Egy többlépcsős ellenőrző kiszűri a nem-arcokat, fél arcokat és egymásba ágyazott dobozokat.
3. **Beágyazás** — igazítja az arcokat és **CPU-n** beágyazást készít: MobileFaceNet TFLite (192 dim.), SFace, InsightFace ArcFace (512 dim.) vagy macOS-en CoreML (ehhez a lépéshez a Coral nincs használatban, mert nyilvánosan nem érhető el gyakorlatban használható Edge TPU ArcFace modell).
4. **Felismerés** — a címkézett arcokból MLP-ensemble osztályozót tanít, és hozzárendelési javaslatokat ad az ismeretlen arcokra (automatikus hozzárendelések áttekintése).
5. **Klaszterezés** — a maradék ismeretlen arcokat koszinusz-hasonlóság alapján DBSCAN-nel csoportosítja.
6. **Címkézés** — a PySide6 GUI-val átnevezheted a klasztereket, összevonhatod a hibás szétválásokat, újra hozzárendelhetsz egyedi arcokat, és megjelölhetsz azonos/különböző párokat.
7. **Rendszerezés** — képböngésző, családfa, helyszínek térképpel és geokódolással, címkézett tárgyak, személycsoportok, kollázsok.
8. **Exportálás** — CSV/JSON jelentések, arcképek, EXIF/metaadatok, statikus webes galéria (Astro), helyi szerver / Docker export, `.facepack` / `.facedb` projektcsomagok.
9. **Google Drive** — opcionális online munkamód egy Drive-mappán (`gdrive` extra).

Minden adat egy helyi SQLite-adatbázisban tárolódik. Nem történik hálózati kommunikáció.

---

## Architektúra

```
app/
├── main.py            Belépési pont (Qt inicializálás, CLI: --config, --db, --debug, --self-test)
├── config.py          AppConfig dataclassok + YAML betöltő
├── app_settings.py    QSettings INI a Documents/localAIFaceRecognizer/settings/ mappában
├── db/                SQLAlchemy ORM (31 tábla: Image, Face, Person, Place, TaggedObject,
│                      Relationship, AutoAssignment, TrainingRun, …) + session_scope()
├── detectors/         YuNet, Coral Edge TPU, OpenCV SSD/Haar, InsightFace SCRFD;
│                      factory.py = create_detector() tartaléklánccal
├── embeddings/        MobileFaceNet TFLite, SFace, InsightFace ArcFace, CoreML;
│                      alignment.py (5 pontos ArcFace igazítás), tta.py (tükrözéses TTA)
├── deep/              MLP-ensemble osztályozó, adatkészlet, kiértékelés, nem-arc szűrő
├── clustering/        cluster_embeddings() — DBSCAN + same-pair korlátozások
├── services/          73 szolgáltatás (*_service.py): scan, detection, embedding,
│                      clustering, deep_recognition, identity, family, place, object,
│                      export, projekt-/adatbáziscsomag, screen_recorder, …
├── workers/           QThreadek: deep_pipeline_worker (beolvasás → érzékelés → beágyazás →
│                      tisztítás → tanítás → felismerés → klaszterezés → javaslatok),
│                      bélyegképek, exportok, geokódolás
├── tasks/             TaskManager — priorizált háttérfeladatok, UI-ban látható állapottal
├── gdrive/            Google Drive online mód (OAuth, kliens, cache, DB szinkron)
├── updater/           Sparkle híd (macOS automatikus frissítés)
├── utils/             EXIF, képmetaadat, bizonytalan dátumok, személykeresés
└── ui/
    ├── main_window/   MainWindow = window.py + témánkénti mixinek (ui_build, pipeline, persons, …)
    ├── panels/        cluster, persons, image_browser/ (mixinek), preview, family_tree,
    │                  family_search, locations, objects, collage, groups, log, sidebar
    ├── dialogs/       beállítások, beolvasási módok, export, egyesítés, javaslatok, identitás-javítás, …
    ├── widgets/       közös widgetek (családfa nézet, térképek, arcrácsok, keresők)
    └── i18n/          t("key"); EN + HU szövegek a strings/sNN_*.py fájlokban
```

A teljes, fájlszintű térkép: [AGENT.md](AGENT.md).

---

## Követelmények

- **Python 3.11+**
- **Linux** (elsődleges célplatform — Ubuntu 22.04-en és Raspberry Pi OS Bookwormön tesztelve)
- **Google Coral USB Accelerator** *(opcionális)* — Edge TPU gyorsításhoz
- Kijelző / X11 vagy Wayland a GUI-hoz

A macOS és a Windows másodlagos célplatformok. A PySide6 és az OpenCV mindkettőn működik, de a pycoral hivatalosan csak Linuxot támogat.

### macOS: első indítás (aláíratlan build)

A kiadott `.app` nincs feltétlenül notarizálva, ezért a macOS Gatekeeper az első indításkor blokkolhatja.

Telepítés után:

**System Settings → Privacy & Security → Open Anyway**

Ha továbbra is blokkolja:

```bash
xattr -dr com.apple.quarantine /Applications/Face-Local.app
```

---

## Telepítés

### 1. Klónozás és virtuális környezet létrehozása

```bash
git clone <repo-url> face-local
cd face-local
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Python-függőségek telepítése

```bash
pip install -r requirements.txt
```

> **Megjegyzés a tflite-runtime-ról:** A csomag Linux x86-64 és ARM platformokra érhető el.
> macOS-en vagy Windowson inkább a TensorFlow telepítése ajánlott: `pip install tensorflow`

### 3. Modellfájlok letöltése

#### Arcérzékelés (CPU) — Caffe SSD res10

```bash
mkdir -p models
wget -P models/ \
  https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

wget -P models/ \
  https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

E fájlok nélkül a detektor az OpenCV Haar cascade megoldására áll vissza (gyengébb minőség, de mindig elérhető).

#### Arcbeágyazás — MobileFaceNet TFLite (csak CPU)

A beágyazó modell nincs benne a repóban. Három lehetőséged van:

**A lehetőség — Közösségi, előre konvertált modell** (legegyszerűbb):
```bash
# Keress rá GitHubon vagy Hugging Face-en erre: "mobilefacenet tflite".
# Egy gyakran használt változat a sirius-ai/MobileFaceNet_TF projektből származik.
# A letöltött .tflite fájlt ide helyezd:
cp /path/to/mobilefacenet.tflite models/mobilefacenet.tflite
```

**B lehetőség — Konvertáld át magad ONNX-ből**:
```bash
pip install onnx onnx-tf tensorflow
# Töltsd le a mobilefacenet.onnx fájlt az InsightFace model zoo-ból
# (https://github.com/deepinsight/insightface/tree/master/model_zoo)
# Majd konvertáld TFLite-ra (lásd a config.example.yaml → embedding megjegyzéseit)
```

**C lehetőség — Használd a HOG csonkot (csak fejlesztéshez)**:
Hagyd üresen az `embedding.model_path` értékét a `config.yaml` fájlban. Az alkalmazás figyelmeztetést ad, és egy gyenge minőségű, HOG-alapú tartalékmegoldást használ. A pipeline működni fog, az arcfelismerés minősége viszont nem lesz megfelelő.

#### Arcérzékelés (Coral Edge TPU)

```bash
wget -P models/ \
  https://raw.githubusercontent.com/google-coral/test_data/master/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite
```

Ezután ezt állítsd be a `config.yaml` fájlban:
```yaml
detection:
  coral_model_path: models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite
```

### 4. pycoral telepítése (csak Coral felhasználóknak)

Kövesd a hivatalos útmutatót: https://coral.ai/docs/accelerator/get-started/

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install libedgetpu1-std   # vagy libedgetpu1-max a maximális órajelhez
pip install pycoral
```

### 5. Konfigurálás

```bash
cp config.example.yaml config.yaml
# Szerkeszd a config.yaml fájlt — állítsd be a modellútvonalakat, küszöböket stb.
```

---

## Az alkalmazás futtatása

```bash
# Alapértelmezett (automatikusan megkeresi a config.yaml fájlt az aktuális mappában)
python -m app.main

# Konfigurációs fájl megadása
python -m app.main --config /path/to/config.yaml

# Hibakeresési naplózás
python -m app.main --debug

# Egyéni adatbázis-hely
python -m app.main --db /tmp/test.db
```

Vagy ha `pip`-pel van telepítve:

```bash
face-local --config config.yaml
```

---

## GUI munkafolyamat

1. **Mappa kiválasztása** — válaszd ki a képeket tartalmazó gyökérkönyvtárat (vagy több mappát).
2. **Beolvasás és karbantartás** — válassz módot a beolvasási ablakban:
   - **Arcfelismerés és hozzárendelés** — új képek beolvasása → érzékelés → beágyazás → tisztítás → AI modell tanítása → felismerés → ismeretlenek klaszterezése → javaslatok
   - **AI modell tanítása** — csak a hiányzó beágyazások és újratanítás
   - **Teljes újrabeolvasás** — minden automatikus arcdoboz újraépítése (a kézi dobozok és a megerősített hozzárendelések megmaradnak)
   - Karbantartás: tárolt arcok újra-ellenőrzése, identitás-helyreállítás, duplikátum- és átfedéstisztítás
3. **Böngészés** — Arcok, Személyek, Képböngésző, Családfa, Helyszínek, Tárgyak, Kollázs és Csoportok fülek.
4. **Áttekintés** — fogadd el vagy utasítsd el az AI automatikus hozzárendeléseit és névajánlatait.
5. **Átnevezés / egyesítés / áthelyezés** — nevezd el a személyeket, vond össze az azonos személy klasztereit, helyezz át vagy távolíts el egyes arcokat (többes kijelöléssel is).
6. **Szerkesztés a képböngészőben** — arcdobozok rajzolása, javítása; dátum, megjegyzés, helyszín és tárgyak megadása.
7. **Exportálás** — CSV/JSON, arcképek, metaadatok, webes galéria, szerver/Docker export vagy projektcsomag.

---

## Coral vs CPU — mi a valóság, és mi nem

| Szakasz | Coral | CPU |
|-------|-------|-----|
| Arcérzékelés | ✅ Valódi pycoral integráció a `CoralDetector` segítségével | ✅ YuNet (alapértelmezett) / OpenCV DNN SSD + Haar tartalék / opcionális InsightFace SCRFD |
| Arcbeágyazás | ❌ Nincs használatban — nem érhető el gyakorlatban használható nyilvános Edge TPU ArcFace modell | ✅ MobileFaceNet TFLite a `TFLiteEmbedder` segítségével, SFace, InsightFace ArcFace, CoreML (macOS) |
| Mély felismerés | ❌ Nem alkalmazható | ✅ MLP-ensemble (`app/deep/classifier.py`) |
| Klaszterezés | ❌ Nem alkalmazható | ✅ scikit-learn DBSCAN |

**Várható teljesítmény:**
- Coral: kb. 50–200 ms képenként az érzékeléshez (USB Accelerator sebesség)
- CPU DNN: kb. 100–500 ms képenként, hardvertől függően
- Haar cascade: gyors, de az arcok kb. 30–40%-át kihagyhatja
- MobileFaceNet beágyazás: kb. 10–50 ms arckivágásonként modern x86 CPU-n

---

## Coral fallback tesztelése

Így ellenőrizheted a CPU-s tartalék útvonalat Coral hardver nélkül:

```bash
# 1. Ne telepítsd a pycoral csomagot — a factory automatikusan visszaáll CPU-ra:
python -m app.main

# 2. Telepített pycoral mellett, de USB Accelerator nélkül:
python -m app.main
# A naplóban ezt látod majd: "pycoral available but no Edge TPU devices found"

# 3. Állíts be coral_model_path értéket, de húzd ki az USB eszközt:
python -m app.main
# Ezt fogod látni: "Coral init failed: ... — falling back to CPU"

# 4. Futtasd a teszteket:
pytest tests/test_detectors.py -v
# A TestFactoryFallback monkeypatch segítségével ellenőrzi a CPU fallback útvonalat
```

---

## Tesztek futtatása

```bash
pytest -v

# Konkrét modulok
pytest tests/test_clustering.py -v
pytest tests/test_database.py -v
pytest tests/test_detectors.py -v
pytest tests/test_scan_service.py -v
```

A tesztekhez NEM szükséges valódi kamera, Coral hardver vagy modellfájl. Memóriában futó SQLite-adatbázisokat és szintetikus beágyazásokat használnak.

---

## Konfigurációs referencia

Az összes opcióért és a beágyazott dokumentációért lásd: [`config.example.yaml`](config.example.yaml).

Fontos küszöbértékek:

| Paraméter | Alapértelmezett | Hatás |
|-----------|------------------|-------|
| `detection.confidence_threshold` | `0.65` | Alacsonyabb érték → több találat, több hamis pozitív |
| `detection.landmark_geometry_enabled` | `true` | Nem-arcok (kártya, kéz, fa-göcsört) kiszűrése a landmark-geometria alapján |
| `ai_face_detection.confidence_threshold` | `0.65` | Az AI-menet detektor-konfidencia küszöbe |
| `ai_face_detection.min_face_size` | `36` | Az AI-menet minimális arc-méret (px) |
| `ai_face_detection.verification_enabled` | `true` | Az AI-találatok újra-ellenőrzése saját kivágásban a hamis pozitívok ellen |
| `clustering.epsilon` | `0.4` | Alacsonyabb érték → szigorúbb egyezés, több klaszter |
| `clustering.min_samples` | `2` | Magasabb érték → az egyedülálló elemek zajnak számítanak |
| `scan.thumbnail_size` | `[128, 128]` | Nagyobb érték → jobb vizuális minőség, több lemezhasználat |

---

## Projektállapot

### Működik
- Teljes pipeline: beolvasás → érzékelés → ellenőrzés → beágyazás → tanítás → felismerés → klaszterezés → áttekintés
- Detektorok: YuNet (landmarkokkal), Coral Edge TPU, OpenCV DNN SSD + Haar tartalék, opcionális InsightFace SCRFD
- Beágyazók: MobileFaceNet TFLite, SFace, InsightFace ArcFace, CoreML (macOS); 5 pontos igazítás + tükrözéses TTA
- Mély felismerés: címkézett arcokból tanított MLP-ensemble, automatikus hozzárendelések áttekintése, névajánlatok
- Tisztító eszközök: nem-arc / fél arc szűrő, duplikátum- és átfedésfeloldás, identitás-helyreállítás, utólagos ellenőrzés
- Képböngésző arcdoboz-szerkesztéssel, dátumokkal, megjegyzésekkel, helyszínekkel (térkép + geokódolás), tárgyakkal
- Családfa, családi keresés, személycsoportok, kollázsok
- Export: CSV, JSON, képek, EXIF/metaadat, Astro webes galéria, helyi szerver / Docker, `.facepack` / `.facedb` csomagok
- Áthelyezett / átnevezett eredeti képek újrakapcsolása tartalom-hash alapján
- Google Drive online mód, képernyőrögzítő, Sparkle automatikus frissítés, EN/HU felület
- Kiadások macOS-re (aláírt, notarizált), Windowsra (telepítő) és Linuxra (.deb)

### Ismert korlátok
- A beágyazó modellfájlokat külön kell letölteni (lásd: Telepítés); nélkülük gyenge minőségű HOG csonk fut
- A felismerési minőség finomhangolása folyamatban (hamis pozitívok, fölösleges ismeretlenek — lásd a nyitott GitHub jegyeket)
- Az ismeretlenek klaszterezése teljes újrafuttatással történik (nem inkrementális)
- Alapból nincs HEIC támogatás (kiterjesztési pont: `ScanConfig.image_extensions`)
- Windowson a rendszerhang rögzítéséhez loopback eszköz kell (pl. Stereo Mix)

---

## Licenc

[KOAI Personal Use License v1.0](LICENSE)
