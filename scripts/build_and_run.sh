#!/usr/bin/env bash
# Build and run script for Linux and macOS

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

cd "$REPO_ROOT"

# ── Find Python 3.11+ ────────────────────────────────────────────────────────
find_python() {
    for candidate in "${PYTHON:-}" python3.13 python3.12 python3.11; do
        [ -z "$candidate" ] && continue
        if command -v "$candidate" &>/dev/null; then
            local major minor
            major=$("$candidate" -c "import sys; print(sys.version_info.major)")
            minor=$("$candidate" -c "import sys; print(sys.version_info.minor)")
            if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
                echo "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

echo "==> Checking Python..."
if ! PYTHON=$(find_python); then
    echo "ERROR: Python 3.11+ not found." >&2
    echo "       Install it from https://www.python.org/downloads/ and retry." >&2
    exit 1
fi

EXPECTED_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "    Using Python $EXPECTED_VER ($PYTHON)"

# ── libedgetpu system driver ─────────────────────────────────────────────────
install_libedgetpu_macos() {
    if ! system_profiler SPUSBDataType 2>/dev/null | grep -q "Global Unichip\|Coral"; then
        echo "    No Coral USB device detected — skipping libedgetpu install."
        return 0
    fi
    if brew list --cask coral-usb-accelerator &>/dev/null 2>&1; then
        echo "    libedgetpu (coral-usb-accelerator) already installed."
        return 0
    fi
    echo "    Installing libedgetpu via Homebrew..."
    if ! command -v brew &>/dev/null; then
        echo "    WARNING: Homebrew not found. Install it from https://brew.sh/ then re-run." >&2
        return 1
    fi
    brew install --cask coral-usb-accelerator
}

install_libedgetpu_linux() {
    if ldconfig -p 2>/dev/null | grep -q "libedgetpu"; then
        echo "    libedgetpu already installed."
        return 0
    fi
    echo "    Installing libedgetpu..."
    if ! command -v apt-get &>/dev/null; then
        echo "    WARNING: apt-get not found. Install libedgetpu manually:" >&2
        echo "             https://coral.ai/software/#debian-packages" >&2
        return 1
    fi
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list > /dev/null
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | sudo apt-key add - > /dev/null 2>&1
    sudo apt-get update -qq
    sudo apt-get install -y libedgetpu1-std
}

echo "==> Checking Coral Edge TPU driver (libedgetpu)..."
OS="$(uname -s)"
case "$OS" in
    Darwin) install_libedgetpu_macos ;;
    Linux)  install_libedgetpu_linux ;;
    *)      echo "    Unsupported OS '$OS' — skipping libedgetpu check." ;;
esac

# ── Virtual environment ───────────────────────────────────────────────────────
echo "==> Setting up virtual environment at $VENV_DIR ..."

if [ -f "$VENV_DIR/bin/python" ]; then
    VENV_VER=$("$VENV_DIR/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
    if [ "$VENV_VER" != "$EXPECTED_VER" ]; then
        echo "    Stale venv (Python $VENV_VER) — removing and rebuilding with $EXPECTED_VER..."
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -f "$VENV_DIR/bin/python" ]; then
    "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── Python dependencies ───────────────────────────────────────────────────────
echo "==> Installing / updating dependencies..."
pip install --upgrade pip setuptools wheel --quiet
pip install -e ".[dev]" --quiet

echo "==> Trying optional TPU packages (ai-edge-litert + pycoral)..."
if pip install -e ".[tflite]" --quiet 2>/dev/null; then
    echo "    TPU support enabled."
else
    echo "    WARNING: ai-edge-litert or pycoral not available for Python $EXPECTED_VER."
    echo "    Coral/TPU features will be disabled at runtime."
    echo "    See: https://coral.ai/software/#pycoral-api"
fi

# ── Download model files ──────────────────────────────────────────────────────
echo "==> Checking model files..."
MODELS_DIR="$REPO_ROOT/models"
mkdir -p "$MODELS_DIR"

download_if_missing() {
    local dest="$1" url="$2" label="$3"
    if [ -f "$dest" ]; then
        echo "    [ok] $label"
        return 0
    fi
    echo "    Downloading $label ..."
    if command -v curl &>/dev/null; then
        curl -fsSL "$url" -o "$dest" || { echo "    WARNING: Failed to download $label" >&2; return 1; }
    elif command -v wget &>/dev/null; then
        wget -q "$url" -O "$dest" || { echo "    WARNING: Failed to download $label" >&2; return 1; }
    else
        echo "    WARNING: curl/wget not found — cannot download $label" >&2
        return 1
    fi
    echo "    [ok] $label downloaded"
}

# Caffe SSD CPU face detector (OpenCV res10)
download_if_missing \
    "$MODELS_DIR/deploy.prototxt" \
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt" \
    "deploy.prototxt (CPU detector)"

download_if_missing \
    "$MODELS_DIR/res10_300x300_ssd_iter_140000.caffemodel" \
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel" \
    "res10_300x300_ssd_iter_140000.caffemodel (CPU detector)"

# Coral Edge TPU face detection model
download_if_missing \
    "$MODELS_DIR/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite" \
    "https://raw.githubusercontent.com/google-coral/test_data/master/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite" \
    "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite (Coral detector)"

# MobileFaceNet embedding model — auto-download unavailable, guide the user
if [ ! -f "$MODELS_DIR/mobilefacenet.tflite" ]; then
    echo ""
    echo "    ┌─────────────────────────────────────────────────────────────────┐"
    echo "    │  EMBEDDING MODEL MISSING — face grouping quality will be poor   │"
    echo "    │                                                                 │"
    echo "    │  Download MobileFaceNet TFLite and place it at:                │"
    echo "    │    models/mobilefacenet.tflite                                 │"
    echo "    │                                                                 │"
    echo "    │  Source: https://github.com/sirius-ai/MobileFaceNet_TF         │"
    echo "    │  (convert the .pb checkpoint to TFLite, or use a community     │"
    echo "    │   pre-converted .tflite from the insightface model zoo)         │"
    echo "    └─────────────────────────────────────────────────────────────────┘"
    echo ""
fi

# ── Auto-generate config.yaml if missing ─────────────────────────────────────
CONFIG="$REPO_ROOT/config.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "==> Generating config.yaml ..."
    cat > "$CONFIG" <<'YAML'
# Auto-generated by build_and_run.sh — edit as needed.

detection:
  confidence_threshold: 0.65
  min_face_size: 50
  # Coral Edge TPU model — comment out to force CPU-only mode
  coral_model_path: models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite
  # CPU DNN fallback model (OpenCV Caffe SSD)
  cpu_model_path: models/res10_300x300_ssd_iter_140000.caffemodel

embedding:
  # Place mobilefacenet.tflite in models/ for production-quality grouping
  model_path: models/mobilefacenet.tflite
  input_size: [112, 112]
  embedding_dim: 192

clustering:
  epsilon: 0.4
  min_samples: 2
  metric: cosine

storage:
  db_path: data/faces.db
  crops_dir: data/crops

scan:
  image_extensions: [.jpg, .jpeg, .png, .webp]
  worker_threads: 2
  thumbnail_size: [128, 128]
YAML
    echo "    [ok] config.yaml created"
else
    echo "==> config.yaml already exists — skipping generation"
fi

echo "==> Build complete. Launching application..."
exec python -m app.main "$@"
