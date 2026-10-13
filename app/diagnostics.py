"""Runtime diagnostics for the face-local application.

Run from the command line::

    python -m app.diagnostics

Or call :func:`check_tflite_backend` programmatically (e.g. from the settings
dialog) to surface actionable errors before the pipeline starts.
"""

from __future__ import annotations

import logging
import platform
import sys
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)


class BackendResult(NamedTuple):
    name: str
    available: bool
    detail: str


class TFLiteDiagnostics(NamedTuple):
    python_version: str
    platform: str
    machine: str
    backends: list[BackendResult]
    active_backend: str | None
    model_exists: bool
    model_path: str
    interpreter_ok: bool
    interpreter_error: str | None

    @property
    def ok(self) -> bool:
        return self.active_backend is not None


def check_tflite_backend(model_path: str | None = None) -> TFLiteDiagnostics:
    """Check which TFLite backends are available and whether the model loads."""
    from app.paths import resource_path

    resolved_model = Path(model_path) if model_path else resource_path("models/mobilefacenet.tflite")

    backends: list[BackendResult] = []

    # --- ai_edge_litert ---
    try:
        from ai_edge_litert.interpreter import Interpreter as _I  # type: ignore[import]
        import ai_edge_litert as _ael
        backends.append(BackendResult("ai_edge_litert", True, getattr(_ael, "__version__", "unknown")))
        _active_cls = _I
        active_backend = "ai_edge_litert"
    except ImportError as e:
        backends.append(BackendResult("ai_edge_litert", False, str(e)))
        _active_cls = None
        active_backend = None

    # --- tflite_runtime ---
    try:
        import tflite_runtime.interpreter as _tflite  # type: ignore[import]
        import tflite_runtime as _tr
        backends.append(BackendResult("tflite_runtime", True, getattr(_tr, "__version__", "unknown")))
        if _active_cls is None:
            _active_cls = _tflite.Interpreter
            active_backend = "tflite_runtime"
    except ImportError as e:
        backends.append(BackendResult("tflite_runtime", False, str(e)))

    # --- tensorflow ---
    try:
        import tensorflow as _tf  # type: ignore[import]
        backends.append(BackendResult("tensorflow", True, getattr(_tf, "__version__", "unknown")))
        if _active_cls is None:
            _active_cls = _tf.lite.Interpreter
            active_backend = "tensorflow"
    except ImportError as e:
        backends.append(BackendResult("tensorflow", False, str(e)))

    # --- Model file and interpreter test ---
    model_exists = resolved_model.exists()
    interpreter_ok = False
    interpreter_error: str | None = None

    if model_exists and _active_cls is not None:
        try:
            interp = _active_cls(model_path=str(resolved_model))
            interp.allocate_tensors()
            interpreter_ok = True
        except Exception as exc:
            interpreter_error = str(exc)

    return TFLiteDiagnostics(
        python_version=sys.version,
        platform=sys.platform,
        machine=platform.machine(),
        backends=backends,
        active_backend=active_backend,
        model_exists=model_exists,
        model_path=str(resolved_model),
        interpreter_ok=interpreter_ok,
        interpreter_error=interpreter_error,
    )


def print_diagnostics(diag: TFLiteDiagnostics | None = None) -> None:
    """Print a formatted diagnostics report to stdout."""
    if diag is None:
        diag = check_tflite_backend()

    print("=" * 60)
    print("Face-Local — TFLite diagnostics")
    print("=" * 60)
    print(f"Python:   {diag.python_version}")
    print(f"Platform: {diag.platform} / {diag.machine}")
    print()
    print("TFLite backends:")
    for b in diag.backends:
        status = "OK " if b.available else "---"
        marker = " <-- active" if b.name == diag.active_backend else ""
        print(f"  [{status}] {b.name}: {b.detail}{marker}")
    print()
    print(f"Active backend: {diag.active_backend or 'NONE (no backend found!)'}")
    print(f"Model file:     {diag.model_path}")
    print(f"Model exists:   {'yes' if diag.model_exists else 'NO'}")
    if diag.model_exists and diag.active_backend:
        print(f"Interpreter:    {'OK' if diag.interpreter_ok else 'FAILED: ' + (diag.interpreter_error or '')}")
    print()

    if not diag.ok:
        print("ACTION REQUIRED: No TFLite backend installed.")
        print("  Run:  pip install ai-edge-litert")
        print("  Supports Python 3.9-3.12 on Linux x64/ARM64, macOS, Windows x64.")
        print("  Alternative: pip install tensorflow")
    elif not diag.model_exists:
        print("ACTION REQUIRED: TFLite model file not found.")
        print(f"  Expected at: {diag.model_path}")
        print("  Download mobilefacenet.tflite or run build_and_run.ps1/sh")
    elif not diag.interpreter_ok:
        print(f"ACTION REQUIRED: Interpreter failed: {diag.interpreter_error}")
    else:
        print("OK — TFLite backend and model are ready.")
    print("=" * 60)


# Haar cascades the false-positive gate uses (multi_stage_face_validator).
HAAR_CASCADES = (
    "haarcascade_frontalface_default.xml",
    "haarcascade_profileface.xml",
    "haarcascade_eye.xml",
    "haarcascade_eye_tree_eyeglasses.xml",
)


def check_detection_components() -> dict:
    """Report which detection / verification components this install has.

    PyInstaller silently drops data files and optional packages, which turned
    the false-positive gate into a two-technology vote in the Windows build.
    ``Face-Local --self-test out.json`` runs this inside the frozen app so the
    release build can fail loudly instead.  Never imports Qt or loads models.
    """
    import cv2

    from app.paths import is_frozen, resource_path

    report: dict = {
        "frozen": is_frozen(),
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
    }
    haar_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
    report["haar_cascades"] = {
        name: bool(haar_dir) and not cv2.CascadeClassifier(haar_dir + name).empty()
        for name in HAAR_CASCADES
    }
    report["models"] = {
        name: resource_path(f"models/{name}").exists()
        for name in (
            "face_detection_yunet_2023mar.onnx",
            "res10_300x300_ssd_iter_140000.caffemodel",
            "deploy.prototxt",
        )
    }
    try:
        import onnxruntime

        report["onnxruntime"] = onnxruntime.__version__
    except Exception as exc:  # noqa: BLE001
        report["onnxruntime"] = None
        report["onnxruntime_error"] = str(exc)
    try:
        import insightface
        from insightface.app import FaceAnalysis  # noqa: F401

        report["insightface"] = insightface.__version__
    except Exception as exc:  # noqa: BLE001
        report["insightface"] = None
        report["insightface_error"] = str(exc)
    report["insightface_detect"] = (
        _probe_insightface() if report["insightface"] is not None else None
    )
    report["ok"] = (
        all(report["haar_cascades"].values())
        and all(report["models"].values())
        and report["insightface"] is not None
        and report["insightface_detect"] in (None, "ok")
    )
    return report


def _probe_insightface() -> str | None:
    """Run SCRFD once when its model pack is already on disk (never downloads).

    ``None`` = skipped (pack not downloaded yet), ``"ok"`` or ``"error: …"``.
    """
    from app.config import DetectionConfig

    pack = Path.home() / ".insightface" / "models" / DetectionConfig().insightface_model_pack
    if not any(pack.glob("det_*.onnx")):
        return None
    try:
        import numpy as np

        from app.detectors.insightface_detector import InsightFaceDetector

        InsightFaceDetector().detect(np.zeros((160, 160, 3), np.uint8))
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc!r}"


def write_self_test(path: str) -> int:
    """Write :func:`check_detection_components` as JSON; 0 when all present."""
    import json

    try:
        report = check_detection_components()
    except Exception as exc:  # noqa: BLE001
        report = {"ok": False, "error": repr(exc)}
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    diag = check_tflite_backend()
    print_diagnostics(diag)
    sys.exit(0 if diag.ok and diag.interpreter_ok else 1)
