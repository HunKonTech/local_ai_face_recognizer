"""Detector factory — Coral auto-detection and CPU fallback."""

from __future__ import annotations

import logging
import platform

from app.config import DetectionConfig
from app.detectors.base import FaceDetector

log = logging.getLogger(__name__)

_EDGETPU_LIB = (
    "libedgetpu.1.dylib" if platform.system() == "Darwin" else "libedgetpu.so.1"
)


def probe_coral() -> bool:
    """Return True if the EdgeTPU delegate loads successfully.

    Tries ai-edge-litert first, then pycoral.  Never raises.
    """
    # ── ai-edge-litert + libedgetpu ───────────────────────────────────────
    try:
        from ai_edge_litert.interpreter import load_delegate  # type: ignore[import]

        load_delegate(_EDGETPU_LIB)
        log.info("Coral probe: EdgeTPU delegate loaded via ai_edge_litert")
        return True
    except ImportError:
        log.debug("ai-edge-litert not installed")
    except Exception as exc:
        log.warning("Coral probe (ai_edge_litert): %s", exc)

    # ── pycoral (legacy) ──────────────────────────────────────────────────
    try:
        from pycoral.utils.edgetpu import list_edge_tpus  # type: ignore[import]

        devices = list_edge_tpus()
        if devices:
            log.info("Coral probe: Edge TPU(s) via pycoral: %s", devices)
            return True
        log.info("pycoral available but no Edge TPU devices found")
        return False
    except ImportError:
        log.debug("pycoral not installed")
    except Exception as exc:
        log.debug("Coral probe (pycoral): %s", exc)

    return False


def create_detector(config: DetectionConfig) -> FaceDetector:
    """Create the best available face detector."""
    if config.coral_model_path:
        if probe_coral():
            try:
                from app.detectors.coral_detector import CoralDetector

                detector = CoralDetector(model_path=config.coral_model_path)
                log.info("Using Coral Edge TPU detector (backend: %s)", detector.backend_name)
                return detector
            except FileNotFoundError as exc:
                log.warning("Coral model file missing: %s — falling back to CPU", exc)
            except ImportError as exc:
                log.warning("Coral backend unavailable: %s — falling back to CPU", exc)
            except Exception as exc:  # noqa: BLE001
                log.warning("Coral init failed: %s — falling back to CPU", exc)
        else:
            log.warning(
                "coral_model_path is set but EdgeTPU delegate could not load. "
                "Install ai-edge-litert and libedgetpu, then reconnect the device."
            )
    else:
        log.info(
            "detection.coral_model_path not set — using CPU detector. "
            "Set this in config.yaml to enable Coral acceleration."
        )

    from app.detectors.cpu_detector import CpuDetector

    detector = CpuDetector(model_path=config.cpu_model_path)
    log.info("Using CPU detector (backend: %s)", detector.backend_name)
    return detector
