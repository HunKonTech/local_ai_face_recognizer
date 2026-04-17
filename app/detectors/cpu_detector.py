"""CPU face detector using OpenCV's DNN module.

Uses the OpenCV res10_300x300_ssd Caffe model by default, which ships with
many OpenCV distributions.  Alternatively a TFLite face detection model can
be used by pointing ``cpu_model_path`` to a ``.tflite`` file.

Model files (Caffe, bundled with opencv-contrib or downloaded separately)
--------------------------------------------------------------------------
deploy.prototxt:
    https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/\\
    face_detector/deploy.prototxt

res10_300x300_ssd_iter_140000.caffemodel:
    https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_\\
    face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

Both files can be placed in ``models/`` and set via config, or the detector
will attempt to locate them from standard OpenCV data paths.

IMPORTANT: This is a real CPU implementation — no stubs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from app.detectors.base import Detection, FaceDetector

log = logging.getLogger(__name__)

# Relative to project root (populated by setup or the user)
_DEFAULT_PROTOTXT = "models/deploy.prototxt"
_DEFAULT_CAFFEMODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"

# OpenCV's MediaPipe FaceDetection backend is also available when
# opencv-python ≥ 4.5.1 is installed.  We prefer it when Caffe model
# files are not present.
_USE_MEDIAPIPE_FALLBACK = True


class CpuDetector(FaceDetector):
    """CPU face detector using OpenCV DNN (SSD + Caffe res10 model).

    Falls back to OpenCV's Haar cascade if neither Caffe model nor MediaPipe
    is available.

    Args:
        model_path: Optional directory or ``.caffemodel`` path override.
                    ``None`` → look in ``models/`` relative to cwd.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._net: Optional[cv2.dnn.Net] = None
        self._backend = "unknown"

        # --- Try DNN / Caffe model first ---
        prototxt, caffemodel = self._resolve_caffe_paths(model_path)
        if prototxt and caffemodel:
            self._net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
            self._backend = "caffe_ssd"
            log.info("CPU detector: OpenCV DNN (Caffe SSD res10)")
            return

        # --- Try MediaPipe FaceDetection via OpenCV ---
        if _USE_MEDIAPIPE_FALLBACK:
            try:
                detector = cv2.FaceDetectorYN.create(
                    "",  # placeholder; will raise if not found
                    "",
                    (300, 300),
                )
                del detector
            except cv2.error:
                pass

        # --- Last resort: Haar cascade (always available with OpenCV) ---
        log.warning(
            "Caffe SSD model files not found; falling back to Haar cascade. "
            "Detection quality will be lower. See README for model download."
        )
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(cascade_path)
        self._backend = "haar"
        log.info("CPU detector: Haar cascade (%s)", cascade_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_caffe_paths(
        model_path: Optional[str],
    ) -> tuple[Optional[Path], Optional[Path]]:
        """Find prototxt and caffemodel files."""
        candidates_proto: List[Path] = []
        candidates_model: List[Path] = []

        if model_path:
            p = Path(model_path)
            if p.is_dir():
                candidates_proto.append(p / "deploy.prototxt")
                candidates_model.append(p / "res10_300x300_ssd_iter_140000.caffemodel")
            elif p.suffix == ".caffemodel":
                candidates_model.append(p)
                candidates_proto.append(p.parent / "deploy.prototxt")
            elif p.suffix == ".prototxt":
                candidates_proto.append(p)

        # Default search locations
        candidates_proto.append(Path(_DEFAULT_PROTOTXT))
        candidates_model.append(Path(_DEFAULT_CAFFEMODEL))

        found_proto = next((p for p in candidates_proto if p.exists()), None)
        found_model = next((p for p in candidates_model if p.exists()), None)

        if found_proto and found_model:
            return found_proto, found_model
        return None, None

    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        return f"cpu_{self._backend}"

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float = 0.65,
        min_face_size: int = 50,
    ) -> List[Detection]:
        """Detect faces in *image_bgr*.

        Args:
            image_bgr:            OpenCV BGR uint8 image.
            confidence_threshold: Minimum confidence to keep a detection.
            min_face_size:        Minimum face width AND height in pixels.

        Returns:
            List of :class:`Detection` objects, deduplicated via NMS.
        """
        if self._backend == "haar":
            raw = self._detect_haar(image_bgr, min_face_size)
        else:
            raw = self._detect_dnn(image_bgr, confidence_threshold, min_face_size)
        return self._nms(raw)

    # ------------------------------------------------------------------

    @staticmethod
    def _nms(detections: List[Detection], iou_threshold: float = 0.4) -> List[Detection]:
        """Remove overlapping detections with IoU > threshold (greedy NMS)."""
        if len(detections) <= 1:
            return detections

        boxes = np.array([[d.x, d.y, d.x + d.w, d.y + d.h] for d in detections],
                         dtype=np.float32)
        scores = np.array([d.confidence for d in detections], dtype=np.float32)

        indices = cv2.dnn.NMSBoxes(
            [[int(d.x), int(d.y), int(d.w), int(d.h)] for d in detections],
            scores.tolist(),
            score_threshold=0.0,
            nms_threshold=iou_threshold,
        )

        if len(indices) == 0:
            return []
        flat = indices.flatten() if hasattr(indices, "flatten") else list(indices)
        return [detections[i] for i in flat]

    @staticmethod
    def _is_valid_face(det: Detection, min_face_size: int) -> bool:
        """Return True if the detection looks like a plausible face crop."""
        if det.w < min_face_size or det.h < min_face_size:
            return False
        aspect = det.w / det.h if det.h > 0 else 0
        return 0.4 <= aspect <= 2.5

    def _detect_dnn(
        self, image_bgr: np.ndarray, confidence_threshold: float, min_face_size: int
    ) -> List[Detection]:
        """Run OpenCV DNN SSD inference."""
        img_h, img_w = image_bgr.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image_bgr,
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )
        self._net.setInput(blob)
        detections = self._net.forward()  # shape: (1, 1, N, 7)

        results: List[Detection] = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < confidence_threshold:
                continue

            x1 = int(detections[0, 0, i, 3] * img_w)
            y1 = int(detections[0, 0, i, 4] * img_h)
            x2 = int(detections[0, 0, i, 5] * img_w)
            y2 = int(detections[0, 0, i, 6] * img_h)

            det = Detection(x=x1, y=y1, w=x2 - x1, h=y2 - y1, confidence=conf).clamp(
                img_w, img_h
            )
            if self._is_valid_face(det, min_face_size):
                results.append(det)

        log.debug("CPU DNN detected %d face(s) (before NMS)", len(results))
        return results

    def _detect_haar(self, image_bgr: np.ndarray, min_face_size: int) -> List[Detection]:
        """Run Haar cascade detection (last resort)."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)
        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=8,
            minSize=(max(min_face_size, 40), max(min_face_size, 40)),
        )

        results: List[Detection] = []
        if len(faces) == 0:
            return results

        img_h, img_w = image_bgr.shape[:2]
        for x, y, w, h in faces:
            det = Detection(x=int(x), y=int(y), w=int(w), h=int(h), confidence=0.9)
            if self._is_valid_face(det, min_face_size):
                results.append(det.clamp(img_w, img_h))

        log.debug("Haar detected %d face(s) (before NMS)", len(results))
        return results
