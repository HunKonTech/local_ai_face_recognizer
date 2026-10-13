"""Tests for the truncated (half) face filter."""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from app.config import AppConfig
from app.detectors.base import Detection, FaceDetector
from app.detectors.truncation import truncation_reject_reason
from app.services.detection_service import DetectionService

_W, _H = 640, 480


def _lms(dx: float = 0.0, dy: float = 0.0) -> List[List[float]]:
    """A plausible 5-point constellation for a 100 px face at (200, 150)."""
    pts = [[230, 190], [270, 190], [250, 215], [233, 235], [267, 235]]
    return [[x + dx, y + dy] for x, y in pts]


# ── Detection.clamp records the cut-off share ────────────────────────────────

def test_clamp_inside_image_records_no_truncation():
    det = Detection(x=10, y=10, w=100, h=100, confidence=0.9).clamp(_W, _H)
    assert det.truncated_fraction == 0.0


def test_clamp_records_half_box_outside_image():
    det = Detection(x=-50, y=100, w=100, h=100, confidence=0.9).clamp(_W, _H)
    assert det.w == 50
    assert det.truncated_fraction == pytest.approx(0.5)


def test_clamp_keeps_the_larger_recorded_fraction():
    first = Detection(x=-50, y=100, w=100, h=100, confidence=0.9).clamp(_W, _H)
    assert first.clamp(_W, _H).truncated_fraction == pytest.approx(0.5)


# ── truncation_reject_reason ─────────────────────────────────────────────────

def test_whole_face_is_kept():
    det = Detection(x=200, y=150, w=100, h=100, confidence=0.9, landmarks=_lms())
    assert truncation_reject_reason(det, _W, _H, 0.35) is None


def test_slightly_cut_face_is_kept():
    # Forehead cut by the top edge: 20 % outside, both eyes visible.
    det = Detection(x=200, y=130, w=100, h=100, confidence=0.9,
                    truncated_fraction=0.2, landmarks=_lms())
    assert truncation_reject_reason(det, _W, _H, 0.35) is None


def test_half_face_box_is_rejected():
    det = Detection(x=0, y=150, w=50, h=100, confidence=0.9, truncated_fraction=0.5)
    assert truncation_reject_reason(det, _W, _H, 0.35) == "box_outside_image"


def test_eye_outside_image_is_rejected():
    det = Detection(x=0, y=150, w=70, h=100, confidence=0.9,
                    truncated_fraction=0.3, landmarks=_lms(dx=-240))
    assert truncation_reject_reason(det, _W, _H, 0.35) == "eye_outside_image"


def test_landmark_less_detector_judged_by_box_only():
    det = Detection(x=0, y=150, w=80, h=100, confidence=0.9, truncated_fraction=0.2)
    assert truncation_reject_reason(det, _W, _H, 0.35) is None


# ── DetectionService wiring ──────────────────────────────────────────────────

class _EdgeDetector(FaceDetector):
    """Returns one whole face and one face sliced in half by the left edge."""

    @property
    def backend_name(self) -> str:
        return "fake_edge"

    def detect(self, image_bgr, confidence_threshold=0.5, min_face_size=50):
        h, w = image_bgr.shape[:2]
        return [
            Detection(x=200, y=150, w=100, h=100, confidence=0.9,
                      landmarks=_lms()).clamp(w, h),
            Detection(x=-50, y=150, w=100, h=100, confidence=0.9,
                      landmarks=_lms(dx=-250)).clamp(w, h),
        ]


def _service(tmp_path, **detection_overrides) -> DetectionService:
    cfg = AppConfig(base_dir=str(tmp_path))
    cfg.detection.verification_enabled = False
    cfg.detection.adaptive_escalation = False
    for key, value in detection_overrides.items():
        setattr(cfg.detection, key, value)
    return DetectionService(session=None, detector=_EdgeDetector(), config=cfg)


def test_detection_service_drops_half_face(tmp_path):
    img = np.zeros((_H, _W, 3), np.uint8)
    dets = _service(tmp_path)._run_detection(img)
    assert [(d.x, d.w) for d in dets] == [(200, 100)]


def test_detection_service_filter_can_be_disabled(tmp_path):
    img = np.zeros((_H, _W, 3), np.uint8)
    dets = _service(tmp_path, truncation_filter_enabled=False)._run_detection(img)
    assert len(dets) == 2
