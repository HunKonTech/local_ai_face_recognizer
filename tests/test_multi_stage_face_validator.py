"""Unit tests for the multi-technology face-verification ensemble."""

from __future__ import annotations

import dataclasses
from typing import List

import numpy as np

from app.config import AppConfig
from app.detectors.base import Detection, FaceDetector
from app.services.multi_stage_face_validator import (
    CONFIRM,
    REJECT,
    VETO,
    MultiStageFaceValidator,
)


class _CropEchoDetector(FaceDetector):
    """Fake YuNet backend: re-finds (or not) a centered face in any crop."""

    def __init__(self, find: bool = True, confidence: float = 0.9) -> None:
        self._find = find
        self._confidence = confidence

    @property
    def backend_name(self) -> str:
        return "fake_yunet"

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float = 0.5,
        min_face_size: int = 50,
    ) -> List[Detection]:
        if not self._find or self._confidence < confidence_threshold:
            return []
        h, w = image_bgr.shape[:2]
        fw, fh = int(w * 0.5), int(h * 0.5)
        x, y = (w - fw) // 2, (h - fh) // 2
        return [Detection(x=x, y=y, w=fw, h=fh, confidence=self._confidence)]


def _img(w: int = 640, h: int = 480) -> np.ndarray:
    return np.full((h, w, 3), 128, np.uint8)


def _cfg(**overrides):
    """A detection config with the ensemble on and given field overrides."""
    base = AppConfig().detection
    fields = dict(
        multistage_enabled=True,
        multistage_min_confirmations=2,
        multistage_min_available=2,
        # The real OpenCV technologies are replaced by injected fakes in tests.
        multistage_use_caffe=False,
        multistage_use_haar=False,
        multistage_use_eyes=False,
        multistage_use_insightface=False,
    )
    fields.update(overrides)
    return dataclasses.replace(base, **fields)


def _validator(find_yunet: bool, fake_votes: List[str], **cfg_overrides):
    """Build a validator with a controllable YuNet + injected tech votes."""
    v = MultiStageFaceValidator(
        _cfg(**cfg_overrides), yunet_detector=_CropEchoDetector(find=find_yunet)
    )
    for i, vote in enumerate(fake_votes):
        v._techs.append((f"fake{i}", 1, lambda c, l, m, _v=vote: _v))
    return v


_DET = Detection(x=100, y=100, w=80, h=80, confidence=0.4)


# ── Fail-safe paths ──────────────────────────────────────────────────────────

def test_disabled_falls_back_to_single_verifier_keep():
    # multistage off → pure FaceVerifier behaviour (YuNet re-finds → keep).
    v = _validator(find_yunet=True, fake_votes=[], multistage_enabled=False)
    assert v.verify(_img(), _DET) is not None


def test_disabled_falls_back_to_single_verifier_drop():
    v = _validator(find_yunet=False, fake_votes=[], multistage_enabled=False)
    assert v.verify(_img(), _DET) is None


def test_too_few_technologies_falls_back_to_yunet():
    # Only YuNet available (no extra techs) but min_available=2 → legacy path.
    v = _validator(find_yunet=False, fake_votes=[], multistage_min_available=2)
    assert v.active_technologies() == ["yunet"]
    assert not v._multistage_live
    # Legacy verifier says "no face" → dropped, not rescued by a lone vote.
    assert v.verify(_img(), _DET) is None


# ── Voting ───────────────────────────────────────────────────────────────────

def test_quorum_confirms_keeps_face():
    # YuNet + one tech confirm → 2 confirmations ≥ 2 → keep (YuNet refined box).
    v = _validator(find_yunet=True, fake_votes=[CONFIRM])
    out = v.verify(_img(), _DET)
    assert out is not None


def test_single_confirmation_drops_face():
    # YuNet rejects, only one tech confirms → 1 < 2 → drop.
    v = _validator(find_yunet=False, fake_votes=[CONFIRM, REJECT])
    assert v.verify(_img(), _DET) is None


def test_quorum_without_yunet_keeps_original_box():
    # YuNet rejects but two independent techs confirm → keep the original det.
    v = _validator(find_yunet=False, fake_votes=[CONFIRM, CONFIRM])
    out = v.verify(_img(), _DET)
    assert out is not None
    assert (out.x, out.y, out.w, out.h) == (_DET.x, _DET.y, _DET.w, _DET.h)


def test_all_reject_drops_face():
    v = _validator(find_yunet=False, fake_votes=[REJECT, REJECT])
    assert v.verify(_img(), _DET) is None


def test_weighted_technology_keeps_face_alone():
    # A weight-2 technology (e.g. InsightFace) confirming alone reaches the
    # quorum and rescues a profile face YuNet and the weak techs miss.
    v = _validator(find_yunet=False, fake_votes=[])
    v._techs.append(("strong", 2, lambda c, l, m: CONFIRM))
    assert v.verify(_img(), _DET) is not None


def test_lone_yunet_still_dropped_with_weighted_tech_present():
    # The weighted tech must not rescue a hair/texture box it rejects: YuNet
    # confirms (weight 1), the strong tech rejects → 1 < 2 → dropped.
    v = _validator(find_yunet=True, fake_votes=[])
    v._techs.append(("strong", 2, lambda c, l, m: REJECT))
    assert v.verify(_img(), _DET) is None


def test_min_confirmations_three():
    v = _validator(
        find_yunet=True, fake_votes=[CONFIRM, REJECT], multistage_min_confirmations=3
    )
    # YuNet + 1 tech = 2 confirmations < 3 → drop.
    assert v.verify(_img(), _DET) is None
    v2 = _validator(
        find_yunet=True,
        fake_votes=[CONFIRM, CONFIRM],
        multistage_min_confirmations=3,
    )
    assert v2.verify(_img(), _DET) is not None


# ── Technology availability ──────────────────────────────────────────────────

def test_real_opencv_technologies_are_available():
    # Haar + eye cascades ship with OpenCV, so they must register without deps.
    v = MultiStageFaceValidator(
        _cfg(multistage_use_haar=True, multistage_use_eyes=True),
        yunet_detector=_CropEchoDetector(),
    )
    techs = v.active_technologies()
    assert "yunet" in techs
    assert "haar" in techs
    assert "eyes" in techs


def test_insightface_abstains_when_unavailable():
    # Opting in must never crash when insightface is not installed; the tech is
    # simply absent from the active list.
    v = MultiStageFaceValidator(
        _cfg(multistage_use_insightface=True),
        yunet_detector=_CropEchoDetector(),
    )
    assert "yunet" in v.active_technologies()


# ── Deep-detector quorum (Haar / eyes only add evidence) ─────────────────────

def _tech(v, name: str, weight: int, vote: str):
    v._techs.append((name, weight, lambda crop, local, min_face, _v=vote: _v))
    return v


def test_haar_and_eyes_cannot_keep_texture_box():
    # Carpet pattern: YuNet re-finds it, Haar + eyes fire on the texture, the
    # independent Caffe SSD does not.  The legacy 3-of-4 vote kept it.
    v = _validator(find_yunet=True, fake_votes=[])
    _tech(v, "caffe_ssd", 1, REJECT)
    _tech(v, "haar", 1, CONFIRM)
    _tech(v, "eyes", 1, CONFIRM)
    assert v.verify(_img(), _DET) is None


def test_two_deep_detectors_keep_face():
    v = _validator(find_yunet=True, fake_votes=[])
    _tech(v, "caffe_ssd", 1, CONFIRM)
    _tech(v, "haar", 1, REJECT)
    assert v.verify(_img(), _DET) is not None


def test_insightface_alone_meets_deep_quorum():
    # Profile face: only InsightFace (weight 2) sees it.
    v = _validator(find_yunet=False, fake_votes=[])
    _tech(v, "caffe_ssd", 1, REJECT)
    _tech(v, "insightface", 2, CONFIRM)
    assert v.verify(_img(), _DET) is not None


def test_weak_votes_count_when_no_second_deep_detector():
    # Only YuNet among the deep detectors → legacy vote (fail-safe).
    v = _validator(find_yunet=True, fake_votes=[])
    _tech(v, "haar", 1, CONFIRM)
    assert v.verify(_img(), _DET) is not None


def test_min_strong_weight_zero_restores_legacy_vote():
    v = _validator(find_yunet=True, fake_votes=[], multistage_min_strong_weight=0)
    _tech(v, "caffe_ssd", 1, REJECT)
    _tech(v, "haar", 1, CONFIRM)
    assert v.verify(_img(), _DET) is not None


# ── InsightFace veto ─────────────────────────────────────────────────────────

def test_insightface_veto_overrides_other_confirmations():
    v = _validator(find_yunet=True, fake_votes=[])
    _tech(v, "caffe_ssd", 1, CONFIRM)
    _tech(v, "insightface", 2, VETO)
    assert v.verify(_img(), _DET) is None


def test_insightface_plain_reject_is_not_a_veto():
    v = _validator(find_yunet=True, fake_votes=[])
    _tech(v, "caffe_ssd", 1, CONFIRM)
    _tech(v, "insightface", 2, REJECT)
    assert v.verify(_img(), _DET) is not None


class _ScoredInsightFace(FaceDetector):
    """Fake SCRFD: a centered face with a fixed score; records each call."""

    def __init__(self, score: float) -> None:
        self._score = score
        self.calls: list = []

    @property
    def backend_name(self) -> str:
        return "insightface_scrfd"

    def detect(self, image_bgr, confidence_threshold=0.5, min_face_size=0,
               *, allow_below_prepared=False):
        self.calls.append((confidence_threshold, allow_below_prepared))
        if self._score < confidence_threshold:
            return []
        h, w = image_bgr.shape[:2]
        fw, fh = int(w * 0.5), int(h * 0.5)
        return [Detection(x=(w - fw) // 2, y=(h - fh) // 2, w=fw, h=fh,
                          confidence=self._score)]


def _insightface_vote(score: float):
    fake = _ScoredInsightFace(score)
    v = MultiStageFaceValidator(
        _cfg(multistage_use_insightface=True),
        yunet_detector=_CropEchoDetector(),
        insightface_detector=fake,
    )
    name, weight, vote_fn = v._techs[-1]
    assert (name, weight) == ("insightface", 2)
    crop, local, min_face = v._prepare_crop(_img(), _DET)
    return vote_fn(crop, local, min_face), fake.calls


def test_insightface_vote_confirm_reject_veto_bands():
    assert _insightface_vote(0.9)[0] == CONFIRM
    assert _insightface_vote(0.4)[0] == REJECT   # between veto 0.30 and 0.55
    assert _insightface_vote(0.1)[0] == VETO


def test_insightface_vote_probes_below_prepared_threshold():
    _, calls = _insightface_vote(0.1)
    assert calls and all(c == (0.30, True) for c in calls)


def test_insightface_detector_lowers_scrfd_threshold_for_one_call():
    from app.detectors.insightface_detector import InsightFaceDetector

    class _DetModel:
        det_thresh = 0.55

    class _App:
        def __init__(self):
            self.det_model = _DetModel()
            self.seen: list = []

        def get(self, img, max_num=0):
            self.seen.append(self.det_model.det_thresh)
            return []

    det = object.__new__(InsightFaceDetector)
    det._app = _App()
    det.detect(_img(), confidence_threshold=0.3, allow_below_prepared=True)
    det.detect(_img(), confidence_threshold=0.3)
    assert det._app.seen == [0.3, 0.55]
    assert det._app.det_model.det_thresh == 0.55
