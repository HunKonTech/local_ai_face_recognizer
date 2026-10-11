"""Tests for the deep engine accuracy plan (#136): phases 0–6.

0  measurement      — open-set evaluation, review-feedback precision
1  quick fixes      — session-grouped folds, k-NN gate, durable vetoes, fingerprint
2  meta-classifier  — calibrated P(correct) with a precision-derived threshold
3  learning/hygiene — hard negatives, unknown class, label noise, dedup
4  context          — lifetime veto, repeat-on-photo strictness, unknown-group vote
5  embeddings       — flip test-time augmentation + reliability score
6  active learning  — near-match suggestions, feedback tightening, self-training
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pytest

from app.config import DeepRecognitionConfig
from app.db.database import init_db, session_scope
from app.db.models import (
    AUTO_ASSIGN_STATUS_AUTO,
    AUTO_ASSIGN_STATUS_REVERTED,
    LABEL_REVIEW_KEPT,
    LABEL_REVIEW_SUSPICIOUS,
    AutoAssignment,
    Face,
    FaceCorrection,
    FaceVeto,
    Image,
    Person,
    TrainingRun,
)
from app.deep.classifier import DeepFaceClassifier, DeepPrediction, _precision_threshold
from app.deep.dataset import ProbeMeta, TrainingDataset, build_training_dataset
from app.deep.evaluation import evaluate_open_set, feedback_precision
from app.services.deep_recognition_service import (
    DeepRecognitionService,
    _Candidate,
)

DIM = 64


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


@dataclass
class StubConfig:
    """Small/fast hyper-parameters; unspecified fields use the defaults."""

    ensemble_size: int = 2
    hidden_layers: Tuple[int, ...] = (32, 16)
    max_iter: int = 300
    min_class_size: int = 4
    calibration_folds: int = 3
    min_persons_for_ensemble: int = 2
    min_examples_for_ensemble: int = 8


def _vec(axis: int, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = np.zeros(DIM, dtype=np.float32)
    v[axis] = 1.0
    v += rng.normal(0, noise, DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _mix(weights: dict, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = np.zeros(DIM, dtype=np.float32)
    for axis, w in weights.items():
        v[axis] = w
    v += rng.normal(0, noise, DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _dataset(persons: dict, groups: dict | None = None, kept: set | None = None) -> TrainingDataset:
    vectors, labels, face_ids, group_keys, kept_flags = [], [], [], [], []
    fid = 1
    for pid, vecs in persons.items():
        for i, v in enumerate(vecs):
            vectors.append(v)
            labels.append(pid)
            face_ids.append(fid)
            group_keys.append(groups[pid][i] if groups else f"f{fid}")
            kept_flags.append(bool(kept and fid in kept))
            fid += 1
    return TrainingDataset(
        embeddings=np.vstack(vectors).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        face_ids=face_ids,
        person_names={pid: f"Person {pid}" for pid in persons},
        group_keys=group_keys,
        kept_flags=kept_flags,
        manual_flags=[True] * len(labels),
        probes=[ProbeMeta() for _ in labels],
    )


def _fast_config(tmp_path, **overrides) -> DeepRecognitionConfig:
    cfg = DeepRecognitionConfig(
        model_dir=str(tmp_path / "deep_model"),
        ensemble_size=2,
        hidden_layers=(32, 16),
        max_iter=300,
        min_class_size=4,
        calibration_folds=2,
        skip_unchanged=True,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture()
def tmp_db(tmp_path):
    db_file = tmp_path / "test.db"
    init_db(db_file)
    return db_file


_IMG_COUNTER = [0]


def _add_image(session, path: str | None = None, photo_date: str | None = None) -> int:
    _IMG_COUNTER[0] += 1
    path = path or f"/img_{_IMG_COUNTER[0]}.jpg"
    image = Image(file_path=path, file_hash=path, file_mtime=0.0, photo_date=photo_date)
    session.add(image)
    session.flush()
    return image.id


def _add_person(session, name: str, *, auto=False, birth=None, death=None) -> int:
    person = Person(name=name, is_auto_named=auto, birth_date=birth, death_date=death)
    session.add(person)
    session.flush()
    return person.id


def _add_face(session, image_id, person_id, embedding, *, source="manual",
              bbox=(0, 0, 40, 40), confidence=0.95) -> int:
    bx, by, bw, bh = bbox
    face = Face(
        image_id=image_id, person_id=person_id,
        bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
        confidence=confidence, detector_backend="cpu", assignment_source=source,
    )
    face.set_embedding(embedding)
    session.add(face)
    session.flush()
    return face.id


def _seed(session, n=6, axes=(0, 30), names=("Nagy Anna", "Kovács Béla")) -> list:
    """Named persons, each face on its own photo (as in a real archive)."""
    ids = []
    for k, (axis, name) in enumerate(zip(axes, names)):
        pid = _add_person(session, name)
        ids.append(pid)
        for i in range(n):
            _add_face(session, _add_image(session), pid, _vec(axis, seed=100 * k + i))
    return ids


# ----------------------------------------------------------------------
# Phase 0 — measurement
# ----------------------------------------------------------------------


class TestEvaluation:
    def test_open_set_evaluation_reports_strangers_and_coverage(self):
        persons = {pid: [_vec(axis, seed=10 * pid + i) for i in range(6)]
                   for pid, axis in ((1, 0), (2, 15), (3, 30), (4, 45), (5, 60))}
        report = evaluate_open_set(
            _dataset(persons), StubConfig(), n_folds=2, holdout_fraction=0.2
        )
        data = report.to_dict()
        assert report.n_holdout_persons == 1
        assert report.n_folds == 2
        overall = data["overall"]
        # Clean, well separated people: most known faces found, none wrong …
        assert overall["coverage"] > 0.7
        assert overall["n_wrong"] == 0
        # … and the held-out person's faces are never given a name.
        assert overall["n_strangers"] > 0
        assert overall["false_accept_rate"] == 0.0
        assert "unknown" in data["by_size"]

    def test_feedback_precision_counts_reviews_but_not_near_rows(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            pid = _add_person(s, "A")
            run = TrainingRun(mode="rescan")
            s.add(run)
            s.flush()
            statuses = ["confirmed"] * 3 + ["reverted"]
            for status in statuses:
                fid = _add_face(s, img, pid, _vec(0))
                s.add(AutoAssignment(training_run_id=run.id, face_id=fid,
                                     person_id=pid, status=status))
            # An answered near-match suggestion is not an automatic assignment.
            fid = _add_face(s, img, pid, _vec(0))
            s.add(AutoAssignment(training_run_id=run.id, face_id=fid, person_id=pid,
                                 status="reverted",
                                 decision_json='{"origin": "near", "reason": "margin"}'))
        with session_scope() as s:
            precision, n = feedback_precision(s)
        assert n == 4
        assert precision == pytest.approx(0.75)


# ----------------------------------------------------------------------
# Phase 1 — grouped folds, k-NN gate, durable vetoes, fingerprint
# ----------------------------------------------------------------------


class TestQuickFixes:
    def test_photo_day_groups_faces(self, tmp_db):
        with session_scope() as s:
            pid = _add_person(s, "Anna")
            same_day = [_add_image(s, photo_date="2001:05:06 10:00:00") for _ in range(2)]
            other = _add_image(s, photo_date=None)
            for img in same_day + [other]:
                _add_face(s, img, pid, _vec(0, seed=img))
            ds = build_training_dataset(s, collect_unknown=False)
        groups = ds.groups()
        assert groups[0] == groups[1]
        assert groups[2] != groups[0]

    def test_folds_never_exceed_photo_sessions(self):
        persons = {1: [_vec(0, seed=i) for i in range(6)],
                   2: [_vec(30, seed=10 + i) for i in range(6)]}
        groups = {1: ["day1"] * 6, 2: ["day1"] * 6}
        clf = DeepFaceClassifier(StubConfig())
        assert clf._n_folds(_dataset(persons, groups)) == 1  # one session → no CV

    def test_fingerprint_tracks_embedding_values_and_embedder(self):
        persons = {1: [_vec(0, seed=1), _vec(0, seed=2)], 2: [_vec(9, seed=3)]}
        ds1 = _dataset(persons)
        ds2 = _dataset({1: [_vec(0, seed=1), _vec(0, seed=99)], 2: [_vec(9, seed=3)]})
        assert ds1.fingerprint() != ds2.fingerprint()  # same ids, new vectors
        ds3 = _dataset(persons)
        ds3.embedder_id = "arcface|aligned|tta"
        assert ds1.fingerprint() != ds3.fingerprint()

    def test_knn_gate_stops_a_mislabeled_face_attracting_lookalikes(self):
        """Person 2's face wrongly labeled as person 1 (and kept by the user):
        a probe nearest to that one face must not be named person 1."""
        p1 = [_vec(0, noise=0.02, seed=i) for i in range(6)]
        p2 = [_vec(30, noise=0.02, seed=20 + i) for i in range(6)]
        mislabeled = _mix({30: 1.0, 31: 0.25}, noise=0.01, seed=7)
        persons = {1: p1 + [mislabeled], 2: p2}
        cfg = StubConfig(min_persons_for_ensemble=99)  # prototype mode
        clf = DeepFaceClassifier(cfg)
        clf.train(_dataset(persons, kept={7}))  # face 7 = the mislabeled one
        probe = _mix({30: 1.0, 31: 0.26}, noise=0.01, seed=8)
        prediction = clf.predict(probe)
        assert prediction.person_id != 1
        assert prediction.reason in ("knn", "margin", "assigned")

    def test_revert_writes_durable_veto(self, tmp_db, tmp_path):
        """The veto must survive the partner face being moved elsewhere."""
        cfg = _fast_config(tmp_path)
        with session_scope() as s:
            anna, bela = _seed(s)
            cand = _add_face(s, _add_image(s), None, _vec(0, seed=999), source=None)
        with session_scope() as s:
            DeepRecognitionService(s, cfg).train_and_recognize()
        with session_scope() as s:
            row = s.query(AutoAssignment).filter_by(face_id=cand).one()
            DeepRecognitionService(s, cfg).revert_assignment(row.id)
        with session_scope() as s:
            assert s.query(FaceVeto).filter_by(face_id=cand, person_id=anna).count() == 1
            # The legacy FaceCorrection partner (first Anna face) moves to Béla.
            corr = s.query(FaceCorrection).filter_by(same_person=False).one()
            partner = corr.face_id_a if corr.face_id_b == cand else corr.face_id_b
            s.get(Face, partner).person_id = bela
        with session_scope() as s:
            result = DeepRecognitionService(s, cfg).train_and_recognize()
        assert result.recognition.n_rejected_correction == 1
        with session_scope() as s:
            assert s.get(Face, cand).person_id is None

    def test_legacy_corrections_are_backfilled_once(self, tmp_db):
        from sqlalchemy import text

        with session_scope() as s:
            pid = _add_person(s, "Anna")
            img = _add_image(s)
            labeled = _add_face(s, img, pid, _vec(0))
            unknown = _add_face(s, img, None, _vec(1), source=None)
            s.add(FaceCorrection(face_id_a=min(labeled, unknown),
                                 face_id_b=max(labeled, unknown), same_person=False))
            s.execute(text("DELETE FROM schema_flags WHERE name = 'face_vetoes_backfill'"))
        init_db(tmp_db)  # re-run migrations
        with session_scope() as s:
            vetoes = s.query(FaceVeto).all()
            assert [(v.face_id, v.person_id) for v in vetoes] == [(unknown, pid)]


# ----------------------------------------------------------------------
# Phase 2 — calibrated meta-classifier
# ----------------------------------------------------------------------


class TestMetaClassifier:
    def test_precision_threshold_picks_lowest_safe_cut(self):
        scores = np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5])
        labels = np.array([1, 1, 1, 1, 0, 1, 0])
        thr, precision, recall = _precision_threshold(scores, labels, np.ones(7), 0.99)
        assert thr == pytest.approx(0.8)
        assert precision == pytest.approx(1.0)
        assert recall == pytest.approx(0.8)
        assert _precision_threshold(scores, 1 - labels, np.ones(7), 0.99)[0] is None

    @staticmethod
    def _family(n_persons=6, n_faces=10, shared=0.8, noise=0.12, seed=0) -> dict:
        """Lookalike relatives: everyone shares one strong common direction."""
        rng = np.random.default_rng(seed)
        persons = {}
        for p in range(n_persons):
            center = np.zeros(DIM)
            center[0] = shared
            center[1 + p] = np.sqrt(1 - shared ** 2)
            vecs = []
            for _ in range(n_faces):
                v = center + rng.normal(0, noise, DIM)
                vecs.append((v / np.linalg.norm(v)).astype(np.float32))
            persons[p + 1] = vecs
        return persons

    def test_meta_model_is_fitted_when_lookalikes_slip_through(self):
        clf = DeepFaceClassifier(StubConfig())
        report = clf.train(_dataset(self._family()))
        assert report.mode == "ensemble"
        assert report.metrics["meta"] == "active"
        assert report.metrics["n_meta_neg"] >= 5
        assert clf.has_meta_model
        assert 0.5 <= clf.meta_threshold <= 0.97
        probe = self._family(seed=1)[3][0]
        prediction = clf.predict(probe)
        assert prediction.p_correct is not None
        assert prediction.score == pytest.approx(prediction.p_correct)
        _, info = clf.predict_debug(probe)
        assert "meta" in [g.name for g in info.gates]

    def test_meta_model_stays_off_when_gates_already_reject_impostors(self):
        """Well separated people: no impostor reaches the calibrated gate, so
        there is nothing to calibrate — the plain probability gate is used."""
        persons = {pid: [_vec(axis, noise=0.08, seed=10 * pid + i) for i in range(8)]
                   for pid, axis in ((1, 0), (2, 12), (3, 24), (4, 36), (5, 48))}
        clf = DeepFaceClassifier(StubConfig())
        report = clf.train(_dataset(persons))
        assert report.metrics["meta"] == "insufficient_data"
        assert not clf.has_meta_model
        assert clf.predict(_vec(24, noise=0.08, seed=999)).person_id == 3


# ----------------------------------------------------------------------
# Phase 3 — learning from mistakes, clean training data
# ----------------------------------------------------------------------


class TestHygieneAndNegatives:
    def _noisy_persons(self):
        p1 = [_vec(0, seed=i) for i in range(6)]
        p2 = [_vec(30, seed=20 + i) for i in range(6)]
        return {1: p1 + [_vec(30, seed=77)], 2: p2}  # face 7 is really person 2

    def test_mislabeled_face_is_flagged_and_excluded(self):
        clf = DeepFaceClassifier(StubConfig(min_persons_for_ensemble=99))
        report = clf.train(_dataset(self._noisy_persons()))
        assert report.suspicious_face_ids == [7]
        # Excluded from the prototypes: a person-2 face is not pulled to person 1.
        assert clf.predict(_vec(30, seed=78)).person_id == 2

    def test_kept_label_is_never_flagged(self):
        clf = DeepFaceClassifier(StubConfig(min_persons_for_ensemble=99))
        report = clf.train(_dataset(self._noisy_persons(), kept={7}))
        assert report.suspicious_face_ids == []

    def test_burst_duplicates_collapse(self):
        burst = [_vec(0, noise=0.001, seed=i) for i in range(5)]
        persons = {1: burst + [_vec(0, seed=50)], 2: [_vec(30, seed=60 + i) for i in range(4)]}
        clf = DeepFaceClassifier(StubConfig(min_persons_for_ensemble=99))
        report = clf.train(_dataset(persons))
        assert report.n_deduplicated == 4

    def test_unknown_class_is_learned(self):
        persons = {pid: [_vec(axis, seed=10 * pid + i) for i in range(6)]
                   for pid, axis in ((1, 0), (2, 15), (3, 30), (4, 45))}
        ds = _dataset(persons)
        ds.unknown_embeddings = np.vstack(
            [_vec(55 + (i % 8), noise=0.1, seed=500 + i) for i in range(20)]
        )
        ds.unknown_face_ids = list(range(1000, 1020))
        clf = DeepFaceClassifier(StubConfig())
        clf.train(ds)
        assert clf._state.has_unknown_class
        probs = clf.class_probabilities(_vec(0, seed=999))
        assert set(probs) == {1, 2, 3, 4}  # no "-1" leaks to callers
        assert clf.predict(_vec(0, seed=999)).person_id == 1

    def test_dataset_collects_hard_negatives_and_unknown_background(self, tmp_db):
        with session_scope() as s:
            anna, bela = _seed(s)
            wrong = _add_face(s, _add_image(s), None, _vec(0, seed=500), source=None)
            s.add(FaceVeto(face_id=wrong, person_id=anna, source="revert"))
            for i in range(3):
                _add_face(s, _add_image(s), None, _vec(50, seed=600 + i), source=None)
            ds = build_training_dataset(s)
        assert list(zip(ds.neg_face_ids, ds.neg_person_ids)) == [(wrong, anna)]
        assert ds.n_unknown == 3  # strangers only — the Anna-like face is not one

    def test_suspicious_label_review_flow(self, tmp_db, tmp_path):
        cfg = _fast_config(tmp_path)
        with session_scope() as s:
            anna, bela = _seed(s)
            wrong = _add_face(s, _add_image(s), anna, _vec(30, seed=777))
        with session_scope() as s:
            DeepRecognitionService(s, cfg).train()
        with session_scope() as s:
            assert s.get(Face, wrong).label_review_status == LABEL_REVIEW_SUSPICIOUS
            items = DeepRecognitionService(s, cfg).list_suspicious_labels()
            assert [i.face_id for i in items] == [wrong]
            assert items[0].lookalike_name == "Kovács Béla"
            DeepRecognitionService(s, cfg).remove_label(wrong)
        with session_scope() as s:
            face = s.get(Face, wrong)
            assert face.person_id is None and face.label_review_status is None
            assert s.query(FaceVeto).filter_by(face_id=wrong, person_id=anna).count() == 1

    def test_keep_label_restores_training(self, tmp_db, tmp_path):
        cfg = _fast_config(tmp_path)
        with session_scope() as s:
            anna, _ = _seed(s)
            wrong = _add_face(s, _add_image(s), anna, _vec(30, seed=777))
        with session_scope() as s:
            DeepRecognitionService(s, cfg).train()
        with session_scope() as s:
            DeepRecognitionService(s, cfg).keep_label(wrong)
        with session_scope() as s:
            DeepRecognitionService(s, cfg).train(force=True)
            assert s.get(Face, wrong).label_review_status == LABEL_REVIEW_KEPT
            assert DeepRecognitionService(s, cfg).count_suspicious_labels() == 0


# ----------------------------------------------------------------------
# Phase 4 — context
# ----------------------------------------------------------------------


class TestContext:
    def test_photo_before_birth_is_vetoed(self, tmp_db, tmp_path):
        cfg = _fast_config(tmp_path)
        with session_scope() as s:
            anna = _add_person(s, "Anna", birth="1990.03.01")
            bela = _add_person(s, "Béla")
            for i in range(6):
                _add_face(s, _add_image(s), anna, _vec(0, seed=i))
                _add_face(s, _add_image(s), bela, _vec(30, seed=50 + i))
            old_photo = _add_image(s, photo_date="1970")
            cand = _add_face(s, old_photo, None, _vec(0, seed=999), source=None)
            fresh = _add_face(s, _add_image(s, photo_date="2005"), None,
                              _vec(0, seed=998), source=None)
        with session_scope() as s:
            result = DeepRecognitionService(s, cfg).train_and_recognize()
        assert result.recognition.n_rejected_date == 1
        with session_scope() as s:
            assert s.get(Face, cand).person_id is None
            assert s.get(Face, fresh).person_id == anna

    def _lookalike_setup(self, s):
        """Anna and her lookalike sister Bea (prototypes 0.8-correlated)."""
        anna = _add_person(s, "Anna")
        bea = _add_person(s, "Bea")
        for i in range(6):
            _add_face(s, _add_image(s), anna, _mix({0: 1.0}, noise=0.01, seed=i))
            _add_face(s, _add_image(s), bea, _mix({0: 0.8, 1: 0.6}, noise=0.01, seed=40 + i))
        return anna, bea

    def test_second_appearance_on_photo_needs_stronger_evidence(self, tmp_db, tmp_path):
        cfg = _fast_config(tmp_path)
        theta = 0.19  # closer to Anna, but only by a modest similarity margin
        probe = {0: float(np.cos(theta)), 1: float(np.sin(theta))}
        with session_scope() as s:
            anna, _ = self._lookalike_setup(s)
            photo = _add_image(s)
            _add_face(s, photo, anna, _mix({0: 1.0}, noise=0.01, seed=90), bbox=(0, 0, 40, 40))
            repeat = _add_face(s, photo, None, _mix(probe, noise=0.005, seed=91),
                               source=None, bbox=(300, 300, 40, 40))
            alone = _add_face(s, _add_image(s), None, _mix(probe, noise=0.005, seed=92),
                              source=None)
        with session_scope() as s:
            result = DeepRecognitionService(s, cfg).train_and_recognize()
        assert result.recognition.n_rejected_repeat_on_image == 1
        with session_scope() as s:
            assert s.get(Face, repeat).person_id is None
            assert s.get(Face, alone).person_id == anna

    def test_unknown_group_vote_rescues_unsure_member(self, tmp_db, tmp_path):
        cfg = _fast_config(tmp_path)
        with session_scope() as s:
            group = _add_person(s, "Unknown 1", auto=True)
            faces = [_add_face(s, _add_image(s), group, _vec(0, seed=i), source="clustering")
                     for i in range(4)]
            svc = DeepRecognitionService(s, cfg)
            members = [s.get(Face, fid) for fid in faces]
            sure = DeepPrediction(7, "A", 0.9, 0.9, 0.9, 0.5, "assigned", candidate_id=7)
            unsure = DeepPrediction(None, "A", 0.6, 0.6, 0.7, 0.2, "below_threshold",
                                    candidate_id=7)
            decided = [_Candidate(m, sure) for m in members[:3]]
            decided.append(_Candidate(members[3], unsure))

            calls = []

            class FakeClassifier:
                def predict(self, embedding, probe=None, relief=0.0):
                    calls.append(relief)
                    return DeepPrediction(7, "A", 0.62, 0.6, 0.7, 0.2, "assigned",
                                          candidate_id=7)

            assert svc._cluster_vote(FakeClassifier(), decided) == 1
            assert calls == [cfg.cluster_vote_relief]
            assert decided[3].origin == "cluster_vote"
            assert decided[3].prediction.person_id == 7

    def test_mixed_unknown_group_never_votes(self, tmp_db, tmp_path):
        cfg = _fast_config(tmp_path)
        with session_scope() as s:
            group = _add_person(s, "Unknown 1", auto=True)
            members = [s.get(Face, _add_face(s, _add_image(s), group, _vec(0, seed=i),
                                             source="clustering")) for i in range(5)]
            svc = DeepRecognitionService(s, cfg)

            def pred(pid):
                return DeepPrediction(pid, "x", 0.9, 0.9, 0.9, 0.5, "assigned", candidate_id=pid)

            unsure = DeepPrediction(None, "A", 0.6, 0.6, 0.7, 0.2, "below_threshold",
                                    candidate_id=7)
            decided = [_Candidate(members[0], pred(7)), _Candidate(members[1], pred(7)),
                       _Candidate(members[2], pred(8)), _Candidate(members[3], pred(8)),
                       _Candidate(members[4], unsure)]
            assert svc._cluster_vote(SimpleNamespace(predict=None), decided) == 0


# ----------------------------------------------------------------------
# Phase 5 — flip test-time augmentation
# ----------------------------------------------------------------------


class _ColumnEmbedder:
    """Fake base embedder: the vector is the mean of each pixel column."""

    _backend = "fake"

    @property
    def embedding_dim(self) -> int:
        return 8

    def embed(self, img):
        v = img.astype(np.float32).mean(axis=(0, 2))
        return v / np.linalg.norm(v)

    def embed_batch(self, imgs):
        return [self.embed(i) for i in imgs]


class TestFlipTTA:
    def test_symmetric_face_is_fully_consistent(self):
        from app.embeddings.tta import FlipTTAEmbedder

        base = _ColumnEmbedder()
        tta = FlipTTAEmbedder(base)
        symmetric = np.tile(np.array([1, 2, 3, 4, 4, 3, 2, 1], np.uint8)[None, :, None], (8, 1, 3))
        vec, consistency = tta.embed_with_consistency(symmetric)
        assert consistency == pytest.approx(1.0, abs=1e-6)
        assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)

    def test_asymmetric_face_scores_lower_and_batches_match(self):
        from app.embeddings.tta import FlipTTAEmbedder

        tta = FlipTTAEmbedder(_ColumnEmbedder())
        lopsided = np.tile(np.array([9, 9, 9, 1, 1, 1, 1, 1], np.uint8)[None, :, None], (8, 1, 3))
        single = tta.embed_with_consistency(lopsided)
        batch = tta.embed_batch_with_consistency([lopsided, lopsided])
        assert single[1] < 0.9
        assert len(batch) == 2
        assert np.allclose(batch[0][0], single[0])

    def test_embedder_identity_tracks_tta(self):
        from app.config import AppConfig
        from app.services.embedding_service import embedder_identity

        cfg = AppConfig()
        before = embedder_identity(cfg)
        cfg.embedding.flip_tta = not cfg.embedding.flip_tta
        assert embedder_identity(cfg) != before


# ----------------------------------------------------------------------
# Phase 6 — active learning and feedback
# ----------------------------------------------------------------------


class TestActiveLearning:
    def _near_setup(self, s):
        anna, bea = TestContext()._lookalike_setup(s)
        # Exactly between the lookalikes → rejected on margin, but close.
        between = _add_face(s, _add_image(s), None,
                            _mix({0: 0.9, 1: 0.3}, noise=0.0, seed=1), source=None)
        return anna, bea, between

    def test_unsure_face_becomes_near_suggestion_and_confirm_assigns(self, tmp_db, tmp_path):
        cfg = _fast_config(tmp_path)
        with session_scope() as s:
            anna, bea, between = self._near_setup(s)
        with session_scope() as s:
            result = DeepRecognitionService(s, cfg).train_and_recognize()
        assert result.recognition.n_near_suggestions == 1
        with session_scope() as s:
            svc = DeepRecognitionService(s, cfg)
            near = svc.list_near_suggestions()
            assert [n.face_id for n in near] == [between]
            assert s.get(Face, between).person_id is None  # not assigned
            # Not an automatic assignment: invisible to the main review list.
            assert svc.list_auto_assignments() == []
            svc.confirm_assignment(near[0].assignment_id)
        with session_scope() as s:
            face = s.get(Face, between)
            assert face.person_id == near[0].person_id
            assert face.assignment_source == "deep_confirmed"

    def test_rejecting_near_suggestion_vetoes_without_moving(self, tmp_db, tmp_path):
        cfg = _fast_config(tmp_path)
        with session_scope() as s:
            _, _, between = self._near_setup(s)
        with session_scope() as s:
            DeepRecognitionService(s, cfg).train_and_recognize()
        with session_scope() as s:
            near = DeepRecognitionService(s, cfg).list_near_suggestions()[0]
            DeepRecognitionService(s, cfg).revert_assignment(near.assignment_id)
        with session_scope() as s:
            assert s.get(Face, between).person_id is None
            assert s.query(FaceVeto).filter_by(
                face_id=between, person_id=near.person_id
            ).count() == 1
            row = s.get(AutoAssignment, near.assignment_id)
            assert row.status == AUTO_ASSIGN_STATUS_REVERTED
        with session_scope() as s:
            precision, n = feedback_precision(s)
            assert n == 0  # a near answer is not an automatic-assignment review

    def test_poor_review_precision_tightens_thresholds(self, tmp_db, tmp_path):
        cfg = _fast_config(tmp_path, feedback_min_reviewed=5)
        with session_scope() as s:
            anna, _ = _seed(s)
            run = TrainingRun(mode="rescan")
            s.add(run)
            s.flush()
            for i in range(6):
                fid = _add_face(s, _add_image(s), None, _vec(5, seed=300 + i), source=None)
                s.add(AutoAssignment(training_run_id=run.id, face_id=fid, person_id=anna,
                                     status="reverted" if i < 3 else "confirmed"))
        with session_scope() as s:
            result = DeepRecognitionService(s, cfg).train_and_recognize()
        rec = result.recognition
        assert rec.measured_precision == pytest.approx(0.5)
        assert rec.strictness == pytest.approx(cfg.max_strictness)
        with session_scope() as s:
            run = s.get(TrainingRun, result.training_run_id)
            assert run.measured_precision == pytest.approx(0.5)
            assert run.n_reviewed == 6

    def test_self_training_round_runs(self, tmp_db, tmp_path):
        cfg = _fast_config(tmp_path, self_training_rounds=1, auto_training_min_confidence=0.5)
        with session_scope() as s:
            _seed(s)
            _add_face(s, _add_image(s), None, _vec(0, seed=999), source=None)
        with session_scope() as s:
            result = DeepRecognitionService(s, cfg).train_and_recognize()
        assert result.recognition.n_assigned >= 1
        with session_scope() as s:
            # Both rounds share one training run (one review list).
            assert s.query(TrainingRun).count() == 1
            assert s.query(AutoAssignment).filter_by(status=AUTO_ASSIGN_STATUS_AUTO).count() >= 1
