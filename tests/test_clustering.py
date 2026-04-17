"""Unit tests for clustering module."""

from __future__ import annotations

import numpy as np
import pytest

from app.clustering.clusterer import (
    cluster_embeddings,
    compute_centroid,
    cosine_distance,
)


def _unit_vec(dim: int, index: int, noise: float = 0.0) -> np.ndarray:
    """Create a unit vector with most mass at *index*, optional noise."""
    v = np.zeros(dim, dtype=np.float32)
    v[index] = 1.0
    if noise > 0:
        v += np.random.default_rng(seed=index).normal(0, noise, dim).astype(np.float32)
    norm = np.linalg.norm(v)
    return v / norm


class TestCosineDistance:
    def test_identical_vectors(self):
        v = _unit_vec(64, 0)
        assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = _unit_vec(64, 0)
        b = _unit_vec(64, 1)
        assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors(self):
        v = _unit_vec(64, 0)
        assert cosine_distance(v, -v) == pytest.approx(2.0, abs=1e-6)


class TestClusterEmbeddings:
    def test_two_clear_clusters(self):
        """Two tight groups of embeddings should form two clusters."""
        dim = 32
        face_ids = list(range(6))
        embeddings = [
            _unit_vec(dim, 0, noise=0.05),
            _unit_vec(dim, 0, noise=0.05),
            _unit_vec(dim, 0, noise=0.05),
            _unit_vec(dim, 15, noise=0.05),
            _unit_vec(dim, 15, noise=0.05),
            _unit_vec(dim, 15, noise=0.05),
        ]

        result = cluster_embeddings(
            face_ids=face_ids,
            embeddings=embeddings,
            epsilon=0.3,
            min_samples=2,
        )

        assert len(result) == 6
        # All items in group A should share a label
        assert result[0] == result[1] == result[2]
        # All items in group B should share a different label
        assert result[3] == result[4] == result[5]
        assert result[0] != result[3]

    def test_empty_input(self):
        result = cluster_embeddings(face_ids=[], embeddings=[])
        assert result == {}

    def test_single_face_becomes_noise(self):
        """A single point always becomes noise (label -1) with min_samples≥2."""
        v = _unit_vec(32, 0)
        result = cluster_embeddings(
            face_ids=[99],
            embeddings=[v],
            min_samples=2,
        )
        assert result[99] == -1

    def test_same_pair_constraint_merges_clusters(self):
        """Two otherwise-separate clusters should merge when same_pairs forces it."""
        dim = 64
        face_ids = [1, 2, 3, 4]
        embeddings = [
            _unit_vec(dim, 0, noise=0.02),
            _unit_vec(dim, 0, noise=0.02),
            _unit_vec(dim, 32, noise=0.02),
            _unit_vec(dim, 32, noise=0.02),
        ]

        result_unconstrained = cluster_embeddings(
            face_ids=face_ids,
            embeddings=embeddings,
            epsilon=0.3,
            min_samples=2,
        )
        # Without constraint, two separate clusters
        assert result_unconstrained[1] != result_unconstrained[3]

        result_constrained = cluster_embeddings(
            face_ids=face_ids,
            embeddings=embeddings,
            epsilon=0.3,
            min_samples=2,
            same_pairs=[(1, 3)],
        )
        # With same-pair constraint, they should share a label
        assert result_constrained[1] == result_constrained[3]


class TestCentroid:
    def test_centroid_of_identical_vectors(self):
        v = _unit_vec(32, 5)
        centroid = compute_centroid([v, v, v])
        assert cosine_distance(centroid, v) == pytest.approx(0.0, abs=1e-5)

    def test_centroid_is_unit_length(self):
        vectors = [_unit_vec(32, i) for i in range(8)]
        centroid = compute_centroid(vectors)
        assert np.linalg.norm(centroid) == pytest.approx(1.0, abs=1e-5)
