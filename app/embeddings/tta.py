"""Flip test-time augmentation for any face embedder.

Embedding models are trained on (roughly) left/right-symmetric data, so the
embedding of a crop and of its mirror image describe the same identity with
independent noise.  Averaging the two is a well-known, free accuracy gain:
the result stays in the *same* vector space as plain embeddings (so old and
new vectors remain comparable), only less noisy.

The cosine agreement of the two halves is a reliability signal on its own: a
frontal, sharp face embeds almost identically when mirrored, while a profile,
blurred or occluded one does not.  It is reported per crop and stored on the
face (``Face.embedding_consistency``) for the deep engine's meta-classifier.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.embeddings.base import FaceEmbedder


def _merge(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return (a if na >= nb else b), None
    a, b = a / na, b / nb
    consistency = float(np.clip(a @ b, -1.0, 1.0))
    merged = a + b
    norm = float(np.linalg.norm(merged))
    return (merged / norm if norm > 1e-8 else a).astype(np.float32), consistency


class FlipTTAEmbedder(FaceEmbedder):
    """Wraps *base*: embedding = normalised mean of crop and mirrored crop."""

    def __init__(self, base: FaceEmbedder) -> None:
        self._base = base
        # Mirrors the backends' diagnostic attribute (pipeline log line).
        self._backend = f"{getattr(base, '_backend', '?')}+flip_tta"

    @property
    def base(self) -> FaceEmbedder:
        return self._base

    @property
    def embedding_dim(self) -> int:
        return self._base.embedding_dim

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        return self.embed_with_consistency(face_bgr)[0]

    def embed_with_consistency(
        self, face_bgr: np.ndarray
    ) -> Tuple[np.ndarray, Optional[float]]:
        return _merge(
            self._base.embed(face_bgr), self._base.embed(cv2.flip(face_bgr, 1))
        )

    def embed_batch(self, faces_bgr: List[np.ndarray]) -> List[np.ndarray]:
        return [vec for vec, _ in self.embed_batch_with_consistency(faces_bgr)]

    def embed_batch_with_consistency(
        self, faces_bgr: List[np.ndarray]
    ) -> List[Tuple[np.ndarray, Optional[float]]]:
        """One batched model call over the crops and their mirrors."""
        if not faces_bgr:
            return []
        flipped = [cv2.flip(img, 1) for img in faces_bgr]
        vectors = list(self._base.embed_batch(list(faces_bgr) + flipped))
        n = len(faces_bgr)
        return [_merge(vectors[i], vectors[n + i]) for i in range(n)]

    def __repr__(self) -> str:
        return f"<FlipTTAEmbedder base={self._base!r}>"
