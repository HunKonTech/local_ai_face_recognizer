"""Abstract face embedder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FaceEmbedder(ABC):
    """Converts a face crop (BGR image array) to a fixed-length embedding vector."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Length of the output embedding vector."""

    @abstractmethod
    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """Generate an L2-normalised embedding for *face_bgr*.

        Args:
            face_bgr: BGR uint8 numpy array of a face crop (any size —
                      implementations must resize internally).

        Returns:
            1-D float32 numpy array of length :attr:`embedding_dim`.
        """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} dim={self.embedding_dim}>"
