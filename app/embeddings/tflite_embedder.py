"""TFLite-based face embedder (CPU).

Uses a MobileFaceNet TFLite model to produce 192-dimensional L2-normalised
face embeddings.  This runs entirely on CPU — Coral is NOT used here.

Why CPU for embeddings?
-----------------------
The Edge TPU excels at accelerating the first few layers of quantised
models, but the full embedding pipeline (detection → alignment → embedding)
benefits more from throughput on CPU at batch level.  More importantly,
no practical, freely-redistributable ArcFace / MobileFaceNet model compiled
for Edge TPU is currently available.  The CPU TFLite path is transparent,
easy to swap, and fast enough for batch processing.

Model download
--------------
MobileFaceNet TFLite (float32, ~2 MB):

    Option A — use the model from the insightface project:
        https://github.com/deepinsight/insightface/tree/master/model_zoo

    Option B — use the ONNX → TFLite conversion from:
        https://github.com/sirius-ai/MobileFaceNet_TF

    Option C — use any compatible 112×112 input ArcFace TFLite model.

Place the model file at ``models/mobilefacenet.tflite`` (or set
``embedding.model_path`` in config.yaml).

Fallback
--------
If the TFLite model file is missing, the embedder falls back to a
deterministic HOG + PCA stub so the rest of the pipeline can still be
exercised.  The stub is clearly documented and produces lower-quality
embeddings.  It is NOT suitable for production.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.embeddings.base import FaceEmbedder

log = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = "models/mobilefacenet.tflite"


class TFLiteEmbedder(FaceEmbedder):
    """Face embedder backed by a MobileFaceNet TFLite model.

    NOTE: Runs on CPU only.  No Edge TPU usage.

    Args:
        model_path:    Path to ``.tflite`` model file.
        embedding_dim: Expected output dimension (must match the model).
        input_size:    (width, height) expected by the model.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_dim: int = 192,
        input_size: tuple[int, int] = (112, 112),
    ) -> None:
        self._embedding_dim = embedding_dim
        self._input_w, self._input_h = input_size
        self._interpreter = None
        self._input_index: int = 0
        self._output_index: int = 0

        resolved = Path(model_path) if model_path else Path(_DEFAULT_MODEL_PATH)

        if resolved.exists():
            self._load_tflite(resolved)
        else:
            log.warning(
                "Embedding model not found at %s — using HOG stub fallback. "
                "Download a MobileFaceNet TFLite model and set "
                "embedding.model_path in config.yaml for production quality.",
                resolved,
            )
            self._backend = "hog_stub"

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_tflite(self, model_path: Path) -> None:
        """Load the TFLite model via ai-edge-litert, tflite-runtime, or tensorflow."""
        try:
            # New name since 2024: ai-edge-litert (replaces tflite-runtime)
            from ai_edge_litert.interpreter import Interpreter  # type: ignore[import]

            log.info("Embedding: using ai_edge_litert")
        except ImportError:
            try:
                import tflite_runtime.interpreter as tflite  # type: ignore[import]

                Interpreter = tflite.Interpreter
                log.info("Embedding: using tflite_runtime")
            except ImportError:
                try:
                    import tensorflow as tf  # type: ignore[import]

                    Interpreter = tf.lite.Interpreter
                    log.info("Embedding: using tensorflow.lite")
                except ImportError as exc:
                    raise ImportError(
                        "No TFLite backend found. Install ai-edge-litert: "
                        "pip install ai-edge-litert"
                    ) from exc

        self._interpreter = Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()

        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        self._input_index = input_details[0]["index"]
        self._output_index = output_details[0]["index"]

        # Validate expected input shape
        _, h, w, c = input_details[0]["shape"]
        if (int(w), int(h)) != (self._input_w, self._input_h):
            log.warning(
                "Model input size %dx%d differs from configured %dx%d — "
                "updating input size to match model.",
                w, h, self._input_w, self._input_h,
            )
            self._input_w, self._input_h = int(w), int(h)

        # Update embedding dim from model output
        out_shape = output_details[0]["shape"]
        if len(out_shape) >= 2:
            self._embedding_dim = int(out_shape[-1])

        self._backend = "tflite"
        log.info(
            "Embedding model loaded: %s  (input=%dx%d, dim=%d)",
            model_path.name, self._input_w, self._input_h, self._embedding_dim,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """Return an L2-normalised embedding for *face_bgr*.

        Args:
            face_bgr: BGR uint8 numpy array (any size).

        Returns:
            float32 numpy array of shape ``(embedding_dim,)``.
        """
        if self._interpreter is not None:
            return self._embed_tflite(face_bgr)
        return self._embed_hog_stub(face_bgr)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """Resize and normalise face crop for model input."""
        resized = cv2.resize(
            face_bgr, (self._input_w, self._input_h), interpolation=cv2.INTER_LINEAR
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalise to [-1, 1] (standard for MobileFaceNet / ArcFace models)
        normalised = (rgb.astype(np.float32) - 127.5) / 128.0
        return np.expand_dims(normalised, axis=0)  # (1, H, W, 3)

    def _embed_tflite(self, face_bgr: np.ndarray) -> np.ndarray:
        """Run TFLite inference."""
        inp = self._preprocess(face_bgr)
        self._interpreter.set_tensor(self._input_index, inp)
        self._interpreter.invoke()
        embedding = self._interpreter.get_tensor(self._output_index)[0]
        return self._l2_normalise(embedding.astype(np.float32))

    def _embed_hog_stub(self, face_bgr: np.ndarray) -> np.ndarray:
        """HOG-based stub embedding — low quality, for development only.

        WARNING: This produces embeddings that are NOT comparable to
        TFLite model outputs.  Only use when the real model is unavailable
        and you are testing pipeline plumbing, not face recognition quality.
        """
        resized = cv2.resize(face_bgr, (64, 64), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        hog = cv2.HOGDescriptor(
            _winSize=(64, 64),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )
        descriptor = hog.compute(gray).flatten()

        # Trim or pad to match configured embedding_dim
        if len(descriptor) >= self._embedding_dim:
            vec = descriptor[: self._embedding_dim]
        else:
            vec = np.pad(descriptor, (0, self._embedding_dim - len(descriptor)))

        return self._l2_normalise(vec.astype(np.float32))

    @staticmethod
    def _l2_normalise(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return vec
        return vec / norm
