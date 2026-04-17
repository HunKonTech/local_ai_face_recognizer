"""Application configuration.

All tuneable parameters live here.  Load from a YAML file at startup;
fall back to sensible defaults when no file is present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DetectionConfig:
    """Parameters for face detection."""

    # Minimum confidence score [0.0 – 1.0] to accept a detection
    confidence_threshold: float = 0.65

    # Minimum face size in pixels (width and height must both exceed this)
    min_face_size: int = 50

    # Path to the Edge TPU compiled face-detection model (.tflite).
    # Set to None to force CPU-only mode regardless of hardware.
    coral_model_path: Optional[str] = None

    # Path to the CPU TFLite / OpenCV DNN model used for fallback detection.
    # Default: OpenCV's bundled res10_300x300_ssd deploy.prototxt / caffemodel.
    cpu_model_path: Optional[str] = None

    # Input size expected by the CPU DNN model (width, height)
    cpu_model_input_size: tuple[int, int] = (300, 300)


@dataclass
class EmbeddingConfig:
    """Parameters for face embedding generation.

    NOTE: Embedding runs on CPU via a local TFLite model.
          Coral is NOT used for embeddings — only for detection.
    """

    # Path to MobileFaceNet or compatible embedding TFLite model.
    # Download instructions in README; set this to a local path.
    model_path: Optional[str] = None

    # Size to which face crops are resized before embedding
    input_size: tuple[int, int] = (112, 112)

    # Length of the embedding vector produced by the model.
    # MobileFaceNet: 192.  ArcFace variants: 512.
    embedding_dim: int = 192


@dataclass
class ClusteringConfig:
    """Parameters for DBSCAN face clustering."""

    # Maximum cosine distance between two faces in the same cluster.
    # Lower → stricter (more clusters).  Tune to your dataset.
    epsilon: float = 0.4

    # Minimum faces required to form a cluster core point.
    min_samples: int = 2

    # Distance metric passed to DBSCAN
    metric: str = "cosine"


@dataclass
class StorageConfig:
    """Paths for persistent data."""

    # Directory where face crop thumbnails are stored
    crops_dir: str = "data/crops"

    # SQLite database file
    db_path: str = "data/faces.db"


@dataclass
class ScanConfig:
    """Parameters controlling image discovery."""

    # File extensions treated as images (lowercase, including the dot)
    image_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"]
    )

    # Number of parallel worker threads for the processing pipeline
    worker_threads: int = 2

    # Size (width, height) of stored face crop thumbnails
    thumbnail_size: tuple[int, int] = (128, 128)


@dataclass
class AppConfig:
    """Top-level application configuration."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)

    # Base directory used to resolve relative paths in sub-configs.
    # Defaults to the current working directory.
    base_dir: str = field(default_factory=lambda: str(Path.cwd()))

    def resolve(self, relative: str) -> Path:
        """Return *relative* resolved against *base_dir*."""
        p = Path(relative)
        return p if p.is_absolute() else Path(self.base_dir) / p

    @property
    def db_path_resolved(self) -> Path:
        return self.resolve(self.storage.db_path)

    @property
    def crops_dir_resolved(self) -> Path:
        return self.resolve(self.storage.crops_dir)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from a YAML file, falling back to defaults.

    Args:
        config_path: Path to a YAML file.  ``None`` → pure defaults.

    Returns:
        Populated :class:`AppConfig`.
    """
    cfg = AppConfig()

    if config_path is None:
        # Auto-discover config.yaml next to the script / in cwd
        candidates = [
            Path("config.yaml"),
            Path("config.example.yaml"),
            Path(os.environ.get("FACE_LOCAL_CONFIG", "config.yaml")),
        ]
        for c in candidates:
            if c.exists():
                config_path = str(c)
                break

    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            raw: dict = yaml.safe_load(fh) or {}

        det = raw.get("detection", {})
        cfg.detection = DetectionConfig(
            confidence_threshold=det.get(
                "confidence_threshold", cfg.detection.confidence_threshold
            ),
            min_face_size=det.get("min_face_size", cfg.detection.min_face_size),
            coral_model_path=det.get("coral_model_path"),
            cpu_model_path=det.get("cpu_model_path"),
            cpu_model_input_size=tuple(
                det.get("cpu_model_input_size", list(cfg.detection.cpu_model_input_size))
            ),
        )

        emb = raw.get("embedding", {})
        cfg.embedding = EmbeddingConfig(
            model_path=emb.get("model_path"),
            input_size=tuple(emb.get("input_size", list(cfg.embedding.input_size))),
            embedding_dim=emb.get("embedding_dim", cfg.embedding.embedding_dim),
        )

        clu = raw.get("clustering", {})
        cfg.clustering = ClusteringConfig(
            epsilon=clu.get("epsilon", cfg.clustering.epsilon),
            min_samples=clu.get("min_samples", cfg.clustering.min_samples),
            metric=clu.get("metric", cfg.clustering.metric),
        )

        sto = raw.get("storage", {})
        cfg.storage = StorageConfig(
            crops_dir=sto.get("crops_dir", cfg.storage.crops_dir),
            db_path=sto.get("db_path", cfg.storage.db_path),
        )

        sc = raw.get("scan", {})
        cfg.scan = ScanConfig(
            image_extensions=sc.get(
                "image_extensions", cfg.scan.image_extensions
            ),
            worker_threads=sc.get("worker_threads", cfg.scan.worker_threads),
            thumbnail_size=tuple(
                sc.get("thumbnail_size", list(cfg.scan.thumbnail_size))
            ),
        )

        if "base_dir" in raw:
            cfg.base_dir = raw["base_dir"]

    return cfg
