"""Local keypoint features for object matching (#164).

The problem this solves is narrow on purpose: find *the same image region*
again in another photo, even when it appears at a different size — the classic
case being one photo pasted into a collage at a quarter of its original size.

The tool for that is local keypoint matching, not a global embedding.  ORB's
image pyramid is natively scale invariant, and a RANSAC-fitted similarity
transform gives both the geometric proof that it really is the same region and
the exact bounding box in the target image.  A global descriptor could never do
the second part: it cannot locate a small patch inside a large picture.

This module owns extraction, (de)serialisation and the cached feature tables.
The matching itself lives in :mod:`app.services.object_matching_service`.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import ObjectMatchingConfig
from app.db.models import (
    OBJECT_GEOMETRY_BBOX,
    Image,
    ImageFeatures,
    ObjectOccurrence,
    ObjectPatchFeatures,
)
from app.tasks.resource_governor import get_resource_governor
from app.utils.image_utils import load_image_bgr_normalized

log = logging.getLogger(__name__)

# ORB descriptors are 32 bytes each.
_DESC_BYTES = 32
# Below this a crop carries too little texture for a trustworthy match.
MIN_USEFUL_FEATURES = 8
# Replicate-border margin added around a downscaled reference crop.  ORB
# refuses images smaller than its descriptor patch and ignores a border of that
# width, so without padding the small end of the query pyramid would yield
# nothing.  Padding keeps the descriptor geometry identical to the one used on
# full images, which is what makes the two comparable at all.
_PATCH_PAD = 48


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSet:
    """Extracted keypoints and descriptors in *original* image coordinates."""

    keypoints: np.ndarray  # float32 (n, 3): x, y, size
    descriptors: np.ndarray  # uint8 (n, 32)
    width: int
    height: int
    work_scale: float

    @property
    def count(self) -> int:
        return int(self.keypoints.shape[0])

    @property
    def usable(self) -> bool:
        return self.count >= MIN_USEFUL_FEATURES


@dataclass
class IndexStats:
    """Outcome of an indexing pass."""

    processed: int = 0
    reused: int = 0
    failed: int = 0
    cancelled: bool = False


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def params_hash(config: ObjectMatchingConfig, kind: str) -> str:
    """Fingerprint the extractor settings that affect stored descriptors.

    Cached rows carrying a different hash are stale: their descriptors are not
    comparable with freshly extracted ones, so they are simply recomputed.
    """
    n_features = config.max_patch_features if kind == "patch" else config.max_features
    parts = [
        "orb",
        kind,
        str(n_features),
        str(config.pyramid_levels),
        f"{config.scale_factor:.3f}",
        str(config.fast_threshold),
        str(config.max_work_edge),
    ]
    if kind == "patch":
        parts.append(",".join(f"{s:.3f}" for s in config.patch_query_scales))
    raw = "|".join(parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:32]


def pack_features(fs: FeatureSet) -> Tuple[bytes, bytes]:
    """Return ``(keypoints_blob, descriptors_blob)`` for storage."""
    return (
        np.ascontiguousarray(fs.keypoints, dtype=np.float32).tobytes(),
        np.ascontiguousarray(fs.descriptors, dtype=np.uint8).tobytes(),
    )


def unpack_features(
    kp_blob: Optional[bytes],
    desc_blob: Optional[bytes],
    width: int,
    height: int,
    work_scale: float = 1.0,
) -> Optional[FeatureSet]:
    """Rebuild a :class:`FeatureSet` from stored blobs, or ``None`` if empty."""
    if not kp_blob or not desc_blob:
        return None
    try:
        desc = np.frombuffer(desc_blob, dtype=np.uint8)
        if desc.size % _DESC_BYTES:
            return None
        desc = desc.reshape(-1, _DESC_BYTES)
        kp = np.frombuffer(kp_blob, dtype=np.float32)
        if kp.size % 3:
            return None
        kp = kp.reshape(-1, 3)
    except ValueError:
        return None
    if kp.shape[0] != desc.shape[0] or kp.shape[0] == 0:
        return None
    return FeatureSet(
        keypoints=kp.copy(),
        descriptors=desc.copy(),
        width=int(width),
        height=int(height),
        work_scale=float(work_scale) or 1.0,
    )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def build_extractor(config: ObjectMatchingConfig, kind: str = "image"):
    """Create the ORB extractor described by *config*."""
    n_features = config.max_patch_features if kind == "patch" else config.max_features
    return cv2.ORB_create(
        nfeatures=max(32, int(n_features)),
        scaleFactor=max(1.05, float(config.scale_factor)),
        nlevels=max(3, int(config.pyramid_levels)),
        fastThreshold=max(1, int(config.fast_threshold)),
    )


def extract_features(
    image_bgr: np.ndarray,
    config: ObjectMatchingConfig,
    kind: str = "image",
) -> Optional[FeatureSet]:
    """Extract ORB features, returning coordinates in original pixels.

    Large images are downscaled to ``max_work_edge`` first so a 40 MP scan does
    not dominate a run; the keypoint coordinates are scaled back up, so callers
    never need to know that happened.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None
    h, w = image_bgr.shape[:2]
    if h < 8 or w < 8:
        return None

    work = image_bgr
    scale = 1.0
    long_edge = max(h, w)
    limit = max(64, int(config.max_work_edge))
    if long_edge > limit:
        scale = limit / float(long_edge)
        work = cv2.resize(
            image_bgr,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
    try:
        keypoints, descriptors = build_extractor(config, kind).detectAndCompute(
            gray, None
        )
    except cv2.error as exc:  # pragma: no cover - defensive
        log.debug("ORB extraction failed: %s", exc)
        return None
    if descriptors is None or not keypoints:
        return None

    inv = 1.0 / scale
    kp = np.array(
        [(k.pt[0] * inv, k.pt[1] * inv, k.size * inv) for k in keypoints],
        dtype=np.float32,
    )
    return FeatureSet(
        keypoints=kp,
        descriptors=np.ascontiguousarray(descriptors, dtype=np.uint8),
        width=w,
        height=h,
        work_scale=scale,
    )


def extract_patch_features(
    crop_bgr: np.ndarray, config: ObjectMatchingConfig
) -> Optional[FeatureSet]:
    """Describe a reference crop at several sizes, in one coordinate frame.

    Describing the crop only at its own size is what breaks on a shrunk copy.
    ORB spends most of its keypoint budget on the finest pyramid level and very
    little on the coarse ones, so against a quarter-size copy — whose detail
    lives at *its* finest level — barely a handful of comparable descriptors
    survive.  Re-running the extractor on explicitly downscaled versions of the
    crop puts a full budget at each size.

    Every keypoint is mapped back to crop pixels, so the result is a single
    feature set: one brute-force match and one RANSAC fit still decide, and
    descriptors from the wrong size simply become outliers.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    height, width = crop_bgr.shape[:2]
    if height < 8 or width < 8:
        return None

    keypoint_blocks: List[np.ndarray] = []
    descriptor_blocks: List[np.ndarray] = []
    seen: set = set()

    for factor in config.patch_query_scales:
        factor = float(factor)
        if factor <= 0 or factor > 1.0:
            continue
        target_w = max(16, int(round(width * factor)))
        target_h = max(16, int(round(height * factor)))
        if (target_w, target_h) in seen:
            continue
        seen.add((target_w, target_h))

        if (target_w, target_h) == (width, height):
            work = crop_bgr
            pad = 0
        else:
            work = cv2.resize(
                crop_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
            # Without the border ORB would drop everything in a small crop.
            pad = _PATCH_PAD
            work = cv2.copyMakeBorder(
                work, pad, pad, pad, pad, cv2.BORDER_REPLICATE
            )

        fs = extract_features(work, config, kind="patch")
        if fs is None:
            continue

        kp = fs.keypoints.copy()
        if pad:
            kp[:, 0] -= pad
            kp[:, 1] -= pad
            inside = (
                (kp[:, 0] >= 0)
                & (kp[:, 0] < target_w)
                & (kp[:, 1] >= 0)
                & (kp[:, 1] < target_h)
            )
            kp = kp[inside]
            desc = fs.descriptors[inside]
        else:
            desc = fs.descriptors
        if kp.shape[0] == 0:
            continue

        back = width / float(target_w)
        kp[:, 0] *= back
        kp[:, 1] *= height / float(target_h)
        kp[:, 2] *= back
        keypoint_blocks.append(kp)
        descriptor_blocks.append(desc)

    if not keypoint_blocks:
        return None
    return FeatureSet(
        keypoints=np.ascontiguousarray(np.vstack(keypoint_blocks), dtype=np.float32),
        descriptors=np.ascontiguousarray(
            np.vstack(descriptor_blocks), dtype=np.uint8
        ),
        width=width,
        height=height,
        work_scale=1.0,
    )


def crop_bbox(
    image_bgr: np.ndarray, bbox: Tuple[int, int, int, int]
) -> Optional[np.ndarray]:
    """Return the *bbox* region of *image_bgr*, clamped to the image."""
    h, w = image_bgr.shape[:2]
    x, y, bw, bh = (int(v) for v in bbox)
    x0 = max(0, min(x, w - 1))
    y0 = max(0, min(y, h - 1))
    x1 = max(x0 + 1, min(x + bw, w))
    y1 = max(y0 + 1, min(y + bh, h))
    crop = image_bgr[y0:y1, x0:x1]
    return crop if crop.size else None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ObjectFeatureService:
    """Computes and caches keypoint features for images and object crops.

    Indexing is lazy: only the images actually in a search scope are ever
    processed, and the result is stored so the next search reuses it.  There is
    no up-front library-wide run to sit through.

    Mutating methods ``flush()`` but never commit — the caller owns the
    transaction via ``session_scope`` — matching the house style of
    :class:`~app.services.object_service.ObjectService`.
    """

    def __init__(
        self, session: Session, config: Optional[ObjectMatchingConfig] = None
    ) -> None:
        self.session = session
        self.config = config or ObjectMatchingConfig()

    # -- images ----------------------------------------------------------

    def load_image_features(self, image_id: int) -> Optional[FeatureSet]:
        """Return the cached, non-stale features of *image_id*."""
        row = self.session.get(ImageFeatures, image_id)
        if row is None or row.params_hash != params_hash(self.config, "image"):
            return None
        return unpack_features(
            row.keypoints, row.descriptors, row.img_w, row.img_h, row.work_scale
        )

    def pending_image_ids(self, image_ids: Sequence[int]) -> List[int]:
        """Which of *image_ids* still need extraction (missing or stale)."""
        if not image_ids:
            return []
        wanted = params_hash(self.config, "image")
        fresh = set(
            self.session.execute(
                select(ImageFeatures.image_id).where(
                    ImageFeatures.image_id.in_(list(image_ids)),
                    ImageFeatures.params_hash == wanted,
                )
            )
            .scalars()
            .all()
        )
        return [int(i) for i in image_ids if int(i) not in fresh]

    def ensure_image_features(
        self,
        image_ids: Sequence[int],
        progress_cb: Optional[Callable[[int, int], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> IndexStats:
        """Extract and store features for every image in *image_ids*.

        Decoding and extraction run on a worker pool whose size comes from the
        resource governor, so a busy machine gets fewer threads.  One unreadable
        image never aborts the batch.
        """
        stats = IndexStats()
        todo = self.pending_image_ids(image_ids)
        stats.reused = len(image_ids) - len(todo)
        if not todo:
            if progress_cb:
                progress_cb(len(image_ids), len(image_ids))
            return stats

        paths = self._resolve_paths(todo)
        workers = get_resource_governor().recommended_workers(
            max(1, int(self.config.max_workers))
        )
        total = len(todo)
        done = 0
        wanted = params_hash(self.config, "image")
        chunk_size = max(1, workers * 2)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, total, chunk_size):
                if cancel_check and cancel_check():
                    stats.cancelled = True
                    break
                chunk = todo[start : start + chunk_size]
                extracted = list(
                    pool.map(lambda iid: self._extract_for_path(paths.get(iid)), chunk)
                )
                for image_id, fs in zip(chunk, extracted):
                    done += 1
                    if fs is None:
                        stats.failed += 1
                        # Store an empty marker so a broken or texture-less file
                        # is not re-decoded on every single search.
                        self._store_image_row(image_id, None, wanted)
                        continue
                    self._store_image_row(image_id, fs, wanted)
                    stats.processed += 1
                self.session.flush()
                if progress_cb:
                    progress_cb(done, total)
        return stats

    # -- object reference patches ---------------------------------------

    def load_patch_features(self, occurrence_id: int) -> Optional[FeatureSet]:
        """Return the cached, non-stale features of one reference crop."""
        row = self.session.get(ObjectPatchFeatures, occurrence_id)
        if row is None or row.params_hash != params_hash(self.config, "patch"):
            return None
        return unpack_features(
            row.keypoints, row.descriptors, row.patch_w, row.patch_h, row.work_scale
        )

    def reference_occurrences(self, object_id: int) -> List[ObjectOccurrence]:
        """Bbox occurrences of *object_id* — the object's reference samples.

        Every accepted match adds one here, which is how the matcher gets better
        at an object the more often it is confirmed.
        """
        stmt = (
            select(ObjectOccurrence)
            .where(
                ObjectOccurrence.object_id == object_id,
                ObjectOccurrence.geometry_type == OBJECT_GEOMETRY_BBOX,
                ObjectOccurrence.bbox_w.isnot(None),
                ObjectOccurrence.bbox_h.isnot(None),
            )
            .order_by(ObjectOccurrence.id)
        )
        return list(self.session.execute(stmt).scalars().all())

    def ensure_patch_features(
        self, occurrences: Sequence[ObjectOccurrence]
    ) -> Dict[int, FeatureSet]:
        """Return usable features per reference occurrence, extracting and
        caching any that are missing or stale."""
        out: Dict[int, FeatureSet] = {}
        wanted = params_hash(self.config, "patch")
        missing: List[ObjectOccurrence] = []
        for occ in occurrences:
            cached = self.load_patch_features(occ.id)
            if cached is not None:
                if cached.usable:
                    out[occ.id] = cached
            else:
                missing.append(occ)
        if not missing:
            return out

        paths = self._resolve_paths([occ.image_id for occ in missing])
        # Group by image so a photo with several tagged objects is decoded once.
        by_image: Dict[int, List[ObjectOccurrence]] = {}
        for occ in missing:
            by_image.setdefault(occ.image_id, []).append(occ)

        for image_id, occs in by_image.items():
            path = paths.get(image_id)
            img = load_image_bgr_normalized(path) if path else None
            for occ in occs:
                fs = None
                if img is not None:
                    bbox = (
                        occ.bbox_x or 0,
                        occ.bbox_y or 0,
                        occ.bbox_w or 0,
                        occ.bbox_h or 0,
                    )
                    crop = crop_bbox(img, bbox)
                    if crop is not None:
                        fs = extract_patch_features(crop, self.config)
                self._store_patch_row(occ, fs, wanted)
                if fs is not None and fs.usable:
                    out[occ.id] = fs
        self.session.flush()
        return out

    def invalidate_patch(self, occurrence_id: int) -> None:
        """Drop the cached features of one occurrence (bbox edited or deleted)."""
        row = self.session.get(ObjectPatchFeatures, occurrence_id)
        if row is not None:
            self.session.delete(row)
            self.session.flush()

    # -- internals -------------------------------------------------------

    def _resolve_paths(self, image_ids: Sequence[int]) -> Dict[int, str]:
        """Map image id to a readable file path for the given ids."""
        if not image_ids:
            return {}
        rows = self.session.execute(
            select(Image.id, Image.file_path).where(Image.id.in_(list(image_ids)))
        ).all()
        return {int(r[0]): r[1] for r in rows if r[1]}

    def _extract_for_path(self, path: Optional[str]) -> Optional[FeatureSet]:
        if not path:
            return None
        try:
            img = load_image_bgr_normalized(path)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("Object features: cannot load %s (%s)", path, exc)
            return None
        if img is None:
            return None
        return extract_features(img, self.config, kind="image")

    def _store_image_row(
        self, image_id: int, fs: Optional[FeatureSet], hash_: str
    ) -> None:
        row = self.session.get(ImageFeatures, image_id)
        if row is None:
            row = ImageFeatures(image_id=image_id)
            self.session.add(row)
        if fs is None:
            row.keypoints = None
            row.descriptors = None
            row.n_features = 0
            row.work_scale = 1.0
        else:
            row.keypoints, row.descriptors = pack_features(fs)
            row.n_features = fs.count
            row.img_w = fs.width
            row.img_h = fs.height
            row.work_scale = fs.work_scale
        row.params_hash = hash_
        row.computed_at = datetime.utcnow()

    def _store_patch_row(
        self, occ: ObjectOccurrence, fs: Optional[FeatureSet], hash_: str
    ) -> None:
        row = self.session.get(ObjectPatchFeatures, occ.id)
        if row is None:
            row = ObjectPatchFeatures(occurrence_id=occ.id, object_id=occ.object_id)
            self.session.add(row)
        row.object_id = occ.object_id
        if fs is None:
            row.keypoints = None
            row.descriptors = None
            row.n_features = 0
            row.work_scale = 1.0
        else:
            row.keypoints, row.descriptors = pack_features(fs)
            row.n_features = fs.count
            row.patch_w = fs.width
            row.patch_h = fs.height
            row.work_scale = fs.work_scale
        row.params_hash = hash_
        row.computed_at = datetime.utcnow()
