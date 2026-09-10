"""Find the same object region in other images (#164).

Given the crop of an already tagged object, locate that same picture region in
other photos and propose the same object there.  Size differences are handled —
the driving case is one photo pasted into a collage at a fraction of its
original size.  Viewpoint changes (the same table shot from another angle) are
explicitly out of scope for now.

How a single comparison works:

1. ORB descriptors of the reference crop are matched against the target image's
   cached descriptors (Hamming brute force, Lowe ratio test).
2. ``cv2.estimateAffinePartial2D`` fits a *similarity* transform (translation,
   rotation, one uniform scale) under RANSAC.  Restricting the model this way is
   exactly the "first pass need not be smarter" requirement, and it makes the
   estimate far more stable than a full homography on few points.
3. The hit survives only if it has enough inliers, a high enough inlier ratio, a
   plausible scale and a bounding box that lands inside the target image.
4. The reference rectangle is projected through the transform, giving the
   proposed box in the target image.

Nothing here writes an object marking directly.  Hits land in
``object_match_suggestions`` for review; only an explicit accept creates an
:class:`~app.db.models.ObjectOccurrence`.  That accepted occurrence then becomes
another reference sample, which is how the matcher learns an object over time,
and a rejected suggestion is kept forever so the same wrong pairing is never
proposed again.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import ObjectMatchingConfig
from app.db.models import (
    OBJECT_GEOMETRY_BBOX,
    OBJECT_MATCH_ACCEPTED,
    OBJECT_MATCH_PENDING,
    OBJECT_MATCH_REJECTED,
    Image,
    ObjectMatchSuggestion,
    ObjectOccurrence,
    TaggedObject,
)
from app.services.object_feature_service import FeatureSet, ObjectFeatureService
from app.services.object_service import ObjectService

log = logging.getLogger(__name__)

# A proposal overlapping an existing marking of the same object this much is
# considered already tagged and is dropped.
_DUPLICATE_IOU = 0.3
# Two proposals for the same object in the same image overlapping this much are
# the same hit; only the better-scoring one is kept.
_MERGE_IOU = 0.4

Bbox = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MatchResult:
    """One geometrically verified hit of a reference crop in a target image."""

    bbox: Bbox
    score: float
    inliers: int
    inlier_ratio: float
    scale: float


@dataclass(frozen=True)
class SuggestionInfo:
    """Flattened suggestion row for the review dialog."""

    suggestion_id: int
    object_id: int
    object_name: str
    image_id: int
    image_path: Optional[str]
    bbox: Bbox
    score: float
    inliers: int
    scale: float
    status: str


@dataclass
class MatchStats:
    """Outcome of a search run."""

    run_id: str = ""
    objects_searched: int = 0
    images_scanned: int = 0
    images_indexed: int = 0
    suggestions_created: int = 0
    skipped_no_reference: int = 0
    cancelled: bool = False
    object_ids: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def bbox_iou(a: Bbox, b: Bbox) -> float:
    """Intersection over union of two ``(x, y, w, h)`` boxes."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# The matcher
# ---------------------------------------------------------------------------

def match_patch_in_image(
    patch: FeatureSet,
    target: FeatureSet,
    config: ObjectMatchingConfig,
) -> Optional[MatchResult]:
    """Locate *patch* inside *target*, or return ``None`` if it is not there.

    Pure function over two feature sets — no database, no I/O — so it is cheap
    to test and safe to call from a worker thread.
    """
    if patch is None or target is None:
        return None
    if not patch.usable or not target.usable:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        knn = matcher.knnMatch(patch.descriptors, target.descriptors, k=2)
    except cv2.error as exc:  # pragma: no cover - defensive
        log.debug("Object match: knnMatch failed (%s)", exc)
        return None

    ratio = float(config.ratio_test)
    src: List[Tuple[float, float]] = []
    dst: List[Tuple[float, float]] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        best, second = pair[0], pair[1]
        if second.distance <= 0 or best.distance < ratio * second.distance:
            src.append((float(patch.keypoints[best.queryIdx][0]),
                        float(patch.keypoints[best.queryIdx][1])))
            dst.append((float(target.keypoints[best.trainIdx][0]),
                        float(target.keypoints[best.trainIdx][1])))

    good = len(src)
    # estimateAffinePartial2D needs at least 3 correspondences to fit at all.
    if good < max(3, int(config.min_inliers)):
        return None

    src_arr = np.array(src, dtype=np.float32).reshape(-1, 1, 2)
    dst_arr = np.array(dst, dtype=np.float32).reshape(-1, 1, 2)
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        src_arr,
        dst_arr,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(config.ransac_reproj_threshold),
        maxIters=3000,
        confidence=0.995,
        refineIters=10,
    )
    if matrix is None or inlier_mask is None:
        return None

    inliers = int(inlier_mask.sum())
    if inliers < int(config.min_inliers):
        return None
    inlier_ratio = inliers / float(good)
    if inlier_ratio < float(config.min_inlier_ratio):
        return None

    # A partial affine is [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]].
    scale = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
    if not (float(config.min_scale) <= scale <= float(config.max_scale)):
        return None

    corners = np.array(
        [
            [0.0, 0.0],
            [float(patch.width), 0.0],
            [float(patch.width), float(patch.height)],
            [0.0, float(patch.height)],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    projected = cv2.transform(corners, matrix).reshape(-1, 2)

    x0 = float(projected[:, 0].min())
    y0 = float(projected[:, 1].min())
    x1 = float(projected[:, 0].max())
    y1 = float(projected[:, 1].max())

    # The projection must mostly land on the target image, otherwise the fit is
    # a coincidence rather than a real occurrence.
    cx0, cy0 = max(0.0, x0), max(0.0, y0)
    cx1, cy1 = min(float(target.width), x1), min(float(target.height), y1)
    if cx1 - cx0 < 4 or cy1 - cy0 < 4:
        return None
    projected_area = max(1.0, (x1 - x0) * (y1 - y0))
    if ((cx1 - cx0) * (cy1 - cy0)) / projected_area < 0.5:
        return None

    bbox = (
        int(round(cx0)),
        int(round(cy0)),
        int(round(cx1 - cx0)),
        int(round(cy1 - cy0)),
    )

    # Score blends raw evidence (how many points agreed) with its purity (what
    # fraction of candidate matches agreed).  Both matter: a handful of very
    # clean matches and a mass of noisy ones are both weak on their own.
    evidence = min(1.0, inliers / float(max(1, config.min_inliers) * 3))
    score = round(min(1.0, 0.5 * evidence + 0.5 * inlier_ratio), 4)
    if score < float(config.min_score):
        return None

    return MatchResult(
        bbox=bbox,
        score=score,
        inliers=inliers,
        inlier_ratio=round(inlier_ratio, 4),
        scale=round(scale, 4),
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ObjectMatchingService:
    """Runs object searches and owns the suggestion review queue.

    Mutating methods ``flush()`` but never commit; the caller owns the
    transaction via ``session_scope``.
    """

    def __init__(
        self, session: Session, config: Optional[ObjectMatchingConfig] = None
    ) -> None:
        self.session = session
        self.config = config or ObjectMatchingConfig()
        self.features = ObjectFeatureService(session, self.config)
        self.objects = ObjectService(session)

    # -- searching -------------------------------------------------------

    def matchable_object_ids(self) -> List[int]:
        """Objects that have at least one bbox reference sample."""
        stmt = (
            select(ObjectOccurrence.object_id)
            .where(
                ObjectOccurrence.geometry_type == OBJECT_GEOMETRY_BBOX,
                ObjectOccurrence.bbox_w.isnot(None),
            )
            .distinct()
        )
        return [int(v) for v in self.session.execute(stmt).scalars().all()]

    def find_object(
        self,
        object_id: int,
        image_ids: Optional[Sequence[int]] = None,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        run_id: Optional[str] = None,
    ) -> MatchStats:
        """Search *object_id* across *image_ids* (whole library when ``None``)."""
        stats = MatchStats(run_id=run_id or uuid.uuid4().hex[:16])
        self._search_one(object_id, image_ids, progress_cb, cancel_check, stats)
        return stats

    def find_all_objects(
        self,
        image_ids: Optional[Sequence[int]] = None,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> MatchStats:
        """Search every object that has a reference sample — the batch mode."""
        stats = MatchStats(run_id=uuid.uuid4().hex[:16])
        object_ids = self.matchable_object_ids()
        for index, object_id in enumerate(object_ids, start=1):
            if cancel_check and cancel_check():
                stats.cancelled = True
                break
            name = self._object_name(object_id)

            def scoped(done: int, total: int, _msg: str, _i=index, _n=name) -> None:
                if progress_cb:
                    progress_cb(_i, len(object_ids), f"{_n} ({done}/{total})")

            self._search_one(object_id, image_ids, scoped, cancel_check, stats)
        return stats

    def _search_one(
        self,
        object_id: int,
        image_ids: Optional[Sequence[int]],
        progress_cb: Optional[Callable[[int, int, str], None]],
        cancel_check: Optional[Callable[[], bool]],
        stats: MatchStats,
    ) -> None:
        stats.objects_searched += 1
        stats.object_ids.append(int(object_id))

        references = self.features.reference_occurrences(object_id)
        if not references:
            stats.skipped_no_reference += 1
            return
        patches = self.features.ensure_patch_features(references)
        if not patches:
            stats.skipped_no_reference += 1
            return

        targets = self._target_image_ids(object_id, references, image_ids)
        if not targets:
            return

        index_stats = self.features.ensure_image_features(
            targets,
            progress_cb=(
                (lambda d, t: progress_cb(d, t, "index")) if progress_cb else None
            ),
            cancel_check=cancel_check,
        )
        stats.images_indexed += index_stats.processed
        if index_stats.cancelled:
            stats.cancelled = True
            return

        existing = self._existing_boxes(object_id)
        total = len(targets)
        best_per_image: Dict[int, Tuple[MatchResult, int]] = {}

        for done, image_id in enumerate(targets, start=1):
            if cancel_check and cancel_check():
                stats.cancelled = True
                break
            stats.images_scanned += 1
            target_fs = self.features.load_image_features(image_id)
            if target_fs is None or not target_fs.usable:
                if progress_cb:
                    progress_cb(done, total, "match")
                continue

            for occ_id, patch_fs in patches.items():
                result = match_patch_in_image(patch_fs, target_fs, self.config)
                if result is None:
                    continue
                if any(
                    bbox_iou(result.bbox, box) >= _DUPLICATE_IOU
                    for box in existing.get(image_id, ())
                ):
                    continue
                current = best_per_image.get(image_id)
                if current is None or result.score > current[0].score:
                    best_per_image[image_id] = (result, occ_id)
            if progress_cb:
                progress_cb(done, total, "match")

        for image_id, (result, occ_id) in best_per_image.items():
            if self._store_suggestion(object_id, image_id, result, occ_id, stats.run_id):
                stats.suggestions_created += 1
        self.session.flush()

    def _target_image_ids(
        self,
        object_id: int,
        references: Sequence[ObjectOccurrence],
        image_ids: Optional[Sequence[int]],
    ) -> List[int]:
        """Images worth scanning: the scope minus everything already decided."""
        if image_ids is None:
            candidates = [
                int(v)
                for v in self.session.execute(select(Image.id).order_by(Image.id))
                .scalars()
                .all()
            ]
        else:
            candidates = [int(v) for v in image_ids]

        # Images holding a reference sample are already tagged with this object.
        skip = {int(occ.image_id) for occ in references}
        # A pairing the user already reviewed is never proposed again — that is
        # the negative half of the learning.
        decided = self.session.execute(
            select(ObjectMatchSuggestion.image_id).where(
                ObjectMatchSuggestion.object_id == object_id,
                ObjectMatchSuggestion.status.in_(
                    (OBJECT_MATCH_REJECTED, OBJECT_MATCH_ACCEPTED)
                ),
            )
        ).scalars()
        skip.update(int(v) for v in decided)
        return [i for i in candidates if i not in skip]

    def _existing_boxes(self, object_id: int) -> Dict[int, List[Bbox]]:
        """Already-marked boxes of *object_id*, keyed by image."""
        rows = self.session.execute(
            select(
                ObjectOccurrence.image_id,
                ObjectOccurrence.bbox_x,
                ObjectOccurrence.bbox_y,
                ObjectOccurrence.bbox_w,
                ObjectOccurrence.bbox_h,
            ).where(
                ObjectOccurrence.object_id == object_id,
                ObjectOccurrence.bbox_w.isnot(None),
            )
        ).all()
        out: Dict[int, List[Bbox]] = {}
        for image_id, x, y, w, h in rows:
            out.setdefault(int(image_id), []).append(
                (int(x or 0), int(y or 0), int(w or 0), int(h or 0))
            )
        return out

    def _store_suggestion(
        self,
        object_id: int,
        image_id: int,
        result: MatchResult,
        source_occurrence_id: int,
        run_id: str,
    ) -> bool:
        """Insert or refresh a pending suggestion; ``True`` if one was created."""
        pending = (
            self.session.execute(
                select(ObjectMatchSuggestion).where(
                    ObjectMatchSuggestion.object_id == object_id,
                    ObjectMatchSuggestion.image_id == image_id,
                    ObjectMatchSuggestion.status == OBJECT_MATCH_PENDING,
                )
            )
            .scalars()
            .all()
        )
        for row in pending:
            existing_box = (row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h)
            if bbox_iou(result.bbox, existing_box) >= _MERGE_IOU:
                if result.score > row.score:
                    row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h = result.bbox
                    row.score = result.score
                    row.inliers = result.inliers
                    row.scale = result.scale
                    row.source_occurrence_id = source_occurrence_id
                    row.run_id = run_id
                    row.created_at = datetime.utcnow()
                return False

        self.session.add(
            ObjectMatchSuggestion(
                object_id=object_id,
                image_id=image_id,
                bbox_x=result.bbox[0],
                bbox_y=result.bbox[1],
                bbox_w=result.bbox[2],
                bbox_h=result.bbox[3],
                score=result.score,
                inliers=result.inliers,
                scale=result.scale,
                source_occurrence_id=source_occurrence_id,
                status=OBJECT_MATCH_PENDING,
                run_id=run_id,
            )
        )
        return True

    # -- review queue ----------------------------------------------------

    def list_pending(
        self, object_id: Optional[int] = None, run_id: Optional[str] = None
    ) -> List[SuggestionInfo]:
        """Pending suggestions, best score first."""
        stmt = (
            select(
                ObjectMatchSuggestion,
                TaggedObject.name,
                Image.file_path,
            )
            .join(TaggedObject, TaggedObject.id == ObjectMatchSuggestion.object_id)
            .join(Image, Image.id == ObjectMatchSuggestion.image_id)
            .where(ObjectMatchSuggestion.status == OBJECT_MATCH_PENDING)
            .order_by(ObjectMatchSuggestion.score.desc(), ObjectMatchSuggestion.id)
        )
        if object_id is not None:
            stmt = stmt.where(ObjectMatchSuggestion.object_id == object_id)
        if run_id:
            stmt = stmt.where(ObjectMatchSuggestion.run_id == run_id)
        return [
            SuggestionInfo(
                suggestion_id=row.id,
                object_id=row.object_id,
                object_name=name,
                image_id=row.image_id,
                image_path=path,
                bbox=(row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h),
                score=row.score,
                inliers=row.inliers,
                scale=row.scale,
                status=row.status,
            )
            for row, name, path in self.session.execute(stmt).all()
        ]

    def pending_count(self, object_id: Optional[int] = None) -> int:
        """How many suggestions are waiting for review."""
        stmt = select(ObjectMatchSuggestion.id).where(
            ObjectMatchSuggestion.status == OBJECT_MATCH_PENDING
        )
        if object_id is not None:
            stmt = stmt.where(ObjectMatchSuggestion.object_id == object_id)
        return len(self.session.execute(stmt).scalars().all())

    def accept_suggestion(self, suggestion_id: int) -> ObjectOccurrence:
        """Turn a suggestion into a real object marking.

        The new occurrence is recorded as AI-sourced with the match score as its
        confidence, and immediately counts as a reference sample for the next
        search — this is where the model learns from a confirmation.
        """
        row = self._require_suggestion(suggestion_id)
        occurrence = self.objects.add_occurrence_bbox(
            object_id=row.object_id,
            image_id=row.image_id,
            x=row.bbox_x,
            y=row.bbox_y,
            w=row.bbox_w,
            h=row.bbox_h,
        )
        occurrence.detection_source = "ai"
        occurrence.confidence = float(row.score)
        row.status = OBJECT_MATCH_ACCEPTED
        row.created_occurrence_id = occurrence.id
        row.reviewed_at = datetime.utcnow()
        self.session.flush()
        return occurrence

    def reject_suggestion(self, suggestion_id: int) -> None:
        """Mark a suggestion wrong; it is never proposed for this pair again."""
        row = self._require_suggestion(suggestion_id)
        row.status = OBJECT_MATCH_REJECTED
        row.reviewed_at = datetime.utcnow()
        self.session.flush()

    def accept_above(self, score: float, object_id: Optional[int] = None) -> int:
        """Accept every pending suggestion scoring at or above *score*."""
        accepted = 0
        for info in self.list_pending(object_id=object_id):
            if info.score >= score:
                self.accept_suggestion(info.suggestion_id)
                accepted += 1
        return accepted

    def clear_pending(self, object_id: Optional[int] = None) -> int:
        """Discard pending suggestions without recording a judgement.

        Used when the user closes the review without deciding; unlike a reject
        this leaves no negative memory, so a later search can propose again.
        """
        stmt = select(ObjectMatchSuggestion).where(
            ObjectMatchSuggestion.status == OBJECT_MATCH_PENDING
        )
        if object_id is not None:
            stmt = stmt.where(ObjectMatchSuggestion.object_id == object_id)
        rows = list(self.session.execute(stmt).scalars().all())
        for row in rows:
            self.session.delete(row)
        self.session.flush()
        return len(rows)

    # -- internals -------------------------------------------------------

    def _require_suggestion(self, suggestion_id: int) -> ObjectMatchSuggestion:
        row = self.session.get(ObjectMatchSuggestion, suggestion_id)
        if row is None:
            raise ValueError(f"Object match suggestion {suggestion_id} not found")
        return row

    def _object_name(self, object_id: int) -> str:
        name = self.session.execute(
            select(TaggedObject.name).where(TaggedObject.id == object_id)
        ).scalar_one_or_none()
        return name or f"#{object_id}"
