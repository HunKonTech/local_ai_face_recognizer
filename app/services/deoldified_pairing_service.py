"""Deoldified image pairing — filename parsing and DB lookup.

Naming convention:
    Original:   some_name.jpg
    Colorized:  some_name-deoldified (artistic).jpg
                some_name-deoldified (stable).jpg
                some_name-deoldified.jpg

The '-deoldified' token (case-insensitive) marks the separation point.
Everything from '-deoldified' onwards is stripped; the original extension
is re-appended.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.db.models import Face, Image

# Two boxes overlapping at least this much are treated as the same face when
# merging a pair.  The variants are pixel-aligned, so a real match scores far
# above this; the margin only absorbs manual bbox tweaks made on one side.
_FACE_MATCH_IOU = 0.5

_RE_DEOLDIFIED = re.compile(r"-deoldified.*$", re.IGNORECASE)
# Captures the text *after* the '-deoldified' token (e.g. " (artistic)"),
# used to label one colorized variant apart from its siblings.
_RE_VARIANT_LABEL = re.compile(r"-deoldified\s*(.*)$", re.IGNORECASE)


@dataclass(frozen=True)
class ComparisonMember:
    """One image in a comparison group (the B&W original or a colorized variant).

    ``label`` is the human-facing variant tag — empty for the B&W original,
    ``"(artistic)"`` / ``"(stable)"`` / ``"deoldified"`` for colorized images.
    ``file_path`` is the on-disk path (already resolved when possible).
    """

    image_id: int
    file_path: str
    label: str
    is_bw: bool


def extract_original_stem(stem: str) -> Optional[str]:
    """Strip the '-deoldified' suffix and everything after it from a stem.

    Returns the original stem, or None if '-deoldified' is not present.

    >>> extract_original_stem("photo-deoldified (artistic)")
    'photo'
    >>> extract_original_stem("photo-deoldified")
    'photo'
    >>> extract_original_stem("normal_photo") is None
    True
    """
    if not _RE_DEOLDIFIED.search(stem):
        return None
    result = _RE_DEOLDIFIED.sub("", stem)
    return result or None


def extract_original_filename(filename: str) -> Optional[str]:
    """Return the expected original filename for a deoldified image, or None.

    Examples:
        "photo-deoldified (artistic).JPG" → "photo.JPG"
        "photo-deoldified (stable).jpg"   → "photo.jpg"
        "photo-deoldified.jpg"            → "photo.jpg"
        "normal_photo.jpg"                → None
    """
    p = Path(filename)
    original_stem = extract_original_stem(p.stem)
    if original_stem is None:
        return None
    return original_stem + p.suffix


def extract_variant_label(stem: str) -> str:
    """Return the colorized-variant label from a stem, or '' if not deoldified.

    The label is the text following the '-deoldified' token, with surrounding
    whitespace trimmed.  A plain '-deoldified' suffix yields ``"deoldified"`` so
    every colorized member has a non-empty, distinguishable label.

    >>> extract_variant_label("photo-deoldified (artistic)")
    '(artistic)'
    >>> extract_variant_label("photo-deoldified (stable)")
    '(stable)'
    >>> extract_variant_label("photo-deoldified")
    'deoldified'
    >>> extract_variant_label("normal_photo")
    ''
    """
    m = _RE_VARIANT_LABEL.search(stem)
    if m is None:
        return ""
    return m.group(1).strip() or "deoldified"


def is_deoldified_path(path: str) -> bool:
    """Return True if the filename contains '-deoldified' (case-insensitive)."""
    return extract_original_stem(Path(path).stem) is not None


def _basename(path: str) -> str:
    """Return the final path component, handling both '/' and '\\' separators."""
    # PurePosix/Windows both store backslashes literally, so normalise first.
    return Path(path.replace("\\", "/")).name


def _like_escape(value: str) -> str:
    """Escape SQL-LIKE wildcards ('%', '_') and the escape char itself."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class DeoldifiedPairingService:
    """Finds paired images in the database for the deoldified feature.

    Pairing is purely filename-based and ignores the containing folder: a
    colorized image and its black-and-white original are considered the same
    photo whenever their filenames match (after stripping '-deoldified ...'),
    even if they live in different directories.
    """

    def __init__(self, session: "Session") -> None:
        self._session = session

    def find_original_for_deoldified(
        self, deoldified_image: "Image"
    ) -> Optional["Image"]:
        """Return the original Image record for a deoldified image, or None.

        Matches purely by filename across all folders. Tries the exact
        extension first, then lowercase/uppercase variants.
        """
        resolved_name = _basename(deoldified_image.file_path)
        if deoldified_image.relative_path:
            resolved_name = _basename(deoldified_image.relative_path)
        # Prefer the on-disk name when resolvable (portability), else file_path.
        from app.services.image_library_service import resolve_image_path

        resolved = resolve_image_path(deoldified_image)
        name = _basename(str(resolved)) if resolved else resolved_name

        original_name = extract_original_filename(name)
        if original_name is None:
            return None

        ext = Path(original_name).suffix
        stem_orig = Path(original_name).stem

        # Build candidate names: exact extension + case variants
        candidates: list[str] = [original_name]
        for alt_ext in (ext.lower(), ext.upper()):
            alt = stem_orig + alt_ext
            if alt not in candidates:
                candidates.append(alt)

        return self._find_by_basenames(candidates, exclude_id=deoldified_image.id)

    def find_deoldified_for_original(
        self, original_image: "Image"
    ) -> Optional["Image"]:
        """Return the first deoldified Image for an original image, or None.

        Convenience wrapper over :meth:`find_all_deoldified_for_original` that
        keeps the original single-pair callers working.
        """
        variants = self.find_all_deoldified_for_original(original_image)
        return variants[0] if variants else None

    def find_all_deoldified_for_original(
        self, original_image: "Image"
    ) -> List["Image"]:
        """Return every deoldified Image for an original, ordered by label.

        Matches any image — in any folder — whose filename is
        '{original_stem}-deoldified...' (case-insensitive).  A single B&W
        original can have several colorized siblings ('(artistic)', '(stable)',
        plain); all are returned, sorted by their variant label for stability.
        """
        import sqlalchemy as sa

        from app.db.models import Image
        from app.services.image_library_service import resolve_image_path

        resolved = resolve_image_path(original_image)
        name = (
            _basename(str(resolved))
            if resolved
            else _basename(
                original_image.relative_path or original_image.file_path
            )
        )
        stem = Path(name).stem

        needle = f"{stem}-deoldified".lower()
        safe = _like_escape(needle)

        # Pre-filter in SQL (substring), then verify the basename in Python so a
        # match like 'other/{stem}-deoldified.jpg' counts but 'x{stem}...' does not.
        rows = (
            self._session.query(Image)
            .filter(
                sa.or_(
                    sa.func.lower(Image.file_path).like(
                        f"%{safe}%", escape="\\"
                    ),
                    sa.func.lower(Image.relative_path).like(
                        f"%{safe}%", escape="\\"
                    ),
                )
            )
            .all()
        )
        matches: List["Image"] = []
        for row in rows:
            if row.id == original_image.id:
                continue
            row_stem = Path(_basename(row.file_path)).stem
            orig_stem = extract_original_stem(row_stem)
            if orig_stem is not None and orig_stem.lower() == stem.lower():
                matches.append(row)
        matches.sort(
            key=lambda r: extract_variant_label(Path(_basename(r.file_path)).stem)
        )
        return matches

    def get_comparison_group(self, image: "Image") -> List[ComparisonMember]:
        """Return the full comparison group for an image: B&W original + variants.

        Resolves the B&W original (``image`` itself when it is the original, or
        its parent when ``image`` is colorized), then lists every colorized
        sibling.  The B&W original is always first (``is_bw=True``), followed by
        the colorized variants in label order.  Returns an empty list when no
        group exists (no original found, or an original with no colorized
        variants), so callers can fall back to single-image behaviour.
        """
        from app.services.image_library_service import resolve_image_path

        if is_deoldified_path(image.file_path):
            original = self.find_original_for_deoldified(image)
            if original is None:
                return []
        else:
            original = image

        variants = self.find_all_deoldified_for_original(original)
        if not variants:
            return []

        def _member(img: "Image", *, is_bw: bool) -> Optional[ComparisonMember]:
            resolved = resolve_image_path(img)
            path = str(resolved) if resolved else img.file_path
            if not path or not Path(path).exists():
                log.debug(
                    "Skipping deoldified comparison member with missing file: %s",
                    path or img.file_path,
                )
                return None
            label = (
                "" if is_bw
                else extract_variant_label(Path(_basename(img.file_path)).stem)
            )
            return ComparisonMember(
                image_id=img.id, file_path=path, label=label, is_bw=is_bw
            )

        bw_member = _member(original, is_bw=True)
        if bw_member is None:
            return []
        variant_members = [
            m for m in (_member(v, is_bw=False) for v in variants) if m is not None
        ]
        if not variant_members:
            return []
        return [bw_member, *variant_members]

    def _find_by_basenames(
        self, candidates: list[str], *, exclude_id: Optional[int] = None
    ) -> Optional["Image"]:
        """Return the first Image whose basename matches any candidate, or None.

        Searches across all folders; matching is case-insensitive on the
        filename only. A SQL LIKE pre-filter narrows the scan, then the exact
        basename is verified in Python to avoid false suffix matches.
        """
        import sqlalchemy as sa

        from app.db.models import Image

        wanted = {c.lower() for c in candidates}
        clauses = []
        for c in candidates:
            safe = _like_escape(c.lower())
            clauses.append(
                sa.func.lower(Image.file_path).like(f"%{safe}", escape="\\")
            )
            clauses.append(
                sa.func.lower(Image.relative_path).like(f"%{safe}", escape="\\")
            )

        rows = self._session.query(Image).filter(sa.or_(*clauses)).all()
        for row in rows:
            if exclude_id is not None and row.id == exclude_id:
                continue
            if _basename(row.file_path).lower() in wanted:
                return row
        return None

    # ──────────────────────────────────────────────────────────────────
    # One-directional data sync between a pair
    # ──────────────────────────────────────────────────────────────────

    # Image-level metadata fields copied between paired images.  Person-group
    # memberships are intentionally absent: they live on Person, so copying a
    # Face's person_id carries them automatically.
    _META_FIELDS = (
        "photo_date",
        "note",
        "place_id",
        "image_latitude",
        "image_longitude",
        "exif_latitude",
        "exif_longitude",
    )

    @classmethod
    def canonical_image_id(cls, session: "Session", image_id: int) -> int:
        """Return the B&W original's id for a colorized image, else *image_id*.

        Object tags are stored on the original so both sides of a pair show the
        same markers.  Returns *image_id* unchanged when pairing is switched
        off, the image is not a colorized variant, or no original is found.
        """
        from app.app_settings import app_qsettings
        from app.db.models import Image

        if not app_qsettings().value("deoldified/auto_pair", False, type=bool):
            return image_id
        img = session.get(Image, image_id)
        if img is None or not is_deoldified_path(img.file_path):
            return image_id
        original = cls(session).find_original_for_deoldified(img)
        return original.id if original is not None else image_id

    @staticmethod
    def image_has_data(image: "Image") -> bool:
        """Return True if the image has any face or user-supplied metadata."""
        if image.faces:
            return True
        for field in DeoldifiedPairingService._META_FIELDS:
            value = getattr(image, field, None)
            if value not in (None, ""):
                return True
        return False

    def sync_pair_data(
        self,
        image_a: "Image",
        image_b: "Image",
        *,
        crops_dir: "Optional[Path]" = None,
        thumbnail_size: "Optional[tuple[int, int]]" = None,
        crop_mode: str = "legacy",
    ) -> Optional[dict]:
        """Merge annotations between a deoldified pair, adding only what is missing.

        The black-and-white original is the canonical side: faces flow from it
        into the colorized variant, matched by bounding-box overlap so nothing
        is duplicated and a face deleted on the original does not come back.
        An already assigned face on the target is never re-assigned; only an
        unassigned one is filled in from its counterpart.

        Faces are replicated with their person assignment, embedding and
        landmarks; crops are regenerated from the *target* image's own pixels
        when ``crops_dir``/``thumbnail_size`` are supplied.  Image-level
        metadata (date, note, place, GPS) is filled in **both** directions, but
        only into fields that are still empty.  Object occurrences recorded on
        the colorized side are moved onto the original, which keeps a tagged
        object from being counted twice for the same photo.

        Returns a summary dict, or None when nothing changed.
        """
        bw, color = self._split_sides(image_a, image_b)
        source, target = self._resolve_direction(image_a, image_b, bw, color)
        if source is None or target is None:
            return None

        faces_copied, faces_updated = self._merge_faces(
            source, target, crops_dir, thumbnail_size, crop_mode
        )
        meta_copied = self._copy_metadata(source, target)
        meta_copied += self._copy_metadata(target, source)
        objects_moved = (
            self._move_object_occurrences(color, bw)
            if bw is not None and color is not None
            else 0
        )
        self._session.flush()
        if not (faces_copied or faces_updated or meta_copied or objects_moved):
            return None
        return {
            "source_id": source.id,
            "target_id": target.id,
            "faces_copied": faces_copied,
            "faces_updated": faces_updated,
            "objects_moved": objects_moved,
            "metadata_fields": meta_copied,
        }

    @staticmethod
    def _split_sides(
        image_a: "Image", image_b: "Image"
    ) -> "tuple[Optional[Image], Optional[Image]]":
        """Return (black_and_white, colorized), or (None, None) if unclear."""
        a_color = is_deoldified_path(image_a.file_path)
        b_color = is_deoldified_path(image_b.file_path)
        if a_color == b_color:
            return None, None
        return (image_b, image_a) if a_color else (image_a, image_b)

    def _resolve_direction(
        self,
        image_a: "Image",
        image_b: "Image",
        bw: "Optional[Image]",
        color: "Optional[Image]",
    ) -> "tuple[Optional[Image], Optional[Image]]":
        """Return (source, target) for the face merge.

        The black-and-white original is canonical, so faces normally flow from
        it into the colorized variant.  The one exception bootstraps a pair
        whose faces were only ever detected on the colorized side: while the
        original still has none, they flow the other way.  Once the original
        holds faces it stays the source, so a face deleted there is not copied
        back from the variant.

        Falls back to the side that holds data when neither or both filenames
        carry the '-deoldified' token, so an oddly named pair still syncs.
        """
        if bw is not None and color is not None:
            if not bw.faces and color.faces:
                return color, bw
            return bw, color
        a_has = self.image_has_data(image_a)
        b_has = self.image_has_data(image_b)
        if a_has == b_has:
            return None, None
        return (image_a, image_b) if a_has else (image_b, image_a)

    @staticmethod
    def _bbox_iou(a: "Face", b: "Face") -> float:
        """Intersection-over-union of two face boxes; 0.0 when they miss."""
        ax2, ay2 = a.bbox_x + a.bbox_w, a.bbox_y + a.bbox_h
        bx2, by2 = b.bbox_x + b.bbox_w, b.bbox_y + b.bbox_h
        ix = min(ax2, bx2) - max(a.bbox_x, b.bbox_x)
        iy = min(ay2, by2) - max(a.bbox_y, b.bbox_y)
        if ix <= 0 or iy <= 0:
            return 0.0
        inter = float(ix * iy)
        union = float(a.bbox_w * a.bbox_h + b.bbox_w * b.bbox_h) - inter
        return inter / union if union > 0 else 0.0

    def _merge_faces(
        self,
        source: "Image",
        target: "Image",
        crops_dir: "Optional[Path]",
        thumbnail_size: "Optional[tuple[int, int]]",
        crop_mode: str,
    ) -> "tuple[int, int]":
        """Add missing source faces to target; return (copied, updated)."""
        from app.db.models import Face

        if not source.faces:
            return 0, 0

        existing = list(target.faces)
        new_faces: list[Face] = []
        updated = 0
        for f in source.faces:
            match = None
            best_iou = _FACE_MATCH_IOU
            for e in existing:
                iou = self._bbox_iou(f, e)
                if iou >= best_iou:
                    best_iou = iou
                    match = e
            if match is not None:
                # Same face already there — only fill an empty assignment.
                if match.person_id is None and f.person_id is not None:
                    match.person_id = f.person_id
                    match.assignment_source = f.assignment_source
                    match.assignment_confidence = f.assignment_confidence
                    match.assigned_at = f.assigned_at
                    updated += 1
                continue
            nf = Face(
                image_id=target.id,
                person_id=f.person_id,
                bbox_x=f.bbox_x,
                bbox_y=f.bbox_y,
                bbox_w=f.bbox_w,
                bbox_h=f.bbox_h,
                confidence=f.confidence,
                detector_backend=f.detector_backend,
                is_excluded=f.is_excluded,
                assignment_source=f.assignment_source,
                assignment_confidence=f.assignment_confidence,
                assigned_at=f.assigned_at,
                quality_score=f.quality_score,
                quality_reasons=f.quality_reasons,
                is_low_quality=f.is_low_quality,
            )
            emb = f.get_embedding()
            if emb is not None:
                nf.set_embedding(emb)
            lm = f.get_landmarks()
            if lm is not None:
                nf.set_landmarks(lm)
            self._session.add(nf)
            new_faces.append(nf)
            existing.append(nf)

        if not new_faces:
            return 0, updated

        self._session.flush()  # assign Face.id values

        # Regenerate crops from the target image's own (colorized/B&W) pixels.
        # The pair is pixel-aligned, so the source bbox is valid on the target.
        # Decoding only happens when a face was actually added, so opening an
        # already synced image stays cheap.
        if crops_dir is not None and thumbnail_size is not None:
            from app.services.face_crop_service import save_crop_for_face
            from app.services.image_library_service import resolve_image_path
            from app.utils.image_utils import load_image_bgr

            resolved = resolve_image_path(target)
            img_bgr = load_image_bgr(
                str(resolved) if resolved else target.file_path
            )
            if img_bgr is not None:
                for nf in new_faces:
                    save_crop_for_face(
                        nf,
                        crops_dir=crops_dir,
                        thumbnail_size=thumbnail_size,
                        img_bgr=img_bgr,
                        crop_mode=crop_mode,
                    )

        target.detection_done = source.detection_done
        target.embedding_done = source.embedding_done
        return len(new_faces), updated

    def _move_object_occurrences(
        self, from_image: "Image", to_image: "Image"
    ) -> int:
        """Re-home object occurrences onto the canonical image; return the count.

        Object tags live on the black-and-white original only, so both views of
        the pair show the same markers and the Objects tab counts one photo
        once.  A tag that would collide with an identical one already on the
        target (same object, same point) is dropped instead of moved.
        """
        from app.db.models import ObjectOccurrence

        rows = (
            self._session.query(ObjectOccurrence)
            .filter(ObjectOccurrence.image_id == from_image.id)
            .all()
        )
        if not rows:
            return 0
        existing = {
            (o.object_id, o.point_x, o.point_y)
            for o in self._session.query(ObjectOccurrence)
            .filter(ObjectOccurrence.image_id == to_image.id)
            .all()
        }
        moved = 0
        for occ in rows:
            key = (occ.object_id, occ.point_x, occ.point_y)
            if key in existing:
                self._session.delete(occ)
            else:
                occ.image_id = to_image.id
                existing.add(key)
            moved += 1
        log.info(
            "Moved %d object occurrence(s) from image %d to %d",
            moved, from_image.id, to_image.id,
        )
        return moved

    def _copy_metadata(self, source: "Image", target: "Image") -> list[str]:
        """Copy image-level metadata into empty target fields; return names."""
        copied: list[str] = []
        for field in self._META_FIELDS:
            src_val = getattr(source, field, None)
            if src_val in (None, ""):
                continue
            if getattr(target, field, None) in (None, ""):
                setattr(target, field, src_val)
                copied.append(field)
        return copied
