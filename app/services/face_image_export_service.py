"""Export the faces of selected images into separate image files (#175).

Image-centric counterpart of :meth:`ExportService.export_person_images`: instead
of collecting one person's stored 128×128 crop thumbnails, this re-crops the
faces out of the *original* photos at full quality and names each output file
from a user-supplied pattern (see :mod:`app.services.filename_pattern`).

Two crop geometries are offered:

``"original"``
    The bounding box plus a margin, cut from the original image with **no
    resizing** — best for portraits, keeps every available pixel.
``"square"``
    An aspect-preserving square crop scaled to a fixed edge length, so all
    exported files share one size.

Faces are processed grouped by image so each photo is decoded exactly once,
which keeps memory bounded on large databases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
from sqlalchemy.orm import Session, joinedload

from app.db.models import Face, Image
from app.db.query_utils import in_chunks
from app.services.face_date_service import FaceDateResolver
from app.services.filename_pattern import safe_pattern_filename
from app.utils.image_utils import load_image_bgr_normalized, save_image_bgr

log = logging.getLogger(__name__)

DEFAULT_PATTERN = "portré-#CSID#-#Vezetéknév# #Keresztnév#-#Dátum#.jpg"

MODE_ORIGINAL = "original"
MODE_SQUARE = "square"


@dataclass
class FaceImageExportOptions:
    """User choices for one export run.

    Attributes:
        pattern:         Filename pattern including the extension.
        target_dir:      Destination directory (created if absent).
        margin_percent:  Context added around the face box, per side, as a
                         percentage of the box size.
        mode:            :data:`MODE_ORIGINAL` or :data:`MODE_SQUARE`.
        square_size:     Edge length in pixels, used by :data:`MODE_SQUARE`.
        jpeg_quality:    JPEG encode quality (ignored for PNG output).
        include_unknown: Also export faces not assigned to a person.
        skip_excluded:   Skip faces flagged excluded or low quality.
        only_face_ids:   When set, restrict the export to these faces (used by
                         the "export just this face" menu entry).
    """

    pattern: str = DEFAULT_PATTERN
    target_dir: str = ""
    margin_percent: int = 30
    mode: str = MODE_ORIGINAL
    square_size: int = 512
    jpeg_quality: int = 92
    include_unknown: bool = True
    skip_excluded: bool = True
    only_face_ids: Optional[Tuple[int, ...]] = None

    @property
    def margin_frac(self) -> float:
        return max(0, int(self.margin_percent)) / 100.0


@dataclass
class FaceImageExportResult:
    """Outcome of an export run."""

    written: int = 0
    skipped: int = 0
    files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class FaceImageExportService:
    """Crops the faces of given images into separate files.

    Args:
        session: SQLAlchemy session.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._dates = FaceDateResolver()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count_faces(
        self,
        image_ids: Sequence[int],
        *,
        include_unknown: bool = True,
        skip_excluded: bool = True,
        only_face_ids: Optional[Sequence[int]] = None,
    ) -> int:
        """How many faces the given options would export (for the dialog header)."""
        return len(
            self._load_faces(
                image_ids,
                include_unknown=include_unknown,
                skip_excluded=skip_excluded,
                only_face_ids=only_face_ids,
            )
        )

    def preview_names(
        self,
        image_ids: Sequence[int],
        options: FaceImageExportOptions,
        limit: int = 3,
    ) -> List[str]:
        """Filenames the first *limit* faces would get — live dialog preview."""
        faces = self._load_faces(
            image_ids,
            include_unknown=options.include_unknown,
            skip_excluded=options.skip_excluded,
            only_face_ids=options.only_face_ids,
        )
        names: List[str] = []
        used: Dict[str, int] = {}
        for face, index in self._with_indices(faces):
            if len(names) >= limit:
                break
            names.append(self._unique_name(face, index, options, used))
        return names

    def export_faces_of_images(
        self,
        image_ids: Sequence[int],
        options: FaceImageExportOptions,
        ctx: Optional[object] = None,
    ) -> FaceImageExportResult:
        """Write one image file per face of *image_ids* into ``options.target_dir``.

        Args:
            image_ids: Images whose faces should be exported.
            options:   Naming and crop settings.
            ctx:       Optional task context for progress and pause/cancel.

        Returns:
            A :class:`FaceImageExportResult` with counts, written paths and any
            per-face error messages.
        """
        result = FaceImageExportResult()
        if not options.target_dir:
            raise ValueError("target_dir is required")

        dest = Path(options.target_dir)
        dest.mkdir(parents=True, exist_ok=True)

        faces = self._load_faces(
            image_ids,
            include_unknown=options.include_unknown,
            skip_excluded=options.skip_excluded,
            only_face_ids=options.only_face_ids,
        )
        if not faces:
            return result

        # Group by image so each photo is decoded exactly once.
        by_image: Dict[int, List[Tuple[Face, int]]] = {}
        for face, index in self._with_indices(faces):
            by_image.setdefault(face.image_id, []).append((face, index))

        used: Dict[str, int] = {}
        total = len(by_image)
        for done, (image_id, entries) in enumerate(sorted(by_image.items())):
            if ctx is not None:
                ctx.checkpoint()
                ctx.report(int(done * 100 / total), str(image_id))

            image = entries[0][0].image
            path = image.file_path if image is not None else None
            if not path or not Path(path).exists():
                result.skipped += len(entries)
                result.errors.append(f"image_id={image_id}: file not found")
                continue

            # The stored bboxes are in the EXIF-normalised frame (detection ran
            # on the normalised image), so the export must load it the same way.
            img_bgr = load_image_bgr_normalized(str(path))
            if img_bgr is None:
                result.skipped += len(entries)
                result.errors.append(f"image_id={image_id}: could not be loaded")
                continue

            for face, index in entries:
                try:
                    self._export_one(face, index, img_bgr, dest, options, used, result)
                except Exception as exc:  # noqa: BLE001 — one bad face must not stop the run
                    log.exception("Face export failed for face_id=%s", face.id)
                    result.skipped += 1
                    result.errors.append(f"face_id={face.id}: {exc}")

            del img_bgr  # release the decoded photo before the next one

        if ctx is not None:
            ctx.report(100, "")
        log.info(
            "Exported %d face image(s) (%d skipped) to %s",
            result.written, result.skipped, dest,
        )
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_faces(
        self,
        image_ids: Sequence[int],
        *,
        include_unknown: bool,
        skip_excluded: bool,
        only_face_ids: Optional[Sequence[int]] = None,
    ) -> List[Face]:
        """Faces of *image_ids* in (image_id, bbox_x) order, eager-loaded."""
        ids = [int(i) for i in dict.fromkeys(image_ids)]
        if not ids:
            return []
        faces: List[Face] = []
        for chunk in in_chunks(ids):
            query = (
                self._session.query(Face)
                .options(joinedload(Face.person), joinedload(Face.image))
                .filter(Face.image_id.in_(chunk))
            )
            faces.extend(query.all())

        if only_face_ids:
            wanted = {int(f) for f in only_face_ids}
            faces = [f for f in faces if f.id in wanted]

        if skip_excluded:
            faces = [
                f
                for f in faces
                if not getattr(f, "is_excluded", False)
                and not getattr(f, "is_low_quality", False)
            ]
        if not include_unknown:
            faces = [f for f in faces if f.person_id is not None]

        # Stable, human-meaningful order: by image, then left-to-right.
        faces.sort(key=lambda f: (f.image_id, f.bbox_x or 0, f.id))
        return faces

    @staticmethod
    def _with_indices(faces: Sequence[Face]) -> List[Tuple[Face, int]]:
        """Pair each face with its 1-based index within its own image."""
        counters: Dict[int, int] = {}
        out: List[Tuple[Face, int]] = []
        for face in faces:
            counters[face.image_id] = counters.get(face.image_id, 0) + 1
            out.append((face, counters[face.image_id]))
        return out

    def _token_values(self, face: Face, index: int) -> Dict[str, object]:
        """Build the substitution mapping for one face."""
        person = face.person
        image = face.image
        fuzzy = self._dates.for_image(image)
        source = Path(image.file_path).stem if image is not None and image.file_path else ""

        def _field(name: str) -> str:
            value = getattr(person, name, None) if person is not None else None
            return str(value).strip() if value else ""

        return {
            "family_code": _field("family_code"),
            "external_family_code": _field("external_family_code"),
            "name": _field("name"),
            "name_prefix": _field("name_prefix"),
            "last_name": _field("last_name"),
            "first_name": _field("first_name"),
            "nickname": _field("nickname"),
            "date": fuzzy.display if fuzzy.is_known else "",
            "year": str(fuzzy.earliest.year) if fuzzy.is_known and fuzzy.earliest else "",
            "source_name": source,
            "index": index,
            "face_id": face.id,
            "image_id": face.image_id,
        }

    def _unique_name(
        self,
        face: Face,
        index: int,
        options: FaceImageExportOptions,
        used: Dict[str, int],
    ) -> str:
        """Rendered filename, de-duplicated with a ``-2``, ``-3`` … suffix."""
        name = safe_pattern_filename(options.pattern, self._token_values(face, index))
        key = name.casefold()
        seen = used.get(key, 0)
        used[key] = seen + 1
        if seen == 0:
            return name
        stem, dot, suffix = name.rpartition(".")
        if dot:
            return f"{stem}-{seen + 1}.{suffix}"
        return f"{name}-{seen + 1}"

    def _export_one(
        self,
        face: Face,
        index: int,
        img_bgr,
        dest: Path,
        options: FaceImageExportOptions,
        used: Dict[str, int],
        result: FaceImageExportResult,
    ) -> None:
        crop = self._crop(face, img_bgr, options)
        if crop is None:
            result.skipped += 1
            result.errors.append(f"face_id={face.id}: empty crop")
            return

        out_path = dest / self._unique_name(face, index, options, used)
        params = []
        if out_path.suffix.lower() in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, int(options.jpeg_quality)]
        if not save_image_bgr(out_path, crop, params):
            result.skipped += 1
            result.errors.append(f"face_id={face.id}: could not be written")
            return

        result.written += 1
        result.files.append(str(out_path))

    @staticmethod
    def _crop(face: Face, img_bgr, options: FaceImageExportOptions):
        """Cut the face out of *img_bgr* according to *options*."""
        x = int(face.bbox_x or 0)
        y = int(face.bbox_y or 0)
        w = int(face.bbox_w or 0)
        h = int(face.bbox_h or 0)
        if w <= 0 or h <= 0:
            return None

        if options.mode == MODE_SQUARE:
            from app.embeddings.alignment import square_crop

            size = max(16, int(options.square_size))
            return square_crop(
                img_bgr, (x, y, w, h), (size, size), margin_frac=options.margin_frac
            )

        # Original resolution: box + margin, clamped to the image, no resize.
        img_h, img_w = img_bgr.shape[:2]
        mx = int(round(w * options.margin_frac))
        my = int(round(h * options.margin_frac))
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(img_w, x + w + mx)
        y2 = min(img_h, y + h + my)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img_bgr[y1:y2, x1:x2]
        return None if crop.size == 0 else crop.copy()


def image_ids_of_faces(session: Session, face_ids: Sequence[int]) -> List[int]:
    """Distinct image ids owning *face_ids* — used by the single-face menu entry."""
    ids: List[int] = []
    for chunk in in_chunks([int(f) for f in face_ids]):
        rows = session.query(Face.image_id).filter(Face.id.in_(chunk)).all()
        ids.extend(int(r[0]) for r in rows)
    return list(dict.fromkeys(ids))


def image_label(session: Session, image_id: int) -> str:
    """Filename of *image_id* for dialog headers, empty when unknown."""
    image = session.get(Image, image_id)
    if image is None or not image.file_path:
        return ""
    return Path(image.file_path).name
