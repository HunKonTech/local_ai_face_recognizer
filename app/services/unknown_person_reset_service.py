"""Reset automatically created Unknown identities for a fresh recognition pass."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.db.models import Face, Person

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnknownPersonResetOptions:
    """Configuration options for resetting Unknown identities."""

    delete_unknown_persons: bool = True
    """Delete auto-created 'Unknown N' persons."""

    delete_face_assignments: bool = True
    """Unassign faces from Unknown persons (delete person_id, etc.)."""

    delete_face_data: bool = False
    """Hard-delete the face rows themselves (box + embedding + crop) - dangerous!"""

    rebuild_clusters: bool = True
    """Re-cluster unassigned faces into new Unknown groups after reset."""


@dataclass(frozen=True)
class UnknownPersonResetResult:
    """Summary of an Unknown identity reset."""

    deleted_persons: int = 0
    unassigned_faces: int = 0
    deleted_faces: int = 0
    """Face rows hard-deleted because ``delete_face_data`` was set."""

    n_crops_removed: int = 0
    """Crop thumbnail files actually unlinked from disk."""

    persons_deleted: bool = False
    """Whether Unknown persons were deleted."""

    faces_unassigned: bool = False
    """Whether face assignments were removed."""

    clusters_rebuilt: bool = False
    """Whether clustering was queued to rebuild Unknown groups."""


class UnknownPersonResetService:
    """Remove auto-named identities, optionally taking their faces with them."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def reset(
        self, options: UnknownPersonResetOptions | None = None
    ) -> UnknownPersonResetResult:
        """Reset auto-named persons according to ``options``.

        By default embeddings and face boxes are preserved so the next pipeline
        run can attempt recognition again before rebuilding any remaining
        Unknown clusters.  When ``delete_face_data`` is set the face rows are
        hard-deleted instead, and only a fresh detection pass brings them back.

        Args:
            options: Configuration for which steps to perform. Defaults to all
                the non-destructive steps.
        """
        if options is None:
            options = UnknownPersonResetOptions()

        if not (
            options.delete_unknown_persons
            or options.delete_face_assignments
            or options.delete_face_data
        ):
            return UnknownPersonResetResult()

        person_ids = [
            person_id
            for (person_id,) in (
                self._session.query(Person.id)
                .filter(Person.is_auto_named == True)  # noqa: E712
                .filter(Person.is_protected == False)  # noqa: E712
                .all()
            )
        ]
        if not person_ids:
            return UnknownPersonResetResult()

        deleted_persons = 0
        unassigned_faces = 0
        deleted_faces = 0
        n_crops_removed = 0

        try:
            # Step 1: dispose of the faces.  This must happen *before* the
            # assignments are cleared - once person_id is NULL there is no way
            # left to tell which faces belonged to an Unknown person.
            if options.delete_face_data:
                # Lazy import avoids a service-layer import cycle.
                from app.services.identity_service import IdentityService

                faces = (
                    self._session.query(Face)
                    .filter(Face.person_id.in_(person_ids))
                    .all()
                )
                for face in faces:
                    if IdentityService._unlink_crop(face.crop_path):
                        n_crops_removed += 1
                    # Hard delete: cascades FaceCorrection (delete-orphan) and
                    # drops bbox / embedding / landmarks with the row.
                    self._session.delete(face)
                deleted_faces = len(faces)
                self._session.flush()
            elif options.delete_face_assignments:
                unassigned_faces = (
                    self._session.query(Face)
                    .filter(Face.person_id.in_(person_ids))
                    .update(
                        {
                            Face.person_id: None,
                            Face.assignment_source: None,
                            Face.assignment_confidence: None,
                            Face.assigned_at: None,
                        },
                        synchronize_session=False,
                    )
                )

            # Step 2: drop the now-empty Unknown person rows.
            if options.delete_unknown_persons:
                deleted_persons = (
                    self._session.query(Person)
                    .filter(Person.id.in_(person_ids))
                    .delete(synchronize_session=False)
                )

            self._session.commit()
        except Exception:
            self._session.rollback()
            log.exception("Unknown identity reset failed; rolled back")
            raise

        log.info(
            "Unknown identity reset: %d person(s) deleted, %d face(s) unassigned, "
            "%d face(s) deleted, %d crop file(s) unlinked",
            deleted_persons, unassigned_faces, deleted_faces, n_crops_removed,
        )
        return UnknownPersonResetResult(
            deleted_persons=deleted_persons,
            unassigned_faces=unassigned_faces,
            deleted_faces=deleted_faces,
            n_crops_removed=n_crops_removed,
            persons_deleted=deleted_persons > 0,
            faces_unassigned=unassigned_faces > 0,
            clusters_rebuilt=options.rebuild_clusters,
        )
