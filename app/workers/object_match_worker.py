"""Background worker for object matching searches (#164).

Comparing one object against a whole library decodes and matches hundreds of
images, so it never runs on the UI thread.  The work runs under the shared
:class:`~app.tasks.manager.TaskManager`, which means the status bar shows
progress, the Task Manager window can pause or cancel it, and the adaptive
resource governor throttles it when the machine is busy elsewhere.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from app.config import ObjectMatchingConfig
from app.db.database import session_scope
from app.services.object_matching_service import MatchStats, ObjectMatchingService

log = logging.getLogger(__name__)


class ObjectMatchWorker:
    """Runs an object search inside a :class:`~app.tasks.manager.BackgroundTask`.

    Deliberately a plain object rather than a ``QThread``: the TaskManager owns
    the thread, and ``run_in_task`` is the single entry point it calls — the
    same shape as :class:`~app.workers.deep_pipeline_worker.DeepPipelineWorker`.

    Args:
        object_id: The object to search for; ``None`` searches every object that
            has a reference marking (the batch mode).
        image_ids: Restrict the search to these images; ``None`` is the whole
            library.
        config: Matching parameters; defaults are used when omitted.
    """

    def __init__(
        self,
        object_id: Optional[int] = None,
        image_ids: Optional[Sequence[int]] = None,
        config: Optional[ObjectMatchingConfig] = None,
    ) -> None:
        self._object_id = object_id
        self._image_ids: Optional[List[int]] = (
            [int(i) for i in image_ids] if image_ids is not None else None
        )
        self._config = config

    def run_in_task(self, ctx) -> MatchStats:
        """TaskManager entry point; returns the run's :class:`MatchStats`."""
        with session_scope() as session:
            service = ObjectMatchingService(session, self._config)

            def progress(done: int, total: int, stage: str) -> None:
                # checkpoint() is what makes cancel responsive and lets the
                # governor slow us down while other work needs the machine.
                ctx.checkpoint()
                percent = int(done * 100 / total) if total else 0
                ctx.report(min(99, percent), f"{stage} {done}/{total}")

            def cancelled() -> bool:
                return ctx.token.cancelled

            if self._object_id is None:
                stats = service.find_all_objects(
                    image_ids=self._image_ids,
                    progress_cb=progress,
                    cancel_check=cancelled,
                )
            else:
                stats = service.find_object(
                    self._object_id,
                    image_ids=self._image_ids,
                    progress_cb=progress,
                    cancel_check=cancelled,
                )
            ctx.report(100, "")
            log.info(
                "Object match run %s: %d objects, %d images, %d suggestions%s",
                stats.run_id,
                stats.objects_searched,
                stats.images_scanned,
                stats.suggestions_created,
                " (cancelled)" if stats.cancelled else "",
            )
            return stats
