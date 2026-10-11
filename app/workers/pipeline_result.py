"""Shared result type for processing-pipeline workers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class PipelineResult:
    """Outcome of a pipeline run, returned to the Task Manager's ``on_done``.

    ``success`` is ``False`` for soft failures that should warn the user but are
    not crashes (e.g. a stage degraded gracefully).  Hard failures raise; user
    cancellation raises :class:`OperationCancelled` and never produces a result.
    """

    success: bool
    summary: str
    n_suggestions: int = 0
    n_auto_assignments: int = 0
    # Images the scan found new or changed; the UI searches tagged objects on
    # them afterwards (#164).  Empty for modes that do not scan.
    new_image_ids: List[int] = field(default_factory=list)
