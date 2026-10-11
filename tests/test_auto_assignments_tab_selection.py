"""#185 — a single-row action in the Name Suggestions tab keeps the batch ticks."""

from __future__ import annotations

from typing import List

import pytest

from app.db.database import init_db
from app.services.deep_recognition_service import (
    AutoAssignmentDTO,
    DeepRecognitionService,
)
from app.ui.dialogs.auto_assignments_tab import AutoAssignmentsTab


def _dto(assignment_id: int, score: float) -> AutoAssignmentDTO:
    return AutoAssignmentDTO(
        assignment_id=assignment_id,
        face_id=assignment_id,
        person_id=1,
        person_name=f"Person {assignment_id}",
        previous_person_name=None,
        score=score,
        status="auto",
        crop_path=None,
        image_path=None,
        bbox=None,
    )


@pytest.fixture()
def open_rows(tmp_path, monkeypatch) -> List[AutoAssignmentDTO]:
    init_db(tmp_path / "auto_assign_sel.db")
    rows = [_dto(1, 0.84), _dto(2, 0.83), _dto(3, 0.82), _dto(4, 0.81)]

    def fake_revert(_svc, assignment_id):
        rows[:] = [d for d in rows if d.assignment_id != assignment_id]

    monkeypatch.setattr(
        DeepRecognitionService, "list_auto_assignments", lambda _svc: list(rows)
    )
    monkeypatch.setattr(DeepRecognitionService, "list_near_suggestions", lambda _svc: [])
    monkeypatch.setattr(DeepRecognitionService, "list_suspicious_labels", lambda _svc: [])
    monkeypatch.setattr(DeepRecognitionService, "revert_assignment", fake_revert)
    return rows


def _selected(tab: AutoAssignmentsTab) -> List[int]:
    return sorted(tab._selected_ids())


def test_revert_one_row_keeps_other_ticks(qtbot, open_rows):
    tab = AutoAssignmentsTab()
    qtbot.addWidget(tab)
    tab.reload()

    for row in tab._rows:
        row.set_selected(row.assignment_id in (1, 3))
    assert _selected(tab) == [1, 3]

    # Undo an unticked row: the ticks on 1 and 3 must survive.
    tab._on_revert(2)
    assert [r.assignment_id for r in tab._rows] == [4, 3, 1]
    assert _selected(tab) == [1, 3]

    # Undo a ticked row: it leaves the list, the other tick stays.
    tab._on_revert(3)
    assert _selected(tab) == [1]
    assert "(1)" in tab._accept_selected_btn.text()


def test_changing_sort_keeps_ticks(qtbot, open_rows):
    tab = AutoAssignmentsTab()
    qtbot.addWidget(tab)
    tab.reload()
    tab._rows[0].set_selected(True)
    ticked = tab._rows[0].assignment_id

    tab._sort_combo.setCurrentIndex(1)  # reloads with the opposite order
    assert _selected(tab) == [ticked]
