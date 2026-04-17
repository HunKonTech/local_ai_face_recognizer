"""Merge persons dialog."""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from app.db.models import Person


class MergeDialog(QDialog):
    """Dialog to merge the currently selected person into another person."""

    def __init__(
        self,
        source_person: Person,
        all_persons: List[Person],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Merge Into …")
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                f"Merge  <b>{source_person.name}</b>  into:"
            )
        )

        self._combo = QComboBox()
        for person in all_persons:
            if person.id == source_person.id:
                continue
            face_count = len(person.faces)
            self._combo.addItem(
                f"{person.name}  ({face_count} faces)", userData=person.id
            )

        layout.addWidget(self._combo)
        layout.addWidget(
            QLabel(
                "<small>The source person will be deleted after merging.</small>"
            )
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def target_person_id(self) -> Optional[int]:
        """Return the selected target person ID, or ``None`` if empty."""
        data = self._combo.currentData()
        return int(data) if data is not None else None
