"""Sidebar panel — person / cluster list with search."""

from __future__ import annotations

import logging
from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.db.models import Person

log = logging.getLogger(__name__)


class PersonListItem(QListWidgetItem):
    """List item that carries a Person reference."""

    def __init__(self, person: Person) -> None:
        face_count = len(person.faces)
        label = f"{person.name}  ({face_count})"
        super().__init__(label)
        self.person_id = person.id
        self.person_name = person.name


class SidebarPanel(QWidget):
    """Left sidebar showing a searchable list of persons.

    Signals:
        person_selected: ``(person_id: int)``
    """

    person_selected = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Search bar
        search_box = QGroupBox("People")
        search_layout = QVBoxLayout(search_box)

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search person name …")
        self._search_input.textChanged.connect(self._on_search_changed)
        search_layout.addWidget(self._search_input)

        self._person_list = QListWidget()
        self._person_list.setAlternatingRowColors(True)
        self._person_list.currentItemChanged.connect(self._on_selection_changed)
        search_layout.addWidget(self._person_list)

        self._count_label = QLabel("0 persons")
        self._count_label.setAlignment(Qt.AlignCenter)
        search_layout.addWidget(self._count_label)

        layout.addWidget(search_box)

        self._recluster_btn = QPushButton("Re-cluster All")
        self._recluster_btn.setToolTip("Re-run clustering with current corrections")
        layout.addWidget(self._recluster_btn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def populate(self, persons: list[Person]) -> None:
        """Rebuild the list with the given persons."""
        self._all_persons = persons
        self._apply_filter(self._search_input.text())

    def set_recluster_callback(self, cb: Callable) -> None:
        self._recluster_btn.clicked.connect(cb)

    def current_person_id(self) -> Optional[int]:
        item = self._person_list.currentItem()
        if isinstance(item, PersonListItem):
            return item.person_id
        return None

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_search_changed(self, text: str) -> None:
        self._apply_filter(text)

    def _apply_filter(self, text: str) -> None:
        self._person_list.clear()
        persons = getattr(self, "_all_persons", [])
        text = text.strip().lower()

        shown = 0
        for person in persons:
            if text and text not in person.name.lower():
                continue
            item = PersonListItem(person)
            self._person_list.addItem(item)
            shown += 1

        self._count_label.setText(f"{shown} person(s)")

    def _on_selection_changed(
        self, current: QListWidgetItem, previous: QListWidgetItem
    ) -> None:
        if isinstance(current, PersonListItem):
            self.person_selected.emit(current.person_id)
