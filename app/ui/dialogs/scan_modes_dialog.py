"""Scan and maintenance chooser dialog.

Two tabs:
  * "AI recognition" — the three primary workflows (scan, train, full rescan).
  * "Maintenance (developer)" — cleanup/repair tools that now run
    automatically as part of the redesigned detection pipeline, but are kept
    here so they can still be triggered manually for debugging / one-off fixes.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.app_settings import app_qsettings
from app.services.duplicate_unknown_face_finder import (
    DEFAULT_OVERLAP_SENSITIVITY,
    OVERLAP_SENSITIVITIES,
)
from app.services.unknown_person_reset_service import UnknownPersonResetOptions
from app.ui.i18n import t

log = logging.getLogger(__name__)

# QSettings key holding the last chosen overlap-search sensitivity preset.
_OVERLAP_SENSITIVITY_SETTING = "overlap_cleanup/sensitivity"



class ScanModesDialog(QDialog):
    """Modal dialog for choosing a scan or maintenance workflow."""

    # "face_detection" | "full_rescan" | "train_model"
    scan_workflow_started = Signal(str)
    # Maintenance action key, e.g. "overlap_cleanup", "identity_repair", …
    maintenance_action_started = Signal(str)
    # Carries the UnknownPersonResetOptions chosen in the maintenance card.
    # Separate from maintenance_action_started because the launch helpers close
    # the dialog before emitting, so the checkbox state has to travel along.
    reset_unknown_requested = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None, config=None) -> None:
        super().__init__(parent)
        self._config = config
        self.setWindowTitle(t("scanModes.title"))
        self.setMinimumWidth(460)
        self.resize(560, 680)
        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        tabs = QTabWidget()
        tabs.addTab(self._build_ai_tab(), t("scanModes.tab.ai"))
        tabs.addTab(self._build_maintenance_tab(), t("scanModes.tab.maintenance"))
        outer.addWidget(tabs)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton(t("scanModes.close"))
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

    def _make_scroll(self) -> tuple[QScrollArea, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        container = QWidget()
        cards = QVBoxLayout(container)
        cards.setContentsMargins(4, 4, 4, 4)
        cards.setSpacing(10)
        scroll.setWidget(container)
        return scroll, cards

    # ------------------------------------------------------------------
    # AI recognition tab — the three primary workflows.
    # ------------------------------------------------------------------

    def _build_ai_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        scroll, cards = self._make_scroll()

        cards.addWidget(self._make_card(
            title=t("workflow_face_detection_title"),
            desc=t("workflow_face_detection_desc"),
            on_click=lambda: self._launch_workflow("face_detection"),
            danger=False,
        ))
        cards.addWidget(self._make_card(
            title=t("workflow_train_model_title"),
            desc=t("workflow_train_model_desc"),
            on_click=lambda: self._launch_workflow("train_model"),
            danger=False,
        ))
        cards.addWidget(self._make_card(
            title=t("workflow_full_rescan_title"),
            desc=t("workflow_full_rescan_desc"),
            on_click=lambda: self._launch_workflow("full_rescan"),
            danger=True,
        ))
        cards.addStretch()
        layout.addWidget(scroll)

        # Multi-technology "verify every face" toggle (no confidence exemption).
        verify_box = QVBoxLayout()
        verify_box.setSpacing(2)
        self._verify_all_chk = QCheckBox(t("scanModes.verifyAll.label"))
        det = getattr(self._config, "detection", None)
        self._verify_all_chk.setChecked(bool(getattr(det, "verification_verify_all", False)))
        self._verify_all_chk.setEnabled(self._config is not None)
        self._verify_all_chk.toggled.connect(self._on_verify_all_toggled)
        verify_box.addWidget(self._verify_all_chk)

        verify_tip = QLabel(t("scanModes.verifyAll.tip"))
        verify_tip.setWordWrap(True)
        verify_tip.setStyleSheet("color: #A6ADC8; font-size: 11px;")
        verify_tip.setContentsMargins(22, 0, 0, 0)
        verify_box.addWidget(verify_tip)
        layout.addLayout(verify_box)

        return tab

    # ------------------------------------------------------------------
    # Maintenance tab — cleanup/repair tools that also run automatically.
    # ------------------------------------------------------------------

    def _build_maintenance_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        note = QLabel(f"ℹ {t('scanModes.maintenance.note')}")
        note.setWordWrap(True)
        note.setStyleSheet(
            "color: #89B4FA; font-size: 11px; "
            "border: 1px solid #45475A; border-radius: 6px; padding: 8px;"
        )
        layout.addWidget(note)

        scroll, cards = self._make_scroll()

        # Built outside the loop below: this is the one card carrying options.
        cards.addWidget(self._make_card(
            title=t("scanModes.resetUnknowns.title"),
            desc=t("scanModes.resetUnknowns.description"),
            on_click=self._launch_reset_unknown,
            danger=False,
            button_label=t("scanModes.resetUnknowns.startButton"),
            warning=t("scanModes.resetUnknowns.warning"),
            options_widget=self._build_reset_unknown_options(),
        ))

        # The overlap search carries its own sensitivity selector.
        cards.addWidget(self._make_card(
            title=t("scanModes.overlapCleanup.title"),
            desc=t("scanModes.overlapCleanup.description"),
            on_click=lambda: self._launch_maintenance("overlap_cleanup"),
            danger=False,
            button_label=t("scanModes.overlapCleanup.startButton"),
            warning=t("scanModes.overlapCleanup.warning"),
            options_widget=self._build_overlap_options(),
        ))

        # Order mirrors the legacy "Klasszikus" maintenance list.
        maintenance = [
            ("embeddingDuplicates", "embedding_duplicates", False),
            ("identityRepair", "identity_repair", False),
            ("cleanupEmptyUnknowns", "cleanup_empty_unknowns", False),
            ("ignoredFaces", "manage_ignored_faces", False),
        ]
        for key, action, danger in maintenance:
            warn_key = f"scanModes.{key}.warning"
            warning = t(warn_key)
            if warning == warn_key:  # no warning string defined for this card
                warning = None
            cards.addWidget(self._make_card(
                title=t(f"scanModes.{key}.title"),
                desc=t(f"scanModes.{key}.description"),
                on_click=lambda a=action: self._launch_maintenance(a),
                danger=danger,
                button_label=t(f"scanModes.{key}.startButton"),
                warning=warning,
            ))
        cards.addStretch()
        layout.addWidget(scroll)
        return tab

    def _build_reset_unknown_options(self) -> QWidget:
        """Checkbox group rendered inside the "Rebuild Unknown identities" card."""
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        defaults = UnknownPersonResetOptions()
        specs = [
            ("_chk_delete_unknown_persons", "deletePersons",
             defaults.delete_unknown_persons),
            ("_chk_delete_face_assignments", "deleteFaceAssignments",
             defaults.delete_face_assignments),
            ("_chk_delete_face_data", "deleteFaceData",
             defaults.delete_face_data),
            ("_chk_rebuild_clusters", "rebuildClusters",
             defaults.rebuild_clusters),
        ]
        for attr, key, checked in specs:
            chk = QCheckBox(t(f"resetUnknownOptions.{key}"))
            chk.setToolTip(t(f"resetUnknownOptions.{key}Tooltip"))
            chk.setChecked(checked)
            setattr(self, attr, chk)
            layout.addWidget(chk)

        # Deleting the face rows outright makes un-assigning them meaningless,
        # and the service treats the two as exclusive — mirror that here so the
        # checkbox cannot promise something that will be ignored.
        self._chk_delete_face_data.toggled.connect(
            lambda on: self._chk_delete_face_assignments.setEnabled(not on)
        )
        return box

    def _build_overlap_options(self) -> QWidget:
        """Sensitivity selector rendered inside the overlap-cleanup card.

        The strict default only lists boxes that heavily overlap; the looser
        levels also list boxes that merely intersect, which is what finds the
        leftovers a strict pass walks past.
        """
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        label = QLabel(t("scanModes.overlapCleanup.sensitivity"))
        layout.addWidget(label)

        self._overlap_sensitivity_combo = QComboBox()
        for preset in OVERLAP_SENSITIVITIES:
            self._overlap_sensitivity_combo.addItem(
                t(f"overlapSensitivity.{preset.key}"), preset.key
            )
        saved = str(
            app_qsettings().value(
                _OVERLAP_SENSITIVITY_SETTING, DEFAULT_OVERLAP_SENSITIVITY
            )
        )
        index = self._overlap_sensitivity_combo.findData(saved)
        self._overlap_sensitivity_combo.setCurrentIndex(max(0, index))
        self._overlap_sensitivity_combo.currentIndexChanged.connect(
            self._on_overlap_sensitivity_changed
        )
        layout.addWidget(self._overlap_sensitivity_combo)

        tip = QLabel(t("scanModes.overlapCleanup.sensitivityTip"))
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #A6ADC8; font-size: 11px;")
        layout.addWidget(tip)
        return box

    def _on_overlap_sensitivity_changed(self, _index: int) -> None:
        key = self._overlap_sensitivity_combo.currentData()
        app_qsettings().setValue(_OVERLAP_SENSITIVITY_SETTING, key)
        log.info("Overlap search sensitivity set to %s", key)

    def reset_unknown_options(self) -> UnknownPersonResetOptions:
        """Current state of the inline Unknown-reset checkboxes."""
        return UnknownPersonResetOptions(
            delete_unknown_persons=self._chk_delete_unknown_persons.isChecked(),
            delete_face_assignments=self._chk_delete_face_assignments.isChecked(),
            delete_face_data=self._chk_delete_face_data.isChecked(),
            rebuild_clusters=self._chk_rebuild_clusters.isChecked(),
        )

    # ------------------------------------------------------------------

    def _make_card(
        self,
        title: str,
        desc: str,
        on_click: Callable[[], None],
        danger: bool,
        button_label: Optional[str] = None,
        warning: Optional[str] = None,
        options_widget: Optional[QWidget] = None,
    ) -> QFrame:
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setFrameShadow(QFrame.Raised)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        title_lbl = QLabel(f"<b>{title}</b>")
        title_lbl.setWordWrap(True)
        layout.addWidget(title_lbl)

        desc_lbl = QLabel(desc)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("color: #A6ADC8;")
        desc_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(desc_lbl)

        if warning:
            warn_lbl = QLabel(f"⚠ {warning}")
            warn_lbl.setWordWrap(True)
            warn_lbl.setStyleSheet("color: #F38BA8;" if danger else "color: #F9E2AF;")
            layout.addWidget(warn_lbl)

        if options_widget is not None:
            layout.addWidget(options_widget)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn = QPushButton(button_label or t("button_start"))
        btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        if danger:
            btn.setStyleSheet(
                "QPushButton { color: #F38BA8; border-color: #6B3040; }"
                "QPushButton:hover { background-color: #3D2030; border-color: #F38BA8; }"
            )
        btn.clicked.connect(on_click)
        btn_row.addWidget(btn)
        layout.addLayout(btn_row)

        return card

    def _on_verify_all_toggled(self, checked: bool) -> None:
        """Update the live config and persist the choice to config.yaml."""
        if self._config is None:
            return
        self._config.detection.verification_verify_all = bool(checked)
        try:
            from app.config import save_detection_values

            save_detection_values({"verification_verify_all": bool(checked)})
            log.info("verification_verify_all set to %s (persisted)", checked)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not persist verification_verify_all: %s", exc)

    def _launch_workflow(self, workflow: str) -> None:
        self.accept()
        self.scan_workflow_started.emit(workflow)

    def _launch_maintenance(self, action: str) -> None:
        self.accept()
        self.maintenance_action_started.emit(action)

    def _launch_reset_unknown(self) -> None:
        options = self.reset_unknown_options()
        self.accept()
        self.reset_unknown_requested.emit(options)
