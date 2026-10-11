"""Review tab for the deep engine's automatic groupings.

Lists every face the last AI run placed with a person.  Each row shows the
face crop, where it came from, where it was put and the model's confidence.
The user can:

* **confirm** — the grouping is right; the face becomes trusted training data;
* **correct** — pick the right person (or create a new one); the engine
  records the mistake as a correction and learns from it;
* **revert** — send the face back to where it was before the run.

Batch shortcuts: tick rows and accept/move/reject them together, or accept
every grouping whose match is at or above a chosen percentage in one go.

Active learning lives here too: **near matches** (faces the AI was close on
but did not assign — one click confirms or rejects) and **suspicious labels**
(labeled faces that look much more like someone else; kept out of training
until the user decides).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.db.models import (
    AUTO_ASSIGN_STATUS_CONFIRMED,
    AUTO_ASSIGN_STATUS_CORRECTED,
    AUTO_ASSIGN_STATUS_REVERTED,
    AutoAssignment,
    Person,
    TrainingRun,
)
from app.services.deep_recognition_service import (
    AutoAssignmentDTO,
    DeepRecognitionService,
    SuspiciousLabelDTO,
)
from app.ui.i18n import t

log = logging.getLogger(__name__)

_ROW_STYLE = "QFrame { background: #232323; border-radius: 6px; }"

# Remembered "accept all at or above N%" threshold (QSettings key + default).
_S_ACCEPT_THRESHOLD = "auto_assignments/accept_threshold_pct"
_DEFAULT_ACCEPT_THRESHOLD = 90


def _score_pct(score: float) -> int:
    """Match score as the whole percentage shown on the row."""
    return int(round(max(0.0, min(1.0, score)) * 100))


def _open_decision_graph(
    dto: AutoAssignmentDTO, parent: QWidget, deep_config=None
) -> None:
    """Open the deep-engine decision graph for *dto* (or a "no graph" notice).

    Rows assigned before the per-row decision was persisted carry no stored
    graph; for those we recompute it on demand from the saved model so the
    button still works.  Only when that also fails do we show "no graph".
    """
    from app.ui.dialogs.merge_decision_graph_dialog import DeepDecisionGraphDialog

    decision = dto.decision
    if decision is None:
        try:
            with session_scope() as session:
                decision = DeepRecognitionService(
                    session, deep_config
                ).decision_for_face(dto.face_id)
        except Exception:  # noqa: BLE001
            log.exception("Failed to recompute decision graph for face %s", dto.face_id)
            decision = None
    DeepDecisionGraphDialog(
        decision, parent, face_id=dto.face_id, deep_config=deep_config
    ).exec()


class _DecidedAssignmentRow(QFrame):
    """One already-reviewed automatic grouping.

    Shows the decision; lets the user *override* it afterwards (undo the
    confirm/correct so the face returns to its pre-run state).
    """

    reverted = Signal(int)  # assignment_id

    def __init__(
        self,
        dto: AutoAssignmentDTO,
        parent: Optional[QWidget] = None,
        deep_config=None,
    ) -> None:
        super().__init__(parent)
        self._dto = dto
        self._deep_config = deep_config
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { background: #1d1d1d; border-radius: 6px; }")

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(8)

        from app.ui.dialogs.suggestion_dialog import _crop_pixmap

        crop = QLabel()
        crop.setFixedSize(36, 36)
        crop.setAlignment(Qt.AlignCenter)
        pixmap = _crop_pixmap(dto.crop_path)
        if pixmap is not None:
            crop.setPixmap(
                pixmap.scaled(36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            crop.setStyleSheet("border: 1px solid #444; border-radius: 4px;")
        else:
            crop.setText("?")
            crop.setStyleSheet(
                "border: 1px solid #444; border-radius: 4px; color: #666;"
            )
        row.addWidget(crop)

        name = QLabel(f"→ {dto.person_name}")
        name.setStyleSheet("color: #ccc; border: none;")
        name.setWordWrap(True)
        row.addWidget(name, stretch=1)

        if dto.status == AUTO_ASSIGN_STATUS_CONFIRMED:
            status_text, color = t("autoAssign.status_confirmed"), "#88ee88"
        elif dto.status == AUTO_ASSIGN_STATUS_CORRECTED:
            status_text = t(
                "autoAssign.status_corrected",
                name=dto.corrected_person_name or "?",
            )
            color = "#88aaff"
        elif dto.status == AUTO_ASSIGN_STATUS_REVERTED:
            status_text, color = t("autoAssign.status_reverted"), "#ff8888"
        else:
            status_text, color = dto.status, "#aaaaaa"
        status = QLabel(status_text)
        status.setStyleSheet(
            f"color: {color}; font-weight: bold; border: none; font-size: 11px;"
        )
        row.addWidget(status)

        if dto.decided_at is not None:
            when = QLabel(dto.decided_at.strftime("%Y-%m-%d %H:%M"))
            when.setStyleSheet("color: #666; border: none; font-size: 10px;")
            row.addWidget(when)

        graph_btn = QPushButton("📊")
        graph_btn.setToolTip(t("autoAssign.graph"))
        graph_btn.setFixedWidth(40)
        graph_btn.clicked.connect(
            lambda: _open_decision_graph(dto, self, self._deep_config)
        )
        row.addWidget(graph_btn)

        # Let the user reverse a confirm/correct after the fact. A row that is
        # already reverted has nothing left to undo.
        if dto.status != AUTO_ASSIGN_STATUS_REVERTED:
            override_btn = QPushButton(t("autoAssign.override"))
            override_btn.setStyleSheet("QPushButton { color: #ffcc66; }")
            override_btn.setToolTip(t("autoAssign.override_tooltip"))
            override_btn.clicked.connect(
                lambda: self.reverted.emit(dto.assignment_id)
            )
            row.addWidget(override_btn)


class _AutoAssignmentRow(QFrame):
    """One reviewable automatic grouping."""

    confirmed = Signal(int)        # assignment_id
    corrected = Signal(int)        # assignment_id
    reverted = Signal(int)         # assignment_id
    selection_changed = Signal()   # any checkbox toggled

    def __init__(
        self,
        dto: AutoAssignmentDTO,
        parent: Optional[QWidget] = None,
        deep_config=None,
        near: bool = False,
    ) -> None:
        super().__init__(parent)
        self._dto = dto
        self._deep_config = deep_config
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(_ROW_STYLE)

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 8, 8, 8)
        row.setSpacing(8)

        # Per-row tick for batch confirm ("Accept selected").  Near-match
        # suggestions are answered one by one.
        self._check = QCheckBox()
        self._check.setToolTip(t("autoAssign.accept_selected_tooltip"))
        self._check.toggled.connect(lambda _on: self.selection_changed.emit())
        self._check.setVisible(not near)
        row.addWidget(self._check, alignment=Qt.AlignVCenter)

        # Reuse the suggestion dialog's clickable crop (hover zoom + full image).
        from app.ui.dialogs.suggestion_dialog import _ClickableCrop

        row.addWidget(
            _ClickableCrop(dto.crop_path, dto.image_path, dto.bbox, dto.face_id)
        )

        info = QVBoxLayout()
        info.setSpacing(2)

        target = QLabel(dto.person_name)
        target.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #eee; border: none;"
        )
        target.setWordWrap(True)
        info.addWidget(target)

        if dto.previous_person_name:
            origin = QLabel(
                t("autoAssign.from_group", name=dto.previous_person_name)
            )
        else:
            origin = QLabel(t("autoAssign.from_unassigned"))
        origin.setStyleSheet("font-size: 10px; color: #888; border: none;")
        origin.setWordWrap(True)
        info.addWidget(origin)
        row.addLayout(info, stretch=1)

        pct = _score_pct(dto.score)
        score = QLabel(t("suggestions_similarity", pct=pct))
        score.setAlignment(Qt.AlignCenter)
        # Min width keeps the score column aligned across rows; it must grow,
        # not clip, so "100% egyezés" stays fully visible (a fixed 70px width
        # used to chop the leading "1", showing a misleading "00%").
        score.setMinimumWidth(96)
        score.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #88ee88; border: none;"
        )
        row.addWidget(score)

        graph_btn = QPushButton("📊")
        graph_btn.setToolTip(t("autoAssign.graph"))
        graph_btn.setFixedWidth(40)
        graph_btn.clicked.connect(
            lambda: _open_decision_graph(dto, self, self._deep_config)
        )
        row.addWidget(graph_btn)

        confirm_key = "autoAssign.near_confirm" if near else "autoAssign.confirm"
        confirm_btn = QPushButton(t(confirm_key))
        confirm_btn.setStyleSheet("QPushButton { color: #88ee88; }")
        confirm_btn.setToolTip(t(f"{confirm_key}_tooltip"))
        confirm_btn.clicked.connect(lambda: self.confirmed.emit(dto.assignment_id))
        row.addWidget(confirm_btn)

        correct_btn = QPushButton(t("autoAssign.correct"))
        correct_btn.setStyleSheet("QPushButton { color: #88aaff; }")
        correct_btn.setToolTip(t("autoAssign.correct_tooltip"))
        correct_btn.clicked.connect(lambda: self.corrected.emit(dto.assignment_id))
        row.addWidget(correct_btn)

        revert_key = "autoAssign.near_reject" if near else "autoAssign.revert"
        revert_btn = QPushButton(t(revert_key))
        revert_btn.setStyleSheet("QPushButton { color: #ff8888; }")
        revert_btn.setToolTip(t(f"{revert_key}_tooltip"))
        revert_btn.clicked.connect(lambda: self.reverted.emit(dto.assignment_id))
        row.addWidget(revert_btn)

    # ------------------------------------------------------------------

    @property
    def assignment_id(self) -> int:
        return self._dto.assignment_id

    @property
    def score_pct(self) -> int:
        return _score_pct(self._dto.score)

    def is_selected(self) -> bool:
        return self._check.isChecked()

    def set_selected(self, value: bool) -> None:
        self._check.setChecked(value)


class _SuspiciousLabelRow(QFrame):
    """A labeled face that looks much more like another person."""

    kept = Signal(int)      # face_id
    removed = Signal(int)   # face_id

    def __init__(self, dto: SuspiciousLabelDTO, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(_ROW_STYLE)

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(8)

        from app.ui.dialogs.suggestion_dialog import _ClickableCrop

        row.addWidget(
            _ClickableCrop(dto.crop_path, dto.image_path, dto.bbox, dto.face_id)
        )
        text = (
            t(
                "autoAssign.suspicious_lookalike",
                name=dto.person_name,
                other=dto.lookalike_name,
            )
            if dto.lookalike_name
            else dto.person_name
        )
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: #eee; border: none;")
        if dto.own_similarity is not None and dto.other_similarity is not None:
            label.setToolTip(
                f"{dto.person_name}: {dto.own_similarity:.2f} · "
                f"{dto.lookalike_name or '?'}: {dto.other_similarity:.2f}"
            )
        row.addWidget(label, stretch=1)

        keep_btn = QPushButton(t("autoAssign.suspicious_keep"))
        keep_btn.setStyleSheet("QPushButton { color: #88ee88; }")
        keep_btn.setToolTip(t("autoAssign.suspicious_keep_tooltip"))
        keep_btn.clicked.connect(lambda: self.kept.emit(dto.face_id))
        row.addWidget(keep_btn)

        remove_btn = QPushButton(t("autoAssign.suspicious_remove"))
        remove_btn.setStyleSheet("QPushButton { color: #ff8888; }")
        remove_btn.setToolTip(t("autoAssign.suspicious_remove_tooltip"))
        remove_btn.clicked.connect(lambda: self.removed.emit(dto.face_id))
        row.addWidget(remove_btn)


def _clear_layout(layout: QVBoxLayout) -> None:
    """Remove every row widget, keeping the trailing stretch."""
    while layout.count() > 1:
        item = layout.takeAt(0)
        widget = item.widget()
        if widget:
            widget.deleteLater()


class AutoAssignmentsTab(QWidget):
    """Tab widget listing the last run's automatic groupings for review.

    Signals:
        data_changed: emitted after any confirm / correct / revert.
        count_changed: ``(n_open: int)`` — for the tab-title badge.
    """

    data_changed = Signal()
    count_changed = Signal(int)

    def __init__(
        self, deep_config=None, app_config=None, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._deep_config = deep_config
        # Full AppConfig (optional): lets the "Move…" pickers compute face-match
        # scores so they expose the shared "order by similarity" toggle, with the
        # deep AI probability overlay.  ``None`` simply falls back to name order.
        self._app_config = app_config
        self._loaded = False
        self._rows: List[_AutoAssignmentRow] = []
        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        intro = QLabel(t("autoAssign.intro"))
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        root.addWidget(intro)

        self._metrics_label = QLabel("")
        self._metrics_label.setWordWrap(True)
        self._metrics_label.setStyleSheet("color: #888; font-size: 11px;")
        self._metrics_label.setVisible(False)
        root.addWidget(self._metrics_label)

        header = QHBoxLayout()
        self._run_label = QLabel("")
        self._run_label.setStyleSheet("color: #888;")
        header.addWidget(self._run_label)
        header.addStretch()
        header.addWidget(QLabel(t("autoAssign.sort_label")))
        # Least-certain first by default: those reviews teach the AI the most.
        self._sort_combo = QComboBox()
        self._sort_combo.addItem(t("autoAssign.sort_uncertain"), "uncertain")
        self._sort_combo.addItem(t("autoAssign.sort_certain"), "certain")
        self._sort_combo.setToolTip(t("autoAssign.sort_tooltip"))
        self._sort_combo.currentIndexChanged.connect(lambda _i: self.reload())
        header.addWidget(self._sort_combo)
        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: #88aaff; font-weight: bold;")
        header.addWidget(self._count_label)
        self._revert_all_btn = QPushButton(t("autoAssign.revert_all"))
        self._revert_all_btn.setToolTip(t("autoAssign.revert_all_tooltip"))
        self._revert_all_btn.clicked.connect(self._on_revert_all)
        self._revert_all_btn.setVisible(False)
        header.addWidget(self._revert_all_btn)
        root.addLayout(header)

        # ── Selection toolbar: tick rows, then accept them in one go ───────
        self._select_bar = QWidget()
        sel = QHBoxLayout(self._select_bar)
        sel.setContentsMargins(0, 0, 0, 0)
        sel.setSpacing(6)

        self._select_all_btn = QPushButton(t("autoAssign.select_all"))
        self._select_all_btn.setToolTip(t("autoAssign.select_all_tooltip"))
        self._select_all_btn.clicked.connect(lambda: self._set_all_selected(True))
        sel.addWidget(self._select_all_btn)

        self._clear_sel_btn = QPushButton(t("autoAssign.clear_selection"))
        self._clear_sel_btn.setToolTip(t("autoAssign.clear_selection_tooltip"))
        self._clear_sel_btn.clicked.connect(lambda: self._set_all_selected(False))
        sel.addWidget(self._clear_sel_btn)

        sel.addStretch()

        self._accept_selected_btn = QPushButton()
        self._accept_selected_btn.setToolTip(t("autoAssign.accept_selected_tooltip"))
        self._accept_selected_btn.setStyleSheet("QPushButton { color: #88ee88; }")
        self._accept_selected_btn.clicked.connect(self._on_accept_selected)
        sel.addWidget(self._accept_selected_btn)

        self._assign_selected_btn = QPushButton()
        self._assign_selected_btn.setToolTip(t("autoAssign.assign_selected_tooltip"))
        self._assign_selected_btn.setStyleSheet("QPushButton { color: #88aaff; }")
        self._assign_selected_btn.clicked.connect(self._on_assign_selected)
        sel.addWidget(self._assign_selected_btn)

        self._reject_selected_btn = QPushButton()
        self._reject_selected_btn.setToolTip(t("autoAssign.reject_selected_tooltip"))
        self._reject_selected_btn.setStyleSheet("QPushButton { color: #ff8888; }")
        self._reject_selected_btn.clicked.connect(self._on_reject_selected)
        sel.addWidget(self._reject_selected_btn)

        self._select_bar.setVisible(False)
        root.addWidget(self._select_bar)

        # ── Threshold toolbar: accept every grouping at/above N% at once ──
        self._threshold_bar = QWidget()
        thr = QHBoxLayout(self._threshold_bar)
        thr.setContentsMargins(0, 0, 0, 0)
        thr.setSpacing(6)

        self._threshold_label = QLabel(t("autoAssign.threshold_label"))
        self._threshold_label.setStyleSheet("color: #aaa;")
        thr.addWidget(self._threshold_label)

        self._threshold_spin = QSpinBox()
        self._threshold_spin.setRange(0, 100)
        self._threshold_spin.setSuffix(" %")
        self._threshold_spin.setValue(self._load_threshold())
        self._threshold_spin.setToolTip(t("autoAssign.accept_above_tooltip"))
        self._threshold_spin.valueChanged.connect(self._on_threshold_changed)
        thr.addWidget(self._threshold_spin)

        self._select_above_btn = QPushButton()
        self._select_above_btn.setToolTip(t("autoAssign.select_above_tooltip"))
        self._select_above_btn.clicked.connect(self._on_select_above)
        thr.addWidget(self._select_above_btn)

        self._accept_above_btn = QPushButton()
        self._accept_above_btn.setToolTip(t("autoAssign.accept_above_tooltip"))
        self._accept_above_btn.setStyleSheet("QPushButton { color: #88ee88; }")
        self._accept_above_btn.clicked.connect(self._on_accept_above)
        thr.addWidget(self._accept_above_btn)

        thr.addStretch()
        self._threshold_bar.setVisible(False)
        root.addWidget(self._threshold_bar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setSpacing(6)
        self._list_layout.addStretch()
        scroll.setWidget(self._list_widget)
        root.addWidget(scroll, stretch=1)

        self._empty_label = QLabel(t("autoAssign.empty"))
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet(
            "color: #888; font-style: italic; padding: 20px;"
        )
        root.addWidget(self._empty_label)

        from app.ui.widgets.collapsible_section import CollapsibleSection

        # ── Near matches (active learning, collapsed by default) ──────────
        self._near_section, self._near_layout = self._make_section(
            CollapsibleSection, t("autoAssign.near_header", n=0),
            t("autoAssign.near_intro"),
        )
        root.addWidget(self._near_section)

        # ── Suspicious labels (training-data hygiene) ─────────────────────
        self._suspicious_section, self._suspicious_layout = self._make_section(
            CollapsibleSection, t("autoAssign.suspicious_header", n=0),
            t("autoAssign.suspicious_intro"),
        )
        root.addWidget(self._suspicious_section)

        # ── "Already reviewed" history (collapsed by default) ─────────────
        decided_scroll = QScrollArea()
        decided_scroll.setWidgetResizable(True)
        decided_scroll.setMaximumHeight(240)
        self._decided_widget = QWidget()
        self._decided_layout = QVBoxLayout(self._decided_widget)
        self._decided_layout.setSpacing(4)
        self._decided_layout.addStretch()
        decided_scroll.setWidget(self._decided_widget)

        self._decided_section = CollapsibleSection(
            t("autoAssign.decided_header", n=0), decided_scroll
        )
        self._decided_section.toggled.connect(self._on_decided_toggled)
        self._decided_loaded = False
        root.addWidget(self._decided_section)

    @staticmethod
    def _make_section(section_cls, title: str, intro_text: str):
        """A collapsible, scrollable list with an explanatory intro line."""
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        intro = QLabel(intro_text)
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #888; font-size: 11px;")
        body_layout.addWidget(intro)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(260)
        inner = QWidget()
        rows = QVBoxLayout(inner)
        rows.setSpacing(4)
        rows.addStretch()
        scroll.setWidget(inner)
        body_layout.addWidget(scroll)
        return section_cls(title, body), rows

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def ensure_loaded(self) -> None:
        """Load on first show (the tab is cheap until the user opens it)."""
        if not self._loaded:
            self.reload()

    def reload(self) -> None:
        self._loaded = True
        dtos: List[AutoAssignmentDTO] = []
        near: List[AutoAssignmentDTO] = []
        suspicious: List[SuspiciousLabelDTO] = []
        latest_run = None
        try:
            with session_scope() as session:
                svc = DeepRecognitionService(session, self._deep_config)
                dtos = svc.list_auto_assignments()
                near = svc.list_near_suggestions()
                suspicious = svc.list_suspicious_labels()
                latest_run = (
                    session.query(TrainingRun)
                    .filter(TrainingRun.finished_at.isnot(None))
                    .order_by(TrainingRun.id.desc())
                    .first()
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to load automatic groupings")
            QMessageBox.critical(self, t("error"), str(exc))

        uncertain_first = self._sort_combo.currentData() != "certain"
        dtos.sort(key=lambda d: d.score, reverse=not uncertain_first)

        # Keep the user's ticks across the rebuild: a single-row action
        # (confirm / correct / revert) must not wipe the batch selection.
        # Rows that were just decided simply drop out of the list.
        selected = set(self._selected_ids())

        _clear_layout(self._list_layout)
        self._rows = []

        for dto in dtos:
            row = _AutoAssignmentRow(dto, deep_config=self._deep_config)
            row.set_selected(dto.assignment_id in selected)
            row.confirmed.connect(self._on_confirm)
            row.corrected.connect(self._on_correct)
            row.reverted.connect(self._on_revert)
            row.selection_changed.connect(self._update_selection_ui)
            self._rows.append(row)
            self._list_layout.insertWidget(self._list_layout.count() - 1, row)

        if dtos and dtos[0].run_finished_at is not None:
            stamp = dtos[0].run_finished_at.strftime("%Y-%m-%d %H:%M")
            self._run_label.setText(t("autoAssign.run_info", date=stamp))
        else:
            self._run_label.setText("")

        self._count_label.setText(t("autoAssign.count", n=len(dtos)))
        self._empty_label.setVisible(not dtos)
        self._revert_all_btn.setVisible(bool(dtos))
        self._select_bar.setVisible(bool(dtos))
        self._threshold_bar.setVisible(bool(dtos))
        self._update_selection_ui()
        self._update_threshold_ui()
        self.count_changed.emit(len(dtos))
        self._fill_near(near)
        self._fill_suspicious(suspicious)
        self._show_metrics(latest_run)
        self._refresh_decided()

    def _fill_near(self, near: List[AutoAssignmentDTO]) -> None:
        _clear_layout(self._near_layout)
        self._near_section.set_title(t("autoAssign.near_header", n=len(near)))
        if not near:
            empty = QLabel(t("autoAssign.near_empty"))
            empty.setStyleSheet("color: #666; font-style: italic; border: none;")
            self._near_layout.insertWidget(0, empty)
        for i, dto in enumerate(near):
            row = _AutoAssignmentRow(dto, deep_config=self._deep_config, near=True)
            row.confirmed.connect(self._on_confirm)
            row.corrected.connect(self._on_correct)
            row.reverted.connect(self._on_revert)
            self._near_layout.insertWidget(i, row)

    def _fill_suspicious(self, items: List[SuspiciousLabelDTO]) -> None:
        _clear_layout(self._suspicious_layout)
        self._suspicious_section.set_title(
            t("autoAssign.suspicious_header", n=len(items))
        )
        if not items:
            empty = QLabel(t("autoAssign.suspicious_empty"))
            empty.setStyleSheet("color: #666; font-style: italic; border: none;")
            self._suspicious_layout.insertWidget(0, empty)
        for i, dto in enumerate(items):
            row = _SuspiciousLabelRow(dto)
            row.kept.connect(
                lambda fid: self._run_action(lambda svc: svc.keep_label(fid), "keep_label")
            )
            row.removed.connect(
                lambda fid: self._run_action(lambda svc: svc.remove_label(fid), "remove_label")
            )
            self._suspicious_layout.insertWidget(i, row)

    def _show_metrics(self, run: Optional[TrainingRun]) -> None:
        """One line: target precision vs. the precision your reviews measured."""
        if run is None or self._deep_config is None:
            self._metrics_label.setVisible(False)
            return
        target = getattr(self._deep_config, "target_precision", None)
        measured = run.measured_precision
        if target is None or measured is None:
            self._metrics_label.setVisible(False)
            return
        self._metrics_label.setText(
            t(
                "autoAssign.metrics",
                target=f"{target * 100:.0f}%",
                measured=f"{measured * 100:.0f}% ({run.n_reviewed or 0})",
            )
        )
        self._metrics_label.setVisible(True)

    def _refresh_decided(self) -> None:
        """Update the "already reviewed" header; reload its rows if open."""
        from app.db.models import (
            AUTO_ASSIGN_STATUS_AUTO,
            AUTO_ASSIGN_STATUS_NEAR,
            AutoAssignment,
        )

        try:
            with session_scope() as session:
                count = (
                    session.query(AutoAssignment)
                    .filter(
                        AutoAssignment.status.notin_(
                            (AUTO_ASSIGN_STATUS_AUTO, AUTO_ASSIGN_STATUS_NEAR)
                        )
                    )
                    .count()
                )
                decided = (
                    DeepRecognitionService(
                        session, self._deep_config
                    ).list_decided_assignments()
                    if self._decided_section.is_expanded()
                    else None
                )
        except Exception:  # noqa: BLE001 — history must never break the tab
            log.exception("Failed to load reviewed groupings")
            return

        self._decided_section.set_title(t("autoAssign.decided_header", n=count))
        if decided is None:
            self._decided_loaded = False
            return
        self._decided_loaded = True

        while self._decided_layout.count() > 1:
            item = self._decided_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        if not decided:
            empty = QLabel(t("autoAssign.decided_empty"))
            empty.setStyleSheet("color: #666; font-style: italic; border: none;")
            empty.setAlignment(Qt.AlignCenter)
            self._decided_layout.insertWidget(0, empty)
        for i, dto in enumerate(decided):
            drow = _DecidedAssignmentRow(dto, deep_config=self._deep_config)
            drow.reverted.connect(self._on_revert)
            self._decided_layout.insertWidget(i, drow)

    def _on_decided_toggled(self, expanded: bool) -> None:
        if expanded and not self._decided_loaded:
            self._refresh_decided()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _face_ids_for(self, session, assignment_ids: List[int]) -> List[int]:
        """Resolve assignment ids to their face ids (cheap indexed lookup).

        The Move… picker scores these faces in the background (the dialog opens
        instantly with a "computing…" hint), so only this quick id resolution
        happens on the GUI thread.
        """
        if not assignment_ids:
            return []
        rows = (
            session.query(AutoAssignment.face_id)
            .filter(AutoAssignment.id.in_(assignment_ids))
            .all()
        )
        return [r[0] for r in rows]

    def _on_confirm(self, assignment_id: int) -> None:
        self._run_action(
            lambda svc: svc.confirm_assignment(assignment_id), "confirm"
        )

    def _on_correct(self, assignment_id: int) -> None:
        from app.ui.dialogs.move_faces_dialog import MoveFacesDialog

        try:
            with session_scope() as session:
                persons = (
                    session.query(Person)
                    .filter(Person.is_auto_named == False)  # noqa: E712
                    .filter(Person.is_protected == False)  # noqa: E712
                    .order_by(Person.name)
                    .all()
                )
                face_ids = self._face_ids_for(session, [assignment_id])
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to load persons for correction")
            QMessageBox.critical(self, t("error"), str(exc))
            return

        dlg = MoveFacesDialog(
            face_count=1,
            persons=persons,
            score_face_ids=face_ids,
            recognition_cfg=getattr(self._app_config, "recognition", None),
            config=self._app_config,
            parent=self.window(),
        )
        if dlg.exec() != MoveFacesDialog.Accepted:
            return

        new_name = dlg.new_person_name()
        person_id = dlg.selected_person_id()

        def action(svc: DeepRecognitionService):
            if new_name:
                svc.correct_assignment_to_new_person(assignment_id, new_name)
            elif person_id is not None:
                svc.correct_assignment(assignment_id, person_id)

        self._run_action(action, "correct")

    def _on_revert(self, assignment_id: int) -> None:
        self._run_action(
            lambda svc: svc.revert_assignment(assignment_id), "revert"
        )

    # ── Batch selection / accept ──────────────────────────────────────

    def _selected_ids(self) -> List[int]:
        return [r.assignment_id for r in self._rows if r.is_selected()]

    def _set_all_selected(self, value: bool) -> None:
        for row in self._rows:
            row.set_selected(value)
        self._update_selection_ui()

    def _update_selection_ui(self) -> None:
        """Keep the batch buttons' labels/enabled state in sync with the ticks."""
        n = len(self._selected_ids())
        self._accept_selected_btn.setText(t("autoAssign.accept_selected", n=n))
        self._assign_selected_btn.setText(t("autoAssign.assign_selected", n=n))
        self._reject_selected_btn.setText(t("autoAssign.reject_selected", n=n))
        for btn in (
            self._accept_selected_btn,
            self._assign_selected_btn,
            self._reject_selected_btn,
        ):
            btn.setEnabled(n > 0)

    def _on_accept_selected(self) -> None:
        ids = self._selected_ids()
        if not ids:
            return
        self._run_action(lambda svc: svc.confirm_assignments(ids), "accept_selected")

    # ── Accept everything at/above a threshold ────────────────────────

    @staticmethod
    def _load_threshold() -> int:
        try:
            from app.app_settings import app_qsettings

            value = int(
                app_qsettings().value(_S_ACCEPT_THRESHOLD, _DEFAULT_ACCEPT_THRESHOLD)
            )
        except Exception:  # noqa: BLE001 — a bad stored value must not break the tab
            return _DEFAULT_ACCEPT_THRESHOLD
        return max(0, min(100, value))

    def _on_threshold_changed(self, value: int) -> None:
        try:
            from app.app_settings import app_qsettings

            app_qsettings().setValue(_S_ACCEPT_THRESHOLD, int(value))
        except Exception:  # noqa: BLE001
            log.exception("Failed to store the accept threshold")
        self._update_threshold_ui()

    def _ids_above_threshold(self) -> List[int]:
        """Open groupings whose shown match % is at or above the threshold."""
        limit = self._threshold_spin.value()
        return [r.assignment_id for r in self._rows if r.score_pct >= limit]

    def _update_threshold_ui(self) -> None:
        pct = self._threshold_spin.value()
        n = len(self._ids_above_threshold())
        self._select_above_btn.setText(t("autoAssign.select_above", pct=pct, n=n))
        self._accept_above_btn.setText(t("autoAssign.accept_above", pct=pct, n=n))
        self._select_above_btn.setEnabled(n > 0)
        self._accept_above_btn.setEnabled(n > 0)

    def _on_select_above(self) -> None:
        """Tick exactly the rows at/above the threshold (for a final look)."""
        limit = self._threshold_spin.value()
        for row in self._rows:
            row.set_selected(row.score_pct >= limit)
        self._update_selection_ui()

    def _on_accept_above(self) -> None:
        ids = self._ids_above_threshold()
        if not ids:
            return
        answer = QMessageBox.question(
            self,
            t("autoAssign.accept_above_confirm_title"),
            t(
                "autoAssign.accept_above_confirm_msg",
                n=len(ids),
                pct=self._threshold_spin.value(),
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._run_action(lambda svc: svc.confirm_assignments(ids), "accept_above")

    def _on_reject_selected(self) -> None:
        ids = self._selected_ids()
        if not ids:
            return
        self._run_action(lambda svc: svc.revert_assignments(ids), "reject_selected")

    def _on_assign_selected(self) -> None:
        ids = self._selected_ids()
        if not ids:
            return
        from app.ui.dialogs.move_faces_dialog import MoveFacesDialog

        try:
            with session_scope() as session:
                persons = (
                    session.query(Person)
                    .filter(Person.is_auto_named == False)  # noqa: E712
                    .filter(Person.is_protected == False)  # noqa: E712
                    .order_by(Person.name)
                    .all()
                )
                face_ids = self._face_ids_for(session, ids)
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to load persons for batch correction")
            QMessageBox.critical(self, t("error"), str(exc))
            return

        dlg = MoveFacesDialog(
            face_count=len(ids),
            persons=persons,
            score_face_ids=face_ids,
            recognition_cfg=getattr(self._app_config, "recognition", None),
            config=self._app_config,
            parent=self.window(),
        )
        if dlg.exec() != MoveFacesDialog.Accepted:
            return

        new_name = dlg.new_person_name()
        person_id = dlg.selected_person_id()

        def action(svc: DeepRecognitionService):
            if new_name:
                svc.correct_assignments_to_new_person(ids, new_name)
            elif person_id is not None:
                svc.correct_assignments(ids, person_id)

        self._run_action(action, "assign_selected")

    def _on_revert_all(self) -> None:
        n_open = self._list_layout.count() - 1
        answer = QMessageBox.question(
            self,
            t("autoAssign.revert_all_confirm_title"),
            t("autoAssign.revert_all_confirm_msg", n=n_open),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._run_action(lambda svc: svc.revert_all_open(), "revert_all")

    def _run_action(self, action, label: str) -> None:
        try:
            with session_scope() as session:
                action(DeepRecognitionService(session, self._deep_config))
        except Exception as exc:  # noqa: BLE001
            log.exception("Automatic-grouping %s failed", label)
            QMessageBox.warning(self, t("error"), str(exc))
            return
        self.reload()
        self.data_changed.emit()
