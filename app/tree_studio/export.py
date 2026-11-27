"""Tree Studio: export the composed chart to PNG / JPG / SVG."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QMessageBox,
    QProgressDialog,
    QSpinBox,
)

from app.ui.i18n import t
from app.ui.widgets.family_tree.scene_export import export_scene

log = logging.getLogger(__name__)


class ExportMixin:
    def export_image(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(t("ts_export_title"))
        form = QFormLayout(dlg)
        fmt = QComboBox()
        for label, value in (("PNG", "png"), ("JPG", "jpg"), ("SVG", "svg")):
            fmt.addItem(label, value)
        form.addRow(t("ts_export_format"), fmt)
        scale = QComboBox()
        for s in (1, 2, 3):
            scale.addItem(f"{s}×", s)
        form.addRow(t("ts_export_scale"), scale)
        quality = QSpinBox()
        quality.setRange(30, 100)
        quality.setValue(90)
        form.addRow(t("ts_export_quality"), quality)
        markers = QCheckBox(t("ts_export_markers"))
        form.addRow(markers)

        def _sync() -> None:
            is_svg = fmt.currentData() == "svg"
            scale.setEnabled(not is_svg)
            quality.setEnabled(fmt.currentData() == "jpg")

        fmt.currentIndexChanged.connect(_sync)
        _sync()
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(dlg.accept)
        box.rejected.connect(dlg.reject)
        form.addRow(box)
        if dlg.exec() != QDialog.Accepted:
            return
        ext = fmt.currentData()
        default = f"{self.view_name or 'family-tree'}.{ext}"
        path, _f = QFileDialog.getSaveFileName(self, t("ts_export_title"), default, f"{ext.upper()} (*.{ext})")
        if not path:
            return
        if not path.lower().endswith(f".{ext}"):
            path += f".{ext}"
        progress = QProgressDialog(t("ts_export_progress"), "", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(300)

        def _on_progress(done: int, total: int) -> None:
            progress.setMaximum(max(total, 1))
            progress.setValue(done)
            QApplication.processEvents()

        try:
            out = export_scene(
                self.tree, path, ext,
                scale=float(scale.currentData()), quality=quality.value(),
                show_markers=markers.isChecked(), progress=_on_progress,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("Tree Studio export failed")
            QMessageBox.warning(self, t("ts_export_title"), t("ts_export_failed", error=str(exc)))
            return
        finally:
            progress.close()
        self.statusBar().showMessage(t("ts_export_done", path=str(out)), 8000)
