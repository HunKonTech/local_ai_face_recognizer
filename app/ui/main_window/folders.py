"""Source folder selection and persistence."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Slot

from app.ui.dialogs.scan_modes_dialog import ScanModesDialog
from app.ui.i18n import t

log = logging.getLogger(__name__)


class FoldersMixin:
    """Source folder selection and persistence."""

    def _set_folder_label(self, folders) -> None:
        """Show folder count (or single path) in the toolbar."""
        from PySide6.QtGui import QFontMetrics

        if isinstance(folders, str):
            folders = [folders]
        if len(folders) == 1:
            path = folders[0]
            self._folder_label.setToolTip(path)
            metrics = QFontMetrics(self._folder_label.font())
            elided = metrics.elidedText(path, Qt.ElideMiddle, 360)
            self._folder_label.setText(f"  {elided}")
        else:
            tooltip = "\n".join(folders)
            self._folder_label.setToolTip(tooltip)
            self._folder_label.setText(
                f"  {t('folders_count', n=len(folders))}"
            )

    def _restore_last_folder(self) -> None:
        """Re-select the folder(s) used in the previous session."""
        import json

        from app.app_settings import app_qsettings

        qs = app_qsettings()
        # New key: JSON list of paths.
        raw = qs.value("paths/source_folders", None)
        if raw:
            try:
                folders = json.loads(raw) if isinstance(raw, str) else list(raw)
            except Exception:
                folders = []
            folders = [f for f in folders if Path(f).is_dir()]
            if folders:
                self._root_folders = folders
                self._set_folder_label(folders)
                self._scan_modes_btn.setEnabled(True)
                log.info("Restored source folders: %s", folders)
                return
        # Legacy fallback: single path key.
        last = qs.value("paths/last_folder", "", type=str)
        if last and Path(last).is_dir():
            self._root_folders = [last]
            self._set_folder_label([last])
            self._scan_modes_btn.setEnabled(True)
            log.info("Restored last folder (legacy): %s", last)

    def _save_folders(self, folders) -> None:
        import json

        from app.app_settings import app_qsettings

        app_qsettings().setValue("paths/source_folders", json.dumps(folders))

    @Slot()
    def _on_open_scan_modes(self) -> None:
        dlg = ScanModesDialog(parent=self, config=self._config)
        dlg.scan_workflow_started.connect(self._on_scan_workflow_started)
        dlg.maintenance_action_started.connect(self._on_scan_maintenance_action)
        dlg.reset_unknown_requested.connect(self._on_reset_unknown_persons)
        dlg.exec()
