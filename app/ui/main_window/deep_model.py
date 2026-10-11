"""AI visualization windows, face debug and deep model import/export."""

from __future__ import annotations

import logging

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QMessageBox,
)

from app.db.database import session_scope
from app.ui.i18n import t

log = logging.getLogger(__name__)


class DeepModelMixin:
    """AI visualization windows, face debug and deep model import/export."""

    # ------------------------------------------------------------------
    # AI visualization window
    # ------------------------------------------------------------------

    def _open_ai_viz_window(self) -> None:
        from app.ui.dialogs.ai_visualization_window import AIVisualizationWindow
        if not hasattr(self, "_ai_viz_window") or self._ai_viz_window is None:
            self._ai_viz_window = AIVisualizationWindow(parent=self)
        self._ai_viz_window.show()
        self._ai_viz_window.raise_()

    def _open_nn_graph_window(self) -> None:
        from app.ui.dialogs.ai_visualization_window import NeuralNetworkGraphDialog
        if not hasattr(self, "_nn_graph_window") or self._nn_graph_window is None:
            self._nn_graph_window = NeuralNetworkGraphDialog(parent=self)
        # Refresh against the CURRENT trained model so the graph reflects the
        # live set of persons, not an old model whose people no longer exist.
        # Falls back to the AI-viz live history; if neither is available, the
        # graph is cleared to its empty state rather than showing a stale run.
        info = self._current_model_debug_info()
        if info is None and (
            hasattr(self, "_ai_viz_window")
            and self._ai_viz_window is not None
            and self._ai_viz_window._history
        ):
            info = self._ai_viz_window._history[-1]
        if info is not None:
            self._nn_graph_window.update_from_info(info)
        else:
            self._nn_graph_window.clear()
        self._nn_graph_window.show()
        self._nn_graph_window.raise_()

    def _current_model_debug_info(self):
        """A DeepDebugInfo for a representative face on the current trained model.

        Returns None (never raises) when there is no model or no embedded face —
        the caller then clears the neural-net graph to its empty state.
        """
        try:
            from app.services.deep_recognition_service import DeepRecognitionService

            with session_scope() as session:
                svc = DeepRecognitionService(
                    session=session,
                    config=self._config.deep_recognition,
                    model_dir=self._config.resolve(
                        self._config.deep_recognition.model_dir
                    ),
                )
                return svc.debug_sample_face()
        except Exception:  # noqa: BLE001
            log.warning("Could not refresh neural-net graph from current model", exc_info=True)
            return None

    @Slot(object)
    def _on_face_debug(self, info: object) -> None:
        if hasattr(self, "_ai_viz_window") and self._ai_viz_window is not None:
            self._ai_viz_window.update_info(info)
        if (
            hasattr(self, "_nn_graph_window")
            and self._nn_graph_window is not None
            and self._nn_graph_window.isVisible()
        ):
            self._nn_graph_window.update_from_info(info)

    # ------------------------------------------------------------------
    # Deep learning model export / import
    # ------------------------------------------------------------------

    def _on_export_deep_model(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from app.db.database import session_scope
        from app.services.deep_recognition_service import DeepRecognitionService

        path, _ = QFileDialog.getSaveFileName(
            self,
            t("deep_model_export_dialog"),
            "face_recognition_model.facemodel",
            "Face model (*.facemodel);;All files (*)",
        )
        if not path:
            return
        with session_scope() as session:
            svc = DeepRecognitionService(
                session=session,
                config=self._config.deep_recognition,
                model_dir=self._config.resolve(self._config.deep_recognition.model_dir),
            )
            ok = svc.export_model(path)
        if ok:
            QMessageBox.information(
                self, t("deep_model_title"), t("deep_model_export_ok", path=path)
            )
        else:
            QMessageBox.warning(
                self, t("deep_model_title"), t("deep_model_export_none")
            )

    def _on_import_deep_model(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from app.db.database import session_scope
        from app.services.deep_recognition_service import DeepRecognitionService

        path, _ = QFileDialog.getOpenFileName(
            self,
            t("deep_model_import_dialog"),
            "",
            "Face model (*.facemodel);;All files (*)",
        )
        if not path:
            return
        with session_scope() as session:
            svc = DeepRecognitionService(
                session=session,
                config=self._config.deep_recognition,
                model_dir=self._config.resolve(self._config.deep_recognition.model_dir),
            )
            ok = svc.import_model(path)
        if ok:
            QMessageBox.information(
                self, t("deep_model_title"), t("deep_model_import_ok")
            )
        else:
            QMessageBox.warning(
                self, t("deep_model_title"), t("deep_model_import_err")
            )
