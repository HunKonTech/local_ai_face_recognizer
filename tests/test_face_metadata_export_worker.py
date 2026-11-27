"""Tests for FaceMetadataExportWorker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.jobs.cancellation import CancellationToken
from app.services.face_metadata_export_service import (
    FaceMetadataExportOptions,
    FaceMetadataExportSummary,
)
from app.workers.face_metadata_export_worker import FaceMetadataExportWorker


class TestFaceMetadataExportWorker:
    def test_emits_summary_on_success(self, qtbot):
        from app.services.face_metadata_export_service import FaceMetadataExportResult
        from app.utils import image_metadata as meta

        summary = FaceMetadataExportSummary(
            results=[
                FaceMetadataExportResult(1, "a.jpg", True, meta.WRITE_MODE_XMP),
                FaceMetadataExportResult(2, "busy.jpg", True, meta.WRITE_MODE_LOCKED),
            ],
            requested_total=2,
        )
        token = CancellationToken()
        options = FaceMetadataExportOptions()

        with patch(
            "app.workers.face_metadata_export_worker.FaceMetadataExportService"
        ) as SvcCls, patch(
            "app.workers.face_metadata_export_worker.session_scope",
            lambda: MagicMock(__enter__=lambda s: object(), __exit__=MagicMock()),
        ):
            SvcCls.return_value.export_all.return_value = summary
            worker = FaceMetadataExportWorker(options, token)
            with qtbot.waitSignal(worker.finished_ok, timeout=3000) as blocker:
                worker.run()
        assert blocker.args[0] is summary
        # The locked outcome reaches the UI unchanged.
        assert blocker.args[0].locked_count == 1
        assert blocker.args[0].locked_paths == ["busy.jpg"]

    def test_reports_progress(self, qtbot):
        token = CancellationToken()
        options = FaceMetadataExportOptions()
        captured = []

        def _export_all(_opts, progress_cb=None, cancel_token=None):
            if progress_cb:
                progress_cb(1, 3, "photo.jpg")
            return FaceMetadataExportSummary()

        with patch(
            "app.workers.face_metadata_export_worker.FaceMetadataExportService"
        ) as SvcCls, patch(
            "app.workers.face_metadata_export_worker.session_scope",
            lambda: MagicMock(__enter__=lambda s: object(), __exit__=MagicMock()),
        ):
            SvcCls.return_value.export_all.side_effect = _export_all
            worker = FaceMetadataExportWorker(options, token)
            worker.progress.connect(lambda *a: captured.append(a))
            worker.run()

        assert captured == [(1, 3, "photo.jpg")]

    def test_emits_failed_on_crash(self, qtbot):
        token = CancellationToken()
        options = FaceMetadataExportOptions()
        with patch(
            "app.workers.face_metadata_export_worker.session_scope",
            side_effect=RuntimeError("db locked"),
        ):
            worker = FaceMetadataExportWorker(options, token)
            with qtbot.waitSignal(worker.failed, timeout=3000) as blocker:
                worker.run()
        assert "db locked" in blocker.args[0]

    def test_cancel_delegates_to_token(self):
        token = CancellationToken()
        worker = FaceMetadataExportWorker(FaceMetadataExportOptions(), token)
        worker.cancel()
        assert token.cancelled
