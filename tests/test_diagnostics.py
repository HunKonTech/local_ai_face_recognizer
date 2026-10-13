"""Tests for runtime TFLite diagnostics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.diagnostics import TFLiteDiagnostics, check_tflite_backend, print_diagnostics


class TestTFLiteDiagnostics:
    def test_ok_true_when_active_backend_present(self) -> None:
        diag = TFLiteDiagnostics(
            python_version="3.11",
            platform="darwin",
            machine="arm64",
            backends=[],
            active_backend="ai_edge_litert",
            model_exists=True,
            model_path="/tmp/model.tflite",
            interpreter_ok=True,
            interpreter_error=None,
        )
        assert diag.ok is True

    def test_ok_false_without_backend(self) -> None:
        diag = TFLiteDiagnostics(
            python_version="3.11",
            platform="darwin",
            machine="arm64",
            backends=[],
            active_backend=None,
            model_exists=False,
            model_path="/tmp/model.tflite",
            interpreter_ok=False,
            interpreter_error=None,
        )
        assert diag.ok is False


class TestCheckTfliteBackend:
    def test_reports_missing_model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        missing = tmp_path / "missing.tflite"
        diag = check_tflite_backend(str(missing))
        assert diag.model_path == str(missing)
        assert diag.model_exists is False
        assert len(diag.backends) == 3

    def test_uses_first_available_backend_for_interpreter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = tmp_path / "model.tflite"
        model.write_bytes(b"fake")

        interp = MagicMock()
        fake_cls = MagicMock(return_value=interp)

        import builtins

        real_import = builtins.__import__

        def _import(name, *args, **kwargs):
            if name == "ai_edge_litert.interpreter":
                mod = MagicMock()
                mod.Interpreter = fake_cls
                return mod
            if name == "ai_edge_litert":
                pkg = MagicMock()
                pkg.__version__ = "1.2.3"
                return pkg
            if name == "tflite_runtime.interpreter":
                raise ImportError("skip runtime")
            if name == "tflite_runtime":
                raise ImportError("skip runtime")
            if name == "tensorflow":
                raise ImportError("skip tf")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _import)
        diag = check_tflite_backend(str(model))
        assert diag.active_backend == "ai_edge_litert"
        assert diag.interpreter_ok is True
        fake_cls.assert_called_once_with(model_path=str(model))
        interp.allocate_tensors.assert_called_once()


class TestPrintDiagnostics:
    def test_prints_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        diag = TFLiteDiagnostics(
            python_version="3.11",
            platform="darwin",
            machine="arm64",
            backends=[],
            active_backend=None,
            model_exists=False,
            model_path="/tmp/model.tflite",
            interpreter_ok=False,
            interpreter_error=None,
        )
        print_diagnostics(diag)
        out = capsys.readouterr().out
        assert "TFLite diagnostics" in out
        assert "ACTION REQUIRED" in out


class TestDetectionComponentSelfTest:
    """``Face-Local --self-test`` — the release build's bundling guard."""

    def test_reports_haar_cascades_and_models(self, monkeypatch) -> None:
        from app import diagnostics

        monkeypatch.setattr(diagnostics, "_probe_insightface", lambda: None)
        report = diagnostics.check_detection_components()
        assert set(report["haar_cascades"]) == set(diagnostics.HAAR_CASCADES)
        assert all(report["haar_cascades"].values())  # pip OpenCV ships them
        assert all(report["models"].values())

    def test_write_self_test_exit_code_follows_ok(self, monkeypatch, tmp_path) -> None:
        import json

        from app import diagnostics

        out = tmp_path / "selftest.json"
        monkeypatch.setattr(diagnostics, "check_detection_components", lambda: {"ok": True})
        assert diagnostics.write_self_test(str(out)) == 0
        assert json.loads(out.read_text(encoding="utf-8")) == {"ok": True}
        monkeypatch.setattr(
            diagnostics, "check_detection_components", lambda: {"ok": False}
        )
        assert diagnostics.write_self_test(str(out)) == 1

    def test_failed_insightface_probe_fails_the_self_test(self, monkeypatch) -> None:
        from app import diagnostics

        monkeypatch.setattr(diagnostics, "_probe_insightface", lambda: "error: boom")
        report = diagnostics.check_detection_components()
        if report["insightface"] is not None:
            assert report["ok"] is False


def test_main_self_test_flag_exits_with_report(monkeypatch, tmp_path) -> None:
    import sys

    from app import diagnostics, main

    out = tmp_path / "r.json"
    monkeypatch.setattr(sys, "argv", ["face-local", "--self-test", str(out)])
    monkeypatch.setattr(diagnostics, "check_detection_components", lambda: {"ok": True})
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0
    assert out.exists()
