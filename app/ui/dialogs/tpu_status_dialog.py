"""TPU status & repair dialog."""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Dict, Any

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QProgressBar,
)

from app.ui.i18n import t

_EDGETPU_LIB = (
    "libedgetpu.1.dylib" if platform.system() == "Darwin" else "libedgetpu.so.1"
)


# ── Probe helpers ─────────────────────────────────────────────────────────────

def probe_tpu() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ai_edge_litert": False,
        "ai_edge_litert_ver": None,
        "pycoral": False,
        "pycoral_ver": None,
        "libedgetpu": False,
        "delegate_ok": False,
        "devices_pycoral": [],
        "error": None,
    }

    # ai-edge-litert
    try:
        import ai_edge_litert as ael  # type: ignore[import]
        result["ai_edge_litert"] = True
        result["ai_edge_litert_ver"] = getattr(ael, "__version__", "installed")
    except ImportError:
        pass

    # pycoral
    try:
        import pycoral  # type: ignore[import]
        result["pycoral"] = True
        result["pycoral_ver"] = getattr(pycoral, "__version__", "installed")
    except ImportError:
        pass

    # libedgetpu via delegate load
    if result["ai_edge_litert"]:
        try:
            from ai_edge_litert.interpreter import load_delegate  # type: ignore[import]
            load_delegate(_EDGETPU_LIB)
            result["libedgetpu"] = True
            result["delegate_ok"] = True
        except FileNotFoundError:
            result["error"] = f"libedgetpu shared library not found ({_EDGETPU_LIB})"
        except Exception as exc:
            result["error"] = str(exc)

    # pycoral device list (fallback)
    if result["pycoral"] and not result["delegate_ok"]:
        try:
            from pycoral.utils.edgetpu import list_edge_tpus  # type: ignore[import]
            result["devices_pycoral"] = list(list_edge_tpus())
            result["libedgetpu"] = True
            result["delegate_ok"] = bool(result["devices_pycoral"])
        except Exception as exc:
            if not result["error"]:
                result["error"] = str(exc)

    return result


def _os_install_commands() -> list[str]:
    """Return shell commands needed to fix a missing TPU setup."""
    cmds = []
    if platform.system() == "Darwin":
        cmds.append("# 1. Install libedgetpu (requires Homebrew)")
        cmds.append("brew install --cask coral-usb-accelerator")
    else:
        cmds.append("# 1. Install libedgetpu (Debian/Ubuntu)")
        cmds.append(
            "echo 'deb https://packages.cloud.google.com/apt coral-edgetpu-stable main'"
            " | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list"
        )
        cmds.append(
            "curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg"
            " | sudo apt-key add -"
        )
        cmds.append("sudo apt-get update && sudo apt-get install -y libedgetpu1-std")

    cmds.append("")
    cmds.append("# 2. Install ai-edge-litert (Python TPU runtime)")
    cmds.append(f"{sys.executable} -m pip install ai-edge-litert")
    return cmds


# ── Background installer ─────────────────────────────────────────────────────

class _InstallerThread(QThread):
    output = Signal(str)
    finished_ok = Signal(bool)

    def __init__(self, commands: list[str]) -> None:
        super().__init__()
        self._commands = commands

    def run(self) -> None:
        ok = True
        for cmd in self._commands:
            if cmd.startswith("#") or not cmd.strip():
                self.output.emit(f"\n{cmd}")
                continue
            self.output.emit(f"$ {cmd}")
            try:
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if proc.stdout.strip():
                    self.output.emit(proc.stdout.strip())
                if proc.returncode != 0:
                    self.output.emit(f"[ERROR] {proc.stderr.strip()}")
                    ok = False
            except subprocess.TimeoutExpired:
                self.output.emit("[ERROR] Command timed out")
                ok = False
            except Exception as exc:
                self.output.emit(f"[ERROR] {exc}")
                ok = False
        self.finished_ok.emit(ok)


# ── Dialog ───────────────────────────────────────────────────────────────────

class TpuStatusDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("tpu_title"))
        self.setMinimumWidth(520)
        self._thread: _InstallerThread | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        info = probe_tpu()
        tpu_ok = info["delegate_ok"]

        # ── Summary ───────────────────────────────────────────────────────
        summary = QLabel(t("tpu_ok_label") if tpu_ok else t("tpu_warn_label"))
        font = summary.font()
        font.setPointSize(13)
        font.setBold(True)
        summary.setFont(font)
        summary.setAlignment(Qt.AlignCenter)
        summary.setStyleSheet(f"color: {'#4caf50' if tpu_ok else '#f57c00'};")
        layout.addWidget(summary)

        # ── Details ────────────────────────────────────────────────────────
        lines: list[str] = []

        if info["ai_edge_litert"]:
            lines.append(f"✓ ai-edge-litert {info['ai_edge_litert_ver']}")
        else:
            lines.append("✗ ai-edge-litert: NINCS telepítve / not installed")

        if info["libedgetpu"]:
            lines.append(f"✓ {t('tpu_libedge_ok')}")
        else:
            lines.append(f"✗ {t('tpu_libedge_miss')}")

        if info["pycoral"]:
            lines.append(f"✓ pycoral {info['pycoral_ver']} (legacy)")

        lines.append("")
        lines.append(t("tpu_devices"))
        if info["devices_pycoral"]:
            for d in info["devices_pycoral"]:
                lines.append(f"  • {d}")
        elif info["delegate_ok"]:
            lines.append("  • EdgeTPU delegate loaded successfully")
        else:
            lines.append(f"  {t('tpu_none')}")

        if info["error"]:
            lines.append("")
            lines.append(t("tpu_error", msg=info["error"]))

        self._details = QTextEdit()
        self._details.setReadOnly(True)
        self._details.setMinimumHeight(150)
        self._details.setPlainText("\n".join(lines))
        layout.addWidget(self._details)

        # ── Fix section (shown only when not OK) ──────────────────────────
        if not tpu_ok:
            self._fix_btn = QPushButton("🔧 " + ("Javítás / Fix"))
            self._fix_btn.clicked.connect(self._on_fix)
            layout.addWidget(self._fix_btn)

            self._progress = QProgressBar()
            self._progress.setRange(0, 0)
            self._progress.setVisible(False)
            layout.addWidget(self._progress)

        # ── Buttons ────────────────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_fix(self) -> None:
        self._fix_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._details.append("\n─── Running fix commands ───\n")

        cmds = _os_install_commands()
        self._thread = _InstallerThread(cmds)
        self._thread.output.connect(lambda line: self._details.append(line))
        self._thread.finished_ok.connect(self._on_fix_done)
        self._thread.start()

    def _on_fix_done(self, ok: bool) -> None:
        self._progress.setVisible(False)
        if ok:
            self._details.append(
                "\n✓ Done! Restart the application and re-check TPU status."
            )
        else:
            self._details.append(
                "\n✗ Some commands failed. Check the output above."
            )
        self._fix_btn.setEnabled(True)
