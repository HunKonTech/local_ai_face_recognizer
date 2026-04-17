"""Application entry point.

Usage::

    python -m app.main                      # default config
    python -m app.main --config config.yaml # explicit config
    python -m app.main --debug              # verbose logging
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face-Local — offline face grouping and person labeling"
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to config.yaml (default: auto-discover)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        default=None,
        help="Override database path (e.g. --db /tmp/test.db)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Logging (before any other imports that log) ---
    from app.logging_setup import setup_logging

    setup_logging(
        level=logging.DEBUG if args.debug else logging.INFO,
        log_file="data/face_local.log",
    )

    log = logging.getLogger(__name__)
    log.info("Starting Face-Local")

    # --- Config ---
    from app.config import load_config

    config = load_config(args.config)

    if args.db:
        config.storage.db_path = args.db

    log.info("Config loaded — DB: %s", config.db_path_resolved)
    log.info("Crops dir:          %s", config.crops_dir_resolved)

    # Ensure data directories exist
    config.db_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    config.crops_dir_resolved.mkdir(parents=True, exist_ok=True)

    # --- Language preferences ---
    from app.ui.i18n import load_prefs
    load_prefs()

    # --- Qt application ---
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QPalette, QColor
    from PySide6.QtCore import Qt

    app = QApplication(sys.argv)
    app.setApplicationName("Face-Local")
    app.setOrganizationName("face-local")

    # Dark palette
    _apply_dark_palette(app)

    from app.ui.main_window import MainWindow

    window = MainWindow(config=config)
    window.show()

    log.info("GUI ready")
    sys.exit(app.exec())


def _apply_dark_palette(app) -> None:  # noqa: ANN001
    """Apply a system-consistent dark color palette."""
    from PySide6.QtGui import QPalette, QColor
    from PySide6.QtCore import Qt

    palette = QPalette()
    dark = QColor(45, 45, 48)
    mid_dark = QColor(60, 60, 63)
    light = QColor(210, 210, 215)
    highlight = QColor(86, 138, 242)
    link = QColor(100, 160, 255)

    palette.setColor(QPalette.Window, dark)
    palette.setColor(QPalette.WindowText, light)
    palette.setColor(QPalette.Base, QColor(30, 30, 32))
    palette.setColor(QPalette.AlternateBase, mid_dark)
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, light)
    palette.setColor(QPalette.Button, mid_dark)
    palette.setColor(QPalette.ButtonText, light)
    palette.setColor(QPalette.BrightText, QColor(255, 80, 80))
    palette.setColor(QPalette.Highlight, highlight)
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.Link, link)
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(120, 120, 120))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 120, 120))

    app.setPalette(palette)


if __name__ == "__main__":
    main()
