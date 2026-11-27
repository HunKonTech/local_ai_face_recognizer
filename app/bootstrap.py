"""Shared start-up steps for Face-Local and Tree Studio.

Both apps parse the same ``--config`` / ``--db`` / ``--debug`` options, log to
the same file, load the same config and DB, and build a themed
``QApplication``. Face-Local adds its own extras (crash diagnostics, Google
Drive, AI warm-up) in :mod:`app.main`.
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.paths import app_icon_path, default_log_file

log = logging.getLogger(__name__)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config", metavar="PATH", default=None,
        help="Path to config.yaml (default: auto-discover)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging")
    parser.add_argument(
        "--db", metavar="PATH", default=None,
        help="Override database path (e.g. --db /tmp/test.db)",
    )
    return parser


def init_logging(debug: bool, app_label: str) -> None:
    from app.logging_setup import setup_logging

    setup_logging(
        level=logging.DEBUG if debug else logging.INFO,
        log_file=str(default_log_file()),
    )
    logging.getLogger(__name__).info("Starting %s", app_label)


def load_app_config(config_path, db_override):  # noqa: ANN001, ANN201
    """Load config.yaml, apply ``--db``, create the data directories."""
    from app.config import load_config

    config = load_config(config_path)
    if db_override:
        config.storage.db_path = db_override
    log.info("Config loaded — DB: %s", config.db_path_resolved)
    config.db_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    config.crops_dir_resolved.mkdir(parents=True, exist_ok=True)
    return config


def create_qapplication(app_name: str):  # noqa: ANN201
    """Language prefs, ``QApplication``, main-thread GC, icon, theme, settings."""
    from app.ui.i18n import load_prefs

    load_prefs()

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    # QtWebEngine (Places map) requires shared OpenGL contexts before the
    # QApplication exists.
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(app_name)
    app.setOrganizationName("face-local")

    # Route cyclic GC onto the main thread (see app.ui.gc_guard).
    from app.ui.gc_guard import install_main_thread_gc

    install_main_thread_gc()
    icon_path = app_icon_path()
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    from app.ui.theme import apply_theme

    apply_theme(app)

    from app.app_settings import migrate_legacy_settings

    migrate_legacy_settings()

    from app.services.family_code_schemes import FamilyCodeSchemeStore

    try:
        active = FamilyCodeSchemeStore().load_active_into_runtime()
        log.info("Active family code scheme: %s", active.name)
    except Exception:  # noqa: BLE001
        log.exception("Could not load the active family code scheme")
    return app
