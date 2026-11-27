"""Application entry point.

Usage::

    python -m app.main                      # default config
    python -m app.main --config config.yaml # explicit config
    python -m app.main --debug              # verbose logging
    python -m app.main --tree-studio        # open Tree Studio instead
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.bootstrap import (
    add_common_args,
    create_qapplication,
    init_logging,
    load_app_config,
)
from app.paths import default_log_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face-Local — offline face grouping and person labeling"
    )
    add_common_args(parser)
    parser.add_argument(
        "--tree-studio",
        action="store_true",
        help="Start Tree Studio (family tree composer) instead of Face-Local",
    )
    parser.add_argument(
        "--self-test",
        metavar="JSON",
        default=None,
        help="Write the detection-component report to JSON and exit "
        "(exit code 1 when a component is missing; used by the release build)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.self_test:
        from app.diagnostics import write_self_test

        sys.exit(write_self_test(args.self_test))

    if args.tree_studio:
        from app.tree_studio.main import run

        sys.exit(run(args))

    # --- Logging (before any other imports that log) ---
    init_logging(args.debug, "Face-Local")
    log = logging.getLogger(__name__)

    # --- Crash & resource diagnostics (observability only) ---
    # A scan can drive the machine into a memory/CPU-exhaustion hard freeze that
    # leaves no Python traceback.  Enable faulthandler + exception hooks as early
    # as possible, and start a lightweight resource watchdog that records the
    # RAM/CPU run-up and snapshots all thread stacks just before a likely freeze.
    try:
        from app.crash_diagnostics import (
            install_crash_handlers,
            start_resource_watchdog,
        )

        _log_dir = default_log_file().parent
        install_crash_handlers(_log_dir)
        start_resource_watchdog(_log_dir)
    except Exception:  # noqa: BLE001 — diagnostics must never block startup
        log.exception("Could not install crash/resource diagnostics")

    # --- Load .env (Google OAuth secrets, etc.) ---
    # Must run BEFORE app.gdrive.oauth_config is imported, so the module
    # picks up the values from the environment.  .env is in .gitignore.
    from pathlib import Path as _P

    from app.gdrive.dotenv_loader import load_dotenv
    from app.paths import project_root
    # Look in both the current working directory and the project root —
    # whichever exists wins, with cwd taking priority.
    for _candidate in (_P.cwd() / ".env", project_root() / ".env"):
        if load_dotenv(_candidate) > 0:
            break

    # --- Config ---
    config = load_app_config(args.config, args.db)
    log.info("Crops dir:          %s", config.crops_dir_resolved)

    # Google Drive cache — remove any leftovers from a previous (possibly crashed) run
    from app.gdrive.cache import get_cache_manager
    _gdrive_cache = get_cache_manager()
    _gdrive_cache.startup_cleanup()

    # --- Qt application (language prefs, theme, settings, code scheme) ---
    app = create_qapplication("Face-Local")

    from app.ui.main_window import MainWindow

    window = MainWindow(config=config)
    window.show()

    # Warm up onnxruntime on the main thread once the window is up, so its heavy
    # C-extension import (and any GC it provokes) never first runs inside a
    # background worker.  Deferred via the event loop so it doesn't block show().
    from PySide6.QtCore import QTimer

    from app.detectors.factory import warm_up_onnxruntime
    QTimer.singleShot(0, lambda: warm_up_onnxruntime(config.detection))

    # Background install of optional AI packages (ai-edge-litert, insightface).
    # Runs after the event loop starts so it never blocks startup.
    from app.workers.dependency_installer_worker import DependencyInstallerWorker
    _dep_worker = DependencyInstallerWorker()
    _dep_worker.package_installing.connect(window.on_dep_installing)
    _dep_worker.package_installed.connect(window.on_dep_installed)
    _dep_worker.install_failed.connect(window.on_dep_install_failed)
    _dep_worker.all_done.connect(window.on_dep_install_done)
    QTimer.singleShot(2000, _dep_worker.start)  # wait 2 s so startup scan doesn't race

    # Google Drive cache — clean up on normal exit
    app.aboutToQuit.connect(_gdrive_cache.shutdown_cleanup)

    log.info("GUI ready")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
