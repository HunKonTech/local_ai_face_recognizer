"""Tree Studio entry point (design D1): shared bootstrap, same DB as Face-Local."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from app.bootstrap import add_common_args, create_qapplication, init_logging, load_app_config

log = logging.getLogger(__name__)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tree Studio — compose family tree views")
    add_common_args(parser)
    parser.add_argument("--tree-studio", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--view", metavar="NAME", default=None, help="Open a saved view by name")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    init_logging(getattr(args, "debug", False), "Tree Studio")
    config = load_app_config(getattr(args, "config", None), getattr(args, "db", None))
    app = create_qapplication("Tree Studio")

    from app.db.database import init_db, session_scope

    init_db(config.db_path_resolved)

    from app.tree_studio.window import TreeStudioWindow

    window = TreeStudioWindow(config=config)
    view_name = getattr(args, "view", None)
    if view_name:
        from app.services.tree_view_service import TreeViewService

        with session_scope() as session:
            match = [v.id for v in TreeViewService(session).list_views() if v.name == view_name]
        if match:
            window.open_view(match[0])
    window.show()
    log.info("Tree Studio ready")
    return app.exec()


def main(argv: Optional[list[str]] = None) -> int:
    return run(parse_args(argv))


def launch_command(db_path: Optional[str] = None, config_path: Optional[str] = None) -> list[str]:
    """Command line that starts Tree Studio as a separate process."""
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--tree-studio"]
    else:
        cmd = [sys.executable, "-m", "app.tree_studio"]
    if db_path:
        cmd += ["--db", str(db_path)]
    if config_path:
        cmd += ["--config", str(config_path)]
    return cmd
