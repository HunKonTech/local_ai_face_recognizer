"""Structured logging configuration for face-local.

Sets up a root logger that writes structured lines to stderr and,
optionally, to a rotating file.  GUI panels can attach a
:class:`QLogHandler` to receive log records through Qt signals.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """Configure the root logger.

    Call once at application startup before any module imports that log.

    Args:
        level:    Minimum log level (e.g. ``logging.DEBUG``).
        log_file: Optional path to a rotating log file.  ``None`` → no file.
    """
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers if called more than once (e.g. in tests)
    if root.handlers:
        return

    # --- stderr handler -------------------------------------------------
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # --- file handler ---------------------------------------------------
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Quiet noisy third-party loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


class QLogHandler(logging.Handler):
    """A :class:`logging.Handler` that forwards records to a Qt signal.

    Attach this handler to the root logger and connect ``log_emitted``
    to a slot in the GUI log panel.

    Usage::

        handler = QLogHandler(signal=main_window.log_signal)
        logging.getLogger().addHandler(handler)
    """

    def __init__(self, signal) -> None:
        super().__init__()
        self._signal = signal
        self.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._signal.emit(msg, record.levelno)
        except Exception:  # noqa: BLE001
            self.handleError(record)
