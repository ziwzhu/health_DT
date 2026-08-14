"""Logging configuration.

Each module does ``logger = logging.getLogger(__name__)``. The run scripts call
:func:`setup` once. Using logging rather than print means output can be
silenced, redirected to a file, or filtered by level without touching the
analysis code.
"""

import logging
import sys
from pathlib import Path

_seen: set[str] = set()


def setup(level: int = logging.INFO, log_file: str | Path | None = None) -> None:
    """Configure root logging for a pipeline run."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=handlers,
        force=True,
    )


def banner(logger: logging.Logger, title: str) -> None:
    """Log a section header."""
    logger.info("\n%s\n%s\n%s", "=" * 70, title, "=" * 70)


def warn_once(logger: logging.Logger, key: str, message: str, *args) -> None:
    """Log a warning the first time only.

    For conditions inside a replication loop: the reader needs to know once,
    not once per simulation.
    """
    if key in _seen:
        return
    _seen.add(key)
    logger.warning(message, *args)
