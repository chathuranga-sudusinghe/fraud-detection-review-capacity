"""Run logging: progress, timing and diagnostics with timestamps.

Library modules only create loggers with ``logging.getLogger(__name__)`` and never configure
logging. The command line configures it once with ``configure_logging``: every message goes to the
terminal (standard error) and, optionally, to a log file kept with the run's evidence. The log
file keeps the traceback and cause of every logged exception; the terminal shows them only when
asked (``terminal_tracebacks``), so that an expected failure reads as one line there.

Two tools keep a long run visible:

* ``log_step`` logs when a step starts, how long it took, or that it failed or was interrupted;
* ``ProgressLogger`` logs bounded periodic progress inside a loop (items done of the total,
  elapsed time and an estimate of the time left), for example over columns, folds, trials or
  training epochs.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from review_capacity.errors import FileAccessError

PACKAGE_LOGGER = "review_capacity"
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
CONFIGURED_LOGGERS = (PACKAGE_LOGGER, "py.warnings")


class _ConciseFormatter(logging.Formatter):
    """Formats a record without its traceback or stack (terminal output)."""

    def format(self, record: logging.LogRecord) -> str:
        saved = record.exc_info, record.exc_text, record.stack_info
        record.exc_info, record.exc_text, record.stack_info = None, None, None
        try:
            return super().format(record)
        finally:
            record.exc_info, record.exc_text, record.stack_info = saved


def _install(handlers: list[logging.Handler], level: int) -> None:
    for name in CONFIGURED_LOGGERS:
        logger = logging.getLogger(name)
        for old in list(logger.handlers):
            logger.removeHandler(old)
            old.close()
        for handler in handlers:
            logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    *,
    terminal_tracebacks: bool = False,
    file_level: int | None = None,
    new_file: bool = False,
) -> None:
    """Send package messages, and Python warnings, to standard error and an optional file.

    ``level`` applies to standard error; ``file_level`` (``level`` when not given) to the file, so
    that a run's evidence log can keep every message while the terminal stays quiet. With
    ``new_file`` the log file is created, never appended to: an existing file is an error.
    The file always receives tracebacks; standard error only with ``terminal_tracebacks``.
    If the log file cannot be opened, logging to standard error is kept and ``FileAccessError``
    is raised, so that the caller can report it and stop.
    """
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    terminal = logging.StreamHandler(sys.stderr)
    terminal.setFormatter(
        formatter if terminal_tracebacks else _ConciseFormatter(LOG_FORMAT, DATE_FORMAT)
    )
    terminal.setLevel(level)
    handlers: list[logging.Handler] = [terminal]
    file_threshold = level if file_level is None else file_level
    logging.captureWarnings(True)
    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            mode = "x" if new_file else "a"
            file_handler = logging.FileHandler(log_file, mode=mode, encoding="utf-8")
        except OSError as error:
            _install(handlers, min(level, file_threshold))
            raise FileAccessError(f"Log file {log_file} cannot be opened: {error}") from error
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_threshold)
        handlers.append(file_handler)
    _install(handlers, min(level, file_threshold))


@contextmanager
def log_step(logger: logging.Logger, step: str) -> Iterator[None]:
    """Log the start of ``step`` and its duration, or its failure or interruption; re-raise."""
    logger.info("%s ...", step)
    start = time.perf_counter()
    try:
        yield
    except KeyboardInterrupt:
        logger.error("%s: interrupted after %.1f s", step, time.perf_counter() - start)
        raise
    except BaseException:
        logger.error("%s: failed after %.1f s", step, time.perf_counter() - start)
        raise
    logger.info("%s: done in %.1f s", step, time.perf_counter() - start)


class ProgressLogger:
    """Bounded periodic progress for a loop of known length.

    Call ``advance()`` after each item. A message is logged when at least ``every_seconds`` have
    passed since the last message, and always for the last item, so a long loop reports regularly
    without flooding the log.
    """

    def __init__(
        self,
        logger: logging.Logger,
        task: str,
        total: int,
        *,
        every_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.logger = logger
        self.task = task
        self.total = total
        self.every_seconds = every_seconds
        self.clock = clock
        self.done = 0
        self.start = clock()
        self.last_report = self.start

    def advance(self, items: int = 1) -> None:
        self.done += items
        now = self.clock()
        if self.done < self.total and now - self.last_report < self.every_seconds:
            return
        self.last_report = now
        elapsed = now - self.start
        left = elapsed / self.done * (self.total - self.done) if self.done else float("nan")
        self.logger.info(
            "%s: %d/%d (%.0f %%), elapsed %.1f s, about %.1f s left",
            self.task,
            self.done,
            self.total,
            100 * self.done / self.total if self.total else 100.0,
            elapsed,
            left,
        )
