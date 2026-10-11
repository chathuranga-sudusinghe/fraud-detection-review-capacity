"""Shared pytest fixtures; helper code lives in ``tests/support``."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

from support.synthetic import Synthetic, build_synthetic


@pytest.fixture
def synthetic(tmp_path: Path) -> Synthetic:
    """Synthetic raw files and matching specifications in a temporary project root."""
    return build_synthetic(tmp_path)


@pytest.fixture
def restore_logging() -> Iterator[None]:
    """Restore the logging state that a test changes with ``configure_logging``.

    Saves the handlers, level and propagation of the configured loggers and the warning hook,
    then puts them back, closing only the handlers the test added.
    """
    names = ("review_capacity", "py.warnings")
    saved = {
        name: (list(logging.getLogger(name).handlers), logging.getLogger(name).level,
               logging.getLogger(name).propagate)
        for name in names
    }  # fmt: skip
    showwarning = warnings.showwarning
    yield
    logging.captureWarnings(False)  # clears logging's own record of the hook it installed
    warnings.showwarning = showwarning
    for name, (handlers, level, propagate) in saved.items():
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            if handler not in handlers:
                handler.close()
        for handler in handlers:
            logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate
