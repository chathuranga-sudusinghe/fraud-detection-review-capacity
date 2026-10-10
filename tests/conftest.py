"""Shared pytest fixtures; helper code lives in ``tests/support``."""

from __future__ import annotations

from pathlib import Path

import pytest

from support.synthetic import Synthetic, build_synthetic


@pytest.fixture
def synthetic(tmp_path: Path) -> Synthetic:
    """Synthetic raw files and matching specifications in a temporary project root."""
    return build_synthetic(tmp_path)
