"""Checksums for files and for partition membership (DR-001)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

_CHUNK_BYTES = 1 << 22


def file_sha256(path: Path) -> str:
    """Hexadecimal SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def id_set_sha256(ids: pd.Series) -> str:
    """SHA-256 of the identifiers, sorted ascending and serialised as little-endian int64.

    This reproduces the partition hashes recorded in DR-001 (notebook 01, §4.3).
    """
    values = np.sort(ids.to_numpy(dtype="int64")).astype("<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()
