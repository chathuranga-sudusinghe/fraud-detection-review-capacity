"""File checksums and the DR-001 partition-hash convention."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from review_capacity.hashing import file_sha256, id_set_sha256


def test_file_sha256_matches_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    assert file_sha256(path) == hashlib.sha256(b"abc").hexdigest()


def test_id_hash_is_order_invariant_and_uses_little_endian_int64() -> None:
    ids = pd.Series([3_000_005, 2_987_000, 3_100_000])
    expected = hashlib.sha256(np.array(sorted(ids), dtype="<i8").tobytes()).hexdigest()
    assert id_set_sha256(ids) == expected
    assert id_set_sha256(ids.iloc[::-1]) == expected
