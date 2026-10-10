"""Value checks of the data validation contract, on synthetic files."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from review_capacity.data.loading import load_raw
from review_capacity.data.validation import check_file_checksums, validate
from review_capacity.errors import DataValidationError
from support.synthetic import Synthetic, build_synthetic, with_partition


def test_contract_passes_and_reproduces_anchors(synthetic: Synthetic) -> None:
    raw = load_raw(synthetic.features, synthetic.dataset)
    summaries = validate(raw, synthetic.features, synthetic.dataset)
    assert [(s.name, s.rows) for s in summaries] == [
        ("train", 8),
        ("v1", 4),
        ("v2", 4),
        ("test", 4),
    ]
    for summary, part in zip(summaries, synthetic.dataset.partitions, strict=True):
        assert summary.transaction_id_sha256.startswith(part.transaction_id_sha256_prefix)


def test_checksum_mismatch_stops(synthetic: Synthetic) -> None:
    check_file_checksums(synthetic.dataset)
    tampered = replace(
        synthetic.dataset, identity=replace(synthetic.dataset.identity, sha256="0" * 64)
    )
    with pytest.raises(DataValidationError, match="SHA-256"):
        check_file_checksums(tampered)


@pytest.mark.parametrize(
    "changes",
    [{"rows": 9}, {"transaction_dt_max": 1}, {"transaction_id_sha256_prefix": "0" * 16}],
)
def test_anchor_mismatch_stops(synthetic: Synthetic, changes: dict[str, object]) -> None:
    raw = load_raw(synthetic.features, synthetic.dataset)
    dataset = with_partition(synthetic.dataset, "v1", **changes)
    with pytest.raises(DataValidationError, match="anchors not reproduced"):
        validate(raw, synthetic.features, dataset)


def test_fractional_numeric_code_stops(tmp_path: Path) -> None:
    data = build_synthetic(
        tmp_path, edit_transaction=lambda f: f.assign(card1=[100.5, *f["card1"].iloc[1:]])
    )
    raw = load_raw(data.features, data.dataset)
    with pytest.raises(DataValidationError, match="card1: 1 values are not whole numbers"):
        validate(raw, data.features, data.dataset)


def test_reserved_label_in_raw_text_stops(tmp_path: Path) -> None:
    data = build_synthetic(
        tmp_path,
        edit_transaction=lambda f: f.assign(ProductCD=["__MISSING__", *f["ProductCD"].iloc[1:]]),
    )
    raw = load_raw(data.features, data.dataset)
    with pytest.raises(DataValidationError, match="reserved label"):
        validate(raw, data.features, data.dataset)


def test_transaction_id_changes_fail_partition_hash_check(tmp_path: Path) -> None:
    def renumber(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(TransactionID=frame["TransactionID"] + 1)

    data = build_synthetic(tmp_path, edit_transaction=renumber, edit_identity=renumber)
    raw = load_raw(data.features, data.dataset)
    with pytest.raises(DataValidationError, match="anchors not reproduced"):
        validate(raw, data.features, data.dataset)


def test_numeric_code_beyond_exact_range_stops(tmp_path: Path) -> None:
    data = build_synthetic(
        tmp_path, edit_transaction=lambda f: f.assign(card1=[1e20, *f["card1"].iloc[1:]])
    )
    raw = load_raw(data.features, data.dataset)
    with pytest.raises(DataValidationError, match=r"card1: 1 values exceed"):
        validate(raw, data.features, data.dataset)
