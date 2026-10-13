"""The content-duplicate audit (Issue #86) on hand-built frames and synthetic processed files.

The synthetic dataset (``tests/support/synthetic.py``) has 16 development rows: Train rows 0-7,
Validation 1 rows 8-11 and Validation 2 rows 12-15. Without edits every row is unique (each has
its own ``TransactionAmt``). ``repeat_transaction_values`` and ``repeat_identity_values`` give rows
2, 4 and 8 the supplied values of row 0; rows 0, 4 and 8 are at the start of their relative day
and row 2 at noon, so the expected counts below follow by hand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from review_capacity.data.duplicates import (
    DEVELOPMENT,
    PRIMARY,
    SECONDARY,
    RepeatCounts,
    SharedVectors,
    audit_definition,
    audit_duplicates,
    audit_summary,
    repeat_counts,
    vector_ids,
)
from review_capacity.data.processed import DevelopmentFeatures, read_development_features
from review_capacity.errors import DataValidationError
from support.synthetic import (
    Synthetic,
    build_processed,
    repeat_identity_values,
    repeat_transaction_values,
)

PARTITIONS = ("train", "v1", "v2")


def _development(data: Synthetic) -> DevelopmentFeatures:
    return read_development_features(data.paths, data.features, data.dataset)


def _same(ids: npt.NDArray[np.int64], first: int, second: int) -> bool:
    return bool(ids[first] == ids[second])


def test_exact_matches_share_an_identifier() -> None:
    frame = pd.DataFrame({"a": ["x", "x", "y", "x"], "b": [1.0, 1.0, 1.0, 2.0]})
    ids = vector_ids(frame)
    assert _same(ids, 0, 1)
    assert not _same(ids, 0, 2)  # differs in a
    assert not _same(ids, 0, 3)  # differs in b


def test_missing_values_are_equal_and_negative_zero_is_zero() -> None:
    frame = pd.DataFrame(
        {
            "label": ["__MISSING__", "__MISSING__", "x"],
            "value": [np.nan, float("nan"), np.nan],
            "zero": [0.0, -0.0, 0.0],
        }
    )
    ids = vector_ids(frame)
    assert _same(ids, 0, 1)
    assert not _same(ids, 0, 2)


def test_a_missing_value_differs_from_a_present_value() -> None:
    frame = pd.DataFrame({"a": ["x", "x"], "value": [np.nan, 0.0]})
    ids = vector_ids(frame)
    assert not _same(ids, 0, 1)


def test_numbers_are_compared_without_rounding_or_tolerance() -> None:
    frame = pd.DataFrame({"value": [1.0, 1.0 + 1e-12, 1.0]})
    ids = vector_ids(frame)
    assert not _same(ids, 0, 1)
    assert _same(ids, 0, 2)


def test_combined_columns_do_not_collide() -> None:
    # (0, 1) and (1, 0) must differ although each column alone has the same values.
    frame = pd.DataFrame({"a": [0, 1, 0, 1], "b": [1, 0, 1, 0]})
    ids = vector_ids(frame)
    assert _same(ids, 0, 2)
    assert _same(ids, 1, 3)
    assert not _same(ids, 0, 1)


def test_repeat_counts_by_hand() -> None:
    # Vector 7 on three rows, vector 3 on two rows, vector 5 once.
    ids = np.array([7, 3, 7, 5, 3, 7], dtype=np.int64)
    assert repeat_counts(ids) == RepeatCounts(rows=6, groups=2, rows_in_groups=5, excess_rows=3)
    assert repeat_counts(np.array([], dtype=np.int64)) == RepeatCounts(0, 0, 0, 0)


def test_counts_within_and_across_partitions() -> None:
    frame = pd.DataFrame({"a": ["p", "p", "q", "p", "r", "q"]})
    partition = pd.Series(["train", "train", "train", "v1", "v2", "v2"])
    audit = audit_definition(PRIMARY, frame, partition, PARTITIONS)
    assert audit.within["train"] == RepeatCounts(3, 1, 2, 1)
    assert audit.within["v1"] == RepeatCounts(1, 0, 0, 0)
    assert audit.within["v2"] == RepeatCounts(2, 0, 0, 0)
    assert audit.within[DEVELOPMENT] == RepeatCounts(6, 2, 5, 3)
    assert audit.vectors_in_several_partitions == 2  # p (Train, V1) and q (Train, V2)
    assert audit.rows_with_such_vectors == 5
    assert audit.shared == (
        SharedVectors(("train", "v1"), 1),
        SharedVectors(("train", "v2"), 1),
        SharedVectors(("v1", "v2"), 0),
    )


def test_a_vector_in_all_three_partitions_is_counted_once() -> None:
    frame = pd.DataFrame({"a": ["p", "p", "p"]})
    partition = pd.Series(list(PARTITIONS))
    audit = audit_definition(PRIMARY, frame, partition, PARTITIONS)
    assert all(audit.within[name] == RepeatCounts(1, 0, 0, 0) for name in PARTITIONS)
    assert audit.within[DEVELOPMENT] == RepeatCounts(3, 1, 3, 2)
    assert (audit.vectors_in_several_partitions, audit.rows_with_such_vectors) == (1, 3)
    # Pairwise counts are inclusive: the one vector is shared by every pair (not additive).
    assert [s.vectors for s in audit.shared] == [1, 1, 1]


def test_no_repeats_in_the_unedited_synthetic_data(tmp_path: Path) -> None:
    data = build_processed(tmp_path)
    audit = audit_duplicates(_development(data), data.features, data.dataset)
    for definition in audit.definitions:
        assert definition.within[DEVELOPMENT] == RepeatCounts(16, 0, 0, 0)
        assert definition.vectors_in_several_partitions == 0


def test_audit_counts_on_synthetic_processed_files(tmp_path: Path) -> None:
    data = build_processed(
        tmp_path,
        edit_transaction=repeat_transaction_values,
        edit_identity=repeat_identity_values,
    )
    assert not data.dataset.transaction.path.exists()  # no raw file to read
    development = _development(data)
    before, partition_before = development.joined.copy(), development.partition.copy()
    audit = audit_duplicates(development, data.features, data.dataset)
    pd.testing.assert_frame_equal(development.joined, before)  # no row changed
    pd.testing.assert_series_equal(development.partition, partition_before)
    assert audit.partitions == PARTITIONS
    primary, secondary = audit.definitions
    assert primary.definition == PRIMARY
    assert primary.columns == data.features.model_inputs
    assert secondary.definition == SECONDARY
    assert secondary.columns == tuple(
        c for c in data.features.model_inputs if c not in data.features.derived
    )
    # Primary: rows 0 and 4 (Train) and 8 (Validation 1); row 2 differs in the time of day.
    assert primary.within["train"] == RepeatCounts(8, 1, 2, 1)
    assert primary.within["v1"] == RepeatCounts(4, 0, 0, 0)
    assert primary.within["v2"] == RepeatCounts(4, 0, 0, 0)
    assert primary.within[DEVELOPMENT] == RepeatCounts(16, 1, 3, 2)
    assert (primary.vectors_in_several_partitions, primary.rows_with_such_vectors) == (1, 3)
    # Secondary: row 2 joins, because the time of day is not compared.
    assert secondary.within["train"] == RepeatCounts(8, 1, 3, 2)
    assert secondary.within[DEVELOPMENT] == RepeatCounts(16, 1, 4, 3)
    assert (secondary.vectors_in_several_partitions, secondary.rows_with_such_vectors) == (1, 4)
    for definition in audit.definitions:
        assert [s.vectors for s in definition.shared] == [1, 0, 0]
    summary = audit_summary(audit)
    assert summary["partitions"] == list(PARTITIONS)
    assert "isFraud" not in str(summary)


def test_the_target_column_is_refused(tmp_path: Path) -> None:
    data = build_processed(tmp_path)
    development = _development(data)
    joined = development.joined.assign(isFraud=0)
    with pytest.raises(DataValidationError, match="target"):
        audit_duplicates(
            DevelopmentFeatures(joined, development.partition, development.record),
            data.features,
            data.dataset,
        )


def test_rows_outside_the_development_partitions_are_refused(tmp_path: Path) -> None:
    data = build_processed(tmp_path)
    development = _development(data)
    partition = development.partition.copy()
    partition.iloc[0] = "test"
    with pytest.raises(DataValidationError, match="outside the development partitions"):
        audit_duplicates(
            DevelopmentFeatures(development.joined, partition, development.record),
            data.features,
            data.dataset,
        )


@pytest.mark.parametrize("misalign", ["truncated", "reversed"])
def test_misaligned_partition_names_are_refused(tmp_path: Path, misalign: str) -> None:
    data = build_processed(tmp_path)
    development = _development(data)
    partition = development.partition
    partition = partition.iloc[:-1] if misalign == "truncated" else partition.iloc[::-1]
    with pytest.raises(DataValidationError, match="Partition"):
        audit_duplicates(
            DevelopmentFeatures(development.joined, partition, development.record),
            data.features,
            data.dataset,
        )
