"""The data validation contract (protocol design item 8; DR-001; DR-008).

Checks that need the raw files are applied while loading (``data.loading``): the raw-file
checksums, headers against ``config/features.yaml``, unique keys, the one-to-one join, the
``TransactionDT`` rule and its file order. This module adds the checks against the recorded
integrity anchors and the column-value rules, and returns a label-free summary for the dataset
manifest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from review_capacity.config import DatasetSpec, FeatureSpec
from review_capacity.data.loading import RawData
from review_capacity.errors import DataValidationError
from review_capacity.hashing import id_set_sha256
from review_capacity.logging_setup import log_step

logger = logging.getLogger(__name__)

# Largest magnitude up to which every whole number is exactly representable in float64 (2**53);
# numeric codes beyond it could not be converted to their exact integer labels.
LARGEST_EXACT_CODE = 2**53


@dataclass(frozen=True)
class PartitionSummary:
    """Label-free description of one partition."""

    name: str
    first_day: int
    last_day: int
    rows: int
    transaction_dt_min: int
    transaction_dt_max: int
    transaction_id_sha256: str


def summarise_partitions(
    raw: RawData, features: FeatureSpec, dataset: DatasetSpec
) -> list[PartitionSummary]:
    """Rows, ``TransactionDT`` bounds and full identifier hash of every partition."""
    summaries = []
    keys_and_times = raw.joined[[features.key, features.time]]
    for part in dataset.partitions:
        rows = keys_and_times[(raw.partition == part.name).to_numpy()]
        if rows.empty:
            raise DataValidationError(f"Partition {part.name} is empty")
        summaries.append(
            PartitionSummary(
                name=part.name,
                first_day=part.first_day,
                last_day=part.last_day,
                rows=len(rows),
                transaction_dt_min=int(rows[features.time].min()),
                transaction_dt_max=int(rows[features.time].max()),
                transaction_id_sha256=id_set_sha256(rows[features.key]),
            )
        )
    return summaries


def check_partition_anchors(summaries: list[PartitionSummary], dataset: DatasetSpec) -> None:
    """Rows, bounds and hash prefixes must reproduce the values recorded in DR-001."""
    problems = []
    for summary in summaries:
        part = dataset.partition(summary.name)
        expected = (
            part.rows,
            part.transaction_dt_min,
            part.transaction_dt_max,
            part.transaction_id_sha256_prefix,
        )
        observed = (
            summary.rows,
            summary.transaction_dt_min,
            summary.transaction_dt_max,
            summary.transaction_id_sha256[: len(part.transaction_id_sha256_prefix)],
        )
        if observed != expected:
            problems.append(f"{part.name}: observed {observed}, recorded {expected}")
    total = sum(s.rows for s in summaries)
    if total != dataset.total_rows:
        problems.append(f"{total} rows in total, {dataset.total_rows} recorded")
    if problems:
        raise DataValidationError("Partition anchors not reproduced: " + "; ".join(problems))
    logger.info(
        "Partition anchors reproduced (rows, TransactionDT bounds, hash prefixes): %s",
        ", ".join(s.name for s in summaries),
    )


def check_categorical_values(joined: pd.DataFrame, features: FeatureSpec) -> None:
    """Numeric-coded categoricals must be whole numbers that convert exactly to integer labels;
    no raw text value may equal the reserved missing label."""
    problems = []
    for col in features.categorical:
        values = joined[col].dropna()
        if col in features.numeric_coded:
            numbers = values.to_numpy(dtype="float64")
            finite = np.isfinite(numbers)
            with np.errstate(invalid="ignore"):
                not_whole = ~finite | (np.floor(numbers) != numbers)
                too_large = finite & (np.abs(numbers) > LARGEST_EXACT_CODE)
            if not_whole.any():
                problems.append(f"{col}: {int(not_whole.sum())} values are not whole numbers")
            if too_large.any():
                problems.append(
                    f"{col}: {int(too_large.sum())} values exceed ±2**53 and cannot be "
                    "converted to exact integer labels"
                )
        elif (values == features.missing_label).any():
            problems.append(f"{col}: contains the reserved label {features.missing_label!r}")
    if problems:
        raise DataValidationError("Categorical values invalid: " + "; ".join(problems))


def check_numeric_types(joined: pd.DataFrame, features: FeatureSpec) -> None:
    """Every numeric column must have been read as float64."""
    wrong = [c for c in features.numeric if joined[c].dtype != np.float64]
    if wrong:
        raise DataValidationError(f"Numeric columns not float64: {wrong}")


def validate(raw: RawData, features: FeatureSpec, dataset: DatasetSpec) -> list[PartitionSummary]:
    """Run the value checks of the contract on loaded data; return the partition summaries."""
    with log_step(logger, "Running the data validation contract"):
        check_categorical_values(raw.joined, features)
        check_numeric_types(raw.joined, features)
        summaries = summarise_partitions(raw, features, dataset)
        check_partition_anchors(summaries, dataset)
    return summaries
