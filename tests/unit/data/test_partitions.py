"""``TransactionDT`` validation, relative days and the DR-001 partitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from review_capacity.config import load_dataset_spec
from review_capacity.data.partitions import assign_partitions, relative_day, validate_transaction_dt
from review_capacity.errors import DataValidationError
from support.synthetic import REPO_PATHS

DAY = 86_400


def test_valid_times_pass() -> None:
    validate_transaction_dt(pd.Series([DAY, 2 * DAY - 1, 182 * DAY + 5]), first_day=1, last_day=182)


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (np.nan, "missing"),
        (np.inf, "non-finite"),
        (DAY + 0.5, "not a whole number"),
        (DAY - 1, "relative day outside"),
        (183 * DAY, "relative day outside"),
    ],
)
def test_invalid_time_stops(value: float, reason: str) -> None:
    with pytest.raises(DataValidationError, match=reason):
        validate_transaction_dt(pd.Series([DAY, value]), first_day=1, last_day=182)


def test_text_time_stops() -> None:
    with pytest.raises(DataValidationError, match="not numeric"):
        validate_transaction_dt(pd.Series([str(DAY), "noon"]), first_day=1, last_day=182)


def test_relative_day_rounds_down() -> None:
    days = relative_day(pd.Series([DAY, 2 * DAY - 1, 2 * DAY]))
    assert days.tolist() == [1, 1, 2]


def test_partition_boundaries_follow_dr001() -> None:
    spec = load_dataset_spec(REPO_PATHS)
    days = pd.Series([1, 105, 106, 126, 127, 154, 155, 182])
    assigned = assign_partitions(days, spec).astype(str).tolist()
    assert assigned == ["train", "train", "v1", "v1", "v2", "v2", "test", "test"]
    with pytest.raises(DataValidationError):
        assign_partitions(pd.Series([183]), spec)
