"""Relative days and chronological partitions from ``TransactionDT`` (DR-001).

* Relative day = ``TransactionDT // 86400``; no calendar date is inferred.
* Validation rule (DR-008 §4): a missing, non-finite or non-integer ``TransactionDT``, or a
  relative day outside the declared range, stops data preparation. No row is dropped or imputed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from review_capacity.config import SECONDS_PER_DAY, DatasetSpec
from review_capacity.errors import DataValidationError


def validate_transaction_dt(values: pd.Series, *, first_day: int, last_day: int) -> None:
    """Raise ``DataValidationError`` unless every value is a whole number of seconds whose
    relative day lies in ``first_day``-``last_day``."""
    missing = values.isna().to_numpy()
    as_float = pd.to_numeric(values, errors="coerce").astype("float64").to_numpy()
    not_numeric = np.isnan(as_float) & ~missing
    finite = np.isfinite(as_float)
    non_finite = ~finite & ~np.isnan(as_float)
    with np.errstate(invalid="ignore"):
        non_integer = finite & (np.floor(as_float) != as_float)
        day = np.floor(as_float / SECONDS_PER_DAY)
        out_of_range = finite & ~non_integer & ((day < first_day) | (day > last_day))
    problems = {
        "missing": int(missing.sum()),
        "not numeric": int(not_numeric.sum()),
        "non-finite": int(non_finite.sum()),
        "not a whole number of seconds": int(non_integer.sum()),
        f"relative day outside {first_day}-{last_day}": int(out_of_range.sum()),
    }
    failed = {name: n for name, n in problems.items() if n}
    if failed:
        raise DataValidationError(f"TransactionDT validation failed: {failed}")


def relative_day(transaction_dt: pd.Series) -> pd.Series:
    """Relative day of each transaction (validated ``TransactionDT`` expected)."""
    return (transaction_dt.astype("int64") // SECONDS_PER_DAY).rename("relative_day")


def assign_partitions(days: pd.Series, spec: DatasetSpec) -> pd.Series:
    """Partition name of each relative day, by the DR-001 boundaries."""
    names = [p.name for p in spec.partitions]
    edges = [p.first_day for p in spec.partitions] + [spec.last_day + 1]
    assigned = pd.cut(days, bins=edges, right=False, labels=names)
    if assigned.isna().any():
        raise DataValidationError("Relative days outside the declared partitions")
    return pd.Series(assigned, index=days.index, name="partition")
