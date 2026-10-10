"""The two time-of-day inputs (DR-008 §4).

theta = 360 degrees x (t mod 86400) / 86400, computed here in radians, where t is
``TransactionDT``; ``tod_sin`` = sin(theta) and ``tod_cos`` = cos(theta). Each pair is computed
from the row's own ``TransactionDT`` only (row-local, label-free, no fitted statistic).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from review_capacity.config import SECONDS_PER_DAY
from review_capacity.errors import DataValidationError

TIME_OF_DAY_COLUMNS = ("tod_sin", "tod_cos")


def time_of_day(transaction_dt: pd.Series) -> pd.DataFrame:
    """``tod_sin`` and ``tod_cos`` of each transaction.

    Raises ``DataValidationError`` unless every value is a finite whole number of seconds; the
    range of relative days is checked by ``data.partitions.validate_transaction_dt``.
    """
    values = pd.to_numeric(transaction_dt, errors="coerce").to_numpy(dtype="float64")
    if not (np.isfinite(values).all() and (np.floor(values) == values).all()):
        raise DataValidationError("TransactionDT must be finite whole numbers of seconds")
    seconds = values.astype("int64") % SECONDS_PER_DAY
    theta = 2.0 * np.pi * seconds.astype("float64") / SECONDS_PER_DAY
    return pd.DataFrame(
        {TIME_OF_DAY_COLUMNS[0]: np.sin(theta), TIME_OF_DAY_COLUMNS[1]: np.cos(theta)},
        index=transaction_dt.index,
    )
