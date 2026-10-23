"""Synthetic canonical frames for the neural preprocessing tests (no project data is read).

``canonical_frame`` builds the model inputs of ``support.synthetic.FEATURES`` in canonical form
(DR-008 §5) with controlled missingness, tied values and a time-of-day pair over the whole cycle,
with the identifiers and times that fix the row order.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from review_capacity.config import SECONDS_PER_DAY
from review_capacity.features.time_of_day import time_of_day
from support.synthetic import FEATURES

MISSING = "__MISSING__"
type Rows = tuple[pd.DataFrame, npt.NDArray[np.int64], npt.NDArray[np.int64]]


def canonical_frame(
    rows: int = 400,
    seed: int = 1,
    *,
    missing: dict[str, float] | None = None,
    categorical_missing: bool = True,
) -> Rows:
    """The canonical frame, its ``TransactionID`` values and its ``TransactionDT`` values.

    ``missing`` gives the share of NaN per numeric input (default: ``TransactionAmt`` 10 %,
    ``V1`` none, ``id_01`` 50 %, so the indicators are not contiguous). ``V1`` is 60 % zeros, a
    dominant repeated value. Times cover the whole day, ascending; identifiers ascend.
    """
    shares = {"TransactionAmt": 0.1, "V1": 0.0, "id_01": 0.5} | (missing or {})
    rng = np.random.default_rng(seed)
    times = np.linspace(0, 2 * SECONDS_PER_DAY - 1, rows).astype(np.int64)
    ids = np.arange(1_000, 1_000 + rows, dtype=np.int64)
    values = {
        "TransactionAmt": np.exp(rng.normal(3.0, 1.0, rows)),
        "V1": np.where(rng.random(rows) < 0.6, 0.0, np.exp(rng.normal(1.0, 1.0, rows))),
        "id_01": rng.normal(-5.0, 3.0, rows),
    }
    for name, share in shares.items():
        values[name][rng.random(rows) < share] = np.nan
    labels = {
        "ProductCD": ["W", "C", "H", "R"],
        "card1": ["1000", "2000", "3000"] + ([MISSING] if categorical_missing else []),
        "DeviceType": ["desktop", "mobile"] + ([MISSING] if categorical_missing else []),
        "id_13": ["49", "52"],
    }
    columns: dict[str, pd.Series] = {
        name: pd.Series(rng.choice(choices, rows), dtype="str") for name, choices in labels.items()
    }
    for name in FEATURES.numeric:
        columns[name] = pd.Series(values[name], dtype="float64")
    derived = time_of_day(pd.Series(times))
    for name in FEATURES.derived:
        columns[name] = derived[name].astype("float64")
    frame = pd.DataFrame(columns)[list(FEATURES.model_inputs)]
    return frame, ids, times


def fitted_artifact(
    rows: int = 400,
    seed: int = 1,
    *,
    missing: dict[str, float] | None = None,
    categorical_missing: bool = True,
) -> tuple[Any, Rows]:
    """A shared preprocessing artifact fitted on ``canonical_frame`` (preprocessing seed 0), with
    the frame, identifiers and times it was fitted on."""
    from review_capacity.features.neural_preprocessing import fit_neural_preprocessing

    data = canonical_frame(rows, seed, missing=missing, categorical_missing=categorical_missing)
    frame, ids, times = data
    return fit_neural_preprocessing(frame, ids, times, FEATURES, 0), data
