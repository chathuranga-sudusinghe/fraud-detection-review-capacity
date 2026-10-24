"""The fine pseudo-entity key of the dependence-aware inference (DR-006; Issue #156).

The dataset has no account or customer identifier. DR-006 resamples frauds by relative day and by
a heuristic fine pseudo-entity built from ``card1``, ``addr1``, ``P_emaildomain`` and the implied
reference day, relative day minus ``D1``. The key is that of the feasibility audit (notebook 01,
"Two HEURISTIC pseudo-entity keys"): each part is written as text, a missing value as ``"NA"``,
and the parts are joined with ``|``; the implied reference day is ``round(day - D1)`` with
half-to-even rounding, written as a float exactly as the audit's string cast does (so ``-0.0`` and
``0.0`` are different parts), and is missing when ``D1`` is. It is used for resampling and
slicing only, never as a model input (protocol Feature Policy).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from review_capacity.errors import DataValidationError
from review_capacity.models.base import Keys

KEY_COLUMNS = ("card1", "addr1", "P_emaildomain")
DAY_OFFSET_COLUMN = "D1"
MISSING_PART = "NA"
SEPARATOR = "|"


def _part(values: pd.Series, missing_label: str) -> npt.NDArray[np.object_]:
    text = values.astype(str).to_numpy(dtype=object)
    text[(values.isna() | (values == missing_label)).to_numpy()] = MISSING_PART
    return text


def start_proxy(days: Keys, offset: npt.NDArray[np.float64]) -> npt.NDArray[np.object_]:
    """``round(day - D1)`` (half to even) written as the audit writes the rounded float
    (``"8.0"``; ``"-0.0"`` stays distinct from ``"0.0"``), ``"NA"`` where ``D1`` is missing."""
    implied = np.round(days.astype(np.float64) - offset)
    text = np.full(len(days), MISSING_PART, dtype=object)
    known = np.isfinite(implied)
    text[known] = [str(float(value)) for value in implied[known]]
    return text


def pseudo_entity_keys(features: pd.DataFrame, days: Keys, missing_label: str) -> list[str]:
    """The fine pseudo-entity key of each row of ``features`` (canonical model inputs) whose
    relative days are ``days``; ``missing_label`` is the declaration's missing-category token."""
    absent = [c for c in (*KEY_COLUMNS, DAY_OFFSET_COLUMN) if c not in features]
    if absent:
        raise DataValidationError(f"The pseudo-entity key needs the inputs {absent}")
    if days.shape != (len(features),):
        raise DataValidationError(f"{len(features)} rows need {len(features)} relative days")
    parts = [_part(features[column], missing_label) for column in KEY_COLUMNS]
    offset = features[DAY_OFFSET_COLUMN].to_numpy(dtype=np.float64)
    parts.append(start_proxy(days, offset))
    return [SEPARATOR.join(row) for row in zip(*parts, strict=True)]
