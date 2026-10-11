"""The canonical model-input frame and the tree-based library forms (DR-008 §§4–5).

Canonical frame, identical for every model:

* categorical inputs as text labels: text kept exactly; numeric codes as whole-number text
  (``100.0`` -> ``"100"``); missing values as the declared label ``"__MISSING__"``;
* numeric inputs as ``float64`` with missing values left as NaN;
* the derived time-of-day inputs ``tod_sin`` and ``tod_cos``;
* columns in the declared order: categorical, numeric, derived.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

from review_capacity.config import FeatureSpec
from review_capacity.data.validation import check_categorical_values
from review_capacity.errors import DataValidationError
from review_capacity.features.time_of_day import time_of_day

TEXT_DTYPE = "str"

logger = logging.getLogger(__name__)


def _categorical_labels(values: pd.Series, numeric_coded: bool, missing_label: str) -> pd.Series:
    present = values.notna().to_numpy()
    labels = np.full(len(values), missing_label, dtype=object)
    if numeric_coded:
        labels[present] = values.to_numpy(dtype="float64")[present].astype("int64").astype(str)
    else:
        labels[present] = values.to_numpy(dtype=object)[present]
    return pd.Series(labels, index=values.index, dtype=TEXT_DTYPE)


def build_canonical_frame(joined: pd.DataFrame, features: FeatureSpec) -> pd.DataFrame:
    """Canonical model inputs of the given joined rows (target and excluded columns dropped)."""
    check_categorical_values(joined, features)
    columns: dict[str, pd.Series] = {}
    for col in features.categorical:
        columns[col] = _categorical_labels(
            joined[col], col in features.numeric_coded, features.missing_label
        )
    for col in features.numeric:
        columns[col] = joined[col].astype("float64")
    derived = time_of_day(joined[features.time])
    for col in features.derived:
        if col not in derived:
            raise DataValidationError(f"No rule implemented for derived input {col}")
        columns[col] = derived[col]
    frame = pd.DataFrame(columns, index=joined.index)
    if tuple(frame.columns) != features.model_inputs:
        raise DataValidationError("Canonical columns differ from the declared model inputs")
    logger.info("Model inputs built: %d rows, %d inputs", len(frame), len(frame.columns))
    return frame


def fit_category_lists(train: pd.DataFrame, features: FeatureSpec) -> dict[str, list[str]]:
    """Sorted category list of every categorical input, from Train rows only (LightGBM)."""
    return {col: sorted(train[col].unique().tolist()) for col in features.categorical}


def to_lightgbm(
    frame: pd.DataFrame, features: FeatureSpec, categories: Mapping[str, list[str]]
) -> pd.DataFrame:
    """LightGBM form: categoricals as pandas ``Categorical`` with the Train-fixed lists, so that
    a value unseen in Train becomes NaN (missing) as in LightGBM's own prediction behaviour."""
    out = frame.copy()
    for col in features.categorical:
        known = frame[col].isin(categories[col])
        out[col] = pd.Categorical(frame[col].where(known), categories=categories[col])
    return out


def to_catboost(frame: pd.DataFrame, features: FeatureSpec) -> pd.DataFrame:
    """CatBoost form: the canonical text columns unchanged (declared via ``cat_features``)."""
    if any(frame[col].isna().any() for col in features.categorical):
        raise DataValidationError("CatBoost categorical columns must not contain NaN")
    return frame.copy()


def numeric_missing_counts(frame: pd.DataFrame, features: FeatureSpec) -> pd.Series:
    """Missing values per numeric input (label-free; used for reporting)."""
    return frame[list(features.numeric_inputs)].isna().sum().astype(np.int64)
