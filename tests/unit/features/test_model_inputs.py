"""The canonical model-input frame and the tree-based forms (DR-008 §§4-5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from review_capacity.data.loading import load_raw
from review_capacity.errors import DataValidationError
from review_capacity.features.model_inputs import (
    build_canonical_frame,
    fit_category_lists,
    to_catboost,
    to_lightgbm,
)
from support.synthetic import Synthetic, build_synthetic


def _canonical(synthetic: Synthetic) -> pd.DataFrame:
    raw = load_raw(synthetic.features, synthetic.dataset)
    return build_canonical_frame(raw.joined, synthetic.features)


def test_columns_follow_declared_input_order(synthetic: Synthetic) -> None:
    frame = _canonical(synthetic)
    assert tuple(frame.columns) == synthetic.features.model_inputs
    assert "TransactionDT" not in frame.columns
    assert "isFraud" not in frame.columns


def test_categorical_labels(synthetic: Synthetic) -> None:
    frame = _canonical(synthetic)
    assert frame["card1"].iloc[:4].tolist() == ["100", "__MISSING__", "-660", "1000"]
    assert frame["ProductCD"].iloc[:4].tolist() == ["W", " C", "__MISSING__", "H"]
    assert frame["id_13"].iloc[:4].tolist() == ["49", "__MISSING__", "__MISSING__", "__MISSING__"]
    assert not any(frame[c].isna().any() for c in synthetic.features.categorical)


def test_numeric_inputs_keep_missing_values(synthetic: Synthetic) -> None:
    frame = _canonical(synthetic)
    for col in synthetic.features.numeric_inputs:
        assert frame[col].dtype == np.float64
    assert frame["V1"].iloc[:4].isna().tolist() == [False, True, False, False]
    assert frame[["tod_sin", "tod_cos"]].notna().all().all()
    np.testing.assert_allclose(frame[["tod_sin", "tod_cos"]].iloc[0], [0.0, 1.0], atol=1e-12)


def test_lightgbm_categories_from_train_only(synthetic: Synthetic) -> None:
    frame = _canonical(synthetic)
    train, later = frame.iloc[:2], frame.iloc[2:]
    categories = fit_category_lists(train, synthetic.features)
    assert categories["card1"] == ["100", "__MISSING__"]
    lgb = to_lightgbm(later, synthetic.features, categories)
    assert isinstance(lgb["card1"].dtype, pd.CategoricalDtype)
    assert pd.isna(lgb["card1"].iloc[0])  # "-660" is unseen in Train: missing for LightGBM
    assert lgb["card1"].iloc[2] == "100"


def test_catboost_form_keeps_text(synthetic: Synthetic) -> None:
    frame = _canonical(synthetic)
    cat = to_catboost(frame, synthetic.features)
    pd.testing.assert_frame_equal(cat, frame)


def test_largest_exact_numeric_code_keeps_its_label(tmp_path: Path) -> None:
    data = build_synthetic(
        tmp_path, edit_transaction=lambda f: f.assign(card1=[2.0**53, *f["card1"].iloc[1:]])
    )
    raw = load_raw(data.features, data.dataset)
    frame = build_canonical_frame(raw.joined, data.features)
    assert frame["card1"].iloc[0] == "9007199254740992"


def test_direct_call_rejects_fractional_time(synthetic: Synthetic) -> None:
    raw = load_raw(synthetic.features, synthetic.dataset)
    joined = raw.joined.astype({"TransactionDT": "float64"})
    joined.loc[0, "TransactionDT"] = 86_400.5
    with pytest.raises(DataValidationError, match="finite whole numbers"):
        build_canonical_frame(joined, synthetic.features)
