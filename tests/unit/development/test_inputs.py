"""The verified Train and V1 inputs of the search, on synthetic processed files (no raw file)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from review_capacity.development import inputs
from review_capacity.development.inputs import read_search_inputs
from review_capacity.errors import DataValidationError
from review_capacity.models.registry import Weighting
from support.synthetic import DAY, Synthetic, build_processed


def _labels(transaction: pd.DataFrame) -> pd.DataFrame:
    """Two frauds among the eight Train rows; the Test labels stay the invalid value ``X``."""
    labels = transaction["isFraud"].copy()
    labels.iloc[:8] = ["0", "1", "0", "0", "0", "1", "0", "0"]
    return transaction.assign(isFraud=labels)


@pytest.fixture
def processed(tmp_path: Path) -> Synthetic:
    return build_processed(tmp_path, edit_transaction=_labels)


def test_train_and_v1_are_read_in_time_order(processed: Synthetic) -> None:
    data = read_search_inputs(processed.paths, processed.features, processed.dataset)
    assert (data.train.rows, data.v1.rows) == (8, 4)
    assert list(data.train.features.columns) == list(processed.features.model_inputs)
    assert data.train.model_data(None).in_time_order()
    assert data.v1.model_data(None).in_time_order()
    assert set((data.train.times // DAY).tolist()) == {1, 2}
    assert set((data.v1.times // DAY).tolist()) == {3}
    assert data.train.provenance.scope == "train"


def test_class_weights_come_from_train_labels_only(processed: Synthetic) -> None:
    data = read_search_inputs(processed.paths, processed.features, processed.dataset)
    assert data.train.frauds == 2
    assert data.weights(Weighting.UNWEIGHTED).positive == 1.0
    assert data.weights(Weighting.BALANCED).positive == 3.0  # 6 legitimate / 2 fraud
    balanced = data.train_data(Weighting.BALANCED)
    assert set(balanced.weights[balanced.labels == 1]) == {3.0}
    assert set(balanced.weights[balanced.labels == 0]) == {1.0}


def test_v1_is_evaluated_unweighted(processed: Synthetic) -> None:
    data = read_search_inputs(processed.paths, processed.features, processed.dataset)
    assert (data.v1_data().weights == 1.0).all()


def test_the_identifier_hash_follows_row_order(processed: Synthetic) -> None:
    data = read_search_inputs(processed.paths, processed.features, processed.dataset)
    assert len(data.train.ordered_ids_sha256) == 64
    assert data.train.ordered_ids_sha256 != data.v1.ordered_ids_sha256


def test_relative_days_must_agree_with_transaction_dt(
    processed: Synthetic, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = inputs._times_by_id

    def shifted(joined: pd.DataFrame, features: object) -> pd.Series:
        times: pd.Series = original(joined, features)  # type: ignore[arg-type]
        return times + DAY

    monkeypatch.setattr(inputs, "_times_by_id", shifted)
    with pytest.raises(DataValidationError, match="relative days differ"):
        read_search_inputs(processed.paths, processed.features, processed.dataset)


def test_rows_out_of_time_order_are_rejected(
    processed: Synthetic, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = inputs._times_by_id

    def swapped(joined: pd.DataFrame, features: object) -> pd.Series:
        times: pd.Series = original(joined, features)  # type: ignore[arg-type]
        first, second = times.index[0], times.index[1]  # two Train rows of the same day
        assert times[first] // DAY == times[second] // DAY
        times[first], times[second] = times[second], times[first]
        return times

    monkeypatch.setattr(inputs, "_times_by_id", swapped)
    with pytest.raises(DataValidationError, match="not in time order"):
        read_search_inputs(processed.paths, processed.features, processed.dataset)


def test_the_joined_file_is_read_without_the_target(
    processed: Synthetic, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: list[tuple[str, list[str] | None]] = []
    read_parquet = pd.read_parquet

    def recording(path: Path, **kwargs: Any) -> pd.DataFrame:
        columns = kwargs.get("columns")
        reads.append((Path(path).name, None if columns is None else list(columns)))
        frame: pd.DataFrame = read_parquet(path, **kwargs)
        return frame

    monkeypatch.setattr(pd, "read_parquet", recording)
    read_search_inputs(processed.paths, processed.features, processed.dataset)
    joined = [columns for name, columns in reads if name.startswith("development_joined")]
    assert joined
    target = processed.features.target
    assert all(columns is not None and target not in columns for columns in joined)
