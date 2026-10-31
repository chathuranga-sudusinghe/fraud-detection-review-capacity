"""Reading, joining and the sealed Test period, on synthetic files."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from review_capacity.data.loading import NUMERIC_MISSING_TOKENS, load_labels, load_raw
from review_capacity.errors import DataValidationError, FileAccessError, SealedTestError
from support.synthetic import Synthetic, build_synthetic


def test_load_joins_without_target(synthetic: Synthetic) -> None:
    raw = load_raw(synthetic.features, synthetic.dataset)
    assert len(raw.joined) == len(synthetic.transaction)
    assert "isFraud" not in raw.joined.columns
    expected = raw.joined["TransactionID"].isin(synthetic.identity["TransactionID"])
    assert raw.has_identity.equals(expected)  # the join flag, not the non-missing fields
    assert set(raw.joined.columns) == set(synthetic.features.joined_columns) - {"isFraud"}
    assert raw.joined["TransactionDT"].dtype == "int64"
    matched = raw.joined["TransactionID"].isin(synthetic.identity["TransactionID"])
    assert raw.joined.loc[~matched, "id_01"].isna().all()
    assert raw.partition.astype(str).value_counts().to_dict() == {
        "train": 8,
        "v1": 4,
        "v2": 4,
        "test": 4,
    }


def test_development_labels_stop_before_sealed_rows(
    synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = load_raw(synthetic.features, synthetic.dataset)
    calls: list[dict[str, Any]] = []
    read_csv = pd.read_csv

    def recording_read_csv(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls.append(kwargs)
        frame: pd.DataFrame = read_csv(*args, **kwargs)
        return frame

    monkeypatch.setattr(pd, "read_csv", recording_read_csv)  # looked up by data.loading per call
    labels = load_labels(raw, synthetic.features, synthetic.dataset, ["train", "v1", "v2"])
    assert len(labels) == 16
    assert set(labels.unique()) == {0, 1}
    (label_read,) = [c for c in calls if "isFraud" in c.get("usecols", [])]
    assert label_read["nrows"] == 16  # the parser stops at the last development row


def test_sealed_labels_refused(synthetic: Synthetic) -> None:
    raw = load_raw(synthetic.features, synthetic.dataset)
    with pytest.raises(SealedTestError):
        load_labels(raw, synthetic.features, synthetic.dataset, ["test"])
    with pytest.raises(SealedTestError):
        load_labels(raw, synthetic.features, synthetic.dataset, ["v2", "test"])


def test_explicit_unsealing_reads_test_labels(tmp_path: Path) -> None:
    data = build_synthetic(tmp_path, test_label="1")
    raw = load_raw(data.features, data.dataset)
    labels = load_labels(raw, data.features, data.dataset, ["test"], unseal_test_labels=True)
    assert labels.tolist() == [1, 1, 1, 1]


def test_undeclared_column_stops(tmp_path: Path) -> None:
    data = build_synthetic(tmp_path, edit_transaction=lambda f: f.assign(extra=1.0))
    with pytest.raises(DataValidationError, match="undeclared columns"):
        load_raw(data.features, data.dataset)


def test_repeated_identity_key_stops(tmp_path: Path) -> None:
    data = build_synthetic(tmp_path, edit_identity=lambda f: pd.concat([f, f.iloc[:1]]))
    with pytest.raises(DataValidationError, match="repeated TransactionID"):
        load_raw(data.features, data.dataset)


def test_unmatched_identity_row_stops(tmp_path: Path) -> None:
    def add_orphan(frame: pd.DataFrame) -> pd.DataFrame:
        orphan = frame.iloc[:1].assign(TransactionID=9_999_999)
        return pd.concat([frame, orphan])

    data = build_synthetic(tmp_path, edit_identity=add_orphan)
    with pytest.raises(DataValidationError, match="match no transaction"):
        load_raw(data.features, data.dataset)


def test_unordered_time_stops(tmp_path: Path) -> None:
    def swap(frame: pd.DataFrame) -> pd.DataFrame:
        order = list(range(len(frame)))
        order[0], order[1] = order[1], order[0]
        return frame.iloc[order]

    data = build_synthetic(tmp_path, edit_transaction=swap)
    with pytest.raises(DataValidationError, match="non-decreasing"):
        load_raw(data.features, data.dataset)


def test_missing_time_stops(tmp_path: Path) -> None:
    def blank(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.astype({"TransactionDT": "float64"})
        frame.loc[3, "TransactionDT"] = None
        return frame

    data = build_synthetic(tmp_path, edit_transaction=blank)
    with pytest.raises(DataValidationError, match="missing"):
        load_raw(data.features, data.dataset)


def test_non_binary_development_label_stops(tmp_path: Path) -> None:
    data = build_synthetic(
        tmp_path, edit_transaction=lambda f: f.assign(isFraud=["2", *f["isFraud"].iloc[1:]])
    )
    raw = load_raw(data.features, data.dataset)
    with pytest.raises(DataValidationError, match="not binary"):
        load_labels(raw, data.features, data.dataset, ["train"])


TEXT_LIKE_MISSING = ["NA", "NULL", "NaN", "None", " "]


def test_literal_text_tokens_are_kept(tmp_path: Path) -> None:
    data = build_synthetic(
        tmp_path,
        edit_transaction=lambda f: f.assign(ProductCD=TEXT_LIKE_MISSING * (len(f) // 5)),
    )
    raw = load_raw(data.features, data.dataset)
    assert raw.joined["ProductCD"].tolist()[:5] == TEXT_LIKE_MISSING
    assert raw.joined["ProductCD"].notna().all()


def test_empty_text_field_is_missing(synthetic: Synthetic) -> None:
    raw = load_raw(synthetic.features, synthetic.dataset)
    assert raw.joined["ProductCD"].isna().tolist()[:4] == [False, False, True, False]


@pytest.mark.parametrize("token", NUMERIC_MISSING_TOKENS)
def test_every_documented_numeric_missing_token_reads_as_missing(
    tmp_path: Path, token: str
) -> None:
    data = build_synthetic(
        tmp_path, edit_transaction=lambda f: f.assign(V1=[token, 1.0, 0.0, 2.0] * (len(f) // 4))
    )
    raw = load_raw(data.features, data.dataset)
    assert raw.joined["V1"].isna().tolist()[:4] == [True, False, False, False]


def test_unparseable_numeric_value_stops(tmp_path: Path) -> None:
    data = build_synthetic(
        tmp_path, edit_transaction=lambda f: f.assign(V1=["abc", *f["V1"].iloc[1:]])
    )
    with pytest.raises(DataValidationError, match="cannot be parsed"):
        load_raw(data.features, data.dataset)


def test_loading_logs_progress(synthetic: Synthetic, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="review_capacity"):
        load_raw(synthetic.features, synthetic.dataset)
    messages = " | ".join(r.getMessage() for r in caplog.records)
    assert "train_transaction.csv" in messages
    assert "train 8, v1 4, v2 4, test 4" in messages


def test_feature_reads_never_include_the_target(
    synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    read_csv = pd.read_csv

    def recording_read_csv(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls.append(kwargs)
        frame: pd.DataFrame = read_csv(*args, **kwargs)
        return frame

    monkeypatch.setattr(pd, "read_csv", recording_read_csv)
    load_raw(synthetic.features, synthetic.dataset)
    assert calls
    assert all("isFraud" not in call["usecols"] for call in calls)


def test_identity_columns_must_match_the_declaration(synthetic: Synthetic) -> None:
    dataset = replace(synthetic.dataset, identity_columns=("id_01", "id_13", "DeviceInfo"))
    with pytest.raises(DataValidationError, match="columns_besides_key"):
        load_raw(synthetic.features, dataset)


def test_missing_raw_file_is_a_file_access_error(synthetic: Synthetic) -> None:
    synthetic.dataset.identity.path.unlink()
    with pytest.raises(FileAccessError):
        load_raw(synthetic.features, synthetic.dataset)
