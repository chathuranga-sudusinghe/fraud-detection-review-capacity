"""The verified processed-file reader, on synthetic data after the Parquet build (no raw file)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from review_capacity.config import features_fingerprint
from review_capacity.data.processed import (
    FINGERPRINT,
    JOINED,
    read_development_features,
    read_processed,
    read_record,
    schema,
    verify_processed,
)
from review_capacity.errors import DataValidationError, FileAccessError
from review_capacity.hashing import file_sha256
from support.synthetic import Synthetic, build_processed, with_partition


@pytest.fixture
def processed(tmp_path: Path) -> Synthetic:
    return build_processed(tmp_path)


def _edit_record(data: Synthetic, stem: str, edit: Callable[[dict[str, Any]], None]) -> None:
    path = data.paths.provenance_file(stem)
    record = json.loads(path.read_text(encoding="utf-8"))
    edit(record)
    path.write_text(json.dumps(record), encoding="utf-8")


def test_partition_file_has_the_declared_schema_and_kinds(processed: Synthetic) -> None:
    frame = read_processed(processed.paths, "train", processed.features, processed.dataset)
    assert list(frame.columns) == list(schema("train", processed.features))
    assert len(frame) == 8
    assert frame["relative_day"].between(1, 2).all()
    assert frame[list(processed.features.categorical)].notna().all().all()


def test_joined_file_holds_development_rows_only(processed: Synthetic) -> None:
    frame = read_processed(processed.paths, JOINED, processed.features, processed.dataset)
    assert set(frame.columns) == set(schema(JOINED, processed.features))
    assert sorted(frame["partition"].unique()) == ["train", "v1", "v2"]
    assert int(frame["TransactionDT"].max()) < 5 * 86_400  # no Test-period row


def test_development_features_never_read_the_target(
    processed: Synthetic, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested: list[Any] = []
    read_parquet = pd.read_parquet

    def recording(path: Path, **kwargs: Any) -> pd.DataFrame:
        requested.append(kwargs.get("columns"))
        frame: pd.DataFrame = read_parquet(path, **kwargs)
        return frame

    monkeypatch.setattr(pd, "read_parquet", recording)
    data = read_development_features(processed.paths, processed.features, processed.dataset)
    assert requested
    assert all(columns is not None and "isFraud" not in columns for columns in requested)
    assert "isFraud" not in data.joined.columns
    assert data.partition.value_counts().to_dict() == {"train": 8, "v1": 4, "v2": 4}
    assert data.record.scope == "development"


def test_altered_file_is_rejected(processed: Synthetic) -> None:
    path = processed.paths.processed_file("v1")
    frame = pd.read_parquet(path)
    frame.loc[0, "TransactionAmt"] = 999.0
    frame.to_parquet(path, index=False)
    with pytest.raises(DataValidationError, match=r"SHA-256 .* differs from its record"):
        read_processed(processed.paths, "v1", processed.features, processed.dataset)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda r: r.update(scope="train"), "scope train, expected v1"),
        (lambda r: r.update(file="v2.parquet"), "record describes v2.parquet"),
        (lambda r: r["sources"].update({"train_identity.csv": "0" * 64}), "raw-source hashes"),
        (lambda r: r.update(transaction_id_sha256="0" * 64), "DR-001 anchor"),
        (lambda r: r.update(rows=5), "record states 5"),
        (lambda r: r.update(rows="4"), "rows must be a non-negative integer"),
        (lambda r: r.pop("sha256"), "sha256 must be a non-empty string"),
        (lambda r: r.update(code=None), "code must be a JSON object"),
    ],
)
def test_record_that_does_not_describe_the_file_is_rejected(
    processed: Synthetic, edit: Callable[[dict[str, Any]], None], message: str
) -> None:
    _edit_record(processed, "v1", edit)
    with pytest.raises(DataValidationError, match=message):
        verify_processed(processed.paths, "v1", processed.features, processed.dataset)


def test_missing_record_or_file_is_a_file_access_error(processed: Synthetic) -> None:
    processed.paths.provenance_file("v2").unlink()
    with pytest.raises(FileAccessError, match="provenance record missing"):
        verify_processed(processed.paths, "v2", processed.features, processed.dataset)
    processed.paths.processed_file("train").unlink()
    with pytest.raises(FileAccessError):
        verify_processed(processed.paths, "train", processed.features, processed.dataset)


def test_record_that_is_not_json_is_rejected(processed: Synthetic) -> None:
    processed.paths.provenance_file("train").write_text("{", encoding="utf-8")
    with pytest.raises(DataValidationError, match="not valid JSON"):
        read_record(processed.paths.provenance_file("train"))


def test_changed_feature_declaration_is_a_schema_difference(processed: Synthetic) -> None:
    features = replace(processed.features, numeric=(*processed.features.numeric[:-1],))
    # the record carries the fingerprint of this declaration, so only the schema check is left
    _edit_record(
        processed,
        "train",
        lambda r: r["configuration"].update({FINGERPRINT: features_fingerprint(features)}),
    )
    with pytest.raises(DataValidationError, match=r"unexpected \['id_01'\]"):
        verify_processed(processed.paths, "train", features, processed.dataset)


def test_changed_anchor_is_rejected(processed: Synthetic) -> None:
    dataset = with_partition(processed.dataset, "train", transaction_id_sha256_prefix="0" * 16)
    with pytest.raises(DataValidationError, match="DR-001 anchor"):
        verify_processed(processed.paths, "train", processed.features, dataset)


def test_test_partition_is_not_a_processed_file(processed: Synthetic) -> None:
    with pytest.raises(DataValidationError, match="not a processed development file"):
        verify_processed(processed.paths, "test", processed.features, processed.dataset)


def test_undeclared_column_request_is_rejected(processed: Synthetic) -> None:
    with pytest.raises(DataValidationError, match="requested columns not in the file"):
        read_processed(
            processed.paths, "train", processed.features, processed.dataset, columns=["extra"]
        )


def test_column_of_the_wrong_kind_is_rejected(processed: Synthetic) -> None:
    path = processed.paths.processed_file("train")
    frame = pd.read_parquet(path)
    frame["TransactionAmt"] = frame["TransactionAmt"].astype("float32")
    frame.to_parquet(path, index=False)
    _edit_record(processed, "train", lambda r: r.update(sha256=file_sha256(path)))
    with pytest.raises(DataValidationError, match=r"undeclared kind: \['TransactionAmt"):
        read_processed(processed.paths, "train", processed.features, processed.dataset)


def test_lineage_is_reported_without_requiring_the_current_commit(
    processed: Synthetic, caplog: pytest.LogCaptureFixture
) -> None:
    _edit_record(processed, "v1", lambda r: r["code"].update(git_commit="a" * 40))
    with caplog.at_level(logging.INFO, logger="review_capacity"):
        verify_processed(processed.paths, "v1", processed.features, processed.dataset)
    assert "built by commit aaaaaaaaaaaa" in caplog.text


def _rewrite(data: Synthetic, stem: str, edit: Callable[[pd.DataFrame], None]) -> None:
    """Change a processed file and make its record agree, as a swapped file would."""
    path = data.paths.processed_file(stem)
    frame = pd.read_parquet(path)
    edit(frame)
    frame.to_parquet(path, index=False)
    _edit_record(data, stem, lambda r: r.update(sha256=file_sha256(path)))


def test_content_must_reproduce_the_identifier_anchor(processed: Synthetic) -> None:
    def new_identifier(frame: pd.DataFrame) -> None:
        frame.loc[0, "TransactionID"] = 9_999_999

    _rewrite(processed, "v1", new_identifier)
    with pytest.raises(DataValidationError, match="v1: identifier hash differs"):
        verify_processed(processed.paths, "v1", processed.features, processed.dataset)


def test_joined_content_must_reproduce_the_time_anchors(processed: Synthetic) -> None:
    def later_first_time(frame: pd.DataFrame) -> None:
        frame.loc[0, "TransactionDT"] = 86_401  # same relative day, one second later

    _rewrite(processed, JOINED, later_first_time)
    with pytest.raises(DataValidationError, match="train: TransactionDT bounds"):
        verify_processed(processed.paths, JOINED, processed.features, processed.dataset)


def test_file_built_under_another_feature_declaration_is_stale(processed: Synthetic) -> None:
    other = replace(processed.features, missing_label="NA")
    match = r"another features\.yaml declaration \(stale\)"
    with pytest.raises(DataValidationError, match=match):
        verify_processed(processed.paths, "train", other, processed.dataset)


def test_a_change_outside_the_stored_values_keeps_the_file_valid(processed: Synthetic) -> None:
    spec = processed.paths.features_spec
    text = spec.read_text(encoding="utf-8")
    old = 'unseen_categories: "handled natively by CatBoost"'
    edited = text.replace(old, 'unseen_categories: "x"')
    assert edited != text
    spec.write_text("# a comment\n" + edited, "utf-8")
    verify_processed(processed.paths, "train", processed.features, processed.dataset)


def test_record_without_a_fingerprint_asks_for_the_migration(processed: Synthetic) -> None:
    _edit_record(processed, "train", lambda r: r["configuration"].pop("features_fingerprint"))
    with pytest.raises(DataValidationError, match="run migrate-provenance"):
        verify_processed(processed.paths, "train", processed.features, processed.dataset)
