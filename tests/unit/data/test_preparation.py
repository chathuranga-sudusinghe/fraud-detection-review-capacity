"""End-to-end data preparation on synthetic files: manifest and processed development files."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from review_capacity.data import preparation
from review_capacity.data.preparation import prepare_data
from review_capacity.errors import SpecificationError
from review_capacity.hashing import file_sha256
from support.synthetic import Synthetic, with_partition


def test_check_only_writes_nothing(synthetic: Synthetic) -> None:
    result = prepare_data(
        synthetic.features,
        synthetic.dataset,
        synthetic.paths,
        write_processed=False,
        config_files=(),
    )
    assert result.written == []
    assert not synthetic.paths.manifest.exists()
    assert not synthetic.paths.processed_dir.exists()


def test_manifest_is_a_label_free_data_package(synthetic: Synthetic) -> None:
    prepare_data(synthetic.features, synthetic.dataset, synthetic.paths, config_files=())
    package = json.loads(synthetic.paths.manifest.read_text(encoding="utf-8"))
    assert package["$schema"] == "https://datapackage.org/profiles/2.0/datapackage.json"
    paths = [r["path"] for r in package["resources"]]
    assert paths == ["raw/train_transaction.csv", "raw/train_identity.csv"]
    for resource in package["resources"]:
        assert resource["hash"].startswith("sha256:") and len(resource["hash"]) == 7 + 64
        assert resource["bytes"] > 0
    partitions = package["_split"]["partitions"]
    assert [p["name"] for p in partitions] == ["train", "v1", "v2", "test"]
    assert partitions[-1]["sealed"] is True
    assert all(
        set(p)
        == {
            "name",
            "first_day",
            "last_day",
            "rows",
            "transaction_dt_min",
            "transaction_dt_max",
            "transaction_id_sha256",
            "sealed",
        }
        for p in partitions
    )
    assert "isFraud" not in json.dumps(package)


def test_processed_files_cover_development_only(synthetic: Synthetic) -> None:
    result = prepare_data(synthetic.features, synthetic.dataset, synthetic.paths, config_files=())
    processed = synthetic.paths.processed_dir
    names = sorted(p.name for p in processed.iterdir())
    assert names == [
        "development_joined.parquet",
        "development_joined.provenance.json",
        "train.parquet",
        "train.provenance.json",
        "v1.parquet",
        "v1.provenance.json",
        "v2.parquet",
        "v2.provenance.json",
    ]
    assert len(result.written) == 5

    train = pd.read_parquet(processed / "train.parquet")
    expected = ["TransactionID", "relative_day", "isFraud", *synthetic.features.model_inputs]
    assert list(train.columns) == expected
    assert len(train) == 8 and train["relative_day"].between(1, 2).all()

    joined = pd.read_parquet(processed / "development_joined.parquet")
    assert len(joined) == 16 and set(joined["partition"]) == {"train", "v1", "v2"}

    record = json.loads((processed / "train.provenance.json").read_text(encoding="utf-8"))
    assert record["rows"] == 8 and record["scope"] == "train"
    assert len(record["sha256"]) == 64
    assert set(record["sources"]) == {"train_transaction.csv", "train_identity.csv"}
    assert record["code"]["package"]["name"] == "review_capacity"


def test_manifest_records_rows_per_raw_file(synthetic: Synthetic) -> None:
    prepare_data(synthetic.features, synthetic.dataset, synthetic.paths, config_files=())
    package = json.loads(synthetic.paths.manifest.read_text(encoding="utf-8"))
    assert [r["_rows"] for r in package["resources"]] == [20, 10]
    assert package["_split"]["relative_day"] == "TransactionDT // 86400"


def test_every_development_file_round_trips_with_matching_record(synthetic: Synthetic) -> None:
    prepare_data(synthetic.features, synthetic.dataset, synthetic.paths, config_files=())
    expected_rows = {"development_joined": 16, "train": 8, "v1": 4, "v2": 4}
    for stem, rows in expected_rows.items():
        path = synthetic.paths.processed_file(stem)
        frame = pd.read_parquet(path)
        record = json.loads(synthetic.paths.provenance_file(stem).read_text(encoding="utf-8"))
        assert len(frame) == rows == record["rows"]
        assert record["sha256"] == file_sha256(path)
        assert record["parquet_engine"] == "pyarrow"
        assert len(record["code"]["package"]["source_sha256"]) == 64
        if stem != "development_joined":
            assert list(frame.columns)[3:] == list(synthetic.features.model_inputs)
            assert frame["isFraud"].isin([0, 1]).all()


def test_interrupted_write_leaves_no_stale_record(
    synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_data(synthetic.features, synthetic.dataset, synthetic.paths, config_files=())
    record = synthetic.paths.provenance_file("development_joined")
    assert record.exists()

    def failing_write(path: Path, content: dict[str, Any]) -> None:
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(preparation, "write_record", failing_write)
    with pytest.raises(RuntimeError, match="simulated failure"):
        prepare_data(synthetic.features, synthetic.dataset, synthetic.paths, config_files=())
    assert not record.exists()  # the old record no longer sits beside the replaced data file
    assert not list(synthetic.paths.processed_dir.glob(".*.tmp"))


def test_unsealed_configuration_refused_before_reading(synthetic: Synthetic) -> None:
    unsealed = with_partition(synthetic.dataset, "test", sealed=False)
    missing = replace(
        unsealed, transaction=replace(unsealed.transaction, path=Path("/nonexistent.csv"))
    )
    with pytest.raises(SpecificationError, match="must be sealed"):
        prepare_data(synthetic.features, missing, synthetic.paths, config_files=())
