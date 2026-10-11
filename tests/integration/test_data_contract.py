"""Checks on the local processed development files, subject to the execution rules in AGENTS.md.

Opt-in: run only with ``REVIEW_CAPACITY_INTEGRATION=1``, so that an ordinary test run never reads
project data:

    REVIEW_CAPACITY_INTEGRATION=1 pytest -m integration

When enabled, a missing processed file fails the run instead of skipping it. No raw file is read
(the raw-file paths are replaced by paths that do not exist) and nothing is written. Every file is
read through the verified reader (``data.processed``): SHA-256 against its provenance record, the
recorded raw-source hashes against ``config/dataset.yaml``, the declared schema and column kinds,
the DR-001 identifier anchors. The checks below add the development partition anchors, membership,
temporal consistency, labels, the reconstruction of every partition file from the joined file
with the current code, and the feature catalog. No Test-period row exists in these files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from review_capacity.config import DatasetSpec, FeatureSpec, load_dataset_spec, load_feature_spec
from review_capacity.data.catalog import read_catalog
from review_capacity.data.partitions import assign_partitions, relative_day
from review_capacity.data.processed import (
    DAY_COLUMN,
    JOINED,
    PARTITION_COLUMN,
    read_development_features,
    read_processed,
    schema,
)
from review_capacity.data.profiling import compare_with_catalog
from review_capacity.features.model_inputs import build_canonical_frame
from review_capacity.hashing import id_set_sha256
from review_capacity.paths import ProjectPaths

PATHS = ProjectPaths.discover(start=Path(__file__).parent)
ENABLED = os.environ.get("REVIEW_CAPACITY_INTEGRATION") == "1"
DEVELOPMENT_ROWS = 511_998  # DR-001: Train 368,632 + V1 65,544 + V2 77,822

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ENABLED, reason="opt-in: set REVIEW_CAPACITY_INTEGRATION=1"),
]


@dataclass(frozen=True)
class Processed:
    features: FeatureSpec
    dataset: DatasetSpec
    joined: pd.DataFrame
    partitions: dict[str, pd.DataFrame]


def _without_raw_files(dataset: DatasetSpec) -> DatasetSpec:
    """The declaration with raw paths that do not exist: any raw read would fail."""
    absent = Path("/nonexistent/raw")
    return replace(
        dataset,
        transaction=replace(dataset.transaction, path=absent / dataset.transaction.path.name),
        identity=replace(dataset.identity, path=absent / dataset.identity.path.name),
    )


@pytest.fixture(scope="module")
def processed() -> Processed:
    features = load_feature_spec(PATHS.features_spec)
    dataset = _without_raw_files(load_dataset_spec(PATHS))
    joined = read_processed(PATHS, JOINED, features, dataset)
    partitions = {
        p.name: read_processed(PATHS, p.name, features, dataset) for p in dataset.development
    }
    return Processed(features, dataset, joined, partitions)


def test_partition_anchors(processed: Processed) -> None:
    key = processed.features.key
    for part in processed.dataset.development:
        frame = processed.partitions[part.name]
        assert len(frame) == part.rows
        assert id_set_sha256(frame[key]).startswith(part.transaction_id_sha256_prefix)
        assert frame[DAY_COLUMN].between(part.first_day, part.last_day).all()
        rows = processed.joined[processed.joined[PARTITION_COLUMN] == part.name]
        time = rows[processed.features.time]
        assert (int(time.min()), int(time.max())) == (
            part.transaction_dt_min,
            part.transaction_dt_max,
        )


def test_membership(processed: Processed) -> None:
    key = processed.features.key
    joined_ids = processed.joined[key]
    assert len(joined_ids) == DEVELOPMENT_ROWS
    assert joined_ids.notna().all()
    assert joined_ids.is_unique
    ids = [set(frame[key]) for frame in processed.partitions.values()]
    assert sum(len(s) for s in ids) == DEVELOPMENT_ROWS  # pairwise disjoint
    assert set.union(*ids) == set(joined_ids)


def test_temporal_consistency(processed: Processed) -> None:
    joined, features, dataset = processed.joined, processed.features, processed.dataset
    days = relative_day(joined[features.time])
    assert (joined[DAY_COLUMN] == days).all()
    assert (joined[PARTITION_COLUMN] == assign_partitions(days, dataset).astype(str)).all()
    assert int(days.min()) == dataset.first_day
    assert int(days.max()) == dataset.last_development_day  # no Test-period row
    assert joined[features.time].is_monotonic_increasing


def test_labels_are_binary_and_agree(processed: Processed) -> None:
    features = processed.features
    labels = processed.joined.set_index(features.key)[features.target]
    assert labels.isin([0, 1]).all()
    for frame in processed.partitions.values():
        stored = frame.set_index(features.key)[features.target]
        assert stored.equals(labels.loc[stored.index])


def test_partition_files_are_rebuilt_exactly_from_the_joined_file(processed: Processed) -> None:
    """The stored model inputs equal what the current code builds from the joined rows."""
    features = processed.features
    for name, stored in processed.partitions.items():
        rows = processed.joined[processed.joined[PARTITION_COLUMN] == name]
        rebuilt = pd.concat(
            [
                rows[[features.key, DAY_COLUMN, features.target]],
                build_canonical_frame(rows, features),
            ],
            axis=1,
        ).reset_index(drop=True)
        assert list(stored.columns) == list(schema(name, features))
        pd.testing.assert_frame_equal(stored.reset_index(drop=True), rebuilt, check_exact=True)


def test_time_of_day_inputs_are_bounded(processed: Processed) -> None:
    for frame in processed.partitions.values():
        values = frame[["tod_sin", "tod_cos"]].to_numpy()
        assert np.isfinite(values).all()
        assert np.allclose(values[:, 0] ** 2 + values[:, 1] ** 2, 1.0)


def test_numeric_inputs_with_missing_values_in_train(processed: Processed) -> None:
    train = processed.partitions["train"]
    with_missing = train[list(processed.features.numeric_inputs)].isna().any()
    assert int(with_missing.sum()) == 324  # DR-012 diagnostic count


def test_feature_catalog_is_reproduced(processed: Processed) -> None:
    data = read_development_features(PATHS, processed.features, processed.dataset)
    report = compare_with_catalog(
        data, processed.features, processed.dataset, read_catalog(PATHS.feature_catalog)
    )
    assert report.differences == []
