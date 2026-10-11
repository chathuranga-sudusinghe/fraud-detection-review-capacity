"""Checks on the local raw data, subject to the execution rules in AGENTS.md.

Skipped unless ``data/raw/`` holds both files **and** ``REVIEW_CAPACITY_INTEGRATION=1`` is set, so
that the real data are never read by an ordinary test run:

    REVIEW_CAPACITY_INTEGRATION=1 pytest -m integration

No Test-period label is read: the sealed-period request must fail.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from review_capacity.config import DatasetSpec, FeatureSpec, load_dataset_spec, load_feature_spec
from review_capacity.data.loading import RawData, load_labels, load_raw
from review_capacity.data.validation import check_file_checksums, validate
from review_capacity.errors import SealedTestError
from review_capacity.features.model_inputs import build_canonical_frame
from review_capacity.paths import ProjectPaths

PATHS = ProjectPaths.discover(start=Path(__file__).parent)
DATASET = load_dataset_spec(PATHS)
ENABLED = os.environ.get("REVIEW_CAPACITY_INTEGRATION") == "1"
PRESENT = DATASET.transaction.path.exists() and DATASET.identity.path.exists()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (ENABLED and PRESENT), reason="needs REVIEW_CAPACITY_INTEGRATION=1 and the raw files"
    ),
]


@pytest.fixture(scope="module")
def loaded() -> tuple[FeatureSpec, DatasetSpec, RawData]:
    features = load_feature_spec(PATHS.features_spec)
    check_file_checksums(DATASET)
    return features, DATASET, load_raw(features, DATASET)


def test_contract_reproduces_dr001_anchors(
    loaded: tuple[FeatureSpec, DatasetSpec, RawData],
) -> None:
    features, dataset, raw = loaded
    summaries = validate(raw, features, dataset)
    assert sum(s.rows for s in summaries) == 590_540


def test_development_labels_and_sealed_test(
    loaded: tuple[FeatureSpec, DatasetSpec, RawData],
) -> None:
    features, dataset, raw = loaded
    labels = load_labels(raw, features, dataset, ["train", "v1", "v2"])
    assert len(labels) == 511_998
    with pytest.raises(SealedTestError):
        load_labels(raw, features, dataset, ["test"])


def test_canonical_train_frame(loaded: tuple[FeatureSpec, DatasetSpec, RawData]) -> None:
    features, _, raw = loaded
    train = raw.joined[(raw.partition == "train").to_numpy()]
    frame = build_canonical_frame(train, features)
    assert frame.shape == (368_632, 433)
    assert frame[list(features.categorical)].notna().all().all()
    tod = frame[["tod_sin", "tod_cos"]].to_numpy()
    assert np.isfinite(tod).all()
    numeric_with_missing = frame[list(features.numeric_inputs)].isna().any().sum()
    assert numeric_with_missing == 324  # DR-012: one indicator per such input
