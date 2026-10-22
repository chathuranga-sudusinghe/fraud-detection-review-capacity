"""The shared neural preprocessing on the local processed development files (DR-012; Issue #149).

Opt-in, as ``test_data_contract``: run only with ``REVIEW_CAPACITY_INTEGRATION=1``, so that an
ordinary test run never reads project data. Reads Train and Validation 1 through the verified
reader; the Test period is never read and nothing is written.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from review_capacity.config import load_dataset_spec, load_feature_spec
from review_capacity.development.inputs import TRAIN, V1, read_partitions
from review_capacity.features.neural_preprocessing import fit_neural_preprocessing
from review_capacity.paths import ProjectPaths

PATHS = ProjectPaths.discover(start=Path(__file__).parent)
ENABLED = os.environ.get("REVIEW_CAPACITY_INTEGRATION") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ENABLED, reason="opt-in: set REVIEW_CAPACITY_INTEGRATION=1"),
]


def test_the_train_fit_has_324_indicators_and_757_inputs() -> None:
    features = load_feature_spec(PATHS.features_spec)
    parts = read_partitions(PATHS, features, load_dataset_spec(PATHS), (TRAIN, V1))
    train, v1 = parts[TRAIN], parts[V1]
    artifact = fit_neural_preprocessing(train.features, train.ids, train.times, features, 0)
    assert artifact.requested_quantiles == 1_000
    assert len(artifact.numeric_inputs) == 384
    assert len(artifact.indicators) == 324
    layout = artifact.layout
    assert len(layout.numeric) + len(layout.categorical) == 757
    inputs = artifact.to_neural_inputs(v1.features)
    inputs.check(layout)
    assert inputs.rows == v1.rows
