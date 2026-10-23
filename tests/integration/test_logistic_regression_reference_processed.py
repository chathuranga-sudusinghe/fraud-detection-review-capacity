"""The logistic-regression reference on the local processed development files (Issue #151).

Opt-in, as ``test_data_contract``: run only with ``REVIEW_CAPACITY_INTEGRATION=1``. Reads Train
through the verified reader (V1, V2 and the Test period are not read) and writes nothing. It is
the candidate check of DR-015 §8: it fits the preprocessing artifact, then each of the ten recorded
candidates of the reference with the declared settings on Train alone, and requires every fit to
converge and to give finite probabilities in [0, 1]. The iteration count and time of each fit are
in the log. No setting is chosen by a validation result.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from review_capacity.config import load_dataset_spec, load_feature_spec
from review_capacity.development.inputs import TRAIN, read_partitions
from review_capacity.features.neural_preprocessing import fit_neural_preprocessing
from review_capacity.models.logistic_regression_model import LogisticRegressionReference
from review_capacity.paths import ProjectPaths
from review_capacity.training.declaration import load_declaration
from review_capacity.training.sampler import generate_candidates

PATHS = ProjectPaths.discover(start=Path(__file__).parent)
ENABLED = os.environ.get("REVIEW_CAPACITY_INTEGRATION") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ENABLED, reason="opt-in: set REVIEW_CAPACITY_INTEGRATION=1"),
]


def test_all_ten_recorded_candidates_converge_on_train() -> None:
    features = load_feature_spec(PATHS.features_spec)
    declaration, _ = load_declaration(PATHS)
    train = read_partitions(PATHS, features, load_dataset_spec(PATHS), (TRAIN,))[TRAIN]
    artifact = fit_neural_preprocessing(
        train.features, train.ids, train.times, features, declaration.preprocessing_seed
    )
    fixed = declaration.space("logistic_regression").fixed
    max_iter = fixed["max_iter"]
    assert isinstance(max_iter, int)
    candidates = generate_candidates(declaration)["logistic_regression"].candidates
    assert len(candidates) == declaration.trials
    for number, candidate in enumerate(candidates, start=1):
        reference = LogisticRegressionReference(artifact, fixed)
        reference.fit(train.model_data(None), dict(candidate))  # raises if it does not converge
        assert reference.fitted_iterations <= max_iter, number
        scores = reference.predict(train.features)
        assert np.isfinite(scores).all(), number
