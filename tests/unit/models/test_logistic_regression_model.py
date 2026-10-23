"""The logistic-regression reference (DR-011 §5; DR-015 §8; Issue #151), on synthetic data only."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from review_capacity.errors import (
    DataValidationError,
    EnvironmentMismatchError,
    FileAccessError,
    ModelError,
    SpecificationError,
)
from review_capacity.features.logistic_inputs import build_logistic_matrix, logistic_columns
from review_capacity.features.neural_preprocessing import fit_neural_preprocessing
from review_capacity.models import logistic_regression_model as module
from review_capacity.models.base import ModelData
from review_capacity.models.logistic_regression_model import (
    ARRAYS_FILE,
    MANIFEST_FILE,
    LogisticRegressionReference,
)
from review_capacity.training.declaration import load_declaration
from support.preprocessing import MISSING, canonical_frame, fitted_artifact
from support.synthetic import FEATURES, REPO_PATHS

FIXED = dict(load_declaration(REPO_PATHS)[0].space("logistic_regression").fixed)


def train_data(rows: int = 400, seed: int = 1) -> tuple[Any, ModelData]:
    """The artifact and a Train partition whose label depends on the inputs, with some noise."""
    artifact, (frame, _, _) = fitted_artifact(rows, seed)
    rng = np.random.default_rng(3)
    signal = frame["V1"].fillna(0.0).to_numpy() > 1.0
    labels = np.where(rng.random(rows) < 0.1, ~signal, signal).astype(np.int8)
    return artifact, ModelData(frame, labels, np.ones(rows, dtype=np.float64))


def fitted(c: float = 1.0) -> tuple[LogisticRegressionReference, ModelData]:
    artifact, data = train_data()
    reference = LogisticRegressionReference(artifact, FIXED)
    reference.fit(data, {"C": c})
    return reference, data


# --- construction and hyperparameters ----------------------------------------------------------


def test_the_declared_settings_are_the_fixed_ones() -> None:
    assert FIXED == {
        "solver": "lbfgs",
        "l1_ratio": 0,
        "fit_intercept": True,
        "class_weight": None,
        "warm_start": False,
        "tol": 1e-6,
        "max_iter": 1000,
    }
    reference, _ = fitted(2.5)
    estimator = reference._fitted()
    params = estimator.get_params()
    assert {name: params[name] for name in FIXED} == FIXED
    assert params["C"] == 2.5
    assert reference.parameters["requested"] == {"C": 2.5}
    assert reference.parameters["fixed"] == FIXED


@pytest.mark.parametrize(
    "fixed",
    [
        {k: v for k, v in FIXED.items() if k != "tol"},
        {**FIXED, "extra": 1},
        {**FIXED, "class_weight": "balanced"},
        {**FIXED, "solver": "saga"},
        {**FIXED, "warm_start": True},
    ],
)
def test_the_fixed_settings_must_be_exactly_the_declared_unweighted_lbfgs_fit(
    fixed: dict[str, Any],
) -> None:
    artifact, _ = train_data()
    with pytest.raises(SpecificationError):
        LogisticRegressionReference(artifact, fixed)


@pytest.mark.parametrize(
    ("hyperparameters", "message"),
    [
        ({}, "one hyperparameter"),
        ({"C": 1.0, "tol": 1e-3}, "override fixed"),
        ({"C": 1.0, "penalty": "l2"}, "one hyperparameter"),
        ({"tol": 1e-3}, "override fixed"),
        ({"C": 0.0}, "positive"),
        ({"C": -1.0}, "positive"),
        ({"C": float("nan")}, "positive"),
        ({"C": float("inf")}, "positive"),
        ({"C": True}, "real number"),
        ({"C": "1"}, "real number"),
    ],
)
def test_only_a_positive_finite_c_may_be_searched(
    hyperparameters: dict[str, Any], message: str
) -> None:
    artifact, data = train_data()
    with pytest.raises(SpecificationError, match=message):
        LogisticRegressionReference(artifact, FIXED).fit(data, hyperparameters)


def test_the_reference_is_fitted_unweighted() -> None:
    artifact, data = train_data()
    weighted = ModelData(data.features, data.labels, np.full(len(data.labels), 3.0))
    with pytest.raises(SpecificationError, match="unweighted"):
        LogisticRegressionReference(artifact, FIXED).fit(weighted, {"C": 1.0})


# --- fitting and scoring -----------------------------------------------------------------------


def test_scores_are_float64_probabilities_that_follow_the_labels() -> None:
    reference, data = fitted()
    scores = reference.predict(data.features)
    assert scores.dtype == np.float64
    assert scores.shape == (len(data.labels),)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert scores[data.labels == 1].mean() > scores[data.labels == 0].mean()
    assert reference.fitted_iterations == reference.parameters["n_iter"]
    assert reference.parameters["matrix"]["columns"] == len(logistic_columns(reference.artifact))


def test_two_fits_give_bitwise_identical_coefficients_and_scores() -> None:
    first, data = fitted()
    second, _ = fitted()
    estimators = (first._fitted(), second._fitted())
    assert estimators[0].coef_.tobytes() == estimators[1].coef_.tobytes()
    assert estimators[0].intercept_.tobytes() == estimators[1].intercept_.tobytes()
    assert first.predict(data.features).tobytes() == second.predict(data.features).tobytes()


def test_the_fit_equals_scikit_learn_fitted_directly_on_the_same_matrix() -> None:
    reference, data = fitted(0.5)
    matrix = build_logistic_matrix(reference.artifact, data.features)
    direct = LogisticRegression(C=0.5, **FIXED).fit(matrix, data.labels)
    estimator = reference._fitted()
    np.testing.assert_array_equal(estimator.coef_, direct.coef_)
    np.testing.assert_array_equal(estimator.intercept_, direct.intercept_)
    np.testing.assert_array_equal(
        reference.predict(data.features), direct.predict_proba(matrix)[:, 1]
    )


def test_values_unseen_in_train_are_scored() -> None:
    reference, data = fitted()
    scored = data.features.head(5).copy()
    scored["ProductCD"] = pd.Series(["NEW", MISSING, "W", "C", "H"], dtype="str")
    assert len(reference.predict(scored)) == 5


def test_a_fit_that_does_not_converge_is_an_error_without_substitution() -> None:
    artifact, data = train_data()
    reference = LogisticRegressionReference(artifact, {**FIXED, "max_iter": 1})
    hook = warnings.showwarning
    with pytest.raises(ModelError, match="did not converge within max_iter = 1") as raised:
        reference.fit(data, {"C": 1.0})
    assert "failed to converge" in str(raised.value)  # the solver's own explanation is kept
    assert warnings.showwarning is hook
    with pytest.raises(ModelError, match="not fitted"):
        reference.predict(data.features)
    with pytest.raises(ModelError, match="not fitted"):
        reference.save(Path("never-written"))


class Broken:
    """An estimator whose fit returns the given (invalid) state."""

    def __init__(self, width: int, **state: Any) -> None:
        self.width, self.state = width, state

    def fit(self, _matrix: Any, _labels: Any) -> None:
        state = {
            "coef_": np.zeros((1, self.width)),
            "intercept_": np.zeros(1),
            "classes_": np.array([0, 1]),
            "n_iter_": np.array([3]),
        } | self.state
        for name, value in state.items():
            setattr(self, name, value)


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({"coef_": "nan"}, "not finite"),
        ({"intercept_": np.array([np.inf])}, "not finite"),
        ({"classes_": np.array([0, 2])}, "classes"),
        ({"coef_": np.zeros((1, 2))}, "shape"),
        ({"n_iter_": np.array([5000])}, "iterations"),
        ({"n_iter_": np.array([-1])}, "iterations"),
    ],
)
def test_an_invalid_fitted_state_is_refused_and_not_published(
    monkeypatch: pytest.MonkeyPatch, state: dict[str, Any], message: str
) -> None:
    artifact, data = train_data()
    width = len(logistic_columns(artifact))
    if isinstance(state.get("coef_"), str):
        state = {"coef_": np.full((1, width), np.nan)}
    monkeypatch.setattr(module, "_estimator", lambda _c, _fixed: Broken(width, **state))
    reference = LogisticRegressionReference(artifact, FIXED)
    with pytest.raises(ModelError, match=message):
        reference.fit(data, {"C": 1.0})
    with pytest.raises(ModelError, match="not fitted"):
        reference.predict(data.features)


def test_invalid_probabilities_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    reference, data = fitted()
    rows = len(data.labels)
    monkeypatch.setattr(
        reference._fitted(),
        "predict_proba",
        lambda _matrix: np.column_stack([np.zeros(rows), np.full(rows, 2.0)]),
    )
    with pytest.raises(ModelError, match="outside"):
        reference.predict(data.features)


def test_a_frame_without_the_declared_inputs_is_refused() -> None:
    reference, data = fitted()
    with pytest.raises(DataValidationError, match="columns"):
        reference.predict(data.features.drop(columns="id_01"))


# --- persistence -------------------------------------------------------------------------------


def saved(tmp_path: Path, name: str = "model") -> tuple[LogisticRegressionReference, Path, str]:
    reference, _ = fitted()
    directory = tmp_path / name
    return reference, directory, reference.save(directory)


def test_save_and_reload_give_identical_scores_and_parameters(tmp_path: Path) -> None:
    reference, directory, digest = saved(tmp_path)
    _, data = fitted()
    assert sorted(path.name for path in directory.iterdir()) == [ARRAYS_FILE, MANIFEST_FILE]
    loaded = LogisticRegressionReference.load(directory, reference.artifact, FIXED)
    assert loaded.predict(data.features).tobytes() == reference.predict(data.features).tobytes()
    assert loaded.parameters == reference.parameters
    assert json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))["state_sha256"] == (
        digest
    )
    with pytest.raises(FileAccessError, match="never replaced"):
        reference.save(directory)


def rewrite(
    directory: Path,
    *,
    arrays: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    recompute: bool = True,
) -> None:
    """Change the stored state; with ``recompute`` both digests are made valid again, so that
    only the semantic checks can refuse the change."""
    text = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    with np.load(directory / ARRAYS_FILE, allow_pickle=False) as archive:
        stored = {name: archive[name] for name in ("coefficients", "intercept")}
    stored.update(arrays or {})
    text.update(manifest or {})
    if recompute:
        text["arrays_sha256"] = module._arrays_sha256(stored["coefficients"], stored["intercept"])
        text["state_sha256"] = module._state_sha256(text)
    with (directory / ARRAYS_FILE).open("wb") as handle:
        np.savez(handle, coefficients=stored["coefficients"], intercept=stored["intercept"])
    (directory / MANIFEST_FILE).write_text(json.dumps(text), encoding="utf-8")


def test_changed_bytes_are_detected_by_the_digests(tmp_path: Path) -> None:
    reference, directory, _ = saved(tmp_path)
    with np.load(directory / ARRAYS_FILE, allow_pickle=False) as archive:
        coefficients = archive["coefficients"].copy()
    coefficients[0] += 1.0
    rewrite(directory, arrays={"coefficients": coefficients}, recompute=False)
    with pytest.raises(DataValidationError, match="arrays differ"):
        LogisticRegressionReference.load(directory, reference.artifact, FIXED)
    _, other, _ = saved(tmp_path, "second")
    rewrite(other, manifest={"n_iter": 3}, recompute=False)
    with pytest.raises(DataValidationError, match="manifest does not match"):
        LogisticRegressionReference.load(other, reference.artifact, FIXED)


def test_another_artifact_other_settings_or_other_packages_are_refused(tmp_path: Path) -> None:
    reference, directory, _ = saved(tmp_path)
    frame, ids, times = canonical_frame()
    other = fit_neural_preprocessing(frame, ids, times, FEATURES, 7)
    with pytest.raises(SpecificationError, match="different input matrix"):
        LogisticRegressionReference.load(directory, other, FIXED)
    with pytest.raises(SpecificationError, match="different fixed settings"):
        LogisticRegressionReference.load(directory, reference.artifact, {**FIXED, "tol": 1e-3})
    versions = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))["versions"]
    rewrite(directory, manifest={"versions": versions | {"scipy": "0.0.1"}})
    with pytest.raises(EnvironmentMismatchError, match="installed"):
        LogisticRegressionReference.load(directory, reference.artifact, FIXED)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("width", "match the matrix columns"),
        ("nan", "finite"),
        ("dtype", "float64"),
        ("c_zero", "malformed C"),
        ("iterations", "malformed iteration count"),
        ("negative", "malformed iteration count"),
    ],
)
def test_malformed_state_with_valid_digests_is_refused(
    tmp_path: Path, change: str, message: str
) -> None:
    reference, directory, _ = saved(tmp_path)
    with np.load(directory / ARRAYS_FILE, allow_pickle=False) as archive:
        coefficients = archive["coefficients"].copy()
    arrays: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    if change == "width":
        arrays["coefficients"] = coefficients[:-1]
    elif change == "nan":
        coefficients[0] = np.nan
        arrays["coefficients"] = coefficients
    elif change == "dtype":
        arrays["coefficients"] = coefficients.astype(np.float32)
    elif change == "c_zero":
        manifest["C"] = 0.0
    elif change == "negative":
        manifest["n_iter"] = -1
    else:
        max_iter = FIXED["max_iter"]
        assert isinstance(max_iter, int)
        manifest["n_iter"] = max_iter + 1
    rewrite(directory, arrays=arrays, manifest=manifest)
    with pytest.raises(DataValidationError, match=message):
        LogisticRegressionReference.load(directory, reference.artifact, FIXED)


def test_a_zero_iteration_state_is_valid_and_reloads(tmp_path: Path) -> None:
    reference, directory, _ = saved(tmp_path)
    rewrite(directory, manifest={"n_iter": 0})
    loaded = LogisticRegressionReference.load(directory, reference.artifact, FIXED)
    assert loaded.fitted_iterations == 0


def test_missing_files_and_foreign_manifests_are_project_errors(tmp_path: Path) -> None:
    artifact, _ = train_data()
    with pytest.raises(FileAccessError):
        LogisticRegressionReference.load(tmp_path / "absent", artifact, FIXED)
    _, directory, _ = saved(tmp_path)
    (directory / MANIFEST_FILE).write_text('{"kind": "other"}', encoding="utf-8")
    with pytest.raises(DataValidationError, match="expected keys"):
        LogisticRegressionReference.load(directory, artifact, FIXED)


def test_an_unfitted_model_cannot_be_saved(tmp_path: Path) -> None:
    artifact, _ = train_data()
    with pytest.raises(ModelError, match="not fitted"):
        LogisticRegressionReference(artifact, FIXED).save(tmp_path / "model")
