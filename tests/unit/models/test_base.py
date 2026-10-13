"""The common model interface: data checks, class weights, the DR-013 guard, score checks and the
model lifecycle (synthetic data only)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from review_capacity.config import SEED_LIMIT, ModelSettings, load_model_settings
from review_capacity.errors import DataValidationError, ModelError, SpecificationError
from review_capacity.models.base import (
    ClassWeights,
    ModelData,
    ScoringModel,
    check_fit_arguments,
    checked_scores,
    class_weights,
    merge_parameters,
    sample_weights,
)
from review_capacity.models.registry import Weighting, configuration
from support.models import MeanModel
from support.synthetic import REPO_PATHS


@pytest.fixture(scope="module")
def settings() -> ModelSettings:
    return load_model_settings(REPO_PATHS)


def _data(labels: list[int], weights: list[float] | None = None) -> ModelData:
    y = np.array(labels, dtype=np.int8)
    w = np.ones(len(y)) if weights is None else np.array(weights, dtype=np.float64)
    return ModelData(pd.DataFrame({"x": np.arange(len(y), dtype=np.float64)}), y, w)


def test_model_data_accepts_valid_input() -> None:
    assert len(_data([0, 1, 0]).features) == 3


@pytest.mark.parametrize(
    ("labels", "weights", "message"),
    [
        (np.array([0, 1], dtype=np.int8), np.ones(2), "feature rows"),
        (np.array([0, 1, 2], dtype=np.int8), np.ones(3), "0 or 1"),
        (np.array([0, 1, 0], dtype=np.int64), np.ones(3), "int8"),
        (np.array([[0, 1, 0]], dtype=np.int8), np.ones(3), "one-dimensional"),
        ([0, 1, 0], np.ones(3), "int8 array"),
        (np.array([0, 1, 0], dtype=np.int8), np.array([1.0, 0.0, 1.0]), "positive"),
        (np.array([0, 1, 0], dtype=np.int8), np.array([1.0, np.nan, 1.0]), "finite"),
        (np.array([0, 1, 0], dtype=np.int8), np.ones(3, dtype=np.float32), "float64"),
        (np.array([0, 1, 0], dtype=np.int8), [1.0, 1.0, 1.0], "NumPy array"),
        (np.array([0, 1, 0], dtype=np.int8), np.ones(2), "feature rows"),
    ],
)
def test_model_data_rejects_invalid_input(
    labels: npt.NDArray[Any], weights: npt.NDArray[Any], message: str
) -> None:
    features = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    with pytest.raises(DataValidationError, match=message):
        ModelData(features, labels, weights)


def test_model_data_rejects_non_dataframe_features() -> None:
    with pytest.raises(DataValidationError, match="DataFrame"):
        ModelData(
            np.zeros((2, 1)),  # type: ignore[arg-type]
            np.array([0, 1], dtype=np.int8),
            np.ones(2),
        )


def test_model_data_rejects_empty_and_duplicate_columns() -> None:
    empty = pd.DataFrame({"x": np.array([], dtype=np.float64)})
    with pytest.raises(DataValidationError, match="must not be empty"):
        ModelData(empty, np.array([], dtype=np.int8), np.array([], dtype=np.float64))
    duplicate = pd.DataFrame([[0.0, 1.0]], columns=["x", "x"])
    with pytest.raises(DataValidationError, match="unique"):
        ModelData(duplicate, np.array([1], dtype=np.int8), np.ones(1))


def test_balanced_weights_follow_the_protocol() -> None:
    labels = np.array([0, 0, 0, 1, 0, 0, 1], dtype=np.int8)  # 5 legitimate, 2 fraud
    weights = class_weights(labels, Weighting.BALANCED)
    assert weights == ClassWeights(positive=2.5)
    assert sample_weights(labels, weights).tolist() == [1, 1, 1, 2.5, 1, 1, 2.5]
    assert sample_weights(labels, weights).dtype == np.float64


def test_unweighted_gives_every_row_weight_one() -> None:
    labels = np.array([0, 1, 0], dtype=np.int8)
    weights = class_weights(labels, Weighting.UNWEIGHTED)
    assert sample_weights(labels, weights).tolist() == [1.0, 1.0, 1.0]


@pytest.mark.parametrize("weighting", [Weighting.UNWEIGHTED, Weighting.BALANCED])
def test_weights_check_labels_in_both_modes(weighting: Weighting) -> None:
    with pytest.raises(DataValidationError, match="0 or 1"):
        class_weights(np.array([0, 2, 1], dtype=np.int8), weighting)
    with pytest.raises(DataValidationError, match="non-empty"):
        class_weights(np.array([], dtype=np.int8), weighting)


def test_unsupported_weighting_is_a_defect() -> None:
    with pytest.raises(ValueError, match="Unsupported weighting"):
        class_weights(np.array([0, 1], dtype=np.int8), "x")  # type: ignore[arg-type]


def test_balanced_weights_need_both_classes() -> None:
    with pytest.raises(DataValidationError, match="both classes"):
        class_weights(np.zeros(4, dtype=np.int8), Weighting.BALANCED)


def test_sample_weights_check_labels() -> None:
    with pytest.raises(DataValidationError, match="0 or 1"):
        sample_weights(np.array([0, 3], dtype=np.int8), ClassWeights(positive=2.0))


@pytest.mark.parametrize("positive", [0.0, -1.0, float("nan"), float("inf")])
def test_class_weight_must_be_finite_and_positive(positive: float) -> None:
    with pytest.raises(DataValidationError, match="finite and positive"):
        ClassWeights(positive=positive)


CATBOOST_ALIASES = {"objective": "loss_function", "max_bin": "border_count"}


def test_fixed_settings_cannot_be_overridden() -> None:
    fixed = {"loss_function": "Logloss", "border_count": 254}
    merged = merge_parameters(fixed, {"depth": 6}, aliases=CATBOOST_ALIASES)
    assert merged == {**fixed, "depth": 6}
    with pytest.raises(SpecificationError, match=r"\['border_count'\]"):
        merge_parameters(fixed, {"border_count": 128}, aliases=CATBOOST_ALIASES)


def test_fixed_settings_cannot_be_overridden_through_an_alias() -> None:
    fixed = {"loss_function": "Logloss"}
    with pytest.raises(SpecificationError, match=r"\['loss_function'\]"):
        merge_parameters(fixed, {"objective": "CrossEntropy"}, aliases=CATBOOST_ALIASES)


def test_two_names_for_one_parameter_are_rejected() -> None:
    with pytest.raises(SpecificationError, match="more than one name"):
        merge_parameters({}, {"max_bin": 63, "border_count": 63}, aliases=CATBOOST_ALIASES)


def test_aliases_are_rewritten_to_canonical_names() -> None:
    merged = merge_parameters({}, {"max_bin": 63}, aliases=CATBOOST_ALIASES)
    assert merged == {"border_count": 63}


def test_checked_scores_accepts_real_scores() -> None:
    assert checked_scores([0.0, 0.5, 1.0], 3, "m").dtype == np.float64
    assert checked_scores(np.array([0, 1, 1]), 3, "m").tolist() == [0.0, 1.0, 1.0]


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        ([0.5, 0.5], "shape"),
        ([0.5, np.nan, 0.5], "non-finite"),
        ([0.5, 1.5, 0.5], r"outside \[0, 1\]"),
        (np.array([0.5, 0.5, 0.5], dtype=np.complex128), "non-real"),
        (np.array(["0.5", "0.5", "0.5"]), "non-real"),
        (np.array([True, False, True]), "non-real"),
        (np.ma.masked_array([0.5, 0.5, 0.5], mask=[False, True, False]), "masked"),
        ([[0.5], [0.1, 0.2], [0.3]], "invalid scores"),
    ],
)
def test_checked_scores_rejects_invalid_output(scores: Any, message: str) -> None:
    with pytest.raises(ModelError, match=message):
        checked_scores(scores, 3, "m")


def test_fit_arguments_need_exactly_one_stopping_rule() -> None:
    data = _data([0, 1])
    check_fit_arguments(data, None)
    check_fit_arguments(None, 100)
    with pytest.raises(ValueError, match="exactly one"):
        check_fit_arguments(data, 100)
    with pytest.raises(ValueError, match="exactly one"):
        check_fit_arguments(None, None)


@pytest.mark.parametrize("iterations", [True, -1, 0, 1.5])
def test_invalid_fixed_iteration_count(iterations: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        check_fit_arguments(None, iterations)


def test_interface_is_abstract(settings: ModelSettings) -> None:
    with pytest.raises(TypeError):
        ScoringModel(  # type: ignore[abstract]
            configuration("LGB-u"), 42, settings=settings, device="gpu"
        )


@pytest.mark.parametrize(
    ("seed", "device", "error"),
    [
        (-1, "gpu", ValueError),
        (0.5, "gpu", TypeError),
        (True, "gpu", TypeError),
        (42, "tpu", ValueError),
        (SEED_LIMIT, "gpu", ValueError),
    ],
)
def test_invalid_seed_or_device_is_a_defect(
    settings: ModelSettings, seed: Any, device: Any, error: type[Exception]
) -> None:
    with pytest.raises(error):
        MeanModel(configuration("LGB-u"), seed, settings=settings, device=device)


def test_a_model_fits_scores_saves_and_reloads(tmp_path: Path, settings: ModelSettings) -> None:
    train = _data([0, 1, 0, 0, 0], [1.0, 3.0, 1.0, 1.0, 1.0])  # weighted fraud rate 3 / 7
    model = MeanModel(configuration("LGB-b"), 42, settings=settings, device="gpu")
    model.fit(train, train, {"num_leaves": 31})
    scores = model.predict(train.features)
    assert scores.tolist() == pytest.approx([3 / 7] * 5)
    model.save(tmp_path / "model")
    reloaded = MeanModel.load(
        tmp_path / "model", configuration("LGB-b"), 42, settings=settings, device="gpu"
    )
    assert reloaded.predict(train.features).tolist() == scores.tolist()
    assert reloaded.parameters == model.parameters
    assert (reloaded.configuration, reloaded.seed) == (model.configuration, model.seed)
    assert reloaded.fitted_iterations == model.fitted_iterations
    assert model.parameters["resolved"]["rows"] == 5


def test_reload_rejects_another_configuration(tmp_path: Path, settings: ModelSettings) -> None:
    model = MeanModel(configuration("LGB-b"), 42, settings=settings, device="gpu")
    model.fit(_data([0, 1]), None, {}, iterations=10)
    assert model.fitted_iterations == 10
    model.save(tmp_path / "model")
    with pytest.raises(ModelError, match="does not match"):
        MeanModel.load(
            tmp_path / "model", configuration("Cat-b"), 42, settings=settings, device="gpu"
        )


@pytest.mark.parametrize("access", ["predict", "save", "parameters", "fitted_iterations"])
def test_unfitted_access_is_rejected(access: str, settings: ModelSettings, tmp_path: Path) -> None:
    model = MeanModel(configuration("LGB-u"), 42, settings=settings, device="cpu")
    actions: dict[str, Callable[[], object]] = {
        "predict": lambda: model.predict(pd.DataFrame({"x": [0.0]})),
        "save": lambda: model.save(tmp_path / "model"),
        "parameters": lambda: model.parameters,
        "fitted_iterations": lambda: model.fitted_iterations,
    }
    with pytest.raises(ModelError, match="not fitted"):
        actions[access]()


def test_reload_rejects_another_seed_or_format(tmp_path: Path, settings: ModelSettings) -> None:
    model = MeanModel(configuration("LGB-b"), 42, settings=settings, device="gpu")
    model.fit(_data([0, 1]), None, {}, iterations=10)
    model.save(tmp_path / "model")
    with pytest.raises(ModelError, match="does not match"):
        MeanModel.load(
            tmp_path / "model", configuration("LGB-b"), 7, settings=settings, device="gpu"
        )
    manifest = tmp_path / "model" / "model.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps({**payload, "format_version": 99}), encoding="utf-8")
    with pytest.raises(ModelError, match="Unsupported format version"):
        MeanModel.load(
            tmp_path / "model", configuration("LGB-b"), 42, settings=settings, device="gpu"
        )
