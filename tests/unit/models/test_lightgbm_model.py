"""The LightGBM model on the common interface, fitted on small synthetic data on the CPU (no
project data; the GPU path is exercised by the GPU probe, Issue #77)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from review_capacity.config import ModelSettings, load_model_settings
from review_capacity.errors import DataValidationError, ModelError, SpecificationError
from review_capacity.models import lightgbm_model
from review_capacity.models.base import ModelData
from review_capacity.models.lightgbm_model import (
    LightGBMModel,
    lightgbm_aliases,
    lightgbm_parameters,
    resolved_parameters,
)
from review_capacity.models.registry import configuration
from review_capacity.models.v1_objective import V1Objective
from support.synthetic import FEATURES, REPO_PATHS

HYPERPARAMETERS: dict[str, Any] = {
    "num_iterations": 60,
    "early_stopping_round": 5,
    "metric": "binary_logloss",
    "learning_rate": 0.1,
    "num_leaves": 7,
    "min_data_in_leaf": 10,
}


@pytest.fixture(autouse=True)
def restore_lightgbm_logger() -> Iterator[None]:
    """LightGBM's registered logger is process-wide: restore it after each test."""
    import lightgbm
    from lightgbm import basic

    previous = (basic._LOGGER, basic._INFO_METHOD_NAME, basic._WARNING_METHOD_NAME)
    lightgbm_model._route_library_messages.cache_clear()
    yield
    lightgbm.register_logger(
        previous[0], info_method_name=previous[1], warning_method_name=previous[2]
    )
    lightgbm_model._route_library_messages.cache_clear()


@pytest.fixture(scope="module")
def settings() -> ModelSettings:
    return load_model_settings(REPO_PATHS)


def _frame(rows: int, seed: int) -> tuple[pd.DataFrame, npt.NDArray[np.int8]]:
    """Canonical model inputs (text categoricals, float64 numerics) and labels that depend on
    a categorical and a numeric input."""
    rng = np.random.default_rng(seed)
    product = rng.choice(["W", "C", "H", FEATURES.missing_label], rows)
    amount = rng.gamma(2.0, 50.0, rows)
    amount[rng.random(rows) < 0.05] = np.nan
    columns: dict[str, Any] = {
        "ProductCD": product,
        "card1": rng.choice(["1001", "1002", "1003", "1004"], rows),
        "DeviceType": rng.choice(["desktop", "mobile", FEATURES.missing_label], rows),
        "id_13": rng.choice(["49", "52", FEATURES.missing_label], rows),
        "TransactionAmt": amount,
        "V1": rng.normal(size=rows),
        "id_01": rng.normal(size=rows),
        "tod_sin": rng.uniform(-1, 1, rows),
        "tod_cos": rng.uniform(-1, 1, rows),
    }
    frame = pd.DataFrame(columns)
    for name in FEATURES.categorical:
        frame[name] = frame[name].astype("str")
    risk = (product == "C") * 1.5 + np.nan_to_num(amount, nan=100.0) / 200.0 - 2.0
    labels = (rng.random(rows) < 1 / (1 + np.exp(-risk))).astype(np.int8)
    return frame[list(FEATURES.model_inputs)], labels


def _data(rows: int, seed: int, fraud_weight: float = 1.0) -> ModelData:
    frame, labels = _frame(rows, seed)
    return ModelData(frame, labels, np.where(labels == 1, fraud_weight, 1.0))


def _model(settings: ModelSettings, name: str = "LGB-u", seed: int = 42) -> LightGBMModel:
    return LightGBMModel(
        configuration(name), seed, settings=settings, features=FEATURES, device="cpu"
    )


@pytest.fixture(scope="module")
def train() -> ModelData:
    return _data(600, 1)


@pytest.fixture(scope="module")
def valid() -> ModelData:
    return _data(300, 2)


UNCHANGED = "cannot be passed to LightGBM unchanged"


def test_alias_table_is_lightgbm_own() -> None:
    aliases = lightgbm_aliases()
    assert aliases["max_bins"] == "max_bin"
    assert aliases["random_state"] == "seed"
    assert aliases["n_jobs"] == "num_threads"
    assert aliases["early_stopping_rounds"] == "early_stopping_round"
    assert aliases["data_seed"] == "data_random_seed"
    assert "max_bin" not in aliases  # canonical names map to themselves implicitly
    parameters = lightgbm_parameters()
    assert {"num_leaves", "max_bin", "bagging_by_query"} <= parameters  # the last has no alias
    assert "num_leafs" not in parameters
    assert set(aliases.values()) <= parameters


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({'"is_unbalance"': True}, "Unknown LightGBM parameter"),
        ({"bagging_freq=0 data_seed": 7}, "Unknown LightGBM parameter"),
        ({"num_leafs": 64}, "Unknown LightGBM parameter"),  # a typo, not an alias
        ({"num_leaves": None}, "LightGBM's default"),
        ({"learning_rate": {"value": 0.1}}, "unsupported value"),
        ({"learning_rate": np.array(["0.1 data_seed=7"])}, UNCHANGED),
        ({"learning_rate": Path("0.1 data_seed=7")}, UNCHANGED),
        ({"metric": "auc=ignored"}, UNCHANGED),
        ({"boosting": '"dart"'}, UNCHANGED),
        ({"metric": "\x00auc"}, UNCHANGED),  # a NUL ends the C string
        ({"learning_rate": np.array(["0.\x001"])}, UNCHANGED),  # NumPy drops only trailing NULs
        ({"metric": "auc\x01"}, UNCHANGED),  # any other control character
        ({"metric": "auc\u00e9"}, UNCHANGED),  # non-ASCII (é)
    ],
)
def test_native_parameter_encoding_cannot_bypass_the_rules(
    settings: ModelSettings, change: dict[str, Any], message: str
) -> None:
    with pytest.raises(SpecificationError, match=message):
        _model(settings)._parameters_for_fit({**HYPERPARAMETERS, **change}, iterations=None)


def test_numeric_arrays_and_nested_lists_are_accepted(settings: ModelSettings) -> None:
    constraints = {
        "monotone_constraints": np.array([1, 0, -1]),
        "interaction_constraints": [[0, 1], [2, 3]],
    }
    params = _model(settings)._parameters_for_fit({**HYPERPARAMETERS, **constraints}, None)
    assert params["interaction_constraints"] == [[0, 1], [2, 3]]


def test_fit_with_early_stopping_and_cpu_prediction(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    model = _model(settings)
    model.fit(train, valid, HYPERPARAMETERS)
    assert 1 <= model.fitted_iterations <= 60
    scores = model.predict(valid.features)
    assert scores.shape == (300,)
    assert scores.dtype == np.float64
    assert ((scores >= 0) & (scores <= 1)).all()
    parameters = model.parameters
    assert parameters["requested"]["seed"] == 42
    assert parameters["requested"]["deterministic"] is True
    assert parameters["requested"]["enable_bundle"] is False
    assert parameters["requested"]["verbosity"] == 1
    resolved = parameters["resolved"]
    assert (resolved["max_bin"], resolved["seed"], resolved["device_type"]) == ("255", "42", "cpu")
    assert "data_random_seed" in resolved  # derived from seed and recorded
    assert parameters["prediction"] == {"backend": "cpu", "num_threads": 24}


def test_v1_weights_enter_the_early_stopping_metric(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    weights = np.where(valid.labels == 1, 3.0, 1.0)
    weighted = ModelData(valid.features, valid.labels, weights)
    model = _model(settings)
    model.fit(train, weighted, HYPERPARAMETERS)
    p = np.clip(model.predict(valid.features), 1e-15, 1 - 1e-15)
    losses = -(valid.labels * np.log(p) + (1 - valid.labels) * np.log1p(-p))
    observed = model._fitted().best_score["v1"]["binary_logloss"]
    np.testing.assert_allclose(observed, np.average(losses, weights=weights), rtol=1e-6)


def test_train_weights_change_the_fit(settings: ModelSettings, valid: ModelData) -> None:
    unweighted, weighted = _model(settings), _model(settings, "LGB-b")
    unweighted.fit(_data(600, 1), None, HYPERPARAMETERS, iterations=10)
    weighted.fit(_data(600, 1, fraud_weight=5.0), None, HYPERPARAMETERS, iterations=10)
    assert not np.allclose(unweighted.predict(valid.features), weighted.predict(valid.features))


def test_refit_at_a_fixed_iteration_count(settings: ModelSettings, train: ModelData) -> None:
    model = _model(settings)
    model.fit(train, None, HYPERPARAMETERS, iterations=12)
    assert model.fitted_iterations == 12
    assert model.parameters["requested"]["num_iterations"] == 12
    assert "early_stopping_round" not in model.parameters["requested"]


def test_refit_that_cannot_grow_every_tree_stops(
    settings: ModelSettings, train: ModelData, caplog: pytest.LogCaptureFixture
) -> None:
    no_split = {**HYPERPARAMETERS, "min_data_in_leaf": 10_000}
    with (
        caplog.at_level(logging.INFO, logger="review_capacity"),
        pytest.raises(ModelError, match="trees"),
    ):
        _model(settings).fit(train, None, no_split, iterations=5)
    steps = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Fit LGB-u")]
    assert any(": failed after" in m for m in steps)
    assert not any(": done in" in m for m in steps)


def _raise_lightgbm_error(*_args: Any, **_kwargs: Any) -> Any:
    from lightgbm.basic import LightGBMError

    raise LightGBMError("injected")


def test_native_training_failure_is_a_model_error(
    settings: ModelSettings, train: ModelData, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lightgbm.basic import LightGBMError

    monkeypatch.setattr("lightgbm.train", _raise_lightgbm_error)
    with pytest.raises(ModelError, match="training failed: injected") as raised:
        _model(settings).fit(train, None, HYPERPARAMETERS, iterations=3)
    assert isinstance(raised.value.__cause__, LightGBMError)


@pytest.mark.parametrize(
    ("method", "use", "message"),
    [
        ("predict", lambda m, data, _path: m.predict(data.features), "prediction failed"),
        ("model_to_string", lambda m, _data, path: m.save(path), "text unavailable"),
    ],
)
def test_native_booster_failure_is_a_model_error(
    settings: ModelSettings,
    train: ModelData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    use: Callable[[LightGBMModel, ModelData, Path], object],
    message: str,
) -> None:
    from lightgbm.basic import LightGBMError

    model = _model(settings)
    model.fit(train, None, HYPERPARAMETERS, iterations=3)
    monkeypatch.setattr(f"lightgbm.Booster.{method}", _raise_lightgbm_error)
    with pytest.raises(ModelError, match=message) as raised:
        use(model, train, tmp_path / "model")
    assert isinstance(raised.value.__cause__, LightGBMError)


def test_same_seed_gives_identical_cpu_scores(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    first, second = _model(settings), _model(settings)
    first.fit(train, valid, HYPERPARAMETERS)
    second.fit(train, valid, HYPERPARAMETERS)
    assert np.array_equal(first.predict(valid.features), second.predict(valid.features))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"early_stopping_round": 0}, "early_stopping_round must be a positive integer"),
        ({"early_stopping_round": True}, "early_stopping_round must be a positive integer"),
        ({"num_iterations": -1}, "num_iterations must be a positive integer"),
        ({"metric": ""}, "empty value"),
        ({"metric": "None"}, "explicit evaluation metric"),
        ({"metric": []}, "empty value"),
        ({"metric": 5}, "explicit evaluation metric"),
        ({"metric": "auc,"}, "explicit evaluation metric"),
        ({"metric": ["auc", "none"]}, "explicit evaluation metric"),
        ({"metric": " auc "}, UNCHANGED),
        ({"metric": "auc, binary_logloss"}, UNCHANGED),
        ({"metric": ["auc", "binary_logloss "]}, UNCHANGED),
        ({"boosting": "gbdt "}, UNCHANGED),
        ({"boosting": "dart"}, "DART"),
        ({"verbose": -1}, "verbosity"),
    ],
)
def test_early_stopping_settings_must_be_valid(
    settings: ModelSettings,
    train: ModelData,
    valid: ModelData,
    change: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(SpecificationError, match=message):
        _model(settings).fit(train, valid, {**HYPERPARAMETERS, **change})


def test_metric_is_normalised_to_one_name_per_entry(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    model = _model(settings)
    model.fit(train, valid, {**HYPERPARAMETERS, "metric": "binary_logloss,auc"})
    assert model.parameters["requested"]["metric"] == ["binary_logloss", "auc"]


def test_early_stopping_metric_must_be_set(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    without_metric = {k: v for k, v in HYPERPARAMETERS.items() if k != "metric"}
    with pytest.raises(SpecificationError, match=r"\['metric'\]"):
        _model(settings).fit(train, valid, without_metric)


@pytest.mark.parametrize(
    "name",
    [
        "max_bin",
        "max_bins",
        "random_state",
        "device",
        "gpu_use_dp",
        "gpu_device_id",
        "gpu_device_id_list",
        "num_gpu",
        "data_seed",
        "bagging_seed",
        "is_unbalance",
        "scale_pos_weight",
        "cat_feature",
        "pos_bagging_fraction",
        "neg_bagging_fraction",
        "neg_bagging",
    ],
)
def test_reserved_settings_cannot_be_tuned(
    settings: ModelSettings, train: ModelData, valid: ModelData, name: str
) -> None:
    with pytest.raises(SpecificationError, match="cannot override fixed research settings"):
        _model(settings).fit(train, valid, {**HYPERPARAMETERS, name: 1})


def test_unseen_category_is_scored_as_missing(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    model = _model(settings)
    model.fit(train, valid, HYPERPARAMETERS)
    expected = [sorted(set(train.features[name])) for name in FEATURES.categorical]
    assert model._fitted().pandas_categorical == expected
    unseen, missing = valid.features.copy(), valid.features.copy()
    unseen["card1"] = "9999"  # never seen in Train
    missing["card1"] = pd.Series(pd.NA, index=missing.index, dtype="str")
    assert np.array_equal(model.predict(unseen), model.predict(missing))


def test_inputs_must_be_the_canonical_inputs(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    model = _model(settings)
    reordered = train.features[list(reversed(train.features.columns))]
    with pytest.raises(DataValidationError, match="canonical inputs"):
        model.fit(ModelData(reordered, train.labels, train.weights), valid, HYPERPARAMETERS)


def test_library_messages_reach_the_run_log(
    settings: ModelSettings, train: ModelData, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="review_capacity"):
        _model(settings).fit(train, None, HYPERPARAMETERS, iterations=3)
    messages = [r.getMessage() for r in caplog.records if r.name == lightgbm_model.__name__]
    assert any(m.startswith("[LightGBM]") for m in messages)
    assert any(m.startswith("Fit LGB-u on cpu") for m in messages)


def test_library_warnings_are_logged_as_warnings(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="review_capacity"):
        lightgbm_model._LibraryLogger.info("[LightGBM] [Warning] something\n")
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.WARNING, "[LightGBM] [Warning] something")
    ]


def _saved(settings: ModelSettings, train: ModelData, directory: Path) -> LightGBMModel:
    model = _model(settings, "LGB-b")
    model.fit(train, None, HYPERPARAMETERS, iterations=5)
    model.save(directory)
    return model


def _reload(
    settings: ModelSettings, directory: Path, seed: int = 42, device: Any = "cpu"
) -> LightGBMModel:
    return LightGBMModel.load(
        directory,
        configuration("LGB-b"),
        seed,
        settings=settings,
        features=FEATURES,
        device=device,
    )


def test_save_and_reload(
    settings: ModelSettings, train: ModelData, valid: ModelData, tmp_path: Path
) -> None:
    model = _saved(settings, train, tmp_path / "model")
    reloaded = _reload(settings, tmp_path / "model")
    assert np.array_equal(reloaded.predict(valid.features), model.predict(valid.features))
    assert reloaded.fitted_iterations == model.fitted_iterations == 5
    assert reloaded.parameters["requested"] == model.parameters["requested"]


def test_v1_fit_reloads_with_its_metric_list(
    settings: ModelSettings, train: ModelData, valid: ModelData, tmp_path: Path
) -> None:
    model = _model(settings, "LGB-b")
    model.fit(train, valid, {**HYPERPARAMETERS, "metric": "binary_logloss,auc"})
    model.save(tmp_path / "model")
    reloaded = _reload(settings, tmp_path / "model")
    assert reloaded.parameters["requested"]["metric"] == ["binary_logloss", "auc"]
    assert reloaded.fitted_iterations == model.fitted_iterations


def _edit_manifest(directory: Path, edit: Callable[[dict[str, Any]], object]) -> None:
    path = directory / "manifest.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    edit(content)
    path.write_text(json.dumps(content), encoding="utf-8")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda m: m.update(format_version=99), "format version"),
        (lambda m: m.update(library_version="4.6.0"), "does not match the requested"),
        (lambda m: m.update(device="gpu"), "does not match the requested"),
        (lambda m: m.update(iterations=3), "tree count does not match"),
        (lambda m: m.update(iterations=0), "positive integer"),
        (lambda m: m["categories"].update(card1=["1001"]), "category lists do not match"),
        (lambda m: m["categories"].update(card1=[]), "Invalid saved category list"),
        (lambda m: m.update(seed=42.0), "seed must be an integer"),
        (lambda m: m["requested"].update(max_bin=63), "'max_bin' does not match the fixed setting"),
        (lambda m: m["requested"].update(enable_bundle=0), "'enable_bundle' does not match"),
        (lambda m: m["requested"].update(max_bin=255.0), "'max_bin' does not match"),
        (lambda m: m.update(model_sha256="0" * 64), "model and manifest do not match"),
        (lambda m: m.pop("categories"), "incomplete"),
    ],
)
def test_reload_rejects_a_tampered_manifest(
    settings: ModelSettings,
    train: ModelData,
    tmp_path: Path,
    edit: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    _saved(settings, train, tmp_path / "model")
    _edit_manifest(tmp_path / "model", edit)
    with pytest.raises(ModelError, match=message):
        _reload(settings, tmp_path / "model")


def test_reload_rejects_another_request(
    settings: ModelSettings, train: ModelData, tmp_path: Path
) -> None:
    _saved(settings, train, tmp_path / "model")
    with pytest.raises(ModelError, match="does not match the requested"):
        _reload(settings, tmp_path / "model", seed=7)
    with pytest.raises(SpecificationError, match="CPU only"):
        _reload(settings, tmp_path / "model", device="gpu")


def test_lightgbm_trains_on_the_cpu_only(settings: ModelSettings) -> None:
    # DR-013 §2: LightGBM's CUDA learner fails on categorical features above 256 categories.
    with pytest.raises(SpecificationError, match="CPU only"):
        LightGBMModel(
            configuration("LGB-u"), 42, settings=settings, features=FEATURES, device="gpu"
        )


def test_native_load_failure_is_a_model_error(
    settings: ModelSettings, train: ModelData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lightgbm.basic import LightGBMError

    _saved(settings, train, tmp_path / "model")
    monkeypatch.setattr("lightgbm.Booster", _raise_lightgbm_error)
    with pytest.raises(ModelError, match="cannot be loaded: injected") as raised:
        _reload(settings, tmp_path / "model")
    assert isinstance(raised.value.__cause__, LightGBMError)


def test_reload_rejects_a_changed_model_file(
    settings: ModelSettings, train: ModelData, tmp_path: Path
) -> None:
    _saved(settings, train, tmp_path / "model")
    with (tmp_path / "model" / "model.txt").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ModelError, match="model and manifest do not match"):
        _reload(settings, tmp_path / "model")


def test_only_lightgbm_configurations(settings: ModelSettings) -> None:
    with pytest.raises(ValueError, match="not a LightGBM configuration"):
        _model(settings, "Cat-u")


def test_unfitted_model_is_rejected(settings: ModelSettings, valid: ModelData) -> None:
    with pytest.raises(ModelError, match="not fitted"):
        _model(settings).predict(valid.features)


def test_resolved_parameters_are_read_from_the_model_text() -> None:
    text = "tree\nparameters:\n[seed: 42]\n[metric: a: b]\n\nend of parameters\n"
    assert resolved_parameters(text) == {"seed": "42", "metric": "a: b"}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("tree\n", "no complete parameter block"),
        ("tree\nparameters:\n[seed: 42]\n", "no complete parameter block"),
        ("tree\nparameters:\n\nend of parameters\n", "empty"),
        ("tree\nparameters:\n[seed: 1]\n[seed: 2]\nend of parameters\n", "repeated"),
        ("tree\nparameters:\nseed=1\nend of parameters\n", "Malformed"),
    ],
)
def test_resolved_parameters_reject_incomplete_blocks(text: str, message: str) -> None:
    with pytest.raises(ModelError, match=message):
        resolved_parameters(text)


# --- the shared V1 objective as LightGBM's evaluation (Issue #123) ---------------------------

V1_HYPERPARAMETERS: dict[str, Any] = {**HYPERPARAMETERS, "metric": "None"}
DAILY_K = 6


def _by_day(data: ModelData, per_day: int = 60, weight: float | None = None) -> ModelData:
    """The same rows with identifiers and times that spread them over relative days 1, 2, ..."""
    rows = len(data.labels)
    index = np.arange(rows, dtype=np.int64)
    weights = data.weights if weight is None else np.where(data.labels == 1, weight, 1.0)
    return ModelData(
        data.features,
        data.labels,
        weights,
        times=(index // per_day + 1) * 86_400 + index % per_day,
        ids=index + 3_000_000,
    )


def _v1_fit(
    settings: ModelSettings,
    train: ModelData,
    valid: ModelData,
    hyperparameters: dict[str, Any] | None = None,
    name: str = "LGB-u",
) -> tuple[LightGBMModel, V1Objective]:
    objective = V1Objective(valid, DAILY_K)
    model = _model(settings, name)
    model.fit(train, valid, hyperparameters or V1_HYPERPARAMETERS, objective=objective)
    return model, objective


def test_fit_early_stops_on_the_v1_objective(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    v1 = _by_day(valid)
    model, objective = _v1_fit(settings, train, v1)
    history = model.training_history
    assert history is not None
    assert model.fitted_iterations == history.best_checkpoint
    assert history.stopped_at >= history.best_checkpoint
    assert history.objective == objective.identity == model.parameters["v1_objective"]
    # The history's best equals the objective recomputed from the model's own scores.
    recall = objective.queue.recall(model.predict(v1.features), v1.labels, DAILY_K)
    assert recall.numerator == history.checkpoints[history.best_checkpoint - 1].numerator
    assert recall.value == history.best_objective
    assert model.parameters["requested"]["metric"] == "None"


def test_improvement_is_strict_and_the_earliest_best_is_kept(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    model, _ = _v1_fit(settings, train, _by_day(valid))
    history = model.training_history
    assert history is not None
    best = -1
    for row in history.checkpoints:
        assert row.improved == (row.numerator > best)
        best = max(best, row.numerator)
        assert row.best_checkpoint == next(
            c.checkpoint for c in history.checkpoints if c.numerator == best
        )
    assert history.best_checkpoint == history.checkpoints[-1].best_checkpoint


def test_patience_stops_training_the_declared_rounds_after_the_best(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    # An objective of at most 30 integer steps cannot improve strictly in 60 rounds in a row.
    model, _ = _v1_fit(
        settings, train, _by_day(valid), {**V1_HYPERPARAMETERS, "early_stopping_round": 1}
    )
    history = model.training_history
    assert history is not None
    assert history.stopping_reason == "early_stopping"
    assert history.stopped_at == history.best_checkpoint + 1
    assert history.stopped_at < 60


def test_training_ends_at_the_maximum_when_patience_does_not_run_out(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    model, _ = _v1_fit(
        settings,
        train,
        _by_day(valid),
        {**V1_HYPERPARAMETERS, "num_iterations": 8, "early_stopping_round": 8},
    )
    history = model.training_history
    assert history is not None
    assert (history.stopped_at, history.stopping_reason) == (8, "maximum")


def test_v1_weights_do_not_enter_the_objective(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    plain, _ = _v1_fit(settings, train, _by_day(valid))
    heavy, _ = _v1_fit(settings, train, _by_day(valid, weight=50.0))
    assert plain.training_history == _same_but_seconds(plain, heavy)
    assert plain.fitted_iterations == heavy.fitted_iterations


def _same_but_seconds(reference: LightGBMModel, other: LightGBMModel) -> Any:
    """``other``'s history with the timings of ``reference`` (runtimes differ between fits)."""
    history = other.training_history
    base = reference.training_history
    assert history is not None
    assert base is not None
    rows = tuple(
        replace(row, seconds=ref.seconds)
        for row, ref in zip(history.checkpoints, base.checkpoints, strict=True)
    )
    return replace(history, checkpoints=rows)


def test_a_fit_without_the_objective_has_no_history(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    model = _model(settings)
    model.fit(train, valid, HYPERPARAMETERS)
    assert model.training_history is None
    assert model.parameters["v1_objective"] is None


def test_the_objective_requires_metric_none(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    v1 = _by_day(valid)
    objective = V1Objective(v1, DAILY_K)
    for metric in ("binary_logloss", "none", ["None"], "custom"):
        with pytest.raises(SpecificationError, match='metric must be "None"'):
            _model(settings).fit(
                train, v1, {**V1_HYPERPARAMETERS, "metric": metric}, objective=objective
            )


def test_metric_none_needs_the_objective(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    with pytest.raises(SpecificationError, match="explicit evaluation metric"):
        _model(settings).fit(train, valid, V1_HYPERPARAMETERS)


def test_the_objective_needs_early_stopping_on_its_own_partition(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    v1 = _by_day(valid)
    objective = V1Objective(v1, DAILY_K)
    with pytest.raises(SpecificationError, match="needs early stopping"):
        _model(settings).fit(train, None, V1_HYPERPARAMETERS, iterations=3, objective=objective)
    with pytest.raises(DataValidationError, match="different V1 rows"):
        _model(settings).fit(train, _by_day(_data(300, 9)), V1_HYPERPARAMETERS, objective=objective)


def test_refit_with_the_declared_metric_has_no_objective(
    settings: ModelSettings, train: ModelData
) -> None:
    model = _model(settings)
    model.fit(train, None, V1_HYPERPARAMETERS, iterations=6)
    assert model.fitted_iterations == 6
    assert model.training_history is None


def test_the_v1_fit_reloads_with_its_objective(
    settings: ModelSettings, train: ModelData, valid: ModelData, tmp_path: Path
) -> None:
    v1 = _by_day(valid)
    model, objective = _v1_fit(settings, train, v1, name="LGB-b")
    model.save(tmp_path / "model")
    reloaded = _reload(settings, tmp_path / "model")
    assert reloaded.parameters["v1_objective"] == objective.identity
    assert reloaded.parameters["requested"]["metric"] == "None"
    assert reloaded.training_history is None
    assert np.array_equal(reloaded.predict(v1.features), model.predict(v1.features))
    manifest = json.loads((tmp_path / "model" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["v1_objective"] == objective.identity


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda m: m["v1_objective"].update(daily_k=0), "positive integer"),
        (lambda m: m["v1_objective"].update(weighting="balanced"), "differs from DR-015"),
        (lambda m: m["v1_objective"].pop("rows"), "malformed"),
        (lambda m: m["requested"].update(metric=["auc"]), 'recorded metric "None"'),
        (lambda m: m.pop("v1_objective"), "incomplete"),
    ],
)
def test_reload_rejects_a_tampered_objective_record(
    settings: ModelSettings,
    train: ModelData,
    valid: ModelData,
    tmp_path: Path,
    edit: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    model, _ = _v1_fit(settings, train, _by_day(valid), name="LGB-b")
    model.save(tmp_path / "model")
    _edit_manifest(tmp_path / "model", edit)
    with pytest.raises(ModelError, match=message):
        _reload(settings, tmp_path / "model")
