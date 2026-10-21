"""The CatBoost model on the common interface, fitted on small synthetic data on the CPU (no
project data; the GPU path is exercised by the GPU probe, Issue #77)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from review_capacity.config import ModelSettings, load_model_settings
from review_capacity.errors import DataValidationError, ModelError, SpecificationError
from review_capacity.models import catboost_model
from review_capacity.models.base import ModelData
from review_capacity.models.catboost_model import (
    SYNONYMS,
    CatBoostModel,
    catboost_parameters,
)
from review_capacity.models.registry import configuration
from review_capacity.models.v1_objective import V1Objective
from support.synthetic import FEATURES, REPO_PATHS

HYPERPARAMETERS: dict[str, Any] = {
    "iterations": 60,
    "od_wait": 5,
    "eval_metric": "Logloss",
    "metric_period": 10,
    "learning_rate": 0.1,
    "depth": 4,
    "model_size_reg": 0.5,
    "fold_permutation_block": 1,
    "permutation_count": 4,
    "random_strength": 1.0,
    "l2_leaf_reg": 3.0,
    "leaf_estimation_iterations": 10,
    "bagging_temperature": 1.0,
}


@pytest.fixture(autouse=True)
def no_files_outside_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run in a temporary directory, so that any file CatBoost writes would be visible."""
    monkeypatch.chdir(tmp_path)


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
    """Rows in time order: increasing TransactionDT and TransactionID."""
    frame, labels = _frame(rows, seed)
    times = (np.arange(rows, dtype=np.int64) * 60) + 86_400
    ids = np.arange(rows, dtype=np.int64) + 3_000_000
    weights = np.where(labels == 1, fraud_weight, 1.0)
    return ModelData(frame, labels, weights, times=times, ids=ids)


def _model(settings: ModelSettings, name: str = "Cat-u", seed: int = 42) -> CatBoostModel:
    return CatBoostModel(
        configuration(name), seed, settings=settings, features=FEATURES, device="cpu"
    )


@pytest.fixture(scope="module")
def train() -> ModelData:
    return _data(600, 1)


@pytest.fixture(scope="module")
def valid() -> ModelData:
    return _data(300, 2)


def test_synonyms_are_catboost_own() -> None:
    from catboost.core import _process_synonyms

    for alias, canonical in SYNONYMS.items():
        params: dict[str, Any] = {alias: 1}
        _process_synonyms(params)
        assert set(params) == {canonical}, alias
    parameters = catboost_parameters()
    assert set(SYNONYMS.values()) <= parameters
    assert {"permutation_count", "has_time", "od_wait"} <= parameters
    assert not set(SYNONYMS) & parameters
    assert "depthh" not in parameters


def test_fit_with_early_stopping_and_cpu_prediction(
    settings: ModelSettings, train: ModelData, valid: ModelData, tmp_path: Path
) -> None:
    model = _model(settings)
    model.fit(train, valid, HYPERPARAMETERS)
    scores = model.predict(valid.features)
    assert scores.shape == (300,)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert 0 < model.fitted_iterations <= 60
    parameters = model.parameters
    resolved = parameters["resolved"]
    assert (resolved["has_time"], resolved["counter_calc_method"]) == (True, "SkipTest")
    assert (resolved["task_type"], resolved["random_seed"]) == ("CPU", 42)
    assert resolved["simple_ctr"]  # the full CTR descriptors are recorded
    requested = parameters["requested"]
    assert (requested["use_best_model"], requested["od_type"]) == (True, "Iter")
    assert requested["allow_writing_files"] is False
    assert parameters["prediction"] == {"backend": "cpu", "task_type": "CPU", "thread_count": 24}
    metadata = parameters["metadata"]
    assert {"params", "output_options", catboost_model.RECORD_KEY} <= set(metadata)
    assert json.loads(metadata[catboost_model.RECORD_KEY]) == json.loads(json.dumps(requested))
    assert list(tmp_path.iterdir()) == []  # no catboost_info directory


def test_refit_at_a_fixed_iteration_count(settings: ModelSettings, train: ModelData) -> None:
    model = _model(settings)
    model.fit(train, None, HYPERPARAMETERS, iterations=12)
    assert model.fitted_iterations == 12
    requested = model.parameters["requested"]
    assert requested["iterations"] == 12
    assert requested["use_best_model"] is False
    assert "od_wait" not in requested
    assert "od_type" not in requested


def test_same_seed_gives_identical_cpu_scores(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    first, second = _model(settings), _model(settings)
    first.fit(train, valid, HYPERPARAMETERS)
    second.fit(train, valid, HYPERPARAMETERS)
    assert np.array_equal(first.predict(valid.features), second.predict(valid.features))


def test_train_weights_change_the_fit(settings: ModelSettings, valid: ModelData) -> None:
    plain, weighted = _model(settings), _model(settings, "Cat-b")
    plain.fit(_data(600, 1), None, HYPERPARAMETERS, iterations=20)
    weighted.fit(_data(600, 1, fraud_weight=5.0), None, HYPERPARAMETERS, iterations=20)
    assert not np.array_equal(plain.predict(valid.features), weighted.predict(valid.features))


def test_train_rows_must_be_in_time_order(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    assert train.times is not None
    reversed_times = ModelData(
        train.features, train.labels, train.weights, times=train.times[::-1].copy(), ids=train.ids
    )
    with pytest.raises(DataValidationError, match="time order"):
        _model(settings).fit(reversed_times, valid, HYPERPARAMETERS)
    unordered = ModelData(train.features, train.labels, train.weights)
    with pytest.raises(DataValidationError, match="needed to check row order"):
        _model(settings).fit(unordered, valid, HYPERPARAMETERS)


def test_order_is_checked_on_train_only_and_before_fitting(
    settings: ModelSettings, train: ModelData, valid: ModelData, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert train.times is not None
    assert train.ids is not None
    assert valid.times is not None
    reversed_valid = ModelData(
        valid.features, valid.labels, valid.weights, times=valid.times[::-1].copy(), ids=valid.ids
    )
    _model(settings).fit(train, reversed_valid, HYPERPARAMETERS)  # V1 order is not required
    tied = train.times.copy()
    tied[1] = tied[0]  # rows 0 and 1 share a time ...
    ids = train.ids.copy()
    ids[[0, 1]] = ids[[1, 0]]  # ... and their identifiers decrease
    bad_ties = ModelData(train.features, train.labels, train.weights, times=tied, ids=ids)

    def never_called(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("CatBoost must not be called")

    monkeypatch.setattr("catboost.CatBoostClassifier.fit", never_called)
    with pytest.raises(DataValidationError, match="time order"):
        _model(settings).fit(bad_ties, valid, HYPERPARAMETERS)


def _without(name: str) -> dict[str, Any]:
    return {k: v for k, v in HYPERPARAMETERS.items() if k != name}


@pytest.mark.parametrize(
    ("hyperparameters", "message"),
    [
        (_without("learning_rate"), r"\['learning_rate'\]"),
        (_without("permutation_count"), r"\['permutation_count'\]"),
        (_without("od_wait"), r"\['od_wait'\]"),
        (_without("eval_metric"), r"\['eval_metric'\]"),
        ({**HYPERPARAMETERS, "learning_rate": True}, "learning_rate must be a positive number"),
        ({**HYPERPARAMETERS, "learning_rate": 0}, "learning_rate must be a positive number"),
        ({**HYPERPARAMETERS, "learning_rate": float("inf")}, "learning_rate must be a positive"),
        ({**HYPERPARAMETERS, "iterations": 0}, "iterations must be a positive integer"),
        ({**HYPERPARAMETERS, "metric_period": "10"}, "metric_period must be a positive integer"),
        ({**HYPERPARAMETERS, "od_wait": 2.5}, "od_wait must be a positive integer"),
        ({**HYPERPARAMETERS, "eval_metric": " "}, "eval_metric must name"),
        ({**HYPERPARAMETERS, "depth": None}, "None would leave"),
        ({**HYPERPARAMETERS, "depthh": 4}, "Unknown CatBoost parameter"),
        ({**HYPERPARAMETERS, "eta": 0.2}, "more than one name"),
    ],
)
def test_explicit_settings_must_be_valid(
    settings: ModelSettings,
    train: ModelData,
    valid: ModelData,
    hyperparameters: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(SpecificationError, match=message):
        _model(settings).fit(train, valid, hyperparameters)


def test_refit_does_not_need_early_stopping_rounds(
    settings: ModelSettings, train: ModelData
) -> None:
    model = _model(settings)
    model.fit(train, None, _without("od_wait"), iterations=5)
    assert model.fitted_iterations == 5


@pytest.mark.parametrize(
    "name",
    [
        "random_seed",
        "random_state",
        "objective",
        "loss_function",
        "border_count",
        "max_bin",
        "has_time",
        "task_type",
        "thread_count",
        "devices",
        "simple_ctr",
        "cat_features",
        "class_weights",
        "auto_class_weights",
        "scale_pos_weight",
        "use_best_model",
        "od_type",
        "train_dir",
        "allow_writing_files",
        "logging_level",
        "verbose",
        "eval_fraction",
        "per_float_feature_quantization",
        "per_feature_ctr",
        "ctr_description",
        "final_ctr_computation_mode",
        "metadata",
    ],
)
def test_reserved_settings_cannot_be_tuned(
    settings: ModelSettings, train: ModelData, valid: ModelData, name: str
) -> None:
    with pytest.raises(SpecificationError, match="cannot override fixed research settings"):
        _model(settings).fit(train, valid, {**HYPERPARAMETERS, name: 1})


def test_unseen_category_is_scored(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    model = _model(settings)
    model.fit(train, valid, HYPERPARAMETERS)
    unseen = valid.features.copy()
    unseen["card1"] = "9999"  # never seen in Train
    assert np.isfinite(model.predict(unseen)).all()


def test_inputs_must_be_the_canonical_inputs(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    reordered = train.features[list(reversed(train.features.columns))]
    data = ModelData(reordered, train.labels, train.weights, times=train.times, ids=train.ids)
    with pytest.raises(DataValidationError, match="canonical inputs"):
        _model(settings).fit(data, valid, HYPERPARAMETERS)


def test_library_output_reaches_the_run_log(
    settings: ModelSettings, train: ModelData, valid: ModelData, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="review_capacity"):
        _model(settings).fit(train, valid, HYPERPARAMETERS)
    messages = [r.getMessage() for r in caplog.records if r.name == catboost_model.__name__]
    assert any("learn" in m for m in messages)  # CatBoost's periodic progress line
    assert any(m.startswith("Fit Cat-u on cpu") for m in messages)


def test_log_stream_logs_complete_lines(caplog: pytest.LogCaptureFixture) -> None:
    stream = catboost_model._LogStream(logging.WARNING)
    with caplog.at_level(logging.INFO, logger="review_capacity"):
        stream.write("first line\nsecond ")
        stream.write("line\n\npartial")
        stream.flush()
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.WARNING, "first line"),
        (logging.WARNING, "second line"),
        (logging.WARNING, "partial"),
    ]


def test_resolved_parameters_must_show_every_fixed_setting() -> None:
    fixed = {
        "loss_function": "Logloss",
        "has_time": True,
        "border_count": 254,
        "simple_ctr": ["Borders", "Counter"],
        "thread_count": 24,  # not reported by CatBoost: checked through the record
        "random_seed": 42,
    }
    resolved = {
        "loss_function": "Logloss",
        "has_time": True,
        "border_count": 254,
        "simple_ctr": ["Borders:CtrBorderCount=15:Prior=0/1", "Counter:CtrBorderCount=15"],
        "random_seed": 42,
    }
    catboost_model._check_resolved(resolved, fixed)
    for name, value in [
        ("has_time", False),
        ("has_time", 1),
        ("border_count", 128),
        ("random_seed", 7),
        ("simple_ctr", ["Borders:CtrBorderCount=15", "FeatureFreq:CtrBorderCount=15"]),
        ("simple_ctr", "Borders"),
    ]:
        with pytest.raises(ModelError, match=name):
            catboost_model._check_resolved({**resolved, name: value}, fixed)
    without = {k: v for k, v in resolved.items() if k != "loss_function"}
    with pytest.raises(ModelError, match="loss_function"):
        catboost_model._check_resolved(without, fixed)


@pytest.mark.parametrize(
    ("classes", "binary"),
    [
        ([0, 1], True),
        (np.array([0, 1], dtype=np.int64), True),
        (["0", "1"], True),
        ([1, 0], False),
        ([0.2, 1.2], False),
        ([0.0, 1.0], False),
        ([float("inf"), 1], False),
        ([False, True], False),
        ([0, 1, 2], False),
    ],
)
def test_class_labels_must_be_exactly_zero_and_one(classes: Any, binary: bool) -> None:
    assert catboost_model._binary_classes(classes) is binary


def test_prediction_uses_cpu_kept_trees_threads_and_the_fraud_column(
    settings: ModelSettings, train: ModelData, valid: ModelData, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(settings)
    model.fit(train, None, HYPERPARAMETERS, iterations=7)
    calls: list[dict[str, Any]] = []

    def spy(_self: Any, pool: Any, **kwargs: Any) -> npt.NDArray[np.float64]:
        calls.append(kwargs)
        rows = pool.num_row()
        return np.column_stack([np.full(rows, 0.9), np.full(rows, 0.1)])

    monkeypatch.setattr("catboost.CatBoostClassifier.predict_proba", spy)
    scores = model.predict(valid.features)
    assert calls == [{"ntree_end": 7, "thread_count": 24, "task_type": "CPU"}]
    assert np.allclose(scores, 0.1)  # column 1: the fraud class


def test_a_model_not_cut_back_to_its_best_iteration_is_rejected(
    settings: ModelSettings, train: ModelData, valid: ModelData, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("catboost.CatBoostClassifier.get_best_iteration", lambda _self: 10_000)
    with pytest.raises(ModelError, match="best iteration"):
        _model(settings).fit(train, valid, HYPERPARAMETERS)


def _saved(settings: ModelSettings, train: ModelData, directory: Path) -> CatBoostModel:
    model = _model(settings, "Cat-b")
    model.fit(train, None, HYPERPARAMETERS, iterations=5)
    model.save(directory)
    return model


def _reload(
    settings: ModelSettings, directory: Path, seed: int = 42, device: Any = "cpu"
) -> CatBoostModel:
    return CatBoostModel.load(
        directory,
        configuration("Cat-b"),
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
    assert reloaded.parameters["requested"] == json.loads(json.dumps(model.parameters["requested"]))
    assert reloaded.parameters["resolved"]["has_time"] is True


def _edit_manifest(directory: Path, edit: Callable[[dict[str, Any]], object]) -> None:
    path = directory / "manifest.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    edit(content)
    path.write_text(json.dumps(content), encoding="utf-8")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda m: m.update(format_version=99), "format version"),
        (lambda m: m.update(library_version="1.2.9"), "does not match the requested"),
        (lambda m: m.update(device="gpu"), "does not match the requested"),
        (lambda m: m.update(iterations=3), "tree count does not match"),
        (lambda m: m.update(iterations=0), "positive integer"),
        (lambda m: m.update(seed=42.0), "seed must be an integer"),
        (lambda m: m["requested"].update(thread_count=8), "'thread_count' does not match"),
        (lambda m: m["requested"].update(has_time=1), "'has_time' does not match"),
        (lambda m: m["requested"].pop("learning_rate"), "model's own record"),
        (lambda m: m["requested"].update(eval_metric="AUC"), "model's own record"),
        (lambda m: m["requested"].update(use_best_model=True), "model's own record"),
        (lambda m: m["requested"].update(use_best_model=0), "model's own record"),
        (lambda m: m["requested"].update(allow_writing_files=0), "model's own record"),
        (lambda m: m["requested"].update(iterations=5.0), "model's own record"),
        (lambda m: m.update(model_sha256="0" * 64), "model and manifest do not match"),
        (lambda m: m.pop("requested"), "incomplete"),
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
    with pytest.raises(ModelError, match="does not match the requested"):
        _reload(settings, tmp_path / "model", device="gpu")


def test_reload_rejects_a_changed_model_file(
    settings: ModelSettings, train: ModelData, tmp_path: Path
) -> None:
    _saved(settings, train, tmp_path / "model")
    with (tmp_path / "model" / "model.cbm").open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(ModelError, match="model and manifest do not match"):
        _reload(settings, tmp_path / "model")


def _raise_catboost_error(*_args: Any, **_kwargs: Any) -> Any:
    from catboost import CatBoostError

    raise CatBoostError("injected")


@pytest.mark.parametrize(
    ("method", "use", "message"),
    [
        (
            "fit",
            lambda m, data, _path: m.fit(data, None, HYPERPARAMETERS, iterations=3),
            "training failed: injected",
        ),
        ("predict_proba", lambda m, data, _path: m.predict(data.features), "prediction failed"),
        ("save_model", lambda m, _data, path: m.save(path), "cannot be saved"),
    ],
)
def test_native_failure_is_a_model_error(
    settings: ModelSettings,
    train: ModelData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    use: Callable[[CatBoostModel, ModelData, Path], object],
    message: str,
) -> None:
    from catboost import CatBoostError

    model = _model(settings)
    model.fit(train, None, HYPERPARAMETERS, iterations=3)
    (tmp_path / "model").mkdir()
    monkeypatch.setattr(f"catboost.CatBoostClassifier.{method}", _raise_catboost_error)
    with pytest.raises(ModelError, match=message) as raised:
        use(model, train, tmp_path / "model")
    assert isinstance(raised.value.__cause__, CatBoostError)


def test_native_load_failure_is_a_model_error(
    settings: ModelSettings, train: ModelData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from catboost import CatBoostError

    _saved(settings, train, tmp_path / "model")
    monkeypatch.setattr("catboost.CatBoostClassifier.load_model", _raise_catboost_error)
    with pytest.raises(ModelError, match="cannot be loaded: injected") as raised:
        _reload(settings, tmp_path / "model")
    assert isinstance(raised.value.__cause__, CatBoostError)


def test_only_catboost_configurations(settings: ModelSettings) -> None:
    with pytest.raises(ValueError, match="not a CatBoost configuration"):
        _model(settings, "LGB-u")


def test_unfitted_model_is_rejected(settings: ModelSettings, valid: ModelData) -> None:
    with pytest.raises(ModelError, match="not fitted"):
        _model(settings).predict(valid.features)


def test_record_must_match_and_be_complete() -> None:
    record = dict.fromkeys((*catboost_model.EXPLICIT, *catboost_model.CONTROLS), 1)
    catboost_model._check_record(record, dict(record))
    with pytest.raises(ModelError, match="model's own record"):
        catboost_model._check_record(record, {**record, "depth": 6})
    with pytest.raises(ModelError, match="model's own record"):
        catboost_model._check_record(record, "not a record")
    partial = {k: v for k, v in record.items() if k != "learning_rate"}
    with pytest.raises(ModelError, match=r"lack required settings: \['learning_rate'\]"):
        catboost_model._check_record(partial, dict(partial))


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("feature_names_", ["x"], "feature names"),
        ("classes_", [0.0, 1.0], "class labels"),
    ],
)
def test_reload_checks_the_loaded_model(
    settings: ModelSettings,
    train: ModelData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    value: Any,
    message: str,
) -> None:
    from catboost import CatBoostClassifier

    _saved(settings, train, tmp_path / "model")
    monkeypatch.setattr(CatBoostClassifier, attribute, property(lambda _self: value))
    with pytest.raises(ModelError, match=message):
        _reload(settings, tmp_path / "model")


def test_reload_checks_the_loaded_categorical_inputs(
    settings: ModelSettings, train: ModelData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _saved(settings, train, tmp_path / "model")
    monkeypatch.setattr("catboost.CatBoostClassifier.get_cat_feature_indices", lambda _self: [0])
    with pytest.raises(ModelError, match="categorical inputs"):
        _reload(settings, tmp_path / "model")


def test_system_options_must_show_the_fixed_threads_and_devices() -> None:
    fixed = {"thread_count": 24, "border_count": 254}
    metadata = {"params": json.dumps({"system_options": {"thread_count": 24}})}
    catboost_model._check_system(metadata, fixed)
    for system in ({"thread_count": 8}, {"thread_count": 24.0}, {}):
        with pytest.raises(ModelError, match="thread_count"):
            catboost_model._check_system({"params": json.dumps({"system_options": system})}, fixed)
    gpu = {"devices": "0"}
    as_text = {"params": json.dumps({"system_options": {"devices": "0"}})}
    as_number = {"params": json.dumps({"system_options": {"devices": 0}})}
    catboost_model._check_system(as_text, gpu)
    with pytest.raises(ModelError, match="devices"):
        catboost_model._check_system(as_number, gpu)
    for broken in ({}, {"params": "not json"}, {"params": json.dumps({"other": 1})}):
        with pytest.raises(ModelError, match="no readable system options"):
            catboost_model._check_system(broken, fixed)


def test_record_comparison_keeps_json_types_apart() -> None:
    record = dict.fromkeys((*catboost_model.EXPLICIT, *catboost_model.CONTROLS), 1)
    record["use_best_model"] = False
    reordered = dict(reversed(list(record.items())))
    catboost_model._check_record(record, reordered)  # key order does not matter
    for name, value in [("use_best_model", 0), ("iterations", 1.0), ("iterations", True)]:
        with pytest.raises(ModelError, match="model's own record"):
            catboost_model._check_record({**record, name: value}, record)
    with pytest.raises(ModelError, match="not valid JSON values"):
        catboost_model._check_record({**record, "learning_rate": float("nan")}, record)


# --- the shared V1 objective as CatBoost's user-defined metric (Issue #123) -------------------

V1_HYPERPARAMETERS: dict[str, Any] = {
    **HYPERPARAMETERS,
    "eval_metric": "v1_objective",
    "boost_from_average": False,
    "metric_period": 1,
}
DAILY_K = 6


def _by_day(data: ModelData, per_day: int = 60) -> ModelData:
    """The same rows (in time order) with times that spread them over relative days 1, 2, ..."""
    rows = len(data.labels)
    index = np.arange(rows, dtype=np.int64)
    return ModelData(
        data.features,
        data.labels,
        data.weights,
        times=(index // per_day + 1) * 86_400 + index % per_day,
        ids=index + 3_000_000,
    )


def _v1_fit(
    settings: ModelSettings,
    train: ModelData,
    valid: ModelData,
    hyperparameters: dict[str, Any] | None = None,
    name: str = "Cat-u",
) -> tuple[CatBoostModel, V1Objective]:
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
    requested = model.parameters["requested"]
    assert requested["eval_metric"] == "v1_objective"
    assert requested["boost_from_average"] is False
    assert requested["v1_objective"] == history.objective == objective.identity
    assert model.parameters["resolved"]["boost_from_average"] is False
    recall = objective.queue.recall(model.predict(v1.features), v1.labels, DAILY_K)
    assert recall.numerator == history.checkpoints[history.best_checkpoint - 1].numerator


def test_patience_stops_training_the_declared_rounds_after_the_best(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    # An objective of at most 30 integer steps cannot improve strictly in 60 rounds in a row.
    model, _ = _v1_fit(settings, train, _by_day(valid), {**V1_HYPERPARAMETERS, "od_wait": 1})
    history = model.training_history
    assert history is not None
    assert history.stopping_reason == "early_stopping"
    assert history.stopped_at == history.best_checkpoint + 1


def test_training_ends_at_the_maximum_when_patience_does_not_run_out(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    model, _ = _v1_fit(
        settings, train, _by_day(valid), {**V1_HYPERPARAMETERS, "iterations": 8, "od_wait": 8}
    )
    history = model.training_history
    assert history is not None
    assert (history.stopped_at, history.stopping_reason) == (8, "maximum")


def test_v1_weights_do_not_enter_the_objective(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    v1 = _by_day(valid)
    heavy = ModelData(
        v1.features,
        v1.labels,
        np.where(v1.labels == 1, 50.0, 1.0),
        times=v1.times,
        ids=v1.ids,
    )
    plain_model, _ = _v1_fit(settings, train, v1)
    heavy_model, _ = _v1_fit(settings, train, heavy)
    plain, other = plain_model.training_history, heavy_model.training_history
    assert plain is not None
    assert other is not None
    assert [c.numerator for c in plain.checkpoints] == [c.numerator for c in other.checkpoints]
    assert plain_model.fitted_iterations == heavy_model.fitted_iterations


def test_a_fit_without_the_objective_has_no_history(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    model = _model(settings)
    model.fit(train, valid, HYPERPARAMETERS)
    assert model.training_history is None
    assert "v1_objective" not in model.parameters["requested"]


def test_refit_records_the_declared_metric_without_passing_it(
    settings: ModelSettings, train: ModelData
) -> None:
    model = _model(settings)
    model.fit(train, None, V1_HYPERPARAMETERS, iterations=6)
    assert model.fitted_iterations == 6
    assert model.training_history is None
    requested = model.parameters["requested"]
    assert (requested["eval_metric"], requested["boost_from_average"]) == ("v1_objective", False)
    assert "v1_objective" not in requested
    assert model.parameters["resolved"]["boost_from_average"] is False


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"boost_from_average": True}, "boost_from_average must be set explicitly to False"),
        ({"boost_from_average": 0}, "boost_from_average must be set explicitly to False"),
        ({"eval_metric": "Logloss"}, "eval_metric must be 'v1_objective'"),
    ],
)
def test_the_research_configuration_is_checked(
    settings: ModelSettings,
    train: ModelData,
    valid: ModelData,
    change: dict[str, Any],
    message: str,
) -> None:
    v1 = _by_day(valid)
    objective = V1Objective(v1, DAILY_K)
    with pytest.raises(SpecificationError, match=message):
        _model(settings).fit(train, v1, {**V1_HYPERPARAMETERS, **change}, objective=objective)


def test_boost_from_average_must_be_explicit_with_the_declared_metric(
    settings: ModelSettings, train: ModelData
) -> None:
    missing = {k: v for k, v in V1_HYPERPARAMETERS.items() if k != "boost_from_average"}
    with pytest.raises(SpecificationError, match="boost_from_average must be set explicitly"):
        _model(settings).fit(train, None, missing, iterations=3)


def test_the_declared_metric_needs_the_objective_for_early_stopping(
    settings: ModelSettings, train: ModelData, valid: ModelData
) -> None:
    with pytest.raises(SpecificationError, match="needs the objective argument"):
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


def test_resolved_boost_from_average_must_be_false() -> None:
    catboost_model._check_boost_from_average({"boost_from_average": False})
    for resolved in ({}, {"boost_from_average": True}, {"boost_from_average": 0}):
        with pytest.raises(ModelError, match="boost_from_average must be False"):
            catboost_model._check_boost_from_average(resolved)


def test_the_v1_fit_reloads_with_its_objective(
    settings: ModelSettings, train: ModelData, valid: ModelData, tmp_path: Path
) -> None:
    v1 = _by_day(valid)
    model, objective = _v1_fit(settings, train, v1, name="Cat-b")
    model.save(tmp_path / "model")
    reloaded = _reload(settings, tmp_path / "model")
    requested = reloaded.parameters["requested"]
    assert requested["v1_objective"] == objective.identity
    assert (requested["eval_metric"], requested["boost_from_average"]) == ("v1_objective", False)
    assert reloaded.parameters["resolved"]["boost_from_average"] is False
    assert reloaded.training_history is None
    assert np.array_equal(reloaded.predict(v1.features), model.predict(v1.features))
    metadata = reloaded.parameters["metadata"]
    embedded = json.loads(metadata[catboost_model.RECORD_KEY])
    assert embedded["v1_objective"] == objective.identity


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda m: m["requested"]["v1_objective"].update(daily_k=0), "positive integer"),
        (lambda m: m["requested"]["v1_objective"].update(version=2), "not this version"),
        (lambda m: m["requested"].update(boost_from_average=True), "boost_from_average"),
        (lambda m: m["requested"].pop("v1_objective"), "model's own record"),
        (lambda m: m["requested"].update(eval_metric="Logloss"), "must record its eval_metric"),
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
    model, _ = _v1_fit(settings, train, _by_day(valid), name="Cat-b")
    model.save(tmp_path / "model")
    _edit_manifest(tmp_path / "model", edit)
    with pytest.raises(ModelError, match=message):
        _reload(settings, tmp_path / "model")
