"""The ResNet-style MLP (REF-087, REF-103) and the shared neural fitted-model behaviour, on small
synthetic tensors on the CPU (no project data; the GPU path is exercised by the GPU probe,
Issue #77). Global PyTorch and random-generator state changed by a fit is restored."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from review_capacity.config import ModelSettings, load_model_settings
from review_capacity.errors import (
    DataValidationError,
    EnvironmentMismatchError,
    ModelError,
    SpecificationError,
)
from review_capacity.models.base import class_weights, sample_weights
from review_capacity.models.neural_training import (
    MANIFEST_FILE,
    STATE_FILE,
    BatchPlan,
    InputLayout,
    NeuralData,
    predict_logits,
)
from review_capacity.models.registry import Weighting, configuration
from review_capacity.models.resnet_model import ResNet, ResNetBlock, ResNetModel
from review_capacity.models.run_record import (
    jsonable,
    load_settings_with_snapshot,
    model_run_record,
)
from support.neural import (
    HYPERPARAMETERS,
    LAYOUT,
    RESNET,
    TRAINING,
    neural_data,
    neural_inputs,
    preserved_torch_state,
    v1_objective,
)
from support.synthetic import REPO_PATHS


@pytest.fixture(autouse=True)
def restore_torch_state() -> Iterator[None]:
    with preserved_torch_state():
        yield


@pytest.fixture(scope="module")
def settings() -> ModelSettings:
    return load_model_settings(REPO_PATHS)


def _balanced(data: NeuralData, train: NeuralData) -> NeuralData:
    """``data`` with the protocol's balanced weights from Train labels."""
    weights = sample_weights(data.labels, class_weights(train.labels, Weighting.BALANCED))
    return replace(data, weights=weights)


@pytest.fixture(scope="module")
def train() -> NeuralData:
    data = neural_data(640, seed=1)
    return _balanced(data, data)


@pytest.fixture(scope="module")
def valid(train: NeuralData) -> NeuralData:
    return _balanced(neural_data(320, seed=2, with_rows=True), train)


def _model(settings: ModelSettings, seed: int = 42, name: str = "RES-b") -> ResNetModel:
    return ResNetModel(configuration(name), seed, settings=settings, layout=LAYOUT, device="cpu")


def _fit(
    settings: ModelSettings, train: NeuralData, valid: NeuralData | None, **kwargs: Any
) -> ResNetModel:
    model = _model(settings, kwargs.pop("seed", 42))
    if valid is not None:
        kwargs.setdefault("objective", v1_objective(valid))
    model.fit(train, valid, kwargs.pop("hyperparameters", HYPERPARAMETERS), **kwargs)
    return model


@pytest.fixture(scope="module")
def fitted(settings: ModelSettings, train: NeuralData, valid: NeuralData) -> ResNetModel:
    with preserved_torch_state():
        return _fit(settings, train, valid)


EARLY_STOPPED = {**HYPERPARAMETERS, "max_epochs": 40}  # long enough for patience to end training


@pytest.fixture(scope="module")
def early_stopped(settings: ModelSettings, train: NeuralData, valid: NeuralData) -> ResNetModel:
    with preserved_torch_state():
        return _fit(settings, train, valid, hyperparameters=EARLY_STOPPED)


@pytest.fixture(scope="module")
def fixed_epochs(settings: ModelSettings, train: NeuralData) -> ResNetModel:
    with preserved_torch_state():
        return _fit(settings, train, None, iterations=3)


def _reload(directory: Path, settings: ModelSettings) -> ResNetModel:
    return ResNetModel.load(
        directory, configuration("RES-b"), 42, settings=settings, layout=LAYOUT, device="cpu"
    )


def _rewrite_manifest(directory: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    path = directory / MANIFEST_FILE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    edit(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


# --- Architecture ---------------------------------------------------------------------------------


def _network(**changes: Any) -> ResNet:
    arguments: dict[str, Any] = {
        "n_numeric": 4,
        "cardinalities": (5, 3),
        "d_embedding": 3,
        "d": 8,
        "d_hidden_factor": 1.5,
        "n_layers": 3,
        "hidden_dropout": 0.2,
        "residual_dropout": 0.1,
    }
    return ResNet(**{**arguments, **changes})


def test_network_follows_the_reference_architecture() -> None:
    network = _network()
    assert network.first_layer.in_features == 4 + 2 * 3  # numeric inputs and two embeddings
    assert network.first_layer.out_features == 8
    assert len(network.blocks) == 3
    for block in network.blocks:
        assert isinstance(block, ResNetBlock)
        assert isinstance(block.norm, torch.nn.BatchNorm1d)
        assert (block.linear0.in_features, block.linear0.out_features) == (8, 12)
        assert (block.linear1.in_features, block.linear1.out_features) == (12, 8)
        assert (block.norm.eps, block.norm.momentum) == (1e-5, 0.1)
    assert isinstance(network.last_normalization, torch.nn.BatchNorm1d)
    assert network.head.out_features == 1
    embeddings = network.category_embeddings
    assert embeddings is not None
    assert tuple(embeddings.weight.shape) == (5 + 3, 3)  # one table, addressed by offsets
    assert network.category_offsets.tolist() == [0, 5]


def test_categories_are_embedded_through_their_offsets() -> None:
    network = _network().eval()
    embeddings = network.category_embeddings
    assert embeddings is not None
    categorical = torch.tensor([[1, 2]])
    rows = embeddings(categorical + network.category_offsets[None])
    assert torch.equal(rows[0, 0], embeddings.weight[1])
    assert torch.equal(rows[0, 1], embeddings.weight[5 + 2])


def test_forward_and_backward_passes() -> None:
    torch.manual_seed(0)
    network = _network()
    inputs = neural_inputs(16, seed=0)
    numeric, categorical = torch.from_numpy(inputs.numeric), torch.from_numpy(inputs.categorical)
    logits = network(numeric, categorical)
    assert logits.shape == (16,)
    assert logits.dtype == torch.float32
    logits.sum().backward()
    for name, parameter in network.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name


def test_numeric_inputs_alone_need_no_embedding() -> None:
    network = _network(cardinalities=(), d_embedding=None)
    assert network.category_embeddings is None
    assert network.first_layer.in_features == 4
    output = network.eval()(torch.zeros(3, 4), torch.zeros(3, 0, dtype=torch.int64))
    assert output.shape == (3,)


# --- Hyperparameters ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"dropout": 0.1}, "Unknown"),
        ({"n_layers": None}, "n_layers"),
        ({"n_layers": 0}, "n_layers"),
        ({"n_layers": True}, "n_layers"),
        ({"d": 16.0}, "d must"),
        ({"d_hidden_factor": 0}, "d_hidden_factor"),
        ({"d": 1, "d_hidden_factor": 0.5}, "hidden size"),
        ({"hidden_dropout": 1.0}, "hidden_dropout"),
        ({"residual_dropout": -0.1}, "residual_dropout"),
        ({"d_embedding": 0}, "d_embedding"),
    ],
)
def test_invalid_hyperparameters_are_rejected(
    settings: ModelSettings,
    train: NeuralData,
    valid: NeuralData,
    changes: dict[str, Any],
    message: str,
) -> None:
    hyperparameters = {**HYPERPARAMETERS, **changes}
    for name, value in changes.items():
        if value is None:
            del hyperparameters[name]
    with pytest.raises(SpecificationError, match=message):
        _model(settings).fit(train, valid, hyperparameters)


def test_embedding_size_follows_the_categorical_inputs(settings: ModelSettings) -> None:
    numeric_only = InputLayout(LAYOUT.numeric, (), (), (), 0.0)
    data = neural_data(130, seed=4)
    inputs = replace(data.inputs, categorical=np.zeros((130, 0), np.int64))
    data = replace(data, inputs=inputs)
    model = ResNetModel(
        configuration("RES-u"), 42, settings=settings, layout=numeric_only, device="cpu"
    )
    with pytest.raises(SpecificationError, match="no categorical inputs"):
        model.fit(data, None, HYPERPARAMETERS, iterations=1)
    without = {key: value for key, value in HYPERPARAMETERS.items() if key != "d_embedding"}
    model.fit(data, None, without, iterations=1)
    assert model.parameters["resolved"]["architecture"]["d_embedding"] is None
    with pytest.raises(SpecificationError, match="d_embedding"):
        _model(settings).fit(data, None, without, iterations=1)


def test_a_single_row_last_batch_is_rejected(settings: ModelSettings, valid: NeuralData) -> None:
    data = neural_data(129, seed=5)  # 64 + 64 + 1 rows
    with pytest.raises(SpecificationError, match="last micro-batch of 1 rows"):
        _model(settings).fit(data, valid, HYPERPARAMETERS, objective=v1_objective(valid))


def test_early_stopping_needs_the_v1_objective_and_its_rows(
    settings: ModelSettings, train: NeuralData, valid: NeuralData
) -> None:
    with pytest.raises(SpecificationError, match="needs the V1 objective"):
        _model(settings).fit(train, valid, HYPERPARAMETERS)
    other = neural_data(320, seed=2, with_rows=True)
    shifted = replace(other, ids=other.ids + 1 if other.ids is not None else None)
    with pytest.raises(DataValidationError, match="different V1 rows"):
        _model(settings).fit(train, shifted, HYPERPARAMETERS, objective=v1_objective(valid))
    no_rows = replace(valid, ids=None, times=None)
    with pytest.raises(DataValidationError, match="different V1 rows"):
        _model(settings).fit(train, no_rows, HYPERPARAMETERS, objective=v1_objective(valid))


def test_the_resnet_runs_each_effective_batch_whole(
    settings: ModelSettings, train: NeuralData
) -> None:
    """Batch normalisation needs the statistics of the whole batch: no accumulation."""
    with pytest.raises(SpecificationError, match="cannot accumulate"):
        _model(settings).fit(train, None, HYPERPARAMETERS, iterations=1, plan=BatchPlan(64, 32))
    with pytest.raises(SpecificationError, match="not the batch_size"):
        _model(settings).fit(train, None, HYPERPARAMETERS, iterations=1, plan=BatchPlan(32, 32))
    model = _model(settings)
    model.fit(train, None, HYPERPARAMETERS, iterations=1, plan=BatchPlan(64, 64))
    batching = model.parameters["resolved"]["batching"]
    assert (batching["effective_batch"], batching["micro_batch"]) == (64, 64)
    assert batching["evaluation_batch"] == HYPERPARAMETERS["eval_batch_size"]
    assert batching["final_batch_rows"] == 64  # 640 rows in batches of 64


def test_model_and_device_are_checked(
    settings: ModelSettings, train: NeuralData, valid: NeuralData, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("FTT-u", "LGB-b"):
        with pytest.raises(ValueError, match="not a resnet configuration"):
            _model(settings, name=name)
    model = ResNetModel(configuration("RES-u"), 42, settings=settings, layout=LAYOUT, device="gpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(EnvironmentMismatchError, match="CUDA"):
        model.fit(train, valid, HYPERPARAMETERS, objective=v1_objective(valid))
    with pytest.raises(ModelError, match="not fitted"):
        model.predict(valid.inputs)


# --- Training -------------------------------------------------------------------------------------


def test_early_stopping_keeps_the_best_v1_epoch(fitted: ResNetModel, valid: NeuralData) -> None:
    resolved = fitted.parameters["resolved"]
    schedule, history = resolved["schedule"], resolved["history"]
    numerators = [epoch["v1_numerator"] for epoch in history]
    assert schedule["mode"] == "early_stopping"
    assert schedule["epochs_run"] == len(history)
    best = numerators.index(max(numerators)) + 1  # the earliest best epoch
    assert fitted.fitted_iterations == schedule["epochs_kept"] == best
    if schedule["stop"] == "patience":
        assert len(history) == best + TRAINING["patience"] + 1
    else:
        assert schedule["stop"] == "max_epochs"
        assert len(history) == TRAINING["max_epochs"]
    # The restored network reproduces the best epoch's V1 objective exactly.
    objective = v1_objective(valid)
    restored = objective.queue.recall(fitted.predict(valid.inputs), objective.labels, objective.k)
    assert restored.numerator == schedule["best_numerator"] == max(numerators)
    assert schedule["best_objective"] == restored.value
    assert schedule["objective"] == objective.identity


def test_restored_state_is_exactly_the_best_epoch(
    settings: ModelSettings, train: NeuralData, early_stopped: ResNetModel
) -> None:
    """V1 evaluation draws no random numbers, so the network restored at the best epoch b must
    equal, bit for bit, the network of a refit for exactly b epochs with the same seed: every
    parameter and every batch-normalisation buffer."""
    schedule = early_stopped.parameters["resolved"]["schedule"]
    assert schedule["stop"] == "patience"
    best = early_stopped.fitted_iterations
    assert best < schedule["epochs_run"]  # later epochs were trained, then discarded
    refit = _fit(settings, train, None, hyperparameters=EARLY_STOPPED, iterations=best)
    restored = early_stopped._network_or_raise().state_dict()
    fixed = refit._network_or_raise().state_dict()
    assert restored.keys() == fixed.keys()
    assert any("running_mean" in key for key in restored)
    for key, value in restored.items():
        assert torch.equal(value, fixed[key]), key


def test_the_network_learns_the_signal(fitted: ResNetModel, valid: NeuralData) -> None:
    """Better than a constant score, whose daily top-K review is decided by the tie order alone."""
    objective = v1_objective(valid)
    constant = objective.queue.recall(
        np.full(valid.inputs.rows, 0.5), objective.labels, objective.k
    )
    assert fitted.parameters["resolved"]["schedule"]["best_objective"] > constant.value + 0.08


def test_the_fit_is_float32_under_any_default_dtype(
    settings: ModelSettings, train: NeuralData
) -> None:
    reference = _fit(settings, train, None, iterations=2)
    torch.set_default_dtype(torch.float64)  # a caller's setting; restored by the autouse fixture
    model = _fit(settings, train, None, iterations=2)
    assert torch.get_default_dtype() == torch.float64  # left as the caller set it
    expected = reference._network_or_raise().state_dict()
    for key, value in model._network_or_raise().state_dict().items():
        assert torch.equal(value, expected[key]), key
    observed = model.parameters["resolved"]["observed_dtypes"]
    assert observed == reference.parameters["resolved"]["observed_dtypes"]


def test_dr013_settings_are_applied_and_recorded(
    fitted: ResNetModel, settings: ModelSettings
) -> None:
    resolved = fitted.parameters["resolved"]
    precision = resolved["precision"]
    assert precision["matmul_fp32_precision"] == settings.neural.matmul_fp32_precision
    assert precision["allow_bf16_reduced_precision_reduction"] is False
    assert precision["deterministic_algorithms"] is True
    assert precision["deterministic_warn_only"] is False
    assert precision["cudnn_benchmark"] is False
    assert (precision["training_dtype"], precision["prediction_dtype"]) == ("bfloat16", "float32")
    assert resolved["seeds"]["run"] == 42
    assert set(resolved["seeds"]) == {"run", "batch_order", "category_dropout"}
    optimizer = resolved["optimizer"]
    assert optimizer["class"] == "AdamW"
    assert optimizer["lr"] == TRAINING["lr"]
    assert optimizer["weight_decay"] == TRAINING["weight_decay"]
    assert (optimizer["betas"], optimizer["eps"]) == ((0.9, 0.999), 1e-8)
    assert (optimizer["foreach"], optimizer["fused"], optimizer["amsgrad"]) == (False,) * 3
    assert resolved["batches_per_epoch"] == 10
    names = [name for name, _ in fitted._network_or_raise().named_parameters()]
    groups = optimizer["parameter_groups"]  # one group: weight decay on every parameter
    assert groups == [{"weight_decay": TRAINING["weight_decay"], "parameters": names}]
    observed = resolved["observed_dtypes"]  # what training actually used
    assert observed["training_logits"] == ["torch.bfloat16"]
    assert observed["optimizer_state"] == {
        "exp_avg": ["torch.float32"],
        "exp_avg_sq": ["torch.float32"],
        "step": ["torch.float32"],
    }
    network_parameters = fitted._network_or_raise().parameters()
    assert {parameter.dtype for parameter in network_parameters} == {torch.float32}


def test_the_same_seed_repeats_exactly(
    settings: ModelSettings, train: NeuralData, valid: NeuralData, fitted: ResNetModel
) -> None:
    again = _fit(settings, train, valid)
    assert np.array_equal(again.predict(valid.inputs), fitted.predict(valid.inputs))

    def trajectory(model: ResNetModel) -> list[tuple[float, float]]:
        history = model.parameters["resolved"]["history"]
        return [(epoch["train_loss"], epoch["v1_numerator"]) for epoch in history]

    assert trajectory(again) == trajectory(fitted)
    other = _fit(settings, train, valid, seed=7)
    assert not np.array_equal(other.predict(valid.inputs), fitted.predict(valid.inputs))


def test_fixed_epochs_train_exactly_that_many(settings: ModelSettings, train: NeuralData) -> None:
    loop_only = {name: TRAINING[name] for name in ("lr", "weight_decay", "batch_size")}
    hyperparameters = {**loop_only, "eval_batch_size": 96, **RESNET}
    model = _fit(settings, train, None, hyperparameters=hyperparameters, iterations=3)
    resolved = model.parameters["resolved"]
    assert model.fitted_iterations == 3
    assert resolved["schedule"]["mode"] == "fixed"
    assert resolved["schedule"]["stop"] == "fixed"
    assert [epoch["v1_numerator"] for epoch in resolved["history"]] == [None] * 3
    assert [epoch["v1_objective"] for epoch in resolved["history"]] == [None] * 3
    assert resolved["schedule"]["objective"] is None
    with pytest.raises(SpecificationError, match="max_epochs"):
        _model(settings).fit(train, train, hyperparameters)


def test_fit_arguments_are_checked(settings: ModelSettings, train: NeuralData) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        _model(settings).fit(train, train, HYPERPARAMETERS, iterations=3)
    with pytest.raises(ValueError, match="exactly one"):
        _model(settings).fit(train, None, HYPERPARAMETERS)


# --- Prediction -----------------------------------------------------------------------------------


def test_prediction_is_float32_and_repeatable(fitted: ResNetModel, valid: NeuralData) -> None:
    scores = fitted.predict(valid.inputs)
    assert scores.dtype == np.float64
    assert ((scores >= 0) & (scores <= 1)).all()
    assert np.array_equal(scores, fitted.predict(valid.inputs))  # no dropout of any kind
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert np.array_equal(scores, fitted.predict(valid.inputs))
    logits = predict_logits(
        fitted._network_or_raise(), valid.inputs, device=torch.device("cpu"), batch_size=1000
    )
    expected = torch.sigmoid(torch.from_numpy(logits).double()).numpy()
    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-6)


def test_prediction_checks_its_inputs(fitted: ResNetModel, valid: NeuralData) -> None:
    categorical = valid.inputs.categorical.copy()
    categorical[0, 0] = 5
    with pytest.raises(DataValidationError, match="vocabularies"):
        fitted.predict(replace(valid.inputs, categorical=categorical))


# --- Persistence and record ----------------------------------------------------------------------


def test_save_and_load_reproduce_the_model(
    fitted: ResNetModel,
    fixed_epochs: ResNetModel,
    valid: NeuralData,
    settings: ModelSettings,
    tmp_path: Path,
) -> None:
    """Both schedules: V1 early stopping and a fixed epoch count."""
    for model in (fitted, fixed_epochs):
        directory = tmp_path / model.parameters["resolved"]["schedule"]["mode"]
        model.save(directory)
        loaded = _reload(directory, settings)
        assert np.array_equal(loaded.predict(valid.inputs), model.predict(valid.inputs))
        assert loaded.fitted_iterations == model.fitted_iterations
        assert jsonable(loaded.parameters) == jsonable(model.parameters)
    assert fixed_epochs.fitted_iterations == 3


def _edit_record(section: str, key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def edit(manifest: dict[str, Any]) -> None:
        manifest["training"][section][key] = value

    return edit


def _edit(key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def edit(manifest: dict[str, Any]) -> None:
        manifest[key] = value

    return edit


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_edit("format_version", 1), "format version"),  # version 1 has no train_seconds
        (_edit("format_version", 2), "format version"),  # no batch plan, no V1 objective
        (_edit("format_version", 4), "format version"),
        (_edit("library_version", "2.10.0"), "does not match"),
        (_edit("seed", 7), "does not match"),
        (_edit("seed", 42.0), "integer"),
        (_edit("configuration", "RES-u"), "does not match"),
        (_edit("device", "gpu"), "does not match"),
        (_edit("model_sha256", "0" * 64), "do not match"),
        (_edit("epochs", 0), "malformed"),
        (_edit("architecture", {}), "architecture"),
        (lambda manifest: manifest["hyperparameters"].update(n_layers=3), "architecture"),
        (lambda manifest: manifest["layout"].update(category_dropout=0.0), "does not match"),
        (lambda manifest: manifest.pop("training"), "incomplete"),
        (_edit("training", {}), "training record is incomplete"),
        (lambda manifest: manifest["training"].pop("observed_dtypes"), "incomplete"),
        (_edit_record("precision", "matmul_fp32_precision", "tf32"), "precision settings"),
        (_edit_record("seeds", "batch_order", 1), "seeds"),
        (_edit_record("schedule", "epochs_run", 1), "inconsistent"),
        (_edit("epochs", 99), "inconsistent"),
        (lambda manifest: manifest["training"]["history"].pop(), "inconsistent"),
        (lambda manifest: manifest["training"]["history"].reverse(), "epochs 1, 2"),
        (
            lambda manifest: manifest["training"]["history"][0].pop("v1_numerator"),
            "history entry 1",
        ),
        (lambda manifest: manifest["training"]["history"][1].update(train_loss=None), "entry 2"),
        (lambda manifest: manifest["training"]["history"][0].update(seconds=10**400), "entry 1"),
        (lambda manifest: manifest["training"]["history"][0].update(train_seconds=1e9), "entry 1"),
        (lambda manifest: manifest["training"].update(optimizer=None), "'optimizer' is malformed"),
        (lambda manifest: manifest["training"]["schedule"].pop("stop"), "schedule is malformed"),
        (_edit_record("optimizer", "lr", 0.5), "optimiser settings"),
        (_edit_record("observed_dtypes", "training_logits", ["torch.float32"]), "observed dtypes"),
        (_edit_record("device", "type", "cuda"), "device type"),
        (_edit_record("schedule", "patience", 5), "does not match the saved hyperparameters"),
        (_edit_record("schedule", "stop", "fixed"), "early-stopping rule"),
        (_edit_record("batching", "effective_batch", 128), "batch plan"),
        (_edit_record("batching", "micro_batch", 32), "batch plan"),  # the ResNet cannot accumulate
        (_edit_record("batching", "micro_batch", 0), "batch plan"),
        (_edit_record("batching", "evaluation_batch", 7), "batch plan"),
        (_edit_record("batching", "loss_reduction", "mean"), "batch plan"),
        (_edit_record("batching", "final_batch_rows", 65), "batch plan"),
        (_edit_record("batching", "final_batch_rows", 1), "batch plan"),  # a one-row BatchNorm tail
        (lambda manifest: manifest["training"]["batching"].pop("micro_batch"), "batch plan"),
        (lambda manifest: manifest["training"].pop("batching"), "incomplete"),
        (
            lambda manifest: manifest["training"]["schedule"].update(objective={"name": "x"}),
            "V1 objective",
        ),
        (
            lambda manifest: manifest["training"]["history"][0].update(v1_objective=0.123456),
            "inconsistent with their numerators",
        ),
    ],
)
def test_load_rejects_a_mismatched_manifest(
    fitted: ResNetModel,
    settings: ModelSettings,
    tmp_path: Path,
    edit: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    fitted.save(tmp_path)
    _rewrite_manifest(tmp_path, edit)
    with pytest.raises(ModelError, match=message):
        _reload(tmp_path, settings)


def _claim_a_later_tied_epoch(manifest: dict[str, Any]) -> None:
    """Give the epoch after the best one the best score, and claim it as the kept epoch."""
    best = manifest["epochs"]
    history = manifest["training"]["history"]
    for key in ("v1_numerator", "v1_objective"):
        history[best][key] = history[best - 1][key]  # epoch best + 1 ties the best
    manifest["epochs"] = manifest["training"]["schedule"]["epochs_kept"] = best + 1


def _continue_after_stopping(manifest: dict[str, Any]) -> None:
    """Append an epoch after the one at which the early-stopping rule ended training."""
    training = manifest["training"]
    training["history"].append({**training["history"][-1], "epoch": len(training["history"]) + 1})
    training["schedule"]["epochs_run"] += 1


@pytest.mark.parametrize("edit", [_claim_a_later_tied_epoch, _continue_after_stopping])
def test_load_replays_the_early_stopping_rule(
    early_stopped: ResNetModel,
    settings: ModelSettings,
    tmp_path: Path,
    edit: Callable[[dict[str, Any]], None],
) -> None:
    early_stopped.save(tmp_path)
    _reload(tmp_path, settings)  # the record as saved replays exactly
    _rewrite_manifest(tmp_path, edit)
    with pytest.raises(ModelError, match="early-stopping rule"):
        _reload(tmp_path, settings)


def _limit_epochs(epochs: int) -> Callable[[dict[str, Any]], None]:
    """Set the epoch limit, in the saved hyperparameters and the schedule alike, to ``epochs``."""

    def edit(manifest: dict[str, Any]) -> None:
        manifest["hyperparameters"]["max_epochs"] = epochs
        manifest["training"]["schedule"]["max_epochs"] = epochs

    return edit


def test_load_applies_the_epoch_limit(
    early_stopped: ResNetModel, settings: ModelSettings, tmp_path: Path
) -> None:
    """A patience stop on the last allowed epoch is valid; a history longer than the limit is
    not, although it ends by patience."""
    early_stopped.save(tmp_path)
    epochs_run = early_stopped.parameters["resolved"]["schedule"]["epochs_run"]
    _rewrite_manifest(tmp_path, _limit_epochs(epochs_run))
    loaded = _reload(tmp_path, settings)
    assert loaded.parameters["resolved"]["schedule"]["stop"] == "patience"
    _rewrite_manifest(tmp_path, _limit_epochs(epochs_run - 1))
    with pytest.raises(ModelError, match="early-stopping rule"):
        _reload(tmp_path, settings)


def _truncate(manifest: dict[str, Any]) -> None:
    manifest["epochs"] = manifest["training"]["schedule"]["epochs_kept"] = 2


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_truncate, "inconsistent"),
        (_edit_record("schedule", "best_numerator", 5), "inconsistent"),
        (_edit_record("schedule", "stop", "patience"), "inconsistent"),
        (lambda manifest: manifest["training"]["history"][0].update(v1_numerator=5), "entry 1"),
    ],
)
def test_load_rejects_an_inconsistent_fixed_epoch_record(
    fixed_epochs: ResNetModel,
    settings: ModelSettings,
    tmp_path: Path,
    edit: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    fixed_epochs.save(tmp_path)
    _rewrite_manifest(tmp_path, edit)
    with pytest.raises(ModelError, match=message):
        _reload(tmp_path, settings)


def test_load_rejects_a_state_of_another_network(
    fitted: ResNetModel, settings: ModelSettings, tmp_path: Path
) -> None:
    fitted.save(tmp_path)
    other = tmp_path / "other"
    _fit(
        settings,
        neural_data(128, seed=8),
        None,
        hyperparameters={**HYPERPARAMETERS, "d": 8},
        iterations=1,
    ).save(other)
    (tmp_path / STATE_FILE).write_bytes((other / STATE_FILE).read_bytes())
    path = tmp_path / MANIFEST_FILE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    other_manifest = json.loads((other / MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest["model_sha256"] = other_manifest["model_sha256"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelError, match="cannot be loaded"):
        _reload(tmp_path, settings)


def test_run_record_carries_the_precision_settings(fitted: ResNetModel) -> None:
    _, snapshot = load_settings_with_snapshot(REPO_PATHS)
    record = model_run_record(
        model=fitted,
        settings=snapshot,
        hyperparameters=HYPERPARAMETERS,
        lightgbm_build=None,
        environment={},
        code={},
    )
    assert record["configuration"] == "RES-b"
    assert (record["model"], record["family"]) == ("resnet", "neural")
    assert record["fitted_iterations"] == fitted.fitted_iterations
    json.dumps(record, allow_nan=False)
    parameters = record["parameters"]
    assert isinstance(parameters, dict)
    resolved = parameters["resolved"]
    assert isinstance(resolved, dict)
    precision = resolved["precision"]
    assert isinstance(precision, dict)
    assert precision["matmul_fp32_precision"] == "ieee"
    assert precision["deterministic_algorithms"] is True
