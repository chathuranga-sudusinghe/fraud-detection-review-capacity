"""Components of the shared neural training loop (DR-012 inputs, DR-013 §4 settings) on small
synthetic tensors on the CPU (no project data; the GPU path is exercised by the GPU probe,
Issue #77). Model-level behaviour is tested through the ResNet-style MLP
(``test_resnet_model.py``)."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from review_capacity.errors import (
    DataValidationError,
    EnvironmentMismatchError,
    ModelError,
    SpecificationError,
)
from review_capacity.models.neural_training import (
    EARLY_STOPPING_PARAMETERS,
    METRICS,
    TRAINING_PARAMETERS,
    EarlyStopping,
    InputLayout,
    NeuralData,
    NeuralInputs,
    derived_seeds,
    drop_categories,
    epoch_batches,
    float32_default,
    generator,
    parameter_group_record,
    predict_logits,
    probabilities,
    torch_device,
    train_network,
    training_settings,
    weighted_bce,
)
from support.neural import LAYOUT, TRAINING, neural_data, neural_inputs, preserved_torch_state


@pytest.fixture(autouse=True)
def restore_torch_state() -> Iterator[None]:
    with preserved_torch_state():
        yield


# --- Layout and inputs ----------------------------------------------------------------------------


NO_INPUTS = {"numeric": (), "categorical": (), "cardinalities": (), "missing_indices": ()}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (NO_INPUTS, "at least"),
        ({"numeric": ("V1", "V1")}, "unique"),
        ({"numeric": ("ProductCD",)}, "unique"),
        ({"numeric": ("",)}, "non-empty"),
        ({"cardinalities": (5,)}, "per categorical input"),
        ({"cardinalities": (5, 0)}, "positive integer"),
        ({"cardinalities": (5, True)}, "positive integer"),
        ({"missing_indices": (0, 3)}, "outside its vocabulary"),
        ({"missing_indices": (0, -1)}, "outside its vocabulary"),
        ({"category_dropout": 1.0}, "float in"),
        ({"category_dropout": -0.01}, "float in"),
        ({"category_dropout": math.nan}, "float in"),
        ({"category_dropout": 0}, "float in"),
    ],
)
def test_invalid_layout_is_rejected(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(SpecificationError, match=message):
        replace(LAYOUT, **changes)


def test_layout_without_categorical_inputs_is_valid() -> None:
    layout = InputLayout(("a", "b"), (), (), (), 0.0)
    assert layout.to_json() == {
        "numeric": ["a", "b"],
        "categorical": [],
        "cardinalities": [],
        "missing_indices": [],
        "category_dropout": 0.0,
    }


@pytest.mark.parametrize(
    ("numeric", "categorical", "message"),
    [
        (np.zeros((3, 4)), np.zeros((3, 2), np.int64), "float32"),
        (np.zeros(3, np.float32), np.zeros((3, 2), np.int64), "two-dimensional"),
        (np.zeros((3, 4), np.float32), np.zeros((3, 2), np.int32), "int64"),
        (np.zeros((3, 4), np.float32), np.zeros((2, 2), np.int64), "categorical rows"),
        (np.zeros((0, 4), np.float32), np.zeros((0, 2), np.int64), "empty"),
    ],
)
def test_malformed_inputs_are_rejected(
    numeric: np.ndarray[Any, Any], categorical: np.ndarray[Any, Any], message: str
) -> None:
    with pytest.raises(DataValidationError, match=message):
        NeuralInputs(numeric=numeric, categorical=categorical)


def test_inputs_are_checked_against_the_layout() -> None:
    inputs = neural_inputs(20, seed=0)
    inputs.check(LAYOUT)
    with pytest.raises(DataValidationError, match="columns"):
        NeuralInputs(inputs.numeric[:, :3].copy(), inputs.categorical).check(LAYOUT)
    numeric = inputs.numeric.copy()
    numeric[4, 1] = np.inf
    with pytest.raises(DataValidationError, match="finite"):
        NeuralInputs(numeric, inputs.categorical).check(LAYOUT)
    for value in (-1, 3):  # the second input has 3 categories: indices 0 to 2
        categorical = inputs.categorical.copy()
        categorical[2, 1] = value
        with pytest.raises(DataValidationError, match="vocabularies"):
            NeuralInputs(inputs.numeric, categorical).check(LAYOUT)


def test_malformed_labels_and_weights_are_rejected() -> None:
    data = neural_data(10, seed=0)
    with pytest.raises(DataValidationError, match="Labels"):
        NeuralData(data.inputs, data.labels.astype(np.int64), data.weights)
    with pytest.raises(DataValidationError, match="Labels"):
        NeuralData(data.inputs, np.full(10, 2, np.int8), data.weights)
    with pytest.raises(DataValidationError, match="Labels"):
        NeuralData(data.inputs, data.labels[:9], data.weights)
    with pytest.raises(DataValidationError, match="float64"):
        NeuralData(data.inputs, data.labels, data.weights.astype(np.float32))
    with pytest.raises(DataValidationError, match="positive"):
        NeuralData(data.inputs, data.labels, np.zeros(10))


# --- Hyperparameters -----------------------------------------------------------------------------


def test_training_settings_are_read_explicitly() -> None:
    settings = training_settings(TRAINING, early_stopping=True)
    assert (settings.lr, settings.batch_size, settings.patience) == (1e-2, 64, 2)
    assert settings.metric == "average_precision"
    loop_only = {name: TRAINING[name] for name in TRAINING_PARAMETERS}
    fixed = training_settings(loop_only, early_stopping=False)
    assert (fixed.max_epochs, fixed.patience, fixed.metric) == (None, None, None)


@pytest.mark.parametrize("name", TRAINING_PARAMETERS + EARLY_STOPPING_PARAMETERS)
def test_every_training_setting_is_required(name: str) -> None:
    values = {key: value for key, value in TRAINING.items() if key != name}
    with pytest.raises(SpecificationError, match=name):
        training_settings(values, early_stopping=True)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("lr", 0.0),
        ("lr", -1e-3),
        ("lr", math.inf),
        ("lr", "0.001"),
        ("lr", True),
        ("weight_decay", -1e-6),
        ("weight_decay", math.nan),
        ("batch_size", 0),
        ("batch_size", 64.0),
        ("batch_size", True),
        ("eval_batch_size", -1),
        ("max_epochs", 0),
        ("patience", -1),
        ("patience", 1.0),
        ("metric", "f1"),
        ("metric", None),
    ],
)
def test_invalid_training_setting_is_rejected(name: str, value: Any) -> None:
    with pytest.raises(SpecificationError, match=name):
        training_settings({**TRAINING, name: value}, early_stopping=True)


# --- Seeds, device and loop components -----------------------------------------------------------


def test_derived_seeds_are_fixed_by_the_run_seed() -> None:
    seeds = derived_seeds(42)
    assert seeds == derived_seeds(42)
    assert set(seeds) == {"batch_order", "category_dropout"}
    assert len(set(seeds.values())) == 2
    assert seeds != derived_seeds(43)
    expected = np.random.SeedSequence(42, spawn_key=(1,)).generate_state(1)[0]
    assert seeds["batch_order"] == int(expected)
    for invalid in (-1, 2**32, True, 1.0):
        with pytest.raises(ValueError, match="Seed"):
            derived_seeds(invalid)  # type: ignore[arg-type]


def test_cpu_device_and_no_gpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    assert torch_device("cpu") == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(EnvironmentMismatchError, match="CUDA"):
        torch_device("gpu")
    with pytest.raises(ValueError, match="Unsupported device"):
        torch_device("tpu")  # type: ignore[arg-type]


def test_category_dropout_replaces_values_by_their_missing_index() -> None:
    categorical = torch.randint(1, 3, (20_000, 2), generator=generator(0))  # values 1 and 2
    missing = torch.tensor([0, 3])  # never equal to a value, so every replacement is a change
    assert drop_categories(categorical, missing, 0.0, generator(1)) is categorical
    dropped = drop_categories(categorical, missing, 0.25, generator(1))
    assert torch.equal(dropped, drop_categories(categorical, missing, 0.25, generator(1)))
    changed = dropped != categorical
    for column in range(2):
        share = changed[:, column].double().mean().item()
        assert 0.23 < share < 0.27
        assert bool((dropped[changed[:, column], column] == missing[column]).all())
    kept = ~changed
    assert torch.equal(dropped[kept], categorical[kept])


def test_weighted_loss_is_the_weighted_row_mean_in_float32() -> None:
    logits = torch.tensor([2.0, -1.0, 0.5, -3.0]).to(torch.bfloat16)
    labels = torch.tensor([1.0, 0.0, 0.0, 1.0])
    weights = torch.tensor([10.0, 1.0, 1.0, 10.0])
    loss = weighted_bce(logits, labels, weights)
    assert loss.dtype == torch.float32
    p = torch.sigmoid(logits.double())
    rows = -(labels.double() * p.log() + (1 - labels.double()) * (1 - p).log())
    assert loss.item() == pytest.approx((weights.double() * rows).sum().item() / 4, rel=1e-6)
    with pytest.raises(TypeError, match="float32"):
        weighted_bce(logits, labels.double(), weights)


def test_early_stopping_counts_patience_plus_one_epochs_without_improvement() -> None:
    tracker = EarlyStopping(2, greater_is_better=True)
    outcomes = [tracker.update(epoch, score) for epoch, score in enumerate([0.5, 0.6], start=1)]
    assert outcomes == [True, True]
    for epoch, score in ((3, 0.6), (4, 0.55)):  # a tie is not an improvement
        assert not tracker.update(epoch, score)
        assert not tracker.stop
    assert not tracker.update(5, 0.6)
    assert tracker.stop
    assert (tracker.best, tracker.best_epoch) == (0.6, 2)  # the earliest best epoch is kept
    lower = EarlyStopping(0, greater_is_better=False)
    assert lower.update(1, 0.3)
    assert lower.update(2, 0.2)
    assert not lower.update(3, 0.25)
    assert lower.stop
    with pytest.raises(ModelError, match="not finite"):
        lower.update(4, math.nan)


def test_scores_keep_the_resolution_of_the_logits() -> None:
    logits = np.array([20.0, 25.0, 30.0, 35.0, -30.0], dtype=np.float32)
    assert bool((torch.sigmoid(torch.from_numpy(logits[:4])) == 1).all())  # float32 rounds to 1
    scores = probabilities(logits)
    assert scores.dtype == np.float64
    assert len(np.unique(scores)) == 5
    assert (scores[:4] < 1).all()
    assert np.all(np.diff(scores[:4]) > 0)


class _Linear(nn.Module):
    """A network with the neural-model interface: numeric inputs and category indices."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(len(LAYOUT.numeric) + len(LAYOUT.categorical), 1)

    def forward(self, numeric: Tensor, categorical: Tensor) -> Tensor:
        logits: Tensor = self.linear(torch.cat([numeric, categorical.float()], dim=-1))
        return logits.squeeze(-1)


def test_prediction_is_float32_with_autocast_disabled() -> None:
    network = _Linear()
    inputs = neural_inputs(50, seed=3)
    device = torch.device("cpu")
    plain = predict_logits(network, inputs, device=device, batch_size=16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):  # a caller's autocast region
        inside = predict_logits(network, inputs, device=device, batch_size=16)
    assert plain.dtype == np.float32
    assert np.array_equal(plain, inside)
    with torch.no_grad():
        expected = network(torch.from_numpy(inputs.numeric), torch.from_numpy(inputs.categorical))
    np.testing.assert_allclose(plain, expected.numpy(), rtol=1e-6, atol=1e-6)
    assert not network.training


def test_epoch_batches_visit_every_row_once() -> None:
    batches = epoch_batches(130, 64, generator(5))
    assert [len(batch) for batch in batches] == [64, 64, 2]  # the last batch keeps the rest
    assert np.array_equal(np.sort(np.concatenate(batches)), np.arange(130))
    again = epoch_batches(130, 64, generator(5))
    assert all(np.array_equal(a, b) for a, b in zip(batches, again, strict=True))
    rng = generator(5)
    first, second = epoch_batches(130, 64, rng), epoch_batches(130, 64, rng)
    assert not np.array_equal(np.concatenate(first), np.concatenate(second))  # a new order


def test_float32_default_restores_the_callers_default() -> None:
    torch.set_default_dtype(torch.float64)  # restored by the autouse fixture
    with float32_default():
        assert torch.get_default_dtype() == torch.float32
        assert torch.zeros(1).dtype == torch.float32
    assert torch.get_default_dtype() == torch.float64


def test_category_dropout_does_not_depend_on_the_default_dtype() -> None:
    categorical = torch.ones((500, 2), dtype=torch.int64)
    missing = torch.tensor([0, 0])
    expected = drop_categories(categorical, missing, 0.3, generator(2))
    torch.set_default_dtype(torch.float64)  # restored by the autouse fixture
    assert torch.equal(drop_categories(categorical, missing, 0.3, generator(2)), expected)


# --- V1 metrics -----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(METRICS))
def test_metric_weights_count_as_repeated_rows(name: str) -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 200).astype(np.int8)
    scores = 0.3 * labels + 0.7 * rng.random(200)
    weights = np.where(rng.random(200) < 0.3, 2.0, 1.0)
    repeat = np.repeat(np.arange(200), weights.astype(np.int64))
    compute = METRICS[name].compute
    weighted = compute(labels, scores, weights)
    assert weighted == pytest.approx(compute(labels[repeat], scores[repeat], np.ones(len(repeat))))
    assert weighted != pytest.approx(compute(labels, scores, np.ones(200)))


def test_metric_directions() -> None:
    labels = np.array([0, 0, 1, 1], np.int8)
    weights = np.ones(4)
    good, bad = np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.9, 0.8, 0.2, 0.1])
    for name, metric in METRICS.items():
        better = metric.compute(labels, good, weights) > metric.compute(labels, bad, weights)
        assert better == metric.greater_is_better, name


def test_only_ranking_metrics_need_both_classes() -> None:
    needing = {name for name, metric in METRICS.items() if metric.needs_both_classes}
    assert needing == {"auc", "average_precision"}
    labels = np.zeros(4, np.int8)
    loss = METRICS["binary_logloss"].compute(labels, np.full(4, 0.1), np.ones(4))
    assert loss == pytest.approx(-math.log(0.9))


# --- Numerical failures ---------------------------------------------------------------------------


class _Constant(nn.Module):
    """A network whose every logit is ``value``, with one trainable weight."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, numeric: Tensor, categorical: Tensor) -> Tensor:
        return (numeric[:, 0] + categorical[:, 0]) * 0 * self.weight + self.value


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_logits_stop_prediction(value: float) -> None:
    inputs = neural_inputs(10, seed=0)
    with pytest.raises(ModelError, match="non-finite logits"):
        predict_logits(_Constant(value), inputs, device=torch.device("cpu"), batch_size=4)


def test_non_finite_loss_stops_before_the_batch_updates_the_network() -> None:
    network = _Constant(math.nan)
    before = network.weight.detach().clone()
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.1)
    settings = training_settings(
        {name: TRAINING[name] for name in TRAINING_PARAMETERS}, early_stopping=False
    )
    data = neural_data(100, seed=0)
    with pytest.raises(ModelError, match="epoch 1, batch 1"):
        train_network(
            network,
            optimizer,
            data,
            None,
            settings,
            layout=LAYOUT,
            device=torch.device("cpu"),
            seeds=derived_seeds(0),
            epochs=1,
            name="test",
        )
    assert torch.equal(network.weight.detach(), before)
    assert not optimizer.state  # no optimiser step was taken


def test_parameter_groups_are_recorded_by_name() -> None:
    network = nn.Linear(2, 1)
    groups: list[dict[str, Any]] = [
        {"params": [network.weight], "weight_decay": 1e-4},
        {"params": [network.bias], "weight_decay": 0},
    ]
    record = parameter_group_record(network, groups)
    assert record == [
        {"weight_decay": 1e-4, "parameters": ["weight"]},
        {"weight_decay": 0.0, "parameters": ["bias"]},
    ]
    foreign = nn.Parameter(torch.zeros(1))
    for invalid in (
        [{"params": [network.weight], "weight_decay": 0.0}],  # the bias is missing
        [{"params": [network.weight, network.bias, foreign], "weight_decay": 0.0}],
        [{"params": [network.weight, network.weight, network.bias], "weight_decay": 0.0}],
        [
            {"params": [network.weight, network.bias], "weight_decay": 0.0},
            {"params": [network.bias], "weight_decay": 0.0},  # in two groups
        ],
    ):
        with pytest.raises(ValueError, match="Parameter groups must hold"):
            parameter_group_record(network, invalid)
