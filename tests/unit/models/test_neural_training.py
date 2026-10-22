"""Components of the shared neural training loop (DR-012 inputs, DR-013 §4 settings) on small
synthetic tensors on the CPU (no project data; the GPU path is exercised by the GPU probe,
Issue #77). Model-level behaviour is tested through the ResNet-style MLP
(``test_resnet_model.py``)."""

from __future__ import annotations

import contextlib
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
    TRAINING_PARAMETERS,
    BatchPlan,
    EarlyStopping,
    InputLayout,
    NeuralData,
    NeuralInputs,
    batching_record,
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
from support.neural import (
    LAYOUT,
    TRAINING,
    neural_data,
    neural_inputs,
    preserved_torch_state,
    v1_objective,
)


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


def test_row_identifiers_and_times_are_metadata_given_together() -> None:
    data = neural_data(10, seed=0, with_rows=True)
    assert data.ids is not None
    assert data.times is not None
    with pytest.raises(DataValidationError, match="together"):
        NeuralData(data.inputs, data.labels, data.weights, ids=data.ids)
    with pytest.raises(DataValidationError, match="identifiers"):
        NeuralData(data.inputs, data.labels, data.weights, ids=data.ids[:9], times=data.times)
    with pytest.raises(DataValidationError, match="times"):
        NeuralData(
            data.inputs, data.labels, data.weights, ids=data.ids, times=data.times.astype(np.int32)
        )


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
    assert (settings.max_epochs, settings.eval_batch_size) == (12, 96)
    loop_only = {name: TRAINING[name] for name in TRAINING_PARAMETERS}
    fixed = training_settings(loop_only, early_stopping=False)
    assert (fixed.max_epochs, fixed.patience) == (None, None)


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


def test_micro_batch_losses_add_up_to_the_effective_batch_loss() -> None:
    logits = torch.tensor([2.0, -1.0, 0.5, -3.0, 1.5])
    labels = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0])
    weights = torch.tensor([10.0, 1.0, 2.0, 10.0, 3.0])
    whole = weighted_bce(logits, labels, weights)
    parts = weighted_bce(logits[:2], labels[:2], weights[:2], rows=5) + weighted_bce(
        logits[2:], labels[2:], weights[2:], rows=5
    )
    assert parts.item() == pytest.approx(whole.item(), rel=1e-6)


def test_early_stopping_counts_patience_plus_one_epochs_without_improvement() -> None:
    tracker = EarlyStopping(2)
    outcomes = [tracker.update(epoch, value) for epoch, value in enumerate([50, 60], start=1)]
    assert outcomes == [True, True]
    for epoch, value in ((3, 60), (4, 55)):  # a tie is not an improvement
        assert not tracker.update(epoch, value)
        assert not tracker.stop
    assert not tracker.update(5, 60)
    assert tracker.stop
    assert (tracker.best, tracker.best_epoch) == (60, 2)  # the earliest best epoch is kept
    immediate = EarlyStopping(0)
    assert immediate.update(1, 3)
    assert not immediate.update(2, 3)
    assert immediate.stop


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
            plan=BatchPlan(64, 64),
            objective=None,
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


# --- The batch plan and the effective-batch loop --------------------------------------------------


def test_a_batch_plan_has_positive_sizes_and_a_micro_batch_within_the_effective_batch() -> None:
    plan = BatchPlan(1024, 256)
    assert plan.micro_sizes(1024) == [256, 256, 256, 256]
    assert plan.micro_sizes(1016) == [256, 256, 256, 248]  # the final batch, kept
    assert BatchPlan(64, 64).micro_sizes(22) == [22]
    assert BatchPlan(5, 2).micro_sizes(5) == [2, 2, 1]
    for effective, micro in ((0, 1), (4, 0), (4, 5), (True, 1), (4, 2.0)):
        with pytest.raises(SpecificationError, match="batch"):
            BatchPlan(effective, micro)  # type: ignore[arg-type]


def test_the_recorded_batching_states_the_plan_the_reduction_and_the_evaluation_batch() -> None:
    record = batching_record(BatchPlan(1024, 256), 368_632, 4096)
    assert record["effective_batch"] == 1024
    assert record["micro_batch"] == 256
    assert record["final_batch_rows"] == 1016  # 368,632 = 359 x 1,024 + 1,016
    assert record["evaluation_batch"] == 4096
    assert "m" in record["loss_reduction"]
    assert batching_record(BatchPlan(64, 64), 128, 96)["final_batch_rows"] == 64


class _Counting(torch.optim.AdamW):
    """AdamW that counts its steps and the clearing of gradients."""

    steps = 0
    clears = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.gradients: list[list[Tensor]] = []  # the accumulated gradients before each step

    def step(self, *args: Any, **kwargs: Any) -> Any:
        self.steps += 1
        parameters = [p for group in self.param_groups for p in group["params"]]
        self.gradients.append([p.grad.clone() for p in parameters if p.grad is not None])
        return super().step(*args, **kwargs)

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        self.clears += 1
        super().zero_grad(*args, **kwargs)


def _train(
    plan: BatchPlan, rows: int = 150, layout: InputLayout = LAYOUT, seed: int = 4
) -> tuple[_Counting, nn.Module, list[float]]:
    torch.manual_seed(seed)
    network = _Linear()
    optimizer = _Counting(network.parameters(), lr=1e-2, weight_decay=1e-3)
    settings = training_settings(
        {**{name: TRAINING[name] for name in TRAINING_PARAMETERS}, "batch_size": plan.effective},
        early_stopping=False,
    )
    result = train_network(
        network,
        optimizer,
        neural_data(rows, seed=1, fraud_weight=3.0),
        None,
        settings,
        layout=layout,
        device=torch.device("cpu"),
        seeds=derived_seeds(0),
        epochs=2,
        name="test",
        plan=plan,
        objective=None,
    )
    return optimizer, network, [record.train_loss for record in result.history]


@pytest.mark.parametrize("dropout", [0.0, 0.3])
@pytest.mark.parametrize("micro", [64, 24, 7, 1])
def test_an_accumulated_update_equals_the_unsplit_update(
    micro: int, dropout: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With autocast off the micro-batch plan changes nothing but the summation order: the same
    losses, parameters and optimiser state as the unsplit effective batch, to float32 rounding,
    with unequal micro-batches, non-uniform weights, a short final batch (22 of 150 rows) and
    category dropout (drawn once per effective batch, so its masks do not depend on the plan)."""
    monkeypatch.setattr(torch, "autocast", lambda **_: contextlib.nullcontext())
    layout = replace(LAYOUT, category_dropout=dropout)
    reference, network_a, losses_a = _train(BatchPlan(64, 64), layout=layout)
    split, network_b, losses_b = _train(BatchPlan(64, micro), layout=layout)
    assert (reference.steps, reference.clears) == (6, 6)  # 3 effective batches x 2 epochs
    assert (split.steps, split.clears) == (6, 6)  # one step and one clearing per effective batch
    np.testing.assert_allclose(losses_b, losses_a, rtol=1e-5)
    assert len(split.gradients) == len(reference.gradients) == 6
    for before_a, before_b in zip(reference.gradients, split.gradients, strict=True):
        for grad_a, grad_b in zip(before_a, before_b, strict=True):
            torch.testing.assert_close(grad_b, grad_a, rtol=1e-4, atol=1e-7)
    for a, b in zip(network_a.parameters(), network_b.parameters(), strict=True):
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-6)
    state_a, state_b = reference.state_dict()["state"], split.state_dict()["state"]
    for key in state_a:
        for name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(state_b[key][name], state_a[key][name], rtol=1e-4, atol=1e-8)


def test_the_study_effective_batch_has_a_1016_row_tail_and_accumulates_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2,040 rows in effective batches of 1,024: 1,024 and a final batch of 1,016 rows, split into
    uneven micro-batches (300, 300, 300, 124 and 300, 300, 300, 116)."""
    monkeypatch.setattr(torch, "autocast", lambda **_: contextlib.nullcontext())
    layout = replace(LAYOUT, category_dropout=0.2)
    reference, network_a, losses_a = _train(BatchPlan(1024, 1024), rows=2040, layout=layout)
    split, network_b, losses_b = _train(BatchPlan(1024, 300), rows=2040, layout=layout)
    assert (reference.steps, split.steps) == (4, 4)  # 2 effective batches x 2 epochs
    assert BatchPlan(1024, 300).micro_sizes(1016) == [300, 300, 300, 116]
    np.testing.assert_allclose(losses_b, losses_a, rtol=1e-5)
    for before_a, before_b in zip(reference.gradients, split.gradients, strict=True):
        for grad_a, grad_b in zip(before_a, before_b, strict=True):
            torch.testing.assert_close(grad_b, grad_a, rtol=1e-4, atol=1e-7)
    for a, b in zip(network_a.parameters(), network_b.parameters(), strict=True):
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-6)


def test_category_dropout_is_drawn_once_per_effective_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shapes: list[int] = []
    original = drop_categories

    def spy(categorical: Tensor, *args: Any) -> Tensor:
        shapes.append(categorical.shape[0])
        return original(categorical, *args)

    monkeypatch.setattr("review_capacity.models.neural_training.drop_categories", spy)
    _train(BatchPlan(64, 24), layout=replace(LAYOUT, category_dropout=0.3))
    assert shapes == [64, 64, 22] * 2  # whole effective batches, never micro-batches


def test_the_same_plan_and_seed_repeat_exactly_with_dropout_on() -> None:
    layout = replace(LAYOUT, category_dropout=0.3)
    _, first, losses_first = _train(BatchPlan(64, 24), layout=layout)
    _, second, losses_second = _train(BatchPlan(64, 24), layout=layout)
    assert losses_first == losses_second
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(a, b)


class _FailsOnCall(nn.Module):
    """A trainable network whose ``call``-th forward pass gives non-finite logits."""

    def __init__(self, call: int) -> None:
        super().__init__()
        self.call, self.calls = call, 0
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, numeric: Tensor, categorical: Tensor) -> Tensor:
        self.calls += 1
        value = math.nan if self.calls == self.call else 0.0
        return (numeric[:, 0] + categorical[:, 0]) * 0 * self.weight + value


class _InfiniteGradient(torch.autograd.Function):
    """The identity whose gradient is infinite: a finite loss with non-finite gradients."""

    @staticmethod
    def forward(_ctx: Any, value: Tensor) -> Tensor:
        return value.clone()

    @staticmethod
    def backward(_ctx: Any, grad: Tensor) -> Tensor:
        return grad * math.inf


class _OverflowingGradient(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, numeric: Tensor, _categorical: Tensor) -> Tensor:
        gradient: Tensor = _InfiniteGradient.apply(self.weight)  # type: ignore[no-untyped-call]
        return numeric[:, 0] * 0 + gradient


def _fit_once(network: nn.Module, optimizer: torch.optim.Optimizer, plan: BatchPlan) -> None:
    settings = training_settings(
        {**{name: TRAINING[name] for name in TRAINING_PARAMETERS}, "batch_size": plan.effective},
        early_stopping=False,
    )
    train_network(
        network,
        optimizer,
        neural_data(100, seed=0),
        None,
        settings,
        layout=LAYOUT,
        device=torch.device("cpu"),
        seeds=derived_seeds(0),
        epochs=1,
        name="test",
        plan=plan,
        objective=None,
    )


def test_a_non_finite_loss_after_successful_micro_batches_gives_no_step() -> None:
    network = _FailsOnCall(call=3)  # micro-batches 1 and 2 of batch 1 are fine
    before = network.weight.detach().clone()
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.1)
    with pytest.raises(ModelError, match="epoch 1, batch 1, micro-batch 3"):
        _fit_once(network, optimizer, BatchPlan(64, 20))
    assert torch.equal(network.weight.detach(), before)
    assert not optimizer.state  # the failing effective batch took no step


def test_finite_losses_with_non_finite_gradients_give_no_step() -> None:
    network = _OverflowingGradient()
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.1)
    with pytest.raises(ModelError, match="non-finite gradients in epoch 1, batch 1"):
        _fit_once(network, optimizer, BatchPlan(64, 32))
    assert torch.equal(network.weight.detach(), torch.ones(1))
    assert not optimizer.state  # no step was taken


def test_earlier_updates_are_kept_when_a_later_batch_fails() -> None:
    network = _FailsOnCall(call=2)  # batch 1 (one micro-batch of 64) is fine, batch 2 fails
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.1)
    settings = training_settings(
        {name: TRAINING[name] for name in TRAINING_PARAMETERS}, early_stopping=False
    )
    with pytest.raises(ModelError, match="batch 2"):
        train_network(
            network,
            optimizer,
            neural_data(100, seed=0),
            None,
            settings,
            layout=LAYOUT,
            device=torch.device("cpu"),
            seeds=derived_seeds(0),
            epochs=1,
            name="test",
            plan=BatchPlan(64, 64),
            objective=None,
        )
    assert len(optimizer.state) == 1  # batch 1's update was applied and is not rolled back


# --- Early stopping on the shared V1 objective ----------------------------------------------------


def _fit_on_v1(patience: int, max_epochs: int, *, v1_weight: float = 1.0) -> Any:
    valid = neural_data(240, seed=9, with_rows=True)
    valid = replace(valid, weights=np.where(valid.labels == 1, v1_weight, 1.0))
    objective = v1_objective(valid)
    torch.manual_seed(0)
    network = _Linear()
    optimizer = torch.optim.AdamW(network.parameters(), lr=5e-2)
    settings = training_settings(
        {**TRAINING, "patience": patience, "max_epochs": max_epochs}, early_stopping=True
    )
    result = train_network(
        network,
        optimizer,
        neural_data(200, seed=2),
        valid,
        settings,
        layout=LAYOUT,
        device=torch.device("cpu"),
        seeds=derived_seeds(0),
        epochs=None,
        name="test",
        plan=BatchPlan(64, 64),
        objective=objective,
    )
    return result, objective


def test_early_stopping_uses_the_shared_unweighted_v1_objective() -> None:
    result, objective = _fit_on_v1(patience=1, max_epochs=8)
    numerators = [record.v1_numerator for record in result.history]
    assert all(type(n) is int and 0 <= n <= objective.frauds for n in numerators)
    assert result.best_numerator == max(numerators)
    assert result.epochs_kept == numerators.index(max(numerators)) + 1  # the earliest best
    assert result.best_objective == result.best_numerator / objective.frauds
    history = result.v1_history
    assert history is not None
    assert history.unit == "epoch"
    assert history.best_checkpoint == result.epochs_kept
    assert [c.numerator for c in history.checkpoints] == numerators


def test_the_v1_sample_weights_do_not_enter_the_objective() -> None:
    plain, _ = _fit_on_v1(patience=1, max_epochs=6)
    weighted, _ = _fit_on_v1(patience=1, max_epochs=6, v1_weight=37.0)
    assert [r.v1_numerator for r in plain.history] == [r.v1_numerator for r in weighted.history]
    assert plain.epochs_kept == weighted.epochs_kept


def test_a_neural_fit_stops_patience_plus_one_epochs_after_the_best() -> None:
    result, _ = _fit_on_v1(patience=1, max_epochs=40)
    if result.stop == "patience":
        assert result.epochs_run == result.epochs_kept + 2  # patience + 1 epochs without gain
    else:
        assert result.epochs_run == 40


def test_the_shared_history_distinguishes_rounds_from_epochs() -> None:
    objective = v1_objective(neural_data(240, seed=9, with_rows=True))
    recorder = objective.recorder()
    for _ in range(3):  # equal numerators: only the first observation improves
        recorder.observe(np.zeros(objective.rows), objective.labels)
    rounds = recorder.history(patience=2, maximum=60)  # a tree stops at best + patience
    assert (rounds.stopped_at, rounds.stopping_reason) == (3, "early_stopping")
    assert rounds.unit == "round"
    epochs = recorder.history(patience=1, maximum=60, unit="epoch")  # best + patience + 1
    assert (epochs.stopped_at, epochs.stopping_reason) == (3, "early_stopping")
    assert epochs.unit == "epoch"
    with pytest.raises(ModelError, match="neither"):
        recorder.history(patience=2, maximum=60, unit="epoch")
    with pytest.raises(ModelError, match="training unit"):
        recorder.history(patience=1, maximum=60, unit="batch")
