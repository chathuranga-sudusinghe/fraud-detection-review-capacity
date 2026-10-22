"""Shared training loop and fitted-model base of the two neural models of DR-011 (ResNet-style MLP
and FT-Transformer), under the device, precision and determinism policy of DR-013 §4.

* **Inputs:** the neural input representation of DR-012 (Accepted), already applied by the caller:
  float32 numeric inputs (transformed values and missingness indicators, all finite) and int64
  category indices, described by an ``InputLayout`` (names, vocabulary sizes, the index of
  ``"__MISSING__"`` per categorical input and the category-dropout probability). This module never
  sees the canonical frame and fits nothing on the data besides the network.
* **Settings (DR-013 §4):** ``configure_torch`` applies the precision and determinism settings of
  ``config/models.yaml`` at the start of every fit and reload, and their effective values are
  recorded. ``seed_everything`` seeds Python, NumPy and PyTorch with the run seed immediately
  before the network is built, so weight initialisation and dropout masks follow the seed. The
  batch order and category dropout each draw from their own CPU generator, whose seeds are
  derived from the run seed (``derived_seeds``) and recorded. No ``DataLoader`` is used: batches
  are taken by index in the main process, so there are no worker processes to seed.
* **Training:** AdamW with a constant learning rate (REF-087, Section 4: no learning-rate
  schedule), its remaining constants set explicitly at the values of the reference implementation
  (REF-103) and the single-tensor implementation on every device; bf16 autocast for the forward
  pass, float32 parameters and optimiser state; weighted binary cross-entropy from logits,
  computed outside autocast in float32 from float32 logits, labels and weights, and reduced as the
  sum of the weighted row losses divided by the rows of the effective batch (``weighted_bce``).
  Every epoch visits every Train row once, in a new random order (one shuffle); the order is cut
  into effective batches (``BatchPlan.effective`` rows, DR-015 §7) and the last batch keeps the
  remaining rows (as REF-103). An effective batch is processed in micro-batches of at most
  ``BatchPlan.micro`` rows: gradients are cleared once, every micro-batch adds the gradients of
  its weighted loss sum divided by the same row count m of the effective batch (float32
  accumulation, no further division), and AdamW steps once per effective batch. A micro-batch
  loss or the accumulated gradients that are not finite stop the run before that effective batch
  updates the network. Category dropout is drawn once per effective batch, so its masks do not
  depend on the micro-batch plan. The network is built and trained with float32 as PyTorch's
  default dtype, whatever the process-wide default (``float32_default``); the dtypes of the
  training logits and of the optimiser state are observed and recorded.
* **Early stopping on V1 (DR-015 §3):** after every epoch, V1 is scored through the float32
  prediction path and the shared V1 objective (unweighted Recall@K under daily top-K review,
  ``v1_objective``) is observed on the full ordered V1 vector. An epoch improves only if its
  integer numerator is strictly greater than the best so far, so the earliest epoch wins a tie;
  training stops after ``patience + 1`` consecutive epochs without improvement (REF-087, Section
  4; the ``ProgressTracker`` of REF-103's ``libzero`` 0.0.3.dev7) or at ``max_epochs``, and the
  parameters and normalisation statistics of the best epoch are restored. With a fixed epoch count
  and no validation, exactly that many epochs are trained.
* **Prediction (DR-013 §4):** float32 with autocast disabled, in evaluation mode and without
  gradients, for V1 early stopping and every exported score. A non-finite logit stops the run.
  The float32 logits are turned into probabilities in float64, so scores keep the resolution of
  the logits: a float32 sigmoid rounds every logit above about 17 to exactly 1.
* **Persistence:** the network state (tensors only, read back with ``weights_only=True``) and a
  manifest with the layout, the hyperparameters, the training record (with the batch plan) and the
  state file's SHA-256; reloading checks the manifest against the request, requires the precision
  settings now applied to equal those of the fit and the training record to be complete and
  consistent (an early-stopped fit is replayed through the stopping rule on its integer
  numerators), and loads the state strictly.

Hyperparameter values, patience and the epoch limit are fixed in the training rules
(protocol design item 3), never defaulted here: every name below must be given explicitly. The
V1 objective and the batch plan are passed to ``fit``.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from torch.nn import functional

from review_capacity.config import DEVICES, SEED_LIMIT, Device, ModelSettings
from review_capacity.errors import (
    DataValidationError,
    EnvironmentMismatchError,
    FileAccessError,
    ModelError,
    SpecificationError,
)
from review_capacity.files import write_atomically, write_json
from review_capacity.logging_setup import ProgressLogger, log_step
from review_capacity.models.base import (
    Keys,
    Labels,
    Scores,
    Weights,
    check_fit_arguments,
    checked_scores,
)
from review_capacity.models.device import configure_torch, seed_everything
from review_capacity.models.registry import Configuration, Family
from review_capacity.models.run_record import jsonable
from review_capacity.models.v1_objective import (
    ObjectiveRecorder,
    TrainingHistory,
    V1Objective,
    check_identity,
)

logger = logging.getLogger(__name__)

FORMAT_VERSION = 3  # 3: V1 objective history and the batch plan; 2: per-epoch train_seconds
STATE_FILE = "model.pt"
MANIFEST_FILE = "manifest.json"

# Hyperparameters of the training loop, required for every fit; with V1 early stopping the three
# early-stopping names are required as well. A fixed-epoch fit accepts them (so that the selected
# hyperparameters can be reused as they are) and records them, but does not use them.
TRAINING_PARAMETERS = ("lr", "weight_decay", "batch_size", "eval_batch_size")
EARLY_STOPPING_PARAMETERS = ("max_epochs", "patience")

# AdamW constants of the reference implementation (REF-103 passes only the learning rate and
# weight decay, so PyTorch's defaults apply), set explicitly; the single-tensor implementation
# (neither ``foreach`` nor ``fused``) on every device.
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8

# Independent random streams derived from the run seed (``derived_seeds``).
SEED_STREAMS = ("batch_order", "category_dropout")

# The sections of a fit's training record and their JSON types, all required when a saved model is
# reloaded.
TRAINING_RECORD_SECTIONS: dict[str, type] = {
    "precision": dict,
    "observed_dtypes": dict,
    "device": dict,
    "seeds": dict,
    "data_loading": str,
    "optimizer": dict,
    "batches_per_epoch": int,
    "batching": dict,
    "schedule": dict,
    "history": list,
}
SCHEDULE_KEYS = (
    "mode",
    "objective",
    "patience",
    "max_epochs",
    "stop",
    "epochs_run",
    "epochs_kept",
    "best_numerator",
    "best_objective",
)
HISTORY_KEYS = ("epoch", "train_loss", "v1_numerator", "v1_objective", "train_seconds", "seconds")
BATCHING_KEYS = (
    "effective_batch",
    "micro_batch",
    "final_batch_rows",
    "loss_reduction",
    "gradient_accumulation",
    "evaluation_batch",
    "evaluation_final_batch",
)
LOSS_REDUCTION = "sum(w * loss) over the effective batch / its rows m, in every micro-batch"
GRADIENT_ACCUMULATION = "float32; one optimiser step per effective batch"
EVALUATION_FINAL_BATCH = "kept"


# --- Inputs ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class InputLayout:
    """The neural inputs of DR-012: names in model order, the vocabulary size of each categorical
    input, the index of its reserved ``"__MISSING__"`` label, and the probability with which
    category dropout replaces a categorical value of a Train row by that index during training."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    cardinalities: tuple[int, ...]
    missing_indices: tuple[int, ...]
    category_dropout: float

    def __post_init__(self) -> None:
        names = (*self.numeric, *self.categorical)
        if not names:
            raise SpecificationError("A neural input layout needs at least one input")
        if not all(isinstance(name, str) and name for name in names):
            raise SpecificationError("Neural input names must be non-empty strings")
        if len(set(names)) != len(names):
            raise SpecificationError("Neural input names must be unique")
        if not len(self.cardinalities) == len(self.missing_indices) == len(self.categorical):
            raise SpecificationError(
                "One vocabulary size and one missing-label index per categorical input"
            )
        for name, size, missing in zip(
            self.categorical, self.cardinalities, self.missing_indices, strict=True
        ):
            if type(size) is not int or size <= 0:
                raise SpecificationError(f"{name}: vocabulary size must be a positive integer")
            if type(missing) is not int or not 0 <= missing < size:
                raise SpecificationError(f"{name}: missing-label index outside its vocabulary")
        dropout = self.category_dropout
        if not (isinstance(dropout, float) and math.isfinite(dropout) and 0.0 <= dropout < 1.0):
            raise SpecificationError(f"Category dropout must be a float in [0, 1): {dropout!r}")

    def to_json(self) -> dict[str, Any]:
        return {
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "cardinalities": list(self.cardinalities),
            "missing_indices": list(self.missing_indices),
            "category_dropout": self.category_dropout,
        }


@dataclass(frozen=True)
class NeuralInputs:
    """Network inputs of one partition: float32 numeric inputs and int64 category indices, one row
    per transaction, in the column order of the layout. Rows are matched by position."""

    numeric: npt.NDArray[np.float32]
    categorical: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        for name, array, dtype in (
            ("Numeric inputs", self.numeric, np.float32),
            ("Category indices", self.categorical, np.int64),
        ):
            if not isinstance(array, np.ndarray) or array.dtype != dtype or array.ndim != 2:
                kind = dtype.__name__
                raise DataValidationError(f"{name} must be a two-dimensional {kind} array")
        if len(self.numeric) != len(self.categorical):
            raise DataValidationError(
                f"{len(self.numeric)} numeric rows, but {len(self.categorical)} categorical rows"
            )
        if len(self.numeric) == 0:
            raise DataValidationError("A model partition must not be empty")

    @property
    def rows(self) -> int:
        return len(self.numeric)

    def check(self, layout: InputLayout) -> None:
        """Check that the inputs follow ``layout``: column counts, finite numeric values and
        category indices inside each vocabulary."""
        shape = (self.numeric.shape[1], self.categorical.shape[1])
        if shape != (len(layout.numeric), len(layout.categorical)):
            raise DataValidationError(
                f"Inputs have {shape} numeric and categorical columns; the layout declares "
                f"{(len(layout.numeric), len(layout.categorical))}"
            )
        if not np.isfinite(self.numeric).all():
            raise DataValidationError("Numeric neural inputs must be finite")
        sizes = np.asarray(layout.cardinalities, dtype=np.int64)
        if ((self.categorical < 0) | (self.categorical >= sizes)).any():
            raise DataValidationError("Category indices outside their vocabularies")


@dataclass(frozen=True)
class NeuralData:
    """Network inputs, labels (1 = fraud, 0 = legitimate) and sample weights of one partition.

    The weights are the protocol's explicit sample weights (``models.base.sample_weights``); V1
    weights are an explicit input set by the training rules, never recomputed from V1 labels.
    ``ids`` and ``times`` (each row's ``TransactionID`` and ``TransactionDT``) are metadata, given
    together, for the V1 objective's queue: they are never network inputs.
    """

    inputs: NeuralInputs
    labels: Labels
    weights: Weights
    ids: Keys | None = None
    times: Keys | None = None

    def __post_init__(self) -> None:
        labels, weights, rows = self.labels, self.weights, self.inputs.rows
        if (self.ids is None) != (self.times is None):
            raise DataValidationError("Row identifiers and times must be given together")
        for name, keys in (("identifiers", self.ids), ("times", self.times)):
            if keys is not None and (
                not isinstance(keys, np.ndarray) or keys.dtype != np.int64 or keys.shape != (rows,)
            ):
                raise DataValidationError(f"Row {name} must be one int64 value per row")
        if (
            not isinstance(labels, np.ndarray)
            or labels.dtype != np.int8
            or labels.shape != (rows,)
            or not np.isin(labels, (0, 1)).all()
        ):
            raise DataValidationError(f"Labels must be {rows} int8 values of 0 or 1")
        if not isinstance(weights, np.ndarray) or weights.dtype != np.float64:
            raise DataValidationError("Sample weights must be a float64 NumPy array")
        if weights.shape != (rows,):
            raise DataValidationError(f"{rows} input rows, but weights {weights.shape}")
        if not (np.isfinite(weights).all() and (weights > 0).all()):
            raise DataValidationError("Sample weights must be finite and positive")


# --- Hyperparameters ------------------------------------------------------------------------------


def positive_integer(value: object, name: str) -> int:
    """``value``, if it is a positive ``int`` (not a Boolean)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SpecificationError(f"{name} must be a positive integer, not {value!r}")
    return value


def real_number(
    value: object, name: str, *, low: float = 0.0, high: float = math.inf, open_low: bool = False
) -> float:
    """``value`` as a float, if it is an int or float (not a Boolean) in [``low``, ``high``), or
    in (``low``, ``high``) with ``open_low``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SpecificationError(f"{name} must be a number, not {value!r}")
    number = float(value)
    above = number > low if open_low else number >= low
    if not (math.isfinite(number) and above and number < high):
        interval = f"{'(' if open_low else '['}{low}, {high})"
        raise SpecificationError(f"{name} must be finite and in {interval}, not {value!r}")
    return number


@dataclass(frozen=True)
class TrainingSettings:
    """The training-loop hyperparameters of one fit. The early-stopping values are ``None`` for a
    fixed-epoch fit."""

    lr: float
    weight_decay: float
    batch_size: int
    eval_batch_size: int
    max_epochs: int | None
    patience: int | None


def training_settings(values: Mapping[str, Any], *, early_stopping: bool) -> TrainingSettings:
    """The training-loop hyperparameters in ``values`` (only those names), checked."""
    missing = [name for name in TRAINING_PARAMETERS if name not in values]
    if early_stopping:
        missing += [name for name in EARLY_STOPPING_PARAMETERS if name not in values]
    if missing:
        raise SpecificationError(f"Hyperparameters must be set explicitly: {missing}")
    max_epochs: int | None = None
    patience: int | None = None
    if early_stopping:
        max_epochs = positive_integer(values["max_epochs"], "max_epochs")
        patience = values["patience"]
        if type(patience) is not int or patience < 0:
            raise SpecificationError(f"patience must be a non-negative integer, not {patience!r}")
    return TrainingSettings(
        lr=real_number(values["lr"], "lr", open_low=True),
        weight_decay=real_number(values["weight_decay"], "weight_decay"),
        batch_size=positive_integer(values["batch_size"], "batch_size"),
        eval_batch_size=positive_integer(values["eval_batch_size"], "eval_batch_size"),
        max_epochs=max_epochs,
        patience=patience,
    )


@dataclass(frozen=True)
class BatchPlan:
    """How an effective batch is processed (DR-015 §7): ``effective`` Train rows per optimiser
    update, in micro-batches of at most ``micro`` rows. The plan comes from the preflight (it is
    fixed before the search and never adapted during a fit); this module only checks and uses it.
    """

    effective: int
    micro: int

    def __post_init__(self) -> None:
        for name, value in (("effective", self.effective), ("micro", self.micro)):
            if type(value) is not int or value < 1:
                raise SpecificationError(f"The {name} batch must be a positive integer")
        if self.micro > self.effective:
            raise SpecificationError(
                f"The micro-batch ({self.micro}) cannot exceed the effective batch "
                f"({self.effective})"
            )

    def micro_sizes(self, rows: int) -> list[int]:
        """The sizes of the micro-batches of an effective batch of ``rows`` rows: full micro-batches
        and a shorter last one."""
        sizes = [self.micro] * (rows // self.micro)
        return [*sizes, rows % self.micro] if rows % self.micro else sizes


def batching_record(plan: BatchPlan, rows: int, evaluation_batch: int) -> dict[str, Any]:
    """The batch plan of a fit as recorded: the effective and micro-batch sizes, the rows of the
    final batch of an epoch, the loss reduction and accumulation, and the evaluation batch."""
    return {
        "effective_batch": plan.effective,
        "micro_batch": plan.micro,
        "final_batch_rows": rows % plan.effective or plan.effective,
        "loss_reduction": LOSS_REDUCTION,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "evaluation_batch": evaluation_batch,
        "evaluation_final_batch": EVALUATION_FINAL_BATCH,
    }


# --- Seeds, device and loop components -----------------------------------------------------------


def _finite_number(value: object) -> bool:
    """Whether ``value`` is a finite int or float (not a Boolean). An integer too large for a
    float (possible in JSON) is not a finite number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _v1_entry_valid(entry: Mapping[str, Any], *, early: bool) -> bool:
    """The V1 fields of a saved history entry: an integer numerator and a finite objective with
    early stopping, none without."""
    numerator, value = entry["v1_numerator"], entry["v1_objective"]
    if not early:
        return numerator is None and value is None
    return type(numerator) is int and numerator >= 0 and _finite_number(value)


def derived_seeds(seed: int) -> dict[str, int]:
    """One seed per random stream of the loop, derived from the run seed with NumPy's
    ``SeedSequence`` (stream ``i`` is the child with spawn key ``(i,)``), so the streams are
    independent of each other and of the global generators."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < SEED_LIMIT:
        raise ValueError(f"Seed must be an integer from 0 to {SEED_LIMIT - 1}, not {seed!r}")
    return {
        name: int(np.random.SeedSequence(seed, spawn_key=(index,)).generate_state(1)[0])
        for index, name in enumerate(SEED_STREAMS, start=1)
    }


@contextmanager
def float32_default() -> Iterator[None]:
    """Make float32 PyTorch's default dtype inside the block and restore the caller's default
    afterwards, so tensors created without an explicit dtype (initial weights, optimiser
    counters) are float32 whatever the process-wide default is (DR-013 §4)."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def generator(seed: int) -> torch.Generator:
    """A CPU generator seeded with ``seed``."""
    return torch.Generator(device="cpu").manual_seed(seed)


def torch_device(device: Device) -> torch.device:
    """The PyTorch device of ``device``. The GPU is the first visible CUDA device (DR-013 §1); if
    CUDA is unavailable the run stops, with no fallback to the CPU."""
    if device == "cpu":
        return torch.device("cpu")
    if device != "gpu":
        raise ValueError(f"Unsupported device {device!r}; expected one of {DEVICES}")
    if not torch.cuda.is_available():
        raise EnvironmentMismatchError("GPU training requested, but CUDA is not available")
    return torch.device("cuda", 0)


def drop_categories(
    categorical: Tensor, missing: Tensor, probability: float, rng: torch.Generator
) -> Tensor:
    """Category dropout of DR-012: each value replaced by its input's ``"__MISSING__"`` index
    (``missing``, one per column) with ``probability``, independently per value."""
    if probability == 0.0:
        return categorical
    draws = torch.rand(categorical.shape, generator=rng, dtype=torch.float32)
    dropped = draws < probability
    return torch.where(dropped, missing.expand_as(categorical), categorical)


def weighted_bce(
    logits: Tensor, labels: Tensor, weights: Tensor, rows: int | None = None
) -> Tensor:
    """Weighted binary cross-entropy from logits in float32: the sum of ``w_i * l_i`` divided by
    ``rows``, the row count of the whole effective batch (DR-015 §7; by default, the rows given),
    so a mini-batch loss is an unbiased estimate of the mean weighted Train loss and the
    micro-batches of one effective batch add up to its loss. Labels and weights must be float32;
    the logits are cast to float32."""
    if labels.dtype != torch.float32 or weights.dtype != torch.float32:
        raise TypeError("Labels and weights of the loss must be float32")
    losses = functional.binary_cross_entropy_with_logits(logits.float(), labels, reduction="none")
    return (losses * weights).sum() / (logits.shape[0] if rows is None else rows)


def predict_logits(
    network: nn.Module,
    inputs: NeuralInputs,
    *,
    device: torch.device,
    batch_size: int,
    task: str = "Prediction",
) -> npt.NDArray[np.float32]:
    """float32 logits of ``inputs``: evaluation mode, no gradients, autocast disabled. A
    non-finite logit stops the run: its sigmoid would otherwise pass as a score of 0 or 1."""
    network.eval()
    outputs: list[Tensor] = []
    progress = ProgressLogger(logger, f"{task} batches", math.ceil(inputs.rows / batch_size))
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, inputs.rows, batch_size):
            stop = min(start + batch_size, inputs.rows)
            numeric = torch.from_numpy(inputs.numeric[start:stop].copy()).to(device)
            categorical = torch.from_numpy(inputs.categorical[start:stop].copy()).to(device)
            logits: Tensor = network(numeric, categorical)
            if logits.dtype != torch.float32:
                raise ModelError(f"{task}: {logits.dtype} logits, not float32")
            if not bool(torch.isfinite(logits).all()):
                raise ModelError(f"{task}: non-finite logits in rows {start} to {stop - 1}")
            outputs.append(logits.cpu())
            progress.advance()
    return torch.cat(outputs).numpy()


def probabilities(logits: npt.NDArray[np.float32]) -> Scores:
    """Fraud scores from float32 logits, computed in float64."""
    return torch.sigmoid(torch.from_numpy(logits.astype(np.float64))).numpy()


def epoch_batches(rows: int, batch_size: int, rng: torch.Generator) -> list[npt.NDArray[np.int64]]:
    """The row indices of each effective batch of one epoch: every row exactly once, in a new
    random order drawn from ``rng`` (one shuffle); the last batch keeps the remaining rows (as
    REF-103)."""
    order = torch.randperm(rows, generator=rng).numpy()
    return [order[start : start + batch_size] for start in range(0, rows, batch_size)]


class EarlyStopping:
    """Best V1 numerator and stopping state: the first epoch always improves; a later epoch
    improves only if its integer numerator is strictly greater than the best; training stops once
    more than ``patience`` consecutive epochs have not improved (``patience + 1`` of them)."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best: int | None = None
        self.best_epoch = 0
        self.bad_epochs = 0

    def update(self, epoch: int, numerator: int) -> bool:
        """Record the numerator of ``epoch``; return whether it is the new best."""
        better = self.best is None or numerator > self.best
        if better:
            self.best, self.best_epoch, self.bad_epochs = numerator, epoch, 0
        else:
            self.bad_epochs += 1
        return better

    @property
    def stop(self) -> bool:
        return self.bad_epochs > self.patience


@dataclass(frozen=True)
class EpochRecord:
    """One epoch: its mean weighted Train loss, its V1 objective numerator and value (None without
    early stopping), the seconds of its training batches alone (up to the synchronising read of
    the loss) and of the whole epoch including V1 scoring."""

    epoch: int
    train_loss: float
    v1_numerator: int | None
    v1_objective: float | None
    train_seconds: float
    seconds: float


@dataclass(frozen=True)
class TrainingResult:
    """What one fit did: the epochs kept and run, why it stopped and the per-epoch history."""

    epochs_kept: int
    epochs_run: int
    stop: str  # "patience", "max_epochs" or "fixed"
    best_numerator: int | None
    best_objective: float | None
    history: tuple[EpochRecord, ...]
    v1_history: TrainingHistory | None
    training_logits_dtypes: tuple[str, ...]
    optimizer_state_dtypes: dict[str, tuple[str, ...]]


def parameter_group_record(
    network: nn.Module, groups: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """The optimiser's parameter groups by parameter name, with each group's weight decay. Every
    parameter of ``network`` must be in exactly one group."""
    names = {id(parameter): name for name, parameter in network.named_parameters()}
    seen: set[int] = set()
    record: list[dict[str, Any]] = []
    for group in groups:
        ids = [id(parameter) for parameter in group["params"]]
        if any(key not in names or key in seen for key in ids) or len(set(ids)) != len(ids):
            raise ValueError("Parameter groups must hold each network parameter exactly once")
        seen.update(ids)
        record.append(
            {
                "weight_decay": float(group["weight_decay"]),
                "parameters": [names[key] for key in ids],
            }
        )
    if seen != set(names):
        raise ValueError("Parameter groups must hold every network parameter")
    return record


def optimizer_state_dtypes(optimizer: torch.optim.Optimizer) -> dict[str, tuple[str, ...]]:
    """The data types of the optimiser's state tensors, by state name (e.g. AdamW's moments
    ``exp_avg`` and ``exp_avg_sq``), as observed after training."""
    found: dict[str, set[str]] = {}
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                found.setdefault(str(key), set()).add(str(value.dtype))
    return {key: tuple(sorted(dtypes)) for key, dtypes in sorted(found.items())}


def _state_copy(network: nn.Module) -> dict[str, Tensor]:
    """A CPU copy of the network's parameters and buffers."""
    state = network.state_dict()
    return {name: value.detach().to("cpu", copy=True) for name, value in state.items()}


def _gradients_finite(network: nn.Module) -> bool:
    """Whether every accumulated gradient is finite (one synchronising read)."""
    checks = [
        torch.isfinite(parameter.grad).all()
        for parameter in network.parameters()
        if parameter.grad is not None
    ]
    return bool(torch.stack(checks).all()) if checks else True


def train_network(
    network: nn.Module,
    optimizer: torch.optim.Optimizer,
    train: NeuralData,
    valid: NeuralData | None,
    settings: TrainingSettings,
    *,
    layout: InputLayout,
    device: torch.device,
    seeds: Mapping[str, int],
    epochs: int | None,
    name: str,
    plan: BatchPlan,
    objective: V1Objective | None,
) -> TrainingResult:
    """Train ``network`` (already on ``device``) with V1 early stopping on ``objective``, or for
    exactly ``epochs`` epochs without validation, and leave it in evaluation mode with the kept
    parameters. ``plan`` splits each effective batch into micro-batches (DR-015 §7)."""
    tracker: EarlyStopping | None = None
    recorder: ObjectiveRecorder | None = None
    if valid is not None:
        if objective is None:
            raise ValueError("V1 early stopping needs the V1 objective")
        if settings.patience is None or settings.max_epochs is None:
            raise ValueError("V1 early stopping needs its patience and epoch limit")
        recorder = objective.recorder()
        tracker = EarlyStopping(settings.patience)
        total_epochs = settings.max_epochs
    else:
        if epochs is None:
            raise ValueError("A fixed-epoch fit needs its epoch count")
        total_epochs = epochs
    rows = train.inputs.rows
    order_rng = generator(seeds["batch_order"])
    dropout_rng = generator(seeds["category_dropout"])
    missing = torch.tensor(layout.missing_indices, dtype=torch.int64)
    labels = train.labels.astype(np.float32)
    weights = train.weights.astype(np.float32)
    batches = math.ceil(rows / plan.effective)
    epoch_progress = ProgressLogger(logger, f"{name} epochs", total_epochs)
    history: list[EpochRecord] = []
    best_state: dict[str, Tensor] | None = None
    logits_dtypes: set[str] = set()
    stop = "fixed"
    for epoch in range(1, total_epochs + 1):
        start_time = time.perf_counter()
        network.train()
        batch_progress = ProgressLogger(logger, f"{name} epoch {epoch} batches", batches)
        loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        for number, index in enumerate(epoch_batches(rows, plan.effective, order_rng), start=1):
            m = len(index)
            # drawn once per effective batch, so the masks do not depend on the micro-batch plan
            categorical = drop_categories(
                torch.from_numpy(train.inputs.categorical[index]),
                missing,
                layout.category_dropout,
                dropout_rng,
            )
            optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), dtype=torch.float32, device=device)
            start = 0
            for micro_number, size in enumerate(plan.micro_sizes(m), start=1):
                part = index[start : start + size]
                numeric_part = torch.from_numpy(train.inputs.numeric[part]).to(device)
                categorical_part = categorical[start : start + size].to(device)
                label_part = torch.from_numpy(labels[part]).to(device)
                weight_part = torch.from_numpy(weights[part]).to(device)
                start += size
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits: Tensor = network(numeric_part, categorical_part)
                logits_dtypes.add(str(logits.dtype))
                loss = weighted_bce(logits, label_part, weight_part, rows=m)
                if not bool(torch.isfinite(loss)):  # before any update from this batch
                    raise ModelError(
                        f"{name}: non-finite Train loss in epoch {epoch}, batch {number}, "
                        f"micro-batch {micro_number}"
                    )
                loss.backward()  # type: ignore[no-untyped-call]
                batch_loss += loss.detach()
            if not _gradients_finite(network):
                raise ModelError(
                    f"{name}: non-finite gradients in epoch {epoch}, batch {number}; "
                    "the optimiser did not step"
                )
            optimizer.step()
            loss_sum += batch_loss.double() * m
            batch_progress.advance()
        train_loss = float(loss_sum.item()) / rows  # waits for the epoch's GPU work
        train_seconds = time.perf_counter() - start_time
        if not math.isfinite(train_loss):
            raise ModelError(f"{name}: Train loss is not finite in epoch {epoch}")
        numerator: int | None = None
        value: float | None = None
        if valid is not None and tracker is not None and recorder is not None:
            logits_v1 = predict_logits(
                network,
                valid.inputs,
                device=device,
                batch_size=settings.eval_batch_size,
                task=f"{name} epoch {epoch} V1",
            )
            scores = checked_scores(probabilities(logits_v1), valid.inputs.rows, name)
            recall = recorder.observe(scores, valid.labels)
            if recall is None:
                raise ModelError(f"{name}: the V1 objective did not accept the V1 evaluation")
            numerator, value = recall.numerator, recall.value
            if tracker.update(epoch, numerator):
                best_state = _state_copy(network)
        seconds = time.perf_counter() - start_time
        history.append(EpochRecord(epoch, train_loss, numerator, value, train_seconds, seconds))
        if tracker is None:
            logger.info("%s epoch %d: Train loss %.6f (%.1f s)", name, epoch, train_loss, seconds)
        elif objective is not None:
            logger.info(
                "%s epoch %d: Train loss %.6f, V1 objective %.6f (best %.6f, epoch %d) (%.1f s)",
                name,
                epoch,
                train_loss,
                value,
                (tracker.best or 0) / objective.frauds,
                tracker.best_epoch,
                seconds,
            )
        epoch_progress.advance()
        if tracker is not None and tracker.stop:
            stop = "patience"
            break
    else:
        stop = "fixed" if tracker is None else "max_epochs"
    v1_history: TrainingHistory | None = None
    if tracker is not None and recorder is not None:
        if best_state is None or settings.patience is None or settings.max_epochs is None:
            raise ModelError(f"{name}: no epoch was kept")
        network.load_state_dict(best_state)
        kept = tracker.best_epoch
        v1_history = recorder.history(
            patience=settings.patience, maximum=settings.max_epochs, unit="epoch"
        )
    else:
        kept = len(history)
    network.eval()
    best_numerator = None if tracker is None else tracker.best
    return TrainingResult(
        epochs_kept=kept,
        epochs_run=len(history),
        stop=stop,
        best_numerator=best_numerator,
        best_objective=None
        if best_numerator is None or objective is None
        else best_numerator / objective.frauds,
        history=tuple(history),
        v1_history=v1_history,
        training_logits_dtypes=tuple(sorted(logits_dtypes)),
        optimizer_state_dtypes=optimizer_state_dtypes(optimizer),
    )


# --- The fitted-model base ------------------------------------------------------------------------


class NeuralModel(ABC):
    """A neural configuration of DR-011 on the inputs of DR-012: fitted once, then scores any
    partition.

    The interface follows ``models.base.ScoringModel`` with the network inputs in place of the
    canonical frame. A subclass names its model and its architecture hyperparameters, checks
    them and builds its network; the loop, the settings, prediction, persistence and the record
    are shared. ``MIN_BATCH_ROWS`` is the smallest training batch the network accepts (2 with
    batch normalisation, which cannot normalise a single row); ``ACCUMULATES`` says whether an
    effective batch may be split into micro-batches (False with batch normalisation, whose
    statistics are those of the whole batch: the micro-batch is then the effective batch).
    """

    MODEL: ClassVar[str]
    ARCHITECTURE: ClassVar[tuple[str, ...]]
    MIN_BATCH_ROWS: ClassVar[int] = 1
    ACCUMULATES: ClassVar[bool] = True

    def __init__(
        self,
        configuration: Configuration,
        seed: int,
        *,
        settings: ModelSettings,
        layout: InputLayout,
        device: Device,
    ) -> None:
        if device not in DEVICES:
            raise ValueError(f"Unsupported device {device!r}; expected one of {DEVICES}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"Seed must be an integer, not {seed!r}")
        if not 0 <= seed < SEED_LIMIT:
            raise ValueError(f"Seed {seed} outside 0 to {SEED_LIMIT - 1}")
        if configuration.family is not Family.NEURAL or configuration.model.name != self.MODEL:
            raise ValueError(f"{configuration.name} is not a {self.MODEL} configuration")
        self.configuration = configuration
        self.seed = seed
        self.settings = settings
        self.layout = layout
        self.device = device
        self._network: nn.Module | None = None
        self._hyperparameters: dict[str, Any] = {}
        self._architecture: dict[str, Any] = {}
        self._early_stopping = True
        self._training: dict[str, Any] = {}
        self._epochs = 0

    @abstractmethod
    def check_architecture(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """The architecture hyperparameters in ``values`` (only ``ARCHITECTURE`` names, of which
        some may be absent), checked and complete; ``SpecificationError`` otherwise."""

    @abstractmethod
    def build_network(self, architecture: Mapping[str, Any]) -> nn.Module:
        """A new network for the checked ``architecture``, on the CPU, in float32."""

    def parameter_groups(self, network: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
        """The optimiser's parameter groups: by default one group, with weight decay applied to
        every parameter (the ResNet-style MLP, as REF-103). The groups are recorded by parameter
        name and checked again when a saved model is reloaded."""
        return [{"params": list(network.parameters()), "weight_decay": weight_decay}]

    def _split(
        self, hyperparameters: Mapping[str, Any], *, early_stopping: bool
    ) -> tuple[TrainingSettings, dict[str, Any]]:
        if not isinstance(hyperparameters, Mapping) or not all(
            isinstance(name, str) for name in hyperparameters
        ):
            raise SpecificationError("Hyperparameters must be a mapping from names to values")
        known = {*TRAINING_PARAMETERS, *EARLY_STOPPING_PARAMETERS, *self.ARCHITECTURE}
        unknown = sorted(set(hyperparameters) - known)
        if unknown:
            raise SpecificationError(f"Unknown {self.MODEL} hyperparameters: {unknown}")
        names = self.ARCHITECTURE
        loop = {key: value for key, value in hyperparameters.items() if key not in names}
        training = training_settings(loop, early_stopping=early_stopping)
        architecture = self.check_architecture(
            {key: value for key, value in hyperparameters.items() if key in names}
        )
        return training, architecture

    def _network_or_raise(self) -> nn.Module:
        if self._network is None:
            raise ModelError(f"{self.configuration.name} is not fitted")
        return self._network

    def _check_batches(self, rows: int, plan: BatchPlan) -> None:
        """The last micro-batch of every effective batch must hold the rows the network needs."""
        last_batch = rows % plan.effective or plan.effective
        for size in {plan.effective, last_batch}:
            last = plan.micro_sizes(size)[-1]
            if last < self.MIN_BATCH_ROWS:
                raise SpecificationError(
                    f"{rows} Train rows in effective batches of {plan.effective} and micro-batches "
                    f"of {plan.micro} leave a last micro-batch of {last} rows; {self.MODEL} needs "
                    f"at least {self.MIN_BATCH_ROWS} rows per batch"
                )

    def _batch_plan(self, plan: BatchPlan | None, training: TrainingSettings) -> BatchPlan:
        """The plan of a fit: the given one, or the whole effective batch at once; it must be
        for the effective batch of the hyperparameters, and a model that cannot accumulate
        processes each effective batch whole."""
        used = plan or BatchPlan(training.batch_size, training.batch_size)
        if used.effective != training.batch_size:
            raise SpecificationError(
                f"The batch plan's effective batch ({used.effective}) is not the batch_size "
                f"hyperparameter ({training.batch_size})"
            )
        if not self.ACCUMULATES and used.micro != used.effective:
            raise SpecificationError(
                f"{self.MODEL} cannot accumulate gradients (its batch normalisation needs the "
                f"whole batch): the micro-batch must be the effective batch ({used.effective}), "
                f"not {used.micro}"
            )
        return used

    def fit(
        self,
        train: NeuralData,
        valid: NeuralData | None,
        hyperparameters: Mapping[str, Any],
        *,
        iterations: int | None = None,
        objective: V1Objective | None = None,
        plan: BatchPlan | None = None,
    ) -> None:
        """Fit on ``train`` with early stopping on ``valid`` (V1) and the shared V1 ``objective``,
        or, with ``valid=None``, for exactly ``iterations`` epochs without validation. ``plan``
        is the micro-batch plan of the effective batch (default: each effective batch at once);
        ``valid`` carries the V1 rows' identifiers and times, which must be those of
        ``objective``."""
        check_fit_arguments(valid, iterations)
        early = valid is not None
        training, architecture = self._split(hyperparameters, early_stopping=early)
        batch_plan = self._batch_plan(plan, training)
        train.inputs.check(self.layout)
        if valid is not None:
            valid.inputs.check(self.layout)
            if objective is None:
                raise SpecificationError("V1 early stopping needs the V1 objective")
            objective.check_rows(valid.ids, valid.times, valid.labels)
        self._check_batches(train.inputs.rows, batch_plan)
        device = torch_device(self.device)
        precision = configure_torch(self.settings.neural)
        seeds = derived_seeds(self.seed)
        name = self.configuration.name
        rows = f"{train.inputs.rows} Train rows"
        if valid is not None:
            rows += f", early stopping on {valid.inputs.rows} V1 rows"
        try:
            with log_step(logger, f"Fit {name} on {self.device} ({rows})"), float32_default():
                seed_everything(self.seed)
                network = self._new_network(architecture).to(device)
                groups = self.parameter_groups(network, training.weight_decay)
                group_record = parameter_group_record(network, groups)
                optimizer = torch.optim.AdamW(
                    groups,
                    lr=training.lr,
                    betas=ADAMW_BETAS,
                    eps=ADAMW_EPS,
                    weight_decay=training.weight_decay,
                    amsgrad=False,
                    foreach=False,
                    fused=False,
                )
                result = train_network(
                    network,
                    optimizer,
                    train,
                    valid,
                    training,
                    layout=self.layout,
                    device=device,
                    seeds=seeds,
                    epochs=iterations,
                    name=name,
                    plan=batch_plan,
                    objective=objective,
                )
        except torch.cuda.OutOfMemoryError as error:
            raise ModelError(f"{name}: out of GPU memory during training") from error
        self._check_float32(network)
        self._network = network
        self._hyperparameters = dict(hyperparameters)
        self._architecture = architecture
        self._early_stopping = early
        self._epochs = result.epochs_kept
        self._training = {
            "precision": self._precision_record(precision, device),
            "observed_dtypes": {
                "training_logits": list(result.training_logits_dtypes),
                "optimizer_state": {
                    key: list(dtypes) for key, dtypes in result.optimizer_state_dtypes.items()
                },
            },
            "device": {
                "type": device.type,
                "index": device.index,
                "name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            },
            "seeds": {"run": self.seed, **seeds},
            "data_loading": "index batches in the main process; no DataLoader, no workers",
            "optimizer": {
                "class": "AdamW",
                **optimizer.defaults,
                "parameter_groups": group_record,
            },
            "batches_per_epoch": math.ceil(train.inputs.rows / batch_plan.effective),
            "batching": batching_record(batch_plan, train.inputs.rows, training.eval_batch_size),
            "schedule": {
                "mode": "early_stopping" if early else "fixed",
                "objective": None if objective is None or not early else objective.identity,
                "patience": training.patience,
                "max_epochs": training.max_epochs,
                "stop": result.stop,
                "epochs_run": result.epochs_run,
                "epochs_kept": result.epochs_kept,
                "best_numerator": result.best_numerator,
                "best_objective": result.best_objective,
            },
            "history": [
                {
                    "epoch": record.epoch,
                    "train_loss": record.train_loss,
                    "v1_numerator": record.v1_numerator,
                    "v1_objective": record.v1_objective,
                    "train_seconds": record.train_seconds,
                    "seconds": record.seconds,
                }
                for record in result.history
            ],
        }
        logger.info(
            "%s fitted: %d of %d epochs kept (%s)",
            name,
            result.epochs_kept,
            result.epochs_run,
            result.stop,
        )

    def _precision_record(
        self, effective: Mapping[str, Any], device: torch.device
    ) -> dict[str, Any]:
        """The precision settings of a fit: the effective values from ``configure_torch`` and the
        fixed precision path of this module."""
        return {
            **effective,
            "training_dtype": self.settings.neural.training_dtype,
            "autocast": f"{device.type}, bfloat16, forward pass only",
            "loss": "binary_cross_entropy_with_logits, float32, "
            "sum(w * loss) / rows of the effective batch",
            "parameter_and_optimizer_dtype": "float32",
            "prediction_dtype": self.settings.neural.prediction_dtype,
        }

    def _new_network(self, architecture: Mapping[str, Any]) -> nn.Module:
        """A new network built in float32 (DR-013 §4), whatever PyTorch's process-wide default
        dtype is; initial values are therefore drawn in float32, as in any float32 run with the
        same seed."""
        with float32_default():
            network = self.build_network(architecture)
        self._check_float32(network)
        return network

    @staticmethod
    def _check_float32(network: nn.Module) -> None:
        dtypes = {tensor.dtype for tensor in network.parameters()}
        if dtypes != {torch.float32}:
            found = sorted(map(str, dtypes))
            raise ModelError(f"Network parameters must be float32, found {found}")

    def predict(self, inputs: NeuralInputs) -> Scores:
        """Fraud scores of ``inputs``, one per row, from the float32 prediction path."""
        network = self._network_or_raise()
        inputs.check(self.layout)
        batch_size = positive_integer(self._hyperparameters["eval_batch_size"], "eval_batch_size")
        name = self.configuration.name
        try:
            with log_step(logger, f"Score {inputs.rows} rows with {name} on {self.device}"):
                logits = predict_logits(
                    network,
                    inputs,
                    device=torch_device(self.device),
                    batch_size=batch_size,
                    task=f"{name} scoring",
                )
        except torch.cuda.OutOfMemoryError as error:
            raise ModelError(f"{name}: out of GPU memory") from error
        return checked_scores(probabilities(logits), inputs.rows, name)

    def save(self, directory: Path) -> None:
        """Save the network state and its manifest (written last, with the state's SHA-256)."""
        network = self._network_or_raise()
        buffer = io.BytesIO()
        torch.save(_state_copy(network), buffer)
        content = buffer.getvalue()
        write_atomically(directory / STATE_FILE, lambda target: target.write_bytes(content))
        write_json(
            directory / MANIFEST_FILE,
            {
                "format_version": FORMAT_VERSION,
                "library": "torch",
                "library_version": torch.__version__,
                "model": self.MODEL,
                "configuration": self.configuration.name,
                "seed": self.seed,
                "device": self.device,
                "layout": self.layout.to_json(),
                "hyperparameters": jsonable(self._hyperparameters),
                "early_stopping": self._early_stopping,
                "architecture": jsonable(self._architecture),
                "epochs": self._epochs,
                "training": jsonable(self._training),
                "model_sha256": hashlib.sha256(content).hexdigest(),
            },
        )

    @classmethod
    def load(
        cls,
        directory: Path,
        configuration: Configuration,
        seed: int,
        *,
        settings: ModelSettings,
        layout: InputLayout,
        device: Device,
    ) -> Self:
        """A model saved by ``save``; its manifest must match the request, and the saved state
        must match the rebuilt network exactly. ``device`` is the training device, on which the
        model also predicts."""
        try:
            manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
            content = (directory / STATE_FILE).read_bytes()
        except OSError as error:
            raise FileAccessError(f"{directory}: saved model cannot be read") from error
        except ValueError as error:
            raise ModelError(f"{directory}: saved model manifest is not valid JSON") from error
        model = cls(configuration, seed, settings=settings, layout=layout, device=device)
        model._restore(manifest, content, directory)
        return model

    def _restore(self, manifest: object, content: bytes, directory: Path) -> None:
        if not isinstance(manifest, dict):
            raise ModelError("Saved model manifest must be a JSON object")
        try:
            version = manifest["format_version"]
            identity = tuple(
                manifest[key]
                for key in (
                    "library",
                    "library_version",
                    "model",
                    "configuration",
                    "seed",
                    "device",
                    "layout",
                )
            )
            hyperparameters = manifest["hyperparameters"]
            early = manifest["early_stopping"]
            saved_architecture = manifest["architecture"]
            epochs = manifest["epochs"]
            training = manifest["training"]
            model_sha256 = manifest["model_sha256"]
        except KeyError as error:
            raise ModelError(f"Saved model manifest is incomplete: missing {error}") from error
        if type(version) is not int or version != FORMAT_VERSION:
            raise ModelError(f"Unsupported model format version {version!r}")
        if type(manifest["seed"]) is not int:
            raise ModelError(f"Saved model seed must be an integer, not {manifest['seed']!r}")
        expected = (
            "torch",
            torch.__version__,
            self.MODEL,
            self.configuration.name,
            self.seed,
            self.device,
            self.layout.to_json(),
        )
        if identity != expected:
            raise ModelError(
                "Saved model does not match the requested library version, model, "
                "configuration, seed, training device or input layout"
            )
        if type(early) is not bool or type(epochs) is not int or epochs <= 0:
            raise ModelError("Saved schedule or epoch count is malformed")
        if not isinstance(training, dict) or not isinstance(hyperparameters, dict):
            raise ModelError("Saved hyperparameters or training record malformed")
        try:
            loop_settings, architecture = self._split(hyperparameters, early_stopping=early)
        except SpecificationError as error:
            raise ModelError(f"Saved hyperparameters are invalid: {error}") from error
        if jsonable(architecture) != saved_architecture:
            raise ModelError("Saved architecture does not match the saved hyperparameters")
        if hashlib.sha256(content).hexdigest() != model_sha256:
            raise ModelError(f"{directory}: saved model and manifest do not match")
        device = torch_device(self.device)
        precision = self._precision_record(configure_torch(self.settings.neural), device)
        self._check_training_record(
            training,
            settings=loop_settings,
            early=early,
            epochs=epochs,
            precision=precision,
            device=device,
        )
        network = self._new_network(architecture)
        try:
            state = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
            network.load_state_dict(state, strict=True)
        except (RuntimeError, ValueError, TypeError) as error:
            message = f"{directory}: saved network state cannot be loaded: {error}"
            raise ModelError(message) from error
        groups = self.parameter_groups(network, loop_settings.weight_decay)
        if training["optimizer"].get("parameter_groups") != parameter_group_record(network, groups):
            raise ModelError("Saved optimiser parameter groups differ from those of the network")
        self._network = network.to(device).eval()
        self._hyperparameters = hyperparameters
        self._architecture = architecture
        self._early_stopping = early
        self._epochs = epochs
        self._training = training

    def _check_training_record(
        self,
        training: dict[str, Any],
        *,
        settings: TrainingSettings,
        early: bool,
        epochs: int,
        precision: dict[str, Any],
        device: torch.device,
    ) -> None:
        """Check that a saved training record is complete and consistent with the saved model.

        Every section has its type; the precision settings now applied equal those of the fit (so
        the reloaded model predicts as it did) and the observed dtypes are those of DR-013 §4; the
        seeds are those derived from the run seed; the device type, optimiser settings and
        schedule agree with the saved hyperparameters; every history entry is complete and
        finite, with epochs 1, 2, ... in order. A fixed-epoch fit keeps all its epochs and has no
        V1 scores; an early-stopped fit is replayed through ``EarlyStopping``, which must give the
        saved kept epoch, best score and stop reason, and end where the saved history ends.
        """
        missing = sorted(set(TRAINING_RECORD_SECTIONS) - set(training))
        if missing:
            raise ModelError(f"Saved training record is incomplete: missing {missing}")
        for name, kind in TRAINING_RECORD_SECTIONS.items():
            if type(training[name]) is not kind:
                raise ModelError(f"Saved training record section {name!r} is malformed")
        if training["precision"] != jsonable(precision):
            raise ModelError(
                "Saved precision settings differ from those now applied: "
                f"{training['precision']} != {jsonable(precision)}"
            )
        if training["seeds"] != {"run": self.seed, **derived_seeds(self.seed)}:
            raise ModelError("Saved seeds are not those derived from the run seed")
        if training["device"].get("type") != device.type:
            raise ModelError("Saved training device type differs from the requested device")
        if type(training["batches_per_epoch"]) is not int or training["batches_per_epoch"] <= 0:
            raise ModelError("Saved batch count per epoch must be positive")
        self._check_batching(training["batching"], settings)
        self._check_optimizer_record(training["optimizer"], settings)
        self._check_observed_dtypes(training["observed_dtypes"])
        schedule, history = training["schedule"], training["history"]
        self._check_schedule(schedule, history, settings=settings, early=early, epochs=epochs)

    @staticmethod
    def _check_optimizer_record(optimizer: dict[str, Any], settings: TrainingSettings) -> None:
        expected = {
            "class": "AdamW",
            "lr": settings.lr,
            "weight_decay": settings.weight_decay,
            "betas": list(ADAMW_BETAS),
            "eps": ADAMW_EPS,
            "amsgrad": False,
            "foreach": False,
            "fused": False,
        }
        if {key: optimizer.get(key) for key in expected} != expected:
            raise ModelError("Saved optimiser settings differ from the saved hyperparameters")

    def _check_observed_dtypes(self, observed: dict[str, Any]) -> None:
        """The dtypes observed in training must be those of DR-013 §4: training logits in the
        configured training dtype, optimiser state in float32."""
        logits = observed.get("training_logits")
        state = observed.get("optimizer_state")
        expected_logits = [f"torch.{self.settings.neural.training_dtype}"]
        if (
            logits != expected_logits
            or not isinstance(state, dict)
            or not state
            or any(dtypes != ["torch.float32"] for dtypes in state.values())
        ):
            raise ModelError(
                f"Saved observed dtypes are not those of DR-013 §4: {observed} (expected training "
                f"logits {expected_logits} and float32 optimiser state)"
            )

    def _check_batching(self, batching: dict[str, Any], settings: TrainingSettings) -> None:
        """The saved batch plan has its fields and agrees with the saved hyperparameters: the
        effective and evaluation batches are theirs, the micro-batch lies within the effective
        batch (and equals it for a model that cannot accumulate), the final batch is a real
        batch, and the reduction and accumulation are those of DR-015 §7."""
        if set(batching) != set(BATCHING_KEYS):
            raise ModelError("Saved batch plan is malformed")
        counts = [batching[key] for key in ("effective_batch", "micro_batch", "final_batch_rows")]
        counts.append(batching["evaluation_batch"])
        if any(type(value) is not int or value < 1 for value in counts):
            raise ModelError("Saved batch plan sizes must be positive integers")
        effective, micro, final = counts[:3]
        consistent = (
            effective == settings.batch_size
            and batching["evaluation_batch"] == settings.eval_batch_size
            and micro <= effective
            and final <= effective
            and (self.ACCUMULATES or micro == effective)
            and batching["loss_reduction"] == LOSS_REDUCTION
            and batching["gradient_accumulation"] == GRADIENT_ACCUMULATION
            and batching["evaluation_final_batch"] == EVALUATION_FINAL_BATCH
        )
        if not consistent:
            raise ModelError("Saved batch plan does not match the saved hyperparameters")

    @staticmethod
    def _check_schedule(
        schedule: dict[str, Any],
        history: list[Any],
        *,
        settings: TrainingSettings,
        early: bool,
        epochs: int,
    ) -> None:
        missing = [key for key in SCHEDULE_KEYS if key not in schedule]
        if missing:
            raise ModelError(f"Saved schedule is malformed: missing {missing}")
        for number, entry in enumerate(history, start=1):
            if (
                not isinstance(entry, dict)
                or any(key not in entry for key in HISTORY_KEYS)
                or not _finite_number(entry["train_loss"])
                or not (_finite_number(entry["seconds"]) and entry["seconds"] >= 0)
                or not (
                    _finite_number(entry["train_seconds"])
                    and 0 <= entry["train_seconds"] <= entry["seconds"]
                )
                or not _v1_entry_valid(entry, early=early)
            ):
                raise ModelError(f"Saved history entry {number} is malformed")
        if [entry["epoch"] for entry in history] != list(range(1, len(history) + 1)):
            raise ModelError("Saved history must list epochs 1, 2, ... in order")
        recorded = (schedule["patience"], schedule["max_epochs"])
        if recorded != (settings.patience, settings.max_epochs):
            raise ModelError("Saved schedule does not match the saved hyperparameters")
        if not (
            schedule["mode"] == ("early_stopping" if early else "fixed")
            and schedule["epochs_run"] == len(history)
            and schedule["epochs_kept"] == epochs
        ):
            raise ModelError("Saved schedule, history and kept epoch count are inconsistent")
        if not early:
            fixed = (
                epochs == len(history)
                and schedule["stop"] == "fixed"
                and schedule["objective"] is None
                and schedule["best_numerator"] is None
                and schedule["best_objective"] is None
            )
            if not fixed:
                raise ModelError("Saved schedule, history and kept epoch count are inconsistent")
            return
        if settings.patience is None or settings.max_epochs is None:
            raise ModelError("Saved early-stopping settings are incomplete")
        if not 1 <= len(history) <= settings.max_epochs:  # the epoch limit is a hard cap
            raise ModelError("Saved schedule does not follow the early-stopping rule")
        frauds = check_identity(schedule["objective"])["frauds"]
        if any(
            entry["v1_numerator"] > frauds
            or entry["v1_objective"] != entry["v1_numerator"] / frauds
            for entry in history
        ):
            raise ModelError("Saved V1 objective values are inconsistent with their numerators")
        tracker = EarlyStopping(settings.patience)
        stop = "max_epochs"
        for entry in history:
            if tracker.stop:  # the history continues after the rule stopped training
                stop = "continued"
                break
            tracker.update(entry["epoch"], entry["v1_numerator"])
            if tracker.stop:
                stop = "patience"
        replayed = (
            stop != "continued"
            and (stop == "patience" or len(history) == settings.max_epochs)
            and schedule["stop"] == stop
            and tracker.best_epoch == epochs
            and schedule["best_numerator"] == tracker.best
            and tracker.best is not None
            and schedule["best_objective"] == tracker.best / frauds
        )
        if not replayed:
            raise ModelError("Saved schedule does not follow the early-stopping rule")

    @property
    def parameters(self) -> dict[str, Any]:
        """The requested hyperparameters, the resolved architecture and training record (precision
        and determinism settings, device, seeds, optimiser, schedule and per-epoch history), and
        the prediction path."""
        network = self._network_or_raise()
        return {
            "requested": self._hyperparameters,
            "resolved": {
                "architecture": self._architecture,
                "parameter_count": sum(tensor.numel() for tensor in network.parameters()),
                **self._training,
            },
            "prediction": {
                "dtype": "float32",
                "autocast": "disabled",
                "mode": "eval, inference_mode",
                "probabilities": "sigmoid of float32 logits, computed in float64",
                "eval_batch_size": self._hyperparameters["eval_batch_size"],
            },
        }

    @property
    def fitted_iterations(self) -> int:
        """The epochs the fitted network keeps."""
        self._network_or_raise()
        return self._epochs
