"""The micro-batch preflight of the two neural models (DR-015 §7; Issue #151, Part 2).

Before the search, a synthetic, data-free preflight on the project GPU fixes the micro-batch of
each of a model's ten candidates:

1. It uses the production implementation (the neural loop of ``models.neural_training``) with
   full-size synthetic inputs and the vocabularies of the shared preprocessing artifact, strict
   determinism and the precision and attention backend of DR-013 §4.
2. **Training phase:** ``PHASE_EPOCHS`` epochs (by model) of **two** effective batches (2,048
   rows) in one fit, so that every batch after the first runs its forward and backward passes
   with the AdamW state initialised by the first update, as in a real epoch. The repetition also
   makes the phase last long enough for the memory sampler to observe it: the fastest candidate
   of each model lasts about 1 s (ResNet-style MLP, 25 epochs) or 2 s (FT-Transformer, 5 epochs),
   ten or more sampling intervals against the three required (Issue #173).
3. **Evaluation phase:** the same fit with early stopping on a synthetic validation set of one full
   evaluation batch (4,096 or 16 rows) and the shared V1 objective, so that the float32 evaluation
   runs after every epoch inside the training loop, with the optimiser state alive, exactly where
   V1 evaluation runs in a search. Its reading includes the training batches and is therefore
   conservative.
4. It tries the declared micro-batches (1,024, 512, 256, 128, 64) in that order and records the
   largest whose training phase completes with at least ``preflight_free_margin_mib`` of device
   memory free at the sampled peak; the evaluation phase is then measured at that micro-batch and
   must pass the same margin. A phase also fails when the sampler took fewer than
   ``MIN_SAMPLES`` samples (a phase shorter than the sampling interval would be read from its
   starting sample alone) or reported a problem.
5. The ResNet-style MLP is tried at the effective batch only, because its batch normalisation
   needs the whole batch (``ACCUMULATES`` is False); FT-Transformer accumulates gradients.
6. A candidate that passes at no listed size, or whose evaluation phase fails, stops the search
   for an issue (``ModelError``); no candidate is dropped or replaced.
7. The synthetic data come from the preflight's own generator, and the Python, NumPy and PyTorch
   random states are restored after every attempt, also when it fails, once the attempt's model
   is released (restoring a CUDA generator state needs free device memory).
8. Every model is released, and the allocator cache cleared, after each attempt, so that the next
   attempt does not start from the previous attempt's allocations.
9. On the GPU, PyTorch's allocator is capped at the device total less the margin
   (``cap_allocator``): under WSL the device otherwise spills into system memory and an attempt
   that does not fit runs on for minutes instead of failing.

The recorded plan (``preflight.json``) is used for the unweighted and balanced configurations and
for every refit, and is never changed after V1 results are seen. ``read_preflight`` checks a
recorded plan in full (identity, settings, candidates, attempt order and every phase's arithmetic)
before it is used.

GPU memory is the whole-device memory sampled by ``nvidia-smi`` (``probe.diagnostics``): a sampled
maximum, not a true peak, including memory held by other processes. It does not cover host memory,
allocator fragmentation over a whole search, or the final short training and evaluation batches
(smaller than the measured full batches). The memory reading is a ``Meter``, so that the procedure
is tested without a GPU.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import random
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from review_capacity.config import SECONDS_PER_DAY, Device, ModelSettings
from review_capacity.errors import (
    DataValidationError,
    FileAccessError,
    ModelError,
    SpecificationError,
)
from review_capacity.files import write_json
from review_capacity.logging_setup import ProgressLogger, log_step
from review_capacity.models.ft_transformer_model import FTTransformerModel
from review_capacity.models.neural_training import (
    BatchPlan,
    InputLayout,
    NeuralData,
    NeuralInputs,
    NeuralModel,
)
from review_capacity.models.registry import configuration
from review_capacity.models.resnet_model import ResNetModel
from review_capacity.models.run_record import jsonable
from review_capacity.models.v1_objective import V1Objective
from review_capacity.training.declaration import TrainingDeclaration
from review_capacity.training.sampler import Candidate

logger = logging.getLogger(__name__)

RECORD = "neural_preflight"
FORMAT_VERSION = 2
FILE = "preflight.json"
RUN_TYPE = "neural_preflight"
MODELS: dict[str, type[NeuralModel]] = {"resnet": ResNetModel, "ft_transformer": FTTransformerModel}
CONFIGURATIONS = {"resnet": "RES-u", "ft_transformer": "FTT-u"}  # weighting plays no part here
SYNTHETIC_FRAUD_SHARE = 0.2
SYNTHETIC_SEED = 0
SYNTHETIC_DAYS = 3  # relative days of the synthetic validation rows
EFFECTIVE_BATCHES = 2  # effective batches of a measured epoch
# Epochs of a measured phase, by model: long enough to be sampled, state alive throughout. The
# ResNet-style MLP's fastest candidate takes about 0.04 s per epoch, so 25 epochs last about 1 s
# (5 epochs lasted 0.2 s on one run and left two readings, Issue #173); the FT-Transformer's fastest
# takes about 0.5 s per epoch, so 5 epochs already last more than 2 s, and more would only make its
# larger candidates (13 s per epoch) run for minutes.
PHASE_EPOCHS: Mapping[str, int] = {"resnet": 25, "ft_transformer": 5}
MIN_SAMPLES = 3  # memory samples a phase needs for its reading to count
SAMPLE_INTERVAL_MS = 100  # nvidia-smi sampling interval of the preflight
RECORD_KEYS = frozenset(
    {
        "record",
        "format_version",
        "identity",
        "margin_mib",
        "micro_batch_candidates",
        "effective_batches_per_epoch",
        "epochs_measured",
        "models",
    }
)
ENTRY_KEYS = frozenset({"effective_batch", "evaluation_batch", "candidates"})
ATTEMPT_KEYS = frozenset({"micro_batch", "training"})
PHASE_KEYS = frozenset(
    {
        "completed",
        "error",
        "baseline_used_mib",
        "sampled_max_used_mib",
        "total_mib",
        "free_at_peak_mib",
        "samples",
        "problems",
        "passes",
    }
)
ITEM_KEYS = frozenset(
    {"trial", "candidate", "candidate_sha256", "micro_batch", "attempts", "evaluation"}
)


# --- Memory readings ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reading:
    """The whole-device GPU memory sampled during one phase."""

    total_mib: int | None
    baseline_used_mib: int | None
    sampled_max_used_mib: int | None
    problems: tuple[str, ...] = ()
    samples: int | None = None

    @property
    def free_mib(self) -> int | None:
        """Memory free at the sampled maximum: the device total less that maximum."""
        if self.total_mib is None or self.sampled_max_used_mib is None:
            return None
        return self.total_mib - self.sampled_max_used_mib


class Watch:
    """The reading of one watched phase; set when the phase ends."""

    reading: Reading | None = None


class Meter(Protocol):
    """Samples device memory around a phase: ``with meter.watch() as watch: ...``, after which
    ``watch.reading`` holds the reading."""

    def watch(self) -> Any: ...


class GpuMeter:
    """The ``nvidia-smi`` sampler of the GPU acceptance probe, started and stopped per phase."""

    @contextmanager
    def watch(self) -> Iterator[Watch]:
        from review_capacity.probe.diagnostics import NVIDIA_SMI, GpuMemorySampler

        sampler = GpuMemorySampler((*NVIDIA_SMI[:-1], f"--loop-ms={SAMPLE_INTERVAL_MS}"))
        sampler.start()
        watch = Watch()
        try:
            yield watch
        finally:
            sampler.stop()
            snapshot = sampler.snapshot()
            watch.reading = Reading(
                snapshot["total_mib"],
                snapshot["baseline_used_mib"],
                snapshot["sampled_max_used_mib"],
                tuple(snapshot["problems"]),
                snapshot["samples"],
            )


# --- Inputs, hyperparameters and random state --------------------------------------------------


def synthetic_data(layout: InputLayout, rows: int, rng: np.random.Generator) -> NeuralData:
    """Full-size synthetic inputs of the layout: standard-normal numeric values, category indices
    over each vocabulary, labels with both classes, unit weights."""
    numeric = rng.standard_normal((rows, len(layout.numeric))).astype(np.float32)
    categorical = np.stack(
        [rng.integers(0, size, rows) for size in layout.cardinalities], axis=1
    ).astype(np.int64)
    labels = (rng.random(rows) < SYNTHETIC_FRAUD_SHARE).astype(np.int8)
    labels[:2] = (0, 1)
    return NeuralData(NeuralInputs(numeric, categorical), labels, np.ones(rows, dtype=np.float64))


def synthetic_validation(layout: InputLayout, rows: int, rng: np.random.Generator) -> NeuralData:
    """Synthetic validation rows with identifiers and times on ``SYNTHETIC_DAYS`` relative days
    (metadata of the V1 objective, never inputs)."""
    data = synthetic_data(layout, rows, rng)
    ids = np.arange(1, rows + 1, dtype=np.int64)
    days = ids * SYNTHETIC_DAYS // (rows + 1) + 1
    times = (days * SECONDS_PER_DAY + 1).astype(np.int64)
    return NeuralData(data.inputs, data.labels, data.weights, ids, times)


def neural_hyperparameters(
    declaration: TrainingDeclaration, model: str, candidate: Candidate
) -> dict[str, Any]:
    """The hyperparameters of one fit: the candidate's searched values, the declared fixed values
    the model takes as hyperparameters (the rest of the fixed values are built into the model and
    checked against DR-015 §7 by the declaration tests), the effective and evaluation batch."""
    model_class = MODELS[model]
    fixed = {
        name: value
        for name, value in declaration.space(model).fixed.items()
        if name in model_class.ARCHITECTURE
    }
    clash = sorted(fixed.keys() & candidate.keys())
    if clash:
        raise SpecificationError(f"{model}: a candidate sets fixed declaration values {clash}")
    return {
        **fixed,
        **candidate,
        "batch_size": declaration.neural.effective_batch,
        "eval_batch_size": declaration.neural.evaluation_batch[model],
    }


def fit_hyperparameters(
    declaration: TrainingDeclaration, model: str, candidate: Candidate
) -> dict[str, Any]:
    """The hyperparameters of a search trial of a neural model: those of the preflight
    (``neural_hyperparameters``) with the declared early stopping (DR-015 §2), so that the fit
    uses exactly the candidate, fixed values and batches the preflight measured."""
    return {
        **neural_hyperparameters(declaration, model, candidate),
        "max_epochs": declaration.neural_stopping.maximum,
        "patience": declaration.neural_stopping.patience,
    }


RandomState = tuple[Any, Any, Any, Any]


def save_random_state(device: Device) -> RandomState:
    """The Python, NumPy and PyTorch (and CUDA) random states: a fit seeds them."""
    import torch

    cuda_states = torch.cuda.get_rng_state_all() if device == "gpu" else None  # initialises CUDA
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), cuda_states


def restore_random_state(saved: RandomState) -> None:
    """Put the saved random states back. Called after the attempt's model is released: with the
    device full, restoring a CUDA generator state fails."""
    import torch

    python_state, numpy_state, torch_state, cuda_states = saved
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def cap_allocator(margin_mib: int) -> None:
    """Cap PyTorch's allocator at the device total less the margin. Under WSL the GPU spills into
    system memory instead of failing, so an attempt that does not fit would run on for minutes
    (in the observed case up to 52 GiB allocated on an 8 GiB device); with the cap it fails at
    once with an out-of-memory error, which the preflight records as a failed attempt. The sampled
    whole-device reading still decides every phase: the cap does not count the CUDA context or
    other processes."""
    import torch

    total_mib = torch.cuda.get_device_properties(0).total_memory // (1 << 20)
    torch.cuda.set_per_process_memory_fraction((total_mib - margin_mib) / total_mib)


def _loaded_torch() -> Any:
    """PyTorch if this process has imported it, else None."""
    return sys.modules.get("torch")


def release_gpu_memory() -> None:
    """Free what a finished configuration leaves on the GPU, so that the next one, possibly of
    another library, can use it: collect the objects that are no longer referenced (a model, its
    optimiser and activations) and hand PyTorch's unused cached blocks back to the device. PyTorch
    keeps freed blocks in its cache, where CatBoost's separate allocator cannot use them. Nothing
    happens when PyTorch is not loaded or no GPU is present."""
    gc.collect()
    torch = _loaded_torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _release(device: Device) -> None:
    gc.collect()
    if device == "gpu":
        import torch

        torch.cuda.empty_cache()


# --- The procedure ------------------------------------------------------------------------------


def _phase(reading: Reading, error: str | None, margin_mib: int) -> dict[str, Any]:
    problems = list(reading.problems)
    if reading.samples is not None and reading.samples < MIN_SAMPLES:
        problems.append(f"{reading.samples} memory samples; at least {MIN_SAMPLES} are required")
    free = reading.free_mib
    completed = error is None
    passes = completed and not problems and free is not None and free >= margin_mib
    return {
        "completed": completed,
        "error": error,
        "baseline_used_mib": reading.baseline_used_mib,
        "sampled_max_used_mib": reading.sampled_max_used_mib,
        "total_mib": reading.total_mib,
        "free_at_peak_mib": free,
        "samples": reading.samples,
        "problems": problems,
        "passes": passes,
    }


def _out_of_memory(error: ModelError) -> bool:
    import torch

    return isinstance(error.__cause__, torch.cuda.OutOfMemoryError)


@dataclass(frozen=True)
class Setting:
    """What the preflight of every candidate shares."""

    declaration: TrainingDeclaration
    settings: ModelSettings
    layout: InputLayout
    device: Device
    meter: Meter
    seed: int
    train: NeuralData
    validation: dict[str, NeuralData]


def _attempt(
    setting: Setting,
    model: str,
    hyperparameters: Mapping[str, Any],
    micro: int,
    *,
    evaluate: bool,
) -> dict[str, Any]:
    """One measured fit at ``micro`` rows per micro-batch: the training phase (two effective
    batches, no validation) or, with ``evaluate``, the evaluation phase (the same epoch followed by
    the float32 evaluation of the synthetic validation set inside the loop). The model does not
    outlive the call."""
    declared = setting.declaration.neural
    error: str | None = None
    instance: NeuralModel | None = None
    saved = save_random_state(setting.device)
    try:
        with setting.meter.watch() as watch:
            try:
                instance = MODELS[model](
                    configuration(CONFIGURATIONS[model]),
                    setting.seed,
                    settings=setting.settings,
                    layout=setting.layout,
                    device=setting.device,
                )
                plan = BatchPlan(declared.effective_batch, micro)
                if evaluate:
                    valid = setting.validation[model]
                    assert valid.ids is not None
                    assert valid.times is not None
                    objective = V1Objective.from_rows(
                        valid.ids, valid.times, valid.labels, setting.declaration.v1_daily_k
                    )
                    instance.fit(
                        setting.train,
                        valid,
                        {
                            **hyperparameters,
                            "max_epochs": PHASE_EPOCHS[model],
                            "patience": PHASE_EPOCHS[model],
                        },
                        objective=objective,
                        plan=plan,
                    )
                else:
                    instance.fit(
                        setting.train,
                        None,
                        hyperparameters,
                        iterations=PHASE_EPOCHS[model],
                        plan=plan,
                    )
            except ModelError as failure:
                if not _out_of_memory(failure):
                    raise
                error = f"out of GPU memory during {'evaluation' if evaluate else 'training'}"
    finally:
        # the model and the exception's frames are dropped, the cache is cleared, and only then
        # are the random states restored (restoring a CUDA generator state needs free memory)
        instance = None
        _release(setting.device)
        restore_random_state(saved)
    assert watch.reading is not None
    return _phase(watch.reading, error, declared.preflight_free_margin_mib)


def candidate_sha256(candidate: Candidate) -> str:
    """SHA-256 of a candidate as canonical JSON (sorted names, no spaces)."""
    text = json.dumps(jsonable(dict(candidate)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def micro_batches(model: str, effective: int, declared: Sequence[int]) -> tuple[int, ...]:
    """The micro-batches tried for ``model``, in order: the declared sizes up to the effective
    batch; for a model that cannot accumulate, the effective batch only."""
    sizes = tuple(size for size in declared if size <= effective)
    return sizes if MODELS[model].ACCUMULATES else (effective,)


def _outcome(phase: Mapping[str, Any]) -> str:
    """Why a phase passed or failed: its error, its sampler problems, or the memory margin."""
    if phase["passes"]:
        return "passes"
    reasons = [phase["error"]] if phase["error"] else []
    reasons += phase["problems"]
    return f"fails ({'; '.join(reasons) or 'memory margin'})"


def preflight_model(
    setting: Setting,
    model: str,
    candidates: Sequence[Candidate],
    progress: ProgressLogger | None = None,
) -> dict[str, Any]:
    """The preflight of one model's candidates; raises ``ModelError`` for a candidate that passes
    at no listed micro-batch or whose evaluation phase fails the memory margin."""
    declared = setting.declaration.neural
    entries: list[dict[str, Any]] = []
    for number, candidate in enumerate(candidates, start=1):
        hyperparameters = neural_hyperparameters(setting.declaration, model, candidate)
        attempts: list[dict[str, Any]] = []
        chosen: int | None = None
        with log_step(logger, f"Preflight {model}, candidate {number} of {len(candidates)}"):
            tried = micro_batches(model, declared.effective_batch, declared.micro_batch_candidates)
            for micro in tried:
                phase = _attempt(setting, model, hyperparameters, micro, evaluate=False)
                _release(setting.device)
                attempts.append({"micro_batch": micro, "training": phase})
                logger.info(
                    "%s candidate %d, micro-batch %d: %s, free at the sampled peak %s MiB",
                    model,
                    number,
                    micro,
                    _outcome(phase),
                    phase["free_at_peak_mib"],
                )
                if phase["passes"]:
                    chosen = micro
                    break
            if chosen is None:
                raise ModelError(
                    f"{model} candidate {number} passes at no listed micro-batch "
                    f"{list(declared.micro_batch_candidates)}: the search stops for an issue"
                )
            evaluation = _attempt(setting, model, hyperparameters, chosen, evaluate=True)
            _release(setting.device)
            logger.info(
                "%s candidate %d, evaluation at micro-batch %d: %s, free at the peak %s MiB",
                model,
                number,
                chosen,
                _outcome(evaluation),
                evaluation["free_at_peak_mib"],
            )
            if not evaluation["passes"]:
                raise ModelError(
                    f"{model} candidate {number}: float32 evaluation (batch "
                    f"{declared.evaluation_batch[model]}) inside the training loop fails the "
                    "memory margin: the search stops for an issue"
                )
        entries.append(
            {
                "trial": number,
                "candidate": jsonable(dict(candidate)),
                "candidate_sha256": candidate_sha256(candidate),
                "micro_batch": chosen,
                "attempts": attempts,
                "evaluation": evaluation,
            }
        )
        if progress is not None:
            progress.advance()
    return {
        "effective_batch": declared.effective_batch,
        "evaluation_batch": declared.evaluation_batch[model],
        "candidates": entries,
    }


def run_preflight(
    declaration: TrainingDeclaration,
    settings: ModelSettings,
    layout: InputLayout,
    candidates: Mapping[str, Sequence[Candidate]],
    *,
    device: Device,
    meter: Meter,
    seed: int,
    models: Sequence[str] = tuple(MODELS),
) -> dict[str, Any]:
    """The preflight of ``models``: one entry per model with its candidates' micro-batches."""
    rng = np.random.default_rng(SYNTHETIC_SEED)
    declared = declaration.neural
    if device == "gpu":
        cap_allocator(declared.preflight_free_margin_mib)
    setting = Setting(
        declaration,
        settings,
        layout,
        device,
        meter,
        seed,
        synthetic_data(layout, EFFECTIVE_BATCHES * declared.effective_batch, rng),
        {
            model: synthetic_validation(layout, declared.evaluation_batch[model], rng)
            for model in models
        },
    )
    total = sum(len(candidates[model]) for model in models)
    progress = ProgressLogger(logger, "Preflight candidates", total, every_seconds=10.0)
    with log_step(logger, f"Micro-batch preflight of {', '.join(models)} on {device}"):
        return {
            model: preflight_model(setting, model, candidates[model], progress) for model in models
        }


def build_record(
    declaration: TrainingDeclaration, identity: Mapping[str, Any], models: Mapping[str, Any]
) -> dict[str, Any]:
    """The versioned preflight record."""
    return {
        "record": RECORD,
        "format_version": FORMAT_VERSION,
        "identity": jsonable(dict(identity)),
        "margin_mib": declaration.neural.preflight_free_margin_mib,
        "micro_batch_candidates": list(declaration.neural.micro_batch_candidates),
        "effective_batches_per_epoch": EFFECTIVE_BATCHES,
        "epochs_measured": dict(PHASE_EPOCHS),
        "models": jsonable(dict(models)),
    }


def write_preflight(directory: Path, record: Mapping[str, Any]) -> Path:
    """Write ``preflight.json`` to a new file of ``directory``, never replacing one."""
    path = directory / FILE
    if path.exists():
        raise FileAccessError(f"{path}: already exists and is never replaced")
    write_json(path, dict(record))
    return path


# --- Reading a recorded plan ---------------------------------------------------------------------


def _mapping(value: object, what: str, keys: frozenset[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict) or (keys is not None and set(value) != keys):
        raise DataValidationError(f"The preflight's {what} is malformed")
    return value


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_phase(phase: object, what: str, margin_mib: int) -> dict[str, Any]:
    """A recorded phase: its fields, the arithmetic of its free memory and the pass flag derived
    again from the measurements, never taken from the saved boolean alone."""
    values = _mapping(phase, f"{what}", PHASE_KEYS)
    for name in ("baseline_used_mib", "sampled_max_used_mib", "total_mib", "samples"):
        if values[name] is not None and not _integer(values[name]):
            raise DataValidationError(f"The preflight's {what} has a malformed {name}")
    problems = values["problems"]
    if not isinstance(problems, list) or not all(isinstance(item, str) for item in problems):
        raise DataValidationError(f"The preflight's {what} has malformed problems")
    if not isinstance(values["completed"], bool) or not isinstance(values["passes"], bool):
        raise DataValidationError(f"The preflight's {what} has malformed flags")
    if values["error"] is not None and not isinstance(values["error"], str):
        raise DataValidationError(f"The preflight's {what} has a malformed error")
    total, peak = values["total_mib"], values["sampled_max_used_mib"]
    free = None if total is None or peak is None else total - peak
    short = values["samples"] is not None and values["samples"] < MIN_SAMPLES
    passes = (
        values["error"] is None
        and not problems
        and not short
        and free is not None
        and free >= margin_mib
    )
    consistent = (
        values["free_at_peak_mib"] == free
        and values["completed"] == (values["error"] is None)
        and values["passes"] == passes
    )
    if not consistent:
        raise SpecificationError(f"The preflight's {what} contradicts its own measurements")
    return values


def _check_candidate(
    model: str,
    number: int,
    candidate: Candidate,
    item: object,
    declaration: TrainingDeclaration,
) -> BatchPlan:
    """One recorded candidate: identity, the attempts as the procedure orders them (a prefix of the
    listed micro-batches that ends at the first pass) and the passing evaluation."""
    declared = declaration.neural
    margin = declared.preflight_free_margin_mib
    where = f"{model} candidate {number}"
    values = _mapping(item, where, ITEM_KEYS)
    if values["trial"] != number:
        raise SpecificationError(f"{where}: the recorded trial number is {values['trial']}")
    same = values["candidate"] == jsonable(dict(candidate))
    if not same or values["candidate_sha256"] != candidate_sha256(candidate):
        raise SpecificationError(f"The preflight of {where} is for another candidate")
    attempts = values["attempts"]
    allowed = micro_batches(model, declared.effective_batch, declared.micro_batch_candidates)
    if not isinstance(attempts, list) or not attempts or len(attempts) > len(allowed):
        raise SpecificationError(f"{where}: the recorded attempts do not follow the procedure")
    for position, attempt in enumerate(attempts):
        entry = _mapping(attempt, f"{where} attempt", ATTEMPT_KEYS)
        phase = _check_phase(entry["training"], f"{where} training phase", margin)
        last = position == len(attempts) - 1
        if entry["micro_batch"] != allowed[position] or phase["passes"] != last:
            raise SpecificationError(f"{where}: the recorded attempts do not follow the procedure")
    micro = values["micro_batch"]
    if micro != attempts[-1]["micro_batch"]:
        raise SpecificationError(f"{where}: the chosen micro-batch is not the passing attempt")
    evaluation = _check_phase(values["evaluation"], f"{where} evaluation phase", margin)
    if not evaluation["passes"]:
        raise SpecificationError(f"{where}: the evaluation phase did not pass")
    return BatchPlan(declared.effective_batch, micro)


def read_preflight(
    path: Path,
    declaration: TrainingDeclaration,
    candidates: Mapping[str, Sequence[Candidate]],
    *,
    identity: Mapping[str, Any],
    models: Sequence[str] = tuple(MODELS),
) -> dict[str, dict[int, BatchPlan]]:
    """The recorded batch plan of every candidate: ``{model: {trial: BatchPlan}}``.

    ``identity`` gives the values the plan must have been made for (for example the preprocessing
    artifact, declaration and settings hashes, the package source fingerprint and the GPU name):
    each must equal the recorded one. The record must also be of this format, use this
    declaration's margin, micro-batch list, effective batch and evaluation batch, and hold exactly
    the candidates now generated, in order, each with a valid attempt history, a passing
    micro-batch and a passing evaluation phase. Anything else is refused."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FileAccessError(f"{path}: cannot be read ({error.strerror})") from error
    except ValueError as error:
        raise DataValidationError(f"{path}: not valid JSON ({error})") from error
    if not isinstance(record, dict) or set(record) != RECORD_KEYS:
        raise DataValidationError(f"{path}: not a {RECORD} record with the expected keys")
    if record["record"] != RECORD or record["format_version"] != FORMAT_VERSION:
        raise DataValidationError(f"{path}: not a {RECORD} record of format {FORMAT_VERSION}")
    recorded = _mapping(record["identity"], "identity")
    for name, expected in identity.items():
        if recorded.get(name) != expected:
            raise SpecificationError(f"The preflight was made for another {name}")
    declared = declaration.neural
    if (
        record["margin_mib"] != declared.preflight_free_margin_mib
        or record["micro_batch_candidates"] != list(declared.micro_batch_candidates)
        or record["effective_batches_per_epoch"] != EFFECTIVE_BATCHES
        or record["epochs_measured"] != PHASE_EPOCHS
    ):
        raise SpecificationError("The preflight used other margin or micro-batch settings")
    entries = _mapping(record["models"], "models")
    plans: dict[str, dict[int, BatchPlan]] = {}
    for model in models:
        if model not in entries:
            raise SpecificationError(f"The preflight holds no entry for {model}")
        entry = _mapping(entries[model], f"{model} entry", ENTRY_KEYS)
        sizes = (entry["effective_batch"], entry["evaluation_batch"])
        if sizes != (declared.effective_batch, declared.evaluation_batch[model]):
            raise SpecificationError(f"The preflight of {model} used other batch sizes")
        listed = entry["candidates"]
        if not isinstance(listed, list) or len(listed) != len(candidates[model]):
            raise SpecificationError(f"The preflight of {model} is for other candidates")
        pairs = zip(candidates[model], listed, strict=True)
        plans[model] = {
            number: _check_candidate(model, number, candidate, item, declaration)
            for number, (candidate, item) in enumerate(pairs, start=1)
        }
    return plans
