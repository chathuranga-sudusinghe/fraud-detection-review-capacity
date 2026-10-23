"""The micro-batch preflight of the two neural models (DR-015 §7; Issue #151, Part 2).

Before the search, a synthetic, data-free preflight on the project GPU fixes the micro-batch of
each of a model's ten candidates:

1. It uses the production implementation (the neural loop of ``models.neural_training``) with
   full-size synthetic inputs and the vocabularies of the shared preprocessing artifact, strict
   determinism and the precision and attention backend of DR-013 §4. One training epoch of one
   effective batch (1,024 rows) is run: forward, backward and one AdamW update, with initialised
   optimiser state.
2. It tries the declared micro-batches (1,024, 512, 256, 128, 64) in that order and records the
   largest that completes with at least ``preflight_free_margin_mib`` of device memory free at
   the sampled peak. Training and float32 evaluation are checked separately: the evaluation is
   one full batch of the model's evaluation batch size (4,096 or 16 rows) on the fitted model.
3. The ResNet-style MLP is tried at the effective batch only, because its batch normalisation
   needs the whole batch (``ACCUMULATES`` is False); FT-Transformer accumulates gradients.
4. A candidate that passes at no listed size, or whose evaluation fails the margin, stops the
   search for an issue (``ModelError``); no candidate is dropped or replaced.
5. The preflight draws its synthetic data from its own generator and restores the Python, NumPy
   and PyTorch random states after every attempt, so the training random state is unchanged.

The recorded plan (``preflight.json``) is used for the unweighted and balanced configurations and
for every refit, and is never changed after V1 results are seen. ``read_preflight`` checks a
recorded plan against the declaration, the preprocessing artifact and the candidate lists before
it is used.

GPU memory is the whole-device memory sampled by ``nvidia-smi`` (``probe.diagnostics``): a
sampled maximum, not a true peak, including memory held by other processes. The memory reading is
a ``Meter``, so that the procedure is tested without a GPU.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import random
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from review_capacity.config import Device, ModelSettings
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
from review_capacity.training.declaration import TrainingDeclaration
from review_capacity.training.sampler import Candidate

logger = logging.getLogger(__name__)

RECORD = "neural_preflight"
FORMAT_VERSION = 1
FILE = "preflight.json"
RUN_TYPE = "neural_preflight"
MODELS: dict[str, type[NeuralModel]] = {"resnet": ResNetModel, "ft_transformer": FTTransformerModel}
CONFIGURATIONS = {"resnet": "RES-u", "ft_transformer": "FTT-u"}  # weighting plays no part here
SYNTHETIC_FRAUD_SHARE = 0.2
SYNTHETIC_SEED = 0
RECORD_KEYS = frozenset(
    {"record", "format_version", "identity", "margin_mib", "micro_batch_candidates", "models"}
)


# --- Memory readings --------------------------------------------------------------------------


@dataclass(frozen=True)
class Reading:
    """The whole-device GPU memory sampled during one phase."""

    total_mib: int | None
    baseline_used_mib: int | None
    sampled_max_used_mib: int | None
    problems: tuple[str, ...] = ()

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
        from review_capacity.probe.diagnostics import GpuMemorySampler

        sampler = GpuMemorySampler()
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
            )


# --- Inputs, hyperparameters and random state -------------------------------------------------


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


@contextmanager
def preserved_random_state(device: Device) -> Iterator[None]:
    """Restore the Python, NumPy and PyTorch (and CUDA) random states on exit: a fit seeds them."""
    import torch

    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if device == "gpu" else None  # initialises CUDA
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _release(device: Device) -> None:
    gc.collect()
    if device == "gpu":
        import torch

        torch.cuda.empty_cache()


# --- The procedure ----------------------------------------------------------------------------


def _phase(reading: Reading, error: str | None, margin_mib: int) -> dict[str, Any]:
    free = reading.free_mib
    completed = error is None
    passes = completed and not reading.problems and free is not None and free >= margin_mib
    return {
        "completed": completed,
        "error": error,
        "baseline_used_mib": reading.baseline_used_mib,
        "sampled_max_used_mib": reading.sampled_max_used_mib,
        "total_mib": reading.total_mib,
        "free_at_peak_mib": free,
        "problems": list(reading.problems),
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
    evaluation: dict[str, NeuralInputs]


def _train_attempt(
    setting: Setting, model: str, hyperparameters: Mapping[str, Any], micro: int
) -> tuple[dict[str, Any], NeuralModel | None]:
    """One epoch of one effective batch at ``micro`` rows per micro-batch."""
    declared = setting.declaration.neural
    fitted: NeuralModel | None = None
    error: str | None = None
    with setting.meter.watch() as watch:
        try:
            with preserved_random_state(setting.device):
                instance = MODELS[model](
                    configuration(CONFIGURATIONS[model]),
                    setting.seed,
                    settings=setting.settings,
                    layout=setting.layout,
                    device=setting.device,
                )
                instance.fit(
                    setting.train,
                    None,
                    hyperparameters,
                    iterations=1,
                    plan=BatchPlan(declared.effective_batch, micro),
                )
                fitted = instance
        except ModelError as failure:
            if not _out_of_memory(failure):
                raise
            error = "out of GPU memory during training"
    assert watch.reading is not None
    phase = _phase(watch.reading, error, declared.preflight_free_margin_mib)
    if not phase["passes"]:
        fitted = None
        _release(setting.device)
    return phase, fitted


def _evaluation_attempt(setting: Setting, model: str, fitted: NeuralModel) -> dict[str, Any]:
    """Float32 evaluation of the fitted model on one full evaluation batch."""
    error: str | None = None
    with setting.meter.watch() as watch:
        try:
            with preserved_random_state(setting.device):
                fitted.predict(setting.evaluation[model])
        except ModelError as failure:
            if not _out_of_memory(failure):
                raise
            error = "out of GPU memory during evaluation"
    assert watch.reading is not None
    return _phase(watch.reading, error, setting.declaration.neural.preflight_free_margin_mib)


def candidate_sha256(candidate: Candidate) -> str:
    """SHA-256 of a candidate as canonical JSON (sorted names, no spaces)."""
    text = json.dumps(jsonable(dict(candidate)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def micro_batches(model: str, effective: int, declared: Sequence[int]) -> tuple[int, ...]:
    """The micro-batches tried for ``model``, in order: the declared sizes up to the effective
    batch; for a model that cannot accumulate, the effective batch only."""
    sizes = tuple(size for size in declared if size <= effective)
    return sizes if MODELS[model].ACCUMULATES else (effective,)


def preflight_model(
    setting: Setting,
    model: str,
    candidates: Sequence[Candidate],
    progress: ProgressLogger | None = None,
) -> dict[str, Any]:
    """The preflight of one model's candidates; raises ``ModelError`` for a candidate that passes
    at no listed micro-batch or whose float32 evaluation fails the memory margin."""
    declared = setting.declaration.neural
    entries: list[dict[str, Any]] = []
    for number, candidate in enumerate(candidates, start=1):
        hyperparameters = neural_hyperparameters(setting.declaration, model, candidate)
        attempts: list[dict[str, Any]] = []
        chosen: int | None = None
        fitted: NeuralModel | None = None
        with log_step(logger, f"Preflight {model}, candidate {number} of {len(candidates)}"):
            tried = micro_batches(model, declared.effective_batch, declared.micro_batch_candidates)
            for micro in tried:
                phase, fitted = _train_attempt(setting, model, hyperparameters, micro)
                attempts.append({"micro_batch": micro, "training": phase})
                logger.info(
                    "%s candidate %d, micro-batch %d: %s, free at the sampled peak %s MiB",
                    model,
                    number,
                    micro,
                    "passes" if phase["passes"] else f"fails ({phase['error'] or 'memory margin'})",
                    phase["free_at_peak_mib"],
                )
                if fitted is not None:
                    chosen = micro
                    break
            if chosen is None or fitted is None:
                raise ModelError(
                    f"{model} candidate {number} passes at no listed micro-batch "
                    f"{list(declared.micro_batch_candidates)}: the search stops for an issue"
                )
            evaluation = _evaluation_attempt(setting, model, fitted)
            del fitted
            _release(setting.device)
            if not evaluation["passes"]:
                raise ModelError(
                    f"{model} candidate {number}: float32 evaluation at batch "
                    f"{declared.evaluation_batch[model]} fails the memory margin: the search stops "
                    "for an issue"
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
    setting = Setting(
        declaration,
        settings,
        layout,
        device,
        meter,
        seed,
        synthetic_data(layout, declared.effective_batch, rng),
        {
            model: synthetic_data(layout, declared.evaluation_batch[model], rng).inputs
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
        "models": jsonable(dict(models)),
    }


def write_preflight(directory: Path, record: Mapping[str, Any]) -> Path:
    """Write ``preflight.json`` to a new file of ``directory``, never replacing one."""
    path = directory / FILE
    if path.exists():
        raise FileAccessError(f"{path}: already exists and is never replaced")
    write_json(path, dict(record))
    return path


# --- Reading a recorded plan ------------------------------------------------------------------


def read_preflight(
    path: Path,
    declaration: TrainingDeclaration,
    candidates: Mapping[str, Sequence[Candidate]],
    *,
    artifact_sha256: str,
    models: Sequence[str] = tuple(MODELS),
) -> dict[str, dict[int, BatchPlan]]:
    """The recorded batch plan of every candidate: ``{model: {trial: BatchPlan}}``.

    The record must be of this format, made for this preprocessing artifact and this declaration's
    margin, effective batch, micro-batch list and evaluation batch, and hold exactly the candidates
    now generated (by their SHA-256) with a passing, listed micro-batch each; a plan for
    a ResNet-style MLP candidate must be the whole effective batch. Anything else is refused."""
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
    declared = declaration.neural
    if record["identity"].get("artifact_sha256") != artifact_sha256:
        raise SpecificationError("The preflight was made for another preprocessing artifact")
    if record["margin_mib"] != declared.preflight_free_margin_mib or record[
        "micro_batch_candidates"
    ] != list(declared.micro_batch_candidates):
        raise SpecificationError("The preflight used other margin or micro-batch settings")
    plans: dict[str, dict[int, BatchPlan]] = {}
    for model in models:
        entry = record["models"].get(model)
        if entry is None:
            raise SpecificationError(f"The preflight holds no entry for {model}")
        if (entry["effective_batch"], entry["evaluation_batch"]) != (
            declared.effective_batch,
            declared.evaluation_batch[model],
        ):
            raise SpecificationError(f"The preflight of {model} used other batch sizes")
        recorded = entry["candidates"]
        expected = [candidate_sha256(candidate) for candidate in candidates[model]]
        if [item["candidate_sha256"] for item in recorded] != expected:
            raise SpecificationError(f"The preflight of {model} is for other candidates")
        plans[model] = {}
        for item in recorded:
            micro = item["micro_batch"]
            allowed = micro_batches(
                model, declared.effective_batch, declared.micro_batch_candidates
            )
            last = item["attempts"][-1]
            passing = last["micro_batch"] == micro and last["training"]["passes"]
            if micro not in allowed or not passing:
                raise SpecificationError(
                    f"{model} candidate {item['trial']}: no passing micro-batch"
                )
            if not item["evaluation"]["passes"]:
                raise SpecificationError(f"{model} candidate {item['trial']}: evaluation failed")
            plans[model][item["trial"]] = BatchPlan(declared.effective_batch, micro)
    return plans
