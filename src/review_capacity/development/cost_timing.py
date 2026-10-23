"""Scoring throughput and single-transaction latency of the selected model (DR-015 §9; Issues #134
and #151).

The measurement primitives and the ``timing`` record builders are those of
``training.timing`` (Issue #121); this module supplies what is timed:

* **Scoring and throughput.** The selected model scores a complete partition, V1 or V2, in one call
  at two boundaries, reported separately. *Model-only*: the native inputs (``model.prepare``) are
  built beforehand and only ``model.predict_prepared`` is timed. *End-to-end*: ``model.predict`` is
  timed from the in-memory features (conversion, prediction, scores on the host). Each boundary has
  one untimed full prediction as warm-up, a settling interval before the block, and the declared
  number of timed passes; throughput is the rows divided by the median pass.
* **Latency.** One batch-of-one call per row on the declared number of distinct V1 rows (the
  smallest SHA-256 digests of the ``TransactionID``, in that fixed order), at both boundaries. The
  one-row inputs are prepared beforehand, outside the clock.

The tree-based models score on the CPU, with their production prediction thread counts and the
process's CPU affinity recorded (DR-013 §5). The reference scores on the CPU too: its model-only
boundary is the prediction from the prepared float64 matrix, its end-to-end boundary builds the
matrix from the canonical rows; the code does not set its threads, so the recorded count is the
largest size of the numerical libraries' thread pools that ``threadpoolctl`` reports in the
process (the pools themselves are recorded with it).

A neural model scores on the GPU through its float32 prediction path, with its evaluation batch.
Model-only: the network inputs of the partition are moved to the device in evaluation batches
beforehand (``model.prepare``) and only ``model.predict_prepared`` is timed, which ends with the
probabilities on the device. End-to-end: the shared preprocessing artifact turns the canonical
rows into network inputs, and ``model.predict`` transfers them, predicts and returns the scores on
the host, all inside the clock. Every timed pass and latency call is bracketed by an explicit
synchronisation of CUDA device 0 (DR-015 §9). The warm-up is the declared number of batch calls
(the normal evaluation batch shape, then the final short batch), not full passes. The measurement
records the allocator's peaks and the sampled GPU maximum. The step logger of the neural model is
silenced during a timed block, because writing log lines is not model work.

Disk loading, model loading, calibration, ranking and metrics lie outside every clock. Timings
are descriptive and never enter selection.
"""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from typing import Any

from review_capacity.config import ModelSettings
from review_capacity.development.memory import run_measured
from review_capacity.development.scoring import (
    MODEL_ROLE,
    ScoredModel,
    ScoredPartition,
    ScoringContext,
)
from review_capacity.development.subjects import NEURAL, kind_of, scoring_device, study_device
from review_capacity.errors import SpecificationError
from review_capacity.logging_setup import ProgressLogger, log_step
from review_capacity.models.neural_training import NeuralInputs, NeuralModel
from review_capacity.models.registry import Subject
from review_capacity.training.timing import (
    BOUNDARIES,
    Clock,
    Timed,
    latency_record,
    measure_latency,
    measure_scoring,
    scoring_record,
    select_latency_rows,
    settle,
)

logger = logging.getLogger(__name__)

THREAD_PARAMETER = {"lightgbm": "num_threads", "catboost": "thread_count"}
NEURAL_LOGGER = "review_capacity.models.neural_training"
REFERENCE_THREADS = (
    "the code does not set the reference's threads (scikit-learn and its numerical libraries "
    "choose them); the count is the largest thread pool reported by threadpoolctl"
)
SCORING_MEMORY_SCOPE = (
    "the {boundary} scoring of {partition}: the partition's rows (and, at model_only, their "
    "prepared native inputs) resident; the warm-up and the timed passes inside"
)
LATENCY_MEMORY_SCOPE = (
    "the {boundary} latency measurement: the one-row inputs (and, at model_only, their prepared "
    "native inputs) of the declared sample resident; the warm-up and the timed rounds inside"
)

type PreparedModel = ScoredModel


def library_threads() -> tuple[int, list[dict[str, Any]]]:
    """The largest thread-pool size of the numerical libraries loaded in this process, and the
    pools (``threadpoolctl``, a dependency of scikit-learn). Without any pool the libraries are
    single-threaded: 1."""
    threadpoolctl = importlib.import_module("threadpoolctl")
    pools: list[dict[str, Any]] = [
        {
            key: pool.get(key)
            for key in ("user_api", "internal_api", "num_threads", "threading_layer")
        }
        for pool in threadpoolctl.threadpool_info()
    ]
    sizes = [pool["num_threads"] for pool in pools if isinstance(pool["num_threads"], int)]
    return max(sizes, default=1), pools


def gpu_clock(clock: Clock) -> Clock:
    """``clock`` with the explicit synchronisation of CUDA device 0 before the clock starts and
    after the work completes (DR-015 §9); a process without CUDA has nothing to synchronise."""
    import torch

    def synchronise() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize(torch.device("cuda", 0))

    return Clock(clock.now, synchronise)


def timed_on(
    settings: ModelSettings,
    session: Mapping[str, Any],
    configuration: Subject,
    role: str | None,
    device: str,
    subject: Mapping[str, str] | None = None,
) -> Timed:
    """Who is timed: ``configuration`` on ``device``. On the GPU, with the attention backend of
    an FT-Transformer; on the CPU, with its threads and the process's CPU affinity."""
    model = configuration.model.name
    if device != "cpu":
        backend = settings.neural.gpu_attention_backend if model == "ft_transformer" else None
        return Timed(configuration.name, role, device, subject=subject, attention_backend=backend)
    affinity = tuple(int(cpu) for cpu in session["allowed_cpus"])
    if configuration.family is None:
        threads, pools = library_threads()
        return Timed(
            configuration.name,
            role,
            device,
            threads,
            affinity,
            subject=subject,
            extra={"threads_note": REFERENCE_THREADS, "thread_pools": pools},
        )
    threads = getattr(settings, model).params("cpu").get(THREAD_PARAMETER[model])
    if not isinstance(threads, int) or isinstance(threads, bool):
        raise SpecificationError(f"{configuration.name}: the CPU thread count is not declared")
    return Timed(configuration.name, role, device, threads, affinity, subject=subject)


@contextmanager
def quiet_neural_logging() -> Iterator[None]:
    """Silence the neural model's step logger inside a timed block (log lines are not model
    work, and a latency measurement makes tens of thousands of calls)."""
    target = logging.getLogger(NEURAL_LOGGER)
    previous = target.level
    target.setLevel(logging.WARNING)
    try:
        yield
    finally:
        target.setLevel(previous)


def _neural_inputs(context: ScoringContext) -> Any:
    if context.neural is None:
        raise SpecificationError("A neural model is timed with the run's neural inputs")
    return context.neural


def _boundary_calls(
    context: ScoringContext, model: PreparedModel, scored: ScoredPartition, boundary: str
) -> tuple[Callable[[], object], list[Callable[[], object]]]:
    """The call that one pass times at ``boundary`` and the untimed warm-up calls: for the
    tree-based models and the reference one full prediction, for a neural model the declared
    number of batch calls, the normal evaluation batch shape and then the final short batch."""
    if boundary not in BOUNDARIES:
        raise SpecificationError(f"The boundary must be one of {BOUNDARIES}")
    features = scored.rows.features
    warmup = context.declaration.timing.warmup_neural
    if isinstance(model, NeuralModel):
        neural = _neural_inputs(context)
        batch = model.parameters["prediction"]["eval_batch_size"]
        rows = scored.rows.rows
        short = rows % batch or min(batch, rows)
        if boundary == "model_only":
            prepared = model.prepare(neural.scoring_inputs(scored.label))
            normal, final = prepared[0], prepared[-1]
            device_calls: list[Callable[[], object]] = [
                lambda: model.predict_prepared([normal])
            ] * (warmup - 1)
            device_calls.append(lambda: model.predict_prepared([final]))
            return lambda: model.predict_prepared(prepared), device_calls
        head, tail = features.iloc[:batch], features.iloc[rows - short :]
        host_calls: list[Callable[[], object]] = [
            lambda: model.predict(neural.artifact.to_neural_inputs(head))
        ] * (warmup - 1)
        host_calls.append(lambda: model.predict(neural.artifact.to_neural_inputs(tail)))
        return lambda: model.predict(neural.artifact.to_neural_inputs(features)), host_calls
    warmup_tree = context.declaration.timing.warmup_tree
    one_pass: Callable[[], object]
    if boundary == "model_only":
        one_pass = partial(model.predict_prepared, model.prepare(features))
    else:
        one_pass = partial(model.predict, features)
    return one_pass, [one_pass] * warmup_tree


def _one_row_call(
    context: ScoringContext,
    model: PreparedModel,
    scored: ScoredPartition,
    positions: Any,
    boundary: str,
) -> Callable[[int], object]:
    """The call that one latency measurement times for sample row ``i`` at ``boundary``; the
    one-row inputs are built here, before the clock starts (a neural model's model-only inputs are
    already on the device; at its end-to-end boundary the canonical row is transformed inside the
    clock)."""
    features = scored.rows.features.iloc[positions].reset_index(drop=True)
    one_row = [features.iloc[[i]] for i in range(len(features))]
    if isinstance(model, NeuralModel):
        neural = _neural_inputs(context)
        if boundary == "model_only":
            source = neural.scoring_inputs(scored.label)
            device_rows = [
                model.prepare(
                    NeuralInputs(
                        source.numeric[p : p + 1].copy(), source.categorical[p : p + 1].copy()
                    )
                )
                for p in positions
            ]
            return lambda i: model.predict_prepared(device_rows[i])
        return lambda i: model.predict(neural.artifact.to_neural_inputs(one_row[i]))
    if boundary == "model_only":
        progress = ProgressLogger(logger, "One-row native inputs", len(one_row))
        natives: list[Any] = []
        for frame in one_row:
            natives.append(model.prepare(frame))
            progress.advance()
        return lambda i: model.predict_prepared(natives[i])
    return lambda i: model.predict(one_row[i])


def time_scoring(
    context: ScoringContext,
    configuration: Subject,
    model: PreparedModel,
    scored: ScoredPartition,
    *,
    clock: Clock | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Path]:
    """Time the complete scoring of ``scored``'s partition at both boundaries; one ``timing``
    record each."""
    clock = clock or Clock()
    timing, name = context.declaration.timing, configuration.name
    neural = kind_of(configuration) == NEURAL
    if neural:
        clock = gpu_clock(clock)
    device = study_device(configuration, scoring_device(context.settings, configuration))
    who = timed_on(
        context.settings, context.session, configuration, MODEL_ROLE, device, scored.record
    )
    batch_rows = (
        model.parameters["prediction"]["eval_batch_size"]
        if isinstance(model, NeuralModel)
        else scored.rows.rows
    )
    paths: list[Path] = []
    for boundary in BOUNDARIES:
        with log_step(logger, f"Timing {name} scoring of {scored.label}, {boundary}"):
            one_pass, warmup_calls = _boundary_calls(context, model, scored, boundary)
            settle(timing.settling_seconds, sleep)
            with quiet_neural_logging() if neural else nullcontext():
                measurement, memory = run_measured(
                    context.memory,
                    SCORING_MEMORY_SCOPE.format(boundary=boundary, partition=scored.label),
                    partial(
                        measure_scoring,
                        one_pass,
                        warmup_calls,
                        timing.scoring_passes,
                        clock,
                    ),
                    gpu=neural,
                    allocator=neural,
                )
        record = scoring_record(
            context.identity,
            who,
            context.session,
            partition=scored.label,
            boundary=boundary,
            rows=scored.rows.rows,
            batch_rows=batch_rows,
            measurement=measurement,
            memory=None if memory is None else memory.reading(),
        )
        paths.append(context.run.write_record(record, name, "scoring", scored.label, boundary))
    return paths


def time_latency(
    context: ScoringContext,
    configuration: Subject,
    model: PreparedModel,
    v1: ScoredPartition,
    *,
    clock: Clock | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Path]:
    """Time single-transaction scoring on the declared V1 sample at both boundaries; one
    ``timing`` record each."""
    clock = clock or Clock()
    timing, name = context.declaration.timing, configuration.name
    neural = kind_of(configuration) == NEURAL
    if neural:
        clock = gpu_clock(clock)
    positions = select_latency_rows(v1.rows.ids, timing.latency_rows)
    sample_ids = v1.rows.ids[positions]
    device = study_device(configuration, scoring_device(context.settings, configuration))
    who = timed_on(context.settings, context.session, configuration, MODEL_ROLE, device, v1.record)
    paths: list[Path] = []
    for boundary in BOUNDARIES:
        with log_step(logger, f"Timing {name} latency, {boundary}"):
            call_one = _one_row_call(context, model, v1, positions, boundary)
            settle(timing.settling_seconds, sleep)
            with quiet_neural_logging() if neural else nullcontext():
                measurement, memory = run_measured(
                    context.memory,
                    LATENCY_MEMORY_SCOPE.format(boundary=boundary),
                    partial(
                        measure_latency,
                        call_one,
                        len(positions),
                        timing.latency_warmup,
                        timing.latency_rounds,
                        clock,
                    ),
                    gpu=neural,
                    allocator=neural,
                )
        record = latency_record(
            context.identity,
            who,
            context.session,
            boundary=boundary,
            sample_ids=sample_ids,
            measurement=measurement,
            memory=None if memory is None else memory.reading(),
        )
        paths.append(context.run.write_record(record, name, "latency", boundary))
    return paths
