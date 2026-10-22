"""Scoring throughput and single-transaction latency of the selected tree-based model (DR-015 §9;
Issue #134).

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
process's CPU affinity recorded (DR-013 §5). Disk loading, model loading, calibration, ranking and
metrics lie outside every clock. Timings are descriptive and never enter selection.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from review_capacity.config import ModelSettings
from review_capacity.development.scoring import (
    MODEL_ROLE,
    SCORING_DEVICE,
    ScoredPartition,
    ScoringContext,
)
from review_capacity.errors import SpecificationError
from review_capacity.logging_setup import log_step
from review_capacity.models.catboost_model import CatBoostModel
from review_capacity.models.lightgbm_model import LightGBMModel
from review_capacity.models.registry import Configuration
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

type TreeModel = LightGBMModel | CatBoostModel


def timed_on(
    settings: ModelSettings,
    session: Mapping[str, Any],
    configuration: Configuration,
    role: str | None,
    device: str,
    subject: Mapping[str, str] | None = None,
) -> Timed:
    """Who is timed: ``configuration`` on ``device``, with its CPU threads and the process's CPU
    affinity when that device is the CPU."""
    if device != "cpu":
        return Timed(configuration.name, role, device, subject=subject)
    model = configuration.model.name
    threads = getattr(settings, model).params("cpu").get(THREAD_PARAMETER[model])
    if not isinstance(threads, int) or isinstance(threads, bool):
        raise SpecificationError(f"{configuration.name}: the CPU thread count is not declared")
    affinity = tuple(int(cpu) for cpu in session["allowed_cpus"])
    return Timed(configuration.name, role, device, threads, affinity, subject=subject)


def _boundary_calls(
    model: TreeModel, features: pd.DataFrame, boundary: str
) -> Callable[[], object]:
    """The call that one pass times at ``boundary``."""
    if boundary == "model_only":
        native = model.prepare(features)
        return lambda: model.predict_prepared(native)
    if boundary == "end_to_end":
        return lambda: model.predict(features)
    raise SpecificationError(f"The boundary must be one of {BOUNDARIES}")


def _one_row_call(
    model: TreeModel, one_row: list[pd.DataFrame], boundary: str
) -> Callable[[int], object]:
    """The call that one latency measurement times for row ``i`` at ``boundary``; the one-row
    inputs are built here, before the clock starts."""
    if boundary == "model_only":
        natives = [model.prepare(frame) for frame in one_row]

        def prepared(i: int) -> object:
            return model.predict_prepared(natives[i])

        return prepared

    def end_to_end(i: int) -> object:
        return model.predict(one_row[i])

    return end_to_end


def time_scoring(
    context: ScoringContext,
    configuration: Configuration,
    model: TreeModel,
    scored: ScoredPartition,
    *,
    clock: Clock | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Path]:
    """Time the complete scoring of ``scored``'s partition at both boundaries; one ``timing``
    record each."""
    clock = clock or Clock()
    timing, name = context.declaration.timing, configuration.name
    who = timed_on(
        context.settings, context.session, configuration, MODEL_ROLE, SCORING_DEVICE, scored.record
    )
    paths: list[Path] = []
    for boundary in BOUNDARIES:
        with log_step(logger, f"Timing {name} scoring of {scored.label}, {boundary}"):
            one_pass = _boundary_calls(model, scored.rows.features, boundary)
            settle(timing.settling_seconds, sleep)
            measurement = measure_scoring(
                one_pass, [one_pass] * timing.warmup_tree, timing.scoring_passes, clock
            )
        record = scoring_record(
            context.identity,
            who,
            context.session,
            partition=scored.label,
            boundary=boundary,
            rows=scored.rows.rows,
            batch_rows=scored.rows.rows,
            measurement=measurement,
        )
        paths.append(context.run.write_record(record, name, "scoring", scored.label, boundary))
    return paths


def time_latency(
    context: ScoringContext,
    configuration: Configuration,
    model: TreeModel,
    v1: ScoredPartition,
    *,
    clock: Clock | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Path]:
    """Time single-transaction scoring on the declared V1 sample at both boundaries; one
    ``timing`` record each."""
    clock = clock or Clock()
    timing, name = context.declaration.timing, configuration.name
    positions = select_latency_rows(v1.rows.ids, timing.latency_rows)
    sample_ids = v1.rows.ids[positions]
    sample = v1.rows.features.iloc[positions].reset_index(drop=True)
    one_row = [sample.iloc[[i]] for i in range(len(sample))]
    who = timed_on(
        context.settings, context.session, configuration, MODEL_ROLE, SCORING_DEVICE, v1.record
    )
    paths: list[Path] = []
    for boundary in BOUNDARIES:
        with log_step(logger, f"Timing {name} latency, {boundary}"):
            call_one = _one_row_call(model, one_row, boundary)
            settle(timing.settling_seconds, sleep)
            measurement = measure_latency(
                call_one, len(one_row), timing.latency_warmup, timing.latency_rounds, clock
            )
        record = latency_record(
            context.identity,
            who,
            context.session,
            boundary=boundary,
            sample_ids=sample_ids,
            measurement=measurement,
        )
        paths.append(context.run.write_record(record, name, "latency", boundary))
    return paths
