"""Scoring throughput and latency timing of the selected model, with a deterministic clock:
LightGBM on synthetic data on the CPU (no project data)."""

from __future__ import annotations

import itertools
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from review_capacity.development.cost_timing import time_latency, time_scoring, timed_on
from review_capacity.development.run_output import load_run
from review_capacity.development.scoring import ScoringContext, load_selected, score_partition
from review_capacity.development.trial import TrialOutcome
from review_capacity.models.registry import configuration
from review_capacity.training.timing import Clock, rows_identity, select_latency_rows
from support.search import V1_ROWS, searched

LGB_U = configuration("LGB-u")
TICK = 0.5

pytestmark = pytest.mark.usefixtures("restore_lightgbm_logger")


def half_second_clock() -> Clock:
    """Every timed call reads the clock twice, half a second apart."""
    ticks = itertools.count()
    return Clock(now=lambda: next(ticks) * TICK)


@pytest.fixture
def scoring(tmp_path: Path) -> tuple[ScoringContext, TrialOutcome]:
    return searched(tmp_path)


def timings(context: ScoringContext, measurement: str) -> list[dict[str, Any]]:
    loaded = load_run(context.run.path, context.run.partitions)
    return [t for t in loaded.of_kind("timing") if t["measurement"] == measurement]


def test_scoring_is_timed_at_both_boundaries(
    scoring: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, selected = scoring
    model = load_selected(context, LGB_U, selected)
    scored = score_partition(context, LGB_U, selected, model, "V1")
    time_scoring(context, LGB_U, model, scored, clock=half_second_clock())
    records = timings(context, "scoring")
    assert sorted(r["boundary"] for r in records) == ["end_to_end", "model_only"]
    for record in records:
        assert record["partition"] == "V1"
        assert record["model_role"] == "selected"
        assert record["device"] == "cpu"
        assert record["threads"] == 24
        assert record["rows"] == record["batch_rows"] == V1_ROWS
        assert record["warmup"] == 1
        assert record["repetitions"] == 2
        assert record["seconds"] == TICK
        assert record["rows_per_second"] == V1_ROWS / TICK
        assert record["settings"]["pass_seconds"] == [TICK, TICK]
        assert record["settings"]["pass_min_seconds"] == record["settings"]["pass_max_seconds"]
        assert record["subject"] == scored.record


def test_the_settling_interval_precedes_each_scoring_block(
    scoring: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, selected = scoring
    declaration = replace(
        context.declaration, timing=replace(context.declaration.timing, settling_seconds=7)
    )
    context = replace(context, declaration=declaration)
    model = load_selected(context, LGB_U, selected)
    scored = score_partition(context, LGB_U, selected, model, "V1")
    slept: list[float] = []
    time_scoring(context, LGB_U, model, scored, clock=half_second_clock(), sleep=slept.append)
    assert slept == [7.0, 7.0]


def test_latency_is_timed_on_the_declared_v1_sample_at_both_boundaries(
    scoring: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, selected = scoring
    model = load_selected(context, LGB_U, selected)
    scored = score_partition(context, LGB_U, selected, model, "V1")
    time_latency(context, LGB_U, model, scored, clock=half_second_clock())
    records = timings(context, "latency")
    assert sorted(r["boundary"] for r in records) == ["end_to_end", "model_only"]
    positions = select_latency_rows(scored.rows.ids, 20)
    for record in records:
        assert record["rows"] == 20
        assert record["batch_rows"] == 1
        assert record["warmup"] == 3
        assert record["repetitions"] == 2
        assert record["rows_sha256"] == rows_identity(scored.rows.ids[positions])
        assert record["settings"]["timed_calls"] == 40
        assert record["latency_ms"] == {
            "median": TICK * 1000,
            "p95": TICK * 1000,
            "p99": TICK * 1000,
        }
        assert record["subject"] == scored.record


def test_the_timed_party_names_its_threads_and_affinity_on_the_cpu_only(
    scoring: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, _ = scoring
    cpu = timed_on(context.settings, context.session, LGB_U, "selected", "cpu")
    assert cpu.threads == 24
    assert cpu.cpu_affinity == tuple(context.session["allowed_cpus"])
    gpu = timed_on(context.settings, context.session, configuration("Cat-u"), None, "gpu")
    assert (gpu.threads, gpu.cpu_affinity) == (None, None)
    catboost_cpu = timed_on(context.settings, context.session, configuration("Cat-u"), None, "cpu")
    assert catboost_cpu.threads == 24
