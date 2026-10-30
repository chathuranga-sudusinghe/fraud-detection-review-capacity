"""Test outcomes and their intervals on a synthetic partition (Issue #197; no data read)."""

from __future__ import annotations

import dataclasses
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from review_capacity.errors import DataValidationError
from review_capacity.evaluation.calibration import fit_isotonic, fit_platt
from review_capacity.evaluation.declaration import EvaluationDeclaration, load_declaration
from review_capacity.evaluation.outcomes import (
    Partition,
    Thresholds,
    calibration_outcome,
    confirmatory_outcomes,
    secondary_resamples,
)
from review_capacity.evaluation.queue import DailyQueue
from review_capacity.evaluation.ranking import Capacity, Truth, r3_outcome
from review_capacity.evaluation.resampling import (
    day_block_indices,
    pigeonhole_ratios,
    stream_generator,
)
from support.synthetic import REPO_PATHS

DAYS, PER_DAY, B = 12, 300, 400
CAPACITIES = (Capacity(Fraction(1, 50), 6), Capacity(Fraction(1, 10), 30))
THRESHOLDS = Thresholds(0.5, 0.35, {0.02: 0.3, 0.1: 0.15})  # R2 differs by capacity


@pytest.fixture(scope="module")
def declaration() -> EvaluationDeclaration:
    base = load_declaration(REPO_PATHS)[0]
    return dataclasses.replace(base, resampling=dataclasses.replace(base.resampling, replicates=B))


@pytest.fixture(scope="module")
def partition() -> Partition:
    rng = np.random.default_rng(11)
    days = np.repeat(np.arange(1, DAYS + 1), PER_DAY).astype(np.int64)
    labels = (rng.random(len(days)) < 0.08).astype(np.int8)
    labels[days == 4] = 0  # a day without frauds
    amounts = np.round(rng.gamma(2.0, 50.0, len(days)), 2)
    ids = np.arange(len(days), dtype=np.int64)
    truth = Truth.build(days, labels, amounts)
    entities = rng.integers(0, 60, int(labels.sum()))
    _, entity_index = np.unique(entities, return_inverse=True)
    return Partition(truth, DailyQueue.build(ids, days), entity_index.astype(np.intp), 250.0)


@pytest.fixture(scope="module")
def scores(partition: Partition) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(12)
    labels = partition.truth.labels
    return {
        "LGB-b": np.round(np.clip(rng.normal(0.2 + 0.5 * labels, 0.2), 0, 1), 3),
        "LR": np.round(np.clip(rng.normal(0.15 + 0.3 * labels, 0.2), 0, 1), 3),
    }


@pytest.fixture(scope="module")
def outcomes(
    declaration: EvaluationDeclaration, partition: Partition, scores: dict[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    return confirmatory_outcomes(declaration, partition, scores, {"LGB-b": THRESHOLDS}, CAPACITIES)


def test_point_values_are_those_of_the_development_functions(
    partition: Partition, scores: dict[str, np.ndarray], outcomes: dict[str, dict[str, Any]]
) -> None:
    direct = r3_outcome(partition.queue, partition.truth, scores["LGB-b"], CAPACITIES[0])
    entry = outcomes["LGB-b"]["r3"][0]
    assert entry["counts"] == direct.counts
    assert entry["metrics"] == direct.metrics
    assert entry["daily"] == direct.daily


def test_the_reference_gets_r3_and_the_ranking_only(outcomes: dict[str, dict[str, Any]]) -> None:
    assert outcomes["LR"]["thresholds"] == []
    assert len(outcomes["LR"]["r3"]) == len(CAPACITIES)
    assert set(outcomes["LR"]["ranking"]["intervals"]) == {"average_precision", "roc_auc"}
    assert len(outcomes["LGB-b"]["thresholds"]) == 3 * len(CAPACITIES)


def test_every_interval_has_the_declared_level_and_replicates(
    declaration: EvaluationDeclaration, outcomes: dict[str, dict[str, Any]]
) -> None:
    found = 0
    for outcome in outcomes.values():
        entries = [*outcome["r3"], *outcome["thresholds"], outcome["ranking"]]
        for entry in entries:
            for interval in entry["intervals"].values():
                found += 1
                assert interval["level"] == declaration.levels.difference
                assert interval["replicates"] == B
                assert 0 <= interval["undefined"] < B
                assert interval["lower"] <= interval["upper"]
    assert found > 40


def test_the_recall_interval_is_the_pigeonhole_interval_of_the_shared_stream(
    declaration: EvaluationDeclaration,
    partition: Partition,
    scores: dict[str, np.ndarray],
    outcomes: dict[str, dict[str, Any]],
) -> None:
    truth = partition.truth
    fraud = truth.labels == 1
    hit = partition.queue.select(scores["LGB-b"], CAPACITIES[1].k).mask[fraud]
    replicates = pigeonhole_ratios(
        stream_generator(declaration, "pigeonhole_h1_h2"),
        hit.astype(np.float64)[np.newaxis],
        partition.fraud_day_index,
        DAYS,
        partition.entity_index,
        replicates=B,
    ).ratios[:, 0]
    interval = outcomes["LGB-b"]["r3"][1]["intervals"]["recall"]
    tail = 100 * (1 - declaration.levels.difference) / 2
    lower, upper = np.percentile(replicates, [tail, 100 - tail])
    assert interval["lower"] == pytest.approx(lower)
    assert interval["upper"] == pytest.approx(upper)


def test_derived_measures_scale_the_recall_replicates(outcomes: dict[str, dict[str, Any]]) -> None:
    entry = outcomes["LGB-b"]["r3"][0]
    counts, recall = entry["counts"], entry["intervals"]["recall"]
    precision = entry["intervals"]["precision"]
    assert precision["lower"] == pytest.approx(
        recall["lower"] * counts["frauds"] / counts["selected"]
    )
    lift = entry["intervals"]["lift"]
    assert lift["upper"] == pytest.approx(
        recall["upper"] * counts["transactions"] / counts["selected"]
    )
    misses = entry["intervals"]["false_negatives"]  # decreasing in recall: the ends swap
    assert misses["lower"] == pytest.approx(counts["frauds"] * (1 - recall["upper"]))
    assert misses["upper"] == pytest.approx(counts["frauds"] * (1 - recall["lower"]))


def test_a_threshold_interval_is_the_day_block_ratio_of_the_concatenated_days(
    declaration: EvaluationDeclaration, outcomes: dict[str, dict[str, Any]]
) -> None:
    indices = day_block_indices(
        stream_generator(declaration, "day_block_secondary"),
        DAYS,
        declaration.resampling.block_length,
        replicates=B,
    )
    entry = next(e for e in outcomes["LGB-b"]["thresholds"] if e["policy"] == "R2")
    daily = entry["daily"]
    tp = np.array([row["true_positives"] for row in daily])
    alerts = np.array([row["selected"] for row in daily])
    frauds = np.array([row["frauds"] for row in daily])
    f1 = 2 * tp[indices].sum(axis=1) / (frauds[indices].sum(axis=1) + alerts[indices].sum(axis=1))
    tail = 100 * (1 - declaration.levels.difference) / 2
    lower, upper = np.percentile(f1, [tail, 100 - tail])
    assert entry["intervals"]["f1"]["lower"] == pytest.approx(lower)
    assert entry["intervals"]["f1"]["upper"] == pytest.approx(upper)


def test_inputs_are_checked(
    declaration: EvaluationDeclaration, partition: Partition, scores: dict[str, np.ndarray]
) -> None:
    with pytest.raises(DataValidationError, match="scores"):
        confirmatory_outcomes(declaration, partition, {"LR": scores["LR"][:-1]}, {}, CAPACITIES)
    short = dataclasses.replace(partition, entity_index=partition.entity_index[:-1])
    with pytest.raises(DataValidationError, match="pseudo-entity"):
        confirmatory_outcomes(declaration, short, scores, {}, CAPACITIES)


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibration_intervals_surround_the_point_diagnostics(
    declaration: EvaluationDeclaration,
    partition: Partition,
    scores: dict[str, np.ndarray],
    method: str,
) -> None:
    labels = partition.truth.labels
    fitting = scores["LGB-b"]
    calibrator = (
        fit_platt(fitting, labels)[0] if method == "platt" else fit_isotonic(fitting, labels)
    )
    resamples = secondary_resamples(declaration, DAYS)
    result = calibration_outcome(declaration, partition, fitting, calibrator, resamples)
    for measure in ("brier", "log_loss"):
        interval = result["intervals"][measure]
        assert interval["lower"] <= result[measure] <= interval["upper"]
        assert interval["undefined"] == 0
    for measure in ("intercept", "slope"):
        interval = result["intervals"][measure]
        assert interval["replicates"] == B
        if result[measure] is not None:
            assert interval["lower"] <= interval["upper"]


def test_r2_uses_its_own_threshold_and_interval_for_every_capacity(
    scores: dict[str, np.ndarray], outcomes: dict[str, dict[str, Any]]
) -> None:
    entries = {e["capacity"]: e for e in outcomes["LGB-b"]["thresholds"] if e["policy"] == "R2"}
    assert {c: e["threshold"] for c, e in entries.items()} == {0.02: 0.3, 0.1: 0.15}
    for capacity, entry in entries.items():
        alerts = int((scores["LGB-b"] >= THRESHOLDS.r2[capacity]).sum())
        assert entry["counts"]["selected"] == alerts
    low, high = entries[0.02], entries[0.1]
    assert low["intervals"]["recall"] != high["intervals"]["recall"]  # separate alert sets
    shared = [e for e in outcomes["LGB-b"]["thresholds"] if e["policy"] == "R1"]
    assert shared[0]["intervals"] == shared[1]["intervals"]  # one alert set for every budget
    assert {e["threshold"] for e in shared} == {THRESHOLDS.r1}


def test_r2_needs_a_threshold_for_every_capacity(
    declaration: EvaluationDeclaration, partition: Partition, scores: dict[str, np.ndarray]
) -> None:
    missing = Thresholds(0.5, 0.35, {0.02: 0.3})
    with pytest.raises(DataValidationError, match=r"no frozen threshold for capacity 0\.1"):
        confirmatory_outcomes(declaration, partition, scores, {"LGB-b": missing}, CAPACITIES)
