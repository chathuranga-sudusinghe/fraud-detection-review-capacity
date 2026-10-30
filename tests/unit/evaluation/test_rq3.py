"""Test-period R4 outcomes and RQ3 intervals on a synthetic partition (Issue #199; no data read)."""

from __future__ import annotations

import dataclasses
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from review_capacity.errors import DataValidationError
from review_capacity.evaluation.cost import analyse, reviewed
from review_capacity.evaluation.declaration import EvaluationDeclaration, load_declaration
from review_capacity.evaluation.outcomes import Partition
from review_capacity.evaluation.queue import DailyQueue
from review_capacity.evaluation.ranking import Capacity, Truth
from review_capacity.evaluation.resampling import (
    day_block_indices,
    pigeonhole_ratios,
    stream_generator,
)
from review_capacity.evaluation.rq3 import CostSubject, regime, rq3_outcomes
from support.synthetic import REPO_PATHS

DAYS, PER_DAY, B = 12, 300, 300
CAPACITIES = (Capacity(Fraction(1, 50), 6), Capacity(Fraction(1, 10), 30))
RATIOS = (2, 5, 10, 20, 50, 100, 200, 500, 1000)


@pytest.fixture(scope="module")
def declaration() -> EvaluationDeclaration:
    base = load_declaration(REPO_PATHS)[0]
    return dataclasses.replace(base, resampling=dataclasses.replace(base.resampling, replicates=B))


@pytest.fixture(scope="module")
def partition() -> Partition:
    rng = np.random.default_rng(31)
    days = np.repeat(np.arange(1, DAYS + 1), PER_DAY).astype(np.int64)
    labels = (rng.random(len(days)) < 0.08).astype(np.int8)
    amounts = np.round(rng.gamma(2.0, 50.0, len(days)), 2)
    truth = Truth.build(days, labels, amounts)
    _, entities = np.unique(rng.integers(0, 50, int(labels.sum())), return_inverse=True)
    queue = DailyQueue.build(np.arange(len(days), dtype=np.int64), days)
    return Partition(truth, queue, entities.astype(np.intp), 250.0)


@pytest.fixture(scope="module")
def subject(partition: Partition) -> CostSubject:
    rng = np.random.default_rng(32)
    labels = partition.truth.labels
    scores = np.round(np.clip(rng.normal(0.15 + 0.5 * labels, 0.2), 0, 1), 3)
    return CostSubject("LGB-b", "platt", scores, np.clip(scores**2, 0, 1))


@pytest.fixture(scope="module")
def outcome(
    declaration: EvaluationDeclaration, partition: Partition, subject: CostSubject
) -> list[dict[str, Any]]:
    return rq3_outcomes(declaration, partition, [subject], CAPACITIES)[("LGB-b", "platt")]


def per_day(partition: Partition, subject: CostSubject, capacity: Capacity, ratio: int) -> Any:
    """u_d, f_d, the queue and the R3 misses per day, derived independently of the module."""
    truth = partition.truth
    selection = partition.queue.select(subject.scores, capacity.k)
    declined = selection.mask & ~reviewed(selection, subject.probabilities, float(ratio))
    fraud = truth.labels == 1
    n = len(truth.day_values)
    u = np.bincount(truth.day_index[declined], minlength=n)
    f = np.bincount(truth.day_index[declined & fraud], minlength=n)
    queue = np.bincount(truth.day_index[selection.mask], minlength=n)
    captured = np.bincount(truth.day_index[selection.mask & fraud], minlength=n)
    return u, f, queue, truth.frauds - captured


def test_point_values_are_those_of_the_cost_analysis(
    partition: Partition, subject: CostSubject, outcome: list[dict[str, Any]]
) -> None:
    for entry, capacity in zip(outcome, CAPACITIES, strict=True):
        direct = analyse(
            partition.truth,
            partition.queue.select(subject.scores, capacity.k),
            subject.probabilities,
            capacity,
        )
        assert entry["capacity"] == capacity.value
        assert entry["k"] == capacity.k
        for name in ("queue", "saturation", "break_even", "curve", "friction"):
            assert entry[name] == direct[name]
        for row, expected in zip(entry["grid"], direct["grid"], strict=True):
            assert {k: v for k, v in row.items() if k in expected} == expected


def test_day_block_intervals_equal_the_brute_force_resampling(
    declaration: EvaluationDeclaration,
    partition: Partition,
    subject: CostSubject,
    outcome: list[dict[str, Any]],
) -> None:
    indices = day_block_indices(
        stream_generator(declaration, "day_block_rq3"), DAYS, 3, replicates=B
    )
    tail = 100 * (1 - declaration.levels.difference) / 2
    for index, capacity in enumerate(CAPACITIES):
        for j, ratio in enumerate(RATIOS):
            u_d, f_d, queue_d, miss_d = per_day(partition, subject, capacity, ratio)
            u, f = u_d[indices].sum(axis=1), f_d[indices].sum(axis=1)
            delta = ratio * f - u
            j3 = queue_d[indices].sum(axis=1) + ratio * miss_d[indices].sum(axis=1)
            row = outcome[index]["grid"][j]["intervals"]
            for name, draws in (
                ("cost_difference", delta),
                ("relative_difference", delta / j3),
                ("unused", u),
                ("frauds_forgone", f),
            ):
                lower, upper = np.percentile(draws, [tail, 100 - tail])
                assert row[name]["lower"] == pytest.approx(lower, rel=1e-9, abs=1e-12)
                assert row[name]["upper"] == pytest.approx(upper, rel=1e-9, abs=1e-12)
            if row["declined_fraud_rate"] is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    rate = f / u
                defined = rate[np.isfinite(rate)]
                lower, upper = np.percentile(defined, [tail, 100 - tail])
                assert row["declined_fraud_rate"]["lower"] == pytest.approx(lower)
                assert row["declined_fraud_rate"]["undefined"] == int((u == 0).sum())


def test_the_pigeonhole_interval_of_the_frauds_forgone_uses_stream_six(
    declaration: EvaluationDeclaration,
    partition: Partition,
    subject: CostSubject,
    outcome: list[dict[str, Any]],
) -> None:
    truth = partition.truth
    fraud = truth.labels == 1
    capacity, ratio = CAPACITIES[1], 20
    selection = partition.queue.select(subject.scores, capacity.k)
    declined = (selection.mask & ~reviewed(selection, subject.probabilities, float(ratio)))[fraud]
    share = pigeonhole_ratios(
        stream_generator(declaration, "pigeonhole_rq3"),
        declined.astype(np.float64)[np.newaxis],
        partition.fraud_day_index,
        DAYS,
        partition.entity_index,
        replicates=B,
    ).ratios[:, 0]
    tail = 100 * (1 - declaration.levels.difference) / 2
    lower, upper = np.percentile(share * int(fraud.sum()), [tail, 100 - tail])
    interval = outcome[1]["grid"][RATIOS.index(ratio)]["frauds_forgone_pigeonhole"]
    assert interval["lower"] == pytest.approx(lower)
    assert interval["upper"] == pytest.approx(upper)


def test_the_regime_follows_the_interval_of_the_cost_difference() -> None:
    assert regime(0, {"lower": -1.0, "upper": 1.0}) == "identical"
    assert regime(5, {"lower": -3.0, "upper": -0.5}) == "r4_cheaper"
    assert regime(5, {"lower": 0.5, "upper": 3.0}) == "r3_cheaper"
    assert regime(5, {"lower": -1.0, "upper": 2.0}) == "inconclusive"
    assert regime(5, {"lower": 0.0, "upper": 2.0}) == "inconclusive"  # touching 0 is not above
    assert regime(5, {"lower": -2.0, "upper": 0.0}) == "inconclusive"


def test_every_cell_has_its_regime_and_the_intervals_of_the_module(
    outcome: list[dict[str, Any]],
) -> None:
    for entry in outcome:
        for row in entry["grid"]:
            assert row["regime"] == regime(row["unused"], row["intervals"]["cost_difference"])
            assert row["intervals"]["cost_difference"]["level"] == 0.95
            assert row["intervals"]["cost_difference"]["replicates"] == B
            assert (row["declined_fraud_rate"] is None) == (
                row["intervals"]["declined_fraud_rate"] is None
            )


def test_a_ratio_undefined_on_some_resamples_is_counted(
    declaration: EvaluationDeclaration, partition: Partition
) -> None:
    """Only day 4 has queue members with a small probability, so the resamples without day 4 have
    no unused review at a high ratio: f / u is undefined there and counted."""
    rng = np.random.default_rng(33)
    labels = partition.truth.labels
    scores = np.round(np.clip(rng.normal(0.15 + 0.5 * labels, 0.2), 0, 1), 3)
    probabilities = np.where(partition.truth.days == 4, scores * 0.0005, 0.5 + 0.5 * scores)
    result = rq3_outcomes(
        declaration,
        partition,
        [CostSubject("LGB-b", "isotonic", scores, probabilities)],
        CAPACITIES[:1],
    )[("LGB-b", "isotonic")]
    row = result[0]["grid"][RATIOS.index(1000)]
    indices = day_block_indices(
        stream_generator(declaration, "day_block_rq3"), DAYS, 3, replicates=B
    )
    without = int((~(indices == 3).any(axis=1)).sum())  # day 4 is position 3
    assert without > 0
    assert row["intervals"]["declined_fraud_rate"]["undefined"] == without


def test_probabilities_must_cover_every_row(
    declaration: EvaluationDeclaration, partition: Partition, subject: CostSubject
) -> None:
    short = dataclasses.replace(subject, probabilities=subject.probabilities[:-1])
    with pytest.raises(DataValidationError, match="probabilities"):
        rq3_outcomes(declaration, partition, [short], CAPACITIES)
