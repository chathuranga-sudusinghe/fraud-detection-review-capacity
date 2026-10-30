"""The Test outcomes of the policies R0 to R3 with their intervals (Issue #197; protocol,
Statistical Plan, "Secondary outcomes").

Point values come from the development functions on the Test labels: ``ranking.r3_outcome`` and
``ranking.ranking_outcome`` for R3 and the whole ranking, ``workload.threshold_outcome`` for the
uncapped R0, R1 and R2 against each daily budget. Every interval is a percentile bootstrap
interval with the declared B and level, chosen by the outcome's dependence (DR-006; Issue #185):

- **two-way pigeonhole** (stream 1, the weights of H1 and H2, shared by all configurations): the
  fraud-based ratios of R3, namely recall, amount-weighted recall, and the measures that are fixed
  multiples of recall once the review count is fixed: for the frauds F, the transactions N, the
  reviews S and the oracle-reachable frauds A of one capacity,

$$
\\text{precision} = \\frac{F}{S}\\,\\text{recall}, \\quad
\\text{lift} = \\frac{N}{S}\\,\\text{recall}, \\quad
\\text{normalised recall} = \\frac{F}{A}\\,\\text{recall}, \\quad
\\text{FN} = F\\,(1 - \\text{recall});
$$

- **day-level moving block** (stream 8, shared by all configurations): Average Precision, ROC-AUC
  and the precision, recall and F1 of R0 to R2, recomputed on the resampled days.

An interval is undefined for a replicate where its measure has no value; such replicates are left
out and counted. A measure with no defined replicate stops the run (DR-006).

Nothing here reads a file: scores, labels, amounts and entity indices are arrays.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from review_capacity.errors import DataValidationError
from review_capacity.evaluation.calibration import EPSILON, Calibrator, diagnostics
from review_capacity.evaluation.declaration import EvaluationDeclaration
from review_capacity.evaluation.inference import percentile_interval
from review_capacity.evaluation.queue import DailyQueue
from review_capacity.evaluation.ranking import (
    Capacity,
    R3Outcome,
    Truth,
    r3_outcome,
    ranking_outcome,
)
from review_capacity.evaluation.resampling import (
    Floats,
    Positions,
    Replicates,
    day_block_indices,
    day_ratios,
    pigeonhole_ratios,
    stream_generator,
)
from review_capacity.evaluation.weighted import (
    RankedRows,
    calibration_replicates,
    day_counts,
    ranking_replicates,
)
from review_capacity.evaluation.workload import ThresholdOutcome, threshold_outcome
from review_capacity.logging_setup import log_step

logger = logging.getLogger(__name__)

THRESHOLD_MEASURES = ("recall", "precision", "f1")
RANKING_MEASURES = ("average_precision", "roc_auc")
R3_MEASURES = (
    "recall",
    "precision",
    "frauds_per_1000_reviews",
    "normalised_recall",
    "lift",
    "amount_weighted_recall",
    "false_negatives",
)


@dataclass(frozen=True)
class Thresholds:
    """The frozen thresholds of one configuration: one each for R0 (0.5) and R1, and for R2 the
    threshold tau_c of every capacity c, keyed by the share as the records write it (0.02)."""

    r0: float
    r1: float
    r2: Mapping[float, float]

    def of(self, policy: str, capacity: Capacity) -> float:
        if policy == "R0":
            return self.r0
        if policy == "R1":
            return self.r1
        if capacity.value not in self.r2:
            raise DataValidationError(f"R2 has no frozen threshold for capacity {capacity.value}")
        return self.r2[capacity.value]


@dataclass(frozen=True)
class Partition:
    """The Test rows as the outcomes need them: labels and amounts (``truth``), the shared daily
    queue, the pseudo-entity of each fraud in row order (0 .. G - 1 without gaps) and the V2
    reference volume of the fidelity decomposition."""

    truth: Truth
    queue: DailyQueue
    entity_index: Positions
    reference_volume: float

    @property
    def fraud_day_index(self) -> Positions:
        fraud: Positions = self.truth.day_index[self.truth.labels == 1]
        return fraud


def interval_record(draws: npt.ArrayLike, level: float) -> dict[str, Any]:
    """The percentile interval of the defined replicates, with the counts the record keeps."""
    interval = percentile_interval(draws, level)
    return {
        "lower": interval.lower,
        "upper": interval.upper,
        "level": interval.level,
        "undefined": interval.undefined,
        "replicates": interval.replicates,
    }


@dataclass(frozen=True)
class _Points:
    """The point outcomes of one configuration."""

    r3: list[tuple[Capacity, R3Outcome, npt.NDArray[np.bool_]]]  # reviewed flags of the frauds
    thresholds: list[tuple[str, float, Capacity, ThresholdOutcome]]
    ranking: dict[str, float | None]
    ranked: RankedRows


def _points(
    partition: Partition,
    scores: npt.NDArray[Any],
    capacities: Sequence[Capacity],
    thresholds: Thresholds | None,
) -> _Points:
    truth = partition.truth
    exact = np.asarray(scores)
    fraud = truth.labels == 1
    r3 = []
    for capacity in capacities:
        outcome = r3_outcome(partition.queue, truth, exact, capacity)
        reviewed = partition.queue.select(exact, capacity.k).mask
        r3.append((capacity, outcome, reviewed[fraud]))
    rules: list[tuple[str, float, Capacity, ThresholdOutcome]] = []
    if thresholds is not None:
        for policy in ("R0", "R1", "R2"):
            for capacity in capacities:
                value = thresholds.of(policy, capacity)
                rules.append(
                    (policy, value, capacity, threshold_outcome(truth, exact, value, capacity))
                )
    ranked = RankedRows.build(
        exact.astype(np.float64), truth.labels, truth.day_index.astype(np.intp)
    )
    return _Points(r3, rules, ranking_outcome(truth, exact), ranked)


def _fraud_rows(
    partition: Partition, points: Mapping[str, _Points]
) -> tuple[Floats, Floats, list[tuple[str, int]]]:
    """Numerators and denominators of the pigeonhole ratios: for every configuration and capacity
    the capture indicator over its frauds, then the amount-weighted capture."""
    amounts = partition.truth.amounts[partition.truth.labels == 1]
    top, bottom, labels = [], [], []
    for name, point in points.items():
        for index, (_, _, hit) in enumerate(point.r3):
            top.append(hit.astype(np.float64))
            bottom.append(np.ones(len(hit)))
            top.append(amounts * hit)
            bottom.append(amounts)
            labels.append((name, index))
    return np.vstack(top), np.vstack(bottom), labels


def _scaled_measures(counts: Mapping[str, int], achievable: int) -> dict[str, tuple[float, float]]:
    """The (scale, offset) of each recall-derived measure: value = scale x recall + offset."""
    frauds, selected, transactions = counts["frauds"], counts["selected"], counts["transactions"]
    return {
        "recall": (1.0, 0.0),
        "precision": (frauds / selected, 0.0),
        "frauds_per_1000_reviews": (1000 * frauds / selected, 0.0),
        "normalised_recall": (frauds / achievable, 0.0),
        "lift": (transactions / selected, 0.0),
        "false_negatives": (-float(frauds), float(frauds)),
    }


def secondary_resamples(declaration: EvaluationDeclaration, days: int) -> tuple[Positions, Floats]:
    """The day-block resamples of stream 8 and their day multiplicities, shared by every
    configuration, policy and calibration method so that their intervals are paired."""
    resampling = declaration.resampling
    indices = day_block_indices(
        stream_generator(declaration, "day_block_secondary"),
        days,
        resampling.block_length,
        replicates=resampling.replicates,
    )
    return indices, day_counts(indices, days)


def calibration_outcome(
    declaration: EvaluationDeclaration,
    partition: Partition,
    scores: npt.NDArray[Any],
    calibrator: Calibrator,
    resamples: tuple[Positions, Floats],
) -> dict[str, Any]:
    """The calibration diagnostics of ``calibrator`` on the Test rows (Brier score, clipped log
    loss, intercept, slope, reliability table) with day-block intervals from ``resamples``."""
    truth = partition.truth
    indices, counts = resamples
    days = len(truth.day_values)
    level = declaration.levels.difference
    values = np.asarray(scores, dtype=np.float64)
    result = diagnostics(
        calibrator,
        values,
        truth.labels,
        rows_used="held_out",
        days={"first": truth.first_day, "last": truth.last_day},
    )
    probabilities = calibrator.predict(values)
    clipped = np.clip(probabilities, EPSILON, 1 - EPSILON)
    y = truth.labels.astype(np.float64)
    daily = np.vstack(
        [
            np.bincount(truth.day_index, weights=(probabilities - y) ** 2, minlength=days),
            np.bincount(
                truth.day_index,
                weights=-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)),
                minlength=days,
            ),
        ]
    )
    losses = day_ratios(daily, truth.transactions.astype(np.float64), indices)
    fits = calibration_replicates(
        np.log(clipped) - np.log1p(-clipped),
        truth.labels,
        truth.day_index.astype(np.intp),
        counts,
        point=(result["intercept"], result["slope"]),
    )
    result["intervals"] = {
        "brier": interval_record(losses.ratios[:, 0], level),
        "log_loss": interval_record(losses.ratios[:, 1], level),
        "intercept": interval_record(fits.ratios[:, 0], level),
        "slope": interval_record(fits.ratios[:, 1], level),
    }
    return result


def confirmatory_outcomes(
    declaration: EvaluationDeclaration,
    partition: Partition,
    scores: Mapping[str, npt.NDArray[Any]],
    thresholds: Mapping[str, Thresholds],
    capacities: Sequence[Capacity],
) -> dict[str, dict[str, Any]]:
    """The R0 to R3 and ranking outcomes of every configuration in ``scores`` with their
    intervals. ``thresholds`` holds the R0, R1 and R2 thresholds (R2 for every capacity) of the
    configurations that take them; a configuration without an entry (the reference) gets R3 and
    the ranking only."""
    validate_inputs(partition, scores)
    level = declaration.levels.difference
    replicates = declaration.resampling.replicates
    days = len(partition.truth.day_values)
    with log_step(logger, f"Test outcomes: {len(scores)} configurations"):
        points = {
            name: _points(partition, s, capacities, thresholds.get(name))
            for name, s in scores.items()
        }
        top, bottom, order = _fraud_rows(partition, points)
        fraud = pigeonhole_ratios(
            stream_generator(declaration, "pigeonhole_h1_h2"),
            top,
            partition.fraud_day_index,
            days,
            partition.entity_index,
            replicates=replicates,
            denominators=bottom,
        )
        indices, counts = secondary_resamples(declaration, days)
        position = {key: 2 * i for i, key in enumerate(order)}
        return {
            name: _assemble(point, name, fraud, position, indices, counts, level)
            for name, point in points.items()
        }


def _assemble(
    point: _Points,
    name: str,
    fraud: Replicates,
    position: Mapping[tuple[str, int], int],
    indices: Positions,
    counts: Floats,
    level: float,
) -> dict[str, Any]:
    r3 = []
    for index, (capacity, outcome, _) in enumerate(point.r3):
        recall = fraud.ratios[:, position[(name, index)]]
        amount = fraud.ratios[:, position[(name, index)] + 1]
        achievable = sum(row["oracle_reviewable"] for row in outcome.daily)
        intervals: dict[str, dict[str, Any] | None] = {}
        for measure, (scale, offset) in _scaled_measures(outcome.counts, achievable).items():
            intervals[measure] = interval_record(scale * recall + offset, level)
        intervals["amount_weighted_recall"] = (
            None
            if outcome.metrics["amount_weighted_recall"] is None
            else interval_record(amount, level)
        )
        r3.append(
            {
                "capacity": capacity.value,
                "k": capacity.k,
                "counts": outcome.counts,
                "daily": outcome.daily,
                "cutoff_ties": outcome.cutoff_ties,
                "metrics": outcome.metrics,
                "intervals": intervals,
            }
        )
    ranking = ranking_replicates(point.ranked, counts)
    result: dict[str, Any] = {
        "ranking": {
            "metrics": point.ranking,
            "intervals": {
                measure: interval_record(ranking.ratios[:, i], level)
                for i, measure in enumerate(RANKING_MEASURES)
            },
        },
        "r3": r3,
        "thresholds": [],
    }
    if point.thresholds:
        result["thresholds"] = _threshold_entries(point, indices, level)
    return result


def _threshold_entries(point: _Points, indices: Positions, level: float) -> list[dict[str, Any]]:
    """R0 to R2 against each budget. The alert set of R0 and R1 does not depend on the budget, so
    their intervals come from one day-block computation per policy; R2 has its own threshold and
    alert set for each capacity, so it has its own computation for each."""

    def key(policy: str, capacity: Capacity) -> tuple[str, float | None]:
        return (policy, capacity.value if policy == "R2" else None)

    draws: dict[tuple[str, float | None], Replicates] = {}
    for policy, _, capacity, outcome in point.thresholds:
        if key(policy, capacity) in draws:
            continue
        daily = outcome.daily
        hits = np.array([row["true_positives"] for row in daily], dtype=np.float64)
        alerts = np.array([row["selected"] for row in daily], dtype=np.float64)
        frauds = np.array([row["frauds"] for row in daily], dtype=np.float64)
        draws[key(policy, capacity)] = day_ratios(
            np.vstack([hits, hits, 2 * hits]),
            np.vstack([frauds, alerts, frauds + alerts]),
            indices,
        )
    entries = []
    for policy, threshold, capacity, outcome in point.thresholds:
        replicates = draws[key(policy, capacity)]
        intervals: dict[str, dict[str, Any] | None] = {}
        for i, measure in enumerate(THRESHOLD_MEASURES):
            intervals[measure] = (
                None
                if outcome.metrics[measure] is None
                else interval_record(replicates.ratios[:, i], level)
            )
        entries.append(
            {
                "policy": policy,
                "capacity": capacity.value,
                "k": capacity.k,
                "threshold": threshold,
                "counts": outcome.counts,
                "daily": outcome.daily,
                "metrics": outcome.metrics,
                "intervals": intervals,
            }
        )
    return entries


def validate_inputs(partition: Partition, scores: Mapping[str, npt.NDArray[Any]]) -> None:
    """The array checks that apply to every configuration before any outcome is computed."""
    rows = partition.truth.rows
    frauds = int(partition.truth.frauds.sum())
    if partition.entity_index.shape != (frauds,):
        raise DataValidationError(f"{frauds} frauds need {frauds} pseudo-entity indices")
    for name, values in scores.items():
        if np.shape(values) != (rows,):
            raise DataValidationError(f"{name}: {rows} rows need {rows} scores")
