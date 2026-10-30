"""The Test-period R4 outcomes and RQ3 cost analysis with their uncertainty (Issue #199;
DR-007 §7 and §9).

Point values come from ``cost.analyse`` (the queue totals, the exact curve with the saturation and
break-even ratios, the C_FP conversions and the grid table) on the Test labels, for every
configuration, calibrator and capacity. The uncertainty is descriptive; RQ3 has no hypothesis test:

- **Day-level moving block** (stream 5 ``day_block_rq3``, block length 3, B replicates): whole days
  are resampled, so each day's R3 and R4 queues stay intact and R3 and R4 are paired in every
  replicate. For each cost ratio rho of the grid, the per-day statistics of the queue (reviews N,
  frauds missed by R3, frauds F), of the unused reviews (u_d), of the frauds among them (f_d), of
  their calibrated probabilities and of the fraud amounts give, summed over the resampled days,

$$
\\Delta J = \\rho f - u, \\qquad
J_{\\mathrm{R3}} = N + \\rho \\, \\mathrm{FN}_{\\mathrm{R3}}, \\qquad
\\frac{\\Delta J}{J_{\\mathrm{R3}}}
$$

  and the ratios f / u, the mean calibrated probability of the unused reviews and the amount
  measures. A ratio whose denominator is zero on a replicate (no unused review, no fraud amount)
  is undefined there: it is left out of its interval and counted.
- **Two-way pigeonhole** (stream 6 ``pigeonhole_rq3``): frauds forgone f as F times the weighted
  share of frauds among the unused reviews, a sensitivity check beside the day-block interval.

The regime of a cell follows the interval of ΔJ: *identical* when no review is unused,
*R4 cheaper* when the interval lies below 0, *R3 cheaper* when it lies above 0, else
*inconclusive*. The saturation and break-even ratios keep their point values (DR-007 §7).

Nothing here reads a file: scores, calibrated probabilities, labels and amounts are arrays.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from review_capacity.errors import DataValidationError
from review_capacity.evaluation.cost import analyse, reviewed
from review_capacity.evaluation.declaration import EvaluationDeclaration
from review_capacity.evaluation.outcomes import Partition, interval_record
from review_capacity.evaluation.ranking import Capacity
from review_capacity.evaluation.resampling import (
    Floats,
    day_block_indices,
    pigeonhole_ratios,
    stream_generator,
)
from review_capacity.evaluation.weighted import day_counts
from review_capacity.logging_setup import log_step

logger = logging.getLogger(__name__)

REGIMES = ("identical", "r4_cheaper", "r3_cheaper", "inconclusive")
GRID_INTERVALS = (
    "unused",
    "frauds_forgone",
    "utilisation",
    "cost_r3",
    "cost_r4",
    "cost_difference",
    "relative_difference",
    "declined_fraud_rate",
    "declined_mean_probability",
    "fraud_value_exposed_r4",
    "fraud_value_forgone",
    "forgone_per_unused",
    "amount_weighted_recall_r4",
)
# The intervals that are null where the point value is (no unused review, no fraud amount).
NULLABLE_INTERVALS = (
    "declined_fraud_rate",
    "declined_mean_probability",
    "forgone_per_unused",
    "amount_weighted_recall_r4",
)
COMMON_ROWS = 5  # N, FN_R3, F, fraud amount, amount R3 does not review
RATIO_ROWS = 4  # u_d, f_d, probability mass and fraud amount of the unused reviews


@dataclass(frozen=True)
class CostSubject:
    """One configuration under one calibrator: its raw scores (the R3 queue order) and the
    calibrated probabilities (R4's threshold)."""

    configuration: str
    calibrator: str
    scores: npt.NDArray[Any]
    probabilities: Floats


def regime(unused: int, interval: dict[str, Any]) -> str:
    """The regime of one cell from its point u and the interval of ΔJ (DR-007 §7)."""
    if unused == 0:
        return "identical"
    if interval["upper"] < 0:
        return "r4_cheaper"
    if interval["lower"] > 0:
        return "r3_cheaper"
    return "inconclusive"


def _safe(top: Floats, bottom: Floats) -> Floats:
    """top / bottom, NaN where the denominator is not positive."""
    defined = bottom > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(defined, top / np.where(defined, bottom, 1.0), np.nan)


def _daily(partition: Partition, member: npt.NDArray[np.bool_]) -> Floats:
    """The common per-day statistics of one capacity (rows ``COMMON_ROWS``)."""
    truth = partition.truth
    fraud = truth.labels == 1
    days = len(truth.day_values)

    def count(mask: npt.NDArray[np.bool_]) -> Floats:
        return np.bincount(truth.day_index[mask], minlength=days).astype(np.float64)

    def amount(mask: npt.NDArray[np.bool_]) -> Floats:
        return np.bincount(truth.day_index[mask], weights=truth.amounts[mask], minlength=days)

    return np.vstack(
        [
            count(member),
            truth.frauds.astype(np.float64) - count(member & fraud),
            truth.frauds.astype(np.float64),
            amount(fraud),
            amount(fraud & ~member),
        ]
    )


def _ratio_daily(
    partition: Partition,
    declined: npt.NDArray[np.bool_],
    probabilities: Floats,
) -> Floats:
    truth = partition.truth
    fraud = truth.labels == 1
    days = len(truth.day_values)
    mask = declined & fraud
    return np.vstack(
        [
            np.bincount(truth.day_index[declined], minlength=days).astype(np.float64),
            np.bincount(truth.day_index[mask], minlength=days).astype(np.float64),
            np.bincount(truth.day_index[declined], weights=probabilities[declined], minlength=days),
            np.bincount(truth.day_index[mask], weights=truth.amounts[mask], minlength=days),
        ]
    )


def _cell_draws(sums: Floats, offset: int, ratio: float, k: int, days: int) -> dict[str, Floats]:
    """The replicates of every interval measure of one grid ratio, from the replicate sums (the
    ratio's columns start at ``offset``)."""
    n, miss, _, fraud_amount, exposed = (sums[:, i] for i in range(COMMON_ROWS))
    u, f, mass, forgone = (sums[:, offset + i] for i in range(RATIO_ROWS))
    j3 = n + ratio * miss
    delta = ratio * f - u
    exposed4 = exposed + forgone
    return {
        "unused": u,
        "frauds_forgone": f,
        "utilisation": (n - u) / (k * days),
        "cost_r3": j3,
        "cost_r4": j3 + delta,
        "cost_difference": delta,
        "relative_difference": delta / j3,
        "declined_fraud_rate": _safe(f, u),
        "declined_mean_probability": _safe(mass, u),
        "fraud_value_exposed_r4": exposed4,
        "fraud_value_forgone": forgone,
        "forgone_per_unused": _safe(forgone, u),
        "amount_weighted_recall_r4": 1 - _safe(exposed4, fraud_amount),
    }


def rq3_outcomes(
    declaration: EvaluationDeclaration,
    partition: Partition,
    subjects: Sequence[CostSubject],
    capacities: Sequence[Capacity],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """The cost analysis of every subject at every capacity, with the intervals of the module
    description: for each (configuration, calibrator) one entry per capacity holding the queue
    totals, the saturation and break-even results, the curve, the C_FP conversions and the grid,
    each grid row extended by its ``intervals``, the pigeonhole interval of the frauds forgone
    and its ``regime``."""
    resampling, level = declaration.resampling, declaration.levels.difference
    days = len(partition.truth.day_values)
    grid = declaration.rq3.ratio_grid
    indices = day_block_indices(
        stream_generator(declaration, "day_block_rq3"),
        days,
        resampling.block_length,
        replicates=resampling.replicates,
    )
    counts = day_counts(indices, days)
    truth = partition.truth
    fraud = truth.labels == 1
    total_frauds = int(truth.frauds.sum())
    result: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with log_step(logger, f"RQ3 on Test: {len(subjects)} subjects"):
        for subject in subjects:
            if subject.probabilities.shape != (truth.rows,):
                raise DataValidationError(
                    f"{subject.configuration}: {truth.rows} rows need {truth.rows} probabilities"
                )
            entries, forgone_rows = [], []
            for capacity in capacities:
                selection = partition.queue.select(subject.scores, capacity.k)
                body = analyse(truth, selection, subject.probabilities, capacity)
                member = selection.mask
                rows = [_daily(partition, member)]
                declined_by_ratio = []
                for ratio in grid:
                    declined = member & ~reviewed(selection, subject.probabilities, float(ratio))
                    declined_by_ratio.append(declined)
                    rows.append(_ratio_daily(partition, declined, subject.probabilities))
                    forgone_rows.append(declined[fraud].astype(np.float64))
                sums = counts @ np.vstack(rows).T
                entries.append((capacity, body, sums))
            pigeonhole = pigeonhole_ratios(
                stream_generator(declaration, "pigeonhole_rq3"),
                np.vstack(forgone_rows),
                partition.fraud_day_index,
                days,
                partition.entity_index,
                replicates=resampling.replicates,
            )
            result[(subject.configuration, subject.calibrator)] = [
                _entry(
                    capacity,
                    body,
                    sums,
                    pigeonhole.ratios[:, i * len(grid) : (i + 1) * len(grid)] * total_frauds,
                    grid,
                    days,
                    level,
                )
                for i, (capacity, body, sums) in enumerate(entries)
            ]
    return result


def _entry(
    capacity: Capacity,
    body: dict[str, Any],
    sums: Floats,
    pigeonhole: Floats,
    grid: Sequence[int],
    days: int,
    level: float,
) -> dict[str, Any]:
    rows = []
    for j, (ratio, point) in enumerate(zip(grid, body["grid"], strict=True)):
        offset = COMMON_ROWS + RATIO_ROWS * j
        draws = _cell_draws(sums, offset, float(ratio), capacity.k, days)
        intervals: dict[str, dict[str, Any] | None] = {}
        for name in GRID_INTERVALS:
            missing = name in NULLABLE_INTERVALS and point[name] is None
            intervals[name] = None if missing else interval_record(draws[name], level)
        delta = interval_record(draws["cost_difference"], level)
        rows.append(
            {
                **point,
                "intervals": intervals,
                "frauds_forgone_pigeonhole": interval_record(pigeonhole[:, j], level),
                "regime": regime(point["unused"], delta),
            }
        )
    return {"capacity": capacity.value, "k": capacity.k, **body, "grid": rows}
