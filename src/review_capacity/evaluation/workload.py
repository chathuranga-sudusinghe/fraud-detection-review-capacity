"""Uncapped threshold outcomes and workload against the fixed daily budget (DR-009; Issue #157).

R0, R1 and R2 are evaluated as **uncapped** threshold policies: every transaction whose score
reaches the threshold is an alert and counts as review demand; none is truncated to K. For each
relative day d, with A_d the alerts and K the fixed daily budget of a capacity scenario,

$$
O_d = \\max(A_d - K, 0), \\qquad U_d = \\max(K - A_d, 0), \\qquad L_d = A_d / K
$$

(overflow, unused capacity, load ratio). The labels give the alert-set measures: TP the frauds
among the alerts, recall = TP / frauds, precision = TP / alerts (undefined without an alert) and
F1 = 2 TP / (frauds + alerts). Frauds among the alerts in excess of K are neither missed nor false
negatives (DR-009), and no measure here counts them.

Nothing here reads a file: scores and labels are arrays, and the ``evaluation`` and
``r1_selection`` records are built for the caller to write.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import numpy.typing as npt

from review_capacity.development.records import RECORD_CONTRACTS
from review_capacity.errors import DataValidationError
from review_capacity.evaluation.ranking import Capacity, Subject, Truth, evaluation_record
from review_capacity.evaluation.thresholds import (
    R1_RULE_VERSION,
    FloatScores,
    R1Selection,
    alert_mask,
    describe,
)
from review_capacity.logging_setup import log_step

logger = logging.getLogger(__name__)

THRESHOLD_POLICIES = ("R0", "R1", "R2")


@dataclass(frozen=True)
class ThresholdOutcome:
    """The results of one threshold at one budget, in the shapes of the ``evaluation`` record."""

    counts: dict[str, int]
    daily: list[dict[str, Any]]
    metrics: dict[str, float | None]


def threshold_outcome(
    truth: Truth, scores: FloatScores, threshold: float, capacity: Capacity
) -> ThresholdOutcome:
    """The uncapped policy ``score >= threshold`` on the rows of ``truth``, against ``capacity``."""
    if np.shape(scores) != (truth.rows,):
        raise DataValidationError(f"{truth.rows} rows need {truth.rows} scores")
    k = capacity.k
    with log_step(logger, f"Threshold {threshold!r} against K = {k} ({capacity.value:.0%})"):
        flagged = alert_mask(scores, threshold)
        fraud = truth.labels == 1
        n_days = len(truth.day_values)
        alerts_by_day = np.bincount(truth.day_index[flagged], minlength=n_days)
        hits_by_day = np.bincount(truth.day_index[flagged & fraud], minlength=n_days)
        daily = []
        for i in range(n_days):
            n_d, f_d = int(truth.transactions[i]), int(truth.frauds[i])
            a_d, tp_d = int(alerts_by_day[i]), int(hits_by_day[i])
            daily.append(
                {
                    "day": int(truth.day_values[i]),
                    "transactions": n_d,
                    "frauds": f_d,
                    "selected": a_d,
                    "true_positives": tp_d,
                    "oracle_reviewable": min(k, f_d),
                    "overflow": max(a_d - k, 0),
                    "unused": max(k - a_d, 0),
                    "load_ratio": a_d / k,
                    "cutoff_tied_group": None,
                }
            )
        frauds, alerts = int(truth.frauds.sum()), int(alerts_by_day.sum())
        hits = int(hits_by_day.sum())
        counts = {
            "days": n_days,
            "transactions": int(truth.transactions.sum()),
            "frauds": frauds,
            "selected": alerts,
            "true_positives": hits,
            "false_positives": alerts - hits,
            "false_negatives": frauds - hits,
        }
        metrics = _metrics(counts, daily, capacity)
    return ThresholdOutcome(counts, daily, metrics)


def _metrics(
    counts: Mapping[str, int], daily: list[dict[str, Any]], capacity: Capacity
) -> dict[str, float | None]:
    frauds, alerts, hits = counts["frauds"], counts["selected"], counts["true_positives"]
    prevalence = Fraction(frauds, counts["transactions"])
    return {
        "recall": hits / frauds,
        "precision": hits / alerts if alerts else None,
        "f1": 2 * hits / (frauds + alerts),
        "capacity_to_prevalence": float(capacity.share / prevalence),
        "total_overflow": sum(row["overflow"] for row in daily),
        "total_unused": sum(row["unused"] for row in daily),
        "mean_load_ratio": sum(row["load_ratio"] for row in daily) / len(daily),
        "days_above_budget": sum(row["overflow"] > 0 for row in daily),
        "days_below_budget": sum(row["unused"] > 0 for row in daily),
        "zero_alert_days": sum(row["selected"] == 0 for row in daily),
    }


def threshold_record(
    identity: Mapping[str, Any],
    subject: Subject,
    truth: Truth,
    policy: str,
    threshold: float,
    capacity: Capacity,
    outcome: ThresholdOutcome,
) -> dict[str, Any]:
    """The ``evaluation`` record of R0, R1 or R2 at ``capacity`` (validated by its writer)."""
    if policy not in THRESHOLD_POLICIES:
        raise DataValidationError(f"{policy!r} is not a threshold policy (R0, R1 or R2)")
    record = evaluation_record(identity, subject, truth, policy)
    record |= {
        "capacity": capacity.value,
        "k": capacity.k,
        "threshold": threshold,
        "counts": outcome.counts,
        "metrics": outcome.metrics,
        "daily": outcome.daily,
    }
    return record


def r1_selection_record(
    identity: Mapping[str, Any],
    configuration: str,
    score: Mapping[str, str],
    period: str,
    selection_days: Mapping[str, int],
    selection: R1Selection,
) -> dict[str, Any]:
    """The ``r1_selection`` record of one selection, a recorded error included."""
    return {
        "record": "r1_selection",
        "schema_version": RECORD_CONTRACTS["r1_selection"].version,
        "identity": dict(identity),
        "configuration": configuration,
        "score": dict(score),
        "period": period,
        "selection_days": dict(selection_days),
        "rule_version": R1_RULE_VERSION,
        **describe(selection),
    }


def day_subset(truth_days: npt.NDArray[np.int64], first: int, last: int) -> npt.NDArray[np.bool_]:
    """The rows whose relative day lies in ``first..last``."""
    return (truth_days >= first) & (truth_days <= last)
