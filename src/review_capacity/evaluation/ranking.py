"""Ranking and capacity outcomes of development scores (Issue #122; Issue #96; DR-002).

Given a partition's scores, labels and transaction amounts, this module computes

* **R3**, daily top-K review, at a capacity share c with its fixed daily budget K_c: pooled
  counts, one row per relative day, the score ties that cross the daily cutoff, and the
  measures of the protocol's Outcomes (``development.records.METRICS``): Recall@K, Precision@K,
  false negatives (in the counts), frauds per 1,000 reviews, the oracle recall ceiling, pooled
  normalised recall, the random-review context and lift, utilisation, capacity relative to
  prevalence, amount-weighted recall and the fraud value not reviewed;
* **the whole ranking**: Average Precision and ROC-AUC with equal scores kept as one group (the
  SHA-256 tie order of ``evaluation.queue`` cuts review queues only and never enters these).
  Average Precision is the step-function (non-interpolated) area under the precision-recall
  curve, ``sum_n (R_n - R_{n-1}) P_n`` over the distinct scores; it is what the protocol's
  "Average Precision / PR-AUC" denotes here, not a trapezoidal PR-AUC. ROC-AUC gives a tied
  positive-negative pair half credit.

and builds the ``evaluation`` records of those results. Nothing here reads a file: scores,
labels and amounts are arrays, and loading them is the caller's task (the search runner).

**Daily budget** (DR-002 §3). K_c is the V2 mean daily transaction volume times c, rounded half
up, computed exactly from the V2 row and day counts of ``config/dataset.yaml``.

**Definitions** (DR-002, protocol Outcomes), with F_d the frauds, N_d the transactions and TP_d
the reviewed frauds of relative day d, and A_d = min(K, N_d) the day's reviews under R3:

$$
\\text{oracle ceiling} = \\frac{\\sum_d \\min(K, F_d)}{\\sum_d F_d}, \\qquad
\\text{normalised recall} = \\frac{\\sum_d \\mathrm{TP}_d}{\\sum_d \\min(K, F_d)}
$$

Days without frauds add 0 to both sums. Random review of the same number of transactions
captures a share ``reviews / transactions`` of the frauds in expectation, so
``lift = recall / (reviews / transactions) = precision / prevalence``. Utilisation is
``reviews / (K x days)``. Amount-weighted recall is the amount of reviewed frauds over the
amount of all frauds; the amount of the other frauds is the fraud value not reviewed (an
exposure proxy, not a loss).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, roc_auc_score

from review_capacity.config import PartitionSpec
from review_capacity.development.records import RECORD_CONTRACTS
from review_capacity.errors import DataValidationError
from review_capacity.evaluation.queue import DailyQueue
from review_capacity.logging_setup import log_step
from review_capacity.models.base import Keys, Labels

logger = logging.getLogger(__name__)

# The capacity scenarios of DR-002 §1: 1 %, 2 % (primary), 5 % and 10 % of transactions.
CAPACITY_SHARES: tuple[Fraction, ...] = (
    Fraction(1, 100),
    Fraction(2, 100),
    Fraction(5, 100),
    Fraction(10, 100),
)

type Amounts = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Capacity:
    """A review-capacity share c and its fixed daily review budget K (reviews per day)."""

    share: Fraction
    k: int

    @property
    def value(self) -> float:
        """The share as the records write it (0.02 for 2 %)."""
        return float(self.share)


def daily_budget(share: Fraction, partition: PartitionSpec) -> int:
    """K_c = round(c x mean daily volume of ``partition``), rounded half up (DR-002 §3)."""
    days = partition.last_day - partition.first_day + 1
    exact = share * partition.rows / days
    return math.floor(exact + Fraction(1, 2))


def capacity_budgets(v2: PartitionSpec) -> tuple[Capacity, ...]:
    """The four capacities with K_c from the mean daily volume of V2 (28, 56, 139, 278)."""
    return tuple(Capacity(share, daily_budget(share, v2)) for share in CAPACITY_SHARES)


@dataclass(frozen=True)
class Truth:
    """The labels and amounts of a partition's rows, checked once; the per-day counts follow.

    The relative days must be consecutive, each with at least one transaction, as the
    ``evaluation`` record lists every day of its range. The partition must hold a fraud.
    """

    days: Keys
    labels: Labels
    amounts: Amounts
    day_values: Keys
    day_index: npt.NDArray[np.intp]
    transactions: npt.NDArray[np.int64]
    frauds: npt.NDArray[np.int64]

    @classmethod
    def build(cls, days: Keys, labels: Labels, amounts: Amounts) -> Truth:
        if days.ndim != 1 or days.dtype != np.int64 or days.size == 0:
            raise DataValidationError("Days must be a non-empty one-dimensional int64 array")
        rows = len(days)
        if labels.shape != (rows,) or labels.dtype != np.int8 or not np.isin(labels, (0, 1)).all():
            raise DataValidationError(f"Labels must be an int8 array of 0 or 1 for {rows} rows")
        if amounts.shape != (rows,) or amounts.dtype != np.float64:
            raise DataValidationError(f"Amounts must be a float64 array for {rows} rows")
        if not np.isfinite(amounts).all():
            raise DataValidationError("TransactionAmt must be finite for every row")
        if (amounts < 0).any():
            raise DataValidationError("TransactionAmt must not be negative for an amount share")
        days, labels, amounts = days.copy(), labels.copy(), amounts.copy()  # owned, not shared
        values, index, counts = np.unique(days, return_inverse=True, return_counts=True)
        if int(values[-1] - values[0]) + 1 != len(values):
            raise DataValidationError("Every relative day of the range needs a transaction")
        fraud = labels == 1
        if not fraud.any():
            raise DataValidationError("The partition has no fraud: its outcomes are undefined")
        frauds = np.bincount(index[fraud], minlength=len(values)).astype(np.int64)
        transactions = counts.astype(np.int64)
        for array in (days, labels, amounts, values, index, transactions, frauds):
            array.flags.writeable = False  # the cached counts must stay those of these arrays
        return cls(days, labels, amounts, values, index, transactions, frauds)

    @property
    def rows(self) -> int:
        return len(self.days)

    @property
    def first_day(self) -> int:
        return int(self.day_values[0])

    @property
    def last_day(self) -> int:
        return int(self.day_values[-1])


@dataclass(frozen=True)
class R3Outcome:
    """The R3 results at one capacity, in the shapes of the ``evaluation`` record."""

    counts: dict[str, int]
    daily: list[dict[str, Any]]
    cutoff_ties: dict[str, Any]
    metrics: dict[str, float | None]


def r3_outcome(
    queue: DailyQueue, truth: Truth, scores: npt.NDArray[Any], capacity: Capacity
) -> R3Outcome:
    """R3 at ``capacity``: the first K transactions of every day's queue are reviewed.

    ``queue`` and ``truth`` describe the same rows in the same order. Equal scores are ordered
    by the shared tie rule; the days and sizes of tied groups that cross a cutoff are reported.
    """
    if queue.rows != truth.rows or not np.array_equal(queue.days, truth.days):
        raise DataValidationError("The queue and the labels describe different rows")
    k = capacity.k
    with log_step(logger, f"R3 at {capacity.value:.0%} (K = {k})"):
        selection = queue.select(scores, k)
        reviewed = selection.mask
        fraud = truth.labels == 1
        n_days = len(truth.day_values)
        selected_by_day = np.bincount(truth.day_index[reviewed], minlength=n_days)
        hits_by_day = np.bincount(truth.day_index[reviewed & fraud], minlength=n_days)
        daily = []
        for i, summary in enumerate(selection.days):
            n_d, f_d = int(truth.transactions[i]), int(truth.frauds[i])
            a_d, tp_d = int(selected_by_day[i]), int(hits_by_day[i])
            if summary.day != int(truth.day_values[i]) or summary.selected != a_d:
                raise DataValidationError(f"The queue and the labels disagree on day {summary.day}")
            daily.append(
                {
                    "day": summary.day,
                    "transactions": n_d,
                    "frauds": f_d,
                    "selected": a_d,
                    "true_positives": tp_d,
                    "oracle_reviewable": min(k, f_d),
                    "overflow": max(a_d - k, 0),
                    "unused": max(k - a_d, 0),
                    "load_ratio": a_d / k,
                    "cutoff_tied_group": summary.cutoff_tied_group,
                }
            )
        transactions = int(truth.transactions.sum())
        frauds = int(truth.frauds.sum())
        selected = int(selected_by_day.sum())
        hits = int(hits_by_day.sum())
        counts = {
            "days": n_days,
            "transactions": transactions,
            "frauds": frauds,
            "selected": selected,
            "true_positives": hits,
            "false_positives": selected - hits,
            "false_negatives": frauds - hits,
        }
        ties = {
            "days": selection.cutoff_tie_days,
            "group_sizes": list(selection.cutoff_tie_group_sizes),
        }
        metrics = _r3_metrics(truth, capacity, reviewed & fraud, counts, daily)
    return R3Outcome(counts, daily, ties, metrics)


def _r3_metrics(
    truth: Truth,
    capacity: Capacity,
    hit_rows: npt.NDArray[np.bool_],
    counts: Mapping[str, int],
    daily: list[dict[str, Any]],
) -> dict[str, float | None]:
    frauds, selected, hits = counts["frauds"], counts["selected"], counts["true_positives"]
    achievable = sum(row["oracle_reviewable"] for row in daily)
    fraud_rows = truth.labels == 1
    try:
        fraud_amount = math.fsum(truth.amounts[fraud_rows].tolist())
        hit_amount = math.fsum(truth.amounts[hit_rows].tolist())
        missed_amount = math.fsum(truth.amounts[fraud_rows & ~hit_rows].tolist())
    except OverflowError as error:
        raise DataValidationError("The fraud amounts are too large to be totalled") from error
    prevalence = Fraction(frauds, counts["transactions"])
    return {
        "recall": hits / frauds,
        "precision": hits / selected,
        "frauds_per_1000_reviews": 1000 * hits / selected,
        "oracle_ceiling": achievable / frauds,
        "normalised_recall": hits / achievable,
        "random_review_recall": selected / counts["transactions"],
        "lift": hits * counts["transactions"] / (frauds * selected),
        "utilisation": selected / (capacity.k * counts["days"]),
        "capacity_to_prevalence": float(capacity.share / prevalence),
        "amount_weighted_recall": hit_amount / fraud_amount if fraud_amount > 0 else None,
        "fraud_value_exposed": missed_amount,
    }


def ranking_outcome(truth: Truth, scores: npt.NDArray[Any]) -> dict[str, float | None]:
    """Average Precision and ROC-AUC of the whole ranking, with equal scores kept as one group
    (scikit-learn merges equal thresholds): no tie-break enters either measure."""
    exact = np.asarray(scores)
    if exact.shape != (truth.rows,) or exact.dtype not in (np.float32, np.float64):
        raise DataValidationError(f"Scores must be a float32 or float64 array of {truth.rows} rows")
    exact = exact.astype(np.float64)
    if not np.isfinite(exact).all():
        raise DataValidationError("Scores must be finite for the ranking measures")
    if truth.labels.min() == 1:
        raise DataValidationError("ROC-AUC is undefined: the partition has no legitimate row")
    with log_step(logger, "Average Precision and ROC-AUC"):
        return {
            "average_precision": float(average_precision_score(truth.labels, exact)),
            "roc_auc": float(roc_auc_score(truth.labels, exact)),
        }


@dataclass(frozen=True)
class Subject:
    """Whose scores an ``evaluation`` record describes (its model, role, seed and score record)."""

    configuration: str
    model_role: str
    seed: int
    score: Mapping[str, str]
    partition: str
    period: str = "primary"


def _record(
    identity: Mapping[str, Any], subject: Subject, truth: Truth, policy: str
) -> dict[str, Any]:
    return {
        "record": "evaluation",
        "schema_version": RECORD_CONTRACTS["evaluation"].version,
        "identity": dict(identity),
        "configuration": subject.configuration,
        "model_role": subject.model_role,
        "seed": subject.seed,
        "score": dict(subject.score),
        "partition": subject.partition,
        "period": subject.period,
        "days": {"first": truth.first_day, "last": truth.last_day},
        "policy": policy,
        "capacity": None,
        "k": None,
        "threshold": None,
        "cost_ratio": None,
        "calibrator": None,
        "calibration": None,
        "counts": None,
        "metrics": {},
        "daily": [],
        "cutoff_ties": None,
    }


def r3_record(
    identity: Mapping[str, Any],
    subject: Subject,
    truth: Truth,
    capacity: Capacity,
    outcome: R3Outcome,
) -> dict[str, Any]:
    """The ``evaluation`` record of R3 at ``capacity`` (validated by its writer)."""
    record = _record(identity, subject, truth, "R3")
    record |= {
        "capacity": capacity.value,
        "k": capacity.k,
        "counts": outcome.counts,
        "metrics": outcome.metrics,
        "daily": outcome.daily,
        "cutoff_ties": outcome.cutoff_ties,
    }
    return record


def ranking_record(
    identity: Mapping[str, Any], subject: Subject, truth: Truth, metrics: Mapping[str, float | None]
) -> dict[str, Any]:
    """The ``evaluation`` record of the whole ranking (Average Precision and ROC-AUC)."""
    record = _record(identity, subject, truth, "ranking")
    record["metrics"] = dict(metrics)
    return record
