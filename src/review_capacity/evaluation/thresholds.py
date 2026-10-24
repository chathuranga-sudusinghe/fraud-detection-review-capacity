"""The thresholds of the static policies R0, R1 and R2 (protocol Decision Policies; DR-003;
DR-009; Issue #157).

* **R0** flags every transaction whose raw score is at least 0.5.
* **R1** flags every transaction whose raw score is at least the F1-optimal threshold of the
  selection set (V2, or its label-latency fitting days). The candidates are the distinct scores
  of the set, including the smallest; transactions with equal scores are never separated. With
  P the frauds of the set, and TP_a and A_a the frauds and the transactions with score at least
  a, the F1 of a candidate is 2 TP_a / (P + A_a). Two candidates a and b are compared through the
  integer cross-products TP_a (P + A_b) and TP_b (P + A_a), computed with Python integers, so
  nothing rounds or overflows; equal products are an exact tie, and the highest tied threshold
  (the fewest alerts) is selected.
* **R2** flags every transaction whose raw score is at least the score ranked ceil(c x N) in
  descending order among the N transactions of V2 (DR-003, "Threshold ties"), for each capacity
  share c. It uses scores only.

A selection that cannot be made (an empty set, no fraud, a non-finite score) is returned as a
recorded error, never raised, so that the caller can write the ``r1_selection`` record first.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import numpy.typing as npt

from review_capacity.errors import DataValidationError
from review_capacity.logging_setup import log_step
from review_capacity.models.base import Labels

logger = logging.getLogger(__name__)

R0_THRESHOLD = 0.5
R1_RULE_VERSION = "1"  # DR-009, R1 threshold selection (Issue #115)

type FloatScores = npt.NDArray[np.float32] | npt.NDArray[np.float64]


def _floats(scores: object) -> npt.NDArray[np.float64]:
    """The scores widened to float64 (exact for float32), with ``-0.0`` equal to ``0.0``."""
    if not isinstance(scores, np.ndarray) or scores.ndim != 1:
        raise DataValidationError("Scores must be a one-dimensional array")
    if scores.dtype not in (np.float32, np.float64):
        raise DataValidationError(
            f"Scores must be floating point (float32 or float64), not {scores.dtype}"
        )
    return scores.astype(np.float64) + 0.0


@dataclass(frozen=True)
class R1Selection:
    """The outcome of one R1 threshold selection, in the shape of the ``r1_selection`` record.

    ``error`` is set, and every selected quantity is ``None``, when no threshold could be chosen.
    """

    transactions: int
    frauds: int
    error: str | None = None
    distinct_scores: int | None = None
    threshold: float | None = None
    true_positives: int | None = None
    false_positives: int | None = None
    false_negatives: int | None = None
    alerts: int | None = None

    @property
    def status(self) -> str:
        return "error" if self.error is not None else "selected"

    @property
    def f1_numerator(self) -> int | None:
        """2 TP at the selected threshold."""
        return None if self.true_positives is None else 2 * self.true_positives

    @property
    def f1_denominator(self) -> int | None:
        """P + A at the selected threshold (= 2 TP + FP + FN)."""
        return None if self.alerts is None else self.frauds + self.alerts


def select_r1(scores: FloatScores, labels: Labels) -> R1Selection:
    """The R1 threshold of the selection set whose ``scores`` and ``labels`` are given."""
    if not isinstance(labels, np.ndarray) or labels.dtype != np.int8 or labels.ndim != 1:
        raise DataValidationError("Labels must be a one-dimensional int8 array")
    if not np.isin(labels, (0, 1)).all():
        raise DataValidationError("Labels must be 0 or 1")
    values = _floats(scores)
    if values.shape != labels.shape:
        raise DataValidationError(
            f"{len(values)} scores for {len(labels)} labels: they must describe the same rows"
        )
    rows, frauds = len(values), int(labels.sum())
    if rows == 0:
        return R1Selection(0, 0, "empty_selection_set")
    if not np.isfinite(values).all():
        return R1Selection(rows, frauds, "non_finite_scores")
    if frauds == 0:
        return R1Selection(rows, 0, "no_fraud")
    with log_step(logger, f"Selecting the R1 threshold on {rows} transactions"):
        distinct, inverse = np.unique(values, return_inverse=True)
        per_score = np.bincount(inverse, minlength=len(distinct))
        frauds_per_score = np.bincount(inverse, weights=labels, minlength=len(distinct))
        # Alerts and true positives at "score >= distinct[j]": the suffix sums, exact integers.
        alerts_at = np.cumsum(per_score[::-1])[::-1].tolist()
        hits_at = np.cumsum(frauds_per_score.astype(np.int64)[::-1])[::-1].tolist()
        best = len(distinct) - 1  # the highest candidate first; only a strictly larger F1 replaces
        for j in range(len(distinct) - 2, -1, -1):
            if hits_at[j] * (frauds + alerts_at[best]) > hits_at[best] * (frauds + alerts_at[j]):
                best = j
    alerts, hits = int(alerts_at[best]), int(hits_at[best])
    return R1Selection(
        transactions=rows,
        frauds=frauds,
        distinct_scores=len(distinct),
        threshold=float(distinct[best]),
        true_positives=hits,
        false_positives=alerts - hits,
        false_negatives=frauds - hits,
        alerts=alerts,
    )


@dataclass(frozen=True)
class R2Threshold:
    """The R2 threshold of capacity share c: the score ranked ``rank`` = ceil(c N) of N."""

    share: Fraction
    threshold: float
    rank: int
    transactions: int

    @property
    def value(self) -> float:
        return float(self.share)


def select_r2(scores: FloatScores, share: Fraction) -> R2Threshold:
    """The R2 threshold of ``share`` on the (V2) ``scores``; labels are not used."""
    values = _floats(scores)
    if values.size == 0:
        raise DataValidationError("The R2 threshold needs at least one score")
    if not np.isfinite(values).all():
        raise DataValidationError("Scores must be finite for the R2 threshold")
    if not 0 < share <= 1:
        raise DataValidationError(f"The capacity share must lie in (0, 1], not {share}")
    rank = math.ceil(share * values.size)  # exact: a Fraction times an integer
    threshold = float(np.sort(values)[::-1][rank - 1])
    return R2Threshold(share, threshold, rank, int(values.size))


def alert_mask(scores: FloatScores, threshold: float) -> npt.NDArray[np.bool_]:
    """The alerts of a static threshold policy: ``score >= threshold`` (DR-003, DR-009)."""
    return _floats(scores) >= threshold


def describe(selection: R1Selection) -> dict[str, Any]:
    """The selected quantities as the fields of an ``r1_selection`` record."""
    return {
        "transactions": selection.transactions,
        "frauds": selection.frauds,
        "distinct_scores": selection.distinct_scores,
        "threshold": selection.threshold,
        "true_positives": selection.true_positives,
        "false_positives": selection.false_positives,
        "false_negatives": selection.false_negatives,
        "alerts": selection.alerts,
        "f1_numerator": selection.f1_numerator,
        "f1_denominator": selection.f1_denominator,
        "status": selection.status,
        "error": selection.error,
    }
