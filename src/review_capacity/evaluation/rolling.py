"""The outcome of one origin of the rolling-origin supplement (protocol, Experimental Split
Structure; DR-001; DR-003; Issue #159).

At an origin the primary configuration scores a **threshold window** and the **transfer window**
that follows it. The R2 threshold of the capacity share c is the score ranked ceil(c N) among the N
transactions of the threshold window (scores only, no labels; ``evaluation.thresholds.select_r2``)
and is applied unchanged to the transfer window: a transaction is an alert when its score is at
least the threshold. The fixed daily review budget of the origin is

$$
K = \\operatorname{round}\\left(c \\cdot \\bar{n}\\right)
$$

with n-bar the mean daily volume of the threshold window, rounded half up (DR-002 §3). For each
transfer day d the realised alert rate is r_d = n_d / N_d (alerts n_d, transactions N_d). The
pooled rate r is the alerts of the transfer window over its transactions. The overflow, the unused
capacity and the rate / volume decomposition are those of DR-003 and DR-009, with n-bar of the
threshold window in place of the mean daily volume of V2.

The outcome describes the transfer; it is descriptive only and never enters a selection.
Nothing here reads a file: scores and relative days are arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import numpy.typing as npt

from review_capacity.errors import DataValidationError
from review_capacity.evaluation.thresholds import FloatScores, alert_mask, select_r2

# The H3 equivalence band of DR-003: the capacity share of 2 % within the tolerance of 0.4
# percentage points. The supplement counts the origins whose pooled rate lies strictly inside it.
BAND_CENTRE = 0.02
BAND_TOLERANCE = 0.004

type Days = npt.NDArray[np.int64]


def daily_budget_of(share: Fraction, transactions: int, days: int) -> int:
    """K = round(share x mean daily volume), rounded half up (DR-002 §3)."""
    return math.floor(share * transactions / days + Fraction(1, 2))


@dataclass(frozen=True)
class OriginOutcome:
    """The result of one origin: the threshold, the budget and the transfer, day by day."""

    share: Fraction
    threshold: float
    rank: int
    window_transactions: int
    window_alerts: int
    k: int
    daily: list[dict[str, Any]]
    transfer_transactions: int
    transfer_alerts: int
    zero_alert_days: int
    rate_component: float | None
    volume_component: float | None
    total_component: float | None

    @property
    def window_rate(self) -> float:
        """The rate of the threshold window itself (above the share when scores tie)."""
        return self.window_alerts / self.window_transactions

    @property
    def transfer_rate(self) -> float:
        """The pooled alert rate of the transfer window."""
        return self.transfer_alerts / self.transfer_transactions

    @property
    def relative_deviation(self) -> float:
        """r / c - 1: the pooled transfer rate against the capacity share."""
        return self.transfer_rate / float(self.share) - 1

    @property
    def overflow(self) -> int:
        return sum(day["overflow"] for day in self.daily)

    @property
    def unused(self) -> int:
        return sum(day["unused"] for day in self.daily)

    @property
    def days_above(self) -> int:
        return sum(day["overflow"] > 0 for day in self.daily)

    @property
    def days_below(self) -> int:
        return sum(day["unused"] > 0 for day in self.daily)

    @property
    def inside_band(self) -> bool:
        """The pooled rate lies strictly inside the H3 band (0.016, 0.024)."""
        return abs(self.transfer_rate - BAND_CENTRE) < BAND_TOLERANCE - 1e-12


def _check_window(days: Days, scores: FloatScores, first: int, last: int, what: str) -> None:
    if days.ndim != 1 or days.dtype != np.int64 or days.size == 0:
        raise DataValidationError(f"{what}: the days are a non-empty one-dimensional int64 array")
    if np.shape(scores) != days.shape:
        raise DataValidationError(f"{what}: {len(days)} rows need {len(days)} scores")
    if days.min() < first or days.max() > last:
        raise DataValidationError(f"{what}: relative days must lie in {first}..{last}")
    present = np.unique(days)
    if len(present) != last - first + 1:
        raise DataValidationError(f"{what}: every relative day {first}..{last} needs a transaction")


def origin_outcome(
    share: Fraction,
    window_days: Days,
    window_scores: FloatScores,
    window: tuple[int, int],
    transfer_days: Days,
    transfer_scores: FloatScores,
    transfer: tuple[int, int],
) -> OriginOutcome:
    """The R2 threshold set on the window and its transfer to the next days; ``window`` and
    ``transfer`` are the first and last relative day of each."""
    if transfer[0] != window[1] + 1:
        raise DataValidationError("The transfer window starts the day after the threshold window")
    _check_window(window_days, window_scores, *window, "threshold window")
    _check_window(transfer_days, transfer_scores, *transfer, "transfer window")
    selected = select_r2(window_scores, share)
    window_transactions = len(window_days)
    window_alerts = int(alert_mask(window_scores, selected.threshold).sum())
    n_days = window[1] - window[0] + 1
    k = daily_budget_of(share, window_transactions, n_days)
    mean_volume = window_transactions / n_days
    flagged = alert_mask(transfer_scores, selected.threshold)
    values, index = np.unique(transfer_days, return_inverse=True)
    volumes = np.bincount(index, minlength=len(values))
    alerts = np.bincount(index[flagged], minlength=len(values))
    daily: list[dict[str, Any]] = []
    rates, volume_terms, totals = [], [], []
    for position, day in enumerate(values.tolist()):
        n_d, a_d = int(volumes[position]), int(alerts[position])
        daily.append(
            {
                "day": day,
                "transactions": n_d,
                "selected": a_d,
                "overflow": max(a_d - k, 0),
                "unused": max(k - a_d, 0),
                "load_ratio": a_d / k,
            }
        )
        if a_d:
            rates.append(math.log(a_d / n_d / float(share)))
            volume_terms.append(math.log(n_d / mean_volume))
            totals.append(math.log(a_d / k))

    def mean(terms: list[float]) -> float | None:
        return sum(terms) / len(terms) if terms else None

    return OriginOutcome(
        share=share,
        threshold=selected.threshold,
        rank=selected.rank,
        window_transactions=window_transactions,
        window_alerts=window_alerts,
        k=k,
        daily=daily,
        transfer_transactions=int(volumes.sum()),
        transfer_alerts=int(alerts.sum()),
        zero_alert_days=sum(day["selected"] == 0 for day in daily),
        rate_component=mean(rates),
        volume_component=mean(volume_terms),
        total_component=mean(totals),
    )
