"""Daily-budget fidelity of RQ2 (DR-003, DR-009; Issue #197).

For each evaluated relative day d, with n_d the threshold-generated alerts, N_d the day's
transactions, K the fixed daily budget of a capacity scenario c, and N̄_V2 the mean daily volume
of the V2 reference (the volume that fixed K), the day's deviation from the budget is n_d - K and
the decomposition of its logarithm, for a day with an alert, is

$$
\\ln\\frac{n_d}{K} = \\underbrace{\\ln\\frac{n_d / N_d}{c}}_{\\text{rate}}
 + \\underbrace{\\ln\\frac{N_d}{\\bar N_{V2}}}_{\\text{volume}}
 + \\underbrace{\\ln\\frac{c \\, \\bar N_{V2}}{K}}_{\\text{budget}}
$$

The budget term is the rounding of K (zero when K = c N̄_V2). A day without an alert has no
logarithm: it is counted, left out of the logarithmic means, and its whole deviation is the
rate component (DR-003). The statistics are point values (protocol, Statistical Plan).

Nothing here reads a file: the inputs are the daily table of an uncapped threshold outcome.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from review_capacity.errors import DataValidationError


def decomposition_terms(
    alerts: int, transactions: int, capacity: float, k: int, reference_volume: float
) -> dict[str, float]:
    """The rate, volume and budget terms and their total ln(n_d / K) for a day with an alert."""
    if alerts < 1 or transactions < 1 or capacity <= 0 or k < 1 or reference_volume <= 0:
        raise DataValidationError("The decomposition needs an alert, a volume, a budget and a rate")
    return {
        "rate": math.log(alerts / transactions / capacity),
        "volume": math.log(transactions / reference_volume),
        "budget": math.log(capacity * reference_volume / k),
        "total": math.log(alerts / k),
    }


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def fidelity(
    daily: Sequence[Mapping[str, Any]], capacity: float, k: int, reference_volume: float
) -> dict[str, Any]:
    """The fidelity statistics of the days in ``daily`` (the ``selected`` alerts and
    ``transactions`` of each) against the budget ``k`` of capacity ``capacity``, and the
    decomposition against the V2 reference volume ``reference_volume``."""
    if not daily:
        raise DataValidationError("Fidelity needs at least one day")
    alerts = [int(row["selected"]) for row in daily]
    deviation = [n - k for n in alerts]
    days = len(daily)
    terms: list[dict[str, Any]] = []
    for row, n_d in zip(daily, alerts, strict=True):
        part = (
            dict.fromkeys(("rate", "volume", "budget", "total"))
            if n_d == 0
            else decomposition_terms(n_d, int(row["transactions"]), capacity, k, reference_volume)
        )
        terms.append(
            {
                "day": int(row["day"]),
                "alerts": n_d,
                "transactions": int(row["transactions"]),
                **part,
            }
        )
    alert_days = [t for t in terms if t["total"] is not None]
    return {
        "days": days,
        "mean_alerts": math.fsum(alerts) / days,
        "mean_deviation": math.fsum(deviation) / days,
        "mean_absolute_deviation": math.fsum(abs(d) for d in deviation) / days,
        "max_absolute_deviation": max(abs(d) for d in deviation),
        "mean_relative_deviation": math.fsum(deviation) / days / k,
        "days_above_budget": sum(d > 0 for d in deviation),
        "days_below_budget": sum(d < 0 for d in deviation),
        "share_above_budget": sum(d > 0 for d in deviation) / days,
        "share_below_budget": sum(d < 0 for d in deviation) / days,
        "total_overflow": sum(max(d, 0) for d in deviation),
        "total_unused": sum(max(-d, 0) for d in deviation),
        "mean_load_ratio": math.fsum(n / k for n in alerts) / days,
        "zero_alert_days": days - len(alert_days),
        "reference_volume": reference_volume,
        "mean_rate": _mean([t["rate"] for t in alert_days]),
        "mean_volume": _mean([t["volume"] for t in alert_days]),
        "mean_budget": _mean([t["budget"] for t in alert_days]),
        "mean_total": _mean([t["total"] for t in alert_days]),
        "terms": terms,
    }
