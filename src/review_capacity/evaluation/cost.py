"""R4 and the cost analysis of RQ3 (protocol Cost Model and Decision Policies; DR-007; Issue #158).

**R4** reviews, on each relative day, the transactions of R3's top-K queue whose calibrated
probability reaches the cost threshold,

$$
\\mathrm{R4}_d(\\rho) = \\lbrace i \\in \\mathrm{R3}_d : \\hat p_i \\ge p^* \\rbrace, \\qquad
p^* = \\frac{1}{\\rho}
$$

(the float ``1.0 / rho`` is compared, so a probability equal to the threshold is reviewed). The
queue order is R3's (raw score and the shared tie rule); R4 leaves the unreviewed slots unused.

**Cost analysis** of one configuration, capacity and calibrator over the evaluated days. With
``N`` the transactions in R3's queues, ``F`` the frauds of the partition, ``u(rho)`` the unused
reviews (R3's queue members that R4 declines) and ``f(rho)`` the frauds among them,

$$
J_\\pi(\\rho) = N_\\pi + \\rho\\,\\mathrm{FN}_\\pi, \\qquad
\\Delta J(\\rho) = J_{\\mathrm{R4}} - J_{\\mathrm{R3}} = \\rho f(\\rho) - u(\\rho)
$$

A queue member with probability ``q`` is reviewed once ``rho ≥ 1/q`` (never when ``q = 0``), so
``u`` and ``f`` are step functions of rho and ΔJ is linear between the finitely many breakpoints
``1/q``: its sign is decided exactly, segment by segment, with no interpolation (DR-007 §4).

* **Saturation ratio** rho_sat = 1 / (smallest probability in R3's queues); infinite when a queue
  member has probability 0 (then R4 never fills every day).
* **Break-even ratio** rho_BE = sup{rho ∈ (1, 1000] : ΔJ(rho) < 0}, the right end of the last
  interval on which ΔJ is negative. Classified as *coincidence* (rho_BE = rho_sat), *crossing*
  (rho_BE < rho_sat), *none* (ΔJ ≥ 0 on the range) or *right-censored* (rho_sat > 1000 and
  ΔJ(1000) < 0; then no ratio is reported). Every interval of negative ΔJ is listed, and the
  points where ΔJ is exactly 0 with unused reviews (ties).
* **Grid table** for rho in {2, 5, 10, 20, 50, 100, 200, 500, 1000}: u, f, the declined-set
  diagnostic (the fraud rate f/u next to the mean calibrated probability of the declined reviews
  and p*), the costs of R3, R4 and of no review (rho F), ΔJ and ΔJ / J_R3, the fixed-policy
  break-even rho_fixed = u / f (infinite when f = 0 < u; identical when u = 0), utilisation, the
  days with unused capacity, and the amount proxies (fraud value exposed, forgone, forgone per
  unused review, amount-weighted recall).
* **C_FP sensitivity:** rho = (1 + φ) rho_BE − φ for φ = C_FP / C_R in {1, 5}.

Amounts are exposure proxies, never losses or savings (DR-007 §6).
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

from review_capacity.development.records import RQ3_FRICTION, RQ3_RATIO_GRID, RQ3_RATIO_MAX
from review_capacity.errors import DataValidationError
from review_capacity.evaluation.queue import Selection
from review_capacity.evaluation.ranking import Capacity, Subject, Truth, evaluation_record
from review_capacity.logging_setup import log_step

logger = logging.getLogger(__name__)

RATIO_GRID = RQ3_RATIO_GRID  # DR-007 §2
RATIO_MAX = float(RQ3_RATIO_MAX)
FRICTION = RQ3_FRICTION  # phi = C_FP / C_R, DR-007 §1

type Floats = npt.NDArray[np.float64]
type Bools = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class R4Outcome:
    """R4 at one capacity and ratio, in the shapes of the ``evaluation`` record."""

    counts: dict[str, int]
    daily: list[dict[str, Any]]
    cutoff_ties: dict[str, Any]
    metrics: dict[str, float | None]


def _check(truth: Truth, selection: Selection, probabilities: Floats) -> None:
    if selection.mask.shape != (truth.rows,) or probabilities.shape != (truth.rows,):
        raise DataValidationError("The queue, the probabilities and the labels need the same rows")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise DataValidationError("Calibrated probabilities must be finite and within [0, 1]")


def reviewed(selection: Selection, probabilities: Floats, ratio: float) -> Bools:
    """R4's reviews at ``ratio``: the R3 queue members whose probability is at least 1 / ratio."""
    if not ratio > 1:
        raise DataValidationError(f"The cost ratio must exceed 1, not {ratio!r}")
    mask: Bools = selection.mask & (probabilities >= 1.0 / ratio)
    return mask


def _amount(values: Floats) -> float:
    try:
        return math.fsum(values.tolist())
    except OverflowError as error:
        raise DataValidationError("The fraud amounts are too large to be totalled") from error


def r4_outcome(
    truth: Truth,
    selection: Selection,
    probabilities: Floats,
    capacity: Capacity,
    ratio: float,
) -> R4Outcome:
    """R4 at ``capacity`` and ``ratio``: counts, one row per day, the R3 cutoff ties and the
    measures of the protocol's Outcomes (the scenario cost J_R4 included)."""
    _check(truth, selection, probabilities)
    k, n_days = capacity.k, len(truth.day_values)
    with log_step(logger, f"R4 at {capacity.value:.0%} (K = {k}), rho = {ratio:g}"):
        mask = reviewed(selection, probabilities, ratio)
        fraud = truth.labels == 1
        selected_by_day = np.bincount(truth.day_index[mask], minlength=n_days)
        hits_by_day = np.bincount(truth.day_index[mask & fraud], minlength=n_days)
        daily = []
        for i, summary in enumerate(selection.days):
            n_d, f_d = int(truth.transactions[i]), int(truth.frauds[i])
            a_d, tp_d = int(selected_by_day[i]), int(hits_by_day[i])
            if summary.day != int(truth.day_values[i]):
                raise DataValidationError(f"The queue and the labels disagree on day {summary.day}")
            daily.append(
                {
                    "day": summary.day,
                    "transactions": n_d,
                    "frauds": f_d,
                    "selected": a_d,
                    "true_positives": tp_d,
                    "oracle_reviewable": min(k, f_d),
                    "overflow": 0,
                    "unused": max(k - a_d, 0),
                    "load_ratio": a_d / k,
                    "cutoff_tied_group": summary.cutoff_tied_group,
                }
            )
        frauds, selected = int(truth.frauds.sum()), int(selected_by_day.sum())
        hits = int(hits_by_day.sum())
        counts = {
            "days": n_days,
            "transactions": int(truth.transactions.sum()),
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
        metrics = _metrics(truth, capacity, mask & fraud, counts, daily, ratio)
    return R4Outcome(counts, daily, ties, metrics)


def r4_record(
    identity: Mapping[str, Any],
    subject: Subject,
    truth: Truth,
    capacity: Capacity,
    ratio: float,
    calibrator: str,
    calibration: Mapping[str, str],
    outcome: R4Outcome,
) -> dict[str, Any]:
    """The ``evaluation`` record of R4 at ``capacity`` and ``ratio`` (validated by its writer):
    the threshold is p* = 1 / ratio, and the record links the calibration it used."""
    record = evaluation_record(identity, subject, truth, "R4")
    record |= {
        "capacity": capacity.value,
        "k": capacity.k,
        "threshold": 1.0 / ratio,
        "cost_ratio": float(ratio),
        "calibrator": calibrator,
        "calibration": dict(calibration),
        "counts": outcome.counts,
        "metrics": outcome.metrics,
        "daily": outcome.daily,
        "cutoff_ties": outcome.cutoff_ties,
    }
    return record


def _metrics(
    truth: Truth,
    capacity: Capacity,
    hit_rows: Bools,
    counts: dict[str, int],
    daily: list[dict[str, Any]],
    ratio: float,
) -> dict[str, float | None]:
    frauds, selected, hits = counts["frauds"], counts["selected"], counts["true_positives"]
    achievable = sum(row["oracle_reviewable"] for row in daily)
    fraud_rows = truth.labels == 1
    fraud_amount = _amount(truth.amounts[fraud_rows])
    hit_amount = _amount(truth.amounts[hit_rows])
    missed_amount = _amount(truth.amounts[fraud_rows & ~hit_rows])
    prevalence = Fraction(frauds, counts["transactions"])
    return {
        "recall": hits / frauds,
        "precision": hits / selected if selected else None,
        "frauds_per_1000_reviews": 1000 * hits / selected if selected else None,
        "oracle_ceiling": achievable / frauds,
        "normalised_recall": hits / achievable,
        "random_review_recall": selected / counts["transactions"],
        "lift": hits * counts["transactions"] / (frauds * selected) if selected else None,
        "utilisation": selected / (capacity.k * counts["days"]),
        "capacity_to_prevalence": float(capacity.share / prevalence),
        "amount_weighted_recall": hit_amount / fraud_amount if fraud_amount > 0 else None,
        "fraud_value_exposed": missed_amount,
        "scenario_cost": selected + ratio * (frauds - hits),
    }


# --- The exact curve -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """An interval of ratios on which ΔJ is negative: from ``start`` (closed, or open for the
    lower end 1 of the range) to ``end`` (open, or closed at the range end 1000)."""

    start: float
    end: float
    end_closed: bool

    def as_record(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "end_closed": self.end_closed}


@dataclass(frozen=True)
class Curve:
    """The exact sign analysis of ΔJ over (1, 1000]."""

    saturation: float | None  # None: infinite
    intervals: tuple[Interval, ...]
    zero_crossings: tuple[float, ...]

    @property
    def break_even(self) -> float | None:
        if not self.intervals or self.intervals[-1].end_closed:
            return None
        return self.intervals[-1].end

    @property
    def classification(self) -> str:
        if not self.intervals:
            return "none"
        if self.intervals[-1].end_closed and (
            self.saturation is None or self.saturation > RATIO_MAX
        ):
            return "right_censored"
        return "coincidence" if self.break_even == self.saturation else "crossing"


def exact_curve(probabilities: Floats, labels: npt.NDArray[np.int8]) -> Curve:
    """The sign of ΔJ(rho) = rho f − u over (1, 1000] for the R3 queue members with calibrated
    ``probabilities`` and fraud ``labels``, exact between the breakpoints 1 / probability."""
    if probabilities.shape != labels.shape or probabilities.ndim != 1 or len(probabilities) == 0:
        raise DataValidationError("The queue members need one probability and one label each")
    members = len(probabilities)
    frauds = int(labels.sum())
    order = np.argsort(-probabilities, kind="stable")  # descending: breakpoints ascending
    ranked, ranked_labels = probabilities[order], labels[order]
    ranked_frauds = np.cumsum(ranked_labels)
    positive = int((ranked > 0).sum())
    last = np.empty(0, dtype=np.intp)  # the last rank of each group of equal positive values
    if positive:
        last = np.concatenate((np.flatnonzero(np.diff(ranked[:positive])), [positive - 1]))
    levels = ranked[last]
    breakpoints = [1.0 / float(level) for level in levels]
    declined = [members, *(members - (last + 1)).tolist()]
    declined_frauds = [frauds, *(frauds - ranked_frauds[last]).tolist()]
    # Segment j (0 .. m): ratios in [lo_j, hi_j) with u = declined[j] and f = declined_frauds[j];
    # segment 0 is the open range (1, b_1).
    starts = [1.0, *breakpoints]
    ends = [*breakpoints, math.inf]
    negative: list[Interval] = []
    ties: list[float] = []
    for j, (lo, hi) in enumerate(zip(starts, ends, strict=True)):
        u, f = declined[j], declined_frauds[j]
        if u == 0 or hi <= 1.0 or lo > RATIO_MAX:
            continue
        root = u / f if f else math.inf
        if f and lo <= root < hi and 1.0 < root <= RATIO_MAX:
            ties.append(root)
        end = min(hi, root)
        if end > RATIO_MAX:
            if lo <= RATIO_MAX:
                negative.append(Interval(lo, RATIO_MAX, True))
        elif end > lo:
            negative.append(Interval(lo, end, False))
    merged: list[Interval] = []
    for interval in negative:
        if merged and not merged[-1].end_closed and merged[-1].end == interval.start:
            merged[-1] = Interval(merged[-1].start, interval.end, interval.end_closed)
        else:
            merged.append(interval)
    saturation = None if positive < members else 1.0 / float(levels[-1])
    return Curve(saturation, tuple(merged), tuple(ties))


# --- The analysis --------------------------------------------------------------------------------


def _fixed(unused: int, forgone: int) -> tuple[str, float | None]:
    if unused == 0:
        return "identical", None
    if forgone == 0:
        return "infinite", None
    return "finite", unused / forgone


def analyse(
    truth: Truth,
    selection: Selection,
    probabilities: Floats,
    capacity: Capacity,
) -> dict[str, Any]:
    """The cost analysis of R4 against R3 at ``capacity`` (the body of a ``cost_analysis``
    record): the queue totals, the exact curve and the grid table. ``selection`` is R3's queue."""
    _check(truth, selection, probabilities)
    k, n_days = capacity.k, len(truth.day_values)
    with log_step(logger, f"Cost analysis at {capacity.value:.0%} (K = {k})"):
        member = selection.mask
        fraud = truth.labels == 1
        q, y = probabilities[member], truth.labels[member]
        queue = int(member.sum())
        frauds_total = int(truth.frauds.sum())
        captured = int(y.sum())
        fn_r3 = frauds_total - captured
        fraud_amount = _amount(truth.amounts[fraud])
        exposed_r3 = _amount(truth.amounts[fraud & ~member])
        awr_r3 = _amount(truth.amounts[fraud & member]) / fraud_amount if fraud_amount > 0 else None
        curve = exact_curve(q, y)
        grid = []
        for ratio in RATIO_GRID:
            r4 = reviewed(selection, probabilities, float(ratio))
            declined = member & ~r4
            unused = int(declined.sum())
            forgone = int((declined & fraud).sum())
            per_day = np.bincount(truth.day_index[r4], minlength=n_days)
            forgone_amount = _amount(truth.amounts[declined & fraud])
            exposed_r4 = _amount(truth.amounts[fraud & ~r4])
            cost_r3 = queue + ratio * fn_r3
            cost_r4 = (queue - unused) + ratio * (fn_r3 + forgone)
            difference = cost_r4 - cost_r3
            status, fixed = _fixed(unused, forgone)
            grid.append(
                {
                    "ratio": float(ratio),
                    "threshold": 1.0 / ratio,
                    "reviews": queue - unused,
                    "unused": unused,
                    "frauds_forgone": forgone,
                    "days_with_unused": int((per_day < k).sum()),
                    "utilisation": (queue - unused) / (k * n_days),
                    "declined_fraud_rate": forgone / unused if unused else None,
                    "declined_mean_probability": (
                        float(probabilities[declined].mean()) if unused else None
                    ),
                    "cost_r3": cost_r3,
                    "cost_r4": cost_r4,
                    "cost_difference": difference,
                    "relative_difference": difference / cost_r3,
                    "cost_no_review": ratio * frauds_total,
                    "fixed_status": status,
                    "fixed_ratio": fixed,
                    "fraud_value_exposed_r3": exposed_r3,
                    "fraud_value_exposed_r4": exposed_r4,
                    "fraud_value_forgone": forgone_amount,
                    "forgone_per_unused": forgone_amount / unused if unused else None,
                    "amount_weighted_recall_r3": awr_r3,
                    "amount_weighted_recall_r4": (
                        _amount(truth.amounts[fraud & r4]) / fraud_amount
                        if fraud_amount > 0
                        else None
                    ),
                }
            )
    break_even = curve.break_even
    return {
        "queue": {"reviews": queue, "frauds": frauds_total, "captured": captured},
        "saturation": {"ratio": curve.saturation, "infinite": curve.saturation is None},
        "break_even": {
            "classification": curve.classification,
            "ratio": break_even,
            "single_interval": len(curve.intervals) <= 1,
            "intervals": [interval.as_record() for interval in curve.intervals],
            "zero_crossings": list(curve.zero_crossings),
        },
        "friction": [
            {
                "phi": float(phi),
                "ratio": None if break_even is None else (1 + phi) * break_even - phi,
            }
            for phi in FRICTION
        ],
        "grid": grid,
    }
