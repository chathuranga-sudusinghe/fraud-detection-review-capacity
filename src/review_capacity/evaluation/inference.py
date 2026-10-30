"""The building blocks of the confirmatory inference (DR-003, DR-005, DR-006; Issue #191).

Pure functions on replicate vectors (the output of ``evaluation.resampling``): add-one tail
p-values with their Monte Carlo standard errors, percentile intervals, the position of an interval
against open bounds, the H1 practical category, the Holm procedure, the smallest H3 tolerance and
the units of a recall difference. The rules were fixed before this code (Issues #183 and #185):

* A bootstrap p-value is ``(1 + count) / (B + 1)`` with B the number of replicates. A replicate
  without a defined value counts as not rejecting: it is added to every one-sided count, so B stays
  the denominator and the p-value can only grow (DR-006).
* Intervals are ``numpy.percentile`` (method ``linear``) over the defined replicates.
* Bounds are open: an interval with an endpoint exactly on a bound is neither inside nor beyond it
  but crosses it (DR-005, DR-003).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
import numpy.typing as npt

from review_capacity.errors import DataValidationError

type Floats = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PValue:
    """A bootstrap p-value and its Monte Carlo standard error (DR-006)."""

    value: float
    mc_se: float


@dataclass(frozen=True)
class Interval:
    """A percentile (or Wald) interval and the number of replicates it excludes as undefined."""

    lower: float
    upper: float
    level: float
    undefined: int = 0
    replicates: int = 0


class Position(StrEnum):
    """Where an interval lies against the open interval (low, high)."""

    INSIDE = "inside"  # entirely within: lower > low and upper < high
    BEYOND = "beyond"  # entirely outside: upper < low or lower > high
    CROSSING = "crossing"  # anything else, an endpoint exactly on a bound included


class H1Category(StrEnum):
    """The five pre-specified interpretations of H1 against ±SESOI (DR-005 §3)."""

    MEANINGFUL = "difference of practically meaningful size"
    UNCERTAIN_SIZE = "difference of uncertain practical size"
    NEGLIGIBLE = "difference too small to matter in practice"
    COMPATIBLE_NEGLIGIBLE = "data compatible only with practically negligible differences"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class HolmResult:
    """Holm-adjusted p-values and the rejections, in the order of the input."""

    adjusted: tuple[float, ...]
    rejected: tuple[bool, ...]


@dataclass(frozen=True)
class EffectUnits:
    """A recall difference in the units of the protocol (Statistical Plan, effect sizes)."""

    percentage_points: float
    frauds_per_partition: float
    frauds_per_day: float
    precision_at_k_points: float
    normalised_recall: float


def _vector(values: npt.ArrayLike, what: str) -> Floats:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size < 1:
        raise DataValidationError(f"{what} must be a non-empty vector of replicates")
    if np.isinf(vector).any():
        raise DataValidationError(
            f"{what} must not contain infinities; an undefined replicate is NaN"
        )
    return vector


def _standard_error(proportion: float, replicates: int) -> float:
    return math.sqrt(proportion * (1 - proportion) / replicates)


def one_sided_p(
    values: npt.ArrayLike, bound: float, *, tail: Literal["at_most", "at_least"]
) -> PValue:
    """The add-one p-value ``(1 + #{value <= bound}) / (B + 1)`` (``at_most``) or
    ``(1 + #{value >= bound}) / (B + 1)`` (``at_least``), with every undefined replicate (NaN)
    added to the count."""
    if not math.isfinite(bound):
        raise DataValidationError(f"The bound must be finite, not {bound}")
    draws = _vector(values, "values")
    replicates = int(draws.size)
    hits = draws <= bound if tail == "at_most" else draws >= bound
    count = int(hits.sum()) + int(np.isnan(draws).sum())
    value = (1 + count) / (replicates + 1)
    return PValue(value, _standard_error(value, replicates))


def two_sided_p_about_zero(values: npt.ArrayLike) -> PValue:
    """The two-sided add-one p-value of H1: ``min(1, 2 min(q_lower, q_upper))`` with the tail
    proportions ``q`` of ``#{value <= 0}`` and ``#{value >= 0}``. Its Monte Carlo standard error
    is twice that of the smaller tail (before the cap at 1)."""
    lower = one_sided_p(values, 0.0, tail="at_most")
    upper = one_sided_p(values, 0.0, tail="at_least")
    smaller = lower if lower.value <= upper.value else upper
    return PValue(min(1.0, 2 * smaller.value), 2 * smaller.mc_se)


def largest(components: Sequence[PValue]) -> PValue:
    """The maximum of one-sided p-values (the intersection-union p-value), with the standard
    error of the component that attains it."""
    if not components:
        raise DataValidationError("The intersection-union p-value needs at least one component")
    return max(components, key=lambda component: component.value)


def percentile_interval(values: npt.ArrayLike, level: float) -> Interval:
    """The ``level`` percentile interval (``numpy.percentile``, method ``linear``) of the defined
    replicates; ``DataValidationError`` if none is defined."""
    draws = _vector(values, "values")
    if not 0 < level < 1:
        raise DataValidationError(f"The interval level must lie in (0, 1), not {level}")
    defined = draws[np.isfinite(draws)]
    if defined.size == 0:
        raise DataValidationError("Every bootstrap replicate was undefined")
    tail = 100 * (1 - level) / 2
    lower, upper = np.percentile(defined, [tail, 100 - tail], method="linear")
    return Interval(
        float(lower), float(upper), level, int(draws.size - defined.size), int(draws.size)
    )


def position_of(interval: Interval, low: float, high: float) -> Position:
    """The position of ``interval`` against the open interval (``low``, ``high``)."""
    if interval.lower > low and interval.upper < high:
        return Position.INSIDE
    if interval.upper < low or interval.lower > high:
        return Position.BEYOND
    return Position.CROSSING


def h1_category(significant: bool, position: Position) -> H1Category:
    """The practical interpretation of H1 (DR-005 §3): ``significant`` is the Holm-adjusted
    rejection of H1; ``position`` is that of its 90 % interval against (-SESOI, +SESOI)."""
    if significant:
        return {
            Position.BEYOND: H1Category.MEANINGFUL,
            Position.CROSSING: H1Category.UNCERTAIN_SIZE,
            Position.INSIDE: H1Category.NEGLIGIBLE,
        }[position]
    return (
        H1Category.COMPATIBLE_NEGLIGIBLE if position is Position.INSIDE else H1Category.INCONCLUSIVE
    )


def interaction_splits_family_effect(interval: Interval, estimate: float, sesoi: float) -> bool:
    """The qualification rule of DR-005 §3: the 95 % interval of the family x weighting interaction
    excludes 0 and its point estimate exceeds the SESOI in magnitude, so the family effect is also
    reported for each weighting condition."""
    excludes_zero = interval.lower > 0 or interval.upper < 0
    return excludes_zero and abs(estimate) > sesoi


def holm(p_values: Sequence[float], alpha: float) -> HolmResult:
    """Holm's step-down procedure: the adjusted p-value of the k-th smallest is the running
    maximum of ``min(1, (m - i + 1) p_(i))`` over i <= k (m the number of hypotheses); a hypothesis
    is rejected if and only if its adjusted p-value is at most ``alpha``."""
    count = len(p_values)
    if count == 0 or not 0 < alpha < 1:
        raise DataValidationError("Holm needs p-values and a level in (0, 1)")
    if any(not 0 <= p <= 1 for p in p_values):
        raise DataValidationError(f"Every p-value must lie in [0, 1]: {list(p_values)}")
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return HolmResult(tuple(adjusted), tuple(value <= alpha for value in adjusted))


def smallest_tolerance(interval: Interval, target: float) -> float:
    """The smallest relative tolerance t whose open band (target (1 - t), target (1 + t)) contains
    ``interval`` for every larger t (DR-003): ``max(|L - target|, |U - target|) / target``. It is
    descriptive, not a test, and not Holm-adjusted."""
    if target <= 0:
        raise DataValidationError("The target rate must be positive")
    return max(abs(interval.lower - target), abs(interval.upper - target)) / target


def effect_units(
    delta: float, *, frauds: int, days: int, k: int, oracle_ceiling: float
) -> EffectUnits:
    """The units of a Recall@2 % difference ``delta`` on a partition of ``days`` days with
    ``frauds`` frauds, a fixed daily budget ``k`` and the oracle recall ceiling: percentage points;
    frauds per partition (delta F) and per day; the Precision@K equivalent delta F / (days k) in
    percentage points; and the normalised recall delta / ceiling."""
    if frauds < 1 or days < 1 or k < 1 or not 0 < oracle_ceiling <= 1:
        raise DataValidationError("The effect units need positive counts and a ceiling in (0, 1]")
    return EffectUnits(
        percentage_points=100 * delta,
        frauds_per_partition=delta * frauds,
        frauds_per_day=delta * frauds / days,
        precision_at_k_points=100 * delta * frauds / (days * k),
        normalised_recall=delta / oracle_ceiling,
    )
