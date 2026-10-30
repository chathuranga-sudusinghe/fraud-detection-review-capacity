"""The pre-declared sensitivity analyses of H1 and H2 (DR-005, DR-006; Issues #183 and #191).

None of them is confirmatory: the decisions of ``hypotheses.decide`` use the primary pigeonhole
bootstrap only. Each sensitivity recomputes p_H1, p_H2 and the H1 practical category, re-applies
Holm with p_H3 held at its primary value, and reports whether a rejection decision or the H1
category differs from the primary one.

* S1 (Owen correction): SE^2 = V_pigeonhole - 2 V_naive with the sample variances (ddof = 1) of
  the same estimand over the pigeonhole and the fraud-level replicates; Wald intervals and Wald
  p-values. Not estimable when the corrected variance is not positive. Holm needs both p_H1 and
  p_H2, so the comparison is reported as not estimable when either is.
* S2: the same hypotheses on day-block recall replicates (non-circular, block length 3).
* SESOI views: the H1 category and the H2 interval positions against SESOIs of 1, 3 and 5 pp,
  descriptive only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from review_capacity.errors import DataValidationError
from review_capacity.evaluation.contrasts import Estimands
from review_capacity.evaluation.declaration import EvaluationDeclaration
from review_capacity.evaluation.hypotheses import (
    Confirmatory,
    H1Result,
    H2Result,
    analyse_h1,
    analyse_h2,
    check_replicates,
    observed_values,
)
from review_capacity.evaluation.inference import (
    H1Category,
    HolmResult,
    Interval,
    Position,
    h1_category,
    holm,
    position_of,
)
from review_capacity.evaluation.resampling import Replicates

type Floats = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Comparison:
    """The sensitivity decisions next to the primary ones; ``differs`` is true when a rejection
    decision for H1 or H2, or the H1 practical category, is not the primary one."""

    p_h1: float
    p_h2: float
    holm: HolmResult
    h1_rejected: bool
    h2_rejected: bool
    h1_category: H1Category
    differs: bool


def compare(
    declaration: EvaluationDeclaration,
    primary: Confirmatory,
    *,
    p_h1: float,
    p_h2: float,
    h1_position: Position,
) -> Comparison:
    """Holm over {p_h1, p_h2, the primary p_H3}, and the comparison with ``primary``."""
    result = holm((p_h1, p_h2, primary.h3.primary.p.value), declaration.inference.alpha)
    rejected_h1, rejected_h2, _ = result.rejected
    category = h1_category(rejected_h1, h1_position)
    return Comparison(
        p_h1=p_h1,
        p_h2=p_h2,
        holm=result,
        h1_rejected=rejected_h1,
        h2_rejected=rejected_h2,
        h1_category=category,
        differs=(
            rejected_h1 != primary.h1_rejected
            or rejected_h2 != primary.h2_equivalent
            or category != primary.h1_category
        ),
    )


@dataclass(frozen=True)
class S2Outcome:
    """S2: H1 and H2 on day-block replicates, and the comparison with the primary."""

    h1: H1Result
    h2: H2Result
    comparison: Comparison


def analyse_s2(
    declaration: EvaluationDeclaration,
    estimands: Estimands,
    recalls: npt.ArrayLike,
    replicates: Replicates,
    primary: Confirmatory,
) -> S2Outcome:
    """S2 from the day-block recall replicates (``resampling.day_ratios`` of the per-day captured
    frauds over the per-day frauds)."""
    h1 = analyse_h1(declaration, estimands, recalls, replicates)
    h2 = analyse_h2(declaration, estimands, recalls, replicates)
    comparison = compare(
        declaration, primary, p_h1=h1.p.value, p_h2=h2.p.value, h1_position=h1.position
    )
    return S2Outcome(h1, h2, comparison)


@dataclass(frozen=True)
class WaldQuantity:
    """An estimand with the two variances of S1."""

    name: str
    estimate: float
    v_pigeonhole: float
    v_naive: float

    @property
    def corrected_variance(self) -> float:
        return self.v_pigeonhole - 2 * self.v_naive

    @property
    def se(self) -> float | None:
        """The corrected standard error, or None when the corrected variance is not positive
        (S1 is then not estimable for this quantity)."""
        variance = self.corrected_variance
        return math.sqrt(variance) if variance > 0 else None


@dataclass(frozen=True)
class S1Outcome:
    """S1: the Wald quantities, the S1 p-values and intervals (None where not estimable) and the
    comparison with the primary (None unless both p_H1 and p_H2 are estimable)."""

    quantities: tuple[WaldQuantity, ...]  # delta_F, then the weighting effects in model order
    p_h1: float | None
    interval_h1: Interval | None  # 90 %
    position_h1: Position | None
    p_h2: float | None
    intervals_h2: tuple[Interval, ...] | None  # 90 %, in model order
    comparison: Comparison | None


def _variance(values: Floats, what: str) -> float:
    if np.isinf(values).any():
        raise DataValidationError(f"{what}: replicates must not contain infinities")
    defined = values[np.isfinite(values)]
    if defined.size < 2:
        raise DataValidationError(f"{what}: a variance needs at least two defined replicates")
    return float(np.var(defined, ddof=1))


def wald_interval(estimate: float, se: float, level: float) -> Interval:
    """``estimate`` ± z SE with the standard-normal quantile z of the two-sided ``level``."""
    z = float(norm.ppf(0.5 + level / 2))
    return Interval(estimate - z * se, estimate + z * se, level)


def analyse_s1(
    declaration: EvaluationDeclaration,
    estimands: Estimands,
    recalls: npt.ArrayLike,
    pigeonhole: Replicates,
    naive: Replicates,
    primary: Confirmatory,
) -> S1Outcome:
    """S1 from the pigeonhole and the fraud-level recall replicates of the same estimands."""
    points = estimands.apply(observed_values(recalls, "recalls", estimands.coefficients.shape[1]))
    check_replicates(pigeonhole, "The pigeonhole replicates")
    check_replicates(naive, "The fraud-level replicates")
    pigeonhole_draws = estimands.apply(pigeonhole.ratios)
    naive_draws = estimands.apply(naive.ratios)
    configurations = declaration.configurations
    models = (*configurations.tree, *configurations.neural)
    names = ("family", *(f"weighting:{model}" for model in models))
    quantities = tuple(
        WaldQuantity(
            name,
            float(points[estimands.names.index(name)]),
            _variance(pigeonhole_draws[:, estimands.names.index(name)], f"S1 {name}"),
            _variance(naive_draws[:, estimands.names.index(name)], f"S1 naive {name}"),
        )
        for name in names
    )
    levels, sesoi = declaration.levels, declaration.inference.sesoi
    family, weighting = quantities[0], quantities[1:]

    p_h1: float | None = None
    interval_h1: Interval | None = None
    position_h1: Position | None = None
    if family.se is not None:
        p_h1 = float(2 * norm.sf(abs(family.estimate) / family.se))
        interval_h1 = wald_interval(family.estimate, family.se, levels.interpretation)
        position_h1 = position_of(interval_h1, -sesoi, sesoi)

    p_h2: float | None = None
    intervals_h2: tuple[Interval, ...] | None = None
    if all(quantity.se is not None for quantity in weighting):
        components = []
        for quantity in weighting:
            se = quantity.se
            assert se is not None
            components.append(float(norm.sf((quantity.estimate + sesoi) / se)))  # H0a
            components.append(float(norm.cdf((quantity.estimate - sesoi) / se)))  # H0b
        p_h2 = max(components)
        intervals_h2 = tuple(
            wald_interval(quantity.estimate, quantity.se, levels.interpretation)
            for quantity in weighting
            if quantity.se is not None
        )

    comparison = (
        compare(declaration, primary, p_h1=p_h1, p_h2=p_h2, h1_position=position_h1)
        if p_h1 is not None and p_h2 is not None and position_h1 is not None
        else None
    )
    return S1Outcome(quantities, p_h1, interval_h1, position_h1, p_h2, intervals_h2, comparison)


@dataclass(frozen=True)
class SesoiView:
    """H1 and H2 read against another SESOI (descriptive; nothing is retested)."""

    sesoi: float
    h1_position: Position
    h1_category: H1Category
    h2_positions: tuple[Position, ...]  # the 90 % interval of each model's weighting effect


def sesoi_views(
    declaration: EvaluationDeclaration, confirmatory: Confirmatory
) -> tuple[SesoiView, ...]:
    """The H1 category and the H2 interval positions against each declared alternative SESOI
    (1, 3 and 5 pp). The H1 rejection is the primary Holm decision; no p-value is recomputed."""
    return tuple(
        SesoiView(
            sesoi=sesoi,
            h1_position=(position := position_of(confirmatory.h1.interval, -sesoi, sesoi)),
            h1_category=h1_category(confirmatory.h1_rejected, position),
            h2_positions=tuple(
                position_of(model.interval, -sesoi, sesoi) for model in confirmatory.h2.models
            ),
        )
        for sesoi in declaration.inference.sesoi_sensitivity
    )
