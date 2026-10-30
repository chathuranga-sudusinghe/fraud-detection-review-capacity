"""The confirmatory hypotheses H1, H2 and H3 from replicate matrices (DR-003, DR-005, DR-006,
DR-010; Issue #191).

Inputs are arrays: the observed Recall@2 % of the eight configurations in the declared order and
the replicate matrices of ``evaluation.resampling`` (a (replicates x configurations) matrix of
recalls for H1 and H2, one of alert rates for each H3 block length). No data is read. The H3
configuration is a parameter: its single source is the H3 selection record (DR-010).

* H1: the two-sided add-one p-value of delta_F, its 90 % interval against ±SESOI, and the
  descriptive contrasts with 95 % intervals.
* H2: the eight one-sided add-one p-values (two per model) and their maximum; the 90 % and 95 %
  intervals of the four weighting effects.
* H3: the one-sided p-values of the TOST against the open band (target - Delta, target + Delta)
  for the H3 configuration, the interval and its position, the block-length and tolerance
  sensitivities and t_min; the other seven configurations descriptively.
* ``decide``: Holm over {H1, H2, H3} with the adjusted p-values deciding.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from review_capacity.errors import DataValidationError
from review_capacity.evaluation.contrasts import Estimands
from review_capacity.evaluation.declaration import EvaluationDeclaration
from review_capacity.evaluation.inference import (
    H1Category,
    HolmResult,
    Interval,
    Position,
    PValue,
    h1_category,
    holm,
    interaction_splits_family_effect,
    largest,
    one_sided_p,
    percentile_interval,
    position_of,
    smallest_tolerance,
    two_sided_p_about_zero,
)
from review_capacity.evaluation.resampling import Replicates

type Floats = npt.NDArray[np.float64]


@dataclass(frozen=True)
class EstimandSummary:
    """A descriptive estimand: its observed value and 95 % percentile interval."""

    name: str
    estimate: float
    interval: Interval


@dataclass(frozen=True)
class H1Result:
    """H1: the family effect delta_F and the descriptive contrasts."""

    estimate: float
    p: PValue
    interval: Interval  # 90 %
    position: Position  # of the 90 % interval against (-SESOI, +SESOI)
    difference_interval: Interval  # 95 %
    within_family: tuple[EstimandSummary, ...]
    interaction: EstimandSummary
    interaction_splits: bool  # true: the family effect is also reported at each weighting
    by_weighting: tuple[EstimandSummary, ...]  # N_w - T_w for w = u, b (95 % intervals)
    cells: tuple[EstimandSummary, ...]


@dataclass(frozen=True)
class ModelEquivalence:
    """H2 for one model: the weighting effect delta_W,m and its two one-sided tests."""

    model: str
    estimate: float
    p_lower: PValue  # H0a: delta <= -SESOI
    p_upper: PValue  # H0b: delta >= +SESOI
    interval: Interval  # 90 %
    position: Position  # against (-SESOI, +SESOI)
    difference_interval: Interval  # 95 %, descriptive


@dataclass(frozen=True)
class H2Result:
    """H2: the intersection-union test over the models; ``p`` is the largest of the eight."""

    models: tuple[ModelEquivalence, ...]
    p: PValue


@dataclass(frozen=True)
class H3Variant:
    """The TOST of H3 for one block length."""

    block_length: int
    p_lower: PValue  # H0a: r <= target - Delta
    p_upper: PValue  # H0b: r >= target + Delta
    p: PValue  # the larger of the two
    interval: Interval  # 90 %
    position: Position  # against the open band (target - Delta, target + Delta)


@dataclass(frozen=True)
class ToleranceView:
    """The 90 % interval against a wider or narrower band (target (1 - t), target (1 + t))."""

    tolerance: float
    lower: float
    upper: float
    position: Position


@dataclass(frozen=True)
class ConfigurationRate:
    """The descriptive transfer of a configuration other than the H3 configuration."""

    configuration: str
    rate: float
    interval: Interval  # 90 %
    position: Position


@dataclass(frozen=True)
class H3Result:
    """H3 on the H3 configuration, with its sensitivities and the other seven descriptively."""

    configuration: str
    rate: float
    primary: H3Variant
    sensitivity: tuple[H3Variant, ...]  # the other block lengths, in the declared order
    tolerances: tuple[ToleranceView, ...]
    smallest_tolerance: float
    others: tuple[ConfigurationRate, ...]


@dataclass(frozen=True)
class Confirmatory:
    """The three hypotheses and the Holm decision (order H1, H2, H3)."""

    h1: H1Result
    h2: H2Result
    h3: H3Result
    holm: HolmResult
    h1_rejected: bool
    h2_equivalent: bool
    h3_preserved: bool
    h1_category: H1Category


def observed_values(values: npt.ArrayLike, what: str, count: int) -> Floats:
    """The observed values, one finite value per configuration."""
    observed = np.asarray(values, dtype=np.float64)
    if observed.shape != (count,) or not np.isfinite(observed).all():
        raise DataValidationError(f"{what} must be one finite value per configuration ({count})")
    return observed


def check_replicates(replicates: Replicates, what: str) -> None:
    """Reject infinities in a replicate matrix: the only undefined value is NaN."""
    if np.isinf(replicates.ratios).any():
        raise DataValidationError(
            f"{what} must not contain infinities; an undefined replicate is NaN"
        )


def _draws(
    estimands: Estimands, recalls: npt.ArrayLike, replicates: Replicates
) -> tuple[dict[str, int], Floats, Floats]:
    column = {name: index for index, name in enumerate(estimands.names)}
    observed = observed_values(recalls, "recalls", estimands.coefficients.shape[1])
    check_replicates(replicates, "The recall replicates")
    return column, estimands.apply(observed), estimands.apply(replicates.ratios)


def _summary(
    name: str, column: dict[str, int], points: Floats, draws: Floats, level: float
) -> EstimandSummary:
    index = column[name]
    return EstimandSummary(name, float(points[index]), percentile_interval(draws[:, index], level))


def analyse_h1(
    declaration: EvaluationDeclaration,
    estimands: Estimands,
    recalls: npt.ArrayLike,
    replicates: Replicates,
) -> H1Result:
    """H1 and its descriptive contrasts from the pigeonhole recall replicates."""
    column, points, draws = _draws(estimands, recalls, replicates)
    levels, sesoi = declaration.levels, declaration.inference.sesoi
    family = draws[:, column["family"]]
    interval = percentile_interval(family, levels.interpretation)
    interaction = _summary("interaction", column, points, draws, levels.difference)
    names = estimands.names
    return H1Result(
        estimate=float(points[column["family"]]),
        p=two_sided_p_about_zero(family),
        interval=interval,
        position=position_of(interval, -sesoi, sesoi),
        difference_interval=percentile_interval(family, levels.difference),
        within_family=tuple(
            _summary(name, column, points, draws, levels.difference)
            for name in names
            if name.startswith("within:")
        ),
        interaction=interaction,
        interaction_splits=interaction_splits_family_effect(
            interaction.interval, interaction.estimate, sesoi
        ),
        by_weighting=tuple(
            _summary(name, column, points, draws, levels.difference)
            for name in names
            if name.startswith("family:")
        ),
        cells=tuple(
            _summary(name, column, points, draws, levels.difference)
            for name in names
            if name.startswith("cell:")
        ),
    )


def analyse_h2(
    declaration: EvaluationDeclaration,
    estimands: Estimands,
    recalls: npt.ArrayLike,
    replicates: Replicates,
) -> H2Result:
    """H2 from the pigeonhole recall replicates: the two one-sided tests of each model's
    weighting effect against ±SESOI, and the maximum of the eight p-values."""
    column, points, draws = _draws(estimands, recalls, replicates)
    levels, sesoi = declaration.levels, declaration.inference.sesoi
    configurations = declaration.configurations
    models = []
    for model in (*configurations.tree, *configurations.neural):
        index = column[f"weighting:{model}"]
        effect = draws[:, index]
        interval = percentile_interval(effect, levels.interpretation)
        models.append(
            ModelEquivalence(
                model=model,
                estimate=float(points[index]),
                p_lower=one_sided_p(effect, -sesoi, tail="at_most"),
                p_upper=one_sided_p(effect, sesoi, tail="at_least"),
                interval=interval,
                position=position_of(interval, -sesoi, sesoi),
                difference_interval=percentile_interval(effect, levels.difference),
            )
        )
    return H2Result(tuple(models), largest([p for m in models for p in (m.p_lower, m.p_upper)]))


def _band(declaration: EvaluationDeclaration) -> tuple[float, float]:
    inference = declaration.inference
    return inference.h3_target - inference.h3_delta, inference.h3_target + inference.h3_delta


def _variant(declaration: EvaluationDeclaration, rates: Floats, block_length: int) -> H3Variant:
    low, high = _band(declaration)
    p_lower = one_sided_p(rates, low, tail="at_most")
    p_upper = one_sided_p(rates, high, tail="at_least")
    interval = percentile_interval(rates, declaration.levels.interpretation)
    return H3Variant(
        block_length,
        p_lower,
        p_upper,
        largest([p_lower, p_upper]),
        interval,
        position_of(interval, low, high),
    )


def _rate(
    declaration: EvaluationDeclaration,
    configuration: str,
    rate: float,
    replicates: Replicates,
    position: int,
) -> ConfigurationRate:
    low, high = _band(declaration)
    interval = percentile_interval(
        replicates.ratios[:, position], declaration.levels.interpretation
    )
    return ConfigurationRate(configuration, rate, interval, position_of(interval, low, high))


def analyse_h3(
    declaration: EvaluationDeclaration,
    configuration: str,
    rates: npt.ArrayLike,
    replicate_sets: Mapping[int, Replicates],
) -> H3Result:
    """H3 on ``configuration`` (the H3 configuration of the selection record) from the observed
    Test alert rates of the eight configurations and, for each block length, the day-block
    replicates of those rates. The other configurations are reported descriptively."""
    order = declaration.configurations.order
    if configuration not in order:
        raise DataValidationError(f"{configuration!r} is not one of the configurations {order}")
    resampling = declaration.resampling
    primary_length = resampling.block_length
    lengths = (primary_length, *resampling.block_length_sensitivity)
    if set(replicate_sets) != set(lengths):
        raise DataValidationError(
            f"H3 needs the replicate sets of the block lengths {list(lengths)}, "
            f"not {sorted(replicate_sets)}"
        )
    observed = observed_values(rates, "rates", len(order))
    for length, replicates in replicate_sets.items():
        check_replicates(replicates, f"The replicates for block length {length}")
        if replicates.ratios.ndim != 2 or replicates.ratios.shape[1] != len(order):
            raise DataValidationError(
                f"The replicates for block length {length} need {len(order)} columns"
            )
    index = order.index(configuration)
    variants = {
        length: _variant(declaration, replicate_sets[length].ratios[:, index], length)
        for length in lengths
    }
    primary = variants[primary_length]
    inference = declaration.inference
    others = tuple(
        _rate(
            declaration, name, float(observed[position]), replicate_sets[primary_length], position
        )
        for position, name in enumerate(order)
        if position != index
    )
    return H3Result(
        configuration=configuration,
        rate=float(observed[index]),
        primary=primary,
        sensitivity=tuple(variants[length] for length in resampling.block_length_sensitivity),
        tolerances=tuple(
            ToleranceView(
                tolerance,
                inference.h3_target * (1 - tolerance),
                inference.h3_target * (1 + tolerance),
                position_of(
                    primary.interval,
                    inference.h3_target * (1 - tolerance),
                    inference.h3_target * (1 + tolerance),
                ),
            )
            for tolerance in inference.h3_tolerance_sensitivity
        ),
        smallest_tolerance=smallest_tolerance(primary.interval, inference.h3_target),
        others=others,
    )


def h3_interval_label(position: Position) -> str:
    """The descriptive label of the position of the H3 interval (DR-003). It never says that the
    rate is preserved: that is said only when the Holm-adjusted p-value rejects H3."""
    return {
        Position.INSIDE: "CI inside the band",
        Position.BEYOND: "CI entirely outside the band",
        Position.CROSSING: "CI overlapping a bound (inconclusive)",
    }[position]


def decide(
    declaration: EvaluationDeclaration, h1: H1Result, h2: H2Result, h3: H3Result
) -> Confirmatory:
    """Holm over {H1, H2, H3} at the declared level: the adjusted p-values decide, and the
    interval positions only describe."""
    result = holm((h1.p.value, h2.p.value, h3.primary.p.value), declaration.inference.alpha)
    rejected_h1, rejected_h2, rejected_h3 = result.rejected
    return Confirmatory(
        h1=h1,
        h2=h2,
        h3=h3,
        holm=result,
        h1_rejected=rejected_h1,
        h2_equivalent=rejected_h2,
        h3_preserved=rejected_h3,
        h1_category=h1_category(rejected_h1, h1.position),
    )
