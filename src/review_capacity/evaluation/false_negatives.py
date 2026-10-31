"""The false-negative analysis of the Test period (Issue #201; protocol, False-Negative Analysis).

Exploratory, descriptive and non-causal. A false negative is a fraud outside the daily top-K of R3
at the primary capacity; frauds among threshold alerts in excess of K are not (DR-009). For each
configuration and each pre-declared slice the analysis reports the fraud count, the misses, the
miss rate, the share of the total false negatives and, for a slice with at least
``minimum_frauds`` Test frauds, the percentile interval of the miss rate

$$
\\text{miss rate} = \\frac{\\sum_i w_i \\, m_i \\, s_i}{\\sum_i w_i \\, s_i}
$$

where:

- `i` = a Test fraud;
- `w_i` = its weight in a two-way pigeonhole replicate (relative day × fine pseudo-entity, the
  stream ``pigeonhole_false_negative``; one joint weight set for all slices and configurations);
- `m_i` = 1 when the fraud is a false negative of the configuration, else 0;
- `s_i` = 1 when the fraud belongs to the slice, else 0.

A replicate with zero weight in the slice is undefined: it is left out of the interval and counted
(DR-006).

The **score band** (the within-day rank of the missed fraud, relative to K) is defined on the missed
frauds only, so its miss rate is 100 % by construction: it reports the fraud count and the share of
the false negatives, with no miss rate and no interval.

The slice boundaries are fixed without Test labels: the amount edges from Train amounts, the pooled
levels from V2 frauds (a level with fewer than ``minimum_frauds`` V2 frauds, and a Test level never
seen in V2, is ``other``), the weeks from the declaration, and the recurring pseudo-entities from
Train frauds. Nothing here reads a file: the inputs are arrays.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt

from review_capacity.errors import DataValidationError, SpecificationError
from review_capacity.evaluation.declaration import EvaluationDeclaration
from review_capacity.evaluation.outcomes import Partition, interval_record
from review_capacity.evaluation.ranking import Capacity
from review_capacity.evaluation.resampling import Positions, pigeonhole_ratios, stream_generator
from review_capacity.logging_setup import log_step

logger = logging.getLogger(__name__)

OTHER = "other"
POOLED_VARIABLES = ("ProductCD", "card4", "card6", "DeviceType")
VARIABLES = (
    "identity",
    *POOLED_VARIABLES,
    "amount_band",
    "week",
    "entity",
    "score_band",
)
IDENTITY_LEVELS = ("absent", "present")
ENTITY_LEVELS = ("new", "recurring")
STATUSES = ("interval", "fewer_than_minimum_frauds", "not_applicable")


def week_label(week: tuple[int, int]) -> str:
    return f"{week[0]}-{week[1]}"


def declared_levels(declaration: EvaluationDeclaration) -> dict[str, tuple[str, ...]]:
    """The levels, in order, of every variable whose levels the declaration fixes (all but the
    pooled categorical variables, whose kept levels come from V2)."""
    rule = declaration.false_negative
    multiples = rule.score_band_multiples
    return {
        "identity": IDENTITY_LEVELS,
        "amount_band": tuple(f"band {i + 1}" for i in range(rule.amount_quantiles)),
        "week": tuple(week_label(week) for week in rule.test_weeks),
        "entity": ENTITY_LEVELS,
        "score_band": tuple(_band_name(i, multiples) for i in range(len(multiples))),
    }


@dataclass(frozen=True)
class Boundaries:
    """The label-free slice boundaries: the Train amount edges and, for each pooled variable, the
    levels kept with their V2 fraud counts."""

    amount_edges: tuple[float, ...]
    kept: Mapping[str, Mapping[str, int]]


@dataclass(frozen=True)
class SliceVariable:
    """One slicing variable: a level code (an index into ``levels``) for every Test fraud."""

    name: str
    levels: tuple[str, ...]
    codes: Positions


@dataclass(frozen=True)
class Misses:
    """The false negatives of one configuration over the Test frauds (in row order): whether each
    fraud is outside the daily top-K and its 0-based within-day rank under the configuration."""

    missed: npt.NDArray[np.bool_]
    day_rank: Positions


def pooling_from_v2(values: npt.NDArray[np.str_], minimum: int) -> dict[str, int]:
    """The levels of the V2 frauds ``values`` with at least ``minimum`` of them, with their counts,
    the most frequent first (ties by name); every other level is pooled into ``other``."""
    counts = Counter(values.tolist())
    kept = {level: n for level, n in counts.items() if n >= minimum}
    if OTHER in kept:
        raise DataValidationError(f"A level named {OTHER!r} would collide with the pooled level")
    return dict(sorted(kept.items(), key=lambda item: (-item[1], item[0])))


def compute_boundaries(
    declaration: EvaluationDeclaration,
    train_amounts: npt.NDArray[np.float64],
    v2_values: Mapping[str, npt.NDArray[np.str_]],
) -> Boundaries:
    """The label-free boundaries from the development data: the Train amount edges and the levels
    of each pooled variable kept from the V2 frauds ``v2_values`` (Issue #211)."""
    rule = declaration.false_negative
    boundaries = Boundaries(
        amount_edges(train_amounts, rule.amount_quantiles),
        {name: pooling_from_v2(v2_values[name], rule.minimum_frauds) for name in POOLED_VARIABLES},
    )
    check_boundaries(boundaries, declaration)
    return boundaries


def check_frozen(boundaries: Boundaries, declaration: EvaluationDeclaration) -> None:
    """The boundaries computed from the development data equal the ones frozen in the
    declaration; a declaration without frozen boundaries is refused."""
    frozen = declaration.false_negative.boundaries
    if frozen is None:
        raise SpecificationError(
            "false_negative.boundaries is not frozen in config/evaluation.yaml: the slice "
            "boundaries must be declared before the Test evaluation"
        )
    if tuple(boundaries.amount_edges) != tuple(frozen.amount_edges):
        raise SpecificationError(
            f"The Train amount edges {list(boundaries.amount_edges)} differ from the frozen "
            f"edges {list(frozen.amount_edges)}"
        )
    computed = {name: list(levels.items()) for name, levels in boundaries.kept.items()}
    declared = {name: list(levels.items()) for name, levels in frozen.kept.items()}
    if computed != declared:
        raise SpecificationError("The V2 pooled levels differ from the frozen levels")


def require_frozen(declaration: EvaluationDeclaration) -> None:
    """Refuse a declaration whose slice boundaries are not frozen."""
    if declaration.false_negative.boundaries is None:
        raise SpecificationError(
            "false_negative.boundaries is not frozen in config/evaluation.yaml: the slice "
            "boundaries must be declared before the Test evaluation"
        )


def amount_edges(train_amounts: npt.NDArray[np.float64], quantiles: int) -> tuple[float, ...]:
    """The interior edges of the ``quantiles`` equal-share bands of the Train amounts
    (``numpy.quantile``, method ``linear``): four edges for quintiles."""
    if quantiles < 2 or train_amounts.ndim != 1 or train_amounts.size == 0:
        raise DataValidationError("Amount bands need at least two bands and some Train amounts")
    if not np.isfinite(train_amounts).all():
        raise DataValidationError("Train amounts must be finite")
    levels = np.arange(1, quantiles) / quantiles
    return tuple(float(x) for x in np.quantile(train_amounts, levels, method="linear"))


def amount_codes(amounts: npt.NDArray[np.float64], edges: Sequence[float]) -> Positions:
    """The band of every amount: a value equal to an edge belongs to the band above it."""
    return np.searchsorted(np.asarray(edges), amounts, side="right").astype(np.intp)


def week_codes(days: npt.NDArray[np.int64], weeks: Sequence[tuple[int, int]]) -> Positions:
    """The index of the declared week containing each relative day."""
    codes = np.full(days.shape, -1, dtype=np.intp)
    for index, (first, last) in enumerate(weeks):
        codes[(days >= first) & (days <= last)] = index
    if (codes < 0).any():
        raise DataValidationError("Every Test day must lie in one of the declared weeks")
    return codes


def score_band_codes(day_rank: Positions, k: int, multiples: Sequence[int]) -> Positions:
    """The band of each missed fraud from its 0-based within-day rank: ranks K + 1 … m_2 K, …
    and above the last multiple (for K, 2K, 5K: near miss, 2K + 1 … 5K, above 5K)."""
    edges = np.asarray(multiples[1:]) * k
    return np.searchsorted(edges, day_rank, side="right").astype(np.intp)


def pooled_variable(
    name: str, values: npt.NDArray[np.str_], boundaries: Boundaries
) -> SliceVariable:
    """The Test frauds' levels of a pooled variable: a kept level keeps its index, every other
    value (a level pooled in V2 or never seen in V2) is ``other``, the last level."""
    kept = tuple(boundaries.kept[name])
    index = {level: i for i, level in enumerate(kept)}
    codes = np.fromiter((index.get(v, len(kept)) for v in values.tolist()), dtype=np.intp)
    return SliceVariable(name, (*kept, OTHER), codes)


def misses_of(partition: Partition, scores: npt.NDArray[Any], capacity: Capacity) -> Misses:
    """The false negatives of R3 at ``capacity`` over the Test frauds, in row order."""
    selection = partition.queue.select(scores, capacity.k)
    fraud = partition.truth.labels == 1
    return Misses(~selection.mask[fraud], selection.day_rank[fraud].astype(np.intp))


def _check_variable(variable: SliceVariable, frauds: int) -> None:
    if variable.codes.shape != (frauds,) or not variable.levels:
        raise DataValidationError(
            f"{variable.name}: one level code for each of the {frauds} frauds"
        )
    if int(variable.codes.min()) < 0 or int(variable.codes.max()) >= len(variable.levels):
        raise DataValidationError(f"{variable.name}: a level code lies outside its levels")


def check_boundaries(boundaries: Boundaries, declaration: EvaluationDeclaration) -> None:
    """The edges are increasing and as many as the declared bands need; every kept level has the
    minimum number of V2 frauds."""
    rule = declaration.false_negative
    edges = boundaries.amount_edges
    if len(edges) != rule.amount_quantiles - 1 or any(b <= a for a, b in pairwise(edges)):
        raise DataValidationError(
            f"Amount bands need {rule.amount_quantiles - 1} increasing edges, not {list(edges)}"
        )
    if set(boundaries.kept) != set(POOLED_VARIABLES):
        raise DataValidationError(f"Kept levels are needed for exactly {list(POOLED_VARIABLES)}")
    for name, levels in boundaries.kept.items():
        small = [level for level, n in levels.items() if n < rule.minimum_frauds or level == OTHER]
        if small:
            raise DataValidationError(f"{name}: levels {small} are not kept by the pooling rule")


def false_negative_analysis(
    declaration: EvaluationDeclaration,
    partition: Partition,
    variables: Sequence[SliceVariable],
    misses: Mapping[str, Misses],
    capacity: Capacity,
) -> dict[str, dict[str, Any]]:
    """The slice tables of every configuration in ``misses`` at ``capacity``.

    ``variables`` holds the level code of every Test fraud for each slicing variable except the
    score band (``VARIABLES`` order); the score band follows from each configuration's ranks."""
    truth = partition.truth
    frauds = int(truth.frauds.sum())
    days = len(truth.day_values)
    if [v.name for v in variables] != [n for n in VARIABLES if n != "score_band"]:
        raise DataValidationError(f"The variables must be {list(VARIABLES[:-1])}, in that order")
    for variable in variables:
        _check_variable(variable, frauds)
    for name, found in misses.items():
        if found.missed.shape != (frauds,) or found.day_rank.shape != (frauds,):
            raise DataValidationError(
                f"{name}: the misses need one entry for each of the {frauds} frauds"
            )
    with log_step(logger, f"False-negative analysis: {len(misses)} configurations"):
        masks = {
            (variable.name, level): variable.codes == index
            for variable in variables
            for index, level in enumerate(variable.levels)
        }
        keys = list(masks)
        slice_matrix = np.vstack([masks[key] for key in keys]).astype(np.float64)
        rows: list[npt.NDArray[np.float64]] = []
        denominators: list[npt.NDArray[np.float64]] = []
        owner: list[tuple[str, tuple[str, str]]] = []
        for name, found in misses.items():
            rows.append(slice_matrix * found.missed.astype(np.float64))
            denominators.append(slice_matrix)
            owner.extend((name, key) for key in keys)
        replicates = pigeonhole_ratios(
            stream_generator(declaration, "pigeonhole_false_negative"),
            np.vstack(rows),
            partition.fraud_day_index,
            days,
            partition.entity_index,
            replicates=declaration.resampling.replicates,
            denominators=np.vstack(denominators),
        )
        column = {item: i for i, item in enumerate(owner)}
        return {
            name: _table(
                declaration, found, variables, masks, replicates.ratios, column, name, capacity
            )
            for name, found in misses.items()
        }


def _table(
    declaration: EvaluationDeclaration,
    found: Misses,
    variables: Sequence[SliceVariable],
    masks: Mapping[tuple[str, str], npt.NDArray[np.bool_]],
    ratios: npt.NDArray[np.float64],
    column: Mapping[tuple[str, tuple[str, str]], int],
    name: str,
    capacity: Capacity,
) -> dict[str, Any]:
    rule = declaration.false_negative
    total = int(found.missed.sum())
    frauds = int(found.missed.shape[0])
    slices: list[dict[str, Any]] = []
    for variable in variables:
        for level in variable.levels:
            mask = masks[(variable.name, level)]
            count, missed = int(mask.sum()), int((mask & found.missed).sum())
            enough = count >= rule.minimum_frauds
            slices.append(
                {
                    "variable": variable.name,
                    "level": level,
                    "frauds": count,
                    "misses": missed,
                    "miss_rate": missed / count if count else None,
                    "share": missed / total if total else None,
                    "status": "interval" if enough else "fewer_than_minimum_frauds",
                    "interval": (
                        interval_record(
                            ratios[:, column[(name, (variable.name, level))]], rule.interval_level
                        )
                        if enough
                        else None
                    ),
                }
            )
    band = score_band_codes(found.day_rank[found.missed], capacity.k, rule.score_band_multiples)
    for index in range(len(rule.score_band_multiples)):
        in_band = int((band == index).sum())
        slices.append(
            {
                "variable": "score_band",
                "level": declared_levels(declaration)["score_band"][index],
                "frauds": in_band,
                "misses": in_band,
                "miss_rate": None,
                "share": in_band / total if total else None,
                "status": "not_applicable",
                "interval": None,
            }
        )
    return {
        "capacity": capacity.value,
        "k": capacity.k,
        "frauds": frauds,
        "false_negatives": total,
        "slices": slices,
    }


def _band_name(index: int, multiples: Sequence[int]) -> str:
    if index == len(multiples) - 1:
        return f"above {multiples[-1]}K"
    first = "K" if multiples[index] == 1 else f"{multiples[index]}K"
    return f"{first}+1 to {multiples[index + 1]}K"
