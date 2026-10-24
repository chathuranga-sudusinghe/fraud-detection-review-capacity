"""The two-way (pigeonhole) paired bootstrap of the fraud-capture difference (DR-006; Issue #156).

Inputs are fixed capture indicators: for each fraud i, c_{a,i} = 1 if configuration a's daily
top-K review captures it, and c_{b,i} likewise for b. The weights depend only on each fraud's
relative day and fine pseudo-entity, never on model outputs, and are shared by both models, so
the replicates are paired. In each replicate

$$
\\begin{aligned}
(a_1, \\dots, a_D) &\\sim \\operatorname{Multinomial}(D;\\ 1/D, \\dots, 1/D) \\\\
(b_1, \\dots, b_G) &\\sim \\operatorname{Multinomial}(G;\\ 1/G, \\dots, 1/G) \\\\
w_i &= a_{d(i)}\\, b_{e(i)}, \\qquad
\\delta^* = \\frac{\\sum_i w_i\\, (c_{a,i} - c_{b,i})}{\\sum_i w_i}
\\end{aligned}
$$

where D is the number of relative days of the partition (all of them, also days without a
fraud), G the number of distinct fraud pseudo-entities, and the day and entity multiplicities are
drawn independently. The top-K ranking is not recomputed (DR-006, "Daily top-K"). The interval is
the percentile interval of the replicates (``numpy.percentile``, method ``linear``). A replicate
whose weights sum to zero has no recall (it happens when the drawn days and the drawn entities
share no fraud, for example when every fraud is on one day that the replicate does not draw); it is
counted and left out of the interval, and reported rather than hidden.

The draws come from ``numpy.random.default_rng(seed)`` (PCG64): all day multiplicities first,
then the entity multiplicities in blocks of ``BLOCK`` replicates, so that a result depends on the
seed and the inputs only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from review_capacity.errors import DataValidationError
from review_capacity.logging_setup import ProgressLogger, log_step

logger = logging.getLogger(__name__)

REPLICATES = 10_000  # DR-006: B = 10,000
SEED = 42  # fixed and recorded (Issue #156)
LEVEL = 0.95  # DR-006: 95 % for difference estimates
BLOCK = 500  # replicates drawn per block; bounds the memory of the entity draws


@dataclass(frozen=True)
class PairedInterval:
    """The point estimate and the percentile interval of a paired capture difference."""

    estimate: float
    lower: float
    upper: float
    level: float
    replicates: int
    undefined_replicates: int
    seed: int
    days: int
    entities: int
    frauds: int


def _check(
    difference: npt.NDArray[np.int8],
    day_index: npt.NDArray[np.intp],
    entity_index: npt.NDArray[np.intp],
) -> None:
    frauds = len(difference)
    if frauds == 0:
        raise DataValidationError("The paired bootstrap needs at least one fraud")
    if day_index.shape != (frauds,) or entity_index.shape != (frauds,):
        raise DataValidationError("Every fraud needs one difference, one day and one entity")
    if not np.isin(difference, (-1, 0, 1)).all():
        raise DataValidationError("Capture differences must be -1, 0 or 1")


def paired_difference(
    capture_a: npt.NDArray[np.bool_],
    capture_b: npt.NDArray[np.bool_],
    day_index: npt.NDArray[np.intp],
    days: int,
    entity_index: npt.NDArray[np.intp],
    *,
    replicates: int = REPLICATES,
    seed: int = SEED,
    level: float = LEVEL,
) -> PairedInterval:
    """The paired bootstrap of recall(a) - recall(b) over the frauds of one partition.

    ``capture_a`` and ``capture_b`` mark, for each fraud, whether the daily top-K review of
    a and of b captures it; ``day_index`` (0 .. ``days`` - 1) and ``entity_index``
    (0 .. G - 1, G the distinct entities) say which day and entity each fraud belongs to.
    """
    if capture_a.dtype != np.bool_ or capture_b.dtype != np.bool_:
        raise DataValidationError("Capture indicators must be boolean arrays")
    if capture_a.shape != capture_b.shape:
        raise DataValidationError("Both models need one capture indicator per fraud")
    difference = capture_a.astype(np.int8) - capture_b.astype(np.int8)
    _check(difference, day_index, entity_index)
    if replicates < 1 or not 0 < level < 1:
        raise DataValidationError(
            "The bootstrap needs a positive replicate count and a level in (0, 1)"
        )
    if days < 1 or int(day_index.max()) >= days or int(day_index.min()) < 0:
        raise DataValidationError("Every fraud's day must be one of the partition's days")
    entities = int(entity_index.max()) + 1
    if int(entity_index.min()) < 0 or len(np.unique(entity_index)) != entities:
        raise DataValidationError("Entity indices must be 0 .. G - 1 without gaps")
    rng = np.random.default_rng(seed)
    with log_step(
        logger, f"Paired bootstrap: {replicates} replicates over {len(difference)} frauds"
    ):
        day_draws = rng.multinomial(days, np.full(days, 1 / days), size=replicates)
        values = np.empty(replicates, dtype=np.float64)
        progress = ProgressLogger(logger, "bootstrap blocks", -(-replicates // BLOCK))
        for start in range(0, replicates, BLOCK):
            stop = min(start + BLOCK, replicates)
            entity_draws = rng.multinomial(
                entities, np.full(entities, 1 / entities), size=stop - start
            )
            weights = day_draws[start:stop][:, day_index] * entity_draws[:, entity_index]
            total = weights.sum(axis=1)
            numerator = weights @ difference.astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                values[start:stop] = np.where(
                    total > 0, numerator / np.where(total > 0, total, 1), np.nan
                )
            progress.advance()
    defined = values[np.isfinite(values)]
    if len(defined) == 0:
        raise DataValidationError("Every bootstrap replicate had zero total weight")
    tail = 100 * (1 - level) / 2
    lower, upper = np.percentile(defined, [tail, 100 - tail], method="linear")
    return PairedInterval(
        estimate=float(difference.mean()),
        lower=float(lower),
        upper=float(upper),
        level=level,
        replicates=replicates,
        undefined_replicates=int(replicates - len(defined)),
        seed=seed,
        days=days,
        entities=entities,
        frauds=len(difference),
    )


def entity_indices(keys: list[str]) -> npt.NDArray[np.intp]:
    """Entity numbers 0 .. G - 1 for the string keys of the frauds, in order of first appearance."""
    numbering: dict[str, int] = {}
    return np.array([numbering.setdefault(key, len(numbering)) for key in keys], dtype=np.intp)
