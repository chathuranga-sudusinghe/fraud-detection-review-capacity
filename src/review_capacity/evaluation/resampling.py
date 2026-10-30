"""The resampling engines of the confirmatory inference (DR-006, DR-003; Issue #189).

Pure functions on arrays: no data is read and no statistic is computed beyond the replicate
ratios. Intervals, p-values, Monte Carlo standard errors, the S1 correction and Holm belong to the
inference (Issue #183 fixed their rules).

* ``pigeonhole_ratios``: the two-way bootstrap of H1 and H2. Day multiplicities (Multinomial over
  the D days) and entity multiplicities (Multinomial over the G fine pseudo-entities) are drawn
  independently, and a fraud's weight is the product of its day's and its entity's multiplicity.
  The same weights serve every row, so all configurations are compared on the same resampled
  frauds. The top-K ranking is not recomputed.
* ``naive_ratios``: the fraud-level iid bootstrap (multiplicities Multinomial over the F frauds),
  the V_naive of S1.
* ``day_block_indices``: the non-circular moving-block sampler of DR-003: block starts uniform over
  0 .. D - l, ceil(D / l) blocks concatenated, truncated to the first D days. ``day_ratios``
  turns per-day numerators and denominators into replicate ratios over such an index set.

Draw conventions (DR-006, Resampling conventions): every engine takes a generator, so that a
stream restarts identically on each call (``stream_generator``). The pigeonhole engine draws the
day multiplicities of all replicates first and then the entity multiplicities replicate by
replicate, which is also the order of ``bootstrap.paired_difference``; ``BLOCK`` only bounds the
memory and never changes a result.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from review_capacity.errors import DataValidationError, SpecificationError
from review_capacity.evaluation.declaration import EvaluationDeclaration
from review_capacity.logging_setup import ProgressLogger, log_step

logger = logging.getLogger(__name__)

BLOCK = 500  # replicates held in memory at once; bounds memory, never changes a result

type Floats = npt.NDArray[np.float64]
type Positions = npt.NDArray[np.intp]


def stream_generator(declaration: EvaluationDeclaration, name: str) -> np.random.Generator:
    """The generator of the child stream ``name``, at the start of that stream (DR-006): the
    children of ``SeedSequence(root_seed).spawn`` in the declared order, with the declared bit
    generator (PCG64). Every call returns a fresh generator, so a stream restarts identically."""
    streams = declaration.resampling.streams
    if name not in streams:
        raise SpecificationError(
            f"Unknown resampling stream {name!r}; expected one of {list(streams)}"
        )
    children = np.random.SeedSequence(declaration.resampling.root_seed).spawn(len(streams))
    return np.random.Generator(np.random.PCG64(children[streams.index(name)]))


@dataclass(frozen=True, eq=False)
class Replicates:
    """Replicate ratios: one row per replicate and one column per quantity. A ratio is NaN where
    the denominator of its replicate is zero."""

    ratios: Floats

    @property
    def replicates(self) -> int:
        return int(self.ratios.shape[0])

    @property
    def undefined(self) -> npt.NDArray[np.bool_]:
        """The replicates with at least one undefined ratio."""
        return np.isnan(self.ratios).any(axis=1)

    @property
    def undefined_count(self) -> int:
        return int(self.undefined.sum())


@dataclass(frozen=True, eq=False)
class DayBlocks:
    """One set of day-block resamples: the day positions of each replicate."""

    block_length: int
    indices: Positions  # (replicates, days)


def _matrix(values: npt.ArrayLike, what: str, columns: int, *, nonnegative: bool = False) -> Floats:
    """``values`` as a finite float matrix with one row per quantity and ``columns`` columns."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] != columns:
        raise DataValidationError(
            f"{what} must have one row per quantity and {columns} columns, not shape {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise DataValidationError(f"{what} must be finite")
    if nonnegative and (matrix < 0).any():
        raise DataValidationError(f"{what} must not be negative")
    return matrix


def _replicate_count(replicates: int) -> int:
    if replicates < 1:
        raise DataValidationError("The bootstrap needs at least one replicate")
    return replicates


def _ratios(
    blocks: Iterator[Floats], numerators: Floats, denominators: Floats | None, replicates: int
) -> Replicates:
    """The ratios sum(w x) / sum(w y) of each row of ``numerators`` (x) and ``denominators`` (y,
    default 1) for the weights in ``blocks``; NaN where the denominator is not positive."""
    ratios = np.empty((replicates, numerators.shape[0]), dtype=np.float64)
    start = 0
    for weights in blocks:
        stop = start + weights.shape[0]
        top = weights @ numerators.T
        bottom = (
            weights.sum(axis=1, keepdims=True) if denominators is None else weights @ denominators.T
        )
        defined = bottom > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios[start:stop] = np.where(defined, top / np.where(defined, bottom, 1.0), np.nan)
        start = stop
    return Replicates(ratios)


def _check_units(day_index: Positions, days: int, entity_index: Positions) -> int:
    """The number of frauds, after checking the day and entity of each; the number of distinct
    entities follows from the entity indices (0 .. G - 1 without gaps)."""
    frauds = int(day_index.shape[0])
    if frauds == 0:
        raise DataValidationError("The bootstrap needs at least one fraud")
    if day_index.shape != (frauds,) or entity_index.shape != (frauds,):
        raise DataValidationError("Every fraud needs one day and one entity")
    if days < 1 or int(day_index.max()) >= days or int(day_index.min()) < 0:
        raise DataValidationError("Every fraud's day must be one of the partition's days")
    entities = int(entity_index.max()) + 1
    if int(entity_index.min()) < 0 or len(np.unique(entity_index)) != entities:
        raise DataValidationError("Entity indices must be 0 .. G - 1 without gaps")
    return frauds


def _pigeonhole_blocks(
    rng: np.random.Generator,
    day_index: Positions,
    days: int,
    entity_index: Positions,
    replicates: int,
) -> Iterator[Floats]:
    entities = int(entity_index.max()) + 1
    day_draws = rng.multinomial(days, np.full(days, 1 / days), size=replicates)
    progress = ProgressLogger(logger, "pigeonhole blocks", -(-replicates // BLOCK))
    for start in range(0, replicates, BLOCK):
        stop = min(start + BLOCK, replicates)
        entity_draws = rng.multinomial(entities, np.full(entities, 1 / entities), size=stop - start)
        weights = day_draws[start:stop][:, day_index] * entity_draws[:, entity_index]
        yield weights.astype(np.float64)
        progress.advance()


def pigeonhole_weights(
    rng: np.random.Generator,
    day_index: Positions,
    days: int,
    entity_index: Positions,
    *,
    replicates: int,
) -> Iterator[Floats]:
    """The fraud weights w_i = a_d(i) * b_e(i) of the pigeonhole bootstrap, in blocks of at most
    ``BLOCK`` replicates (rows) over the frauds (columns). ``day_index`` (0 .. ``days`` - 1) and
    ``entity_index`` (0 .. G - 1) give each fraud's relative day and fine pseudo-entity."""
    _check_units(day_index, days, entity_index)
    return _pigeonhole_blocks(rng, day_index, days, entity_index, _replicate_count(replicates))


def pigeonhole_ratios(
    rng: np.random.Generator,
    numerators: npt.ArrayLike,
    day_index: Positions,
    days: int,
    entity_index: Positions,
    *,
    replicates: int,
    denominators: npt.ArrayLike | None = None,
) -> Replicates:
    """The pigeonhole replicates of sum(w x) / sum(w y) for each row x of ``numerators`` (and the
    matching row y of ``denominators``, default 1) over the frauds of one partition. The same
    weights serve every row. A capture-indicator row gives the replicate recall of one
    configuration; an amount-weighted or slice ratio gives its own numerator and denominator
    rows."""
    frauds = _check_units(day_index, days, entity_index)
    top = _matrix(numerators, "numerators", frauds)
    bottom = (
        None
        if denominators is None
        else _matrix(denominators, "denominators", frauds, nonnegative=True)
    )
    if bottom is not None and bottom.shape != top.shape:
        raise DataValidationError("denominators need the shape of the numerators")
    with log_step(
        logger,
        f"Pigeonhole bootstrap: {replicates} replicates, {top.shape[0]} quantities, "
        f"{frauds} frauds",
    ):
        blocks = pigeonhole_weights(rng, day_index, days, entity_index, replicates=replicates)
        return _ratios(blocks, top, bottom, replicates)


def _naive_blocks(rng: np.random.Generator, frauds: int, replicates: int) -> Iterator[Floats]:
    for start in range(0, replicates, BLOCK):
        stop = min(start + BLOCK, replicates)
        draws = rng.multinomial(frauds, np.full(frauds, 1 / frauds), size=stop - start)
        yield draws.astype(np.float64)


def naive_weights(rng: np.random.Generator, frauds: int, *, replicates: int) -> Iterator[Floats]:
    """The multiplicities of the fraud-level iid bootstrap, in blocks of at most ``BLOCK``
    replicates (rows) over the frauds (columns)."""
    if frauds < 1:
        raise DataValidationError("The bootstrap needs at least one fraud")
    return _naive_blocks(rng, frauds, _replicate_count(replicates))


def naive_ratios(
    rng: np.random.Generator,
    numerators: npt.ArrayLike,
    *,
    replicates: int,
    denominators: npt.ArrayLike | None = None,
) -> Replicates:
    """As ``pigeonhole_ratios`` with iid fraud multiplicities: the variance of these replicates is
    V_naive of S1."""
    shaped = np.asarray(numerators, dtype=np.float64)
    if shaped.ndim != 2:
        raise DataValidationError(
            "numerators must have one row per quantity and one column per fraud"
        )
    frauds = int(shaped.shape[1])
    top = _matrix(shaped, "numerators", frauds)
    bottom = (
        None
        if denominators is None
        else _matrix(denominators, "denominators", frauds, nonnegative=True)
    )
    if bottom is not None and bottom.shape != top.shape:
        raise DataValidationError("denominators need the shape of the numerators")
    with log_step(
        logger,
        f"Fraud-level bootstrap: {replicates} replicates, {top.shape[0]} quantities, "
        f"{frauds} frauds",
    ):
        return _ratios(naive_weights(rng, frauds, replicates=replicates), top, bottom, replicates)


def day_block_indices(
    rng: np.random.Generator, days: int, block_length: int, *, replicates: int
) -> Positions:
    """The non-circular moving-block resamples of DR-003: for each replicate, block starts drawn
    uniformly from 0 .. ``days`` - ``block_length`` (``Generator.integers``), ceil(days / length)
    blocks of consecutive days concatenated and truncated to the first ``days`` positions."""
    if not 1 <= block_length <= days:
        raise DataValidationError(
            f"The block length must lie between 1 and the {days} days, not {block_length}"
        )
    blocks = math.ceil(days / block_length)
    starts = rng.integers(0, days - block_length + 1, size=(_replicate_count(replicates), blocks))
    positions = starts[:, :, np.newaxis] + np.arange(block_length)
    return positions.reshape(replicates, blocks * block_length)[:, :days].astype(np.intp)


def day_block_sets(
    rng: np.random.Generator, days: int, block_lengths: Sequence[int], *, replicates: int
) -> tuple[DayBlocks, ...]:
    """One set of day-block resamples per block length, drawn one after the other from ``rng``."""
    return tuple(
        DayBlocks(length, day_block_indices(rng, days, length, replicates=replicates))
        for length in block_lengths
    )


def h3_day_block_sets(declaration: EvaluationDeclaration, days: int) -> tuple[DayBlocks, ...]:
    """The three resample sets of the H3 stream (DR-006): block lengths 3, 1 and 7 in that order,
    shared by all configurations because the Test days are common to them."""
    resampling = declaration.resampling
    return day_block_sets(
        stream_generator(declaration, "day_block_h3"),
        days,
        (resampling.block_length, *resampling.block_length_sensitivity),
        replicates=resampling.replicates,
    )


def day_ratios(
    numerators: npt.ArrayLike, denominators: npt.ArrayLike, indices: Positions
) -> Replicates:
    """The replicate ratios sum(x) / sum(y) over the resampled days: ``numerators`` has one row
    per quantity and one column per day; ``denominators`` has the same shape, or one value per
    day shared by all quantities; ``indices`` is a (replicates x days) day-position matrix such
    as ``day_block_indices``. Quantities that are compared (for example R3 and R4 costs) use the
    same index set, so they are paired within each replicate."""
    if indices.ndim != 2 or indices.shape[0] < 1 or indices.shape[1] < 1:
        raise DataValidationError("indices must be a (replicates x days) matrix")
    days = indices.shape[1]
    if int(indices.min()) < 0 or int(indices.max()) >= days:
        raise DataValidationError("Every index must be one of the days")
    top = _matrix(numerators, "numerators", days)
    denominator = np.asarray(denominators, dtype=np.float64)
    if denominator.ndim == 1:
        denominator = np.broadcast_to(denominator, top.shape)
    bottom = _matrix(denominator, "denominators", days, nonnegative=True)
    if bottom.shape != top.shape:
        raise DataValidationError("denominators need the shape of the numerators")
    counts = (indices[:, :, np.newaxis] == np.arange(days)).sum(axis=1).astype(np.float64)
    above = counts @ bottom.T
    defined = above > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(defined, (counts @ top.T) / np.where(defined, above, 1.0), np.nan)
    return Replicates(ratios)
