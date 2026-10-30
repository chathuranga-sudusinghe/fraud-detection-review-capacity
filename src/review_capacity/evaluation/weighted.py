"""Row-level metrics on day-block resamples, from integer day multiplicities (Issue #197).

A day-block resample lists days with repeats. Concatenating the resampled days gives every
transaction of a day the multiplicity of that day, so a metric of the concatenation is the same
metric with the multiplicity as a row weight. The functions here compute the weighted Average
Precision, ROC-AUC, calibration intercept and calibration slope of many resamples from one sort of
the rows. They match the unweighted functions (scikit-learn and the calibration diagnostics) on the
concatenated days, equal scores included; the tests prove it by brute force.

A replicate is undefined where its measure has no value (no fraud, no legitimate transaction, a
calibration fit that does not converge or reproduces every label). The value is NaN, and the
caller counts and reports it (DR-006).

Nothing here reads a file: scores, labels and day positions are arrays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import expit

from review_capacity.errors import DataValidationError
from review_capacity.evaluation.calibration import (
    DIAGNOSTIC_ITERATIONS,
    DIAGNOSTIC_TOLERANCE,
    SEPARATION,
)
from review_capacity.evaluation.resampling import Floats, Positions, Replicates
from review_capacity.logging_setup import ProgressLogger, log_step

logger = logging.getLogger(__name__)

CHUNK = 100  # replicates held in memory at once; bounds memory, never changes a result
SINGULAR = 1e-12  # relative size of the determinant below which a fit has no unique optimum


def day_counts(indices: Positions, days: int) -> Floats:
    """The multiplicity of every day in every resample: (replicates x days) integers as floats."""
    if indices.ndim != 2 or indices.shape[0] < 1 or indices.shape[1] != days:
        raise DataValidationError("indices must be a (replicates x days) matrix of the days")
    if int(indices.min()) < 0 or int(indices.max()) >= days:
        raise DataValidationError("Every index must be one of the days")
    counts = np.empty((indices.shape[0], days), dtype=np.float64)
    for start in range(0, indices.shape[0], CHUNK):
        block = indices[start : start + CHUNK]
        counts[start : start + CHUNK] = (block[:, :, np.newaxis] == np.arange(days)).sum(axis=1)
    return counts


def _rows(day_index: Positions, labels: npt.NDArray[np.int8], *others: Floats) -> int:
    rows = int(labels.shape[0])
    if labels.ndim != 1 or day_index.shape != (rows,) or any(o.shape != (rows,) for o in others):
        raise DataValidationError("Scores, labels and days must describe the same rows")
    if not np.isin(labels, (0, 1)).all():
        raise DataValidationError("Labels must be 0 or 1")
    return rows


@dataclass(frozen=True)
class RankedRows:
    """The rows of one configuration in descending score order, with its groups of equal scores.

    ``day_index`` and ``labels`` are in score order; ``starts`` are the first positions of the
    groups of equal scores, so that no tie-break enters a measure.
    """

    day_index: Positions
    labels: Floats
    starts: npt.NDArray[np.intp]

    @classmethod
    def build(
        cls, scores: Floats, labels: npt.NDArray[np.int8], day_index: Positions
    ) -> RankedRows:
        rows = _rows(day_index, labels, scores)
        if not np.isfinite(scores).all():
            raise DataValidationError("Scores must be finite for the ranking measures")
        order = np.argsort(-scores, kind="stable")
        ordered = scores[order]
        starts = np.concatenate(([0], np.flatnonzero(np.diff(ordered)) + 1)).astype(np.intp)
        if rows == 0:
            raise DataValidationError("The ranking needs at least one row")
        return cls(day_index[order], labels[order].astype(np.float64), starts)


def ranking_replicates(ranked: RankedRows, counts: Floats) -> Replicates:
    """Average Precision and ROC-AUC of every resample: columns ``[average_precision, roc_auc]``.

    NaN for Average Precision without a fraud, and for ROC-AUC without a fraud or a legitimate
    transaction in the resample."""
    replicates = counts.shape[0]
    out = np.full((replicates, 2), np.nan)
    with log_step(logger, f"Weighted ranking measures: {replicates} replicates"):
        progress = ProgressLogger(logger, "ranking chunks", -(-replicates // CHUNK))
        for start in range(0, replicates, CHUNK):
            weights = counts[start : start + CHUNK][:, ranked.day_index]
            positive = np.add.reduceat(weights * ranked.labels, ranked.starts, axis=1)
            total = np.add.reduceat(weights, ranked.starts, axis=1)
            negative = total - positive
            cum_positive = np.cumsum(positive, axis=1)
            cum_total = np.cumsum(total, axis=1)
            frauds = cum_positive[:, -1]
            legitimate = cum_total[:, -1] - frauds
            with np.errstate(divide="ignore", invalid="ignore"):
                precision = np.where(
                    positive > 0, cum_positive / np.where(cum_total > 0, cum_total, 1.0), 0.0
                )
                average = (positive * precision).sum(axis=1) / frauds
                area = (negative * (cum_positive - positive / 2)).sum(axis=1) / (
                    frauds * legitimate
                )
            stop = start + weights.shape[0]
            out[start:stop, 0] = np.where(frauds > 0, np.clip(average, 0.0, 1.0), np.nan)
            out[start:stop, 1] = np.where(
                (frauds > 0) & (legitimate > 0), np.clip(area, 0.0, 1.0), np.nan
            )
            progress.advance()
    return Replicates(out)


def _loss(predictor: Floats, labels: Floats, weights: Floats, total: Floats) -> Floats:
    """The weighted mean logistic loss of every resample at ``predictor`` (inf if not finite)."""
    with np.errstate(all="ignore"):
        loss = (weights * (np.logaddexp(0.0, predictor) - labels * predictor)).sum(axis=1) / total
    return np.where(np.isfinite(loss), loss, np.inf)


def _newton(
    logit: Floats, labels: Floats, weights: Floats, theta: Floats, *, slope: bool
) -> tuple[Floats, npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Newton steps with step-halving, as ``calibration._logistic`` takes them, for every row of
    ``weights``. Returns the parameters, the converged resamples and the resamples with no unique
    finite optimum (singular curvature or complete separation)."""
    replicates = weights.shape[0]
    total = weights.sum(axis=1)

    def predictor(t: Floats) -> Floats:
        return t[:, :1] + t[:, 1:] * logit if slope else t[:, :1] + logit

    theta = theta.copy()
    loss = _loss(predictor(theta), labels, weights, total)
    converged = np.zeros(replicates, dtype=bool)
    ended = np.zeros(replicates, dtype=bool)  # singular, or the halving found no improvement
    with np.errstate(all="ignore"):
        for _ in range(DIAGNOSTIC_ITERATIONS):
            probability = expit(predictor(theta))
            residual = weights * (probability - labels)
            curvature = weights * probability * (1 - probability)
            g0, h00 = residual.sum(axis=1), curvature.sum(axis=1)
            if slope:
                g1 = (residual * logit).sum(axis=1)
                h01, h11 = (curvature * logit).sum(axis=1), (curvature * logit**2).sum(axis=1)
                det = h00 * h11 - h01**2
                step = np.column_stack(((h11 * g0 - h01 * g1) / det, (h00 * g1 - h01 * g0) / det))
                gradient = np.maximum(np.abs(g0), np.abs(g1))
                singular = ~(det > SINGULAR * h00 * h11)
            else:
                step = (g0 / h00)[:, np.newaxis]
                gradient = np.abs(g0)
                singular = ~(h00 > 0)
            converged |= (gradient / total <= DIAGNOSTIC_TOLERANCE) & ~singular
            ended |= singular
            active = ~(converged | ended)
            if not active.any():
                break
            scale = np.ones(replicates)
            pending = active.copy()
            for _ in range(60):
                trial = theta - scale[:, np.newaxis] * step
                trial_loss = _loss(predictor(trial), labels, weights, total)
                accepted = pending & (trial_loss <= loss)
                theta[accepted], loss[accepted] = trial[accepted], trial_loss[accepted]
                pending &= ~accepted
                if not pending.any():
                    break
                scale[pending] /= 2
            ended |= pending
        fitted = expit(predictor(theta))
        separated = np.where(weights > 0, np.abs(fitted - labels), 0.0).max(axis=1) < SEPARATION
    return (
        theta,
        converged & ~separated,
        (ended & ~converged) | separated | ~np.isfinite(theta).all(axis=1),
    )


def _solve(
    logit: Floats,
    labels: Floats,
    day_index: Positions,
    counts: Floats,
    start: tuple[float, ...],
    *,
    slope: bool,
) -> Floats:
    """The weighted logistic fits of one chunk of resamples: the intercept with a fixed log-odds
    offset (``slope`` false) or the intercept and slope of the log-odds (``slope`` true). The
    Newton steps start from the fit of the whole partition; a resample that does not converge from
    there is refitted from the cold start of ``calibration._logistic``. NaN where a fit has no
    unique finite optimum or does not converge from either start."""
    weights = counts[:, day_index]
    theta, converged, undefined = _newton(
        logit,
        labels,
        weights,
        np.tile(np.array(start, dtype=np.float64), (weights.shape[0], 1)),
        slope=slope,
    )
    retry = ~converged & ~undefined
    if retry.any():
        again, again_converged, _ = _newton(
            logit,
            labels,
            weights[retry],
            np.zeros((int(retry.sum()), 2 if slope else 1)),
            slope=slope,
        )
        theta[retry], converged[retry] = again, again_converged
    result = theta[:, -1].copy()
    result[~converged] = np.nan
    return result


def calibration_replicates(
    logit: Floats,
    labels: npt.NDArray[np.int8],
    day_index: Positions,
    counts: Floats,
    *,
    point: tuple[float | None, float | None],
) -> Replicates:
    """Calibration intercept (with the log-odds as a fixed offset) and slope of every resample:
    columns ``[intercept, slope]``. ``point`` holds the two fits of the whole partition, the
    starting values of the Newton steps; NaN where a resample's fit does not exist."""
    rows = _rows(day_index, labels, logit)
    if not np.isfinite(logit).all():
        raise DataValidationError("The log-odds must be finite")
    target = labels.astype(np.float64)
    out = np.full((counts.shape[0], 2), np.nan)
    intercept, slope = point
    with log_step(logger, f"Weighted calibration fits: {counts.shape[0]} replicates, {rows} rows"):
        progress = ProgressLogger(logger, "calibration chunks", -(-counts.shape[0] // CHUNK))
        for start in range(0, counts.shape[0], CHUNK):
            block = counts[start : start + CHUNK]
            stop = start + block.shape[0]
            out[start:stop, 0] = _solve(
                logit, target, day_index, block, (intercept or 0.0,), slope=False
            )
            out[start:stop, 1] = _solve(
                logit, target, day_index, block, (0.0, 1.0 if slope is None else slope), slope=True
            )
            progress.advance()
    return Replicates(out)
