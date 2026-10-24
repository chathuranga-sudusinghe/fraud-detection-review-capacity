"""Calibrators of the raw fraud score and their diagnostics (protocol Calibration; DR-007 §8;
Issue #158).

Calibration serves R4 only: R4 compares a calibrated probability with the cost threshold
1/rho (rho: the miss-to-review cost ratio), while R0–R3 use the raw score. Two calibrators are
fitted on V2 scores of a configuration's selected main-seed model, unweighted:

* **Platt scaling** (primary) on the float64 log-odds of the clipped raw score,

  $$
  s_c = \\min(\\max(s, \\epsilon), 1 - \\epsilon), \\qquad z = \\ln s_c - \\ln(1 - s_c), \\qquad
  \\hat p = \\frac{1}{1 + \\exp(-(\\kappa_0 + \\kappa_1 z))}
  $$

  with ε = 10⁻¹⁵ and κ₁ > 0. The fit minimises the binary cross-entropy against Platt's smoothed
  targets t₊ = (N₊ + 1)/(N₊ + 2) and t₋ = 1/(N₋ + 2), where N₊ and N₋ count the frauds and the
  legitimate rows of the fitting rows (a smoothed-target fit, not maximum likelihood against the
  labels), with no coefficient penalty. **Solver:** Newton–Raphson on (κ₀, κ₁) with the exact
  Hessian and step-halving (a step is accepted only if it lowers the loss), started at the
  identity map (κ₀, κ₁) = (0, 1); it stops when every component of the mean gradient is at most
  ``GRADIENT_TOLERANCE`` and fails after ``MAX_ITERATIONS`` iterations. The loss is strictly
  convex when there are two distinct clipped log-odds, so the minimum is unique.
* **Isotonic regression** (secondary check) on the raw scores and the original labels, increasing,
  bounded to [0, 1], identical scores sharing their mean label; between its fitted knots the
  prediction is linear in the raw score, and outside the fitting range it takes the end values.

Both require the two classes (and Platt two distinct clipped log-odds); Platt also stops on
non-convergence, non-finite parameters or a non-positive slope, with no substitute and no forcing
of the slope. A calibrator is saved and reloaded, and its predictions are verified to be equal.

**Diagnostics** (separate from the fit; on V2 they are in-sample): the Brier score of the original
probabilities; the log loss of a metric-only copy clipped to [ε, 1 − ε] with the number of
probabilities clipped; the calibration intercept (labels on logit(p̂) as an offset, intercept
only; target 0) and slope (labels on logit(p̂) with a free intercept; target 1), both unweighted
logistic fits reported as not estimable when Newton–Raphson does not converge; the counts of
scores outside the fitting range; and the reliability diagram. **Reliability bins:** 20
equal-count bins formed from the probabilities alone: a group of identical probabilities with
``c`` smaller probabilities below it lies in bin ``floor(20 c / n)`` (the bin where it starts), so
identical probabilities are never split and a bin that no group starts in does not exist (fewer
than 20 bins).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import sklearn
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

from review_capacity.errors import DataValidationError, FileAccessError, ModelError
from review_capacity.files import write_atomically
from review_capacity.logging_setup import log_step
from review_capacity.models.base import Labels

logger = logging.getLogger(__name__)

EPSILON = 1e-15
SOLVER = "newton_step_halving"
INITIAL = (0.0, 1.0)  # the identity map on the log-odds
MAX_ITERATIONS = 100
GRADIENT_TOLERANCE = 1e-10
DIAGNOSTIC_ITERATIONS = 100
DIAGNOSTIC_TOLERANCE = 1e-9
SEPARATION = 1e-4  # a diagnostic fit that reproduces every label this closely has no finite optimum
BINS = 20

type Floats = npt.NDArray[np.float64]


def _scores(scores: object, what: str = "scores") -> Floats:
    if not isinstance(scores, np.ndarray) or scores.ndim != 1:
        raise DataValidationError(f"The {what} must be a one-dimensional array")
    if scores.dtype not in (np.float32, np.float64):
        raise DataValidationError(f"The {what} must be float32 or float64, not {scores.dtype}")
    values = scores.astype(np.float64) + 0.0
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise DataValidationError(f"The {what} must be finite and within [0, 1]")
    return values


def _labels(labels: object, rows: int) -> Labels:
    if not isinstance(labels, np.ndarray) or labels.dtype != np.int8 or labels.shape != (rows,):
        raise DataValidationError(f"The labels must be an int8 array of {rows} rows")
    if not np.isin(labels, (0, 1)).all():
        raise DataValidationError("The labels must be 0 or 1")
    return labels


def clipped_log_odds(scores: Floats, epsilon: float = EPSILON) -> tuple[Floats, int, int]:
    """The log-odds of the scores clipped to [ε, 1 − ε], and how many were raised and lowered."""
    low, high = int((scores < epsilon).sum()), int((scores > 1 - epsilon).sum())
    clipped = np.clip(scores, epsilon, 1 - epsilon)
    return np.log(clipped) - np.log1p(-clipped), low, high


def software() -> dict[str, str]:
    """The versions of the numerical libraries that fit and reload a calibrator."""
    return {"numpy": np.__version__, "scikit-learn": sklearn.__version__}


# --- Platt scaling -------------------------------------------------------------------------------


@dataclass(frozen=True)
class PlattFit:
    """What fitting recorded: the clipping, the smoothed targets and the solver outcome."""

    clipped_low: int
    clipped_high: int
    target_positive: float
    target_negative: float
    iterations: int
    converged: bool
    gradient: float
    loss: float
    fitting_low: float
    fitting_high: float


@dataclass(frozen=True)
class PlattCalibrator:
    """The frozen map: intercept κ₀ and slope κ₁ on the clipped log-odds, and the fitting range of
    the raw scores (for the out-of-range counts)."""

    intercept: float
    slope: float
    fitting_low: float
    fitting_high: float
    epsilon: float = EPSILON
    method: str = "platt"

    def predict(self, scores: Floats) -> Floats:
        z, _, _ = clipped_log_odds(_scores(scores), self.epsilon)
        probabilities: Floats = expit(self.intercept + self.slope * z)
        return probabilities

    def out_of_range(self, scores: Floats) -> tuple[int, int]:
        return int((scores < self.fitting_low).sum()), int((scores > self.fitting_high).sum())

    def save(self, path: Path) -> None:
        document = {
            "method": self.method,
            "epsilon": self.epsilon,
            "intercept": self.intercept,
            "slope": self.slope,
            "fitting_low": self.fitting_low,
            "fitting_high": self.fitting_high,
        }
        text = json.dumps(document, indent=2, sort_keys=True)
        write_atomically(path, lambda target: target.write_text(text, encoding="utf-8"))

    @classmethod
    def load(cls, path: Path) -> PlattCalibrator:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if document["method"] != "platt":
                raise ModelError(f"{path}: not a Platt calibrator")
            return cls(
                float(document["intercept"]),
                float(document["slope"]),
                float(document["fitting_low"]),
                float(document["fitting_high"]),
                float(document["epsilon"]),
            )
        except (OSError, ValueError, KeyError) as error:
            raise FileAccessError(f"{path}: the Platt calibrator cannot be read") from error


def _platt_loss(
    theta: npt.NDArray[np.float64], z: Floats, target: Floats
) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Mean smoothed cross-entropy (softplus(a) − t a), its mean gradient and its mean Hessian."""
    a = theta[0] + theta[1] * z
    p = expit(a)
    loss = float(np.mean(np.logaddexp(0.0, a) - target * a))
    residual = p - target
    gradient = np.array([residual.mean(), (residual * z).mean()])
    weight = p * (1 - p)
    hessian = np.array(
        [[weight.mean(), (weight * z).mean()], [(weight * z).mean(), (weight * z * z).mean()]]
    )
    return loss, gradient, hessian


def fit_platt(scores: Floats, labels: Labels) -> tuple[PlattCalibrator, PlattFit]:
    """Fit Platt scaling on ``scores`` and ``labels`` as described in the module description."""
    values = _scores(scores)
    y = _labels(labels, len(values))
    positives, negatives = int(y.sum()), int((y == 0).sum())
    if positives == 0 or negatives == 0:
        raise DataValidationError("Platt scaling needs both classes among the fitting rows")
    z, low, high = clipped_log_odds(values)
    if len(np.unique(z)) < 2:
        raise DataValidationError("Platt scaling needs at least two distinct clipped log-odds")
    t_pos, t_neg = (positives + 1) / (positives + 2), 1 / (negatives + 2)
    target = np.where(y == 1, t_pos, t_neg)
    theta = np.array(INITIAL, dtype=np.float64)
    iterations = 0
    with log_step(logger, f"Platt scaling on {len(values)} rows"):
        loss, gradient, hessian = _platt_loss(theta, z, target)
        converged = bool(np.abs(gradient).max() <= GRADIENT_TOLERANCE)
        while not converged and iterations < MAX_ITERATIONS:
            iterations += 1
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError as error:
                raise ModelError("Platt scaling: the Hessian is singular") from error
            scale, accepted = 1.0, False
            for _ in range(60):  # step-halving: accept only a step that lowers the loss
                trial = theta - scale * step
                trial_loss, trial_gradient, trial_hessian = _platt_loss(trial, z, target)
                if np.isfinite(trial_loss) and trial_loss <= loss:
                    theta, loss, gradient, hessian, accepted = (
                        trial,
                        trial_loss,
                        trial_gradient,
                        trial_hessian,
                        True,
                    )
                    break
                scale /= 2
            if not accepted:
                break
            converged = bool(np.abs(gradient).max() <= GRADIENT_TOLERANCE)
    if not np.isfinite(theta).all():
        raise ModelError("Platt scaling: a fitted parameter is not finite")
    if not converged:
        raise ModelError(
            f"Platt scaling did not converge in {iterations} iterations "
            f"(largest mean-gradient component {np.abs(gradient).max():.3g})"
        )
    if not theta[1] > 0:
        raise ModelError(f"Platt scaling: the fitted slope {theta[1]:.6g} is not positive")
    calibrator = PlattCalibrator(
        float(theta[0]), float(theta[1]), float(values.min()), float(values.max())
    )
    fit = PlattFit(
        low,
        high,
        t_pos,
        t_neg,
        iterations,
        converged,
        float(np.abs(gradient).max()),
        loss,
        calibrator.fitting_low,
        calibrator.fitting_high,
    )
    return calibrator, fit


# --- Isotonic regression -------------------------------------------------------------------------


@dataclass(frozen=True)
class IsotonicCalibrator:
    """The fitted knots: increasing raw scores and their calibrated values; linear between the
    knots and constant beyond the end knots."""

    scores: Floats
    probabilities: Floats
    method: str = "isotonic"

    @property
    def knots(self) -> int:
        return len(self.scores)

    @property
    def fitting_low(self) -> float:
        return float(self.scores[0])

    @property
    def fitting_high(self) -> float:
        return float(self.scores[-1])

    def predict(self, scores: Floats) -> Floats:
        probabilities: Floats = np.interp(_scores(scores), self.scores, self.probabilities)
        return probabilities

    def out_of_range(self, scores: Floats) -> tuple[int, int]:
        return int((scores < self.fitting_low).sum()), int((scores > self.fitting_high).sum())

    def save(self, path: Path) -> None:
        def write(target: Path) -> None:
            with target.open("wb") as handle:
                np.savez(handle, scores=self.scores, probabilities=self.probabilities)

        write_atomically(path, write)

    @classmethod
    def load(cls, path: Path) -> IsotonicCalibrator:
        try:
            with np.load(path) as archive:
                return cls(archive["scores"].astype(np.float64), archive["probabilities"])
        except (OSError, ValueError, KeyError) as error:
            raise FileAccessError(f"{path}: the isotonic calibrator cannot be read") from error


def fit_isotonic(scores: Floats, labels: Labels) -> IsotonicCalibrator:
    """Fit isotonic regression on the raw ``scores`` and the original ``labels`` (unit weights)."""
    values = _scores(scores)
    y = _labels(labels, len(values))
    if int(y.sum()) == 0 or int((y == 0).sum()) == 0:
        raise DataValidationError("Isotonic regression needs both classes among the fitting rows")
    with log_step(logger, f"Isotonic regression on {len(values)} rows"):
        regression = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        regression.fit(values, y.astype(np.float64))
    knots = np.asarray(regression.X_thresholds_, dtype=np.float64)
    heights = np.asarray(regression.y_thresholds_, dtype=np.float64)
    if not (np.isfinite(heights).all() and (np.diff(knots) > 0).all()):
        raise ModelError("Isotonic regression returned invalid knots")
    return IsotonicCalibrator(knots, heights)


type Calibrator = PlattCalibrator | IsotonicCalibrator


def load_calibrator(method: str, path: Path) -> Calibrator:
    """Reload a saved calibrator of ``method`` (``platt`` or ``isotonic``)."""
    if method == "platt":
        return PlattCalibrator.load(path)
    if method == "isotonic":
        return IsotonicCalibrator.load(path)
    raise DataValidationError(f"Unknown calibration method {method!r}")


# --- Diagnostics ---------------------------------------------------------------------------------


def _logistic(
    design: npt.NDArray[np.float64], y: Floats, offset: Floats
) -> npt.NDArray[np.float64] | None:
    """Unweighted logistic regression of ``y`` on ``design`` with a fixed ``offset``, by
    Newton–Raphson with step-halving; ``None`` when the design is rank deficient, the fit does not
    converge, is not finite or reproduces every label to within ``SEPARATION`` (complete
    separation: the maximum likelihood estimate does not exist)."""
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return None  # a constant predictor cannot identify its slope
    theta = np.zeros(design.shape[1])

    def evaluate(
        t: npt.NDArray[np.float64],
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        a = offset + design @ t
        p = expit(a)
        loss = float(np.mean(np.logaddexp(0.0, a) - y * a))
        gradient = design.T @ (p - y) / len(y)
        hessian = (design * (p * (1 - p))[:, None]).T @ design / len(y)
        return loss, gradient, hessian

    loss, gradient, hessian = evaluate(theta)
    for _ in range(DIAGNOSTIC_ITERATIONS):
        if np.abs(gradient).max() <= DIAGNOSTIC_TOLERANCE:
            if not np.isfinite(theta).all():
                return None
            fitted = expit(offset + design @ theta)
            return None if np.abs(fitted - y).max() < SEPARATION else theta
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return None
        scale, accepted = 1.0, False
        for _ in range(60):
            trial = theta - scale * step
            trial_loss, trial_gradient, trial_hessian = evaluate(trial)
            if np.isfinite(trial_loss) and trial_loss <= loss:
                theta, loss, gradient, hessian, accepted = (
                    trial,
                    trial_loss,
                    trial_gradient,
                    trial_hessian,
                    True,
                )
                break
            scale /= 2
        if not accepted:
            return None
    return None


def reliability_bins(probabilities: Floats, labels: Labels) -> list[dict[str, Any]]:
    """The reliability diagram: equal-count bins formed from the probabilities alone (see the
    module description)."""
    n = len(probabilities)
    order = np.argsort(probabilities, kind="stable")
    sorted_p, sorted_y = probabilities[order], labels[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_p)) + 1))
    ends = np.concatenate((starts[1:], [n]))
    bin_of_group = (BINS * starts) // n
    bins: list[dict[str, Any]] = []
    for number in np.unique(bin_of_group):
        members = np.flatnonzero(bin_of_group == number)
        begin, end = int(starts[members[0]]), int(ends[members[-1]])
        p, y = sorted_p[begin:end], sorted_y[begin:end]
        frauds = int(y.sum())
        bins.append(
            {
                "bin": len(bins) + 1,
                "min_probability": float(p[0]),
                "max_probability": float(p[-1]),
                "mean_probability": float(min(max(p.mean(), p[0]), p[-1])),  # rounding stays inside
                "transactions": end - begin,
                "frauds": frauds,
                "observed_rate": frauds / (end - begin),
            }
        )
    return bins


def diagnostics(
    calibrator: Calibrator, scores: Floats, labels: Labels, *, rows_used: str, days: dict[str, int]
) -> dict[str, Any]:
    """The ``calibration`` record's diagnostic of ``calibrator`` on the rows ``scores`` and
    ``labels`` (``rows_used``: ``fitting`` for the calibrator's own rows, ``held_out`` after)."""
    values = _scores(scores)
    y = _labels(labels, len(values))
    probabilities = calibrator.predict(values)
    brier = float(np.mean((probabilities - y) ** 2))
    metric = np.clip(probabilities, EPSILON, 1 - EPSILON)
    clipped = int((metric != probabilities).sum())
    log_loss = float(-np.mean(y * np.log(metric) + (1 - y) * np.log1p(-metric)))
    logit = np.log(metric) - np.log1p(-metric)
    target = y.astype(np.float64)
    intercept_fit = _logistic(np.ones((len(y), 1)), target, logit)
    slope_fit = _logistic(np.column_stack([np.ones(len(y)), logit]), target, np.zeros(len(y)))
    below, above = calibrator.out_of_range(values)
    return {
        "rows_used": rows_used,
        "days": dict(days),
        "rows": len(y),
        "frauds": int(y.sum()),
        "brier": brier,
        "log_loss": log_loss,
        "metric_clipped": clipped,
        "intercept": None if intercept_fit is None else float(intercept_fit[0]),
        "slope": None if slope_fit is None else float(slope_fit[1]),
        "out_of_range": {"below": below, "above": above},
        "reliability": reliability_bins(probabilities, y),
    }
