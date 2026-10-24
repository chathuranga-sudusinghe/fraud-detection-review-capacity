"""Platt scaling, isotonic regression and the calibration diagnostics."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

from review_capacity.errors import DataValidationError, FileAccessError, ModelError
from review_capacity.evaluation.calibration import (
    BINS,
    EPSILON,
    IsotonicCalibrator,
    PlattCalibrator,
    clipped_log_odds,
    diagnostics,
    fit_isotonic,
    fit_platt,
    load_calibrator,
    reliability_bins,
)

DAYS = {"first": 127, "last": 154}


def sample(rows: int = 4000, seed: int = 0, shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Scores whose log-odds are the truth shifted by ``shift`` (miscalibrated when not 0)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(-3.0, 1.5, rows)
    labels = (rng.random(rows) < expit(z)).astype(np.int8)
    return expit(z + shift), labels


def test_platt_scaling_matches_an_independent_minimiser_of_the_smoothed_loss() -> None:
    scores, labels = sample(shift=0.8)
    calibrator, fit = fit_platt(scores, labels)
    z, _, _ = clipped_log_odds(scores)
    positives = int(labels.sum())
    target = np.where(
        labels == 1, (positives + 1) / (positives + 2), 1 / (len(labels) - positives + 2)
    )

    def loss(theta: np.ndarray) -> float:
        a = theta[0] + theta[1] * z
        return float(np.mean(np.logaddexp(0.0, a) - target * a))

    reference = minimize(loss, [0.0, 1.0], method="BFGS", options={"gtol": 1e-12})
    assert calibrator.intercept == pytest.approx(reference.x[0], abs=1e-5)
    assert calibrator.slope == pytest.approx(reference.x[1], abs=1e-5)
    assert fit.converged
    assert fit.target_positive == (positives + 1) / (positives + 2)
    assert fit.target_negative == 1 / (len(labels) - positives + 2)


def test_platt_scaling_recovers_the_shift_of_a_miscalibrated_score() -> None:
    scores, labels = sample(rows=40_000, shift=0.8)
    calibrator, _ = fit_platt(scores, labels)
    assert calibrator.slope == pytest.approx(1.0, abs=0.08)
    assert calibrator.intercept == pytest.approx(-0.8, abs=0.1)


def test_the_identity_map_is_the_start_and_a_calibrated_score_stays_close() -> None:
    scores, labels = sample(rows=40_000)
    calibrator, _ = fit_platt(scores, labels)
    assert calibrator.intercept == pytest.approx(0.0, abs=0.1)
    assert calibrator.slope == pytest.approx(1.0, abs=0.08)


def test_the_calibrated_probability_is_increasing_and_clipping_is_counted() -> None:
    scores = np.array([0.0, 1e-20, 0.2, 0.5, 0.9, 1.0])
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    calibrator, fit = fit_platt(scores, labels)
    probabilities = calibrator.predict(scores)
    assert (np.diff(probabilities) >= 0).all()
    assert probabilities[0] == probabilities[1]  # clipping merges the extreme scores
    assert (fit.clipped_low, fit.clipped_high) == (2, 1)
    assert calibrator.slope > 0


def test_platt_scaling_refuses_what_it_cannot_fit() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(DataValidationError, match="both classes"):
        fit_platt(scores, np.zeros(4, dtype=np.int8))
    with pytest.raises(DataValidationError, match="two distinct"):
        fit_platt(np.full(4, 0.3), np.array([0, 1, 0, 1], dtype=np.int8))
    with pytest.raises(ModelError, match="not positive"):
        fit_platt(scores, np.array([1, 1, 0, 0], dtype=np.int8))  # a decreasing relation
    with pytest.raises(DataValidationError, match="within"):
        fit_platt(np.array([0.1, 1.5]), np.array([0, 1], dtype=np.int8))
    with pytest.raises(DataValidationError, match="int8"):
        fit_platt(scores, np.array([0, 1, 0, 1]))


def test_a_saved_calibrator_predicts_the_same_after_reloading(tmp_path: Path) -> None:
    scores, labels = sample()
    platt, _ = fit_platt(scores, labels)
    isotonic = fit_isotonic(scores, labels)
    for name, calibrator in (("platt", platt), ("isotonic", isotonic)):
        path = tmp_path / f"{name}.bin"
        calibrator.save(path)
        again = load_calibrator(name, path)
        assert np.array_equal(again.predict(scores), calibrator.predict(scores))
    with pytest.raises(FileAccessError):
        PlattCalibrator.load(tmp_path / "missing.json")
    with pytest.raises(FileAccessError):
        IsotonicCalibrator.load(tmp_path / "missing.npz")
    with pytest.raises(DataValidationError, match="Unknown"):
        load_calibrator("beta", tmp_path / "x")


def test_isotonic_regression_is_monotone_and_matches_scikit_learn() -> None:
    scores, labels = sample(shift=-0.5)
    calibrator = fit_isotonic(scores, labels)
    reference = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    reference.fit(scores, labels.astype(float))
    grid = np.concatenate([scores, [0.0, 1.0, scores.min(), scores.max()]])
    assert calibrator.predict(grid) == pytest.approx(reference.predict(grid), abs=1e-12)
    ordered = calibrator.predict(np.sort(scores))
    assert (np.diff(ordered) >= 0).all()
    assert ((ordered >= 0) & (ordered <= 1)).all()


def test_isotonic_regression_shares_the_mean_label_of_identical_scores() -> None:
    scores = np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.9])
    labels = np.array([1, 0, 0, 0, 1, 1], dtype=np.int8)
    calibrator = fit_isotonic(scores, labels)
    assert calibrator.predict(np.array([0.1]))[0] == pytest.approx(0.25)
    assert calibrator.predict(np.array([0.0]))[0] == pytest.approx(0.25)  # the end value
    assert calibrator.predict(np.array([1.0]))[0] == pytest.approx(1.0)
    with pytest.raises(DataValidationError, match="both classes"):
        fit_isotonic(scores, np.zeros(6, dtype=np.int8))


def test_scores_outside_the_fitting_range_are_counted() -> None:
    scores, labels = sample()
    calibrator, _ = fit_platt(scores, labels)
    probe = np.array([scores.min() / 2, scores.min(), scores.max(), min(1.0, scores.max() + 1e-3)])
    below, above = calibrator.out_of_range(probe)
    assert (below, above) == (1, 1 if probe[3] > scores.max() else 0)


def test_the_diagnostics_of_a_hand_example() -> None:
    scores = np.array([0.2, 0.8])
    labels = np.array([0, 1], dtype=np.int8)
    calibrator = PlattCalibrator(0.0, 1.0, 0.2, 0.8)  # the identity map
    found = diagnostics(calibrator, scores, labels, rows_used="fitting", days=DAYS)
    assert found["brier"] == pytest.approx(((0.2) ** 2 + (0.2) ** 2) / 2)
    assert found["log_loss"] == pytest.approx(-np.log(0.8))
    assert found["metric_clipped"] == 0
    assert (found["rows"], found["frauds"]) == (2, 1)
    assert found["out_of_range"] == {"below": 0, "above": 0}
    assert found["days"] == DAYS


def test_a_calibrated_sample_has_intercept_near_zero_and_slope_near_one() -> None:
    scores, labels = sample(rows=40_000)
    identity = PlattCalibrator(0.0, 1.0, float(scores.min()), float(scores.max()))
    found = diagnostics(identity, scores, labels, rows_used="fitting", days=DAYS)
    assert found["intercept"] == pytest.approx(0.0, abs=0.1)
    assert found["slope"] == pytest.approx(1.0, abs=0.1)
    shifted = diagnostics(
        PlattCalibrator(0.8, 1.0, float(scores.min()), float(scores.max())),
        scores,
        labels,
        rows_used="held_out",
        days=DAYS,
    )
    assert shifted["intercept"] == pytest.approx(-0.8, abs=0.1)
    assert shifted["rows_used"] == "held_out"


def test_probabilities_outside_the_metric_range_are_clipped_and_counted() -> None:
    scores = np.array([0.1, 0.1, 0.5, 0.9])
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    isotonic = fit_isotonic(scores, labels)  # probabilities 0, 0, 1, 1: all four are clipped
    found = diagnostics(isotonic, scores, labels, rows_used="fitting", days=DAYS)
    assert found["metric_clipped"] == 4
    assert found["brier"] == 0.0  # the Brier score uses the original probabilities
    assert found["log_loss"] == pytest.approx(1e-15, abs=1e-14)


def test_a_fit_that_does_not_converge_is_reported_as_not_estimable() -> None:
    scores = np.array([0.1, 0.2, 0.7, 0.8])
    labels = np.array([0, 0, 1, 1], dtype=np.int8)  # separable: the slope diverges
    identity = PlattCalibrator(0.0, 1.0, 0.1, 0.8)
    found = diagnostics(identity, scores, labels, rows_used="fitting", days=DAYS)
    assert found["slope"] is None


def test_a_constant_calibrated_probability_leaves_the_slope_not_estimable() -> None:
    scores = np.array([0.1, 0.9])
    labels = np.array([1, 0], dtype=np.int8)
    constant = fit_isotonic(scores, labels)  # a decreasing relation: one flat step of 0.5
    found = diagnostics(constant, scores, labels, rows_used="fitting", days=DAYS)
    assert found["slope"] is None
    assert found["intercept"] == pytest.approx(0.0, abs=1e-6)  # the offset-only fit is estimable
    flat = PlattCalibrator(0.0, 0.0, 0.1, 0.9)  # p-hat = 0.5 for every row
    again = diagnostics(flat, scores, labels, rows_used="fitting", days=DAYS)
    assert again["slope"] is None


def test_the_reliability_diagram_has_twenty_equal_count_bins() -> None:
    probabilities = np.linspace(0.01, 0.99, 40)
    labels = (np.arange(40) % 4 == 0).astype(np.int8)
    bins = reliability_bins(probabilities, labels)
    assert len(bins) == BINS
    assert [b["transactions"] for b in bins] == [2] * BINS
    assert [b["bin"] for b in bins] == list(range(1, BINS + 1))
    assert sum(b["frauds"] for b in bins) == int(labels.sum())
    assert all(a["max_probability"] < b["min_probability"] for a, b in pairwise(bins))
    assert bins[0]["mean_probability"] == pytest.approx((0.01 + probabilities[1]) / 2)


def test_identical_probabilities_stay_in_one_bin_and_bins_can_be_fewer() -> None:
    probabilities = np.array([0.1] * 30 + [0.2] * 10)
    labels = np.zeros(40, dtype=np.int8)
    labels[::5] = 1
    bins = reliability_bins(probabilities, labels)
    assert [b["transactions"] for b in bins] == [30, 10]  # 0.1 starts in bin 0, 0.2 in bin 15
    assert [b["bin"] for b in bins] == [1, 2]
    assert bins[0]["observed_rate"] == bins[0]["frauds"] / 30


def test_the_clipping_constant_is_that_of_the_protocol() -> None:
    assert EPSILON == 1e-15
    low, lows, highs = clipped_log_odds(np.array([0.0, 0.5, 1.0]))
    assert (lows, highs) == (1, 1)
    assert low[1] == 0.0
    assert low[0] == pytest.approx(np.log(EPSILON) - np.log1p(-EPSILON))


def test_the_mean_of_identical_probabilities_never_leaves_the_bin_range() -> None:
    probabilities = np.full(
        49, 0.1 + 1e-9
    )  # the float mean of equal values can differ in the last bit
    labels = np.zeros(49, dtype=np.int8)
    (only,) = reliability_bins(probabilities, labels)
    assert only["min_probability"] <= only["mean_probability"] <= only["max_probability"]
