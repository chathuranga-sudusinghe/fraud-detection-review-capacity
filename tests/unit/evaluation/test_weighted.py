"""Weighted row-level metrics equal the metrics of the concatenated resampled days (Issue #197)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from review_capacity.errors import DataValidationError
from review_capacity.evaluation.calibration import _logistic
from review_capacity.evaluation.resampling import day_block_indices
from review_capacity.evaluation.weighted import (
    RankedRows,
    calibration_replicates,
    day_counts,
    ranking_replicates,
)

DAYS = 8


def world(
    seed: int = 1, *, ties: bool = True
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int8], npt.NDArray[np.intp]]:
    """Scores with ties, labels and day positions of 8 days of 25 rows; day 3 has no fraud."""
    rng = np.random.default_rng(seed)
    days = np.repeat(np.arange(DAYS), 25).astype(np.intp)
    labels = (rng.random(len(days)) < 0.2).astype(np.int8)
    labels[days == 3] = 0
    labels[0] = 1
    scores = np.clip(rng.normal(0.3 + 0.4 * labels, 0.25), 0.001, 0.999)
    return (np.round(scores, 1) if ties else scores), labels, days


def concatenated(indices: npt.NDArray[np.intp], days: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
    return np.concatenate([np.flatnonzero(days == d) for d in indices])


@pytest.mark.parametrize("ties", [True, False])
def test_ranking_measures_equal_those_of_the_concatenated_days(ties: bool) -> None:
    scores, labels, days = world(ties=ties)
    indices = day_block_indices(np.random.default_rng(5), DAYS, 3, replicates=40)
    counts = day_counts(indices, DAYS)
    got = ranking_replicates(RankedRows.build(scores, labels, days), counts).ratios
    for r in range(40):
        rows = concatenated(indices[r], days)
        assert got[r, 0] == pytest.approx(
            average_precision_score(labels[rows], scores[rows]), abs=1e-12
        )
        assert got[r, 1] == pytest.approx(roc_auc_score(labels[rows], scores[rows]), abs=1e-12)


def test_a_resample_without_frauds_or_legitimate_rows_is_undefined() -> None:
    scores, labels, days = world()
    only_clean = np.full((1, DAYS), 3)  # day 3 has no fraud
    got = ranking_replicates(RankedRows.build(scores, labels, days), day_counts(only_clean, DAYS))
    assert np.isnan(got.ratios).all()
    all_fraud = labels.copy()
    all_fraud[days == 0] = 1
    only_fraud = np.full((1, DAYS), 0)
    got = ranking_replicates(
        RankedRows.build(scores, np.where(days == 0, 1, 0).astype(np.int8), days),
        day_counts(only_fraud, DAYS),
    )
    assert got.ratios[0, 0] == pytest.approx(1.0)  # Average Precision exists
    assert np.isnan(got.ratios[0, 1])  # ROC-AUC needs a legitimate transaction


def test_a_perfect_ranking_stays_inside_the_unit_interval() -> None:
    labels = np.array([1, 1, 0, 0], dtype=np.int8)
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    days = np.array([0, 0, 1, 1], dtype=np.intp)
    got = ranking_replicates(
        RankedRows.build(scores, labels, days), day_counts(np.array([[0, 1]]), 2)
    )
    assert got.ratios[0].tolist() == [1.0, 1.0]


def test_calibration_fits_equal_those_of_the_concatenated_days() -> None:
    scores, labels, days = world(seed=3, ties=False)
    logit = np.log(scores) - np.log1p(-scores)
    target = labels.astype(np.float64)
    intercept = _logistic(np.ones((len(labels), 1)), target, logit)
    slope = _logistic(np.column_stack([np.ones(len(labels)), logit]), target, np.zeros(len(labels)))
    assert intercept is not None
    assert slope is not None
    indices = day_block_indices(np.random.default_rng(6), DAYS, 3, replicates=25)
    got = calibration_replicates(
        logit, labels, days, day_counts(indices, DAYS), point=(float(intercept[0]), float(slope[1]))
    ).ratios
    for r in range(25):
        rows = concatenated(indices[r], days)
        y = labels[rows].astype(np.float64)
        one = _logistic(np.ones((len(rows), 1)), y, logit[rows])
        two = _logistic(np.column_stack([np.ones(len(rows)), logit[rows]]), y, np.zeros(len(rows)))
        if one is None or two is None:
            assert np.isnan(got[r]).any()
            continue
        assert got[r, 0] == pytest.approx(float(one[0]), abs=1e-6)
        assert got[r, 1] == pytest.approx(float(two[1]), abs=1e-6)


def test_a_fit_without_an_optimum_is_undefined() -> None:
    logit = np.array([-2.0, -1.0, 1.0, 2.0] * 5)
    labels = (logit > 0).astype(np.int8)  # complete separation
    days = np.repeat(np.arange(4), 5).astype(np.intp)
    got = calibration_replicates(
        logit, labels, days, day_counts(np.array([[0, 1, 2, 3]]), 4), point=(None, None)
    )
    assert got.ratios[0, 0] == pytest.approx(0.0, abs=1e-6)  # the intercept alone is identified
    assert np.isnan(got.ratios[0, 1])  # the slope diverges: complete separation
    constant = np.zeros(20)  # a constant predictor cannot identify a slope
    mixed = np.array([0, 1] * 10, dtype=np.int8)
    got = calibration_replicates(
        constant, mixed, days, day_counts(np.array([[0, 1, 2, 3]]), 4), point=(None, None)
    )
    assert got.ratios[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(got.ratios[0, 1])


def test_inputs_are_checked() -> None:
    scores, labels, days = world()
    with pytest.raises(DataValidationError, match="same rows"):
        RankedRows.build(scores[:-1], labels, days)
    with pytest.raises(DataValidationError, match="finite"):
        RankedRows.build(np.where(scores > 0.5, np.nan, scores), labels, days)
    with pytest.raises(DataValidationError, match="0 or 1"):
        RankedRows.build(scores, labels + 1, days)
    with pytest.raises(DataValidationError, match="one of the days"):
        day_counts(np.array([[0, 9]]), 2)
    with pytest.raises(DataValidationError, match="replicates x days"):
        day_counts(np.array([0, 1]), 2)


def test_a_resample_far_from_the_whole_partition_still_converges() -> None:
    """The fit of the whole partition is a poor start for a resample of the one dense day; the
    step-halving and the cold-start retry still reach the optimum that ``_logistic`` finds."""
    rng = np.random.default_rng(8)
    days = np.concatenate([np.zeros(100), np.repeat([1, 2, 3], 2000)]).astype(np.intp)
    labels = np.zeros(len(days), dtype=np.int8)
    labels[:50] = 1  # day 0: half frauds
    labels[100 + np.arange(6) * 1000] = 1  # days 1 to 3: 6 frauds in 6,000 rows
    logit = rng.normal(0.0, 1.0, len(days))
    target = labels.astype(np.float64)
    whole_one = _logistic(np.ones((len(days), 1)), target, logit)
    whole_two = _logistic(np.column_stack([np.ones(len(days)), logit]), target, np.zeros(len(days)))
    assert whole_one is not None
    assert whole_two is not None
    counts = day_counts(np.array([[0, 0, 0, 0], [1, 2, 3, 1]]), 4)
    got = calibration_replicates(
        logit, labels, days, counts, point=(float(whole_one[0]), float(whole_two[1]))
    ).ratios
    for r, picked in enumerate(([0, 0, 0, 0], [1, 2, 3, 1])):
        rows = concatenated(np.array(picked), days)
        y = labels[rows].astype(np.float64)
        one = _logistic(np.ones((len(rows), 1)), y, logit[rows])
        two = _logistic(np.column_stack([np.ones(len(rows)), logit[rows]]), y, np.zeros(len(rows)))
        assert one is not None
        assert two is not None
        assert got[r, 0] == pytest.approx(float(one[0]), abs=1e-6)
        assert got[r, 1] == pytest.approx(float(two[1]), abs=1e-6)


def test_each_calibration_fit_is_undefined_on_its_own() -> None:
    """A resample with a constant predictor has an intercept but no slope, and the reverse is not
    possible: the two columns are judged independently."""
    logit = np.array([0.0] * 10 + [1.0, -1.0] * 5)
    labels = np.array([0, 1] * 10, dtype=np.int8)
    days = np.repeat([0, 1], 10).astype(np.intp)
    got = calibration_replicates(
        logit, labels, days, day_counts(np.array([[0, 0]]), 2), point=(0.0, 1.0)
    )
    assert np.isfinite(got.ratios[0, 0])
    assert np.isnan(got.ratios[0, 1])


def test_the_chunk_size_does_not_change_a_result(monkeypatch: pytest.MonkeyPatch) -> None:
    scores, labels, days = world(seed=4, ties=False)
    logit = np.log(scores) - np.log1p(-scores)
    counts = day_counts(day_block_indices(np.random.default_rng(9), DAYS, 3, replicates=23), DAYS)
    ranked = RankedRows.build(scores, labels, days)
    first = (
        ranking_replicates(ranked, counts).ratios,
        calibration_replicates(logit, labels, days, counts, point=(0.0, 1.0)).ratios,
    )
    monkeypatch.setattr("review_capacity.evaluation.weighted.CHUNK", 5)
    second = (
        ranking_replicates(ranked, counts).ratios,
        calibration_replicates(logit, labels, days, counts, point=(0.0, 1.0)).ratios,
    )
    for one, two in zip(first, second, strict=True):
        np.testing.assert_allclose(one, two, rtol=1e-12, atol=1e-12, equal_nan=True)
