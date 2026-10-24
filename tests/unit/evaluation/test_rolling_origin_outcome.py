"""The outcome of one rolling-origin transfer on hand-made scores (no files, no model)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from review_capacity.errors import DataValidationError
from review_capacity.evaluation.rolling import daily_budget_of, origin_outcome

SHARE = Fraction(2, 100)


def window_of(days: int, per_day: int) -> tuple[np.ndarray, np.ndarray]:
    """Scores 0.00, 0.01, ... within each day, so the day's top score is the largest."""
    values = np.tile(np.arange(per_day) / per_day, days)
    return np.repeat(np.arange(1, days + 1), per_day).astype(np.int64), values


def test_the_threshold_is_the_ranked_window_score_and_k_follows_the_window_volume() -> None:
    days, scores = window_of(3, 100)  # 300 transactions, 100 a day
    transfer_days, transfer_scores = (
        np.repeat([4, 5, 6], 100).astype(np.int64),
        np.tile(np.arange(100) / 100, 3),
    )
    outcome = origin_outcome(SHARE, days, scores, (1, 3), transfer_days, transfer_scores, (4, 6))
    assert outcome.rank == 6  # ceil(0.02 x 300)
    # the sixth largest of the window: the scores 0.99, 0.98 and 0.97 occur three times each
    assert outcome.threshold == 0.98
    assert outcome.window_alerts == 6  # score >= 0.98
    assert outcome.k == 2  # round(0.02 x 100)
    assert outcome.transfer_alerts == 6
    assert outcome.transfer_rate == pytest.approx(0.02)
    assert outcome.relative_deviation == pytest.approx(0.0)
    assert outcome.inside_band
    assert [(d["day"], d["selected"], d["overflow"], d["unused"]) for d in outcome.daily] == [
        (4, 2, 0, 0),
        (5, 2, 0, 0),
        (6, 2, 0, 0),
    ]
    assert outcome.days_above == outcome.days_below == outcome.zero_alert_days == 0


def test_a_shifted_transfer_shows_overflow_unused_capacity_and_the_components() -> None:
    days, scores = window_of(2, 100)
    transfer_days = np.repeat([3, 4], 100).astype(np.int64)
    high = np.tile(np.arange(100) / 100 + 1.0, 1)  # every score above the threshold on day 3
    low = np.zeros(100)  # none on day 4
    outcome = origin_outcome(
        SHARE, days, scores, (1, 2), transfer_days, np.concatenate([high, low]), (3, 4)
    )
    first, second = outcome.daily
    assert first["selected"] == 100
    assert first["overflow"] == 98
    assert second["selected"] == 0
    assert second["unused"] == 2
    assert outcome.zero_alert_days == 1
    assert (outcome.days_above, outcome.days_below) == (1, 1)
    assert outcome.transfer_rate == pytest.approx(0.5)
    assert not outcome.inside_band
    # one day with alerts: the components are its own terms
    assert outcome.rate_component == pytest.approx(np.log(1.0 / 0.02))
    assert outcome.volume_component == pytest.approx(0.0)
    assert outcome.total_component == pytest.approx(np.log(100 / 2))


def test_a_transfer_without_any_alert_has_no_components() -> None:
    days, scores = window_of(2, 50)
    transfer_days = np.repeat([3], 50).astype(np.int64)
    outcome = origin_outcome(SHARE, days, scores, (1, 2), transfer_days, np.zeros(50), (3, 3))
    assert outcome.zero_alert_days == 1
    assert (outcome.rate_component, outcome.volume_component, outcome.total_component) == (
        None,
        None,
        None,
    )


def test_the_daily_budget_rounds_half_up() -> None:
    assert daily_budget_of(SHARE, 2_500, 1) == 50
    assert daily_budget_of(SHARE, 75, 1) == 2  # 1.5 rounds up
    assert daily_budget_of(SHARE, 74, 1) == 1


def test_the_windows_must_be_consecutive_and_complete() -> None:
    days, scores = window_of(2, 20)
    other = np.repeat([4], 20).astype(np.int64)
    with pytest.raises(DataValidationError, match="the day after"):
        origin_outcome(SHARE, days, scores, (1, 2), other, np.zeros(20), (4, 4))
    with pytest.raises(DataValidationError, match="needs a transaction"):
        origin_outcome(
            SHARE, days[:20], scores[:20], (1, 2), np.repeat([3], 20), np.zeros(20), (3, 3)
        )
    with pytest.raises(DataValidationError, match="scores"):
        origin_outcome(SHARE, days, scores[:-1], (1, 2), np.repeat([3], 20), np.zeros(20), (3, 3))
    with pytest.raises(DataValidationError, match="must lie in"):
        origin_outcome(SHARE, days, scores, (1, 2), np.repeat([5], 20), np.zeros(20), (3, 3))


@pytest.mark.parametrize(("alerts", "inside"), [(16, False), (17, True), (24, False), (23, True)])
def test_the_band_is_strict_at_both_ends(alerts: int, inside: bool) -> None:
    days, scores = window_of(2, 50)  # the threshold is the second-largest window score, 0.98
    transfer_scores = np.concatenate([np.ones(alerts), np.zeros(1000 - alerts)])
    outcome = origin_outcome(
        SHARE, days, scores, (1, 2), np.repeat([3], 1000).astype(np.int64), transfer_scores, (3, 3)
    )
    assert outcome.transfer_rate == pytest.approx(alerts / 1000)
    assert outcome.inside_band is inside


def test_the_volume_component_follows_the_unequal_daily_volumes() -> None:
    days, scores = window_of(2, 25)  # mean daily volume 25, K = round(0.02 x 25) = 1
    transfer_days = np.concatenate([np.repeat([3], 50), np.repeat([4], 100)]).astype(np.int64)
    transfer_scores = np.concatenate(
        [np.r_[np.ones(2), np.zeros(48)], np.r_[np.ones(6), np.zeros(94)]]
    )
    outcome = origin_outcome(SHARE, days, scores, (1, 2), transfer_days, transfer_scores, (3, 4))
    assert outcome.k == 1
    assert outcome.volume_component == pytest.approx((np.log(50 / 25) + np.log(100 / 25)) / 2)
    assert outcome.rate_component == pytest.approx(
        (np.log(2 / 50 / 0.02) + np.log(6 / 100 / 0.02)) / 2
    )
    assert outcome.total_component == pytest.approx((np.log(2 / 1) + np.log(6 / 1)) / 2)
