"""R4 and the exact cost analysis of RQ3: hand examples and a brute-force oracle."""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import pairwise

import numpy as np
import pytest

from review_capacity.errors import DataValidationError
from review_capacity.evaluation.cost import (
    RATIO_GRID,
    Curve,
    analyse,
    exact_curve,
    r4_outcome,
    r4_record,
    reviewed,
)
from review_capacity.evaluation.queue import DailyQueue
from review_capacity.evaluation.ranking import Capacity, Subject, Truth
from support.development import IDENTITY, file

SUBJECT = Subject("Cat-u", "selected", 42, file("records/score.json"), "V2")


def small() -> tuple[Truth, DailyQueue, np.ndarray, np.ndarray]:
    """Two days of four transactions; frauds are ids 1, 3 and 6. With K = 2 the R3 queues are
    ids 1, 2 (day 1) and 5, 6 (day 2), with calibrated probabilities .9, .05, .03, .8."""
    days = np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    labels = np.array([1, 0, 1, 0, 0, 1, 0, 0], dtype=np.int8)
    amounts = np.array([100.0, 10, 50, 10, 10, 200, 10, 10])
    raw = np.array([0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.7, 0.6])
    probabilities = np.array([0.9, 0.05, 0.5, 0.4, 0.03, 0.8, 0.1, 0.1])
    truth = Truth.build(days, labels, amounts)
    queue = DailyQueue.build(np.arange(1, 9, dtype=np.int64), days)
    return truth, queue, raw, probabilities


CAPACITY = Capacity(Fraction(1, 50), 2)


def test_r4_reviews_the_r3_queue_members_that_reach_the_threshold() -> None:
    truth, queue, raw, probabilities = small()
    selection = queue.select(raw, CAPACITY.k)
    mask = reviewed(selection, probabilities, 10.0)  # p* = 0.1
    assert np.flatnonzero(mask).tolist() == [0, 5]  # ids 1 and 6; the 0.5 and 0.4 are not in R3
    outcome = r4_outcome(truth, selection, probabilities, CAPACITY, 10.0)
    assert outcome.counts == {
        "days": 2,
        "transactions": 8,
        "frauds": 3,
        "selected": 2,
        "true_positives": 2,
        "false_positives": 0,
        "false_negatives": 1,
    }
    assert [(r["selected"], r["unused"], r["overflow"]) for r in outcome.daily] == [
        (1, 1, 0),
        (1, 1, 0),
    ]
    metrics = outcome.metrics
    assert metrics["scenario_cost"] == 2 + 10 * 1  # reviews + rho x missed frauds
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["precision"] == 1.0
    assert metrics["utilisation"] == pytest.approx(2 / 4)
    assert metrics["amount_weighted_recall"] == pytest.approx(300 / 350)
    assert metrics["fraud_value_exposed"] == 50.0


def test_a_probability_equal_to_the_threshold_is_reviewed() -> None:
    _, queue, raw, probabilities = small()
    selection = queue.select(raw, CAPACITY.k)
    assert probabilities[1] == 1.0 / 20  # id 2 has exactly p* at rho = 20
    assert reviewed(selection, probabilities, 20.0)[1]
    assert not reviewed(selection, probabilities, 19.0)[1]  # p* = 1/19 is above 0.05


def test_r4_that_reviews_nothing_leaves_precision_undefined() -> None:
    truth, queue, raw, _ = small()
    selection = queue.select(raw, CAPACITY.k)
    outcome = r4_outcome(truth, selection, np.zeros(8), CAPACITY, 2.0)
    assert outcome.counts["selected"] == 0
    assert outcome.metrics["precision"] is None
    assert outcome.metrics["lift"] is None
    assert outcome.metrics["recall"] == 0
    assert outcome.metrics["scenario_cost"] == 2.0 * 3


def test_r4_is_always_a_subset_of_the_r3_queue_and_needs_matching_rows() -> None:
    truth, queue, raw, probabilities = small()
    selection = queue.select(raw, CAPACITY.k)
    for ratio in (1.5, 3.0, 40.0, 1000.0):
        assert not (reviewed(selection, probabilities, ratio) & ~selection.mask).any()
    with pytest.raises(DataValidationError, match="exceed 1"):
        reviewed(selection, probabilities, 1.0)
    with pytest.raises(DataValidationError, match="same rows"):
        r4_outcome(truth, selection, probabilities[:-1], CAPACITY, 10.0)
    with pytest.raises(DataValidationError, match="within"):
        r4_outcome(truth, selection, probabilities + 1.0, CAPACITY, 10.0)


def test_the_curve_of_a_hand_example_is_one_interval_ending_at_saturation() -> None:
    q = np.array([0.9, 0.8, 0.05, 0.03])
    y = np.array([1, 1, 0, 0], dtype=np.int8)
    curve = exact_curve(q, y)
    assert curve.saturation == pytest.approx(1 / 0.03)
    assert len(curve.intervals) == 1
    assert curve.intervals[0].start == 1.0
    assert curve.intervals[0].end == pytest.approx(1 / 0.03)
    assert curve.break_even == curve.saturation
    assert curve.classification == "coincidence"
    assert curve.zero_crossings == ()


def test_a_crossing_has_its_ties_and_a_break_even_below_saturation() -> None:
    # Breakpoints 2, 5 and 10; ΔJ = rho f - u is negative only for rho < 1.5 on the first segment.
    q = np.array([0.5, 0.2, 0.1])
    y = np.array([1, 0, 1], dtype=np.int8)
    curve = exact_curve(q, y)
    assert curve.saturation == pytest.approx(10.0)
    assert [(i.start, i.end, i.end_closed) for i in curve.intervals] == [(1.0, 1.5, False)]
    assert curve.break_even == 1.5
    assert curve.classification == "crossing"
    assert curve.zero_crossings == (1.5, 2.0)  # ΔJ = 0 at 1.5 (u/f = 3/2) and at 2 (u/f = 2/1)


def test_no_break_even_when_r4_is_never_cheaper() -> None:
    curve = exact_curve(np.array([0.5]), np.array([1], dtype=np.int8))
    assert curve.intervals == ()
    assert curve.break_even is None
    assert curve.classification == "none"
    assert curve.saturation == 2.0


def test_a_probability_of_zero_makes_the_saturation_infinite_and_can_censor_the_break_even() -> (
    None
):
    curve = exact_curve(np.array([0.0, 0.5]), np.array([0, 1], dtype=np.int8))
    assert curve.saturation is None
    assert [(i.start, i.end, i.end_closed) for i in curve.intervals] == [(1.0, 1000.0, True)]
    assert curve.break_even is None
    assert curve.classification == "right_censored"
    # a fraud among the never-reviewed members ends the negative range at the ratio u / f
    fraud = exact_curve(np.array([0.0, 0.5, 0.4]), np.array([1, 0, 0], dtype=np.int8))
    assert fraud.saturation is None
    assert fraud.classification == "crossing"
    assert fraud.break_even == pytest.approx(2.0)
    assert fraud.zero_crossings == (2.0,)


def test_a_negative_result_exactly_at_the_range_end_is_a_closed_point() -> None:
    # Both members are declined below 1000, where ΔJ = rho - 2; at 1000 the fraud is reviewed
    # (0.001 >= 1/1000) and ΔJ = -1: a single negative ratio at the end of the range.
    curve = exact_curve(np.array([0.001, 0.0]), np.array([1, 0], dtype=np.int8))
    assert [(i.start, i.end, i.end_closed) for i in curve.intervals] == [
        (1.0, 2.0, False),
        (1000.0, 1000.0, True),
    ]
    assert curve.saturation is None
    assert curve.classification == "right_censored"
    assert curve.break_even is None


def test_a_probability_of_one_is_reviewed_at_every_ratio_above_one() -> None:
    curve = exact_curve(np.array([1.0, 1.0]), np.array([1, 0], dtype=np.int8))
    assert curve.saturation == 1.0
    assert curve.intervals == ()


def delta(q: np.ndarray, y: np.ndarray, rho: float) -> float:
    declined = q < 1.0 / rho
    return rho * float(y[declined].sum()) - float(declined.sum())


@pytest.mark.parametrize("seed", range(25))
def test_the_curve_matches_a_direct_evaluation_of_delta_j(seed: int) -> None:
    rng = np.random.default_rng(seed)
    members = int(rng.integers(2, 40))
    q = rng.integers(0, 41, members) / 40.0  # many exact ties, some zeros and ones
    y = (rng.random(members) < 0.25).astype(np.int8)
    curve = exact_curve(q, y)
    critical = sorted(
        {1.0, 1000.0} | {1 / v for v in q if v > 0 and 1 < 1 / v < 1000} | set(curve.zero_crossings)
    )
    midpoints = [(a + b) / 2 for a, b in pairwise(critical)]
    inside_end = any(i.end_closed and i.start <= 1000.0 <= i.end for i in curve.intervals)
    assert inside_end == (delta(q, y, 1000.0) < 0)  # the range end is part of the range
    for rho in midpoints:
        inside = any(
            i.start <= rho < i.end or (i.end_closed and i.start <= rho <= i.end)
            for i in curve.intervals
        )
        assert inside == (delta(q, y, rho) < 0), (rho, curve)
    if curve.break_even is not None:
        e = curve.break_even
        assert delta(q, y, e - 1e-9) < 0
        assert delta(q, y, e + 1e-9) >= 0 or e + 1e-9 > 1000
    for tie in curve.zero_crossings:
        assert math.isclose(delta(q, y, tie), 0.0, abs_tol=1e-9)
    positive = q[q > 0]
    if (q > 0).all():
        assert curve.saturation == pytest.approx(1 / positive.min())
    else:
        assert curve.saturation is None
    if curve.saturation is not None:
        assert curve.break_even is None or curve.break_even <= curve.saturation


def test_the_grid_table_of_a_hand_example() -> None:
    truth, queue, raw, probabilities = small()
    selection = queue.select(raw, CAPACITY.k)
    analysis = analyse(truth, selection, probabilities, CAPACITY)
    assert analysis["queue"] == {"reviews": 4, "frauds": 3, "captured": 2}
    grid = {row["ratio"]: row for row in analysis["grid"]}
    assert list(grid) == [float(r) for r in RATIO_GRID]
    ten = grid[10.0]  # p* = 0.1: ids 2 (0.05) and 5 (0.03) are declined; neither is a fraud
    assert (ten["reviews"], ten["unused"], ten["frauds_forgone"]) == (2, 2, 0)
    assert ten["cost_r3"] == 4 + 10 * 1
    assert ten["cost_r4"] == 2 + 10 * 1
    assert ten["cost_difference"] == -2.0 == 10 * 0 - 2
    assert ten["declined_fraud_rate"] == 0.0
    assert ten["declined_mean_probability"] == pytest.approx((0.05 + 0.03) / 2)
    assert ten["fixed_status"] == "infinite"
    assert ten["days_with_unused"] == 2
    assert ten["cost_no_review"] == 30.0
    assert ten["fraud_value_forgone"] == 0.0
    assert ten["fraud_value_exposed_r3"] == 50.0
    assert ten["amount_weighted_recall_r4"] == pytest.approx(300 / 350)
    two = grid[2.0]  # p* = 0.5: only ids 1 (0.9) and 6 (0.8) stay
    assert (two["reviews"], two["unused"]) == (2, 2)
    fifty = grid[50.0]  # p* = 0.02: every member is reviewed
    assert (fifty["unused"], fifty["fixed_status"], fifty["fixed_ratio"]) == (0, "identical", None)
    assert fifty["declined_fraud_rate"] is None
    assert fifty["cost_difference"] == 0.0
    assert analysis["break_even"]["classification"] == "coincidence"
    assert analysis["saturation"] == {"ratio": pytest.approx(1 / 0.03), "infinite": False}
    assert analysis["friction"][0]["ratio"] is None or analysis["friction"][0]["phi"] == 1.0


def test_a_fraud_among_the_unused_reviews_gives_a_finite_fixed_break_even() -> None:
    days = np.array([1, 1, 1, 1], dtype=np.int64)
    labels = np.array([1, 1, 0, 0], dtype=np.int8)
    truth = Truth.build(days, labels, np.array([10.0, 20.0, 5.0, 5.0]))
    queue = DailyQueue.build(np.arange(1, 5, dtype=np.int64), days)
    raw = np.array([0.9, 0.8, 0.7, 0.6])
    probabilities = np.array([0.5, 0.01, 0.02, 0.0])
    capacity = Capacity(Fraction(1, 50), 3)  # queue: ids 1, 2, 3
    analysis = analyse(truth, queue.select(raw, 3), probabilities, capacity)
    row = {r["ratio"]: r for r in analysis["grid"]}[20.0]  # p* = 0.05: ids 2 and 3 declined
    assert (row["unused"], row["frauds_forgone"]) == (2, 1)
    assert (row["fixed_status"], row["fixed_ratio"]) == ("finite", 2.0)
    assert row["forgone_per_unused"] == 10.0
    assert row["fraud_value_forgone"] == 20.0
    # members .5, .01, .02: breakpoints 2, 100 and 50; ΔJ = rho f - u is negative only below
    # rho = 3/2 (u = 3, f = 2 before id 1 is admitted at 2), so the break-even is 1.5
    assert analysis["break_even"]["classification"] == "crossing"
    assert analysis["break_even"]["ratio"] == 1.5
    assert analysis["break_even"]["zero_crossings"] == [1.5, 2.0]
    assert analysis["saturation"]["ratio"] == pytest.approx(100.0)
    # friction: rho = (1 + phi) rho_BE - phi
    assert analysis["friction"] == [
        {"phi": 1.0, "ratio": 2 * 1.5 - 1},
        {"phi": 5.0, "ratio": 6 * 1.5 - 5},
    ]


def test_the_r4_record_carries_the_threshold_ratio_and_calibration() -> None:
    truth, queue, raw, probabilities = small()
    selection = queue.select(raw, CAPACITY.k)
    outcome = r4_outcome(truth, selection, probabilities, CAPACITY, 10.0)
    record = r4_record(
        IDENTITY, SUBJECT, truth, CAPACITY, 10.0, "platt", file("records/calibration.json"), outcome
    )
    assert record["policy"] == "R4"
    assert record["threshold"] == 1 / 10.0
    assert record["cost_ratio"] == 10.0
    assert (record["calibrator"], record["k"], record["capacity"]) == ("platt", 2, 0.02)
    assert record["cutoff_ties"] == {"days": 0, "group_sizes": []}


def test_curve_values_are_plain_data() -> None:
    curve = Curve(None, (), ())
    assert curve.classification == "none"
    assert curve.break_even is None
