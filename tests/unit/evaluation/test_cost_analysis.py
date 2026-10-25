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
    boundary,
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


def test_a_negative_result_at_the_range_end_is_a_closed_interval_of_the_last_floats() -> None:
    # Both members are declined below 1000, where ΔJ = rho - 2; the fraud is reviewed from the
    # float 999.9999999999999 on (0.001 >= 1 / 1000, as reviewed() compares) and ΔJ = -1 there:
    # a negative stretch of the last floats of the range, closed at 1000.
    curve = exact_curve(np.array([0.001, 0.0]), np.array([1, 0], dtype=np.int8))
    assert boundary(0.001) == 999.9999999999999
    assert [(i.start, i.end, i.end_closed) for i in curve.intervals] == [
        (1.0, 2.0, False),
        (boundary(0.001), 1000.0, True),
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


def declined_counts(q: np.ndarray, y: np.ndarray, rho: float) -> tuple[int, int]:
    declined = q < 1.0 / rho
    return int(declined.sum()), int(y[declined].sum())


def segment_at(curve: Curve, rho: float) -> tuple[int, int]:
    (found,) = [
        s
        for s in curve.segments
        if s.start <= rho < s.end or (s.end_closed and s.start <= rho <= s.end)
    ]
    return found.unused, found.frauds_forgone


@pytest.mark.parametrize("seed", range(25))
def test_the_segments_give_the_exact_curve_at_every_ratio(seed: int) -> None:
    rng = np.random.default_rng(seed)
    members = int(rng.integers(2, 40))
    q = rng.integers(0, 41, members) / 40.0  # exact ties, zeros and ones
    y = (rng.random(members) < 0.25).astype(np.int8)
    curve = exact_curve(q, y)
    segments = curve.segments
    assert segments[0].start == 1.0
    assert segments[-1].end == 1000.0
    assert segments[-1].end_closed
    assert all(a.end == b.start for a, b in pairwise(segments))  # no gap, no overlap
    assert all(not s.end_closed for s in segments[:-1])
    breakpoints = {boundary(float(v)) for v in q if v > 0 and 1 < boundary(float(v)) < 1000}
    # on every breakpoint and on its neighbouring floats, where the float division 1 / rho and
    # the reciprocal 1 / q of the probability can disagree
    probes = {1.0 + 1e-9, 1000.0}
    for b in breakpoints:
        probes |= {b, math.nextafter(b, 0.0), math.nextafter(b, math.inf)}
        probes |= {b * (1 + 1e-9), b * (1 - 1e-9)}
    probes |= set(rng.uniform(1.0, 1000.0, 50))
    for rho in sorted(probes):
        if 1.0 < rho <= 1000.0:
            assert segment_at(curve, rho) == declined_counts(q, y, rho), rho
    # the cost difference and the utilisation of the curve are those of the direct evaluation
    for rho in (1.5, 2.0, 7.0, 55.5, 1000.0):
        unused, forgone = segment_at(curve, rho)
        assert rho * forgone - unused == pytest.approx(delta(q, y, rho))


def test_the_grid_of_a_cost_analysis_is_a_table_of_its_curve() -> None:
    truth, queue, raw, probabilities = small()
    capacity = Capacity(Fraction(2, 100), 2)
    body = analyse(truth, queue.select(raw, 2), probabilities, capacity)
    curve = body["curve"]
    assert curve[0]["start"] == 1.0
    assert curve[-1]["end"] == 1000.0
    for row in body["grid"]:
        (found,) = [
            s
            for s in curve
            if s["start"] <= row["ratio"] < s["end"]
            or (s["end_closed"] and row["ratio"] == s["end"])
        ]
        assert (found["unused"], found["frauds_forgone"]) == (row["unused"], row["frauds_forgone"])


def test_a_breakpoint_is_the_smallest_float_ratio_at_which_the_member_is_reviewed() -> None:
    # 1 / 0.95 = 1.0526315789473684 has the reciprocal 0.9500000000000001, which exceeds 0.95: the
    # member is still declined at that ratio, and reviewed from the next float on
    assert 1.0 / (1.0 / 0.95) > 0.95
    assert boundary(0.95) == math.nextafter(1.0 / 0.95, math.inf)
    for value in (0.95, 0.9, 0.7, 0.3, 0.1, 0.05, 0.0123, 1 / 3, 0.999999, 1.0):
        edge = boundary(value)
        assert 1.0 / edge <= value
        assert 1.0 / math.nextafter(edge, 0.0) > value or edge == 1.0
    # the segments of a queue that holds such a member start at that boundary
    curve = exact_curve(np.array([0.95, 0.5]), np.array([1, 0], dtype=np.int8))
    assert [s.start for s in curve.segments] == [1.0, boundary(0.95), boundary(0.5)]
    assert curve.saturation == boundary(0.5)


def test_the_edge_cases_of_the_segments() -> None:
    # a probability of 1 is reviewed at every ratio above 1: no zero-length first segment
    curve = exact_curve(np.array([1.0, 0.5]), np.array([0, 1], dtype=np.int8))
    assert [(s.start, s.end, s.unused) for s in curve.segments] == [(1.0, 2.0, 1), (2.0, 1000.0, 0)]
    # a queue with a zero probability keeps that member declined to the end: one point at 1000
    zero = exact_curve(np.array([0.5, 0.0]), np.array([0, 1], dtype=np.int8))
    assert zero.saturation is None
    assert [(s.start, s.end, s.end_closed, s.unused, s.frauds_forgone) for s in zero.segments] == [
        (1.0, 2.0, False, 2, 1),
        (2.0, 1000.0, True, 1, 1),
    ]
    # an entirely zero queue: one segment, declining everything, over the whole range
    nothing = exact_curve(np.zeros(3), np.array([0, 1, 0], dtype=np.int8))
    assert [(s.start, s.end, s.end_closed, s.unused) for s in nothing.segments] == [
        (1.0, 1000.0, True, 3)
    ]
    # a member first reviewed at the last floats of the range: a segment that ends at 1000, included
    last = exact_curve(np.array([0.001]), np.array([1], dtype=np.int8))
    assert last.segments[-1].start == boundary(0.001)
    assert last.segments[-1].end == 1000.0
    assert last.segments[-1].end_closed
    assert last.saturation == boundary(0.001)
    # a member whose boundary lies above 1000 is declined to the end of the range
    beyond = exact_curve(np.array([0.0009999999999999998]), np.array([1], dtype=np.int8))
    assert [(s.start, s.end, s.end_closed, s.unused) for s in beyond.segments] == [
        (1.0, 1000.0, True, 1)
    ]


def test_the_utilisation_of_a_segment_is_the_reviews_left_over_the_budget() -> None:
    q = np.array([0.9, 0.5, 0.2, 0.05])
    y = np.array([1, 0, 1, 0], dtype=np.int8)
    curve = exact_curve(q, y)
    reviews, k, days = 4, 1, 4  # R3's four queue members over four days at K = 1
    for rho in (1.2, 3.0, 8.0, 40.0):
        unused, _ = segment_at(curve, rho)
        reviewed_now = int((q >= 1.0 / rho).sum())
        assert (reviews - unused) / (k * days) == reviewed_now / (k * days)


@pytest.mark.parametrize("seed", range(60))
def test_every_record_the_analysis_produces_satisfies_the_contract(seed: int) -> None:
    """The rules of the ``cost_analysis`` contract accept what ``analyse`` writes: ties, zeros,
    ones, no unused review, a queue that ends before 1000 and one that does not."""
    from review_capacity.development.records import contract_of
    from support.development import example

    rng = np.random.default_rng(seed)
    n_days, per_day, k = 6, 40, int(rng.integers(2, 9))
    days = np.repeat(np.arange(1, n_days + 1), per_day).astype(np.int64)
    labels = (rng.random(len(days)) < 0.2).astype(np.int8)
    labels[0] = 1  # the partition has a fraud
    amounts = rng.gamma(2.0, 50.0, len(days))
    raw = rng.random(len(days))
    levels = rng.integers(0, 41, len(days)) / 40.0  # many exact ties, some zeros and ones
    if seed % 3 == 0:
        levels = np.clip(levels, 0.002, 1.0)  # no zero probability: a finite saturation ratio
    if seed % 5 == 0:
        levels = np.clip(levels, 0.5, 1.0)  # saturation soon: no unused review at large ratios
    truth = Truth.build(days, labels, amounts)
    queue = DailyQueue.build(np.arange(1, len(days) + 1, dtype=np.int64), days)
    capacity = Capacity(Fraction(2, 100), k)
    body = analyse(truth, queue.select(raw, k), levels, capacity)
    record = example("cost_analysis") | body
    record |= {"k": k, "days": {"first": 127, "last": 127 + n_days - 1}}
    assert contract_of(record).problems(record) == []


def record_problems_for(levels: np.ndarray) -> list[str]:
    """The contract problems of the record that ``analyse`` writes for one queue of 8 members
    with ``levels`` as calibrated probabilities (two days of four transactions, K = 4)."""
    from review_capacity.development.records import contract_of
    from support.development import example

    days = np.repeat([1, 2], 4).astype(np.int64)
    labels = np.array([1, 0, 1, 0, 0, 1, 0, 0], dtype=np.int8)
    truth = Truth.build(days, labels, np.full(8, 10.0))
    queue = DailyQueue.build(np.arange(1, 9, dtype=np.int64), days)
    raw = np.linspace(0.9, 0.1, 8)
    body = analyse(truth, queue.select(raw, 4), levels, Capacity(Fraction(2, 100), 4))
    record = example("cost_analysis") | body | {"k": 4, "days": {"first": 127, "last": 128}}
    return contract_of(record).problems(record)


@pytest.mark.parametrize("value", [5e-324, 1e-310, 1e-309])
def test_a_probability_whose_boundary_overflows_makes_the_saturation_infinite(
    value: float,
) -> None:
    assert math.isinf(boundary(value))  # no finite ratio reaches it
    curve = exact_curve(np.array([0.5, value]), np.array([0, 1], dtype=np.int8))
    assert curve.saturation is None  # not infinity: the record could not hold it
    assert curve.segments[-1].unused == 1  # the member stays declined over the whole range
    levels = np.array([0.9, 0.5, value, 0.3, 0.2, 0.4, 0.6, 0.8])
    assert record_problems_for(levels) == []  # and the record is written and valid


def test_a_tiny_probability_with_a_finite_boundary_keeps_a_finite_saturation() -> None:
    value = 1e-308  # its reciprocal 1e308 is finite, far beyond the range
    assert boundary(value) == pytest.approx(1e308)
    curve = exact_curve(np.array([0.5, value]), np.array([0, 1], dtype=np.int8))
    assert curve.saturation == boundary(value)
    levels = np.array([0.9, 0.5, value, 0.3, 0.2, 0.4, 0.6, 0.8])
    assert record_problems_for(levels) == []
