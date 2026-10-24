"""Ranking and capacity outcomes: hand-computed counts and measures on small rankings, exact
ties, days without frauds and a budget above a day's volume; K_c of DR-002; records that satisfy
their contract on a V2-sized synthetic partition (no data read)."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from review_capacity.development.records import RECORD_CONTRACTS, dump_record, validate_record
from review_capacity.development.schema import loads
from review_capacity.errors import DataValidationError
from review_capacity.evaluation.queue import DailyQueue
from review_capacity.evaluation.ranking import (
    CAPACITY_SHARES,
    Capacity,
    Subject,
    Truth,
    capacity_budgets,
    daily_budget,
    r3_outcome,
    r3_record,
    ranking_outcome,
    ranking_record,
)
from support.development import IDENTITY, PARTITIONS, file

SUBJECT = Subject("LGB-u", "selected", 42, file("scores/LGB-u/v2.json"), "V2")


def keys(*values: int) -> np.ndarray:
    return np.array(values, dtype=np.int64)


def small() -> tuple[DailyQueue, Truth, np.ndarray]:
    """Three days. Day 1: five transactions, frauds at ranks 1 and 3. Day 2: four transactions,
    no fraud, three equal scores across the cutoff. Day 3: one fraud, fewer than K."""
    ids = keys(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    days = keys(1, 1, 1, 1, 1, 2, 2, 2, 2, 3)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.1, 0.4])
    labels = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.int8)
    amounts = np.array([10, 20, 30, 40, 50, 1, 2, 3, 4, 100], dtype=np.float64)
    return DailyQueue.build(ids, days), Truth.build(days, labels, amounts), scores


CAPACITY = Capacity(Fraction(1, 5), 2)


def test_r3_counts_and_daily_rows_match_the_hand_calculation() -> None:
    queue, truth, scores = small()
    outcome = r3_outcome(queue, truth, scores, CAPACITY)
    assert outcome.counts == {
        "days": 3,
        "transactions": 10,
        "frauds": 3,
        "selected": 5,
        "true_positives": 2,
        "false_positives": 3,
        "false_negatives": 1,
    }
    assert [(r["day"], r["selected"], r["true_positives"]) for r in outcome.daily] == [
        (1, 2, 1),
        (2, 2, 0),
        (3, 1, 1),
    ]
    assert outcome.daily[2] == {
        "day": 3,
        "transactions": 1,
        "frauds": 1,
        "selected": 1,
        "true_positives": 1,
        "oracle_reviewable": 1,
        "overflow": 0,
        "unused": 1,  # K = 2 above the day's volume
        "load_ratio": 0.5,
        "cutoff_tied_group": None,
    }
    assert outcome.daily[1]["frauds"] == 0
    assert outcome.daily[1]["oracle_reviewable"] == 0  # a day without frauds
    assert outcome.cutoff_ties == {"days": 1, "group_sizes": [3]}
    assert outcome.daily[1]["cutoff_tied_group"] == 3


def test_r3_measures_match_the_hand_calculation() -> None:
    queue, truth, scores = small()
    metrics = r3_outcome(queue, truth, scores, CAPACITY).metrics
    assert metrics == pytest.approx(
        {
            "recall": 2 / 3,
            "precision": 2 / 5,
            "frauds_per_1000_reviews": 400.0,
            "oracle_ceiling": 1.0,  # (2 + 0 + 1) of 3 frauds
            "normalised_recall": 2 / 3,  # 2 of the 3 achievable
            "random_review_recall": 0.5,  # 5 reviews of 10 transactions
            "lift": 4 / 3,  # recall 2/3 over 1/2 = precision 0.4 over prevalence 0.3
            "utilisation": 5 / 6,  # 5 reviews of K x days = 6
            "capacity_to_prevalence": 2 / 3,  # 0.2 over 0.3
            "amount_weighted_recall": 110 / 140,  # amounts 10 + 100 of 10 + 30 + 100
            "fraud_value_exposed": 30.0,
        }
    )


def test_lift_is_precision_over_prevalence() -> None:
    queue, truth, scores = small()
    metrics = r3_outcome(queue, truth, scores, CAPACITY).metrics
    precision = metrics["precision"]
    assert precision is not None
    assert metrics["lift"] == pytest.approx(precision / (3 / 10))


def test_a_budget_above_every_days_volume_reviews_everything() -> None:
    queue, truth, scores = small()
    outcome = r3_outcome(queue, truth, scores, Capacity(Fraction(1, 2), 100))
    assert outcome.counts["selected"] == 10
    assert outcome.counts["false_negatives"] == 0
    assert outcome.cutoff_ties == {"days": 0, "group_sizes": []}
    assert outcome.metrics["recall"] == 1.0
    assert outcome.metrics["normalised_recall"] == 1.0
    assert outcome.metrics["oracle_ceiling"] == 1.0
    assert outcome.metrics["fraud_value_exposed"] == 0.0
    assert outcome.metrics["utilisation"] == pytest.approx(10 / 300)


def test_a_cutoff_tie_is_cut_by_the_shared_tie_rule() -> None:
    queue, truth, scores = small()
    mask = queue.select(scores, 2).mask
    tied_day = np.flatnonzero(truth.days == 2)
    assert int(mask[tied_day].sum()) == 2
    reviewed = {int(i) for i in tied_day[mask[tied_day]]}
    assert reviewed <= {5, 6, 7}  # two of the three equal scores, never the lower one


def test_fraud_amounts_of_zero_leave_amount_weighted_recall_undefined() -> None:
    days = keys(1, 1)
    labels = np.array([1, 0], dtype=np.int8)
    truth = Truth.build(days, labels, np.zeros(2))
    queue = DailyQueue.build(keys(1, 2), days)
    metrics = r3_outcome(queue, truth, np.array([0.9, 0.1]), Capacity(Fraction(1, 2), 1)).metrics
    assert metrics["amount_weighted_recall"] is None  # no fraud amount: not estimable
    assert metrics["fraud_value_exposed"] == 0.0


def test_whole_ranking_measures_keep_equal_scores_together() -> None:
    days = keys(1, 1, 1)
    scores = np.array([0.5, 0.5, 0.1])
    for labels in ([1, 0, 1], [0, 1, 1]):  # the fraud first or second among the equal scores
        truth = Truth.build(days, np.array(labels, dtype=np.int8), np.ones(3))
        measures = ranking_outcome(truth, scores)
        assert measures["roc_auc"] == pytest.approx(0.25)  # (0.5 + 0) of 2 pairs
        assert measures["average_precision"] == pytest.approx(0.25 + 0.5 * (2 / 3))


def test_whole_ranking_measures_need_finite_float_scores_and_both_classes() -> None:
    truth = Truth.build(keys(1, 1), np.array([1, 0], dtype=np.int8), np.ones(2))
    with pytest.raises(DataValidationError, match="finite"):
        ranking_outcome(truth, np.array([0.5, np.nan]))
    with pytest.raises(DataValidationError, match="float32 or float64"):
        ranking_outcome(truth, np.array([1, 2]))
    with pytest.raises(DataValidationError, match="float32 or float64"):
        ranking_outcome(truth, np.array([0.5, 0.1], dtype=np.float16))
    only_fraud = Truth.build(keys(1, 1), np.array([1, 1], dtype=np.int8), np.ones(2))
    with pytest.raises(DataValidationError, match="no legitimate"):
        ranking_outcome(only_fraud, np.array([0.1, 0.2]))


def test_truth_is_checked() -> None:
    days, labels, amounts = keys(1, 1, 3), np.array([1, 0, 0], dtype=np.int8), np.ones(3)
    with pytest.raises(DataValidationError, match="Every relative day"):
        Truth.build(days, labels, amounts)
    with pytest.raises(DataValidationError, match="no fraud"):
        Truth.build(keys(1, 1), np.zeros(2, dtype=np.int8), np.ones(2))
    with pytest.raises(DataValidationError, match="finite"):
        Truth.build(keys(1, 1), np.array([1, 0], dtype=np.int8), np.array([1.0, np.inf]))
    with pytest.raises(DataValidationError, match="int8"):
        Truth.build(keys(1, 1), np.array([1, 0]), np.ones(2))
    with pytest.raises(DataValidationError, match="float64"):
        Truth.build(keys(1, 1), np.array([1, 0], dtype=np.int8), np.ones(2, dtype=np.float32))


def test_r3_needs_the_queue_and_the_labels_to_describe_the_same_rows() -> None:
    queue, truth, scores = small()
    other = Truth.build(
        keys(1, 1, 1, 1, 1, 2, 2, 2, 2, 2), truth.labels, truth.amounts
    )  # the last row moved to day 2
    with pytest.raises(DataValidationError, match="different rows"):
        r3_outcome(queue, other, scores, CAPACITY)


def test_the_daily_budgets_of_dr_002_on_the_v2_partition() -> None:
    budgets = capacity_budgets(PARTITIONS.v2)
    assert [(c.value, c.k) for c in budgets] == [(0.01, 28), (0.02, 56), (0.05, 139), (0.1, 278)]
    assert tuple(c.share for c in budgets) == CAPACITY_SHARES


def test_the_budget_rounds_half_up_exactly() -> None:
    v2 = replace(PARTITIONS.v2, rows=5, first_day=1, last_day=2)  # mean daily volume 2.5
    assert daily_budget(Fraction(1), v2) == 3  # not 2 (round half to even)
    assert daily_budget(Fraction(1, 5), v2) == 1


def v2_sized() -> tuple[DailyQueue, Truth, np.ndarray]:
    v2 = PARTITIONS.v2
    n_days = v2.last_day - v2.first_day + 1
    per_day = np.full(n_days, v2.rows // n_days)
    per_day[: v2.rows % n_days] += 1
    days = np.repeat(np.arange(v2.first_day, v2.last_day + 1, dtype=np.int64), per_day)
    rng = np.random.default_rng(0)
    labels = (rng.random(v2.rows) < 0.034).astype(np.int8)
    scores = rng.random(v2.rows) + 0.5 * labels
    amounts = rng.uniform(1.0, 500.0, v2.rows)
    ids = np.arange(1, v2.rows + 1, dtype=np.int64)
    return DailyQueue.build(ids, days), Truth.build(days, labels, amounts), scores


def test_records_satisfy_their_contract_on_a_v2_sized_partition() -> None:
    queue, truth, scores = v2_sized()
    for capacity in capacity_budgets(PARTITIONS.v2):
        outcome = r3_outcome(queue, truth, scores, capacity)
        record = r3_record(IDENTITY, SUBJECT, truth, capacity, outcome)
        validate_record(record, PARTITIONS, "evaluation")
        assert loads(dump_record(record, PARTITIONS), "record") == record  # exact JSON round trip
        assert outcome.counts["transactions"] == PARTITIONS.v2.rows
        assert outcome.counts["selected"] <= capacity.k * outcome.counts["days"]
        assert 0 < record["metrics"]["normalised_recall"] <= 1
        assert record["metrics"]["recall"] <= record["metrics"]["oracle_ceiling"] + 1e-12
    ranking = ranking_record(IDENTITY, SUBJECT, truth, ranking_outcome(truth, scores))
    validate_record(ranking, PARTITIONS, "evaluation")
    assert ranking["policy"] == "ranking"
    assert ranking["counts"] is None
    assert set(ranking["metrics"]) == {"average_precision", "roc_auc"}


def test_the_reference_is_reported_under_ranking_and_r3_only() -> None:
    queue, truth, scores = small()
    reference = Subject("LR", "selected", 42, file("scores/LR/v2.json"), "V2")
    outcome = r3_outcome(queue, truth, scores, CAPACITY)
    record = r3_record(IDENTITY, reference, truth, CAPACITY, outcome)
    assert RECORD_CONTRACTS["evaluation"].problems(record) == []
    whole = ranking_record(IDENTITY, reference, truth, ranking_outcome(truth, scores))
    assert RECORD_CONTRACTS["evaluation"].problems(whole) == []
    record["policy"] = "R2"  # the contract refuses any other policy for the reference
    assert any("reference" in p for p in RECORD_CONTRACTS["evaluation"].problems(record))


def test_a_binding_budget_gives_an_oracle_ceiling_below_one() -> None:
    """K = 2 below the daily fraud counts: day 1 has 3 frauds in 4 transactions, day 2 has 2
    frauds in 3. Achievable: min(2, 3) + min(2, 2) = 4 of 5 frauds."""
    days = keys(1, 1, 1, 1, 2, 2, 2)
    labels = np.array([1, 0, 1, 1, 1, 1, 0], dtype=np.int8)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.1])
    truth = Truth.build(days, labels, np.ones(7))
    queue = DailyQueue.build(keys(1, 2, 3, 4, 5, 6, 7), days)
    outcome = r3_outcome(queue, truth, scores, Capacity(Fraction(1, 5), 2))
    assert [r["oracle_reviewable"] for r in outcome.daily] == [2, 2]
    assert outcome.counts["true_positives"] == 3  # 1 on day 1, 2 on day 2
    assert outcome.metrics["recall"] == pytest.approx(3 / 5)
    assert outcome.metrics["oracle_ceiling"] == pytest.approx(4 / 5)
    assert outcome.metrics["normalised_recall"] == pytest.approx(3 / 4)
    assert outcome.metrics["random_review_recall"] == pytest.approx(4 / 7)


@pytest.mark.parametrize("fraud_id", [101, 102, 103])
def test_r3_reviews_the_tied_transaction_with_the_lowest_digest(fraud_id: int) -> None:
    """Three equal scores, K = 1: the reviewed one is the first by SHA-256 of its id, and the
    captured count depends on it."""
    ids = (101, 102, 103)
    first = min(ids, key=lambda i: (hashlib.sha256(str(i).encode()).digest(), i))
    days = keys(1, 1, 1)
    labels = np.array([1 if i == fraud_id else 0 for i in ids], dtype=np.int8)
    truth = Truth.build(days, labels, np.ones(3))
    queue = DailyQueue.build(keys(*ids), days)
    outcome = r3_outcome(queue, truth, np.full(3, 0.5), Capacity(Fraction(1, 3), 1))
    assert outcome.counts["true_positives"] == (1 if fraud_id == first else 0)
    assert outcome.cutoff_ties == {"days": 1, "group_sizes": [3]}


def test_outcomes_do_not_depend_on_the_input_row_order() -> None:
    queue, truth, scores = small()
    order = np.random.default_rng(3).permutation(truth.rows)
    ids = keys(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)[order]
    shuffled = Truth.build(truth.days[order], truth.labels[order], truth.amounts[order])
    other = DailyQueue.build(ids, shuffled.days)
    first = r3_outcome(queue, truth, scores, CAPACITY)
    second = r3_outcome(other, shuffled, scores[order], CAPACITY)
    assert (first.counts, first.daily, first.cutoff_ties) == (
        second.counts,
        second.daily,
        second.cutoff_ties,
    )
    assert first.metrics == second.metrics
    assert ranking_outcome(truth, scores) == pytest.approx(ranking_outcome(shuffled, scores[order]))


def test_the_truth_owns_its_arrays() -> None:
    days, labels, amounts = keys(1, 1), np.array([1, 0], dtype=np.int8), np.ones(2)
    truth = Truth.build(days, labels, amounts)
    labels[1] = 1  # the caller's later change must not reach the cached fraud counts
    amounts[0] = 5.0
    assert truth.labels.tolist() == [1, 0]
    assert truth.amounts.tolist() == [1.0, 1.0]
    with pytest.raises(ValueError, match="read-only"):
        truth.labels[1] = 1
    queue = DailyQueue.build(keys(1, 2), days)
    outcome = r3_outcome(queue, truth, np.array([0.9, 0.1]), Capacity(Fraction(1, 2), 1))
    assert outcome.counts["frauds"] == 1
    assert outcome.metrics["recall"] == 1.0


def test_negative_amounts_and_overflowing_totals_are_validation_errors() -> None:
    labels = np.array([1, 1], dtype=np.int8)
    with pytest.raises(DataValidationError, match="negative"):
        Truth.build(keys(1, 1), labels, np.array([10.0, -9.0]))
    huge = Truth.build(keys(1, 1), labels, np.array([1e308, 1e308]))
    queue = DailyQueue.build(keys(1, 2), keys(1, 1))
    with pytest.raises(DataValidationError, match="too large"):
        r3_outcome(queue, huge, np.array([0.9, 0.1]), Capacity(Fraction(1, 2), 1))


def test_a_scalar_day_array_is_a_validation_error() -> None:
    with pytest.raises(DataValidationError, match="one-dimensional"):
        Truth.build(np.array(1, dtype=np.int64), np.array([1], dtype=np.int8), np.ones(1))


def test_the_ranking_measures_treat_float32_and_signed_zero_scores_like_float64() -> None:
    truth = Truth.build(keys(1, 1, 1), np.array([1, 0, 1], dtype=np.int8), np.ones(3))
    wide = ranking_outcome(truth, np.array([0.25, 0.25, 0.5]))
    narrow = ranking_outcome(truth, np.array([0.25, 0.25, 0.5], dtype=np.float32))
    assert narrow == wide
    zeros = Truth.build(keys(1, 1), np.array([1, 0], dtype=np.int8), np.ones(2))
    tied = ranking_outcome(zeros, np.array([0.0, -0.0]))
    assert tied == {"average_precision": 0.5, "roc_auc": 0.5}  # equal scores: one group


def test_r3_refuses_scores_other_than_float32_or_float64() -> None:
    queue, truth, scores = small()
    with pytest.raises(DataValidationError, match="floating point"):
        r3_outcome(queue, truth, scores.astype(np.float16), CAPACITY)


def test_a_perfect_ranking_has_measures_of_exactly_one_even_when_the_sums_round_up() -> None:
    days = keys(1, 1, 1, 1, 2, 2, 2, 2)
    labels = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8)
    truth = Truth.build(days, labels, np.ones(8))
    scores = np.array([0.9, 0.8, 0.3, 0.2, 0.95, 0.85, 0.3, 0.1])
    measures = ranking_outcome(truth, scores)
    assert measures["average_precision"] == 1.0
    assert measures["roc_auc"] == 1.0
    rng = np.random.default_rng(3)
    wide = rng.random(400) < 0.3
    labels = wide.astype(np.int8)
    truth = Truth.build(np.repeat(np.arange(1, 5, dtype=np.int64), 100), labels, np.ones(400))
    perfect = ranking_outcome(truth, labels * 0.5 + rng.random(400) * 0.01 + 0.4 * labels)
    average_precision = perfect["average_precision"]
    assert average_precision is not None
    assert 0.0 <= average_precision <= 1.0  # the contract's SHARE bound
