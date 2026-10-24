"""Uncapped threshold outcomes against the fixed daily budget, and their records."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from review_capacity.development.records import RECORD_CONTRACTS, dump_record, validate_record
from review_capacity.development.schema import loads
from review_capacity.errors import DataValidationError
from review_capacity.evaluation.ranking import Capacity, Subject, Truth, capacity_budgets
from review_capacity.evaluation.thresholds import select_r1, select_r2
from review_capacity.evaluation.workload import (
    r1_selection_record,
    threshold_outcome,
    threshold_record,
)
from support.development import IDENTITY, PARTITIONS, file

SUBJECT = Subject("LGB-u", "selected", 42, file("records/score-LGB-u-selected-V2.json"), "V2")


def keys(*values: int) -> np.ndarray:
    return np.array(values, dtype=np.int64)


def small() -> tuple[Truth, np.ndarray]:
    """Day 1: 4 transactions (frauds at scores 0.9, 0.3); day 2: 3 transactions (fraud at 0.6)."""
    days = keys(1, 1, 1, 1, 2, 2, 2)
    labels = np.array([1, 0, 1, 0, 0, 1, 0], dtype=np.int8)
    scores = np.array([0.9, 0.8, 0.3, 0.1, 0.7, 0.6, 0.2])
    return Truth.build(days, labels, np.ones(7)), scores


def test_counts_daily_rows_and_measures_match_the_hand_calculation() -> None:
    truth, scores = small()
    outcome = threshold_outcome(truth, scores, 0.5, Capacity(Fraction(1, 50), 2))
    # Alerts (score >= 0.5): day 1 -> 0.9, 0.8; day 2 -> 0.7, 0.6.  Frauds among them: 1 and 1.
    assert outcome.counts == {
        "days": 2,
        "transactions": 7,
        "frauds": 3,
        "selected": 4,
        "true_positives": 2,
        "false_positives": 2,
        "false_negatives": 1,
    }
    assert [(r["selected"], r["true_positives"]) for r in outcome.daily] == [(2, 1), (2, 1)]
    assert outcome.metrics["recall"] == pytest.approx(2 / 3)
    assert outcome.metrics["precision"] == pytest.approx(2 / 4)
    assert outcome.metrics["f1"] == pytest.approx(2 * 2 / (3 + 4))


def test_capacity_relative_to_prevalence_is_reported_with_every_result() -> None:
    truth, scores = small()  # prevalence 3 / 7
    outcome = threshold_outcome(truth, scores, 0.5, Capacity(Fraction(2, 100), 2))
    assert outcome.metrics["capacity_to_prevalence"] == pytest.approx(0.02 / (3 / 7))


def test_overflow_unused_and_load_ratio_follow_dr_009() -> None:
    truth, scores = small()
    outcome = threshold_outcome(truth, scores, 0.5, Capacity(Fraction(1, 50), 1))  # K = 1
    assert [(r["overflow"], r["unused"], r["load_ratio"]) for r in outcome.daily] == [
        (1, 0, 2.0),
        (1, 0, 2.0),
    ]
    assert outcome.metrics["total_overflow"] == 2
    assert outcome.metrics["total_unused"] == 0
    assert outcome.metrics["days_above_budget"] == 2
    assert outcome.metrics["days_below_budget"] == 0
    assert outcome.metrics["mean_load_ratio"] == 2.0
    # Demand is never truncated to K: the alerts stay 2 a day, and frauds in excess of K are
    # not counted as false negatives (only the fraud below the threshold is).
    assert outcome.counts["selected"] == 4
    assert outcome.counts["false_negatives"] == 1


@pytest.mark.parametrize("k", [1, 2, 3, 50])
def test_the_signed_gap_splits_into_overflow_and_unused_capacity(k: int) -> None:
    truth, scores = small()
    outcome = threshold_outcome(truth, scores, 0.25, Capacity(Fraction(1, 50), k))
    for row in outcome.daily:
        assert row["selected"] - k == row["overflow"] - row["unused"]
        assert row["overflow"] * row["unused"] == 0
        assert row["oracle_reviewable"] == min(k, row["frauds"])


def test_a_day_without_alerts_is_counted_and_precision_is_undefined_without_alerts() -> None:
    truth, scores = small()
    outcome = threshold_outcome(truth, scores, 0.75, Capacity(Fraction(1, 50), 2))
    assert [r["selected"] for r in outcome.daily] == [2, 0]
    assert outcome.metrics["zero_alert_days"] == 1
    assert outcome.metrics["days_below_budget"] == 1
    none = threshold_outcome(truth, scores, 2.0, Capacity(Fraction(1, 50), 2))
    assert none.metrics["precision"] is None
    assert none.metrics["recall"] == 0
    assert none.metrics["f1"] == 0
    assert none.metrics["zero_alert_days"] == 2


def test_the_alert_set_includes_scores_equal_to_the_threshold() -> None:
    truth, scores = small()
    outcome = threshold_outcome(truth, scores, 0.6, Capacity(Fraction(1, 50), 2))
    assert outcome.counts["selected"] == 4  # 0.9, 0.8, 0.7 and the 0.6 itself


def test_the_scores_must_pair_with_the_rows() -> None:
    truth, scores = small()
    with pytest.raises(DataValidationError, match="7 scores"):
        threshold_outcome(truth, scores[:-1], 0.5, Capacity(Fraction(1, 50), 2))


def v2_sized() -> tuple[Truth, np.ndarray]:
    v2 = PARTITIONS.v2
    n_days = v2.last_day - v2.first_day + 1
    per_day = np.full(n_days, v2.rows // n_days)
    per_day[: v2.rows % n_days] += 1
    days = np.repeat(np.arange(v2.first_day, v2.last_day + 1, dtype=np.int64), per_day)
    rng = np.random.default_rng(0)
    labels = (rng.random(v2.rows) < 0.034).astype(np.int8)
    scores = np.clip(rng.random(v2.rows) * 0.4 + 0.5 * labels, 0, 1)
    return Truth.build(days, labels, rng.uniform(1.0, 500.0, v2.rows)), scores


def test_records_satisfy_their_contracts_on_a_v2_sized_partition() -> None:
    truth, scores = v2_sized()
    selection = select_r1(scores, truth.labels)
    assert selection.threshold is not None
    days = {"first": truth.first_day, "last": truth.last_day}
    selection_record = r1_selection_record(
        IDENTITY, "LGB-u", file("records/score.json"), "primary", days, selection
    )
    validate_record(selection_record, PARTITIONS, "r1_selection")
    assert selection_record["status"] == "selected"
    for capacity in capacity_budgets(PARTITIONS.v2):
        thresholds = {
            "R0": 0.5,
            "R1": selection.threshold,
            "R2": select_r2(scores, capacity.share).threshold,
        }
        for policy, threshold in thresholds.items():
            outcome = threshold_outcome(truth, scores, threshold, capacity)
            record = threshold_record(
                IDENTITY, SUBJECT, truth, policy, threshold, capacity, outcome
            )
            validate_record(record, PARTITIONS, "evaluation")
            assert loads(dump_record(record, PARTITIONS), "record") == record
            assert outcome.counts["transactions"] == PARTITIONS.v2.rows
            assert len(record["daily"]) == 28
        r2 = threshold_outcome(truth, scores, thresholds["R2"], capacity)
        assert r2.counts["selected"] >= round(capacity.value * PARTITIONS.v2.rows)


def test_a_failed_selection_is_recorded_as_an_error_record() -> None:
    labels = np.zeros(4, dtype=np.int8)
    selection = select_r1(np.array([0.1, 0.2, 0.3, 0.4]), labels)
    days = {"first": 127, "last": 147}
    record = r1_selection_record(
        IDENTITY, "LGB-u", file("records/score.json"), "latency", days, selection
    )
    assert record["status"] == "error"
    assert record["error"] == "no_fraud"
    assert RECORD_CONTRACTS["r1_selection"].problems(record) == []


def test_only_the_threshold_policies_are_built_here() -> None:
    truth, scores = small()
    capacity = Capacity(Fraction(1, 50), 2)
    outcome = threshold_outcome(truth, scores, 0.5, capacity)
    with pytest.raises(DataValidationError, match="not a threshold policy"):
        threshold_record(IDENTITY, SUBJECT, truth, "R3", 0.5, capacity, outcome)
