"""R0, R1 and R2 thresholds: hand cases, an exact brute-force oracle and the recorded errors."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from review_capacity.errors import DataValidationError
from review_capacity.evaluation.thresholds import (
    R0_THRESHOLD,
    alert_mask,
    select_r1,
    select_r2,
)


def labels(*values: int) -> np.ndarray:
    return np.array(values, dtype=np.int8)


def brute_force(scores: np.ndarray, y: np.ndarray) -> tuple[float, int, int]:
    """The best exact F1 over the distinct scores, with the highest threshold among equals."""
    frauds = int(y.sum())
    best: tuple[Fraction, float, int, int] | None = None
    for threshold in sorted({float(s) for s in scores}, reverse=True):
        flagged = scores >= threshold
        alerts, hits = int(flagged.sum()), int(y[flagged].sum())
        f1 = Fraction(2 * hits, frauds + alerts)
        if best is None or f1 > best[0]:
            best = (f1, threshold, alerts, hits)
    assert best is not None
    return best[1], best[2], best[3]


def test_r0_is_the_default_half_threshold_with_at_least_flagging() -> None:
    assert R0_THRESHOLD == 0.5
    mask = alert_mask(np.array([0.2, 0.5, 0.7]), R0_THRESHOLD)
    assert mask.tolist() == [False, True, True]


def test_r1_selects_the_f1_optimal_threshold_of_a_hand_example() -> None:
    # Scores 0.9 (fraud), 0.8 (legit), 0.7 (fraud), 0.1 (legit): P = 2.
    # a = 0.9: TP 1, A 1, F1 = 2/3; a = 0.8: TP 1, A 2, F1 = 1/2;
    # a = 0.7: TP 2, A 3, F1 = 4/5 (best); a = 0.1: TP 2, A 4, F1 = 2/3.
    selection = select_r1(np.array([0.9, 0.8, 0.7, 0.1]), labels(1, 0, 1, 0))
    assert selection.status == "selected"
    assert (selection.threshold, selection.alerts, selection.true_positives) == (0.7, 3, 2)
    assert (selection.false_positives, selection.false_negatives) == (1, 0)
    assert (selection.f1_numerator, selection.f1_denominator) == (4, 5)
    assert selection.distinct_scores == 4


def test_r1_never_separates_equal_scores() -> None:
    # Scores 0.9 (fraud), then 0.6 shared by a fraud, a legitimate and a fraud, then 0.1 (legit):
    # P = 3.  a = 0.9: 2/4;  a = 0.6: TP 3, A 4 -> 6/7 (best);  a = 0.1: TP 3, A 5 -> 6/8.
    scores = np.array([0.9, 0.6, 0.6, 0.6, 0.1])
    selection = select_r1(scores, labels(1, 0, 1, 1, 0))
    assert (selection.threshold, selection.alerts, selection.true_positives) == (0.6, 4, 3)


def test_r1_takes_the_highest_threshold_among_equal_f1_maxima() -> None:
    # P = 2.  a = 0.9: TP 1, A 1 -> 2/3;  a = 0.8: 2/4;  a = 0.75: 2/5;  a = 0.7: TP 2, A 4 -> 4/6.
    # The first and the last are exactly equal (2/3).
    scores = np.array([0.9, 0.8, 0.75, 0.7])
    selection = select_r1(scores, labels(1, 0, 0, 1))
    assert (selection.true_positives, selection.alerts) == (1, 1)
    assert selection.threshold == 0.9  # the highest of the tied candidates: the fewest alerts


def test_r1_allows_the_all_flag_optimum() -> None:
    # Every transaction is fraud: flagging all of them gives F1 = 1.
    selection = select_r1(np.array([0.4, 0.3, 0.2]), labels(1, 1, 1))
    assert (selection.threshold, selection.alerts) == (0.2, 3)
    assert selection.f1_numerator == 6
    assert selection.f1_denominator == 6


@pytest.mark.parametrize("seed", range(8))
def test_r1_matches_an_exact_brute_force_search(seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = 200
    y = (rng.random(rows) < 0.2).astype(np.int8)
    if y.sum() == 0:
        y[0] = 1
    scores = np.round(rng.random(rows) * 0.6 + 0.3 * y, 2)  # many exact ties
    expected_threshold, expected_alerts, expected_hits = brute_force(scores, y)
    selection = select_r1(scores, y)
    assert (selection.threshold, selection.alerts, selection.true_positives) == (
        expected_threshold,
        expected_alerts,
        expected_hits,
    )
    hits, misses = selection.true_positives, selection.false_negatives
    false_alerts = selection.false_positives
    assert hits is not None
    assert misses is not None
    assert false_alerts is not None
    assert hits + misses == int(y.sum())
    assert hits + false_alerts == selection.alerts


def test_r1_widens_float32_and_treats_signed_zero_as_zero() -> None:
    y = labels(1, 0, 1)
    wide = select_r1(np.array([0.5, 0.25, 0.0]), y)
    narrow = select_r1(np.array([0.5, 0.25, -0.0], dtype=np.float32), y)
    assert wide == narrow


@pytest.mark.parametrize(
    ("scores", "y", "error"),
    [
        (np.array([]), labels(), "empty_selection_set"),
        (np.array([0.1, np.nan]), labels(1, 0), "non_finite_scores"),
        (np.array([0.1, np.inf]), labels(1, 0), "non_finite_scores"),
        (np.array([0.1, 0.2]), labels(0, 0), "no_fraud"),
    ],
)
def test_r1_returns_a_recorded_error_without_a_threshold(
    scores: np.ndarray, y: np.ndarray, error: str
) -> None:
    selection = select_r1(scores, y)
    assert selection.status == "error"
    assert selection.error == error
    assert selection.threshold is None
    assert selection.alerts is None
    assert selection.f1_numerator is None
    assert selection.f1_denominator is None


def test_r1_refuses_malformed_arguments() -> None:
    with pytest.raises(DataValidationError, match="same rows"):
        select_r1(np.array([0.1, 0.2]), labels(1))
    with pytest.raises(DataValidationError, match="floating point"):
        select_r1(np.array([1, 2]), labels(1, 0))
    with pytest.raises(DataValidationError, match="int8"):
        select_r1(np.array([0.1, 0.2]), np.array([1, 0]))
    with pytest.raises(DataValidationError, match="0 or 1"):
        select_r1(np.array([0.1, 0.2]), labels(2, 0))


def test_r2_is_the_score_ranked_ceil_c_n() -> None:
    scores = np.arange(1, 101, dtype=np.float64) / 100  # N = 100
    chosen = select_r2(scores, Fraction(2, 100))
    assert (chosen.rank, chosen.threshold) == (2, 0.99)
    assert alert_mask(scores, chosen.threshold).sum() == 2


def test_r2_rounds_the_rank_up_exactly() -> None:
    scores = np.linspace(0.0, 1.0, 77_822)
    chosen = select_r2(scores, Fraction(2, 100))
    assert chosen.rank == 1557  # 0.02 x 77,822 = 1,556.44
    assert chosen.value == 0.02
    assert select_r2(scores, Fraction(1)).rank == 77_822


def test_r2_keeps_equal_scores_together_so_the_rate_can_exceed_c() -> None:
    scores = np.array([0.9, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    chosen = select_r2(scores, Fraction(2, 10))  # rank 2 has score 0.5, shared by three
    assert chosen.threshold == 0.5
    assert alert_mask(scores, chosen.threshold).sum() == 4


def test_r2_uses_scores_only_and_checks_them() -> None:
    with pytest.raises(DataValidationError, match="at least one"):
        select_r2(np.array([]), Fraction(1, 50))
    with pytest.raises(DataValidationError, match="finite"):
        select_r2(np.array([0.1, np.nan]), Fraction(1, 50))
    with pytest.raises(DataValidationError, match="share"):
        select_r2(np.array([0.1, 0.2]), Fraction(0))
    with pytest.raises(DataValidationError, match="share"):
        select_r2(np.array([0.1, 0.2]), Fraction(11, 10))
