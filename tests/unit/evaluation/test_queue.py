"""The shared daily review queue and V1 objective (synthetic scores; no data read)."""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pytest

from review_capacity.errors import DataValidationError
from review_capacity.evaluation import queue
from review_capacity.evaluation.queue import DailyQueue, id_digest, tie_ranks


def keys(*values: int) -> np.ndarray:
    return np.array(values, dtype=np.int64)


def labels(*values: int) -> np.ndarray:
    return np.array(values, dtype=np.int8)


def reference_order(scores: list[float], ids: list[int], days: list[int]) -> list[int]:
    """Row positions in queue order, by the rule written out in plain Python."""
    return sorted(
        range(len(ids)),
        key=lambda row: (
            days[row],
            -scores[row],
            hashlib.sha256(str(ids[row]).encode("utf-8")).digest(),
            ids[row],
        ),
    )


def ids_with_unlike_orders() -> tuple[int, int]:
    """Two TransactionIDs whose digest order differs from their numeric order."""
    for first in range(1, 50):
        if id_digest(first) > id_digest(first + 1):
            return first, first + 1
    raise AssertionError("no such pair among 1..50")


def test_digest_is_sha256_of_the_decimal_string() -> None:
    assert id_digest(3_000_123) == hashlib.sha256(b"3000123").digest()
    assert id_digest(np.int64(7)) == hashlib.sha256(b"7").digest()
    assert id_digest(10) != id_digest(1)


def test_tie_ranks_follow_the_digest_then_the_id() -> None:
    ids = keys(5, 1, 9, 2, 7)
    ranks = tie_ranks(ids)
    assert sorted(ranks.tolist()) == list(range(5))
    by_rank = [int(i) for _, i in sorted(zip(ranks.tolist(), ids.tolist(), strict=True))]
    assert by_rank == sorted(ids.tolist(), key=lambda i: (id_digest(i), i))


def test_the_id_breaks_an_identical_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(queue, "id_digest", lambda _: b"same")
    assert tie_ranks(keys(30, 10, 20)).tolist() == [2, 0, 1]


def test_tie_order_is_not_ascending_id() -> None:
    low, high = ids_with_unlike_orders()
    q = DailyQueue.build(keys(low, high), keys(1, 1))
    first = q.order(np.array([0.5, 0.5]))[0]
    assert (low, high)[first] == high  # the digest, not the identifier, decides


def test_queue_orders_by_day_then_score_descending() -> None:
    scores = np.array([0.1, 0.9, 0.5, 0.7, 0.3])
    q = DailyQueue.build(keys(1, 2, 3, 4, 5), keys(2, 1, 2, 1, 2))
    expected = reference_order(scores.tolist(), [1, 2, 3, 4, 5], [2, 1, 2, 1, 2])
    assert q.order(scores).tolist() == [1, 3, 2, 4, 0]  # day 1: 0.9, 0.7; day 2: 0.5, 0.3, 0.1
    assert q.order(scores).tolist() == expected


def test_equal_scores_are_ordered_by_the_tie_rule() -> None:
    ids = [101, 7, 55, 2024, 9]
    scores = [0.4, 0.4, 0.4, 0.4, 0.4]
    q = DailyQueue.build(np.array(ids, dtype=np.int64), keys(1, 1, 1, 1, 1))
    order = q.order(np.array(scores))
    assert [ids[i] for i in order] == sorted(ids, key=lambda i: (id_digest(i), i))


def test_equal_scores_are_compared_exactly() -> None:
    q = DailyQueue.build(keys(1, 2), keys(1, 1))
    nearly = np.nextafter(0.5, 1.0)
    assert q.order(np.array([0.5, nearly])).tolist() == [1, 0]  # the larger score first
    assert q.order(np.array([nearly, 0.5])).tolist() == [0, 1]


def test_negative_zero_equals_zero_and_float32_widens_exactly() -> None:
    low, high = ids_with_unlike_orders()
    q = DailyQueue.build(keys(low, high), keys(1, 1))
    assert q.order(np.array([-0.0, 0.0])).tolist() == q.order(np.array([0.0, 0.0])).tolist()
    single = np.array([0.1, 0.7], dtype=np.float32)
    assert q.order(single).tolist() == q.order(single.astype(np.float64)).tolist()
    widened = single.astype(np.float64)
    assert q.recall(single, labels(1, 0), 1) == q.recall(widened, labels(1, 0), 1)


def test_selection_takes_the_first_k_of_every_day() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    q = DailyQueue.build(keys(*range(1, 8)), keys(1, 1, 1, 2, 2, 3, 3))
    selection = q.select(scores, 2)
    assert selection.mask.tolist() == [True, True, False, True, True, True, True]
    assert selection.day_rank.tolist() == [0, 1, 2, 0, 1, 0, 1]
    assert [(d.day, d.transactions, d.selected) for d in selection.days] == [
        (1, 3, 2),
        (2, 2, 2),
        (3, 2, 2),
    ]


def test_a_day_with_fewer_rows_than_k_is_selected_whole() -> None:
    q = DailyQueue.build(keys(1, 2, 3), keys(1, 1, 2))
    selection = q.select(np.array([0.3, 0.2, 0.1]), 5)
    assert selection.mask.all()
    assert [(d.transactions, d.selected, d.cutoff_tied_group) for d in selection.days] == [
        (2, 2, None),
        (1, 1, None),
    ]


def test_days_need_not_be_consecutive_or_sorted() -> None:
    q = DailyQueue.build(keys(1, 2, 3, 4), keys(9, 3, 9, 3))
    selection = q.select(np.array([0.1, 0.2, 0.3, 0.4]), 1)
    assert selection.mask.tolist() == [False, False, True, True]
    assert [d.day for d in selection.days] == [3, 9]


def test_a_tie_crossing_the_cutoff_reports_its_whole_group() -> None:
    scores = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
    q = DailyQueue.build(keys(1, 2, 3, 4, 5), keys(1, 1, 1, 1, 1))
    crossing = {k: q.select(scores, k).days[0].cutoff_tied_group for k in range(1, 7)}
    assert crossing == {1: None, 2: 3, 3: 3, 4: None, 5: None, 6: None}
    selection = q.select(scores, 2)
    assert (selection.cutoff_tie_days, selection.cutoff_tie_group_sizes) == (1, (3,))


def test_the_tied_member_inside_the_queue_is_the_first_in_tie_order() -> None:
    ids = [11, 12, 13, 14]
    q = DailyQueue.build(np.array(ids, dtype=np.int64), keys(1, 1, 1, 1))
    selection = q.select(np.array([0.5, 0.5, 0.5, 0.5]), 1)
    chosen = ids[int(np.flatnonzero(selection.mask)[0])]
    assert chosen == min(ids, key=lambda i: (id_digest(i), i))


def test_ties_are_counted_per_day() -> None:
    scores = np.array([0.5, 0.5, 0.2, 0.9, 0.9, 0.9])
    q = DailyQueue.build(keys(*range(1, 7)), keys(1, 1, 1, 2, 2, 2))
    selection = q.select(scores, 1)
    assert [d.cutoff_tied_group for d in selection.days] == [2, 3]
    assert selection.cutoff_tie_group_sizes == (2, 3)


def test_recall_counts_frauds_inside_the_daily_queues() -> None:
    scores = np.array([0.9, 0.8, 0.1, 0.7, 0.6, 0.2])
    fraud = labels(1, 0, 1, 0, 1, 0)
    q = DailyQueue.build(keys(*range(1, 7)), keys(1, 1, 1, 2, 2, 2))
    result = q.recall(scores, fraud, 1)
    assert (result.numerator, result.frauds) == (1, 3)
    assert result.value == 1 / 3
    assert q.recall(scores, fraud, 2).numerator == 2  # day 1: rows 0, 1; day 2: rows 3, 4
    assert q.recall(scores, fraud, 3).numerator == 3


def test_recall_is_unweighted_and_pools_the_days() -> None:
    scores = np.array([0.9, 0.1, 0.9, 0.1])
    fraud = labels(1, 1, 0, 1)
    q = DailyQueue.build(keys(1, 2, 3, 4), keys(1, 1, 2, 2))
    assert q.recall(scores, fraud, 1).value == 1 / 3  # one of three frauds; no weighting


def test_days_without_fraud_add_nothing_to_either_sum() -> None:
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    fraud = labels(0, 0, 1, 0)
    q = DailyQueue.build(keys(1, 2, 3, 4), keys(1, 1, 2, 2))
    result = q.recall(scores, fraud, 1)
    assert (result.numerator, result.frauds) == (1, 1)


def test_recall_is_undefined_without_fraud() -> None:
    q = DailyQueue.build(keys(1, 2), keys(1, 1))
    with pytest.raises(DataValidationError, match="no fraud"):
        q.recall(np.array([0.5, 0.4]), labels(0, 0), 1)


def test_the_objective_does_not_depend_on_row_order() -> None:
    rng = np.random.default_rng(0)
    rows = 600
    ids = rng.permutation(np.arange(1000, 1000 + rows)).astype(np.int64)
    days = rng.integers(1, 7, rows).astype(np.int64)
    scores = np.round(rng.random(rows), 1)  # many exact ties
    fraud = (rng.random(rows) < 0.2).astype(np.int8)
    base = DailyQueue.build(ids, days).recall(scores, fraud, 12)
    shuffle = rng.permutation(rows)
    again = DailyQueue.build(ids[shuffle], days[shuffle]).recall(
        scores[shuffle], fraud[shuffle], 12
    )
    assert again == base


@pytest.mark.parametrize("seed", range(5))
def test_queue_matches_a_plain_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = 400
    ids = rng.choice(10**9, rows, replace=False).astype(np.int64)
    days = rng.integers(1, 6, rows).astype(np.int64)
    scores = np.round(rng.random(rows), 1)
    fraud = (rng.random(rows) < 0.3).astype(np.int8)
    k = 15
    q = DailyQueue.build(ids, days)
    expected_order = reference_order(scores.tolist(), ids.tolist(), days.tolist())
    assert q.order(scores).tolist() == expected_order
    selected = set()
    position = dict.fromkeys(range(1, 6), 0)
    for row in expected_order:
        if position[int(days[row])] < k:
            selected.add(row)
        position[int(days[row])] += 1
    selection = q.select(scores, k)
    assert set(np.flatnonzero(selection.mask).tolist()) == selected
    assert q.recall(scores, fraud, k).numerator == int(fraud[sorted(selected)].sum())


def test_a_v1_sized_partition_selects_62_per_day() -> None:
    rng = np.random.default_rng(1)
    days = np.repeat(np.arange(106, 127, dtype=np.int64), 3121)
    ids = np.arange(len(days), dtype=np.int64) + 3_000_000
    scores = rng.random(len(days))
    selection = DailyQueue.build(ids, days).select(scores, 62)
    assert selection.mask.sum() == 21 * 62
    assert {d.selected for d in selection.days} == {62}


def test_inputs_are_checked() -> None:
    with pytest.raises(DataValidationError, match="one length"):
        DailyQueue.build(keys(1, 2), keys(1))
    with pytest.raises(DataValidationError, match="one length"):
        DailyQueue.build(keys(), keys())
    with pytest.raises(DataValidationError, match="int64"):
        DailyQueue.build(np.array([1, 2], dtype=np.int32), keys(1, 1))
    with pytest.raises(DataValidationError, match="unique"):
        DailyQueue.build(keys(1, 1), keys(1, 1))
    with pytest.raises(DataValidationError, match="start at 1"):
        DailyQueue.build(keys(1, 2), keys(0, 1))


def test_scores_k_and_labels_are_checked() -> None:
    q = DailyQueue.build(keys(1, 2), keys(1, 1))
    for bad in (np.array([0.5, np.nan]), np.array([0.5, np.inf])):
        with pytest.raises(DataValidationError, match="finite"):
            q.select(bad, 1)
    with pytest.raises(DataValidationError, match="floating point"):
        q.select(np.array([1, 0]), 1)
    with pytest.raises(DataValidationError, match="2 rows"):
        q.select(np.array([0.5]), 1)
    for k in (0, -1, True, 1.5):
        with pytest.raises(DataValidationError, match="positive integer"):
            q.select(np.array([0.5, 0.4]), k)  # type: ignore[arg-type]
    for bad_labels in (np.array([1, 0]), labels(1, 2), labels(1, 0, 0)):
        with pytest.raises(DataValidationError, match="Labels"):
            q.recall(np.array([0.5, 0.4]), bad_labels, 1)


def test_building_logs_its_step(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="review_capacity.evaluation.queue"):
        DailyQueue.build(keys(1, 2), keys(1, 1))
    assert "Building the daily review queue of 2 transactions: done" in caplog.text


def test_the_queue_copies_its_days() -> None:
    days = keys(1, 1)
    q = DailyQueue.build(keys(1, 2), days)
    days[0] = 9
    assert q.select(np.array([0.5, 0.4]), 1).days[0].day == 1


def test_the_queue_exposes_its_days_read_only() -> None:
    days = keys(1, 1, 2)
    exposed = DailyQueue.build(keys(1, 2, 3), days).days
    assert exposed.tolist() == [1, 1, 2]
    with pytest.raises(ValueError, match="read-only"):
        exposed[0] = 9
