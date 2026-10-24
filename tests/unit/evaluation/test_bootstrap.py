"""The paired two-way bootstrap of the fraud-capture difference."""

from __future__ import annotations

import numpy as np
import pytest

from review_capacity.errors import DataValidationError
from review_capacity.evaluation import bootstrap
from review_capacity.evaluation.bootstrap import entity_indices, paired_difference


def frauds(n: int = 60, days: int = 6, entities: int = 20) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(1)
    day_index = rng.integers(0, days, n).astype(np.intp)
    entity_index = entity_indices([str(e) for e in rng.integers(0, entities, n)])
    capture_a = rng.random(n) < 0.7
    capture_b = rng.random(n) < 0.6
    return capture_a, capture_b, day_index, entity_index


def test_identical_captures_give_a_zero_interval() -> None:
    a, _, day, entity = frauds()
    interval = paired_difference(a, a.copy(), day, 6, entity, replicates=200)
    assert (interval.estimate, interval.lower, interval.upper) == (0.0, 0.0, 0.0)


def test_the_estimate_is_the_difference_of_recalls() -> None:
    a, b, day, entity = frauds()
    interval = paired_difference(a, b, day, 6, entity, replicates=200)
    assert interval.estimate == pytest.approx(a.mean() - b.mean())
    assert interval.lower <= interval.upper
    assert interval.frauds == 60
    assert interval.days == 6
    assert interval.entities == int(entity.max()) + 1


def test_the_same_seed_gives_the_same_interval_and_another_seed_another() -> None:
    a, b, day, entity = frauds()
    first = paired_difference(a, b, day, 6, entity, replicates=300, seed=7)
    again = paired_difference(a, b, day, 6, entity, replicates=300, seed=7)
    other = paired_difference(a, b, day, 6, entity, replicates=300, seed=8)
    assert first == again
    assert (first.lower, first.upper) != (other.lower, other.upper)


def test_the_result_does_not_depend_on_the_block_size(monkeypatch: pytest.MonkeyPatch) -> None:
    a, b, day, entity = frauds()
    wide = paired_difference(a, b, day, 6, entity, replicates=300)
    monkeypatch.setattr(bootstrap, "BLOCK", 7)
    narrow = paired_difference(a, b, day, 6, entity, replicates=300)
    assert wide == narrow


def test_the_defaults_are_those_of_dr_006() -> None:
    assert (bootstrap.REPLICATES, bootstrap.SEED, bootstrap.LEVEL) == (10_000, 42, 0.95)


def test_days_without_a_fraud_still_take_part_in_the_draw() -> None:
    a, b, day, entity = frauds(days=3)
    narrow = paired_difference(a, b, day, 3, entity, replicates=400)
    wide = paired_difference(a, b, day, 12, entity, replicates=400)  # nine days hold no fraud
    assert wide.days == 12
    assert wide.undefined_replicates >= narrow.undefined_replicates


def test_an_interval_that_excludes_a_large_constant_difference() -> None:
    n = 200
    day = np.arange(n, dtype=np.intp) % 10
    entity = np.arange(n, dtype=np.intp)
    a = np.ones(n, dtype=bool)
    b = np.zeros(n, dtype=bool)
    interval = paired_difference(a, b, day, 10, entity, replicates=200)
    assert (interval.lower, interval.upper) == (1.0, 1.0)


@pytest.mark.parametrize(
    "change",
    [
        lambda a, b, d, e: (a.astype(int), b, d, e),
        lambda a, b, d, e: (a, b[:-1], d, e),
        lambda a, b, d, e: (a, b, d + 99, e),
        lambda a, b, d, e: (a[:0], b[:0], d[:0], e[:0]),
        lambda a, b, d, e: (a, b, d, e + 1),
    ],
)
def test_malformed_inputs_are_refused(change: object) -> None:
    a, b, day, entity = frauds()
    args = change(a, b, day, entity)  # type: ignore[operator]
    with pytest.raises(DataValidationError):
        paired_difference(args[0], args[1], args[2], 6, args[3], replicates=50)


def test_entity_indices_number_keys_by_first_appearance() -> None:
    assert entity_indices(["b", "a", "b", "c"]).tolist() == [0, 1, 0, 2]


def test_the_entity_weights_enter_the_replicates() -> None:
    # One day and two entities: a = [captured, not], b = [not, captured]. The entity draws
    # (2, 0), (1, 1) and (0, 2) give differences +1, 0 and -1, so the interval is [-1, 1]; a
    # bootstrap over days alone would give 0 in every replicate.
    a = np.array([True, False])
    b = np.array([False, True])
    day = np.zeros(2, dtype=np.intp)
    entity = np.array([0, 1], dtype=np.intp)
    interval = paired_difference(a, b, day, 1, entity, replicates=2_000)
    assert (interval.lower, interval.upper) == (-1.0, 1.0)
    assert interval.estimate == 0.0


def test_the_replicates_follow_a_direct_calculation_from_the_same_draws() -> None:
    a, b, day, entity = frauds(n=12, days=3, entities=5)
    days, entities, replicates = 3, int(entity.max()) + 1, 200
    rng = np.random.default_rng(11)
    day_draws = rng.multinomial(days, np.full(days, 1 / days), size=replicates)
    entity_draws = rng.multinomial(entities, np.full(entities, 1 / entities), size=replicates)
    values = []
    for r in range(replicates):
        weights = [day_draws[r][day[i]] * entity_draws[r][entity[i]] for i in range(len(day))]
        total = sum(weights)
        if total > 0:
            values.append(sum(w * (int(a[i]) - int(b[i])) for i, w in enumerate(weights)) / total)
    lower, upper = np.percentile(values, [2.5, 97.5], method="linear")
    interval = paired_difference(a, b, day, days, entity, replicates=replicates, seed=11)
    assert interval.lower == pytest.approx(lower)
    assert interval.upper == pytest.approx(upper)
    assert interval.undefined_replicates == replicates - len(values)


def test_a_replicate_without_weight_is_counted_not_hidden() -> None:
    # Every fraud is on day 0 of two days: the replicates that draw day 1 twice have no weight.
    a = np.array([True, False, True, False])
    b = np.array([False, False, True, True])
    day = np.zeros(4, dtype=np.intp)
    entity = np.array([0, 1, 2, 3], dtype=np.intp)
    interval = paired_difference(a, b, day, 2, entity, replicates=2_000)
    assert 300 < interval.undefined_replicates < 700  # about a quarter of 2,000
