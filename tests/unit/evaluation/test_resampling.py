"""The resampling engines: streams, the pigeonhole and fraud-level bootstraps, the non-circular day
blocks and the day-level ratios, on small synthetic inputs (no data is read)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from review_capacity.errors import DataValidationError, SpecificationError
from review_capacity.evaluation import resampling
from review_capacity.evaluation.bootstrap import SEED, entity_indices, paired_difference
from review_capacity.evaluation.declaration import EvaluationDeclaration, load_declaration
from review_capacity.evaluation.resampling import (
    day_block_indices,
    day_block_sets,
    day_ratios,
    h3_day_block_sets,
    naive_ratios,
    naive_weights,
    pigeonhole_ratios,
    pigeonhole_weights,
    stream_generator,
)
from support.synthetic import REPO_PATHS


@pytest.fixture(scope="module")
def declaration() -> EvaluationDeclaration:
    return load_declaration(REPO_PATHS)[0]


def frauds(n: int = 60, days: int = 6, entities: int = 20) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(1)
    day_index = rng.integers(0, days, n).astype(np.intp)
    entity_index = entity_indices([str(e) for e in rng.integers(0, entities, n)])
    capture_a = rng.random(n) < 0.7
    capture_b = rng.random(n) < 0.6
    return capture_a, capture_b, day_index, entity_index


# --- streams -------------------------------------------------------------------------------------


def test_a_stream_restarts_identically_on_every_call(declaration: EvaluationDeclaration) -> None:
    first = stream_generator(declaration, "day_block_h3").random(5)
    again = stream_generator(declaration, "day_block_h3").random(5)
    assert np.array_equal(first, again)


def test_streams_are_the_children_of_the_root_seed_in_the_declared_order(
    declaration: EvaluationDeclaration,
) -> None:
    streams = declaration.resampling.streams
    children = np.random.SeedSequence(declaration.resampling.root_seed).spawn(len(streams))
    for index, name in enumerate(streams):
        expected = np.random.Generator(np.random.PCG64(children[index])).random(3)
        assert np.array_equal(stream_generator(declaration, name).random(3), expected)


def test_different_streams_give_different_draws(declaration: EvaluationDeclaration) -> None:
    draws = [
        stream_generator(declaration, name).random(4) for name in declaration.resampling.streams
    ]
    assert len({tuple(draw) for draw in draws}) == len(draws)


def test_an_unknown_stream_is_rejected(declaration: EvaluationDeclaration) -> None:
    with pytest.raises(SpecificationError, match="Unknown resampling stream"):
        stream_generator(declaration, "no_such_stream")


# --- pigeonhole ----------------------------------------------------------------------------------


def test_the_engine_reproduces_the_development_paired_bootstrap() -> None:
    a, b, day, entity = frauds()
    replicates = pigeonhole_ratios(
        np.random.default_rng(SEED), np.vstack([a, b]), day, 6, entity, replicates=300
    )
    interval = paired_difference(a, b, day, 6, entity, replicates=300)
    difference = replicates.ratios[:, 0] - replicates.ratios[:, 1]
    defined = difference[np.isfinite(difference)]
    lower, upper = np.percentile(defined, [2.5, 97.5], method="linear")
    assert replicates.undefined_count == interval.undefined_replicates
    assert lower == pytest.approx(interval.lower, rel=1e-9, abs=1e-12)
    assert upper == pytest.approx(interval.upper, rel=1e-9, abs=1e-12)


def test_the_result_does_not_depend_on_the_block_size(monkeypatch: pytest.MonkeyPatch) -> None:
    a, b, day, entity = frauds()
    captures = np.vstack([a, b])
    wide = pigeonhole_ratios(np.random.default_rng(3), captures, day, 6, entity, replicates=300)
    monkeypatch.setattr(resampling, "BLOCK", 7)
    narrow = pigeonhole_ratios(np.random.default_rng(3), captures, day, 6, entity, replicates=300)
    assert np.array_equal(wide.ratios, narrow.ratios, equal_nan=True)


def test_the_weights_are_the_product_of_day_and_entity_multiplicities() -> None:
    days, entities = 4, 5
    day_index = np.repeat(np.arange(days), entities).astype(np.intp)
    entity_index = np.tile(np.arange(entities), days).astype(np.intp)
    blocks = list(
        pigeonhole_weights(np.random.default_rng(5), day_index, days, entity_index, replicates=40)
    )
    weights = np.vstack(blocks)
    assert weights.shape == (40, days * entities)
    assert (weights.sum(axis=1) == days * entities).all()  # (sum of a_d) * (sum of b_e)
    grid = weights.reshape(40, days, entities)
    for replicate in grid:  # a rank-one table: every row is a multiple of the same vector
        assert np.linalg.matrix_rank(replicate) <= 1


def test_every_row_uses_the_same_weights() -> None:
    a, b, day, entity = frauds()
    together = pigeonhole_ratios(
        np.random.default_rng(9), np.vstack([a, b]), day, 6, entity, replicates=200
    )
    alone = pigeonhole_ratios(
        np.random.default_rng(9), a[np.newaxis], day, 6, entity, replicates=200
    )
    assert np.array_equal(together.ratios[:, :1], alone.ratios, equal_nan=True)


def test_numerators_and_denominators_give_the_weighted_ratio() -> None:
    _, _, day, entity = frauds()
    rng = np.random.default_rng(2)
    amount = rng.random(60) + 0.5
    missed = (rng.random(60) < 0.4).astype(np.float64)
    ratios = pigeonhole_ratios(
        np.random.default_rng(11),
        (missed * amount)[np.newaxis],
        day,
        6,
        entity,
        replicates=120,
        denominators=amount[np.newaxis],
    )
    weights = np.vstack(
        list(pigeonhole_weights(np.random.default_rng(11), day, 6, entity, replicates=120))
    )
    expected = (weights @ (missed * amount)) / (weights @ amount)
    assert ratios.ratios[:, 0] == pytest.approx(expected)


def test_the_replicate_recalls_centre_on_the_point_estimate() -> None:
    a, _, day, entity = frauds(n=400, days=8, entities=120)
    ratios = pigeonhole_ratios(
        np.random.default_rng(4), a[np.newaxis], day, 8, entity, replicates=2000
    )
    assert ratios.ratios.mean() == pytest.approx(a.mean(), abs=0.03)
    assert ratios.undefined_count == 0


def test_replicates_without_weight_are_undefined() -> None:
    n = 12
    day = np.zeros(n, dtype=np.intp)  # every fraud on day 0 of 6: undefined when day 0 is not drawn
    entity = entity_indices([str(i % 4) for i in range(n)])
    capture = (np.arange(n) % 2 == 0)[np.newaxis]
    result = pigeonhole_ratios(np.random.default_rng(6), capture, day, 6, entity, replicates=400)
    assert result.undefined_count > 0
    assert np.isnan(result.ratios[result.undefined]).all()
    assert np.isfinite(result.ratios[~result.undefined]).all()
    assert result.undefined_count == int(np.isnan(result.ratios[:, 0]).sum())


def test_the_inputs_are_checked() -> None:
    a, b, day, entity = frauds()
    rng = np.random.default_rng(0)
    captures = np.vstack([a, b])
    with pytest.raises(DataValidationError, match="at least one replicate"):
        pigeonhole_ratios(rng, captures, day, 6, entity, replicates=0)
    with pytest.raises(DataValidationError, match="one of the partition's days"):
        pigeonhole_ratios(rng, captures, day, 3, entity, replicates=5)
    with pytest.raises(DataValidationError, match="without gaps"):
        pigeonhole_ratios(rng, captures, day, 6, entity + 1, replicates=5)
    with pytest.raises(DataValidationError, match="60 columns"):
        pigeonhole_ratios(rng, captures[:, :10], day, 6, entity, replicates=5)
    with pytest.raises(DataValidationError, match="shape of the numerators"):
        pigeonhole_ratios(rng, captures, day, 6, entity, replicates=5, denominators=captures[:1])
    with pytest.raises(DataValidationError, match="must not be negative"):
        pigeonhole_ratios(
            rng, captures, day, 6, entity, replicates=5, denominators=-captures.astype(np.float64)
        )
    with pytest.raises(DataValidationError, match="at least one fraud"):
        pigeonhole_ratios(rng, np.zeros((1, 0)), day[:0], 6, entity[:0], replicates=5)


# --- fraud-level bootstrap -----------------------------------------------------------------------


def test_fraud_level_multiplicities_sum_to_the_fraud_count() -> None:
    weights = np.vstack(list(naive_weights(np.random.default_rng(8), 30, replicates=50)))
    assert weights.shape == (50, 30)
    assert (weights.sum(axis=1) == 30).all()


def test_fraud_level_ratios_are_defined_and_centre_on_the_estimate() -> None:
    a, _, _, _ = frauds(n=300)
    ratios = naive_ratios(np.random.default_rng(10), a[np.newaxis], replicates=1500)
    assert ratios.undefined_count == 0
    assert ratios.ratios.mean() == pytest.approx(a.mean(), abs=0.02)
    weights = np.vstack(list(naive_weights(np.random.default_rng(10), 300, replicates=1500)))
    assert ratios.ratios[:, 0] == pytest.approx((weights @ a.astype(np.float64)) / 300)


def test_the_fraud_level_inputs_are_checked() -> None:
    with pytest.raises(DataValidationError, match="at least one fraud"):
        list(naive_weights(np.random.default_rng(0), 0, replicates=5))
    with pytest.raises(DataValidationError, match="one column per fraud"):
        naive_ratios(np.random.default_rng(0), np.zeros(5), replicates=5)


# --- day blocks ----------------------------------------------------------------------------------


@pytest.mark.parametrize("length", [1, 3, 7, 28])
def test_day_blocks_are_consecutive_non_circular_and_cut_to_the_days(length: int) -> None:
    days = 28
    indices = day_block_indices(np.random.default_rng(12), days, length, replicates=400)
    assert indices.shape == (400, days)
    assert indices.min() >= 0
    assert indices.max() < days
    for start in range(0, days, length):  # each block holds consecutive days, never wrapping
        block = indices[:, start : start + length]
        assert (np.diff(block, axis=1) == 1).all()
        assert (block[:, 0] <= days - length).all()


def test_the_last_block_is_cut_when_the_length_does_not_divide_the_days() -> None:
    indices = day_block_indices(np.random.default_rng(13), 28, 3, replicates=300)
    assert indices.shape[1] == 28  # ten blocks of three, the last two days dropped
    starts = indices[:, ::3]
    assert starts.shape[1] == 10
    assert starts.max() <= 28 - 3


def test_a_block_length_of_one_resamples_single_days() -> None:
    indices = day_block_indices(np.random.default_rng(14), 28, 1, replicates=2000)
    counts = np.bincount(indices.ravel(), minlength=28)
    assert counts.min() > 0  # every day can be drawn
    assert counts.std() / counts.mean() < 0.1


def test_the_block_length_must_fit_the_days() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(DataValidationError, match="between 1 and the 28 days"):
        day_block_indices(rng, 28, 0, replicates=5)
    with pytest.raises(DataValidationError, match="between 1 and the 28 days"):
        day_block_indices(rng, 28, 29, replicates=5)
    with pytest.raises(DataValidationError, match="at least one replicate"):
        day_block_indices(rng, 28, 3, replicates=0)


def test_the_h3_sets_follow_the_stream_in_the_order_three_one_seven(
    declaration: EvaluationDeclaration,
) -> None:
    sets = h3_day_block_sets(declaration, 28)
    assert [s.block_length for s in sets] == [3, 1, 7]
    replicates = declaration.resampling.replicates
    assert all(s.indices.shape == (replicates, 28) for s in sets)
    rng = stream_generator(declaration, "day_block_h3")
    for block_set in sets:  # the sets are consecutive draws of one stream
        expected = day_block_indices(rng, 28, block_set.block_length, replicates=replicates)
        assert np.array_equal(block_set.indices, expected)
    assert not np.array_equal(sets[0].indices, sets[1].indices)


def test_day_block_sets_draw_one_after_the_other() -> None:
    rng = np.random.default_rng(15)
    sets = day_block_sets(rng, 28, (3, 7), replicates=20)
    again = np.random.default_rng(15)
    assert np.array_equal(sets[0].indices, day_block_indices(again, 28, 3, replicates=20))
    assert np.array_equal(sets[1].indices, day_block_indices(again, 28, 7, replicates=20))


# --- day-level ratios ----------------------------------------------------------------------------


def test_day_ratios_are_the_ratio_of_the_resampled_sums() -> None:
    numerators = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    denominators = np.array([10.0, 10.0, 10.0, 10.0])
    indices = np.array([[0, 0, 1, 1], [2, 3, 3, 3]], dtype=np.intp)
    result = day_ratios(numerators, denominators, indices)
    assert result.ratios == pytest.approx(np.array([[6 / 40, 14 / 40], [15 / 40, 5 / 40]]))
    # replicate 0 takes days 0, 0, 1, 1 and replicate 1 takes days 2, 3, 3, 3; the columns follow
    # the numerator rows: 1+1+2+2 = 6 and 4+4+3+3 = 14 (replicate 0), 3+4+4+4 = 15 and
    # 2+1+1+1 = 5 (replicate 1), each over 40


def test_day_ratios_accept_a_denominator_for_each_quantity() -> None:
    numerators = np.array([[1.0, 2.0], [3.0, 4.0]])
    denominators = np.array([[2.0, 2.0], [4.0, 8.0]])
    indices = np.array([[0, 1], [1, 1]], dtype=np.intp)
    assert day_ratios(numerators, denominators, indices).ratios == pytest.approx(
        np.array([[3 / 4, 7 / 12], [4 / 4, 8 / 16]])
    )


def test_day_ratios_without_a_denominator_are_undefined() -> None:
    numerators = np.array([[1.0, 2.0, 3.0]])
    denominators = np.array([0.0, 0.0, 5.0])
    indices = np.array([[0, 1, 0], [2, 2, 2]], dtype=np.intp)
    result = day_ratios(numerators, denominators, indices)
    assert np.isnan(result.ratios[0, 0])
    assert result.ratios[1, 0] == pytest.approx(9 / 15)
    assert result.undefined_count == 1


def test_day_ratios_check_their_inputs() -> None:
    numerators = np.ones((1, 3))
    with pytest.raises(DataValidationError, match="replicates x days"):
        day_ratios(numerators, np.ones(3), np.zeros(3, dtype=np.intp))
    with pytest.raises(DataValidationError, match="one of the days"):
        day_ratios(numerators, np.ones(3), np.array([[0, 1, 3]], dtype=np.intp))
    with pytest.raises(DataValidationError, match="3 columns"):
        day_ratios(np.ones((1, 4)), np.ones(3), np.zeros((2, 3), dtype=np.intp))
    with pytest.raises(DataValidationError, match="shape of the numerators"):
        day_ratios(numerators, np.ones((2, 3)), np.zeros((2, 3), dtype=np.intp))


def test_h3_style_ratios_over_the_declared_block_sets(declaration: EvaluationDeclaration) -> None:
    rng = np.random.default_rng(16)
    volume = rng.integers(2500, 3000, 28).astype(np.float64)
    alerts = np.vstack([volume * rate for rate in (0.019, 0.021)])
    for block_set in h3_day_block_sets(declaration, 28):
        rates = day_ratios(alerts, volume, block_set.indices)
        assert rates.undefined_count == 0
        assert rates.ratios.mean(axis=0) == pytest.approx([0.019, 0.021], abs=1e-6)


def test_replicates_are_immutable_records() -> None:
    result = day_ratios(np.ones((1, 2)), np.ones(2), np.zeros((3, 2), dtype=np.intp))
    assert result.replicates == 3
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.ratios = np.zeros(1)  # type: ignore[misc]


def test_a_test_sized_run_completes_with_the_declared_replicates(
    declaration: EvaluationDeclaration,
) -> None:
    rng = np.random.default_rng(21)
    frauds_in_test, days, entities = 2660, 28, 1350  # the projected Test size (protocol)
    day_index = rng.integers(0, days, frauds_in_test).astype(np.intp)
    entity_index = entity_indices([str(e) for e in rng.integers(0, entities, frauds_in_test)])
    captures = rng.random((8, frauds_in_test)) < 0.4
    result = pigeonhole_ratios(
        stream_generator(declaration, "pigeonhole_h1_h2"),
        captures,
        day_index,
        days,
        entity_index,
        replicates=declaration.resampling.replicates,
    )
    assert result.ratios.shape == (declaration.resampling.replicates, 8)
    assert result.undefined_count == 0
    assert result.ratios.mean() == pytest.approx(0.4, abs=0.01)
