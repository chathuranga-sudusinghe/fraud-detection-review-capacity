"""The false-negative slices and their intervals on a synthetic partition (Issue #201; no data
read)."""

from __future__ import annotations

import dataclasses
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from review_capacity.errors import DataValidationError
from review_capacity.evaluation.declaration import EvaluationDeclaration, load_declaration
from review_capacity.evaluation.false_negatives import (
    OTHER,
    Boundaries,
    SliceVariable,
    amount_codes,
    amount_edges,
    check_boundaries,
    declared_levels,
    false_negative_analysis,
    misses_of,
    pooled_variable,
    pooling_from_v2,
    score_band_codes,
    week_codes,
)
from review_capacity.evaluation.outcomes import Partition
from review_capacity.evaluation.queue import DailyQueue
from review_capacity.evaluation.ranking import Capacity, Truth, r3_outcome
from review_capacity.evaluation.resampling import (
    pigeonhole_ratios,
    pigeonhole_weights,
    stream_generator,
)
from support.synthetic import REPO_PATHS

DAYS, PER_DAY, B = 28, 400, 200
CAPACITY = Capacity(Fraction(1, 50), 6)


@pytest.fixture(scope="module")
def declaration() -> EvaluationDeclaration:
    base = load_declaration(REPO_PATHS)[0]
    return dataclasses.replace(base, resampling=dataclasses.replace(base.resampling, replicates=B))


@pytest.fixture(scope="module")
def world() -> tuple[Partition, dict[str, np.ndarray]]:
    rng = np.random.default_rng(41)
    days = np.repeat(np.arange(155, 155 + DAYS), PER_DAY).astype(np.int64)
    labels = (rng.random(len(days)) < 0.04).astype(np.int8)
    amounts = np.round(rng.gamma(2.0, 60.0, len(days)), 2)
    truth = Truth.build(days, labels, amounts)
    _, entities = np.unique(rng.integers(0, 80, int(labels.sum())), return_inverse=True)
    partition = Partition(
        truth,
        DailyQueue.build(np.arange(len(days), dtype=np.int64), days),
        entities.astype(np.intp),
        250.0,
    )
    scores = {
        "LGB-b": np.round(np.clip(rng.normal(0.1 + 0.5 * labels, 0.2), 0, 1), 3),
        "Cat-b": np.round(np.clip(rng.normal(0.15 + 0.4 * labels, 0.2), 0, 1), 3),
    }
    return partition, scores


def variables(partition: Partition, declaration: EvaluationDeclaration) -> list[SliceVariable]:
    rng = np.random.default_rng(42)
    frauds = int(partition.truth.frauds.sum())
    fraud_days = partition.truth.days[partition.truth.labels == 1]
    amounts = partition.truth.amounts[partition.truth.labels == 1]
    rule = declaration.false_negative
    edges = amount_edges(partition.truth.amounts, rule.amount_quantiles)
    levels = ("A", "B", "C", "D")
    kept = {
        name: {"A": 400, "B": 300, "C": 120}
        for name in ("ProductCD", "card4", "card6", "DeviceType")
    }
    boundaries = Boundaries(edges, kept)
    check_boundaries(boundaries, declaration)
    return [
        SliceVariable(
            "identity", ("absent", "present"), rng.integers(0, 2, frauds).astype(np.intp)
        ),
        *(
            pooled_variable(name, np.array(rng.choice(levels, frauds)), boundaries)
            for name in ("ProductCD", "card4", "card6", "DeviceType")
        ),
        SliceVariable(
            "amount_band",
            declared_levels(declaration)["amount_band"],
            amount_codes(amounts, edges),
        ),
        SliceVariable(
            "week",
            declared_levels(declaration)["week"],
            week_codes(fraud_days, rule.test_weeks),
        ),
        SliceVariable("entity", ("new", "recurring"), rng.integers(0, 2, frauds).astype(np.intp)),
    ]


def test_the_misses_are_the_false_negatives_of_the_r3_outcome(
    world: tuple[Partition, dict[str, np.ndarray]],
) -> None:
    partition, scores = world
    found = misses_of(partition, scores["LGB-b"], CAPACITY)
    direct = r3_outcome(partition.queue, partition.truth, scores["LGB-b"], CAPACITY)
    assert int(found.missed.sum()) == direct.counts["false_negatives"]
    assert (found.day_rank[found.missed] >= CAPACITY.k).all()
    assert (found.day_rank[~found.missed] < CAPACITY.k).all()


def test_every_fraud_lies_in_one_level_and_the_levels_sum_to_the_totals(
    declaration: EvaluationDeclaration, world: tuple[Partition, dict[str, np.ndarray]]
) -> None:
    partition, scores = world
    result = false_negative_analysis(
        declaration,
        partition,
        variables(partition, declaration),
        {name: misses_of(partition, s, CAPACITY) for name, s in scores.items()},
        CAPACITY,
    )
    for table in result.values():
        for variable in {s["variable"] for s in table["slices"]}:
            rows = [s for s in table["slices"] if s["variable"] == variable]
            if variable == "score_band":
                assert sum(r["misses"] for r in rows) == table["false_negatives"]
            else:
                assert sum(r["frauds"] for r in rows) == table["frauds"]
                assert sum(r["misses"] for r in rows) == table["false_negatives"]
            assert sum(r["share"] for r in rows) == pytest.approx(1.0)


def test_the_slice_interval_is_the_pigeonhole_interval_of_the_shared_weights(
    declaration: EvaluationDeclaration, world: tuple[Partition, dict[str, np.ndarray]]
) -> None:
    partition, scores = world
    slices = variables(partition, declaration)
    found = {name: misses_of(partition, s, CAPACITY) for name, s in scores.items()}
    result = false_negative_analysis(declaration, partition, slices, found, CAPACITY)
    row = next(
        s for s in result["Cat-b"]["slices"] if s["variable"] == "week" and s["level"] == "155-161"
    )
    mask = slices[6].codes == 0
    ratios = pigeonhole_ratios(
        stream_generator(declaration, "pigeonhole_false_negative"),
        (mask & found["Cat-b"].missed).astype(np.float64)[np.newaxis],
        partition.fraud_day_index,
        DAYS,
        partition.entity_index,
        replicates=B,
        denominators=mask.astype(np.float64)[np.newaxis],
    ).ratios[:, 0]
    tail = 100 * (1 - declaration.false_negative.interval_level) / 2
    lower, upper = np.percentile(ratios[np.isfinite(ratios)], [tail, 100 - tail])
    assert row["status"] == "interval"
    assert row["interval"]["lower"] == pytest.approx(lower)
    assert row["interval"]["upper"] == pytest.approx(upper)
    assert row["interval"]["undefined"] == int((~np.isfinite(ratios)).sum())
    assert row["miss_rate"] == pytest.approx(
        int((mask & found["Cat-b"].missed).sum()) / int(mask.sum())
    )


def test_a_small_slice_has_no_interval_and_the_score_band_has_no_miss_rate(
    declaration: EvaluationDeclaration, world: tuple[Partition, dict[str, np.ndarray]]
) -> None:
    partition, scores = world
    result = false_negative_analysis(
        declaration,
        partition,
        variables(partition, declaration),
        {"LGB-b": misses_of(partition, scores["LGB-b"], CAPACITY)},
        CAPACITY,
    )["LGB-b"]
    small = [s for s in result["slices"] if s["frauds"] < 100 and s["variable"] != "score_band"]
    assert small
    assert all(s["interval"] is None and s["status"] == "fewer_than_minimum_frauds" for s in small)
    bands = [s for s in result["slices"] if s["variable"] == "score_band"]
    assert [b["level"] for b in bands] == ["K+1 to 2K", "2K+1 to 5K", "above 5K"]
    assert all(b["miss_rate"] is None and b["interval"] is None for b in bands)
    assert all(b["status"] == "not_applicable" for b in bands)


def test_the_boundaries_do_not_depend_on_labels() -> None:
    amounts = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    edges = amount_edges(amounts, 5)
    assert edges == pytest.approx((2.8, 4.6, 6.4, 8.2))
    assert amount_codes(np.array([2.8, 2.79, 10.0, 0.0]), edges).tolist() == [1, 0, 4, 0]
    values = np.array(["a"] * 120 + ["b"] * 99 + ["c"] * 100 + ["d"] * 5)
    assert pooling_from_v2(values, 100) == {"a": 120, "c": 100}
    with pytest.raises(DataValidationError, match="collide"):
        pooling_from_v2(np.array([OTHER] * 100), 100)
    assert score_band_codes(np.array([56, 111, 112, 279, 280, 1000]), 56, (1, 2, 5)).tolist() == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    weeks = ((155, 161), (162, 168))
    assert week_codes(np.array([155, 161, 162, 168]), weeks).tolist() == [0, 0, 1, 1]
    with pytest.raises(DataValidationError, match="declared weeks"):
        week_codes(np.array([169]), weeks)


def test_a_test_level_never_seen_in_v2_is_other() -> None:
    boundaries = Boundaries(
        (1.0, 2.0, 3.0, 4.0), {n: {"A": 150} for n in ("ProductCD", "card4", "card6", "DeviceType")}
    )
    variable = pooled_variable("card4", np.array(["A", "Z", "A", "B"]), boundaries)
    assert variable.levels == ("A", OTHER)
    assert variable.codes.tolist() == [0, 1, 0, 1]


def test_inputs_are_checked(
    declaration: EvaluationDeclaration, world: tuple[Partition, dict[str, Any]]
) -> None:
    partition, scores = world
    good = variables(partition, declaration)
    found = {"LGB-b": misses_of(partition, scores["LGB-b"], CAPACITY)}
    with pytest.raises(DataValidationError, match="in that order"):
        false_negative_analysis(declaration, partition, good[:-1], found, CAPACITY)
    bad = [*good[:-1], dataclasses.replace(good[-1], codes=good[-1].codes + 5)]
    with pytest.raises(DataValidationError, match="outside its levels"):
        false_negative_analysis(declaration, partition, bad, found, CAPACITY)
    short = {"LGB-b": dataclasses.replace(found["LGB-b"], missed=found["LGB-b"].missed[:-1])}
    with pytest.raises(DataValidationError, match="one entry"):
        false_negative_analysis(declaration, partition, good, short, CAPACITY)
    with pytest.raises(DataValidationError, match="increasing edges"):
        check_boundaries(Boundaries((2.0, 1.0, 3.0, 4.0), good_kept()), declaration)
    with pytest.raises(DataValidationError, match="not kept"):
        check_boundaries(Boundaries((1.0, 2.0, 3.0, 4.0), good_kept({"A": 5})), declaration)


def good_kept(level: dict[str, int] | None = None) -> dict[str, dict[str, int]]:
    return {n: dict(level or {"A": 150}) for n in ("ProductCD", "card4", "card6", "DeviceType")}


def test_the_slice_interval_equals_a_direct_computation_from_the_weights(
    declaration: EvaluationDeclaration, world: tuple[Partition, dict[str, np.ndarray]]
) -> None:
    """The stream-7 day and entity multiplicities are taken from the weight generator and the
    ratios are formed directly, with empty slices, a slice without misses and the replicates in
    which a slice has no weight (the intervals are forced by a minimum of one fraud)."""
    partition, scores = world
    one = dataclasses.replace(
        declaration,
        false_negative=dataclasses.replace(declaration.false_negative, minimum_frauds=1),
    )
    slices = variables(partition, one)
    frauds = int(partition.truth.frauds.sum())
    rare = np.zeros(frauds, dtype=np.intp)
    rare[:2] = 1  # a slice of two frauds: some replicates give it no weight
    slices[-1] = SliceVariable("entity", ("new", "recurring"), rare)
    found = {"LGB-b": misses_of(partition, scores["LGB-b"], CAPACITY)}
    table = false_negative_analysis(one, partition, slices, found, CAPACITY)["LGB-b"]
    weights = np.vstack(
        list(
            pigeonhole_weights(
                stream_generator(one, "pigeonhole_false_negative"),
                partition.fraud_day_index,
                DAYS,
                partition.entity_index,
                replicates=B,
            )
        )
    )
    tail = 100 * (1 - one.false_negative.interval_level) / 2
    for variable in slices:
        for index, level in enumerate(variable.levels):
            row = next(
                s for s in table["slices"] if s["variable"] == variable.name and s["level"] == level
            )
            mask = variable.codes == index
            top = weights @ (mask & found["LGB-b"].missed).astype(np.float64)
            bottom = weights @ mask.astype(np.float64)
            if not mask.any():
                assert row["frauds"] == 0
                assert row["miss_rate"] is None
                continue
            defined = bottom > 0
            draws = top[defined] / bottom[defined]
            lower, upper = np.percentile(draws, [tail, 100 - tail])
            assert row["interval"]["lower"] == pytest.approx(lower)
            assert row["interval"]["upper"] == pytest.approx(upper)
            assert row["interval"]["undefined"] == int((~defined).sum())
    rare_row = next(
        s for s in table["slices"] if s["variable"] == "entity" and s["level"] == "recurring"
    )
    assert rare_row["frauds"] == 2
    assert (
        rare_row["interval"]["undefined"] > 0
    )  # replicates in which the two frauds have no weight


def test_a_slice_without_misses_has_a_zero_miss_rate_and_a_zero_share(
    declaration: EvaluationDeclaration, world: tuple[Partition, dict[str, np.ndarray]]
) -> None:
    partition, scores = world
    slices = variables(partition, declaration)
    found = misses_of(partition, scores["LGB-b"], CAPACITY)
    frauds = int(partition.truth.frauds.sum())
    codes = np.zeros(frauds, dtype=np.intp)
    codes[~found.missed] = 1  # level 1 holds exactly the captured frauds
    slices[-1] = SliceVariable("entity", ("new", "recurring"), codes)
    table = false_negative_analysis(declaration, partition, slices, {"LGB-b": found}, CAPACITY)
    captured = next(
        s
        for s in table["LGB-b"]["slices"]
        if s["variable"] == "entity" and s["level"] == "recurring"
    )
    assert captured["misses"] == 0
    assert captured["miss_rate"] == 0.0
    assert captured["share"] == 0.0
