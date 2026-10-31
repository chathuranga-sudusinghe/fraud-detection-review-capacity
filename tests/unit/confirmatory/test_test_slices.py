"""The slice inputs of the Test frauds come from label-free or development-only sources
(Issue #205)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from review_capacity.config import FeatureSpec
from review_capacity.confirmatory.inputs import TestInputs
from review_capacity.confirmatory.slices import SliceInputs, build_slice_inputs
from review_capacity.development.inputs import PartitionRows
from review_capacity.errors import DataValidationError
from review_capacity.evaluation.declaration import EvaluationDeclaration
from support import confirmatory as c

MISSING = "__MISSING__"
FEATURES = cast(FeatureSpec, SimpleNamespace(missing_label=MISSING))
DAY = 86_400


def frame(rows: int, rng: np.random.Generator, **fixed: object) -> pd.DataFrame:
    data: dict[str, object] = {
        "ProductCD": rng.choice(["W", "C"], rows),
        "card4": rng.choice(["visa", "mastercard"], rows),
        "card6": rng.choice(["debit", "credit"], rows),
        "DeviceType": rng.choice(["desktop", MISSING], rows),
        "TransactionAmt": rng.gamma(2.0, 50.0, rows),
        "card1": rng.choice(["1", "2", "3"], rows),
        "addr1": rng.choice(["10", MISSING], rows),
        "P_emaildomain": rng.choice(["gmail.com", MISSING], rows),
        "D1": rng.integers(0, 30, rows).astype(np.float64),
    }
    data.update(fixed)
    return pd.DataFrame(data)


def partition(name: str, rows: int, rng: np.random.Generator, **fixed: object) -> PartitionRows:
    labels = (rng.random(rows) < 0.5).astype(np.int8)
    times = np.sort(rng.integers(1, 100, rows)) * DAY
    ids = np.arange(rows, dtype=np.int64)
    return PartitionRows(name, frame(rows, rng, **fixed), labels, ids, times, None)  # type: ignore[arg-type]


@dataclass(frozen=True)
class World:
    declaration: EvaluationDeclaration
    test: TestInputs
    labels: npt.NDArray[np.int8]
    train: PartitionRows
    v2: PartitionRows


@pytest.fixture(scope="module")
def world() -> World:
    rng = np.random.default_rng(5)
    rows = 400
    days = rng.integers(155, 183, rows).astype(np.int64)
    test_frame = frame(rows, rng, ProductCD=rng.choice(["W", "C", "Z"], rows))
    # the first Test fraud repeats a Train fraud: same card, address, domain and start day 40
    test_frame.loc[0, "card1"] = "1"
    test_frame.loc[0, "addr1"] = "10"
    test_frame.loc[0, "P_emaildomain"] = "gmail.com"
    test_frame.loc[0, "D1"] = 120.0
    days[0] = 160
    test = TestInputs(
        test_frame,
        np.arange(rows, dtype=np.int64),
        days * DAY,
        days,
        "e" * 64,
        rng.random(rows) < 0.3,
    )
    labels = (rng.random(rows) < 0.3).astype(np.int8)
    labels[0] = 1
    train = partition("train", 800, rng)
    train.features.loc[0, "card1"] = "1"
    train.features.loc[0, "addr1"] = "10"
    train.features.loc[0, "P_emaildomain"] = "gmail.com"
    train.features.loc[0, "D1"] = 10.0
    train.labels[0] = 1
    train.times[0] = 50 * DAY
    v2 = partition("v2", 900, rng)
    v2.labels[:] = 1  # 900 frauds: ProductCD W is far above 100, RARE far below
    v2.features["ProductCD"] = np.where(np.arange(900) < 5, "RARE", "W")
    return World(c.declaration(), test, labels, train, v2)


def build(w: World) -> SliceInputs:
    return build_slice_inputs(w.declaration, FEATURES, w.test, w.labels, w.train, w.v2)


def test_every_variable_has_one_code_for_each_test_fraud(world: World) -> None:
    result = build(world)
    frauds = int((world.labels == 1).sum())
    assert [v.name for v in result.variables] == [
        "identity",
        "ProductCD",
        "card4",
        "card6",
        "DeviceType",
        "amount_band",
        "week",
        "entity",
    ]
    for variable in result.variables:
        assert variable.codes.shape == (frauds,)
        assert variable.codes.min() >= 0
        assert variable.codes.max() < len(variable.levels)
    assert result.entity_index.shape == (frauds,)


def test_identity_comes_from_the_join_and_levels_from_v2_pooling(world: World) -> None:
    result = build(world)
    fraud = world.labels == 1
    identity = next(v for v in result.variables if v.name == "identity")
    assert world.test.has_identity is not None
    assert identity.codes.tolist() == world.test.has_identity[fraud].astype(int).tolist()
    product = next(v for v in result.variables if v.name == "ProductCD")
    assert product.levels == ("W", "other")  # RARE has 5 V2 frauds; C and Z were never kept
    assert set(result.boundaries.kept["ProductCD"]) == {"W"}
    text = world.test.features["ProductCD"].to_numpy()[fraud]
    assert (product.codes == (text != "W")).all()


def test_the_amount_edges_are_the_train_quintiles_and_the_weeks_are_declared(world: World) -> None:
    result = build(world)
    expected = np.quantile(world.train.features["TransactionAmt"], [0.2, 0.4, 0.6, 0.8])
    assert result.boundaries.amount_edges == pytest.approx(tuple(expected))
    week = next(v for v in result.variables if v.name == "week")
    assert set(week.codes.tolist()) <= {0, 1, 2, 3}


def test_a_fraud_of_a_train_fraud_entity_is_recurring_and_equal_keys_share_an_index(
    world: World,
) -> None:
    result = build(world)
    entity = next(v for v in result.variables if v.name == "entity")
    assert entity.levels == ("new", "recurring")
    assert entity.codes[0] == 1  # the first Test fraud repeats a Train fraud's entity
    assert sorted(set(result.entity_index.tolist())) == list(
        range(int(result.entity_index.max()) + 1)
    )


def test_inputs_are_checked(world: World) -> None:
    short = dataclasses.replace(world, labels=np.zeros(399, dtype=np.int8))
    with pytest.raises(DataValidationError, match="one label and one join flag"):
        build(short)
    none = dataclasses.replace(world, labels=np.zeros(400, dtype=np.int8))
    with pytest.raises(DataValidationError, match="no fraud"):
        build(none)
