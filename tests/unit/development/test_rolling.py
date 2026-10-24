"""The rolling-origin supplement on synthetic partitions (no project data): the rows of an
origin, the refit on the days before the gap, the R2 threshold set on the window's scores and its
transfer, for a tree-based and a neural configuration, on the CPU."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from review_capacity.config import SECONDS_PER_DAY, RollingOriginSpec
from review_capacity.development.inputs import PartitionRows, SearchInputs
from review_capacity.development.records import DevelopmentPartitions
from review_capacity.development.rolling import (
    OriginGeometry,
    refitted_neural,
    rows_between,
    run_rolling,
)
from review_capacity.development.run_output import RunDirectory, load_run
from review_capacity.development.scoring import ScoringContext
from review_capacity.development.search import search_configuration
from review_capacity.development.trial import SearchContext, TrialOutcome
from review_capacity.errors import DataValidationError, SpecificationError
from review_capacity.models.registry import Weighting, configuration
from review_capacity.models.v1_objective import V1Objective
from support.development import example
from support.runs import RUN_ID
from support.search import (
    RESNET_CANDIDATES,
    TRAIN_ROWS,
    V1_ROWS,
    V2_ROWS,
    candidate_file,
    candidate_list,
    mixed_setup,
    partition,
    scoring_context,
    search_context,
    synthetic_partitions,
)

LGB_U = configuration("LGB-u")
RES_U = configuration("RES-u")
SPEC = RollingOriginSpec(
    origins=(5, 7), gap_days=1, window_days=3, transfer_days=3, minimum_training_days=1
)
# Train covers days 1-8, V1 days 9-10 and V2 days 11-16: the second origin's transfer window
# (days 10-12) crosses from V1 into V2.
RANGES = {"train": (1, 8), "v1": (9, 10), "v2": (11, 16)}

pytestmark = pytest.mark.usefixtures("restore_lightgbm_logger")


def dated(name: str, rows: int, seed: int, first_id: int) -> PartitionRows:
    """A synthetic partition spread evenly over the days of ``RANGES[name]``, in time order."""
    first, last = RANGES[name]
    part = partition(name, rows, seed, first_id)
    days = first + (np.arange(rows) * (last - first + 1)) // rows
    offsets = np.sort(np.random.default_rng(seed).integers(0, SECONDS_PER_DAY, rows))
    times = np.sort(days * SECONDS_PER_DAY + offsets).astype(np.int64)
    order = np.argsort(times, kind="stable")
    return replace(part, times=times[order])


def parts() -> list[PartitionRows]:
    return [
        dated("train", TRAIN_ROWS, 1, 1_000_000),
        dated("v1", V1_ROWS, 2, 2_000_000),
        dated("v2", V2_ROWS, 3, 3_000_000),
    ]


def partitions() -> DevelopmentPartitions:
    base = synthetic_partitions()
    return replace(
        base,
        train=replace(base.train, first_day=1, last_day=8),
        v1=replace(base.v1, first_day=9, last_day=10),
        v2=replace(base.v2, first_day=11, last_day=16),
        rolling=SPEC,
    )


def state(root: Path) -> tuple[SearchContext, ScoringContext, TrialOutcome, list[PartitionRows]]:
    rows = parts()
    base = search_context(root / "base", write_run_record=False)
    run = RunDirectory.create(root, partitions(), "development", RUN_ID)
    run.write_record(example("run"))
    inputs = SearchInputs(rows[0], rows[1])
    objective = V1Objective(inputs.v1_data(), base.declaration.v1_daily_k)
    search = replace(base, run=run, inputs=inputs, objective=objective)
    result = search_configuration(search, LGB_U, candidate_list(), candidate_file(run))
    scoring = replace(scoring_context(search), inputs=replace(scoring_context(search).inputs))
    return search, scoring, result.selected, rows


def test_the_rows_of_a_day_range_cross_the_partitions_in_time_order() -> None:
    rows = parts()
    window = rows_between(rows, 7, 12)
    days = window.times // SECONDS_PER_DAY
    assert days.min() == 7
    assert days.max() == 12
    assert bool((np.diff(window.times) >= 0).all())
    expected = sum(
        int(((p.times // SECONDS_PER_DAY >= 7) & (p.times // SECONDS_PER_DAY <= 12)).sum())
        for p in rows
    )
    assert window.rows == expected
    assert set(window.features.columns) == set(rows[0].features.columns)
    with pytest.raises(DataValidationError, match="No rows"):
        rows_between(rows, 100, 101)


def test_the_geometry_of_an_origin_follows_the_declared_design() -> None:
    geometry = OriginGeometry.of(SPEC, 7, 1)
    assert geometry.training == (1, 5)
    assert geometry.window == (7, 9)
    assert geometry.transfer == (10, 12)


def test_a_tree_based_origin_refits_before_the_gap_and_transfers_the_threshold(
    tmp_path: Path,
) -> None:
    search, scoring, selected, rows = state(tmp_path)
    paths = run_rolling(search, scoring, LGB_U, selected, SPEC, rows)
    loaded = load_run(search.run.path, search.run.partitions)
    records = sorted(loaded.of_kind("rolling_origin"), key=lambda r: r["origin"])
    assert [p.name for p in paths] == [
        "rolling_origin-LGB-u-s5.json",
        "rolling_origin-LGB-u-s7.json",
    ]
    assert [r["origin"] for r in records] == [5, 7]
    for record in records:
        origin = record["origin"]
        training = record["training"]
        assert training["last_day"] == origin - 2  # the gap of one day
        train_rows = rows_between(rows[:1], 1, training["last_day"])
        assert training["rows"] == train_rows.rows
        assert training["frauds"] == train_rows.frauds
        assert record["fit"]["length"] == selected.fitted_iterations
        assert record["fit"]["source_fit"] == selected.fit
        scores = pd.read_parquet(search.run.path / record["scores"]["path"])
        window = rows_between(rows, origin, origin + 5)
        assert scores["TransactionID"].to_numpy().tolist() == window.ids.tolist()
        days = window.times // SECONDS_PER_DAY
        values = scores["score"].to_numpy()
        in_window = days <= origin + 2
        rank = math.ceil(0.02 * int(in_window.sum()))
        threshold = np.sort(values[in_window])[::-1][rank - 1]
        assert record["threshold"]["value"] == threshold
        assert record["threshold"]["rank"] == rank
        transfer = record["transfer"]
        flagged = values[~in_window] >= threshold
        assert transfer["alerts"] == int(flagged.sum())
        assert transfer["transactions"] == int((~in_window).sum())
        assert [row["day"] for row in transfer["daily"]] == [origin + 3, origin + 4, origin + 5]


def test_a_neural_origin_fits_its_preprocessing_on_the_training_days_only(
    tmp_path: Path,
) -> None:
    setup = mixed_setup(tmp_path, ("RES-u",))
    assert setup.neural is not None
    rows = parts()
    prefix = rows_between(rows[:1], 1, 3)
    inputs = SearchInputs(prefix, rows[1])
    neural = refitted_neural(
        setup.neural, setup.features, 0, inputs, rows[2], tmp_path / "rolling-preprocessing"
    )
    assert neural.artifact_sha256 != setup.neural.artifact_sha256
    assert neural.artifact.train_rows == prefix.rows
    assert neural.plans is setup.neural.plans
    weights = {w: data.weights for w, data in neural.train.items()}
    assert all(len(v) == prefix.rows for v in weights.values())
    balanced = inputs.train_data(Weighting.BALANCED).weights
    assert np.array_equal(weights[Weighting.BALANCED], balanced)
    assert not np.array_equal(
        balanced, setup.search_inputs.train_data(Weighting.BALANCED).weights[: len(balanced)]
    )
    assert len(neural.v1_inputs.numeric) == rows[1].rows


def test_a_neural_configuration_is_refitted_and_scored_with_its_origin_artifact(
    tmp_path: Path,
) -> None:
    setup = mixed_setup(tmp_path, ("RES-u",))
    rows = parts()
    base, scoring0, lgb_selected, _ = state(tmp_path / "tree")
    selected = replace(
        lgb_selected,
        configuration="RES-u",
        trial=1,
        hyperparameters=dict(RESNET_CANDIDATES[0]),
        fitted_iterations=2,
    )
    search = replace(
        base,
        declaration=setup.declaration,
        settings=setup.settings,
        neural=setup.neural,
    )
    scoring = replace(scoring0, declaration=setup.declaration, settings=setup.settings)
    run_rolling(search, scoring, RES_U, selected, replace(SPEC, origins=(7,)), rows)
    (record,) = load_run(search.run.path, search.run.partitions).of_kind("rolling_origin")
    assert record["configuration"] == "RES-u"
    assert record["training"]["preprocessing_sha256"] is not None
    assert setup.neural is not None
    assert record["training"]["preprocessing_sha256"] != setup.neural.artifact_sha256
    assert (search.run.path / "rolling-preprocessing" / "s7" / "manifest.json").exists()


def test_a_batch_plan_that_leaves_a_one_row_batch_stops_the_supplement_before_any_fit(
    tmp_path: Path,
) -> None:
    setup = mixed_setup(tmp_path, ("RES-u",))
    base, scoring0, lgb_selected, rows = state(tmp_path / "tree")
    selected = replace(
        lgb_selected, trial=1, hyperparameters=dict(RESNET_CANDIDATES[0]), fitted_iterations=2
    )
    search = replace(
        base, declaration=setup.declaration, settings=setup.settings, neural=setup.neural
    )
    scoring = replace(scoring0, declaration=setup.declaration, settings=setup.settings)
    before = len(load_run(search.run.path, search.run.partitions).records)
    with pytest.raises(SpecificationError, match=r"origin 5.*last micro-batch of 1 rows"):
        run_rolling(search, scoring, RES_U, selected, SPEC, rows)
    assert len(load_run(search.run.path, search.run.partitions).records) == before
    assert not (search.run.path / "rolling-preprocessing").exists()


def test_labels_after_an_origins_training_days_cannot_change_its_fit_or_threshold(
    tmp_path: Path,
) -> None:
    search, scoring, selected, rows = state(tmp_path / "a")
    spec = replace(SPEC, origins=(5,))  # trains on days 1-3
    run_rolling(search, scoring, LGB_U, selected, spec, rows)
    flipped = [
        replace(
            p,
            labels=np.where(p.times // SECONDS_PER_DAY >= 4, 1 - p.labels, p.labels).astype(
                np.int8
            ),
        )
        for p in rows
    ]
    other, other_scoring, _, _ = state(tmp_path / "b")
    run_rolling(other, other_scoring, LGB_U, selected, spec, flipped)
    (first,) = load_run(search.run.path, search.run.partitions).of_kind("rolling_origin")
    (second,) = load_run(other.run.path, other.run.partitions).of_kind("rolling_origin")
    assert first["training"] == second["training"]
    assert first["threshold"] == second["threshold"]
    assert first["transfer"] == second["transfer"]
    a = pd.read_parquet(search.run.path / first["scores"]["path"])
    b = pd.read_parquet(other.run.path / second["scores"]["path"])
    assert a.equals(b)
