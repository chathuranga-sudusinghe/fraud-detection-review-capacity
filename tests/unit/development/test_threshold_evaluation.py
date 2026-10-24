"""R0, R1 and R2 on the saved scores of a selected model: LightGBM on synthetic data on the CPU
(no project data)."""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from review_capacity.development.evaluation import truth_of
from review_capacity.development.run_output import load_run
from review_capacity.development.scoring import (
    ScoredPartition,
    ScoringContext,
    load_selected,
    read_scores,
    score_partition,
)
from review_capacity.development.thresholds import evaluate_thresholds, saved_partition
from review_capacity.development.trial import TrialOutcome, run_file
from review_capacity.errors import DataValidationError, ModelError, RecordContractError
from review_capacity.evaluation.ranking import capacity_budgets
from review_capacity.evaluation.thresholds import select_r1
from review_capacity.models.registry import configuration
from support.search import V1_ROWS, V2_ROWS, searched

LGB_U = configuration("LGB-u")

pytestmark = pytest.mark.usefixtures("restore_lightgbm_logger")


@pytest.fixture
def scored(tmp_path: Path) -> tuple[ScoringContext, TrialOutcome]:
    return searched(tmp_path)


def score_both(
    context: ScoringContext, selected: TrialOutcome
) -> tuple[ScoredPartition, ScoredPartition]:
    model = load_selected(context, LGB_U, selected)
    return (
        score_partition(context, LGB_U, selected, model, "V1"),
        score_partition(context, LGB_U, selected, model, "V2"),
    )


def evaluate(context: ScoringContext, v1: ScoredPartition, v2: ScoredPartition) -> dict[str, Path]:
    return evaluate_thresholds(
        context.run,
        context.run,
        context.run.partitions,
        context.identity,
        context.seed,
        v1,
        v2,
    )


def test_the_policies_are_recorded_where_the_protocol_evaluates_them(
    scored: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, selected = scored
    v1, v2 = score_both(context, selected)
    written = evaluate(context, v1, v2)
    loaded = load_run(context.run.path, context.run.partitions)
    evaluations = [r for r in loaded.of_kind("evaluation") if r["policy"] in ("R0", "R1", "R2")]
    cells = sorted((r["policy"], r["partition"], r["period"]) for r in evaluations)
    expected = (
        [("R0", "V1", "primary")] * 4
        + [("R0", "V2", "primary")] * 4
        + [("R1", "V2", "primary")] * 4
        + [("R1", "V2", "latency")] * 4
        + [("R2", "V2", "primary")] * 4
    )
    assert cells == sorted(expected)  # R1 and R2 are not applied to V1; R2 has no latency variant
    assert len(written) == 20 + 2
    budgets = capacity_budgets(context.run.partitions.v2)
    for record in evaluations:
        assert record["k"] in [b.k for b in budgets]
        assert record["score"] in (v1.record, v2.record)
        assert record["model_role"] == "selected"
        assert record["seed"] == context.seed
        assert record["cutoff_ties"] is None


def test_r1_is_selected_on_all_of_v2_and_on_the_fitting_days(
    scored: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, selected = scored
    v1, v2 = score_both(context, selected)
    evaluate(context, v1, v2)
    loaded = load_run(context.run.path, context.run.partitions)
    by_period = {r["period"]: r for r in loaded.of_kind("r1_selection")}
    assert set(by_period) == {"primary", "latency"}
    truth, scores = truth_of(v2.rows), read_scores(context.run, v2.file, v2.rows)
    expected = select_r1(scores, truth.labels)
    primary = by_period["primary"]
    assert (primary["threshold"], primary["alerts"], primary["transactions"]) == (
        expected.threshold,
        expected.alerts,
        V2_ROWS,
    )
    assert primary["selection_days"] == {"first": 1, "last": 3}
    latency = by_period["latency"]
    assert latency["selection_days"] == {"first": 1, "last": 2}
    held_out = [
        r for r in loaded.of_kind("evaluation") if r["policy"] == "R1" and r["period"] == "latency"
    ]
    assert {(r["days"]["first"], r["days"]["last"]) for r in held_out} == {(3, 3)}
    assert {r["threshold"] for r in held_out} == {latency["threshold"]}


def test_r2_thresholds_are_the_order_statistics_of_the_saved_v2_scores(
    scored: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, selected = scored
    v1, v2 = score_both(context, selected)
    evaluate(context, v1, v2)
    scores = read_scores(context.run, v2.file, v2.rows)
    loaded = load_run(context.run.path, context.run.partitions)
    r2 = [r for r in loaded.of_kind("evaluation") if r["policy"] == "R2"]
    assert len(r2) == 4
    for record in r2:
        rank = math.ceil(Fraction(round(record["capacity"] * 100), 100) * V2_ROWS)
        expected = float(np.sort(scores)[::-1][rank - 1])
        assert record["threshold"] == expected
        assert record["counts"]["selected"] == int((scores >= expected).sum())
        assert record["counts"]["selected"] >= rank


def test_r0_is_evaluated_on_both_partitions_at_one_half(
    scored: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, selected = scored
    v1, v2 = score_both(context, selected)
    evaluate(context, v1, v2)
    loaded = load_run(context.run.path, context.run.partitions)
    r0 = [r for r in loaded.of_kind("evaluation") if r["policy"] == "R0"]
    assert {r["threshold"] for r in r0} == {0.5}
    rows = {"V1": V1_ROWS, "V2": V2_ROWS}
    assert all(r["counts"]["transactions"] == rows[r["partition"]] for r in r0)


def test_scores_that_do_not_pair_with_the_rows_are_refused_before_anything_is_written(
    scored: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, selected = scored
    v1, v2 = score_both(context, selected)
    path = context.run.path / v2.file["path"]
    frame = pd.read_parquet(path)
    frame.iloc[::-1].reset_index(drop=True).to_parquet(path, index=False)
    refreshed = replace(v2, file=run_file(context.run, path))
    with pytest.raises(DataValidationError, match="identifier sequence differs"):
        evaluate(context, v1, refreshed)
    loaded = load_run(context.run.path, context.run.partitions)
    assert [r for r in loaded.of_kind("evaluation") if r["policy"] in ("R0", "R1", "R2")] == []


def test_a_v2_without_a_fraud_is_a_recorded_primary_selection_error(
    scored: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, selected = scored
    v1, v2 = score_both(context, selected)
    changed = replace(v2, rows=replace(v2.rows, labels=np.zeros(V2_ROWS, dtype=np.int8)))
    with pytest.raises(DataValidationError, match="primary R1 threshold cannot be selected"):
        evaluate(context, v1, changed)
    loaded = load_run(context.run.path, context.run.partitions)
    (record,) = loaded.of_kind("r1_selection")
    assert (record["period"], record["status"], record["error"]) == ("primary", "error", "no_fraud")
    assert [r for r in loaded.of_kind("evaluation") if r["policy"] in ("R0", "R1", "R2")] == []


def test_the_policies_need_the_v1_and_v2_scores_of_one_model(
    scored: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, selected = scored
    v1, v2 = score_both(context, selected)
    with pytest.raises(DataValidationError, match="V1 and V2 scores of one model"):
        evaluate(context, v2, v1)
    with pytest.raises(DataValidationError, match="V1 and V2 scores of one model"):
        evaluate(context, replace(v1, configuration="LGB-b"), v2)


def test_a_selection_without_a_fraud_is_recorded_and_then_stops_the_run(
    scored: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, selected = scored
    v1, v2 = score_both(context, selected)
    # Day 3 only: the fitting days hold no fraud once their labels are cleared.
    labels = v2.rows.labels.copy()
    days = v2.rows.times // 86_400
    labels[days <= 2] = 0
    labels[np.flatnonzero(days == 3)[0]] = 1
    changed = replace(v2, rows=replace(v2.rows, labels=labels))
    with pytest.raises(DataValidationError, match="latency R1 threshold cannot be selected"):
        evaluate(context, v1, changed)
    loaded = load_run(context.run.path, context.run.partitions)
    errors = [r for r in loaded.of_kind("r1_selection") if r["status"] == "error"]
    assert [(r["period"], r["error"]) for r in errors] == [("latency", "no_fraud")]


def test_the_saved_scoring_is_found_from_a_loaded_run(
    scored: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, selected = scored
    v1, v2 = score_both(context, selected)
    loaded = load_run(context.run.path, context.run.partitions)
    again = saved_partition(loaded, context.inputs, "LGB-u", "V2")
    assert (again.file, again.record) == (v2.file, v2.record)
    assert again.rows is context.inputs.v2
    assert saved_partition(loaded, context.inputs, "LGB-u", "V1").file == v1.file
    with pytest.raises(RecordContractError, match="0 score records"):
        saved_partition(loaded, context.inputs, "LGB-b", "V2")
    evaluate_thresholds(
        context.run,
        loaded,  # the scores are read from the loaded records' directory
        context.run.partitions,
        context.identity,
        context.seed,
        saved_partition(loaded, context.inputs, "LGB-u", "V1"),
        again,
    )


def test_only_the_selected_main_seed_model_gets_threshold_policies(
    scored: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, selected = scored
    v1, v2 = score_both(context, selected)
    refit = replace(v2, role="seed_refit", seed=43, tag="seed_refit-s43")
    with pytest.raises(ModelError, match="not those of the selected main-seed model"):
        evaluate(context, v1, refit)
    with pytest.raises(ModelError, match="not those of the selected main-seed model"):
        evaluate(context, replace(v1, role="cpu_refit"), v2)
    loaded = load_run(context.run.path, context.run.partitions)
    assert [r for r in loaded.of_kind("evaluation") if r["policy"] in ("R0", "R1", "R2")] == []
