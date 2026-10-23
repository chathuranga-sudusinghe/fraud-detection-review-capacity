"""Ranking and capacity evaluation of the scored partitions: LightGBM on synthetic data on the
CPU (no project data)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from review_capacity.development.evaluation import evaluate_partition, truth_of
from review_capacity.development.run_output import load_run
from review_capacity.development.scoring import ScoringContext, load_selected, score_partition
from review_capacity.development.trial import TrialOutcome, run_file
from review_capacity.errors import DataValidationError
from review_capacity.evaluation.ranking import capacity_budgets
from review_capacity.models.lightgbm_model import LightGBMModel
from review_capacity.models.registry import configuration
from support.search import V1_ROWS, V2_ROWS, searched

LGB_U = configuration("LGB-u")


pytestmark = pytest.mark.usefixtures("restore_lightgbm_logger")


@pytest.fixture
def scoring(tmp_path: Path) -> tuple[ScoringContext, TrialOutcome]:
    return searched(tmp_path)


@pytest.mark.parametrize(("label", "rows"), [("V1", V1_ROWS), ("V2", V2_ROWS)])
def test_r3_at_the_four_capacities_and_the_ranking_are_recorded(
    scoring: tuple[ScoringContext, TrialOutcome], label: str, rows: int
) -> None:
    context, selected = scoring
    model = load_selected(context, LGB_U, selected)
    scored = score_partition(context, LGB_U, selected, model, label)
    written = evaluate_partition(context, scored)
    assert list(written) == ["R3-c01", "R3-c02", "R3-c05", "R3-c10", "ranking"]
    records = [
        r
        for r in load_run(context.run.path, context.run.partitions).of_kind("evaluation")
        if r["partition"] == label
    ]
    r3 = sorted((r for r in records if r["policy"] == "R3"), key=lambda r: r["capacity"])
    budgets = capacity_budgets(context.run.partitions.v2)
    assert [r["capacity"] for r in r3] == [b.value for b in budgets]
    assert [r["k"] for r in r3] == [b.k for b in budgets]  # K_c is fixed from V2 for both
    for record in r3:
        assert record["counts"]["transactions"] == rows
        assert record["counts"]["selected"] <= record["k"] * record["counts"]["days"]
        assert record["score"] == scored.record
        assert record["model_role"] == "selected"
        assert record["period"] == "primary"
    (ranking,) = [r for r in records if r["policy"] == "ranking"]
    assert 0.0 < ranking["metrics"]["average_precision"] <= 1.0
    assert 0.0 <= ranking["metrics"]["roc_auc"] <= 1.0


def test_r3_recall_follows_the_daily_top_k_of_the_scores(
    scoring: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, selected = scoring
    model = load_selected(context, LGB_U, selected)
    assert isinstance(model, LightGBMModel)
    scored = score_partition(context, LGB_U, selected, model, "V2")
    evaluate_partition(context, scored)
    (record,) = [
        r
        for r in load_run(context.run.path, context.run.partitions).of_kind("evaluation")
        if r["policy"] == "R3" and r["capacity"] == 0.1
    ]
    rows = scored.rows
    scores = model.predict(rows.features)
    days = rows.times // 86_400
    hits = 0
    for day in np.unique(days):
        member = np.flatnonzero(days == day)
        ordered = member[np.lexsort((rows.ids[member], -scores[member]))]  # ties: see below
        hits += int(rows.labels[ordered[: record["k"]]].sum())
    # Equal scores are ordered by the shared SHA-256 tie rule, not by identifier; with distinct
    # scores the two agree, and a tie at a cutoff is reported in the record.
    if record["cutoff_ties"]["days"] == 0:
        assert record["counts"]["true_positives"] == hits
    assert record["counts"]["true_positives"] + record["counts"]["false_negatives"] == int(
        rows.labels.sum()
    )


def test_scores_that_do_not_pair_with_the_rows_are_refused(
    scoring: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, selected = scoring
    model = load_selected(context, LGB_U, selected)
    scored = score_partition(context, LGB_U, selected, model, "V1")
    path = context.run.path / scored.file["path"]
    frame = pd.read_parquet(path)
    frame.iloc[::-1].reset_index(drop=True).to_parquet(path, index=False)
    refreshed = replace(scored, file=run_file(context.run, path))
    with pytest.raises(DataValidationError, match="identifier sequence differs"):
        evaluate_partition(context, refreshed)
    loaded = load_run(context.run.path, context.run.partitions)
    assert loaded.of_kind("evaluation") == []


def test_the_truth_uses_the_partitions_labels_and_amounts(
    scoring: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, _ = scoring
    rows = context.inputs.v1
    truth = truth_of(rows)
    assert np.array_equal(truth.labels, rows.labels)
    assert np.array_equal(truth.amounts, rows.features["TransactionAmt"].to_numpy())
    assert np.array_equal(truth.days, rows.times // 86_400)
    without = replace(rows, features=rows.features.drop(columns=["TransactionAmt"]))
    with pytest.raises(DataValidationError, match="TransactionAmt"):
        truth_of(without)
