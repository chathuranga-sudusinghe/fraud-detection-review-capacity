"""Scoring the selected model on V1 and V2 and the row-alignment check: LightGBM on synthetic
data on the CPU (no project data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from review_capacity.development.run_output import load_run
from review_capacity.development.scoring import (
    ScoringContext,
    check_sequence,
    load_selected,
    read_scores,
    score_partition,
    score_path,
)
from review_capacity.development.trial import TrialOutcome, run_file
from review_capacity.errors import DataValidationError, FileAccessError, SpecificationError
from review_capacity.models.lightgbm_model import LightGBMModel
from review_capacity.models.registry import configuration
from support.search import V1_ROWS, V2_ROWS, searched

LGB_U = configuration("LGB-u")


pytestmark = pytest.mark.usefixtures("restore_lightgbm_logger")


@pytest.fixture
def scoring(tmp_path: Path) -> tuple[ScoringContext, TrialOutcome]:
    return searched(tmp_path)


def test_the_selected_model_is_reloaded_and_scores_like_the_saved_one(
    scoring: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, selected = scoring
    first = load_selected(context, LGB_U, selected)
    second = load_selected(context, LGB_U, selected)
    assert isinstance(first, LightGBMModel)
    assert isinstance(second, LightGBMModel)
    features = context.inputs.v1.features
    assert first.fitted_iterations == selected.fitted_iterations
    assert np.array_equal(first.predict(features), second.predict(features))


def test_only_tree_models_are_loaded_here(
    scoring: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, selected = scoring
    with pytest.raises(SpecificationError, match="neural inputs"):
        load_selected(context, configuration("RES-u"), selected)


@pytest.mark.parametrize(("label", "rows"), [("V1", V1_ROWS), ("V2", V2_ROWS)])
def test_a_partition_is_scored_to_a_local_file_and_a_record(
    scoring: tuple[ScoringContext, TrialOutcome], label: str, rows: int
) -> None:
    context, selected = scoring
    model = load_selected(context, LGB_U, selected)
    assert isinstance(model, LightGBMModel)
    scored = score_partition(context, LGB_U, selected, model, label)
    frame = pd.read_parquet(context.run.path / scored.file["path"])
    partition = context.inputs.partition(label)
    assert list(frame.columns) == ["TransactionID", "score"]
    assert np.array_equal(frame["TransactionID"].to_numpy(), partition.ids)
    assert np.array_equal(frame["score"].to_numpy(), model.predict(partition.features))
    assert (
        scored.file["path"]
        == score_path(context.run, "LGB-u", label).relative_to(context.run.path).as_posix()
    )
    (record,) = [
        r
        for r in load_run(context.run.path, context.run.partitions).of_kind("score")
        if r["partition"] == label
    ]
    assert record["rows"] == rows
    assert record["fit"] == selected.fit
    assert record["model_role"] == "selected"
    assert record["device"] == "cpu"
    assert record["min_score"] <= record["max_score"]
    assert 1 <= record["distinct_scores"] <= rows
    assert record["file"] == scored.file


def test_identical_sequences_pass_and_any_difference_stops_the_run() -> None:
    ids = np.array([5, 7, 9, 11], dtype=np.int64)
    check_sequence(ids, ids.copy(), "scores")
    with pytest.raises(DataValidationError, match="differs from the partition's at row 1"):
        check_sequence(ids, np.array([5, 9, 7, 11], dtype=np.int64), "scores")
    with pytest.raises(DataValidationError, match="3 identifiers, the partition has 4"):
        check_sequence(ids, ids[:3], "scores")


def test_scores_in_another_row_order_are_refused(
    scoring: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, selected = scoring
    model = load_selected(context, LGB_U, selected)
    scored = score_partition(context, LGB_U, selected, model, "V1")
    path = context.run.path / scored.file["path"]
    frame = pd.read_parquet(path)
    swapped = frame.iloc[[1, 0, *range(2, len(frame))]].reset_index(drop=True)
    swapped.to_parquet(path, index=False)
    with pytest.raises(DataValidationError, match="differs from the score file"):
        read_scores(context.run, scored.file, scored.rows)  # the file changed since the record
    with pytest.raises(DataValidationError, match="identifier sequence differs"):
        read_scores(context.run, run_file(context.run, path), scored.rows)


def test_a_score_file_with_other_columns_is_refused(
    scoring: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, selected = scoring
    model = load_selected(context, LGB_U, selected)
    scored = score_partition(context, LGB_U, selected, model, "V1")
    path = context.run.path / scored.file["path"]
    pd.read_parquet(path).rename(columns={"score": "probability"}).to_parquet(path, index=False)
    with pytest.raises(DataValidationError, match="columns"):
        read_scores(context.run, run_file(context.run, path), scored.rows)


def test_a_score_directory_that_cannot_be_created_is_a_project_error(
    scoring: tuple[ScoringContext, TrialOutcome],
) -> None:
    context, selected = scoring
    model = load_selected(context, LGB_U, selected)
    (context.run.path / "scores").write_text("a file where the directory belongs", encoding="utf-8")
    with pytest.raises(FileAccessError, match="cannot be written"):
        score_partition(context, LGB_U, selected, model, "V1")
