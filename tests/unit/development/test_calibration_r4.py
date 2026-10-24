"""Calibration, R4 and the RQ3 cost analysis on the saved scores of a selected model: LightGBM on
synthetic data on the CPU (no project data)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from review_capacity.development.calibration import (
    METHODS,
    PERIODS,
    FittedCalibrator,
    artifact_path,
    fit_calibrators,
)
from review_capacity.development.rq3 import evaluate_r4
from review_capacity.development.run_output import load_run
from review_capacity.development.scoring import (
    ScoredPartition,
    ScoringContext,
    load_selected,
    read_scores,
    score_partition,
)
from review_capacity.errors import DataValidationError, ModelError
from review_capacity.evaluation.calibration import load_calibrator
from review_capacity.evaluation.cost import RATIO_GRID
from review_capacity.models.registry import configuration
from support.search import V2_ROWS, searched

LGB_U = configuration("LGB-u")

pytestmark = pytest.mark.usefixtures("restore_lightgbm_logger")


@pytest.fixture
def scored(tmp_path: Path) -> tuple[ScoringContext, ScoredPartition]:
    context, selected = searched(tmp_path)
    model = load_selected(context, LGB_U, selected)
    return context, score_partition(context, LGB_U, selected, model, "V2")


def calibrate(context: ScoringContext, v2: ScoredPartition) -> list[FittedCalibrator]:
    return fit_calibrators(context.run, context.run, context.run.partitions, context.identity, v2)


def test_four_calibrators_are_fitted_saved_reloaded_and_recorded(
    scored: tuple[ScoringContext, ScoredPartition],
) -> None:
    context, v2 = scored
    fitted = calibrate(context, v2)
    assert [(f.method, f.period) for f in fitted] == [
        (method, period) for period in PERIODS for method in METHODS
    ]
    loaded = load_run(context.run.path, context.run.partitions)
    records = {(r["method"], r["period"]): r for r in loaded.of_kind("calibration")}
    assert set(records) == {(f.method, f.period) for f in fitted}
    scores = read_scores(context.run, v2.file, v2.rows)
    for f in fitted:
        record = records[(f.method, f.period)]
        path = artifact_path(context.run, "LGB-u", f.method, f.period)
        assert record["artifact"]["path"] == path.relative_to(context.run.path).as_posix()
        assert record["status"] == "completed"
        assert record["reload_verified"] is True
        again = load_calibrator(f.method, path)
        assert np.array_equal(again.predict(scores), f.calibrator.predict(scores))
        assert record["score"] == v2.record
    primary = records[("platt", "primary")]
    assert primary["fitting_days"] == {"first": 1, "last": 3}  # the synthetic V2 spans days 1–3
    assert primary["rows"] == V2_ROWS
    assert [d["rows_used"] for d in primary["diagnostics"]] == ["fitting"]
    assert primary["parameters"]["slope"] > 0
    assert primary["targets"]["positive"] == (primary["frauds"] + 1) / (primary["frauds"] + 2)
    latency = records[("isotonic", "latency")]
    assert latency["fitting_days"] == {"first": 1, "last": 2}
    assert [d["rows_used"] for d in latency["diagnostics"]] == ["fitting", "held_out"]
    assert latency["diagnostics"][1]["days"] == {"first": 3, "last": 3}
    assert latency["knots"] > 1
    assert latency["rows"] < V2_ROWS


def test_r4_is_evaluated_at_every_capacity_and_ratio_for_every_calibrator(
    scored: tuple[ScoringContext, ScoredPartition],
) -> None:
    context, v2 = scored
    fitted = calibrate(context, v2)
    written = evaluate_r4(
        context.run, context.run, context.run.partitions, context.identity, context.seed, v2, fitted
    )
    loaded = load_run(context.run.path, context.run.partitions)
    r4 = [r for r in loaded.of_kind("evaluation") if r["policy"] == "R4"]
    assert len(r4) == len(METHODS) * len(PERIODS) * 4 * len(RATIO_GRID)
    assert {r["cost_ratio"] for r in r4} == {float(g) for g in RATIO_GRID}
    assert all(r["threshold"] == 1 / r["cost_ratio"] for r in r4)
    assert {(r["partition"], r["model_role"], r["seed"]) for r in r4} == {("V2", "selected", 42)}
    primary = [r for r in r4 if r["period"] == "primary"]
    latency = [r for r in r4 if r["period"] == "latency"]
    assert {(r["days"]["first"], r["days"]["last"]) for r in primary} == {(1, 3)}
    assert {(r["days"]["first"], r["days"]["last"]) for r in latency} == {(3, 3)}
    analyses = loaded.of_kind("cost_analysis")
    assert len(analyses) == len(METHODS) * len(PERIODS) * 4
    assert len(written) == len(r4) + len(analyses)
    for record in r4:
        assert record["calibration"]["path"].startswith("records/calibration-LGB-u-")
        assert record["counts"]["selected"] <= record["k"] * record["counts"]["days"]
        assert record["metrics"]["scenario_cost"] == pytest.approx(
            record["counts"]["selected"]
            + record["cost_ratio"] * record["counts"]["false_negatives"]
        )


def test_the_cost_analysis_agrees_with_the_r4_evaluation_records(
    scored: tuple[ScoringContext, ScoredPartition],
) -> None:
    context, v2 = scored
    evaluate_r4(
        context.run,
        context.run,
        context.run.partitions,
        context.identity,
        context.seed,
        v2,
        calibrate(context, v2),
    )
    loaded = load_run(context.run.path, context.run.partitions)
    cells = {
        (r["calibrator"], r["period"], r["capacity"], r["cost_ratio"]): r
        for r in loaded.of_kind("evaluation")
        if r["policy"] == "R4"
    }
    for analysis in loaded.of_kind("cost_analysis"):
        for row in analysis["grid"]:
            cell = cells[
                (analysis["calibrator"], analysis["period"], analysis["capacity"], row["ratio"])
            ]
            assert row["reviews"] == cell["counts"]["selected"]
            assert (
                row["frauds_forgone"]
                == analysis["queue"]["captured"] - cell["counts"]["true_positives"]
            )
            assert row["cost_r4"] == pytest.approx(cell["metrics"]["scenario_cost"])
            assert row["utilisation"] == pytest.approx(cell["metrics"]["utilisation"])
            assert row["unused"] == analysis["queue"]["reviews"] - row["reviews"]
        assert analysis["break_even"]["classification"] in {
            "coincidence",
            "crossing",
            "none",
            "right_censored",
        }


def test_r4_reviews_more_as_the_ratio_rises_and_never_more_than_r3(
    scored: tuple[ScoringContext, ScoredPartition],
) -> None:
    context, v2 = scored
    evaluate_r4(
        context.run,
        context.run,
        context.run.partitions,
        context.identity,
        context.seed,
        v2,
        calibrate(context, v2),
    )
    loaded = load_run(context.run.path, context.run.partitions)
    by_cell: dict[tuple[str, str, float], list[tuple[float, int]]] = {}
    for r in loaded.of_kind("evaluation"):
        if r["policy"] == "R4":
            key = (r["calibrator"], r["period"], r["capacity"])
            by_cell.setdefault(key, []).append((r["cost_ratio"], r["counts"]["selected"]))
    for key, cells in by_cell.items():
        reviews = [selected for _, selected in sorted(cells)]
        assert reviews == sorted(reviews), key
    for analysis in loaded.of_kind("cost_analysis"):
        assert all(row["reviews"] <= analysis["queue"]["reviews"] for row in analysis["grid"])


def test_a_v2_without_a_fraud_stops_the_calibration_with_a_failed_record(
    scored: tuple[ScoringContext, ScoredPartition],
) -> None:
    context, v2 = scored
    changed = replace(v2, rows=replace(v2.rows, labels=np.zeros(V2_ROWS, dtype=np.int8)))
    with pytest.raises(DataValidationError, match="both classes"):
        calibrate(context, changed)
    loaded = load_run(context.run.path, context.run.partitions)
    (record,) = loaded.of_kind("calibration")
    assert (record["method"], record["period"], record["status"]) == ("platt", "primary", "failed")
    assert "both classes" in record["error"]
    assert record["artifact"] is None
    assert record["frauds"] == 0


def test_only_the_selected_main_seed_model_is_calibrated_and_analysed(
    scored: tuple[ScoringContext, ScoredPartition],
) -> None:
    context, v2 = scored
    refit = replace(v2, role="seed_refit", seed=43, tag="seed_refit-s43")
    with pytest.raises(ModelError, match="not those of the selected main-seed model"):
        calibrate(context, refit)
    fitted = calibrate(context, v2)
    with pytest.raises(ModelError, match="not those of the selected main-seed model"):
        evaluate_r4(
            context.run,
            context.run,
            context.run.partitions,
            context.identity,
            context.seed,
            refit,
            fitted,
        )
    loaded = load_run(context.run.path, context.run.partitions)
    assert [r for r in loaded.of_kind("evaluation") if r["policy"] == "R4"] == []


def test_the_scores_must_be_those_of_v2(scored: tuple[ScoringContext, ScoredPartition]) -> None:
    context, v2 = scored
    with pytest.raises(ModelError, match="V2"):
        calibrate(context, replace(v2, label="V1"))
