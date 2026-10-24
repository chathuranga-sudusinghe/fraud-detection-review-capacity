"""Fixed-length refits, their records and the CPU–GPU comparison: LightGBM on synthetic data on
the CPU (no project data)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from review_capacity.development.refits import (
    CPU_REFIT,
    SAME_SEED_REPEAT,
    SEED_REFIT,
    RefitJob,
    RefitSpec,
    compare_scores,
    device_comparison,
    fit_refit,
    refit_directory,
    refit_specs,
    run_refit,
    run_refits,
)
from review_capacity.development.run_output import load_run
from review_capacity.development.scoring import (
    ScoredPartition,
    ScoringContext,
    load_selected,
    score_partition,
)
from review_capacity.development.search import search_configuration
from review_capacity.development.trial import SearchContext, TrialOutcome
from review_capacity.errors import ModelError, SpecificationError
from review_capacity.models.lightgbm_model import LightGBMModel
from review_capacity.models.registry import REFERENCE_SUBJECT, configuration
from support.search import (
    candidate_file,
    candidate_list,
    scoring_context,
    search_context,
)

LGB_U = configuration("LGB-u")
CAT_U = configuration("Cat-u")

pytestmark = pytest.mark.usefixtures("restore_lightgbm_logger")


@pytest.fixture
def state(
    tmp_path: Path,
) -> tuple[SearchContext, ScoringContext, TrialOutcome, ScoredPartition]:
    search = search_context(tmp_path)
    result = search_configuration(search, LGB_U, candidate_list(), candidate_file(search.run))
    scoring = scoring_context(search)
    model = load_selected(scoring, LGB_U, result.selected)
    reference = score_partition(scoring, LGB_U, result.selected, model, "V2")
    return search, scoring, result.selected, reference


def fresh_like(search: SearchContext) -> object:
    """Stands in for the child process: fits and saves in this process."""

    def fresh(job: RefitJob, directory: Path) -> float:
        subject = configuration(job.configuration)
        return fit_refit(search, subject, job, directory)[0]

    return fresh


def test_the_refits_of_a_configuration_follow_dr_015_and_dr_013() -> None:
    lightgbm = refit_specs(LGB_U, "cpu", 42, (43, 44))
    assert [(s.kind, s.seed, s.device, s.fresh_process) for s in lightgbm] == [
        (SEED_REFIT, 43, "cpu", False),
        (SEED_REFIT, 44, "cpu", False),
        (SAME_SEED_REPEAT, 42, "cpu", True),
    ]
    catboost = refit_specs(CAT_U, "gpu", 42, (43, 44))
    assert [(s.kind, s.seed, s.device, s.fresh_process) for s in catboost][-2:] == [
        (SAME_SEED_REPEAT, 42, "gpu", True),
        (CPU_REFIT, 42, "cpu", False),
    ]
    assert [s.tag for s in catboost] == [
        "seed_refit-s43",
        "seed_refit-s44",
        "same_seed_repeat",
        "cpu_refit",
    ]
    with pytest.raises(SpecificationError, match="no refits"):
        refit_specs(REFERENCE_SUBJECT, "cpu", 42, (43, 44))


def test_a_seed_refit_grows_exactly_the_selected_length_and_is_recorded(
    state: tuple[SearchContext, ScoringContext, TrialOutcome, ScoredPartition],
) -> None:
    search, scoring, selected, reference = state
    spec = RefitSpec(SEED_REFIT, 43, "cpu")
    scored = run_refit(search, scoring, LGB_U, selected, spec, reference)
    loaded = load_run(search.run.path, search.run.partitions)
    (fit,) = [r for r in loaded.of_kind("fit") if r["purpose"] == SEED_REFIT]
    assert (fit["seed"], fit["length"], fit["fitted_iterations"]) == (
        43,
        selected.fitted_iterations,
        selected.fitted_iterations,
    )
    assert fit["trial"] == selected.trial
    assert fit["early_stopping"] is False
    assert fit["fresh_process"] is False
    assert fit["status"] == "completed"
    (refit,) = loaded.of_kind("refit")
    assert (refit["kind"], refit["seed"], refit["comparison"]) == (SEED_REFIT, 43, None)
    assert refit["reference_score"] == reference.record
    assert refit["score"] == scored.record
    evaluations = [r for r in loaded.of_kind("evaluation") if r["model_role"] == SEED_REFIT]
    assert sorted(r["capacity"] for r in evaluations) == [0.01, 0.02, 0.05, 0.1]
    assert {(r["policy"], r["partition"], r["seed"]) for r in evaluations} == {("R3", "V2", 43)}
    (score,) = [r for r in loaded.of_kind("score") if r["model_role"] == SEED_REFIT]
    assert score["fit"] == refit["fit"]


def test_different_seeds_give_different_scores_for_the_same_setting(
    state: tuple[SearchContext, ScoringContext, TrialOutcome, ScoredPartition],
) -> None:
    search, scoring, selected, reference = state
    run_refit(search, scoring, LGB_U, selected, RefitSpec(SEED_REFIT, 43, "cpu"), reference)
    model = load_selected(
        scoring,
        LGB_U,
        selected,
        directory=refit_directory(search.run.path, "LGB-u", RefitSpec(SEED_REFIT, 43, "cpu")),
        seed=43,
    )
    other = load_selected(scoring, LGB_U, selected)
    rows = scoring.inputs.v2.features
    assert isinstance(model, LightGBMModel)
    assert isinstance(other, LightGBMModel)
    assert model.fitted_iterations == other.fitted_iterations
    assert not np.array_equal(model.predict(rows), other.predict(rows))


def test_a_same_seed_repeat_runs_through_the_fresh_process_and_reproduces_the_scores(
    state: tuple[SearchContext, ScoringContext, TrialOutcome, ScoredPartition],
) -> None:
    search, scoring, selected, reference = state
    spec = RefitSpec(SAME_SEED_REPEAT, 42, "cpu", fresh_process=True)
    run_refit(search, scoring, LGB_U, selected, spec, reference, fresh=fresh_like(search))  # type: ignore[arg-type]
    loaded = load_run(search.run.path, search.run.partitions)
    (refit,) = loaded.of_kind("refit")
    assert refit["kind"] == SAME_SEED_REPEAT
    assert refit["comparison"]["rows"] == scoring.inputs.v2.rows
    assert refit["comparison"]["identical"] is True  # LightGBM CPU training is deterministic
    assert refit["comparison"]["max_abs_difference"] == 0
    (fit,) = [r for r in loaded.of_kind("fit") if r["purpose"] == SAME_SEED_REPEAT]
    assert fit["fresh_process"] is True
    assert fit["seed"] == 42


def test_a_same_seed_repeat_needs_a_way_to_start_a_fresh_process(
    state: tuple[SearchContext, ScoringContext, TrialOutcome, ScoredPartition],
) -> None:
    search, scoring, selected, reference = state
    spec = RefitSpec(SAME_SEED_REPEAT, 42, "cpu", fresh_process=True)
    with pytest.raises(SpecificationError, match="needs a fresh process"):
        run_refit(search, scoring, LGB_U, selected, spec, reference)
    loaded = load_run(search.run.path, search.run.partitions)
    (refit,) = loaded.of_kind("refit")
    assert refit["status"] == "failed"
    assert refit["score"] is None
    assert "needs a fresh process" in refit["error"]
    (fit,) = [r for r in loaded.of_kind("fit") if r["purpose"] == SAME_SEED_REPEAT]
    assert fit["status"] == "failed"
    assert fit["model"] is None


def test_a_failed_fresh_fit_is_recorded_and_raised(
    state: tuple[SearchContext, ScoringContext, TrialOutcome, ScoredPartition],
) -> None:
    search, scoring, selected, reference = state
    spec = RefitSpec(SAME_SEED_REPEAT, 42, "cpu", fresh_process=True)

    def broken(_job: RefitJob, _directory: Path) -> float:
        raise ModelError("the child process failed")

    with pytest.raises(ModelError, match="child process failed"):
        run_refit(search, scoring, LGB_U, selected, spec, reference, fresh=broken)
    loaded = load_run(search.run.path, search.run.partitions)
    (refit,) = loaded.of_kind("refit")
    assert refit["status"] == "failed"
    assert refit["error"] == "ModelError: the child process failed"


def test_all_refits_of_lightgbm_have_no_device_pair(
    state: tuple[SearchContext, ScoringContext, TrialOutcome, ScoredPartition],
) -> None:
    search, scoring, selected, reference = state
    scored = run_refits(search, scoring, LGB_U, selected, reference, fresh=fresh_like(search))  # type: ignore[arg-type]
    assert list(scored) == ["seed_refit-s43", "seed_refit-s44", "same_seed_repeat"]
    loaded = load_run(search.run.path, search.run.partitions)
    assert [r["kind"] for r in loaded.of_kind("refit")] == [
        "same_seed_repeat",
        "seed_refit",
        "seed_refit",
    ]
    assert loaded.of_kind("device_comparison") == []


def test_score_comparison_is_exact() -> None:
    reference = np.array([0.1, 0.2, 0.3])
    assert compare_scores(reference, reference.copy()) == {
        "rows": 3,
        "identical": True,
        "differing_rows": 0,
        "max_abs_difference": 0.0,
    }
    changed = compare_scores(reference, np.array([0.1, 0.25, 0.3 + 1e-12]))
    assert changed["identical"] is False
    assert changed["differing_rows"] == 2
    assert changed["max_abs_difference"] == pytest.approx(0.05)
    with pytest.raises(ModelError, match="different numbers of rows"):
        compare_scores(reference, reference[:2])


def with_entity_inputs(scored: ScoredPartition) -> ScoredPartition:
    """The scored V2 with the inputs of the pseudo-entity key, which the synthetic model inputs
    lack."""
    rows = scored.rows
    n = rows.rows
    features = rows.features.assign(
        addr1=[str(300 + i % 7) for i in range(n)],
        P_emaildomain=["a.com" if i % 3 else "b.com" for i in range(n)],
        D1=[float(i % 5) for i in range(n)],
    )
    return replace(scored, rows=replace(rows, features=features))


def test_the_device_comparison_records_differences_and_a_bootstrap_interval(
    state: tuple[SearchContext, ScoringContext, TrialOutcome, ScoredPartition],
) -> None:
    # The pair is a LightGBM selected model and its seed-43 refit under the name of a CatBoost
    # configuration: the comparison only reads two sets of V2 scores and the contract names
    # CatBoost; no CatBoost model is trained here.
    search, scoring, selected, reference = state
    cpu = run_refit(search, scoring, LGB_U, selected, RefitSpec(SEED_REFIT, 43, "cpu"), reference)
    path = device_comparison(
        scoring,
        CAT_U,
        with_entity_inputs(reference),
        with_entity_inputs(cpu),
        {"path": "x", "sha256": "0" * 64},
    )
    record = load_run(search.run.path, search.run.partitions).of_kind("device_comparison")[0]
    assert path.name == "device_comparison-Cat-u.json"
    assert [c["capacity"] for c in record["capacities"]] == [0.01, 0.02, 0.05, 0.1]
    assert all("recall" in c["differences"] for c in record["capacities"])
    interval = record["bootstrap"]
    assert (interval["replicates"], interval["seed"], interval["level"]) == (10_000, 42, 0.95)
    assert interval["capacity"] == 0.02
    assert interval["lower"] <= interval["upper"]
    recall_at_primary = next(c for c in record["capacities"] if c["capacity"] == 0.02)
    assert interval["estimate"] == pytest.approx(recall_at_primary["differences"]["recall"])
    assert interval["days"] == 3
    assert interval["frauds"] == int(reference.rows.labels.sum())


def test_the_device_comparison_of_identical_scores_is_zero(
    state: tuple[SearchContext, ScoringContext, TrialOutcome, ScoredPartition],
) -> None:
    search, scoring, selected, reference = state
    spec = RefitSpec(SAME_SEED_REPEAT, 42, "cpu", fresh_process=True)
    again = run_refit(search, scoring, LGB_U, selected, spec, reference, fresh=fresh_like(search))  # type: ignore[arg-type]
    device_comparison(
        scoring,
        CAT_U,
        with_entity_inputs(reference),
        with_entity_inputs(again),
        {"path": "x", "sha256": "0" * 64},
    )
    record = load_run(search.run.path, search.run.partitions).of_kind("device_comparison")[0]
    assert (
        record["bootstrap"]["estimate"],
        record["bootstrap"]["lower"],
        record["bootstrap"]["upper"],
    ) == (0, 0, 0)
    assert all(not any(c["differences"].values()) for c in record["capacities"])


def test_a_later_failure_keeps_the_completed_fit_record(
    state: tuple[SearchContext, ScoringContext, TrialOutcome, ScoredPartition],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search, scoring, selected, reference = state

    def failing(*_args: object, **_kwargs: object) -> ScoredPartition:
        raise ModelError("scoring failed")

    monkeypatch.setattr("review_capacity.development.refits.score_model", failing)
    spec = RefitSpec(SEED_REFIT, 43, "cpu")
    with pytest.raises(ModelError, match="scoring failed"):
        run_refit(search, scoring, LGB_U, selected, spec, reference)
    loaded = load_run(search.run.path, search.run.partitions)
    (fit,) = [r for r in loaded.of_kind("fit") if r["purpose"] == SEED_REFIT]
    assert fit["status"] == "completed"  # the model was fitted and saved
    (refit,) = loaded.of_kind("refit")
    assert refit["status"] == "failed"
    assert refit["error"] == "ModelError: scoring failed"
    assert refit["score"] is None
