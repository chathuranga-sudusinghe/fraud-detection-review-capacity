"""The sequential search of one configuration, its selection and its records: LightGBM on
synthetic data on the CPU (no project data)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from review_capacity.development.run_output import load_run
from review_capacity.development.search import search_configuration, select
from review_capacity.development.startup import Startup
from review_capacity.development.trial import SearchContext, TrialOutcome
from review_capacity.development.trial import run_trial as real_run_trial
from review_capacity.errors import ReviewCapacityError, SpecificationError
from review_capacity.models.registry import configuration
from review_capacity.training.sampler import CandidateList
from support.memory import memory_setup
from support.search import CANDIDATES, TRIALS, candidate_file, candidate_list, search_context

LGB_U = configuration("LGB-u")


pytestmark = pytest.mark.usefixtures("restore_lightgbm_logger")


@pytest.fixture
def context(tmp_path: Path) -> SearchContext:
    return search_context(tmp_path)


def outcome(trial: int, objective: float) -> TrialOutcome:
    return TrialOutcome(
        "LGB-u",
        trial,
        {},
        objective,
        3,
        8,
        "early_stopping",
        0.1,
        0.05,
        None,
        3,
        Path("m"),
        {},
        {},
        {},
    )


def timings(context: SearchContext, measurement: str) -> list[dict[str, Any]]:
    loaded = load_run(context.run.path, context.run.partitions)
    return [t for t in loaded.of_kind("timing") if t["measurement"] == measurement]


def test_the_highest_objective_is_selected() -> None:
    selected, tied = select([outcome(1, 0.2), outcome(2, 0.5), outcome(3, 0.4)])
    assert (selected.trial, tied) == (2, (2,))


def test_exact_ties_go_to_the_lowest_trial_number() -> None:
    selected, tied = select([outcome(1, 0.2), outcome(2, 0.5), outcome(3, 0.5), outcome(4, 0.1)])
    assert (selected.trial, tied) == (2, (2, 3))
    selected, tied = select([outcome(1, 0.5), outcome(2, 0.5)])
    assert (selected.trial, tied) == (1, (1, 2))


def test_a_near_tie_is_not_a_tie() -> None:
    selected, tied = select([outcome(1, 0.5), outcome(2, 0.5 + 1e-12)])
    assert (selected.trial, tied) == (2, (2,))


def test_a_selection_needs_trials_numbered_in_order() -> None:
    with pytest.raises(SpecificationError, match="at least one"):
        select([])
    with pytest.raises(SpecificationError, match="numbered 1, 2"):
        select([outcome(1, 0.1), outcome(3, 0.2)])


def test_every_declared_trial_runs_and_the_best_is_selected(context: SearchContext) -> None:
    result = search_configuration(context, LGB_U, candidate_list(), candidate_file(context.run))
    assert [t.trial for t in result.trials] == [1, 2, 3]
    assert [t.hyperparameters for t in result.trials] == list(CANDIDATES)
    best = max(t.objective for t in result.trials)
    assert result.selected.objective == best
    assert result.selected.trial == min(t.trial for t in result.trials if t.objective == best)
    assert result.tied == tuple(t.trial for t in result.trials if t.objective == best)
    for number in (1, 2, 3):
        assert (context.run.path / "models" / "LGB-u" / f"trial-{number:02d}").is_dir()


def test_the_configuration_record_matches_the_selection(context: SearchContext) -> None:
    result = search_configuration(context, LGB_U, candidate_list(), candidate_file(context.run))
    loaded = load_run(context.run.path, context.run.partitions)
    (record,) = loaded.of_kind("configuration")
    assert [t["trial"] for t in record["trials"]] == [1, 2, 3]
    assert record["selected_trial"] == result.selected.trial
    assert record["selected_objective"] == result.selected.objective
    assert record["tied_trials"] == list(result.tied)
    assert record["selected_length"] == result.selected.best_checkpoint
    assert record["selected_hyperparameters"] == result.selected.hyperparameters
    assert record["selected_fit"] == result.selected.fit
    assert record["selected_checkpoint"] == result.selected.checkpoint
    assert len(loaded.of_kind("trial")) == len(loaded.of_kind("fit")) == TRIALS
    assert len(loaded.of_kind("checkpoint")) == TRIALS


def test_the_search_and_the_selected_fit_are_timed(context: SearchContext) -> None:
    result = search_configuration(context, LGB_U, candidate_list(), candidate_file(context.run))
    (whole,) = timings(context, "search")
    assert whole["model_role"] is None
    assert whole["device"] == "cpu"
    assert whole["settings"]["trials_completed"] == whole["settings"]["trials_declared"] == TRIALS
    assert whole["settings"]["interrupted"] is False
    assert whole["seconds"] == pytest.approx(result.search_seconds, rel=0.5, abs=1.0)
    (fit,) = timings(context, "fit")
    assert fit["model_role"] == "selected"
    assert fit["subject"] == result.selected.fit
    assert fit["seconds"] == result.selected.fit_seconds
    assert fit["settings"]["retained_length"] == result.selected.best_checkpoint
    assert fit["settings"]["completed_length"] == result.selected.stopped_at


def test_an_interrupted_search_is_recorded_as_partial_and_raised(
    context: SearchContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def interrupting(*args: Any) -> TrialOutcome:
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        outcome_: TrialOutcome = real_run_trial(*args)
        return outcome_

    monkeypatch.setattr("review_capacity.development.search.run_trial", interrupting)
    with pytest.raises(KeyboardInterrupt):
        search_configuration(context, LGB_U, candidate_list(), candidate_file(context.run))
    (whole,) = timings(context, "search")
    assert whole["settings"]["interrupted"] is True
    assert whole["settings"]["trials_completed"] == 1
    loaded = load_run(context.run.path, context.run.partitions)
    assert loaded.of_kind("configuration") == []
    assert timings(context, "fit") == []


def test_a_failed_trial_ends_the_search_without_a_selection(context: SearchContext) -> None:
    candidates = (CANDIDATES[0], {**CANDIDATES[1], "not_a_lightgbm_parameter": 1}, CANDIDATES[2])
    with pytest.raises(ReviewCapacityError):
        search_configuration(
            context, LGB_U, candidate_list(candidates), candidate_file(context.run)
        )
    loaded = load_run(context.run.path, context.run.partitions)
    assert [t["status"] for t in loaded.of_kind("trial")] == ["completed", "failed"]
    assert loaded.of_kind("configuration") == []
    assert timings(context, "search") == []


def test_the_candidate_list_must_be_the_models_and_complete(context: SearchContext) -> None:
    other = CandidateList("catboost", 1, (1,), CANDIDATES)
    with pytest.raises(SpecificationError, match="list is that of catboost"):
        search_configuration(context, LGB_U, other, candidate_file(context.run))
    with pytest.raises(SpecificationError, match="2 candidates, 3 are declared"):
        search_configuration(
            context, LGB_U, candidate_list(CANDIDATES[:2]), candidate_file(context.run)
        )
    assert load_run(context.run.path, context.run.partitions).of_kind("trial") == []


def test_the_search_time_ends_after_selection_and_its_record(
    context: SearchContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DR-015 §9: the clock runs to the end of the selection and the ``configuration`` record, so
    the search timing is written after that record."""
    written: list[str] = []
    write_record = context.run.write_record

    def recording(record: Any, *parts: str | int) -> Path:
        written.append(record["record"] + record.get("measurement", ""))
        path: Path = write_record(record, *parts)
        return path

    monkeypatch.setattr(context.run, "write_record", recording)
    search_configuration(context, LGB_U, candidate_list(), candidate_file(context.run))
    assert written.index("configuration") < written.index("timingsearch")


def test_the_selected_fit_carries_its_own_trials_readings(context: SearchContext) -> None:
    peaks = [(900 - 100 * i) * 1024 for i in range(TRIALS)]  # a different peak for every trial
    measured = replace(context, memory=memory_setup(peaks))
    result = search_configuration(measured, LGB_U, candidate_list(), candidate_file(measured.run))
    expected = peaks[result.selected.trial - 1] / 1024
    (fit,) = timings(measured, "fit")
    assert fit["peak_host_mib"] == expected
    assert fit["settings"]["host_memory_method"] == "clear_refs"
    assert fit["settings"]["memory_problems"] == []
    assert "the trial's fit" in fit["memory_scope"]
    (whole,) = timings(measured, "search")
    assert whole["peak_host_mib"] == max(peaks) / 1024
    assert "not a whole-search peak" in whole["memory_scope"]
    assert whole["peak_gpu_mib"] is None


def test_the_selected_fit_records_the_shared_startup_and_its_persistence(
    context: SearchContext,
) -> None:
    startup = Startup(1.25, "library import, once per process")
    started = replace(context, startups={"lightgbm": startup})
    result = search_configuration(started, LGB_U, candidate_list(), candidate_file(started.run))
    (fit,) = timings(started, "fit")
    assert fit["settings"]["startup_seconds"] == 1.25
    assert fit["settings"]["startup_scope"] == startup.scope
    assert fit["settings"]["persistence_seconds"] == result.selected.persistence_seconds
    assert fit["settings"]["preprocessing_seconds"] is None
    assert fit["seconds"] == result.selected.fit_seconds  # the start-up is outside the fit time


def test_without_a_startup_or_memory_setup_the_components_are_absent(
    context: SearchContext,
) -> None:
    search_configuration(context, LGB_U, candidate_list(), candidate_file(context.run))
    (fit,) = timings(context, "fit")
    assert fit["peak_host_mib"] is None
    assert fit["memory_scope"] is None
    assert fit["settings"]["startup_seconds"] is None
    assert fit["settings"]["startup_scope"] is None


def test_an_interrupted_search_records_the_memory_of_its_completed_trials(
    context: SearchContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    measured = replace(context, memory=memory_setup([200 * 1024, 400 * 1024, 300 * 1024]))
    calls = {"n": 0}

    def stop_after_two(*args: Any, **kwargs: Any) -> TrialOutcome:
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real_run_trial(*args, **kwargs)

    monkeypatch.setattr("review_capacity.development.search.run_trial", stop_after_two)
    with pytest.raises(KeyboardInterrupt):
        search_configuration(measured, LGB_U, candidate_list(), candidate_file(measured.run))
    (whole,) = timings(measured, "search")
    assert whole["settings"]["interrupted"] is True
    assert whole["settings"]["trials_completed"] == 2
    assert whole["peak_host_mib"] == 400.0


def test_a_selected_trial_that_is_not_the_largest_keeps_its_own_reading(
    context: SearchContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    peaks = [900 * 1024, 300 * 1024, 700 * 1024]  # trial 2 uses the least memory
    measured = replace(context, memory=memory_setup(peaks))

    def second(trials: list[TrialOutcome]) -> tuple[TrialOutcome, tuple[int, ...]]:
        return trials[1], (trials[1].trial,)  # tied trials include the selected one

    monkeypatch.setattr("review_capacity.development.search.select", second)
    result = search_configuration(measured, LGB_U, candidate_list(), candidate_file(measured.run))
    assert result.selected.trial == 2
    (fit,) = timings(measured, "fit")
    assert fit["peak_host_mib"] == 300.0  # its own, not the search's maximum
    (whole,) = timings(measured, "search")
    assert whole["peak_host_mib"] == 900.0


def test_a_startup_that_could_not_be_measured_is_stated_beside_the_fit_time(
    context: SearchContext,
) -> None:
    started = replace(
        context, startups={"lightgbm": Startup(None, "library import, once per process")}
    )
    search_configuration(started, LGB_U, candidate_list(), candidate_file(started.run))
    (fit,) = timings(started, "fit")
    assert fit["settings"]["startup_seconds"] is None
    assert fit["settings"]["startup_status"] == "not_measured"


def test_a_measured_startup_is_stated_as_measured(context: SearchContext) -> None:
    started = replace(context, startups={"lightgbm": Startup(0.5, "library import")})
    search_configuration(started, LGB_U, candidate_list(), candidate_file(started.run))
    (fit,) = timings(started, "fit")
    assert fit["settings"]["startup_status"] == "measured"
