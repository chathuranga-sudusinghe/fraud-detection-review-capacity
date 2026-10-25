"""The development run end to end on synthetic partitions: search, selection, scoring, evaluation,
timing, closing and the generated summary (LightGBM on the CPU; no project data)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from review_capacity.development.run_output import END_KIND, SUMMARY_NAME, RunDirectory, load_run
from review_capacity.development.runner import (
    DevelopmentOptions,
    chosen_configurations,
    prepare_development,
    run_development,
    run_directory_for,
)
from review_capacity.development.scoring import ScoringContext, load_selected
from review_capacity.development.startup import Startup
from review_capacity.development.trial import TrialOutcome, trial_directory
from review_capacity.errors import (
    DataValidationError,
    FileAccessError,
    ModelError,
    SpecificationError,
)
from review_capacity.models.device import LightGBMIdentity
from review_capacity.models.neural_training import BatchPlan
from review_capacity.models.registry import RUN_CONFIGURATIONS, configuration
from support.memory import memory_setup
from support.search import FTT_CANDIDATES, NEURAL_BATCH, development_setup, mixed_setup
from support.synthetic import REPO_PATHS

pytestmark = pytest.mark.usefixtures("restore_lightgbm_logger")


def test_a_run_searches_scores_evaluates_times_and_summarises(tmp_path: Path) -> None:
    setup = development_setup(tmp_path)
    directory = run_development(setup)
    loaded = load_run(directory, setup.run.partitions)
    assert loaded.end is not None
    assert loaded.end["outcome"] == "completed"
    counts = {kind: len(loaded.of_kind(kind)) for kind in ("run", "trial", "fit", "checkpoint")}
    assert counts == {"run": 1, "trial": 6, "fit": 6, "checkpoint": 6}
    assert len(loaded.of_kind("configuration")) == 2
    assert len(loaded.of_kind("score")) == 4  # V1 and V2 of each configuration
    assert len(loaded.of_kind("evaluation")) == 20  # four R3 capacities and the ranking, x4
    timings = loaded.of_kind("timing")
    by_measurement = {m: sum(t["measurement"] == m for t in timings) for m in ("fit", "search")}
    assert by_measurement == {"fit": 2, "search": 2}
    assert sum(t["measurement"] == "scoring" for t in timings) == 8
    assert sum(t["measurement"] == "latency" for t in timings) == 4
    summary = (directory / SUMMARY_NAME).read_text(encoding="utf-8")
    assert "LGB-u" in summary
    assert "LGB-b" in summary


def test_the_gpu_is_released_after_every_configuration_before_the_next_starts(
    tmp_path: Path,
) -> None:
    setup = development_setup(tmp_path, ("LGB-u", "LGB-b"))
    finished: list[int] = []

    def release() -> None:
        # called when the configuration just run has written its record and no other has started
        finished.append(
            len(load_run(setup.run.path, setup.run.partitions).of_kind("configuration"))
        )

    run_development(replace(setup, release=release))
    assert finished == [1, 2]


def test_the_reference_runs_first_and_the_configurations_follow_dr011_order() -> None:
    chosen = chosen_configurations(["Cat-b", "RES-u", "LR", "LGB-u", "LGB-b"])
    assert [c.name for c in chosen] == ["LR", "LGB-u", "RES-u", "LGB-b", "Cat-b"]
    everything = chosen_configurations(RUN_CONFIGURATIONS)
    assert [c.name for c in everything] == list(RUN_CONFIGURATIONS)
    assert RUN_CONFIGURATIONS[0] == "LR"
    assert len(RUN_CONFIGURATIONS) == 9
    with pytest.raises(SpecificationError, match="Not configurations of this run"):
        chosen_configurations(["LGB-u", "XGB-u"])
    with pytest.raises(SpecificationError, match="at least one"):
        chosen_configurations([])


def test_the_neural_models_and_the_reference_need_their_inputs_before_anything_is_read() -> None:
    options = DevelopmentOptions(plan=Path("plan"), approved_cost_sha256="a" * 64)
    with pytest.raises(SpecificationError, match="--preprocessing"):
        prepare_development(REPO_PATHS, replace(options, configurations=("LR",)))
    with pytest.raises(SpecificationError, match="--preprocessing"):
        prepare_development(REPO_PATHS, replace(options, configurations=("RES-u",)))
    with_artifact = replace(options, configurations=("RES-u",), preprocessing=Path("artifact"))
    with pytest.raises(SpecificationError, match="--preflight"):
        prepare_development(REPO_PATHS, with_artifact)


def test_a_failed_run_still_closes_and_writes_its_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(*_args: object) -> object:
        raise ModelError("the search failed")

    monkeypatch.setattr("review_capacity.development.runner.search_configuration", failing)
    setup = development_setup(tmp_path)
    with pytest.raises(ModelError, match="the search failed"):
        run_development(setup)
    loaded = load_run(setup.run.path, setup.run.partitions)
    assert loaded.end is not None
    assert loaded.end["outcome"] == "failed"
    assert "the search failed" in loaded.end["error"]
    assert (setup.run.path / SUMMARY_NAME).is_file()


def test_an_interrupted_run_still_closes_and_writes_its_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupting(*_args: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr("review_capacity.development.runner.search_configuration", interrupting)
    setup = development_setup(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        run_development(setup)
    loaded = load_run(setup.run.path, setup.run.partitions)
    assert loaded.end is not None
    assert loaded.end["outcome"] == "interrupted"
    assert (setup.run.path / SUMMARY_NAME).is_file()


def test_a_development_run_writes_inside_outputs_only() -> None:
    started = "2026-10-22T000000Z"
    options = DevelopmentOptions(plan=Path("plan"), approved_cost_sha256="a" * 64)
    default = run_directory_for(REPO_PATHS, options, started)
    assert default == REPO_PATHS.root / "outputs" / "development" / started
    inside = replace(options, output=Path("outputs/custom/run"))
    expected = (REPO_PATHS.root / "outputs" / "custom" / "run").resolve()
    assert run_directory_for(REPO_PATHS, inside, started) == expected
    unsafe_outputs = (
        Path("evidence/development/custom"),
        Path("/tmp/elsewhere"),
        Path("outputs/.."),
    )
    for unsafe in unsafe_outputs:
        with pytest.raises(SpecificationError, match="inside outputs/"):
            run_directory_for(REPO_PATHS, replace(options, output=unsafe), started)


def test_a_start_up_failure_after_the_run_record_closes_the_run(tmp_path: Path) -> None:
    """V1 without a fraud fails the V1 objective, which is built after the run record."""
    setup = development_setup(tmp_path)
    v1 = setup.search_inputs.v1
    no_fraud = replace(v1, labels=np.zeros_like(v1.labels))
    setup = replace(setup, search_inputs=replace(setup.search_inputs, v1=no_fraud))
    with pytest.raises(DataValidationError, match="V1 has no fraud"):
        run_development(setup)
    loaded = load_run(setup.run.path, setup.run.partitions)
    assert loaded.end is not None
    assert loaded.end["outcome"] == "failed"
    assert len(loaded.of_kind("run")) == 1
    assert (setup.run.path / SUMMARY_NAME).is_file()


def test_a_failure_while_adopting_the_candidate_files_closes_the_run(tmp_path: Path) -> None:
    def unreadable(_run: RunDirectory) -> dict[str, dict[str, str]]:
        raise FileAccessError("the candidate file cannot be read")

    setup = replace(development_setup(tmp_path), adopt=unreadable)
    with pytest.raises(FileAccessError, match="cannot be read"):
        run_development(setup)
    loaded = load_run(setup.run.path, setup.run.partitions)
    assert loaded.end is not None
    assert loaded.end["outcome"] == "failed"
    assert loaded.of_kind("trial") == []


def test_a_run_record_that_cannot_be_published_raises_and_leaves_no_closing_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = development_setup(tmp_path)

    def refuse(_record: object, *_parts: object) -> Path:
        raise FileAccessError("the run record cannot be written")

    monkeypatch.setattr(setup.run, "write_record", refuse)
    with pytest.raises(FileAccessError, match="run record cannot be written"):
        run_development(setup)
    assert not (setup.run.records_dir / f"{END_KIND}.json").exists()
    assert not (setup.run.path / SUMMARY_NAME).exists()


def test_the_session_conditions_are_recorded_in_every_timing(tmp_path: Path) -> None:
    setup = development_setup(tmp_path, ("LGB-u",))
    run_development(setup)
    timings = load_run(setup.run.path, setup.run.partitions).of_kind("timing")
    assert timings
    assert all("session_confirmed" not in t["settings"] for t in timings)
    assert all("load_average_1m" in t["settings"]["session"] for t in timings)


def test_every_timing_of_a_measured_run_carries_its_memory_and_the_fit_its_startup(
    tmp_path: Path,
) -> None:
    setup = replace(
        development_setup(tmp_path, ("LGB-u",)),
        memory=memory_setup(),
        startups={"lightgbm": Startup(0.5, "library import, once per process")},
    )
    run_development(setup)
    loaded = load_run(setup.run.path, setup.run.partitions)
    timings = loaded.of_kind("timing")
    assert {t["measurement"] for t in timings} == {"fit", "search", "scoring", "latency"}
    for timing in timings:
        assert timing["peak_host_mib"] == 512.0
        assert timing["memory_scope"]
        assert timing["settings"]["memory_problems"] == []
        assert timing["settings"]["host_memory_method"] == "clear_refs"
    (fit,) = (t for t in timings if t["measurement"] == "fit")
    assert fit["settings"]["startup_seconds"] == 0.5
    assert fit["settings"]["persistence_seconds"] >= 0
    summary = (setup.run.path / SUMMARY_NAME).read_text(encoding="utf-8")
    assert "Memory not measured" not in summary


def test_an_unapproved_cost_stops_the_run_before_a_directory_or_any_data_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    (plan / "cost_prediction.json").write_text("{}\n", encoding="utf-8")
    output = REPO_PATHS.root / "outputs" / "development" / "unit-test-never-created"
    monkeypatch.setattr(
        "review_capacity.development.runner.verify_lightgbm_build",
        lambda _expected: LightGBMIdentity("4.7.0", Path("lib"), "c" * 64),
    )

    def started(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("monitoring or start-up began before the plan was approved")

    monkeypatch.setattr("review_capacity.development.runner.measure_startups", started)
    monkeypatch.setattr("review_capacity.development.runner.MemorySetup.detect", started)
    options = DevelopmentOptions(
        plan=plan, approved_cost_sha256="0" * 64, output=output, configurations=("LGB-u",)
    )
    with pytest.raises(SpecificationError, match="not the approved cost prediction"):
        prepare_development(REPO_PATHS, options)
    assert not output.exists()


def test_a_run_of_the_reference_and_a_tree_and_both_neural_models_is_complete_and_valid(
    tmp_path: Path,
) -> None:
    setup = mixed_setup(tmp_path)
    directory = run_development(setup)
    loaded = load_run(directory, setup.run.partitions)  # every record satisfies its contract
    assert loaded.end is not None
    assert loaded.end["outcome"] == "completed"
    assert len(loaded.of_kind("preprocessing")) == 1
    assert len(loaded.of_kind("preflight")) == 6  # three candidates of each neural model
    configurations = loaded.of_kind("configuration")
    assert {r["configuration"] for r in configurations} == {"LR", "LGB-u", "RES-u", "FTT-b"}
    trials = {(t["configuration"], t["trial"]): t for t in loaded.of_kind("trial")}
    assert len(trials) == 12
    assert {t["unit"] for (name, _), t in trials.items() if name == "LR"} == {"convergence"}
    assert {t["unit"] for (name, _), t in trials.items() if name == "RES-u"} == {"epoch"}
    assert {t["unit"] for (name, _), t in trials.items() if name == "LGB-u"} == {"round"}
    fits = {(f["configuration"], f["trial"]): f for f in loaded.of_kind("fit")}
    assert fits[("LR", 1)]["preflight"] is None
    assert fits[("LR", 1)]["early_stopping"] is False
    assert fits[("FTT-b", 2)]["preflight"] is not None
    assert fits[("FTT-b", 2)]["preprocessing"] == fits[("RES-u", 1)]["preprocessing"]
    assert fits[("LGB-u", 1)]["preprocessing"] is None
    checkpoints = {c["configuration"] for c in loaded.of_kind("checkpoint")}
    assert checkpoints == {"LGB-u", "RES-u", "FTT-b"}  # the reference has none
    scores = {(s["configuration"], s["partition"]): s for s in loaded.of_kind("score")}
    assert len(scores) == 8
    assert scores[("RES-u", "V1")]["batch_rows"] == 64
    assert scores[("FTT-b", "V2")]["batch_rows"] == 8
    assert scores[("LR", "V2")]["batch_rows"] is None
    timings = loaded.of_kind("timing")
    assert {t["configuration"] for t in timings} == {"LR", "LGB-u", "RES-u", "FTT-b"}
    summary = (directory / SUMMARY_NAME).read_text(encoding="utf-8")
    for name in ("LR", "RES-u", "FTT-b"):
        assert name in summary
    assert "| Configuration | Family |" in summary
    assert "### Descriptive contrasts of V2 Recall@2%" in summary
    assert "**Contrasts not computed:**" in summary  # this run scored only four configurations


def test_a_reloaded_neural_model_must_match_its_candidate_and_its_batch_plan(
    tmp_path: Path,
) -> None:
    setup = mixed_setup(tmp_path, ("FTT-b",))
    assert setup.neural is not None
    run_development(setup)
    directory = trial_directory(setup.run, "FTT-b", 1)
    scoring = ScoringContext(
        run=setup.run,
        identity=setup.identity,
        declaration=setup.declaration,
        settings=setup.settings,
        features=setup.features,
        inputs=setup.scoring_inputs,
        session=setup.session,
        neural=setup.neural,
    )
    chosen = configuration("FTT-b")
    selected = TrialOutcome(
        "FTT-b", 1, dict(FTT_CANDIDATES[0]), 0.5, 1, 2, "maximum", 1.0, 0.1, None, 1,
        directory, {}, {}, {},
    )  # fmt: skip
    model = load_selected(scoring, chosen, selected)  # the saved model is the selected candidate
    assert model.fitted_iterations >= 1
    other_plan = replace(
        setup.neural, plans={"ft_transformer": {1: BatchPlan(NEURAL_BATCH, NEURAL_BATCH)}}
    )
    with pytest.raises(ModelError, match="recorded batch plan"):
        load_selected(replace(scoring, neural=other_plan), chosen, selected)
    other_candidate = replace(selected, hyperparameters=dict(FTT_CANDIDATES[1]))
    with pytest.raises(ModelError, match="selected candidate"):
        load_selected(scoring, chosen, other_candidate)
