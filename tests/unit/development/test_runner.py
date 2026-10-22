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
from review_capacity.development.startup import Startup
from review_capacity.errors import (
    DataValidationError,
    FileAccessError,
    ModelError,
    SpecificationError,
)
from review_capacity.models.device import LightGBMIdentity
from support.memory import memory_setup
from support.search import development_setup
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


def test_the_configurations_follow_dr011_order_and_are_tree_based() -> None:
    chosen = chosen_configurations(["Cat-b", "LGB-u", "LGB-b"])
    assert [c.name for c in chosen] == ["LGB-u", "LGB-b", "Cat-b"]
    with pytest.raises(SpecificationError, match="Not tree-based configurations"):
        chosen_configurations(["LGB-u", "RES-u"])
    with pytest.raises(SpecificationError, match="at least one"):
        chosen_configurations([])


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
    options = DevelopmentOptions(plan=plan, approved_cost_sha256="0" * 64, output=output)
    with pytest.raises(SpecificationError, match="not the approved cost prediction"):
        prepare_development(REPO_PATHS, options)
    assert not output.exists()
