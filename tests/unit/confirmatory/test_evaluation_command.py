"""The evaluation phase of the Test command, end to end with the heavy steps stubbed (Issue #207):
the two phases under one lock, the attempts and their records, the consumption marker and
``labels_accessed``, the resume, a killed attempt, the whole-run check and the closing record. The
outcome and inference records are those the production code builds on the synthetic outcome world,
rewritten to the identity and the score files of the run under test."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from review_capacity.confirmatory import closing, command
from review_capacity.confirmatory.attempts import allocate, allocated, phase_records
from review_capacity.confirmatory.command import TestRunOptions, run_test
from review_capacity.confirmatory.records import ConfirmatoryRecords, phase_sequence_problems
from review_capacity.confirmatory.state import (
    is_consumed_by,
    marker_path,
    scored_path,
)
from review_capacity.development.run_output import RunDirectory, load_run, timestamp
from review_capacity.errors import SpecificationError
from support import command_stubs as stubs
from support import confirmatory as c

ScoreFiles = dict[str, str]


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return stubs.install_evaluation_stubs(tmp_path, monkeypatch)


def options(**changes: Any) -> TestRunOptions:
    values: dict[str, Any] = {
        "tag": stubs.TAG,
        "development": Path("outputs/development/x"),
        "analyses": (Path("outputs/analysis/y"),),
        "started": c.RUN_ID,
        "phase": "all",
    }
    return TestRunOptions(**{**values, **changes})


def run(project: dict[str, Any], **changes: Any) -> Path:
    return run_test(project["paths"], options(**changes), release=lambda: None)


def read(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def state_of(directory: Path) -> str:
    return str(read(directory / "records" / "test_run.json")["identity"]["frozen_state_sha256"])


def test_the_two_phases_run_under_one_lock_and_the_run_is_closed(project: dict[str, Any]) -> None:
    directory = run(project)
    records = directory / "records"
    names = sorted(p.name for p in records.iterdir())
    assert "test_run_end.json" in names
    assert "test_inference.json" in names
    assert "test_phase-scoring-1.json" in names
    assert "test_phase-evaluation-1.json" in names
    end = read(records / "test_run_end.json")
    assert end["outcome"] == "completed"
    assert f"records/{'test_inference.json'}" in [e["path"] for e in end["records"]]
    evaluation = read(records / "test_phase-evaluation-1.json")
    assert (evaluation["outcome"], evaluation["labels_accessed"]) == ("completed", True)
    scoring = read(records / "test_phase-scoring-1.json")
    assert evaluation["scores_sha256"] == scoring["scores_sha256"]
    state = state_of(directory)
    assert is_consumed_by(project["paths"].root, state, c.RUN_ID)
    assert scored_path(project["paths"].root, state).exists()
    loaded = load_run(directory, records=ConfirmatoryRecords(c.outcome_context()))
    assert loaded.end is not None
    assert project["plan"].calls == 1


def test_a_failure_before_the_marker_records_no_label_access_and_the_resume_completes(
    project: dict[str, Any],
) -> None:
    project["plan"].fail_before_marker = True
    with pytest.raises(RuntimeError, match="features could not be read"):
        run(project)
    (directory,) = (project["paths"].root / "outputs" / "confirmatory").glob("20*")
    failed = read(directory / "records" / "test_phase-evaluation-1.json")
    assert (failed["outcome"], failed["labels_accessed"]) == ("failed", False)
    assert not marker_path(project["paths"].root, state_of(directory)).exists()
    assert not (directory / "records" / "test_run_end.json").exists()  # the run stays open
    run(project, phase="evaluation", resume=directory, started="2026-11-02T091000Z")
    phases = [read(directory / "records" / f"test_phase-evaluation-{n}.json") for n in (1, 2)]
    assert [p["outcome"] for p in phases] == ["failed", "completed"]
    assert (
        phase_sequence_problems(
            [read(directory / "records" / "test_phase-scoring-1.json"), *phases]
        )
        == []
    )
    assert (directory / "records" / "test_run_end.json").exists()


def test_a_failure_after_the_marker_records_label_access_and_the_scores_are_not_recomputed(
    project: dict[str, Any],
) -> None:
    project["plan"].fail_after_marker = True
    with pytest.raises(RuntimeError, match="labels could not be read"):
        run(project)
    (directory,) = (project["paths"].root / "outputs" / "confirmatory").glob("20*")
    failed = read(directory / "records" / "test_phase-evaluation-1.json")
    assert (failed["outcome"], failed["labels_accessed"]) == ("failed", True)
    scoring_calls = project["counters"].scoring_calls
    run(project, phase="evaluation", resume=directory, started="2026-11-02T091000Z")
    assert project["counters"].scoring_calls == scoring_calls  # no new scoring
    assert (directory / "records" / "test_run_end.json").exists()
    with pytest.raises(SpecificationError, match="consumed"):  # no new scoring phase either
        run(project, phase="scoring", started="2026-11-03T090000Z")


def test_a_killed_evaluation_attempt_is_recorded_as_interrupted(project: dict[str, Any]) -> None:
    project["plan"].fail_before_marker = True
    with pytest.raises(RuntimeError):
        run(project)
    (directory,) = (project["paths"].root / "outputs" / "confirmatory").glob("20*")
    reopened = RunDirectory.reopen(
        directory, "confirmatory", ConfirmatoryRecords(c.outcome_context())
    )
    allocate(reopened, 2, timestamp())  # allocated, then the process was killed
    assert allocated(reopened) == [1, 2]
    run(project, phase="evaluation", resume=directory, started="2026-11-02T091000Z")
    outcomes = [p["outcome"] for p in phase_records(reopened, "evaluation")]
    assert outcomes == ["failed", "interrupted", "completed"]
    interrupted = phase_records(reopened, "evaluation")[1]
    assert interrupted["labels_accessed"] is False
    assert "left no terminal record" in interrupted["error"]


def test_the_evaluation_needs_a_completed_scoring_and_an_unfinished_run(
    project: dict[str, Any],
) -> None:
    project["counters"].fail_next = True
    with pytest.raises(RuntimeError, match="failed to load"):
        run(project, phase="scoring")
    (directory,) = (project["paths"].root / "outputs" / "confirmatory").glob("20*")
    with pytest.raises(SpecificationError, match="follows a completed scoring"):
        run(project, phase="evaluation", resume=directory, started="2026-11-02T091000Z")
    assert project["plan"].calls == 0
    run(project, phase="all", resume=directory, started="2026-11-02T093000Z")  # scoring, then more
    with pytest.raises(SpecificationError, match="is closed"):
        run(project, phase="evaluation", resume=directory, started="2026-11-02T094000Z")


def test_a_run_that_fails_the_whole_run_check_is_not_recorded_as_completed(
    project: dict[str, Any],
) -> None:
    project["plan"].skip_a_workload = True
    with pytest.raises(SpecificationError, match="not complete"):
        run(project)
    (directory,) = (project["paths"].root / "outputs" / "confirmatory").glob("20*")
    failed = read(directory / "records" / "test_phase-evaluation-1.json")
    assert failed["outcome"] == "failed"
    assert not (directory / "records" / "test_run_end.json").exists()


def test_a_crash_between_the_completed_attempt_and_the_closing_is_reconciled(
    project: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    real_close = closing.close_run

    def failing(_run: RunDirectory) -> Path:
        raise OSError("the disk is full")

    monkeypatch.setattr(command, "close_run", failing)
    with pytest.raises(OSError, match="disk is full"):
        run(project)
    (directory,) = (project["paths"].root / "outputs" / "confirmatory").glob("20*")
    assert not (directory / "records" / "test_run_end.json").exists()
    calls = project["plan"].calls
    monkeypatch.setattr(command, "close_run", real_close)
    run(project, phase="evaluation", resume=directory, started="2026-11-02T091000Z")
    assert project["plan"].calls == calls  # nothing is computed again
    assert read(directory / "records" / "test_run_end.json")["outcome"] == "completed"


def test_labels_accessed_describes_the_attempt_and_not_the_run(project: dict[str, Any]) -> None:
    """Attempt 1 reaches the marker and fails; attempt 2 fails before it: only attempt 1 says so."""
    project["plan"].fail_after_marker = True
    with pytest.raises(RuntimeError, match="labels could not be read"):
        run(project)
    (directory,) = (project["paths"].root / "outputs" / "confirmatory").glob("20*")
    project["plan"].fail_before_marker = True
    with pytest.raises(RuntimeError, match="features could not be read"):
        run(project, phase="evaluation", resume=directory, started="2026-11-02T091000Z")
    run(project, phase="evaluation", resume=directory, started="2026-11-02T092000Z")
    flags = [
        (p["outcome"], p["labels_accessed"])
        for p in (
            read(directory / "records" / f"test_phase-evaluation-{n}.json") for n in (1, 2, 3)
        )
    ]
    assert flags == [("failed", True), ("failed", False), ("completed", True)]


def test_a_killed_attempt_that_had_reached_the_marker_is_reported_as_such(
    project: dict[str, Any],
) -> None:
    from review_capacity.confirmatory.attempts import mark_reaching

    project["plan"].fail_before_marker = True
    with pytest.raises(RuntimeError):
        run(project)
    (directory,) = (project["paths"].root / "outputs" / "confirmatory").glob("20*")
    reopened = RunDirectory.reopen(
        directory, "confirmatory", ConfirmatoryRecords(c.outcome_context())
    )
    allocate(reopened, 2, timestamp())
    mark_reaching(reopened, 2)  # killed after the intent and before the end of the attempt
    allocate(reopened, 3, timestamp())  # killed before it reached the marker
    run(project, phase="evaluation", resume=directory, started="2026-11-02T091000Z")
    records = {n: read(directory / "records" / f"test_phase-evaluation-{n}.json") for n in (2, 3)}
    assert records[2]["labels_accessed"] is True
    assert records[3]["labels_accessed"] is False
    assert all(r["outcome"] == "interrupted" for r in records.values())


@pytest.mark.parametrize("phase", ["all", "evaluation"])
def test_the_scoring_marker_is_repaired_before_any_evaluation_and_survives_an_early_failure(
    project: dict[str, Any], phase: str
) -> None:
    run(project, phase="scoring")
    (directory,) = (project["paths"].root / "outputs" / "confirmatory").glob("20*")
    marker = scored_path(project["paths"].root, state_of(directory))
    marker.unlink()  # a crash between the completed scoring record and its marker
    project["plan"].fail_before_marker = True
    with pytest.raises(RuntimeError, match="features could not be read"):
        run(project, phase=phase, resume=directory, started="2026-11-02T091000Z")
    assert marker.exists()  # repaired before the evaluation started
    with pytest.raises(SpecificationError, match="owns the state until its run ends"):
        run(project, phase="scoring", started="2026-11-03T090000Z")  # no second scoring anywhere
    assert not (project["paths"].root / "outputs" / "confirmatory" / "2026-11-03T090000Z").exists()


def test_a_scoring_marker_of_other_scores_is_refused(project: dict[str, Any]) -> None:
    from review_capacity.confirmatory.state import record_scoring

    run(project, phase="scoring")
    (directory,) = (project["paths"].root / "outputs" / "confirmatory").glob("20*")
    marker = scored_path(project["paths"].root, state_of(directory))
    marker.unlink()
    record_scoring(
        project["paths"].root, state_of(directory), run_id=c.RUN_ID, scores_sha256="d" * 64
    )
    with pytest.raises(SpecificationError, match="another run or other scores"):
        run(project, phase="evaluation", resume=directory, started="2026-11-02T091000Z")
    assert project["plan"].calls == 0


def test_a_resume_with_other_frozen_inputs_is_refused_before_the_labels(
    project: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    project["plan"].fail_after_marker = True
    with pytest.raises(RuntimeError):
        run(project)  # scoring done, the evaluation consumed and failed
    (directory,) = (project["paths"].root / "outputs" / "confirmatory").glob("20*")
    calls = project["plan"].calls
    real = command._prepare

    def other_policies(*args: Any, **kwargs: Any) -> Any:
        prepared = real(*args, **kwargs)
        inputs = dict(prepared.binding.frozen_inputs)
        inputs["thresholds"] = [{"path": "analysis/other.json", "sha256": "f" * 64}]
        inputs["h3_selection"] = {"path": "analysis/other-h3.json", "sha256": "e" * 64}
        binding = dataclasses.replace(prepared.binding, frozen_inputs=inputs)
        return dataclasses.replace(prepared, binding=binding)

    monkeypatch.setattr(command, "_prepare", other_policies)
    with pytest.raises(SpecificationError, match="was frozen with other inputs"):
        run(project, phase="evaluation", resume=directory, started="2026-11-02T091000Z")
    assert project["plan"].calls == calls  # the evaluation did not start again
