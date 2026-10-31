"""The scoring phase of the Test command with the heavy parts stubbed (Issue #203): the order of
the steps, the records, the repeat of a failed attempt, the lock, the consumption marker and the
refusals. No research data and no model is read."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from review_capacity.confirmatory import command
from review_capacity.confirmatory.command import TestRunOptions, run_test
from review_capacity.confirmatory.frozen import EvaluatorState
from review_capacity.confirmatory.records import (
    ConfirmatoryRecords,
    phase_sequence_problems,
    run_problems,
    scores_bundle_sha256,
)
from review_capacity.confirmatory.scoring import start_attempt
from review_capacity.confirmatory.state import consume, lock_path, locked, scored_path
from review_capacity.development.records import ALL_CONFIGURATIONS
from review_capacity.development.run_output import RunDirectory
from review_capacity.errors import SpecificationError
from review_capacity.hashing import file_sha256
from support import command_stubs as stubs
from support import confirmatory as c

TAG = stubs.TAG
COMMIT = stubs.COMMIT


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return stubs.install_stubs(tmp_path, monkeypatch)


def options(**changes: Any) -> TestRunOptions:
    values: dict[str, Any] = {
        "tag": TAG,
        "development": Path("outputs/development/x"),
        "analyses": (Path("outputs/analysis/y"),),
        "started": c.RUN_ID,
        "phase": "scoring",
    }
    return TestRunOptions(**{**values, **changes})


def run(project: dict[str, Any], **changes: Any) -> Path:
    return run_test(project["paths"], options(**changes), release=lambda: None)


def read(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def test_the_scoring_phase_writes_the_run_the_scores_and_the_phase_and_stays_open(
    project: dict[str, Any],
) -> None:
    directory = run(project)
    records = directory / "records"
    names = sorted(p.name for p in records.iterdir())
    assert "test_run.json" in names
    assert "test_phase-scoring-1.json" in names
    assert all(f"test_scores-{name}.json" in names for name in ALL_CONFIGURATIONS)
    assert not any(n.startswith("test_run_end") for n in names)  # the run stays open
    phase = read(records / "test_phase-scoring-1.json")
    assert phase["outcome"] == "completed"
    assert phase["labels_accessed"] is False
    files = [read(records / f"test_scores-{n}.json")["file"]["sha256"] for n in ALL_CONFIGURATIONS]
    assert phase["scores_sha256"] == scores_bundle_sha256(files)
    run_record = read(records / "test_run.json")
    assert run_record["identity"]["freeze"] == {"tag": TAG, "tag_commit": COMMIT}
    assert run_record["identity"]["run_type"] == "confirmatory"
    assert project["counters"].read_test_rows == 1
    assert not lock_path(
        project["paths"].root, run_record["identity"]["frozen_state_sha256"]
    ).exists()


def test_a_failed_attempt_is_kept_and_a_repeat_completes_the_phase(project: dict[str, Any]) -> None:
    project["counters"].fail_next = True
    with pytest.raises(RuntimeError, match="failed to load"):
        run(project)
    (directory,) = (project["paths"].root / "outputs" / "confirmatory").glob("20*")
    names = sorted(p.name for p in (directory / "records").iterdir())
    assert names == ["test_phase-scoring-1.json", "test_run.json"]  # no score record
    failed = read(directory / "records" / "test_phase-scoring-1.json")
    assert failed["outcome"] == "failed"
    assert "failed to load" in failed["error"]
    assert failed["scores_sha256"] is None
    again = run(project, resume=directory, started="2026-11-02T091000Z")
    assert again == directory
    phases = [read(directory / "records" / f"test_phase-scoring-{n}.json") for n in (1, 2)]
    assert [p["outcome"] for p in phases] == ["failed", "completed"]
    assert phase_sequence_problems(phases) == []
    assert (directory / "records" / "test_scores-LR.json").exists()


def test_a_completed_scoring_is_not_repeated(project: dict[str, Any]) -> None:
    directory = run(project)
    with pytest.raises(SpecificationError, match="scoring phase of this run is complete"):
        run(project, resume=directory, started="2026-11-02T091000Z")
    assert project["counters"].scoring_calls == 1


def test_a_consumed_frozen_state_refuses_a_new_scoring_before_anything_is_written(
    project: dict[str, Any],
) -> None:
    directory = run(project)
    state = read(directory / "records" / "test_run.json")["identity"]["frozen_state_sha256"]
    consume(project["paths"].root, state, run_id="r", scores_sha256="c" * 64)
    with pytest.raises(SpecificationError, match="consumed"):
        run(project, started="2026-11-03T090000Z")
    with pytest.raises(SpecificationError, match="consumed"):
        run(project, resume=directory, started="2026-11-03T091000Z")
    assert not (project["paths"].root / "outputs" / "confirmatory" / "2026-11-03T090000Z").exists()
    assert project["counters"].scoring_calls == 1


def test_a_held_lock_refuses_a_second_run(project: dict[str, Any]) -> None:
    project["counters"].fail_next = True
    with pytest.raises(RuntimeError):
        run(project)  # a failed scoring: nothing completed, nothing consumed
    (directory,) = (project["paths"].root / "outputs" / "confirmatory").glob("20*")
    state = state_of(directory)
    with (
        locked(project["paths"].root, state, run_id="other"),
        pytest.raises(SpecificationError, match="never removes a lock"),
    ):
        run(project, started="2026-11-03T090000Z")
    assert not (project["paths"].root / "outputs" / "confirmatory" / "2026-11-03T090000Z").exists()


def test_a_dirty_tree_stops_before_any_test_row_is_read(project: dict[str, Any]) -> None:
    project["state"]["evaluator"] = EvaluatorState(COMMIT, COMMIT, False, True)
    with pytest.raises(SpecificationError, match="uncommitted"):
        run(project)
    assert project["counters"].read_test_rows == 0
    assert not (project["paths"].root / "outputs" / "confirmatory").exists()


def test_a_head_other_than_the_tag_commit_stops_the_run(project: dict[str, Any]) -> None:
    project["state"]["evaluator"] = EvaluatorState("b" * 40, COMMIT, True, True)
    with pytest.raises(SpecificationError, match="not the commit of the freeze tag"):
        run(project)
    assert project["counters"].read_test_rows == 0


def test_a_resumed_run_keeps_the_identity_it_started_with(project: dict[str, Any]) -> None:
    project["counters"].fail_next = True
    with pytest.raises(RuntimeError):
        run(project)
    (directory,) = (project["paths"].root / "outputs" / "confirmatory").glob("20*")
    project["state"]["evaluator"] = EvaluatorState("b" * 40, "b" * 40, True, True)  # a new commit
    with pytest.raises(SpecificationError, match="another identity"):
        run(project, resume=directory, started="2026-11-02T091000Z")


def test_an_unknown_phase_and_an_evaluation_without_a_run_are_refused(
    project: dict[str, Any],
) -> None:
    with pytest.raises(SpecificationError, match="phase must be one of"):
        run(project, phase="inference")
    with pytest.raises(SpecificationError, match="continues an existing run"):
        run(project, phase="evaluation")
    assert not (project["paths"].root / "outputs" / "confirmatory").exists()


def test_the_records_of_a_run_load_as_a_confirmatory_record_set(project: dict[str, Any]) -> None:
    directory = run(project)
    context = ConfirmatoryRecords(c.context())
    record = context.read(directory / "records" / "test_run.json", "test_run")
    assert record["frozen_inputs"]["h3_configuration"] == "LGB-b"


def state_of(directory: Path) -> str:
    return str(read(directory / "records" / "test_run.json")["identity"]["frozen_state_sha256"])


def test_a_completed_scoring_is_recorded_against_the_frozen_state(project: dict[str, Any]) -> None:
    directory = run(project)
    marker = scored_path(project["paths"].root, state_of(directory))
    assert marker.exists()
    assert read(marker)["run_id"] == directory.name
    phase = read(directory / "records" / "test_phase-scoring-1.json")
    assert read(marker)["scores_sha256"] == phase["scores_sha256"]


def test_a_new_directory_cannot_score_again_once_a_scoring_completed(
    project: dict[str, Any],
) -> None:
    run(project)
    with pytest.raises(SpecificationError, match="scoring phase of this frozen state completed"):
        run(project, started="2026-11-03T090000Z")
    assert not (project["paths"].root / "outputs" / "confirmatory" / "2026-11-03T090000Z").exists()
    assert project["counters"].scoring_calls == 1


def test_a_failure_while_publishing_the_scores_is_recovered_without_scoring_again(
    project: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = RunDirectory.write_record
    written: list[str] = []

    def failing(self: RunDirectory, record: Any, *parts: Any) -> Path:
        if record["record"] == "test_scores":
            written.append(parts[0])
            if len(written) == 3:
                raise OSError("the disk is full")
        return original(self, record, *parts)

    monkeypatch.setattr(RunDirectory, "write_record", failing)
    with pytest.raises(OSError, match="disk is full"):
        run(project)
    monkeypatch.setattr(RunDirectory, "write_record", original)
    (directory,) = (project["paths"].root / "outputs" / "confirmatory").glob("20*")
    published = sorted((directory / "records").glob("test_scores-*.json"))
    assert len(published) == 2  # a part of the bundle was published
    assert project["counters"].scoring_calls == 1
    run(project, resume=directory, started="2026-11-02T091000Z")
    assert project["counters"].scoring_calls == 1  # the verified bundle was published, not rescored
    assert len(list((directory / "records").glob("test_scores-*.json"))) == len(ALL_CONFIGURATIONS)
    phases = [read(directory / "records" / f"test_phase-scoring-{n}.json") for n in (1, 2)]
    assert [p["outcome"] for p in phases] == ["failed", "completed"]
    assert phase_sequence_problems(phases) == []
    assert scored_path(project["paths"].root, state_of(directory)).exists()


def test_a_killed_attempt_is_recorded_and_never_overwritten(project: dict[str, Any]) -> None:
    project["counters"].fail_next = True
    with pytest.raises(RuntimeError):
        run(project)
    (directory,) = (project["paths"].root / "outputs" / "confirmatory").glob("20*")
    # a later attempt that was killed after it allocated its directory and wrote a score file
    records = ConfirmatoryRecords(c.context())
    reopened = RunDirectory.reopen(directory, "confirmatory", records)
    killed = start_attempt(reopened, 2, "2026-11-02T09:05:00.000000Z")
    (killed / "LR.parquet").write_text("partial scores", encoding="utf-8")
    run(project, resume=directory, started="2026-11-02T091000Z")
    outcomes = {n: read(directory / "records" / f"test_phase-scoring-{n}.json") for n in (1, 2, 3)}
    assert [outcomes[n]["outcome"] for n in (1, 2, 3)] == ["failed", "interrupted", "completed"]
    assert "left no terminal record" in outcomes[2]["error"]
    assert (killed / "LR.parquet").read_text(encoding="utf-8") == "partial scores"  # kept
    assert phase_sequence_problems(list(outcomes.values())) == []
    assert (directory / "scores" / "attempt-03" / "LR.parquet").exists()  # its own directory


def test_a_completed_run_without_its_marker_is_reconciled(project: dict[str, Any]) -> None:
    directory = run(project)
    marker = scored_path(project["paths"].root, state_of(directory))
    marker.unlink()  # a crash between the completed phase record and the marker
    with pytest.raises(SpecificationError, match="scoring phase of this run is complete"):
        run(project, resume=directory, started="2026-11-02T091000Z")
    assert marker.exists()
    assert project["counters"].scoring_calls == 1


def test_the_records_of_the_scoring_phase_are_consistent_as_a_run(project: dict[str, Any]) -> None:
    directory = run(project)
    records = [read(path) for path in sorted((directory / "records").glob("*.json"))]
    assert run_problems(records, complete=False) == []


def test_the_files_read_after_the_binding_must_be_the_frozen_ones(tmp_path: Path) -> None:
    frozen = {"preprocessing": {"path": "p/manifest.json", "sha256": ""}}
    target = tmp_path / "p" / "manifest.json"
    target.parent.mkdir()
    target.write_text("manifest", encoding="utf-8")
    frozen["preprocessing"]["sha256"] = file_sha256(target)
    command._check_frozen_files(tmp_path, frozen, ("preprocessing",))
    target.write_text("another manifest", encoding="utf-8")
    with pytest.raises(SpecificationError, match="differs from the frozen preprocessing"):
        command._check_frozen_files(tmp_path, frozen, ("preprocessing",))
