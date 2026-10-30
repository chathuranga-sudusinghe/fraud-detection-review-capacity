"""A confirmatory run directory: the development run machinery (atomic records, identity, closing
hashes) driven by the confirmatory record set, and unchanged for the development one."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from review_capacity.config import load_dataset_spec
from review_capacity.confirmatory.records import RECORD_CONTRACTS, ConfirmatoryRecords
from review_capacity.development.records import RECORD_CONTRACTS as DEVELOPMENT_CONTRACTS
from review_capacity.development.records import DevelopmentPartitions
from review_capacity.development.run_output import (
    DevelopmentRecords,
    RunDirectory,
    load_run,
)
from review_capacity.errors import RecordContractError, SpecificationError
from support import confirmatory as c
from support.synthetic import REPO_PATHS

RECORDS = ConfirmatoryRecords(c.context())


def new_run(root: Path) -> RunDirectory:
    return RunDirectory.create(root, None, "confirmatory", c.RUN_ID, records=RECORDS)


def write_all(run: RunDirectory) -> None:
    run.write_record(c.test_run())
    run.write_record(c.phase("scoring", 1, "failed"), "scoring", 1)
    run.write_record(c.phase("scoring", 2, "completed"), "scoring", 2)
    run.write_record(c.scores("LGB-b"), "LGB-b")
    run.write_record(c.scores("LR"), "LR")
    run.write_record(c.phase("evaluation", 1, "completed"), "evaluation", 1)
    run.write_record(c.inference())


def test_a_confirmatory_run_is_written_closed_and_read_back(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    assert run.path == tmp_path / "outputs" / "confirmatory" / c.RUN_ID
    write_all(run)
    end = run.finish("completed")
    assert end.name == "test_run_end.json"

    loaded = load_run(run.path, records=RECORDS)
    assert loaded.end is not None
    assert loaded.end["outcome"] == "completed"
    assert [name for name, _ in loaded.records] == [
        "test_run.json",
        "test_phase-evaluation-1.json",
        "test_phase-scoring-1.json",
        "test_phase-scoring-2.json",
        "test_scores-LGB-b.json",
        "test_scores-LR.json",
        "test_inference.json",
    ]
    assert loaded.identity["run_type"] == "confirmatory"
    assert len(loaded.of_kind("test_scores")) == 2
    assert {entry["path"] for entry in loaded.end["records"]} == {
        f"records/{name}" for name, _ in loaded.records
    }


def test_the_closing_hashes_detect_a_changed_record(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    write_all(run)
    run.finish("completed")
    path = run.path / "records" / "test_scores-LR.json"
    path.write_text(path.read_text(encoding="utf-8").replace("0.0001", "0.0002"), encoding="utf-8")
    with pytest.raises(RecordContractError, match="records differ from test_run_end"):
        load_run(run.path, records=RECORDS)


def test_the_first_record_must_open_the_run(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    with pytest.raises(SpecificationError, match="first record of a run is test_run"):
        run.write_record(c.phase())


def test_the_run_record_has_no_name_part_and_the_end_record_is_written_by_finish(
    tmp_path: Path,
) -> None:
    run = new_run(tmp_path)
    with pytest.raises(SpecificationError, match=r"test_run record is test_run\.json"):
        run.write_record(c.test_run(), "extra")
    run.write_record(c.test_run())
    with pytest.raises(SpecificationError, match="written by finish"):
        run.write_record(c.run_end())


def test_a_record_is_never_overwritten_and_a_closed_run_takes_no_record(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    run.write_record(c.test_run())
    run.write_record(c.scores("LGB-b"), "LGB-b")
    with pytest.raises(SpecificationError, match="never overwritten"):
        run.write_record(c.scores("LGB-b"), "LGB-b")
    run.finish("completed")
    with pytest.raises(SpecificationError, match="is closed"):
        run.write_record(c.scores("LR"), "LR")
    with pytest.raises(SpecificationError, match="already closed"):
        run.finish("completed")


def test_an_identity_other_than_the_runs_is_refused(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    run.write_record(c.test_run())
    other = c.mutated(c.phase(), "identity.freeze.tag", "another-tag")
    with pytest.raises(SpecificationError, match="identity differs from the run's own"):
        run.write_record(other, "scoring", 1)
    wrong_id = c.mutated(c.test_run(), "identity.run_id", "2026-11-03T090000Z")
    with pytest.raises(SpecificationError, match="differs from the run"):
        new_run(tmp_path / "second").write_record(wrong_id)


def test_a_record_that_breaks_its_contract_is_not_written(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    run.write_record(c.test_run())
    with pytest.raises(RecordContractError, match="does not satisfy its contract"):
        run.write_record(c.mutated(c.scores(), "days.first", 127), "LGB-b")
    assert not (run.path / "records" / "test_scores-LGB-b.json").exists()


def test_only_confirmatory_kinds_are_known_to_a_confirmatory_run(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    with pytest.raises(SpecificationError, match="Unknown record kind"):
        run.record_path("evaluation")
    assert not set(RECORD_CONTRACTS) & set(DEVELOPMENT_CONTRACTS)


def fail_inside(run: RunDirectory) -> None:
    with run.closing():
        run.write_record(c.test_run())
        raise RuntimeError("boom")


def test_a_failed_run_closes_with_its_error(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        fail_inside(run)
    loaded = load_run(run.path, records=RECORDS)
    assert loaded.end is not None
    assert loaded.end["outcome"] == "failed"
    assert "boom" in loaded.end["error"]


def test_a_run_directory_needs_partitions_or_a_record_set(tmp_path: Path) -> None:
    with pytest.raises(SpecificationError, match="needs the development partitions"):
        RunDirectory(tmp_path / "x", None, "development", c.RUN_ID)
    with pytest.raises(SpecificationError, match="needs the development partitions"):
        load_run(tmp_path)
    run = new_run(tmp_path)
    with pytest.raises(SpecificationError, match="no development partitions"):
        _ = run.partitions


def test_the_development_record_set_is_the_default_and_unchanged() -> None:
    partitions = DevelopmentPartitions.from_spec(load_dataset_spec(REPO_PATHS), REPO_PATHS)
    development = DevelopmentRecords(partitions)
    assert development.contracts is DEVELOPMENT_CONTRACTS
    assert (development.run_kind, development.end_kind) == ("run", "run_end")
    assert (RECORDS.run_kind, RECORDS.end_kind) == ("test_run", "test_run_end")
    run: Any = RunDirectory(Path("x"), partitions, "development", "2026-10-25T134341Z")
    assert run.partitions is partitions
    assert isinstance(run.records, DevelopmentRecords)
