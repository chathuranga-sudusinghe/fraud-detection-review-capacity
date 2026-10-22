"""Run directories and record persistence: new directories only, atomic validated records,
preservation after an interruption, closing with hashes (synthetic records; no data read)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from review_capacity.development.records import dump_record, validate_record
from review_capacity.development.run_output import (
    END_KIND,
    RunDirectory,
    check_new_directory,
    default_directory,
    load_run,
    run_id,
)
from review_capacity.errors import FileAccessError, RecordContractError, SpecificationError
from review_capacity.files import write_new_atomically
from support.development import PARTITIONS, example
from support.runs import RUN_ID, new_run, write_records


def test_run_id_is_the_utc_start_time() -> None:
    assert run_id(datetime(2026, 10, 20, 9, 15, 30, tzinfo=UTC)) == "2026-10-20T091530Z"


def test_the_default_directory_is_dated_under_outputs(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    assert run.path == default_directory(tmp_path, "development", RUN_ID)
    assert run.path == tmp_path / "outputs" / "development" / RUN_ID
    assert run.records_dir.is_dir()


def test_a_directory_in_use_is_never_reused(tmp_path: Path) -> None:
    new_run(tmp_path)
    with pytest.raises(SpecificationError, match="already exists"):
        new_run(tmp_path)


def test_an_explicit_output_must_be_new_and_inside_the_project(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    run = RunDirectory.create(root, PARTITIONS, "development", RUN_ID, Path("outputs/mine"))
    assert run.path == root / "outputs" / "mine"
    with pytest.raises(SpecificationError, match="already exists"):
        RunDirectory.create(root, PARTITIONS, "development", RUN_ID, Path("outputs/mine"))
    with pytest.raises(SpecificationError, match="outside the project root"):
        check_new_directory(tmp_path / "elsewhere", root)
    with pytest.raises(SpecificationError, match="outside the project root"):
        check_new_directory(Path("../elsewhere"), root)


def test_only_the_runs_own_log_may_be_in_a_new_directory(tmp_path: Path) -> None:
    target = tmp_path / "outputs" / "development" / RUN_ID
    target.mkdir(parents=True)
    (target / "run.log").write_text("log\n", encoding="utf-8")
    with pytest.raises(SpecificationError, match="already exists"):
        check_new_directory(target, tmp_path)
    assert check_new_directory(target, tmp_path, log_opened=True) == target.resolve()
    (target / "other.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SpecificationError, match="already exists"):
        check_new_directory(target, tmp_path, log_opened=True)


def test_an_existing_file_is_not_a_directory_for_a_run(tmp_path: Path) -> None:
    (tmp_path / "file").write_text("x", encoding="utf-8")
    with pytest.raises(SpecificationError, match="already exists"):
        check_new_directory(tmp_path / "file", tmp_path)


def test_records_are_validated_and_written_as_new_files(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    path = run.write_record(example("run"))
    assert path == run.records_dir / "run.json"
    validate_record(load_run(run.path, PARTITIONS).records[0][1], PARTITIONS, "run")
    trial = run.write_record(example("trial"), "LGB-u", 1)
    assert trial.name == "trial-LGB-u-1.json"
    with pytest.raises(SpecificationError, match="never overwritten"):
        run.write_record(example("trial"), "LGB-u", 1)


def test_a_record_breaking_its_contract_is_refused_and_leaves_no_file(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    run.write_record(example("run"))
    with pytest.raises(RecordContractError):
        run.write_record(example("trial") | {"status": "bogus"}, "LGB-u", 1)
    assert [p.name for p in run.records_dir.iterdir()] == ["run.json"]


def test_the_run_record_comes_first(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    with pytest.raises(SpecificationError, match="first record"):
        run.write_record(example("trial"), "LGB-u", 1)


def test_every_record_carries_the_runs_identity(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    other = example("run")
    other["identity"]["run_id"] = "2026-10-21T000000Z"
    with pytest.raises(SpecificationError, match="differs from the run"):
        run.write_record(other)
    run.write_record(example("run"))
    changed = example("trial")
    changed["identity"]["settings_sha256"] = "0" * 64
    with pytest.raises(SpecificationError, match="identity differs"):
        run.write_record(changed, "LGB-u", 1)


def test_record_names_are_checked(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    run.write_record(example("run"))
    with pytest.raises(SpecificationError, match="letters, digits"):
        run.write_record(example("trial"), "../x")
    with pytest.raises(SpecificationError, match="Unknown record kind"):
        run.record_path("notes")


def test_run_end_is_written_by_finish_only(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    run.write_record(example("run"))
    with pytest.raises(SpecificationError, match="finish"):
        run.write_record(example(END_KIND))


def test_finish_lists_every_record_with_its_hash_and_closes_the_run(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    write_records(run)
    run.finish("completed")
    loaded = load_run(run.path, PARTITIONS)
    assert loaded.end is not None
    assert [e["path"] for e in loaded.end["records"]] == sorted(
        f"records/{name}" for name, _ in loaded.records
    )
    assert loaded.end["outcome"] == "completed"
    with pytest.raises(SpecificationError, match="closed"):
        run.write_record(example("trial"), "LGB-u", 2)


def test_a_run_without_a_run_record_cannot_close(tmp_path: Path) -> None:
    with pytest.raises(SpecificationError, match="cannot close"):
        new_run(tmp_path).finish("completed")


def test_completed_records_survive_an_interruption(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    summarised: list[Path] = []

    def interrupted() -> None:
        with run.closing(summarised.append):
            run.write_record(example("run"))
            run.write_record(example("trial"), "LGB-u", 1)
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interrupted()
    loaded = load_run(run.path, PARTITIONS)
    assert [name for name, _ in loaded.records] == ["run.json", "trial-LGB-u-1.json"]
    assert loaded.end is not None
    assert (loaded.end["outcome"], loaded.end["error"]) == ("interrupted", "KeyboardInterrupt")
    assert summarised == [run.path]


def test_a_failure_is_recorded_and_re_raised(tmp_path: Path) -> None:
    run = new_run(tmp_path)

    def failing() -> None:
        with run.closing():
            run.write_record(example("run"))
            raise ZeroDivisionError("boom")

    with pytest.raises(ZeroDivisionError):
        failing()
    end = load_run(run.path, PARTITIONS).end
    assert end is not None
    assert (end["outcome"], end["error"]) == ("failed", "ZeroDivisionError: boom")


def test_a_killed_run_has_no_end_and_ignores_a_partial_write(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    run.write_record(example("run"))
    (run.records_dir / f".trial-LGB-u-1.json.{123}.tmp").write_text('{"partial', encoding="utf-8")
    loaded = load_run(run.path, PARTITIONS)
    assert loaded.end is None
    assert [name for name, _ in loaded.records] == ["run.json"]


def test_a_directory_without_a_run_record_is_not_a_run(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    with pytest.raises(RecordContractError, match="exactly one run record"):
        load_run(run.path, PARTITIONS)


def test_a_closed_run_detects_changed_extra_and_missing_records(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    write_records(run)
    run.finish("completed")
    trial = run.records_dir / "trial-LGB-u-1.json"
    original = trial.read_text(encoding="utf-8")
    trial.write_text(original.replace('"runtime_seconds": 12.5', '"runtime_seconds": 13.5'))
    with pytest.raises(RecordContractError, match="changed"):
        load_run(run.path, PARTITIONS)
    trial.write_text(original, encoding="utf-8")
    load_run(run.path, PARTITIONS)
    extra = run.records_dir / "trial-LGB-u-2.json"
    extra.write_text(original, encoding="utf-8")
    with pytest.raises(RecordContractError, match="not listed"):
        load_run(run.path, PARTITIONS)
    extra.unlink()
    trial.unlink()
    with pytest.raises(RecordContractError, match="missing"):
        load_run(run.path, PARTITIONS)


def test_records_of_different_runs_are_refused(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    run.write_record(example("run"))
    foreign = example("trial")
    foreign["identity"]["run_id"] = "2026-10-21T000000Z"
    (run.records_dir / "trial-LGB-u-1.json").write_text(
        dump_record(foreign, PARTITIONS), encoding="utf-8"
    )
    with pytest.raises(RecordContractError, match="different runs"):
        load_run(run.path, PARTITIONS)


def test_a_file_name_must_match_its_record_kind(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    run.write_record(example("run"))
    (run.records_dir / "run.json").rename(run.records_dir / "trial.json")
    with pytest.raises(RecordContractError, match="does not match"):
        load_run(run.path, PARTITIONS)


def test_other_files_in_records_are_refused(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    run.write_record(example("run"))
    (run.records_dir / "scores.parquet").write_bytes(b"x")
    with pytest.raises(RecordContractError, match="unexpected entries"):
        load_run(run.path, PARTITIONS)


def test_a_run_is_closed_once(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    write_records(run)
    run.finish("failed", "boom")
    with pytest.raises(SpecificationError, match="already closed"):
        run.finish("completed")
    end = load_run(run.path, PARTITIONS).end
    assert end is not None
    assert (end["outcome"], end["error"]) == ("failed", "boom")


def test_a_record_that_appears_while_writing_is_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = new_run(tmp_path)
    run.write_record(example("run"))
    path = run.record_path("trial", "LGB-u", 1)

    def competing(target: Path, text: str) -> None:
        target.write_text("competitor\n", encoding="utf-8")
        write_new_atomically(target, text)

    monkeypatch.setattr("review_capacity.development.run_output.write_new_atomically", competing)
    with pytest.raises(FileAccessError, match="already exists"):
        run.write_record(example("trial"), "LGB-u", 1)
    assert path.read_text(encoding="utf-8") == "competitor\n"


def test_a_failed_first_write_does_not_fix_the_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = new_run(tmp_path)

    def failing(_target: Path, _text: str) -> None:
        raise FileAccessError("disk full")

    monkeypatch.setattr("review_capacity.development.run_output.write_new_atomically", failing)
    with pytest.raises(FileAccessError, match="disk full"):
        run.write_record(example("run"))
    monkeypatch.undo()
    with pytest.raises(SpecificationError, match="first record"):
        run.write_record(example("trial"), "LGB-u", 1)
    run.write_record(example("run"))


def test_there_is_one_run_record_named_run_json(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    run.write_record(example("run"))
    with pytest.raises(SpecificationError, match=r"run\.json"):
        run.write_record(example("run"), "alternate")
    original = (run.records_dir / "run.json").read_bytes()
    (run.records_dir / "run-alternate.json").write_bytes(original)
    with pytest.raises(RecordContractError, match=r"named run\.json"):
        load_run(run.path, PARTITIONS)


def test_run_end_exists_only_as_run_end_json(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    write_records(run)
    run.finish("completed")
    ended = run.records_dir / "run_end.json"
    (run.records_dir / "run_end-copy.json").write_bytes(ended.read_bytes())
    with pytest.raises(RecordContractError, match=r"named run_end\.json"):
        load_run(run.path, PARTITIONS)


def test_the_original_exception_survives_failing_cleanup(tmp_path: Path) -> None:
    run = new_run(tmp_path)

    def summarise(_path: Path) -> None:
        raise RuntimeError("summary")

    def body() -> None:
        with run.closing(summarise):
            run.write_record(example("run"))
            raise ValueError("original")

    with pytest.raises(ValueError, match="original"):
        body()
    end = load_run(run.path, PARTITIONS).end
    assert end is not None
    assert end["outcome"] == "failed"


def test_an_interrupt_survives_a_failing_close_and_still_summarises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = new_run(tmp_path)
    summarised: list[Path] = []

    def failing(*_args: Any, **_kwargs: Any) -> Path:
        raise FileAccessError("cannot close")

    monkeypatch.setattr(run, "finish", failing)

    def body() -> None:
        with run.closing(summarised.append):
            run.write_record(example("run"))
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        body()
    assert summarised == [run.path]


def test_a_completed_run_whose_cleanup_fails_is_an_error(tmp_path: Path) -> None:
    run = new_run(tmp_path)

    def summarise(_path: Path) -> None:
        raise RuntimeError("summary")

    def body() -> None:
        with run.closing(summarise):
            run.write_record(example("run"))

    with pytest.raises(RuntimeError, match="summary"):
        body()
    assert load_run(run.path, PARTITIONS).end is not None  # closed before the summary failed


def test_a_relative_root_is_applied_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "project").mkdir()
    monkeypatch.chdir(tmp_path)
    run = RunDirectory.create(Path("project"), PARTITIONS, "development", RUN_ID)
    assert run.path == tmp_path / "project" / "outputs" / "development" / RUN_ID


def test_a_records_directory_that_is_a_link_is_refused(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    run.write_record(example("run"))
    real = run.path / "elsewhere"
    run.records_dir.rename(real)
    run.records_dir.symlink_to(real)
    with pytest.raises(RecordContractError, match="not a link"):
        load_run(run.path, PARTITIONS)


def test_a_record_file_that_is_a_link_is_refused(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    run.write_record(example("run"))
    target = tmp_path / "outside.json"
    target.write_bytes((run.records_dir / "run.json").read_bytes())
    (run.records_dir / "trial-LGB-u-1.json").symlink_to(target)
    with pytest.raises(RecordContractError, match="unexpected entries"):
        load_run(run.path, PARTITIONS)


def test_an_interrupt_inside_closing_does_not_replace_the_original_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = new_run(tmp_path)
    summarised: list[Path] = []

    def interrupted(*_args: Any, **_kwargs: Any) -> Path:
        raise KeyboardInterrupt

    monkeypatch.setattr(run, "finish", interrupted)

    def body() -> None:
        with run.closing(summarised.append):
            run.write_record(example("run"))
            raise ValueError("original")

    with pytest.raises(ValueError, match="original"):
        body()
    assert summarised == [run.path]


def test_an_interrupt_while_closing_a_completed_run_is_raised_after_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = new_run(tmp_path)
    summarised: list[Path] = []

    def interrupted(*_args: Any, **_kwargs: Any) -> Path:
        raise KeyboardInterrupt

    monkeypatch.setattr(run, "finish", interrupted)

    def body() -> None:
        with run.closing(summarised.append):
            run.write_record(example("run"))

    with pytest.raises(KeyboardInterrupt):
        body()
    assert summarised == [run.path]


def test_an_interrupt_in_the_summary_does_not_replace_the_original_exception(
    tmp_path: Path,
) -> None:
    run = new_run(tmp_path)

    def interrupted(_path: Path) -> None:
        raise KeyboardInterrupt

    def body() -> None:
        with run.closing(interrupted):
            run.write_record(example("run"))
            raise ValueError("original")

    with pytest.raises(ValueError, match="original"):
        body()
    end = load_run(run.path, PARTITIONS).end
    assert end is not None
    assert end["outcome"] == "failed"
