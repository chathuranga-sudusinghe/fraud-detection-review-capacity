"""Command-line parsing, the error boundary, exit codes and profile-data on synthetic processed
files (no project data is read)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from review_capacity import cli
from review_capacity.cli import build_parser, main
from review_capacity.data.processed import read_development_features
from review_capacity.data.profiling import profile_columns
from review_capacity.errors import (
    DataValidationError,
    EnvironmentMismatchError,
    FileAccessError,
    ProjectRootNotFoundError,
    ReviewCapacityError,
    SealedTestError,
    SpecificationError,
)
from review_capacity.hashing import file_sha256
from review_capacity.paths import ProjectPaths
from support.catalog import catalog_text
from support.synthetic import Synthetic, build_processed

pytestmark = pytest.mark.usefixtures("restore_logging")


def test_prepare_data_options() -> None:
    args = build_parser().parse_args(["prepare-data", "--check-only", "--root", "/tmp/project"])
    assert args.command == "prepare-data"
    assert args.check_only is True
    assert args.root == Path("/tmp/project")


def test_prepare_data_defaults() -> None:
    args = build_parser().parse_args(["prepare-data"])
    assert args.check_only is False
    assert args.root is None


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_logging_options() -> None:
    args = build_parser().parse_args(["--quiet", "--log-file", "run.log", "prepare-data"])
    assert args.quiet is True
    assert args.log_file == Path("run.log")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--quiet", "--verbose", "prepare-data"])


def _error_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if " ERROR " in line]


def test_project_error_is_one_line_with_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["prepare-data", "--root", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert len(_error_lines(err)) == 1
    assert "ProjectRootNotFoundError" in err
    assert "Traceback" not in err


def test_debug_adds_the_traceback_and_keeps_the_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_file = tmp_path / "run.log"
    code = main(["--debug", "--log-file", str(log_file), "prepare-data", "--root", str(tmp_path)])
    assert code == 2
    assert "Traceback" in capsys.readouterr().err
    assert "Traceback" in log_file.read_text(encoding="utf-8")


def test_unopenable_log_file_exits_with_code_5(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("", encoding="utf-8")
    assert main(["--log-file", str(blocker / "run.log"), "prepare-data"]) == 5
    assert "FileAccessError" in capsys.readouterr().err


def test_unexpected_error_is_logged_with_traceback_and_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(_root: Path | None = None) -> ProjectPaths:
        raise RuntimeError("defect")

    monkeypatch.setattr(ProjectPaths, "discover", broken)
    log_file = tmp_path / "run.log"
    with pytest.raises(RuntimeError, match="defect"):
        main(["--log-file", str(log_file), "prepare-data"])
    text = log_file.read_text(encoding="utf-8")
    assert "Unexpected error (a defect in the code)" in text
    assert "Traceback" in text


def test_interruption_exits_with_code_130(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(_root: Path | None = None) -> ProjectPaths:
        raise KeyboardInterrupt

    monkeypatch.setattr(ProjectPaths, "discover", interrupted)
    assert main(["prepare-data"]) == 130


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (SpecificationError("s"), 2),
        (ProjectRootNotFoundError("r"), 2),
        (DataValidationError("d"), 3),
        (SealedTestError("t"), 4),
        (FileAccessError("f"), 5),
        (EnvironmentMismatchError("e"), 6),
    ],
)
def test_every_project_error_has_its_exit_code(
    monkeypatch: pytest.MonkeyPatch, error: ReviewCapacityError, code: int
) -> None:
    def failing(_root: Path | None = None) -> ProjectPaths:
        raise error

    monkeypatch.setattr(ProjectPaths, "discover", failing)
    assert main(["prepare-data"]) == code


def test_log_file_keeps_the_cause_chain_without_debug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing(_root: Path | None = None) -> ProjectPaths:
        try:
            int("not a number")
        except ValueError as cause:
            raise DataValidationError("bad value") from cause
        raise AssertionError  # not reached

    monkeypatch.setattr(ProjectPaths, "discover", failing)
    log_file = tmp_path / "run.log"
    assert main(["--log-file", str(log_file), "prepare-data"]) == 3
    terminal = capsys.readouterr().err
    assert "DataValidationError: bad value" in terminal
    assert "Traceback" not in terminal
    text = log_file.read_text(encoding="utf-8")
    assert "Traceback" in text
    assert "invalid literal for int()" in text  # the cause


def _profile_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Synthetic:
    """Synthetic processed files with no raw file, a matching catalog and the specifications."""
    data = build_processed(tmp_path)
    development = read_development_features(data.paths, data.features, data.dataset)
    data.paths.feature_catalog.parent.mkdir(parents=True)
    data.paths.feature_catalog.write_text(
        catalog_text(profile_columns(development, data.features)), encoding="utf-8"
    )
    monkeypatch.setattr(ProjectPaths, "discover", lambda _root=None: data.paths)
    monkeypatch.setattr(cli, "load_feature_spec", lambda _path: data.features)
    monkeypatch.setattr(cli, "load_dataset_spec", lambda _paths: data.dataset)
    return data


def test_profile_data_reads_no_raw_file_and_saves_with_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _profile_project(tmp_path, monkeypatch)
    assert not data.paths.raw_dir.joinpath("train_transaction.csv").exists()
    assert main(["profile-data", "--save"]) == 0
    profile = data.paths.processed_dir / "feature_profile.csv"
    record = json.loads(
        (data.paths.processed_dir / "feature_profile.provenance.json").read_text(encoding="utf-8")
    )
    assert record["sha256"] == file_sha256(profile)
    assert record["inputs"]["development_joined.parquet"] == file_sha256(
        data.paths.processed_file("development_joined")
    )
    assert record["differences"] == 0
    assert record["checks"] > 0
    assert set(record["configuration"]) == {"features.yaml", "dataset.yaml"}


def test_profile_data_reports_a_difference_with_exit_code_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _profile_project(tmp_path, monkeypatch)
    catalog = data.paths.feature_catalog
    text = catalog.read_text(encoding="utf-8")
    catalog.write_text(text.replace("| 25.0 % | 3 |", "| 30.0 % | 3 |", 1), encoding="utf-8")
    assert main(["profile-data"]) == 1
    assert not (data.paths.processed_dir / "feature_profile.csv").exists()


def test_profile_data_without_processed_files_is_a_file_access_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _profile_project(tmp_path, monkeypatch)
    data.paths.processed_file("development_joined").unlink()
    assert main(["profile-data"]) == 5
