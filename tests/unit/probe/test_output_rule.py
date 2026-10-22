"""The probe commands' output rule (Issue #96, #120): a run writes to a dated directory of its own
inside the repository, never reusing a directory that holds another run (synthetic; no data)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from review_capacity import cli
from review_capacity.cli import PROBE_OUTPUT_ROOT, build_parser
from review_capacity.errors import SpecificationError
from review_capacity.paths import ProjectPaths
from review_capacity.probe.records import (
    ACCEPTANCE_LOG_NAME,
    ACCEPTANCE_ROOT,
    LOG_NAME,
    check_acceptance_output,
    check_output,
)
from support.synthetic import REPO_PATHS

pytestmark = pytest.mark.usefixtures("restore_logging")

DATED = re.compile(r"\d{4}-\d{2}-\d{2}T\d{6}Z")


def test_the_output_roots_are_inside_the_ignored_outputs_directory() -> None:
    assert PROBE_OUTPUT_ROOT.parts == ("outputs", "gpu_acceptance_probe")
    assert ACCEPTANCE_ROOT.parts == ("outputs", "catboost_objective_acceptance")
    ignored = (REPO_PATHS.root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "outputs/" in ignored


def test_the_default_probe_output_is_a_new_dated_directory_fixed_on_the_arguments(
    tmp_path: Path,
) -> None:
    args = build_parser().parse_args(["probe-gpu"])
    paths = ProjectPaths(tmp_path)
    output, previous = cli._probe_paths(args, paths)
    assert previous is None
    assert output.parent == tmp_path / PROBE_OUTPUT_ROOT
    assert DATED.fullmatch(output.name)
    assert cli._probe_paths(args, paths)[0] == output  # same on a second call


def test_relative_paths_are_taken_from_the_project_root(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["probe-gpu", "--output", "outputs/mine", "--previous", "outputs/first/probe.json"]
    )
    output, previous = cli._probe_paths(args, ProjectPaths(tmp_path))
    assert output == tmp_path / "outputs" / "mine"
    assert previous == tmp_path / "outputs" / "first" / "probe.json"


@pytest.mark.parametrize("entry", ["probe.json", "probe.md", "scores", LOG_NAME])
def test_a_probe_run_never_reuses_a_directory_holding_records(tmp_path: Path, entry: str) -> None:
    if entry == "scores":
        (tmp_path / entry).mkdir()
    else:
        (tmp_path / entry).write_text("x", encoding="utf-8")
    with pytest.raises(SpecificationError, match="already holds a probe run"):
        check_output(tmp_path, None)


def test_a_second_probe_run_needs_a_directory_apart_from_the_first(tmp_path: Path) -> None:
    first = tmp_path / "first"
    first.mkdir()
    previous = first / "probe.json"
    with pytest.raises(SpecificationError, match="separate from the first run's"):
        check_output(first, previous)
    with pytest.raises(SpecificationError, match="separate from the first run's"):
        check_output(first / "inside", previous)
    with pytest.raises(SpecificationError, match="separate from the first run's"):
        check_output(tmp_path, previous)
    check_output(tmp_path / "second", previous)


def test_the_probe_command_refuses_a_log_file_and_a_used_directory(tmp_path: Path) -> None:
    output = tmp_path / "run"
    root = str(REPO_PATHS.root)
    refused = build_parser().parse_args(
        ["--log-file", "x.log", "probe-gpu", "--root", root, "--output", str(output)]
    )
    with pytest.raises(SpecificationError, match="omit --log-file"):
        cli._log_file(refused)
    args = build_parser().parse_args(["probe-gpu", "--root", root, "--output", str(output)])
    assert cli._log_file(args) == output / LOG_NAME
    output.mkdir()
    (output / "probe.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SpecificationError, match="already holds a probe run"):
        cli._log_file(args)


def test_the_acceptance_run_needs_a_new_or_empty_directory(tmp_path: Path) -> None:
    target = tmp_path / "acceptance"
    check_acceptance_output(target)  # absent: free
    target.mkdir()
    check_acceptance_output(target)  # empty: free
    (target / ACCEPTANCE_LOG_NAME).write_text("log\n", encoding="utf-8")
    with pytest.raises(SpecificationError):
        check_acceptance_output(target)
    check_acceptance_output(target, log_opened=True)  # only this run's own log
    (target / "acceptance.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SpecificationError):
        check_acceptance_output(target, log_opened=True)
