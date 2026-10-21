"""The CatBoost tree-objective acceptance (probe specification, Section 8) end to end on the CPU
with small synthetic inputs, plus its agreement with the training declaration and its command. A
CPU run exercises the code paths; it is never acceptance evidence. No research data."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from review_capacity.cli import _acceptance_output, _log_file, build_parser
from review_capacity.config import load_model_settings
from review_capacity.errors import FileAccessError, ModelError, SpecificationError
from review_capacity.models.base import ModelData
from review_capacity.models.catboost_model import CatBoostModel
from review_capacity.models.lightgbm_model import LightGBMModel
from review_capacity.models.registry import configuration
from review_capacity.models.v1_objective import V1Objective
from review_capacity.probe import catboost_objective
from review_capacity.probe.catboost_objective import (
    CHECKS,
    classify_warnings,
    daily_budget,
    evidence_blockers,
    hyperparameters,
    run_acceptance,
)
from review_capacity.probe.records import CheckBook, check_acceptance_output
from review_capacity.probe.runner import WarningCapture
from review_capacity.probe.synthetic import ProbeSizes
from review_capacity.training.declaration import load_declaration
from support.synthetic import FEATURES, REPO_PATHS

SIZES = ProbeSizes(
    train=600, valid=300, score=100, neural_train=16, neural_valid=16, neural_score=16
)
DEPTH, ITERATIONS = 3, 15


@pytest.fixture(scope="module")
def acceptance(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], Path]:
    output = tmp_path_factory.mktemp("acceptance") / "run"
    record = run_acceptance(
        REPO_PATHS, output, device="cpu", sizes=SIZES, depth=DEPTH, iterations=ITERATIONS
    )
    return record, output


def test_every_check_runs_and_the_deterministic_ones_pass(
    acceptance: tuple[dict[str, Any], Path],
) -> None:
    record, _ = acceptance
    assert list(record["checks"]) == list(CHECKS)
    for check in ("G1", "G2", "G3", "G4", "G5", "G7"):
        assert record["checks"][check]["status"] == "pass", record["checks"][check]
    assert record["checks"]["G6"]["status"] in ("pass", "review")
    assert record["overall"] in ("pass", "review")
    assert record["run"]["stage"] == "finished"
    assert record["run"]["error"] is None


def test_a_cpu_run_is_not_acceptance_evidence(acceptance: tuple[dict[str, Any], Path]) -> None:
    record, output = acceptance
    assert record["run"]["acceptance_evidence"] is False
    blockers = record["run"]["evidence_blockers"]
    assert any("device cpu" in blocker for blocker in blockers)
    summary = (output / "acceptance.md").read_text(encoding="utf-8")
    assert "This run is not acceptance evidence: " in summary
    assert "device cpu, not gpu" in summary


def test_the_fit_trains_every_round_and_records_its_history(
    acceptance: tuple[dict[str, Any], Path],
) -> None:
    record, _ = acceptance
    history = record["history"]
    assert history["stopped_at"] == ITERATIONS == len(history["checkpoints"])
    assert history["stopping_reason"] == "maximum"
    assert history["patience"] == history["maximum"] == ITERATIONS
    assert history["objective"]["daily_k"] == record["run"]["daily_k"] >= 1
    assert record["fit"]["kept"] == history["best_checkpoint"] <= ITERATIONS
    assert record["fit"]["parameters"]["resolved"]["depth"] == DEPTH
    rows = history["checkpoints"]
    assert [row["checkpoint"] for row in rows] == list(range(1, ITERATIONS + 1))
    assert rows[history["best_checkpoint"] - 1]["objective"] == history["best_objective"]


def test_the_records_are_written_and_the_summary_is_generated(
    acceptance: tuple[dict[str, Any], Path],
) -> None:
    record, output = acceptance
    assert json.loads((output / "acceptance.json").read_text(encoding="utf-8")) == record
    summary = (output / "acceptance.md").read_text(encoding="utf-8")
    assert "# CatBoost tree-objective acceptance: run summary" in summary
    assert "## Symbols and notation" in summary
    assert f"**Overall result: {record['overall']}.**" in summary
    specification = REPO_PATHS.root / "docs" / "gpu-acceptance-probe.md"
    assert f"({Path(os.path.relpath(specification, output)).as_posix()})" in summary


def test_a_run_never_reuses_a_directory(acceptance: tuple[dict[str, Any], Path]) -> None:
    _, output = acceptance
    with pytest.raises(SpecificationError, match="is not empty"):
        run_acceptance(REPO_PATHS, output, device="cpu", sizes=SIZES, iterations=ITERATIONS)


def test_a_failed_fit_leaves_a_failed_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def broken(*_args: Any, **_kwargs: Any) -> None:
        raise ModelError("GPU learner rejected the metric")

    monkeypatch.setattr(CatBoostModel, "fit", broken)
    output = tmp_path / "run"
    with pytest.raises(ModelError, match="rejected the metric"):
        run_acceptance(
            REPO_PATHS, output, device="cpu", sizes=SIZES, depth=DEPTH, iterations=ITERATIONS
        )
    record = json.loads((output / "acceptance.json").read_text(encoding="utf-8"))
    assert record["overall"] == "failed"
    assert record["run"]["stage"] == "failed"
    assert "rejected the metric" in record["run"]["error"]
    assert record["history"] is None
    assert (output / "acceptance.md").exists()


def test_the_acceptance_values_are_those_of_the_specification() -> None:
    values = hyperparameters(catboost_objective.DEPTH, catboost_objective.ITERATIONS)
    assert (values["depth"], values["iterations"], values["od_wait"]) == (10, 2000, 2000)
    assert (values["eval_metric"], values["boost_from_average"], values["metric_period"]) == (
        "v1_objective",
        False,
        1,
    )


def test_the_acceptance_values_agree_with_the_training_declaration() -> None:
    declaration = load_declaration(REPO_PATHS)[0]
    space = declaration.space("catboost")
    reference = {parameter.name: parameter.reference for parameter in space.parameters}
    values = hyperparameters(catboost_objective.DEPTH, declaration.tree_stopping.maximum)
    assert values["iterations"] == declaration.tree_stopping.maximum
    assert values["od_wait"] == declaration.tree_stopping.maximum  # the detector cannot stop it
    for name in ("learning_rate", "l2_leaf_reg", "bagging_temperature"):
        assert values[name] == reference[name]
    assert values["leaf_estimation_iterations"] == reference["leaf_estimation_iterations"]
    for name, value in space.fixed.items():
        if name not in ("od_wait", "iterations"):
            assert values[name] == value, name


def _small_v1() -> tuple[ModelData, V1Objective]:
    index = np.arange(40, dtype=np.int64)
    labels = (index % 4 == 0).astype(np.int8)
    frame = pd.DataFrame({name: [0.0] * 40 for name in FEATURES.model_inputs})
    valid = ModelData(
        frame,
        labels,
        np.ones(40),
        times=(index // 10 + 1) * 86_400,
        ids=index + 1,
    )
    return valid, V1Objective(valid, 2)


def test_the_declared_tree_settings_are_accepted_by_both_adapters() -> None:
    """Each adapter takes the declared fixed settings and the reference candidate unchanged."""
    declaration = load_declaration(REPO_PATHS)[0]
    settings = load_model_settings(REPO_PATHS)
    _, objective = _small_v1()
    for library, name, cls in (
        ("lightgbm", "LGB-u", LightGBMModel),
        ("catboost", "Cat-u", CatBoostModel),
    ):
        space = declaration.space(library)
        candidate = {**space.fixed, **{p.name: p.reference for p in space.parameters}}
        model: Any = cls(
            configuration(name), 42, settings=settings, features=FEATURES, device="cpu"
        )
        params = model._parameters_for_fit(candidate, None, objective)
        stopping = declaration.tree_stopping
        patience = params.get("early_stopping_round", params.get("od_wait"))
        maximum = params.get("num_iterations", params.get("iterations"))
        assert (patience, maximum) == (stopping.patience, stopping.maximum)


def test_the_daily_budget_follows_the_rule_of_dr_015() -> None:
    times = np.repeat(np.arange(106, 127, dtype=np.int64), 3) * 86_400 + 5
    assert daily_budget(times, 65_544) == (62, 21)  # round(0.02 × 65,544 / 21)
    assert daily_budget(times[:3], 3) == (1, 1)  # at least 1


def test_warnings_are_classified() -> None:
    capture = WarningCapture()
    capture.records = [
        {"logger": "py.warnings", "phase": "fit", "text": catboost_objective.NUMBA_NOTICE},
        {"logger": "review_capacity.models.catboost_model", "phase": "fit", "text": "x ignored"},
        {"logger": "review_capacity.models.catboost_model", "phase": "fit", "text": "something"},
    ]
    book = CheckBook(CHECKS)
    classify_warnings(capture, book)
    assert [item["outcome"] for item in book.items["G6"]] == [True, False, "review"]
    assert book.status("G6") == "fail"
    empty = CheckBook(CHECKS)
    classify_warnings(WarningCapture(), empty)
    assert empty.status("G6") == "pass"


def test_the_overall_result_is_incomplete_until_every_check_ran() -> None:
    book = CheckBook(CHECKS)
    book.add("G1", "x", True)
    assert book.overall(compared=True) == "incomplete"


def test_command_is_registered_and_logs_in_its_output_directory() -> None:
    args = build_parser().parse_args(["probe-catboost-objective", "--output", "evidence/x"])
    assert args.output == Path("evidence/x")
    assert _log_file(args) == REPO_PATHS.root / "evidence" / "x" / "acceptance.log"
    args = build_parser().parse_args(["probe-catboost-objective"])
    log_file = _log_file(args)
    assert log_file is not None
    assert log_file.parent.parent == REPO_PATHS.root / "outputs" / "catboost_objective_acceptance"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{6}Z", log_file.parent.name)
    assert _acceptance_output(args, REPO_PATHS) == log_file.parent  # one directory per run
    args = build_parser().parse_args(["--log-file", "a.log", "probe-catboost-objective"])
    with pytest.raises(SpecificationError, match="omit --log-file"):
        _log_file(args)


def test_the_command_refuses_a_directory_with_an_earlier_run(tmp_path: Path) -> None:
    (tmp_path / "acceptance.log").write_text("earlier", encoding="utf-8")
    args = build_parser().parse_args(["probe-catboost-objective", "--output", str(tmp_path)])
    with pytest.raises(SpecificationError, match="is not empty"):
        _log_file(args)


PACKAGE = (REPO_PATHS.root / "src" / "review_capacity").as_posix()
CLEAN = {
    "git_uncommitted_changes": False,
    "git_commit": "0" * 40,
    "package": {"location": PACKAGE},
}
ROOT = REPO_PATHS.root


def test_only_the_specified_gpu_run_from_a_clean_tree_is_acceptance_evidence() -> None:
    full = ProbeSizes()
    assert evidence_blockers("gpu", 10, 2000, full, CLEAN, ROOT) == []
    dirty = {**CLEAN, "git_uncommitted_changes": True}
    assert evidence_blockers("gpu", 10, 2000, full, dirty, ROOT)
    unknown = {**CLEAN, "git_uncommitted_changes": None}
    assert evidence_blockers("gpu", 10, 2000, full, unknown, ROOT)
    assert evidence_blockers("gpu", 10, 2000, full, None, ROOT)
    assert evidence_blockers("cpu", 10, 2000, full, CLEAN, ROOT) == ["device cpu, not gpu"]
    assert evidence_blockers("gpu", 6, 2000, full, CLEAN, ROOT) == ["depth 6, not 10"]
    assert evidence_blockers("gpu", 10, 300, full, CLEAN, ROOT) == ["300 rounds, not 2000"]
    small = evidence_blockers("gpu", 10, 2000, SIZES, CLEAN, ROOT)
    assert len(small) == 1
    assert "Train and" in small[0]


def test_the_running_package_must_be_the_inspected_checkout(tmp_path: Path) -> None:
    """An editable install from one checkout, run with ``--root`` of another that is clean, is
    not evidence; neither is a record without its commit."""
    other = (tmp_path / "other" / "review_capacity").as_posix()
    elsewhere = {**CLEAN, "package": {"location": other}}
    blockers = evidence_blockers("gpu", 10, 2000, ProbeSizes(), elsewhere, ROOT)
    assert blockers == ["the running package is not the inspected checkout's"]
    no_package = {k: v for k, v in CLEAN.items() if k != "package"}
    assert evidence_blockers("gpu", 10, 2000, ProbeSizes(), no_package, ROOT)
    no_commit = {**CLEAN, "git_commit": None}
    assert evidence_blockers("gpu", 10, 2000, ProbeSizes(), no_commit, ROOT) == [
        "the commit is not recorded"
    ]


def test_a_reduced_run_from_a_clean_tree_records_its_reasons(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        catboost_objective,
        "code_provenance",
        lambda _root: {**CLEAN},
    )
    record = run_acceptance(
        REPO_PATHS, tmp_path / "run", device="cpu", sizes=SIZES, depth=DEPTH, iterations=ITERATIONS
    )
    blockers = record["run"]["evidence_blockers"]
    assert record["run"]["acceptance_evidence"] is False
    assert any("Train and" in blocker for blocker in blockers)
    assert not any("working tree" in blocker for blocker in blockers)


def test_the_output_directory_must_be_new_or_empty(tmp_path: Path) -> None:
    check_acceptance_output(tmp_path / "does-not-exist")
    check_acceptance_output(tmp_path)  # exists, empty
    for name in ("probe.json", "probe.log", "search_plan.md", "acceptance.json", "acceptance.log"):
        occupied = tmp_path / f"with-{name}"
        occupied.mkdir()
        (occupied / name).write_text("earlier", encoding="utf-8")
        with pytest.raises(SpecificationError, match="is not empty"):
            check_acceptance_output(occupied)
    opened = tmp_path / "opened"
    opened.mkdir()
    (opened / "acceptance.log").write_text("this run's log", encoding="utf-8")
    check_acceptance_output(opened, log_opened=True)  # only the run's own log is allowed
    (opened / "probe.log").write_text("another run", encoding="utf-8")
    with pytest.raises(SpecificationError, match=r"probe\.log"):
        check_acceptance_output(opened, log_opened=True)


def test_an_output_path_that_is_a_file_is_a_file_access_error(tmp_path: Path) -> None:
    target = tmp_path / "file"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(FileAccessError, match="cannot be inspected"):
        check_acceptance_output(target)


def test_the_summary_presents_every_recorded_field_and_every_round(
    acceptance: tuple[dict[str, Any], Path],
) -> None:
    record, output = acceptance
    summary = (output / "acceptance.md").read_text(encoding="utf-8")
    rows = [line for line in summary.splitlines() if re.match(r"\| \d+ \| \d+ \| ", line)]
    assert len(rows) == ITERATIONS  # one row per round, none dropped
    for checkpoint in record["history"]["checkpoints"]:
        assert repr(checkpoint["objective"]) in summary
        assert repr(checkpoint["seconds"]) in summary
    assert f"| Record schema version | {record['schema_version']} |" in summary
    sections = [
        {k: v for k, v in record["run"].items() if k != "hyperparameters"},
        record["run"]["hyperparameters"],
        record["fit"],
        {"specification": record["specification"], "settings": record["settings"]},
        {"code": record["code"], "environment": record["environment"]},
        record["synthetic"],
        {"gpu_memory": record["gpu_memory"], "host_peak_bytes": record["host_peak_bytes"]},
        {k: v for k, v in record["history"].items() if k != "checkpoints"},
    ]
    for section in sections:  # every field, bound to its own path and value
        for path, text in catboost_objective._flatten(section):
            assert f"| `{path}` | {text} |" in summary, path
    for check in record["checks"].values():
        for item in check["items"]:
            assert item["subject"] in summary


def test_floats_are_shown_with_every_digit() -> None:
    value = 0.11582901234567891
    assert float(catboost_objective._scalar(value)) == value
    assert catboost_objective._scalar(1e-07) == "1e-07"
    flat = dict(catboost_objective._flatten({"a": {"b": [0.1, 2], "c": [{"d": 1.5}], "e": {}}}))
    assert flat == {"a.b": "0.1, 2", "a.c[0].d": "1.5", "a.e": "{}"}
