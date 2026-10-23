"""The search plan and the ``plan-search`` command: records, Markdown, new-directory rule and exit
codes (the project's real declaration; no data read, no model fitted)."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from review_capacity.cli import build_parser, main
from review_capacity.development.records import MODELS_SEARCHED, read_record
from review_capacity.development.summary import cell
from review_capacity.errors import FileAccessError, SpecificationError
from review_capacity.hashing import file_sha256
from review_capacity.models.run_record import load_settings_with_snapshot
from review_capacity.training import plan
from review_capacity.training.declaration import TrainingDeclaration, load_declaration
from support.development import PARTITIONS
from support.synthetic import REPO_PATHS

STARTED = "2026-10-21T120000Z"

pytestmark = pytest.mark.usefixtures("restore_logging")


def build(started: str = STARTED) -> tuple[TrainingDeclaration, dict[str, Any]]:
    settings, settings_file = load_settings_with_snapshot(REPO_PATHS)
    declaration, declaration_file = load_declaration(REPO_PATHS)
    run = plan.identity(
        REPO_PATHS.root,
        started,
        declaration,
        declaration_file,
        settings_file,
        settings.main_seed,
        REPO_PATHS.manifest,
    )
    return declaration, run


def test_the_identity_names_the_declaration_settings_manifest_and_seeds() -> None:
    declaration, run = build()
    _, declaration_file = load_declaration(REPO_PATHS)
    assert run["run_id"] == STARTED
    assert run["run_type"] == "search_plan"
    assert run["declaration_sha256"] == declaration_file.sha256
    assert run["settings_sha256"] == file_sha256(REPO_PATHS.models_spec)
    assert run["dataset_manifest_sha256"] == file_sha256(REPO_PATHS.manifest)
    assert run["seeds"] == {
        "main": 42,
        "additional": [43, 44],
        "sampler": declaration.sampler_seed,
        "preprocessing": 0,
    }


def test_run_ids_are_utc_start_times() -> None:
    assert len(plan.run_id()) == len(STARTED)
    assert plan.run_id().endswith("Z")
    assert plan.OUTPUT_ROOT.parts == ("outputs", "search_plan")


def test_the_plan_is_written_as_records_and_markdown(tmp_path: Path) -> None:
    declaration, run = build()
    output = tmp_path / STARTED
    cost = plan.write_plan(output, STARTED, run, declaration, PARTITIONS)
    assert sorted(p.name for p in output.iterdir()) == [
        "candidates",
        "cost_prediction.json",
        "search_plan.md",
    ]
    assert sorted(p.stem for p in (output / "candidates").iterdir()) == sorted(MODELS_SEARCHED)
    read_record(output / "cost_prediction.json", PARTITIONS, "cost_prediction")
    for model in MODELS_SEARCHED:
        record = read_record(output / "candidates" / f"{model}.json", PARTITIONS, "candidates")
        assert len(record["trials"]) == declaration.trials
        link = cost["candidates"][model]
        assert link["path"] == f"candidates/{model}.json"
        assert link["sha256"] == hashlib.sha256((output / link["path"]).read_bytes()).hexdigest()


def test_the_two_configurations_of_a_model_share_its_list(tmp_path: Path) -> None:
    declaration, run = build()
    plan.write_plan(tmp_path / "p", STARTED, run, declaration, PARTITIONS)
    configurations = {
        model: json.loads((tmp_path / "p" / "candidates" / f"{model}.json").read_text("utf-8"))[
            "configurations"
        ]
        for model in MODELS_SEARCHED
    }
    assert configurations == {
        "lightgbm": ["LGB-u", "LGB-b"],
        "catboost": ["Cat-u", "Cat-b"],
        "resnet": ["RES-u", "RES-b"],
        "ft_transformer": ["FTT-u", "FTT-b"],
        "logistic_regression": ["LR"],
    }


def test_the_candidate_lists_do_not_depend_on_the_run(tmp_path: Path) -> None:
    declaration, first_run = build("2026-10-21T120000Z")
    _, second_run = build("2026-10-22T080000Z")
    plan.write_plan(tmp_path / "a", "2026-10-21T120000Z", first_run, declaration, PARTITIONS)
    plan.write_plan(tmp_path / "b", "2026-10-22T080000Z", second_run, declaration, PARTITIONS)
    for model in MODELS_SEARCHED:
        first = json.loads((tmp_path / "a" / "candidates" / f"{model}.json").read_text("utf-8"))
        second = json.loads((tmp_path / "b" / "candidates" / f"{model}.json").read_text("utf-8"))
        assert first["trials"] == second["trials"]
        assert first["identity"]["run_id"] != second["identity"]["run_id"]


def test_the_cost_is_a_pending_decision(tmp_path: Path) -> None:
    declaration, run = build()
    cost = plan.write_plan(tmp_path / "p", STARTED, run, declaration, PARTITIONS)
    assert cost["decision"] == "pending_approval"
    estimated = {e["model"] for e in cost["estimates"] if e["estimated"]}
    assert estimated == {"resnet", "ft_transformer"}
    assert cost["totals"]["search_hours_high"] > cost["totals"]["search_hours_low"] > 0


def test_the_markdown_shows_every_candidate_exactly(tmp_path: Path) -> None:
    declaration, run = build()
    output = tmp_path / "p"
    plan.write_plan(output, STARTED, run, declaration, PARTITIONS)
    text = (output / "search_plan.md").read_text(encoding="utf-8")
    assert "not edited by hand" in text
    assert "decided explicitly before any run" in text
    for model in MODELS_SEARCHED:
        assert f"## {model}" in text
        record = json.loads((output / "candidates" / f"{model}.json").read_text("utf-8"))
        for name in record["trials"][0]["parameters"]:
            assert f"`parameters.{name}`" in text
        for trial in record["trials"]:
            for value in trial["parameters"].values():
                assert cell(value) in text
    assert "## Predicted cost" in text
    assert "**Total**" in text


def test_the_markdown_presents_every_field_of_every_record(tmp_path: Path) -> None:
    declaration, run = build()
    output = tmp_path / "p"
    cost = plan.write_plan(output, STARTED, run, declaration, PARTITIONS)
    text = (output / "search_plan.md").read_text(encoding="utf-8")
    assert cell("search_plan") in text  # identity.run_type, shown once for the plan
    assert text.count("`identity.run_type`") == 1
    assert cell("pending_approval") in text  # cost_prediction.decision
    for kind in ("candidates", "cost_prediction"):
        assert cell(kind) in text
    for field in ("`record`", "`schema_version`", "`sampler.numpy`", "`stream.spawn_key`"):
        assert field in text
    for estimate in cost["estimates"]:
        assert cell(estimate["basis"]) in text  # also for the models that are not estimated
    for link in cost["candidates"].values():
        assert cell(link) in text


def test_candidate_records_name_the_sampling_software(tmp_path: Path) -> None:
    declaration, run = build()
    plan.write_plan(tmp_path / "p", STARTED, run, declaration, PARTITIONS)
    record = read_record(tmp_path / "p" / "candidates" / "lightgbm.json", PARTITIONS)
    assert record["sampler"] == {
        "generator": "numpy.random.default_rng",
        "bit_generator": "PCG64",
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def test_the_cost_is_described_as_an_estimate_with_its_limits(tmp_path: Path) -> None:
    declaration, run = build()
    cost = plan.write_plan(tmp_path / "p", STARTED, run, declaration, PARTITIONS)
    basis = {e["model"]: e["basis"] for e in cost["estimates"]}
    assert "extrapolated" in basis["ft_transformer"]
    assert "not modelled" in basis["resnet"]
    assert "bound" not in basis["resnet"]
    text = (tmp_path / "p" / "search_plan.md").read_text(encoding="utf-8")
    assert "not a bound" in text
    assert "upper bound" not in text


def test_a_plan_never_reuses_a_directory(tmp_path: Path) -> None:
    declaration, run = build()
    output = tmp_path / "p"
    plan.write_plan(output, STARTED, run, declaration, PARTITIONS)
    with pytest.raises(SpecificationError, match="already exists"):
        plan.write_plan(output, STARTED, run, declaration, PARTITIONS)
    with pytest.raises(SpecificationError, match="already exists"):
        plan.check_output(output)


def test_an_unreadable_output_directory_is_a_file_access_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "p").mkdir()

    def denied(_self: Path) -> Any:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "iterdir", denied)
    with pytest.raises(FileAccessError, match="cannot be inspected"):
        plan.check_output(tmp_path / "p")


def test_a_directory_with_only_this_runs_log_is_free(tmp_path: Path) -> None:
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / plan.LOG_NAME).write_text("", encoding="utf-8")
    plan.check_output(tmp_path / "p")
    plan.check_output(tmp_path / "missing")
    (tmp_path / "p" / "other.txt").write_text("", encoding="utf-8")
    with pytest.raises(SpecificationError, match="already exists"):
        plan.check_output(tmp_path / "p")
    (tmp_path / "file").write_text("", encoding="utf-8")
    with pytest.raises(SpecificationError, match="already exists"):
        plan.check_output(tmp_path / "file")


# --- the command ---------------------------------------------------------------------------------


def test_plan_search_options() -> None:
    args = build_parser().parse_args(["plan-search", "--root", "/tmp/project"])
    assert args.command == "plan-search"
    assert args.output is None


def test_plan_search_writes_the_plan_and_its_log(tmp_path: Path) -> None:
    output = tmp_path / "plan"
    assert main(["plan-search", "--root", str(REPO_PATHS.root), "--output", str(output)]) == 0
    assert (output / "cost_prediction.json").is_file()
    assert (output / "search_plan.md").is_file()
    log = (output / plan.LOG_NAME).read_text(encoding="utf-8")
    assert "Search plan written" in log
    assert "Decision pending" in log


def test_plan_search_refuses_an_existing_plan_and_a_separate_log_file(tmp_path: Path) -> None:
    output = tmp_path / "plan"
    arguments = ["plan-search", "--root", str(REPO_PATHS.root), "--output", str(output)]
    assert main(arguments) == 0
    assert main(arguments) == 2  # a new directory only
    assert main(["--log-file", str(tmp_path / "x.log"), *arguments[:-1], str(tmp_path / "q")]) == 2
    assert not (tmp_path / "q").exists()
