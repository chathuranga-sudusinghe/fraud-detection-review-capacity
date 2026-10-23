"""The ``run-development`` command from the command line to the closed run (Issue #124).

``plan-search`` writes the plan and ``run-development`` approves it, verifies it against the
declaration, prepares the run, searches, selects, scores, evaluates, times and closes it, through
``main`` and the production preparation. Only what needs the project's data or hardware is
replaced: the verified processed partitions (synthetic ones stand in), the LightGBM build
verification, the Git state and the GPU telemetry. The verification of the processed files
(hashes, time order, sealed Test, provenance records) and the hashes the run record states for
them are tested on processed synthetic files in ``test_inputs.py`` and ``test_run_start.py``.
The project root is a temporary directory that holds copies of the project's settings and a
small training declaration (a few trials, short early stopping, small timing passes, no settling
interval).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from review_capacity.cli import main
from review_capacity.development import run_start
from review_capacity.development.records import DevelopmentPartitions
from review_capacity.development.run_output import load_run
from review_capacity.hashing import file_sha256
from review_capacity.models.device import LightGBMIdentity
from review_capacity.paths import ProjectPaths
from review_capacity.training.timing import session_conditions
from support.search import (
    DAILY_K,
    MAXIMUM,
    PATIENCE,
    TRIALS,
    V2_ROWS,
    partition,
    search_inputs,
    synthetic_partitions,
)
from support.synthetic import FEATURES, REPO_PATHS

pytestmark = pytest.mark.usefixtures("restore_logging", "restore_lightgbm_logger")

GIT_COMMIT = "c" * 40
LIGHTGBM_BUILD = LightGBMIdentity("4.7.0", Path("lib_lightgbm.so"), "d" * 64)


def small_declaration() -> str:
    """The project's training declaration with a few trials, short early stopping, small timing
    passes and no settling interval."""
    content = yaml.safe_load(REPO_PATHS.training_spec.read_text(encoding="utf-8"))
    content["search"]["trials"] = TRIALS
    content["v1_objective"]["daily_k"] = DAILY_K
    content["early_stopping"]["tree"] = {"patience": PATIENCE, "maximum": MAXIMUM}
    models = content["models"]
    models["lightgbm"]["fixed"].update(num_iterations=MAXIMUM, early_stopping_round=PATIENCE)
    models["catboost"]["fixed"].update(iterations=MAXIMUM, od_wait=PATIENCE)
    timing = content["timing"]
    timing["scoring"]["passes"] = 2
    timing["latency"].update(sample_rows=20, warmup_calls=3, rounds=2)
    timing["session"]["settling_seconds"] = 0
    return yaml.safe_dump(content, sort_keys=False)


def synthetic_parts() -> dict[str, Any]:
    searched = search_inputs()
    return {
        "train": searched.train,
        "v1": searched.v1,
        "v2": partition("v2", V2_ROWS, 3, 3_000_000),
    }


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectPaths:
    root = tmp_path / "project"
    paths = ProjectPaths(root)
    paths.config_dir.mkdir(parents=True)
    paths.data_dir.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n", encoding="utf-8")
    paths.features_spec.write_text("version: 1\n", encoding="utf-8")  # a root marker only
    paths.manifest.write_text(json.dumps({"name": "synthetic"}), encoding="utf-8")
    shutil.copy(REPO_PATHS.models_spec, paths.models_spec)
    paths.training_spec.write_text(small_declaration(), encoding="utf-8")

    parts = synthetic_parts()
    partitions = synthetic_partitions()
    monkeypatch.setattr(
        DevelopmentPartitions, "from_spec", classmethod(lambda _cls, *_a, **_k: partitions)
    )
    monkeypatch.setattr("review_capacity.cli.load_dataset_spec", lambda _paths: SimpleNamespace())
    runner = "review_capacity.development.runner"
    monkeypatch.setattr(f"{runner}.load_dataset_spec", lambda _paths: SimpleNamespace())
    monkeypatch.setattr(f"{runner}.load_feature_spec", lambda _path: FEATURES)
    monkeypatch.setattr(f"{runner}.read_partitions", lambda _p, _f, _d, _stems: dict(parts))
    monkeypatch.setattr(f"{runner}.verify_lightgbm_build", lambda _settings: LIGHTGBM_BUILD)
    monkeypatch.setattr(
        f"{runner}.session_conditions",
        lambda: session_conditions(telemetry=lambda: {"available": False}),
    )
    monkeypatch.setattr(
        run_start,
        "processed_files",
        lambda _paths, stems: {
            stem: {
                "path": f"data/processed/{stem}.parquet",
                "sha256": parts[stem].provenance.sha256 if stem in parts else "a" * 64,
            }
            for stem in stems
        },
    )

    def git(_root: Path, *arguments: str) -> str:
        return "" if arguments[0] == "status" else GIT_COMMIT

    monkeypatch.setattr("review_capacity.provenance._git", git)
    return paths


def make_plan(paths: ProjectPaths) -> tuple[Path, str]:
    """Run ``plan-search``; return the plan directory and the SHA-256 of its cost prediction."""
    assert main(["plan-search", "--root", str(paths.root)]) == 0
    (plan,) = (paths.root / "outputs" / "search_plan").iterdir()
    return plan, file_sha256(plan / "cost_prediction.json")


def run_command(paths: ProjectPaths, plan: Path, approved: str) -> int:
    return main(
        [
            "run-development",
            "--root",
            str(paths.root),
            "--plan",
            str(plan),
            "--approved-cost",
            approved,
            "--configurations",
            "LGB-u",
        ]
    )


def test_the_approved_plan_runs_to_a_closed_run_whose_records_name_what_was_used(
    project: ProjectPaths,
) -> None:
    plan, approved = make_plan(project)
    assert run_command(project, plan, approved) == 0

    (run_directory,) = (project.root / "outputs" / "development").iterdir()
    assert (run_directory / "run.log").is_file()
    assert (run_directory / "summary.md").is_file()
    loaded = load_run(run_directory, synthetic_partitions())  # validates every record
    assert loaded.end is not None  # the run closed
    assert loaded.end["outcome"] == "completed"
    listed = {Path(entry["path"]).stem for entry in loaded.end["records"]}
    assert listed == {
        name.removesuffix(".json") for name, _ in loaded.records if name != "run_end.json"
    }
    (run,) = loaded.of_kind("run")
    cost = json.loads((plan / "cost_prediction.json").read_text(encoding="utf-8"))

    assert run["plan"] == {
        "path": f"outputs/search_plan/{plan.name}/cost_prediction.json",
        "sha256": approved,
    }
    assert run["declaration"]["sha256"] == file_sha256(project.training_spec)
    assert run["settings"]["sha256"] == file_sha256(project.models_spec)
    assert run["dataset"]["manifest"]["sha256"] == file_sha256(project.manifest)
    assert run["configurations"] == ["LGB-u"]
    identity = run["identity"]
    assert identity["declaration_sha256"] == run["declaration"]["sha256"]
    assert identity["settings_sha256"] == run["settings"]["sha256"]
    assert identity["code_commit"] == GIT_COMMIT
    assert identity["uncommitted_changes"] is False
    assert run["dataset"]["processed"]["v2"]["sha256"] == synthetic_parts()["v2"].provenance.sha256

    adopted = run_directory / "candidates" / "lightgbm.json"
    assert file_sha256(adopted) == cost["candidates"]["lightgbm"]["sha256"]
    (configuration,) = loaded.of_kind("configuration")
    assert configuration["configuration"] == "LGB-u"
    trials = sorted(loaded.of_kind("trial"), key=lambda record: record["trial"])
    assert len(trials) == TRIALS
    approved_trials = {
        t["trial"]: t["parameters"]
        for t in json.loads(adopted.read_text(encoding="utf-8"))["trials"]
    }
    assert [t["hyperparameters"] for t in trials] == [approved_trials[t["trial"]] for t in trials]


def test_a_cost_prediction_that_was_not_approved_starts_no_run(project: ProjectPaths) -> None:
    """The command stops before any record: its directory holds only the log that states why."""
    plan, approved = make_plan(project)
    assert approved != "0" * 64
    assert run_command(project, plan, "0" * 64) == 2
    (directory,) = (project.root / "outputs" / "development").iterdir()
    assert [path.name for path in directory.iterdir()] == ["run.log"]
    assert "is not the approved cost prediction" in (directory / "run.log").read_text(
        encoding="utf-8"
    )
