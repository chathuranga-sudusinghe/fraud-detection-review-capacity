"""The GPU acceptance probe end to end on the CPU, with small synthetic sizes and small networks: a
first run in this process and a second run in a fresh process, compared with the first (C11),
plus the failure paths of the acceptance gates. A CPU run exercises the code paths; it is never
acceptance evidence. No research data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from review_capacity.cli import _log_file, _probe_gpu, build_parser, main
from review_capacity.config import load_feature_spec
from review_capacity.data.catalog import read_catalog
from review_capacity.errors import EnvironmentMismatchError, ModelError, SpecificationError
from review_capacity.models import neural_training
from review_capacity.models.base import ModelData
from review_capacity.models.neural_training import NeuralModel
from review_capacity.models.registry import Weighting, configuration
from review_capacity.models.resnet_model import ResNetModel
from review_capacity.models.run_record import load_settings_with_snapshot
from review_capacity.probe import runner
from review_capacity.probe.records import CHECKS
from review_capacity.probe.synthetic import category_levels, neural_data, neural_layout, tree_data
from support.neural import preserved_torch_state
from support.probe_small import SIZES, SMALL, run_small
from support.synthetic import REPO_PATHS

type Runs = tuple[dict[str, Any], dict[str, Any], Path]


@pytest.fixture(autouse=True)
def restore_torch_state() -> Iterator[None]:
    with preserved_torch_state():
        yield


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Runs:
    """A first CPU run in this process; a second in a fresh process, given the first's record."""
    root = tmp_path_factory.mktemp("probe")
    with preserved_torch_state():
        first = run_small(root / "run-1")
    environment = {**os.environ, "PYTHONPATH": str(REPO_PATHS.root / "tests")}
    command = [sys.executable, "-m", "support.probe_small", str(root / "run-2")]
    command.append(str(root / "run-1" / "probe.json"))
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        cwd=REPO_PATHS.root,
        check=False,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    second = json.loads((root / "run-2" / "probe.json").read_text(encoding="utf-8"))
    return first, second, root


def _items(record: dict[str, Any], check: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = record["checks"][check]["items"]
    return items


def _probe(tmp_path: Path) -> runner.Probe:
    return runner.Probe(
        REPO_PATHS,
        tmp_path,
        "cpu",
        SIZES,
        load_settings_with_snapshot(REPO_PATHS)[0],
        load_feature_spec(REPO_PATHS.features_spec),
        runner.WarningCapture(),
    )


def test_every_check_is_run_and_recorded(runs: Runs) -> None:
    first, _, root = runs
    assert set(first["checks"]) == set(CHECKS)
    assert all(first["checks"][check]["status"] != "not run" for check in CHECKS)
    assert first["run"]["acceptance_evidence"] is False  # a CPU run never is
    assert first["run"]["stage"] == "finished"
    assert first["overall"] in ("pending second run", "fail")  # a first run never passes alone
    saved = json.loads((root / "run-1" / "probe.json").read_text(encoding="utf-8"))
    assert saved == first
    summary = (root / "run-1" / "probe.md").read_text(encoding="utf-8")
    assert summary.startswith("# Graphics processing unit (GPU) acceptance probe")
    assert set(first["run"]["phases"]) >= {"start_up_seconds", "input_preparation_seconds"}


def test_the_deterministic_checks_pass_on_the_cpu(runs: Runs) -> None:
    first, _, _ = runs
    for check in ("C1", "C3", "C4", "C6", "C7", "C8", "C10", "C11", "C12"):
        failed = [item for item in _items(first, check) if item["outcome"] is False]
        assert not failed, (check, failed)
    assert first["checks"]["C2"]["status"] != "fail"
    sampling = [item for item in _items(first, "C3") if item["subject"].endswith("sampling_unit")]
    assert sampling
    assert all(item["outcome"] == "review" for item in sampling)  # not natively observable


def test_all_eight_configurations_refits_and_scores_are_recorded(runs: Runs) -> None:
    first, _, _ = runs
    subjects = [fit["subject"] for fit in first["fits"]]
    for name in ("LGB-u", "LGB-b", "Cat-u", "Cat-b"):
        assert f"{name} cpu" in subjects
        assert f"{name} cpu refit" in subjects
    for name in ("RES-u", "RES-b"):
        assert f"{name} small fit 1" in subjects
        assert f"{name} small fit 2" in subjects
    for name in ("FTT-u", "FTT-b"):
        for variant in ("small", "padded"):
            assert f"{name} {variant} fit 1" in subjects
    assert all("parameters" in fit for fit in first["fits"])
    keys = set(first["scores"])
    assert "LGB-u.default.cpu.fit1.reload" in keys
    assert "Cat-b.default.cpu.refit.repeat" in keys
    assert "FTT-b.padded.cpu.fit1.reference" in keys
    assert "RES-u.small.cpu.fit2.score" in keys
    assert len(keys) == 4 * 6 + 6 * 5  # trees: fit and refit x 3 paths; neural: 5 arrays


def test_c4_tolerances_and_timing_are_recorded(runs: Runs) -> None:
    first, _, _ = runs
    (dataset,) = _items(first, "C4")
    assert dataset["outcome"] is True
    assert len(set(dataset["sha256"].values())) == 1
    for item in _items(first, "C8"):
        assert item["tolerance"] == 1e-4
        assert item["max_abs_logit_difference"] <= 1e-4
    timing = [item for item in _items(first, "C13") if "end_to_end" in item]
    assert len(timing) == 4 + 6
    neural = [item for item in timing if "cpu_scoring" in item]
    assert all(item["full_train_epoch_minutes_extrapolated"] > 0 for item in neural)


def test_the_second_run_compares_with_the_first_across_processes(runs: Runs) -> None:
    first, second, _ = runs
    assert first["run"]["process"]["token"] != second["run"]["process"]["token"]
    comparison = second["comparison"]
    assert comparison["comparable"] is True
    assert set(comparison["scores"]) == set(first["scores"])
    across = [item for item in _items(second, "C11") if item["subject"].endswith("across runs")]
    neural = [item for item in across if item["subject"].startswith(("RES", "FTT"))]
    assert neural
    assert all(item["outcome"] is True for item in neural)  # T3 across fresh processes
    trees = [item for item in across if item["subject"].startswith(("LGB", "Cat"))]
    assert trees
    assert all(item["outcome"] is None for item in trees)  # descriptive
    assert second["overall"] != "pending second run"


# --- Acceptance gates -----------------------------------------------------------------------------


def test_a_run_never_reuses_a_directory(runs: Runs) -> None:
    _, _, root = runs
    with pytest.raises(SpecificationError, match="already holds a probe run"):
        run_small(root / "run-1")
    with pytest.raises(SpecificationError, match="separate from the first run"):
        run_small(root / "run-1" / "nested", root / "run-1" / "probe.json")


@pytest.mark.parametrize(
    ("change", "difference"),
    [
        (lambda r: r["settings"].update(sha256="0" * 64), "settings.sha256"),
        (lambda r: r["code"]["package"].update(source_sha256="0"), "code.package.source_sha256"),
        (lambda r: r["environment"].update(torch_threads={"intraop": 1}), "environment"),
        (lambda r: r["synthetic"].update(neural_sha256="0" * 64), "synthetic.neural_sha256"),
        (lambda r: r["run"].update(stage="failed"), "first run not finished"),
        (lambda r: r["scores"].pop(next(iter(r["scores"]))), "score keys"),
    ],
)
def test_an_incomparable_first_run_fails_c11(
    runs: Runs, tmp_path: Path, change: Any, difference: str
) -> None:
    first, second, _ = runs
    changed = json.loads(json.dumps(first))
    change(changed)
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    probe = _probe(tmp_path)
    probe.arrays = {key: (np.zeros(1), np.zeros(1, dtype=np.int64)) for key in second["scores"]}
    result = runner.compare_runs(probe, second, path)
    assert result["comparable"] is False
    assert difference in result["differences"]
    assert probe.book.status("C11") == "fail"


def test_the_same_process_is_never_a_second_run(runs: Runs, tmp_path: Path) -> None:
    first, _, root = runs
    probe = _probe(tmp_path)
    probe.arrays = {key: (np.zeros(1), np.zeros(1, dtype=np.int64)) for key in first["scores"]}
    result = runner.compare_runs(probe, first, root / "run-1" / "probe.json")
    assert "same process" in result["differences"]


def test_weights_must_follow_the_class_weight_rule(tmp_path: Path) -> None:
    features = load_feature_spec(REPO_PATHS.features_spec)
    levels = category_levels(read_catalog(REPO_PATHS.feature_catalog), features)
    data = tree_data(features, levels, SIZES)
    train, valid = data.train[Weighting.BALANCED], data.valid[Weighting.BALANCED]
    probe = _probe(tmp_path)
    runner.weights_item(probe, "valid", train, valid, Weighting.BALANCED, 5)
    twos = np.full(len(train.labels), 2.0)
    constant = ModelData(train.features, train.labels, twos, train.times, train.ids)
    runner.weights_item(probe, "constant balanced weights", constant, valid, Weighting.BALANCED, 5)
    ones = np.ones(len(valid.labels))
    unit_valid = ModelData(valid.features, valid.labels, ones, valid.times, valid.ids)
    runner.weights_item(probe, "V1 weights dropped", train, unit_valid, Weighting.BALANCED, 5)
    received = train.weights.astype(np.float32) * np.float32(2)
    runner.weights_item(probe, "received", train, valid, Weighting.BALANCED, 5, received)
    outcomes = {item["subject"]: item["outcome"] for item in probe.book.items["C6"]}
    assert outcomes == {
        "valid": True,
        "constant balanced weights": False,
        "V1 weights dropped": False,
        "received": False,
    }


def test_warnings_are_classified(tmp_path: Path) -> None:
    probe = _probe(tmp_path)
    accepted = "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf"
    probe.capture.records = [
        {"logger": runner.LIGHTGBM_LOGGER, "phase": "a", "text": accepted},
        {"logger": "review_capacity.other", "phase": "b", "text": accepted},
        {"logger": "py.warnings", "phase": "c", "text": "option x is ignored"},
        {"logger": "py.warnings", "phase": "d", "text": "something new"},
    ]
    runner.classify_warnings(probe)
    outcomes = [item["outcome"] for item in probe.book.items["C2"]]
    assert outcomes == [True, "review", False, "review"]


def test_a_failure_leaves_a_failed_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def broken(probe: runner.Probe, _tree: Any) -> None:
        probe.book.add("C6", "before the failure", True)
        raise ModelError("a fit failed")

    monkeypatch.setattr(runner, "run_trees", broken)
    with pytest.raises(ModelError, match="a fit failed"):
        run_small(tmp_path / "failed")
    record = json.loads((tmp_path / "failed" / "probe.json").read_text(encoding="utf-8"))
    assert record["run"]["stage"] == "failed"
    assert record["overall"] == "failed"
    assert record["run"]["error"] == "ModelError: a fit failed"
    assert record["checks"]["C6"]["items"][0]["subject"] == "before the failure"


def test_a_start_up_failure_leaves_a_failed_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def no_cuda(_device: Any) -> None:
        raise EnvironmentMismatchError("no CUDA")

    monkeypatch.setattr(runner, "torch_device", no_cuda)
    with pytest.raises(EnvironmentMismatchError, match="no CUDA"):
        run_small(tmp_path / "start-up")
    record = json.loads((tmp_path / "start-up" / "probe.json").read_text(encoding="utf-8"))
    assert (record["run"]["stage"], record["overall"]) == ("failed", "failed")
    assert record["run"]["error"] == "EnvironmentMismatchError: no CUDA"
    assert record["code"] is not None  # recorded before the device check
    assert {entry["status"] for entry in record["checks"].values()} == {"not run"}
    assert "no CUDA" in (tmp_path / "start-up" / "probe.md").read_text(encoding="utf-8")


def test_an_interruption_during_start_up_is_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def interrupted(_path: Path) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "load_feature_spec", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_small(tmp_path / "interrupted")
    record = json.loads((tmp_path / "interrupted" / "probe.json").read_text(encoding="utf-8"))
    assert (record["run"]["stage"], record["overall"]) == ("interrupted", "interrupted")


def _probe_never_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    def started(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("the probe must not start")

    monkeypatch.setattr(runner, "run_probe", started)


@pytest.mark.usefixtures("restore_logging")
@pytest.mark.parametrize("entries", [("probe.json", "probe.log"), ("probe.log",)])
def test_the_command_never_writes_to_an_earlier_runs_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entries: tuple[str, ...]
) -> None:
    for entry in entries:
        (tmp_path / entry).write_text("the earlier run\n", encoding="utf-8")
    before = (tmp_path / "probe.log").read_bytes()
    _probe_never_starts(monkeypatch)
    assert main(["probe-gpu", "--output", str(tmp_path)]) == 2  # SpecificationError
    assert (tmp_path / "probe.log").read_bytes() == before


@pytest.mark.usefixtures("restore_logging")
def test_the_command_refuses_another_log_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    earlier = tmp_path / "earlier.log"
    earlier.write_text("the earlier run\n", encoding="utf-8")
    before = earlier.read_bytes()
    _probe_never_starts(monkeypatch)
    arguments = ["--log-file", str(earlier), "probe-gpu", "--output", str(tmp_path / "new")]
    assert main(arguments) == 2
    assert earlier.read_bytes() == before
    assert not (tmp_path / "new").exists()


def test_dropout_replay_counts_the_training_replacements(monkeypatch: pytest.MonkeyPatch) -> None:
    """C7: replaying the recorded seeds gives exactly the replacements of the training loop."""
    features = load_feature_spec(REPO_PATHS.features_spec)
    levels = category_levels(read_catalog(REPO_PATHS.feature_catalog), features)
    layout = neural_layout(features, levels)
    data = neural_data(layout, SIZES)
    counted: list[int] = []
    original = neural_training.drop_categories

    def counting(categorical: torch.Tensor, *args: Any) -> torch.Tensor:
        dropped = original(categorical, *args)
        counted.append(int((dropped != categorical).sum()))
        return dropped

    monkeypatch.setattr(neural_training, "drop_categories", counting)
    settings = load_settings_with_snapshot(REPO_PATHS)[0]
    model: NeuralModel = ResNetModel(
        configuration("RES-u"), 42, settings=settings, layout=layout, device="cpu"
    )
    hyperparameters = {
        **SMALL["NEURAL_TRAINING"],
        **SMALL["NEURAL_SHARED"]["resnet"],
        **SMALL["NEURAL_VARIANTS"]["resnet"]["small"],
    }
    model.fit(data.train[Weighting.UNWEIGHTED], data.valid[Weighting.UNWEIGHTED], hyperparameters)
    epochs = model.parameters["resolved"]["schedule"]["epochs_run"]
    train = data.train[Weighting.UNWEIGHTED]
    replayed = runner.dropout_replacements(train, layout, 64, 42, epochs)
    assert replayed == sum(counted) > 0


def test_probe_command_is_registered() -> None:
    args = build_parser().parse_args(["probe-gpu", "--output", "out", "--previous", "p.json"])
    assert args.handler is _probe_gpu
    assert (args.output, args.previous) == (Path("out"), Path("p.json"))
    with pytest.raises(SystemExit):
        build_parser().parse_args(["probe-gpu"])  # --output is required


def test_the_probe_log_is_in_the_output_directory() -> None:
    args = build_parser().parse_args(["probe-gpu", "--output", "evidence/x"])
    assert _log_file(args) == REPO_PATHS.root / "evidence" / "x" / "probe.log"
    args = build_parser().parse_args(["--log-file", "a.log", "probe-gpu", "--output", "x"])
    with pytest.raises(SpecificationError, match="omit --log-file"):
        _log_file(args)
    args = build_parser().parse_args(["--log-file", "a.log", "prepare-data"])
    assert _log_file(args) == Path("a.log")
