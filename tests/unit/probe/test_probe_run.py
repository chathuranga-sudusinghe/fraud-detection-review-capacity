"""The GPU acceptance probe end to end on the CPU, with small synthetic sizes and small networks: a
first run in this process and a second run in a fresh process, compared with the first (C11),
plus the failure paths of the acceptance gates. A CPU run exercises the code paths; it is never
acceptance evidence. No research data."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from review_capacity.cli import _log_file, _probe_gpu, _probe_paths, build_parser, main
from review_capacity.config import load_feature_spec
from review_capacity.data.catalog import read_catalog
from review_capacity.errors import EnvironmentMismatchError, ModelError, SpecificationError
from review_capacity.models import neural_training
from review_capacity.models.base import ModelData
from review_capacity.models.lightgbm_model import LightGBMModel
from review_capacity.models.neural_training import NeuralModel
from review_capacity.models.registry import Weighting, configuration
from review_capacity.models.resnet_model import ResNetModel
from review_capacity.models.run_record import load_settings_with_snapshot
from review_capacity.probe import diagnostics, runner
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
    for name in ("LGB-u", "LGB-b"):
        assert subjects.count(f"{name} cpu") == 2  # fitted twice with the same seed (C11)
        assert f"{name} cpu refit" not in subjects  # LightGBM trains on the CPU only
    for name in ("Cat-u", "Cat-b"):
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
    assert "LGB-b.default.cpu.fit2.score" in keys
    assert "Cat-b.default.cpu.refit.repeat" in keys
    assert "FTT-b.padded.cpu.fit1.reference" in keys
    assert "RES-u.small.cpu.fit2.score" in keys
    # LightGBM: fit 1 x 3 paths and fit 2; CatBoost: fit and refit x 3 paths; neural: 5 arrays
    assert len(keys) == 2 * 4 + 2 * 6 + 6 * 5


def test_c4_tolerances_and_timing_are_recorded(runs: Runs) -> None:
    first, _, _ = runs
    (dataset,) = _items(first, "C4")
    assert dataset["outcome"] is True
    assert dataset["subject"] == "LightGBM Datasets, two cpu constructions"
    assert len(dataset["sha256"]) == 2
    assert len(set(dataset["sha256"].values())) == 1
    assert dataset["largest_categorical_bins"] >= 2
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
    deterministic = [item for item in across if item["subject"].startswith(("LGB", "RES", "FTT"))]
    assert any(item["subject"].startswith("LGB") for item in deterministic)
    assert all(item["outcome"] is True for item in deterministic)  # T3 across fresh processes
    catboost = [item for item in across if item["subject"].startswith("Cat")]
    assert catboost
    assert all(item["outcome"] is None for item in catboost)  # descriptive
    models = [item for item in across if item["subject"].endswith(".model across runs")]
    assert len(models) == 4  # LGB-u and LGB-b, fits 1 and 2: identical saved model text (T3)
    assert all(item["outcome"] is True for item in models)
    assert set(first["model_hashes"]) == set(second["model_hashes"])
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


def test_a_changed_lightgbm_model_fails_c11_across_runs(runs: Runs, tmp_path: Path) -> None:
    """Identical scores are not enough: LightGBM's complete saved model must repeat (DR-013 §8)."""
    first, second, root = runs
    changed = json.loads(json.dumps(first))
    model = next(key for key in changed["model_hashes"] if key.startswith("LGB-u"))
    changed["model_hashes"][model] = "0" * 64
    shutil.copytree(root / "run-1" / "scores", tmp_path / "first" / "scores")
    (tmp_path / "first" / "probe.json").write_text(json.dumps(changed), encoding="utf-8")
    probe = _probe(tmp_path / "second")
    probe.arrays = {
        key: diagnostics.load_scores(root / "run-2" / "scores", record)
        for key, record in second["scores"].items()
    }
    probe.model_hashes = dict(second["model_hashes"])
    result = runner.compare_runs(probe, second, tmp_path / "first" / "probe.json")
    assert result["comparable"] is True
    outcomes = {item["subject"]: item["outcome"] for item in probe.book.items["C11"]}
    assert outcomes[f"{model} across runs"] is False
    scores = [o for subject, o in outcomes.items() if subject.startswith("LGB-u.default.cpu.fit1.")]
    assert scores.count(True) == 3  # the score, repeat and reload arrays are unchanged
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


@pytest.mark.parametrize(
    ("bins", "outcomes"),
    [
        (300, [True, True]),
        # without a categorical feature above max_bin, the notices follow the general rule: the
        # first needs a recorded review, the second ("may be ignored") fails
        (255, ["review", False]),
        (None, ["review", False]),
    ],
)
def test_categorical_bin_notices_need_a_categorical_feature_above_max_bin(
    tmp_path: Path, bins: int | None, outcomes: list[bool | str]
) -> None:
    probe = _probe(tmp_path)
    probe.categorical_bins = bins  # max_bin is 255 (config/models.yaml)
    probe.capture.records = [
        {"logger": runner.LIGHTGBM_LOGGER, "phase": "a", "text": f"[LightGBM] [Warning] {text}"}
        for text in runner.CATEGORICAL_BIN_WARNINGS
    ]
    runner.classify_warnings(probe)
    assert [item["outcome"] for item in probe.book.items["C2"]] == outcomes


@pytest.mark.parametrize(
    ("logger", "text", "outcome"),
    [
        # an approved notice with anything appended is not the notice: the failure rule applies
        (
            runner.LIGHTGBM_LOGGER,
            f"{runner.CATEGORICAL_BIN_WARNINGS[0]} option x is ignored",
            False,
        ),
        (runner.LIGHTGBM_LOGGER, f"{runner.CATEGORICAL_BIN_WARNINGS[0]} and more", "review"),
        # the notices are accepted from LightGBM's own logger only
        ("py.warnings", runner.CATEGORICAL_BIN_WARNINGS[0], "review"),
        ("py.warnings", runner.CATEGORICAL_BIN_WARNINGS[1], False),
        # the failure rule takes precedence over "no further splits"
        (runner.LIGHTGBM_LOGGER, f"{runner.ACCEPTED_WARNING}; option x is ignored", False),
    ],
)
def test_only_the_exact_notices_are_accepted(
    tmp_path: Path, logger: str, text: str, outcome: bool | str
) -> None:
    probe = _probe(tmp_path)
    probe.categorical_bins = 300  # above max_bin 255
    probe.capture.records = [{"logger": logger, "phase": "a", "text": text}]
    runner.classify_warnings(probe)
    assert [item["outcome"] for item in probe.book.items["C2"]] == [outcome]


def test_a_gpu_run_trains_lightgbm_on_the_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The routing of run_trees on a GPU run, with fitting, checking and scoring replaced by stand-
    ins (no CUDA): LightGBM twice on the CPU, CatBoost on the GPU with its CPU refit only."""
    fits: list[tuple[str, str, str, int | None]] = []
    ids = np.arange(3, dtype=np.int64)

    def fit(
        _state: Any, library: str, name: str, device: str, _data: Any, iterations: int | None
    ) -> Any:
        fits.append((library, name, device, iterations))
        return SimpleNamespace(
            configuration=configuration(name),
            device=device,
            fitted_iterations=5,
            predict=lambda _frame: np.zeros(3),
        )

    monkeypatch.setattr(runner, "_fit_tree", fit)
    monkeypatch.setattr(runner, "_tree_fit_checks", lambda _p, m, _t, _w: f"{m.device}")
    monkeypatch.setattr(runner, "_tree_scores", lambda *_args: np.zeros(3))
    monkeypatch.setattr(runner, "_tree_timing", lambda *_args: {})
    monkeypatch.setattr(runner, "_tree_checks", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "_model_text",
        lambda model: "model text" if model.configuration.model.name == "lightgbm" else None,
    )
    probe = _probe(tmp_path)
    probe.device = "gpu"
    weightings = dict.fromkeys(Weighting)
    tree: Any = SimpleNamespace(train=weightings, valid=weightings, score=None, score_ids=ids)
    runner.run_trees(probe, tree)
    assert fits == [
        ("lightgbm", "LGB-u", "cpu", None),
        ("lightgbm", "LGB-u", "cpu", None),
        ("lightgbm", "LGB-b", "cpu", None),
        ("lightgbm", "LGB-b", "cpu", None),
        ("catboost", "Cat-u", "gpu", None),
        ("catboost", "Cat-u", "cpu", 5),
        ("catboost", "Cat-b", "gpu", None),
        ("catboost", "Cat-b", "cpu", 5),
    ]
    assert set(probe.model_hashes) == {
        f"{name}.default.cpu.{fit}.model" for name in ("LGB-u", "LGB-b") for fit in ("fit1", "fit2")
    }


def test_each_model_trains_on_its_device(tmp_path: Path) -> None:
    probe = _probe(tmp_path)
    devices = {model: probe.training_device(model) for model in runner.LABELS}
    assert devices == dict.fromkeys(runner.LABELS, "cpu")  # a CPU run trains everything on the CPU
    probe.device = "gpu"  # no CUDA needed: only the device map is read
    assert probe.training_device("lightgbm") == "cpu"  # DR-013 §1-§2
    assert probe.training_device("catboost") == "gpu"
    assert probe.training_device("ft_transformer") == "gpu"


def test_the_record_names_the_running_fit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A native abort ends the process without a Python handler: the checkpoint before each fit
    leaves the running fit in the record."""
    seen: list[str] = []

    def aborted(_self: Any, *_args: Any, **_kwargs: Any) -> None:
        record = json.loads((tmp_path / "abort" / "probe.json").read_text(encoding="utf-8"))
        seen.append(record["run"]["stage"])
        raise ModelError("native abort")

    monkeypatch.setattr(LightGBMModel, "fit", aborted)
    with pytest.raises(ModelError, match="native abort"):
        run_small(tmp_path / "abort")
    assert seen == ["fitting LGB-u cpu"]


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
    model.fit(
        data.train[Weighting.UNWEIGHTED],
        data.valid[Weighting.UNWEIGHTED],
        hyperparameters,
        objective=data.objective,
    )
    epochs = model.parameters["resolved"]["schedule"]["epochs_run"]
    train = data.train[Weighting.UNWEIGHTED]
    replayed = runner.dropout_replacements(train, layout, 64, 42, epochs)
    assert replayed == sum(counted) > 0


def test_probe_command_is_registered() -> None:
    args = build_parser().parse_args(["probe-gpu", "--output", "out", "--previous", "p.json"])
    assert args.handler is _probe_gpu
    assert (args.output, args.previous) == (Path("out"), Path("p.json"))
    assert build_parser().parse_args(["probe-gpu"]).output is None  # dated default


def test_the_probe_log_is_in_the_output_directory() -> None:
    args = build_parser().parse_args(["probe-gpu", "--output", "evidence/x"])
    assert _log_file(args) == REPO_PATHS.root / "evidence" / "x" / "probe.log"
    args = build_parser().parse_args(["probe-gpu"])
    log_file = _log_file(args)
    assert log_file is not None
    assert log_file.parent.parent == REPO_PATHS.root / "outputs" / "gpu_acceptance_probe"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{6}Z", log_file.parent.name)
    assert _probe_paths(args, REPO_PATHS)[0] == log_file.parent  # one directory per run
    args = build_parser().parse_args(["--log-file", "a.log", "probe-gpu", "--output", "x"])
    with pytest.raises(SpecificationError, match="omit --log-file"):
        _log_file(args)
    args = build_parser().parse_args(["--log-file", "a.log", "prepare-data"])
    assert _log_file(args) == Path("a.log")
