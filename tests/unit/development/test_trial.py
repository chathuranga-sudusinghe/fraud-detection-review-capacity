"""One search trial of a tree-based configuration: LightGBM on synthetic data on the CPU (no
project data). CatBoost trains on the GPU and is covered by its adapter tests and the GPU
acceptance runs, not here."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from review_capacity.development import trial as trial_module
from review_capacity.development.run_output import load_run
from review_capacity.development.trial import (
    SearchContext,
    hyperparameters,
    run_trial,
    tree_model,
)
from review_capacity.errors import FileAccessError, ReviewCapacityError, SpecificationError
from review_capacity.hashing import file_sha256
from review_capacity.models.catboost_model import CatBoostModel
from review_capacity.models.lightgbm_model import LightGBMModel
from review_capacity.models.registry import configuration
from review_capacity.training.timing import Clock
from support.memory import memory_setup
from support.search import CANDIDATES, PATIENCE, candidate_file, search_context

pytestmark = pytest.mark.usefixtures("restore_lightgbm_logger")


@pytest.fixture
def context(tmp_path: Path) -> SearchContext:
    return search_context(tmp_path)


def test_a_trial_fits_saves_and_records(context: SearchContext) -> None:
    name = "LGB-u"
    outcome = run_trial(context, configuration(name), 1, CANDIDATES[0], candidate_file(context.run))
    assert 0.0 <= outcome.objective <= 1.0
    assert outcome.best_checkpoint <= outcome.stopped_at
    assert outcome.fitted_iterations == outcome.best_checkpoint
    assert (outcome.model_directory / "model.txt").is_file()
    assert (outcome.model_directory / "model_run.json").is_file()
    loaded = load_run(context.run.path, context.run.partitions)
    (trial,), (fit,), (checkpoint,) = (loaded.of_kind(k) for k in ("trial", "fit", "checkpoint"))
    assert trial["status"] == fit["status"] == "completed"
    assert trial["best_objective"] == outcome.objective == checkpoint["objective"]
    assert checkpoint["index"] == trial["best_checkpoint"] == outcome.best_checkpoint
    assert len(trial["checkpoints"]) == trial["stopped_at"] == outcome.stopped_at
    assert trial["patience"] == PATIENCE
    assert fit["purpose"] == "search_trial"
    assert fit["fitted_iterations"] == outcome.fitted_iterations
    assert fit["positive_weight"] == 1
    assert outcome.fit["sha256"] == file_sha256(context.run.path / outcome.fit["path"])
    assert trial["fit"] == outcome.fit
    assert trial["checkpoint"] == outcome.checkpoint
    assert trial["hyperparameters"] == CANDIDATES[0]


def test_a_balanced_configuration_uses_the_train_class_weight(context: SearchContext) -> None:
    run_trial(context, configuration("LGB-b"), 1, CANDIDATES[0], candidate_file(context.run))
    (fit,) = load_run(context.run.path, context.run.partitions).of_kind("fit")
    train = context.inputs.train
    assert fit["positive_weight"] == (train.rows - train.frauds) / train.frauds
    assert fit["positive_weight"] > 1
    assert fit["train"]["rows"] == train.rows
    assert fit["train"]["transaction_ids_sha256"] == train.ordered_ids_sha256


def test_a_failed_trial_is_recorded_and_raised(context: SearchContext) -> None:
    bad = {**CANDIDATES[0], "not_a_lightgbm_parameter": 1}
    with pytest.raises(ReviewCapacityError):
        run_trial(context, configuration("LGB-u"), 2, bad, candidate_file(context.run))
    loaded = load_run(context.run.path, context.run.partitions)
    (trial,), (fit,) = loaded.of_kind("trial"), loaded.of_kind("fit")
    assert trial["status"] == fit["status"] == "failed"
    assert trial["trial"] == fit["trial"] == 2
    assert trial["error"]
    assert fit["error"]
    assert trial["best_objective"] is None
    assert loaded.of_kind("checkpoint") == []


def test_only_the_tree_models_are_searched_here(context: SearchContext) -> None:
    with pytest.raises(SpecificationError, match="neural inputs"):
        run_trial(context, configuration("RES-u"), 1, CANDIDATES[0], candidate_file(context.run))
    assert load_run(context.run.path, context.run.partitions).of_kind("trial") == []


def test_models_are_built_for_their_library(context: SearchContext) -> None:
    assert isinstance(tree_model(context, configuration("LGB-b")), LightGBMModel)
    cat = tree_model(context, configuration("Cat-u"))
    assert isinstance(cat, CatBoostModel)
    assert cat.device == "gpu"


def test_a_candidate_cannot_set_a_fixed_declaration_value(context: SearchContext) -> None:
    clashing: dict[str, Any] = {**CANDIDATES[0], "metric": "auc"}
    with pytest.raises(SpecificationError, match="fixed declaration values"):
        hyperparameters(context.declaration, configuration("LGB-u"), clashing)
    merged = hyperparameters(context.declaration, configuration("LGB-u"), CANDIDATES[0])
    assert merged["metric"] == "None"
    assert merged["learning_rate"] == CANDIDATES[0]["learning_rate"]


def test_a_directory_that_cannot_be_created_is_recorded_and_raised(
    context: SearchContext,
) -> None:
    models = context.run.path / "models"
    models.mkdir()
    (models / "LGB-u").write_text("a file where the trial directory belongs", encoding="utf-8")
    with pytest.raises(FileAccessError, match="cannot be created"):
        run_trial(context, configuration("LGB-u"), 1, CANDIDATES[0], candidate_file(context.run))
    loaded = load_run(context.run.path, context.run.partitions)
    assert [r["status"] for r in loaded.of_kind("trial")] == ["failed"]
    assert [r["status"] for r in loaded.of_kind("fit")] == ["failed"]
    assert loaded.of_kind("checkpoint") == []


def test_a_model_run_record_that_cannot_be_written_is_recorded_and_raised(
    context: SearchContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(path: Path, _record: object) -> None:
        raise FileAccessError(f"{path}: cannot be written")

    monkeypatch.setattr("review_capacity.development.trial.write_model_run_record", refuse)
    with pytest.raises(FileAccessError, match="cannot be written"):
        run_trial(context, configuration("LGB-u"), 1, CANDIDATES[0], candidate_file(context.run))
    loaded = load_run(context.run.path, context.run.partitions)
    assert [r["status"] for r in loaded.of_kind("trial")] == ["failed"]
    assert [r["status"] for r in loaded.of_kind("fit")] == ["failed"]
    assert loaded.of_kind("checkpoint") == []


def test_the_fit_and_the_persistence_are_timed_by_their_own_boundaries(
    context: SearchContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One clock: it advances one second at each reading of the trial, and 100 seconds at each
    start and stop of the memory monitor. Only the fit's own pair and the persistence's own pair
    of readings lie inside the trial's clocks, so no monitor time and no start-up enters either
    duration even while memory is measured."""
    now = {"t": 0.0}

    def reading() -> float:
        now["t"] += 1.0
        return now["t"]

    def monitor() -> None:
        now["t"] += 100.0

    monkeypatch.setattr(trial_module, "time", SimpleNamespace(perf_counter=reading))
    measured = replace(context, memory=memory_setup(on_reset=monitor, on_read=monitor))
    outcome = run_trial(
        measured, configuration("LGB-u"), 1, CANDIDATES[0], candidate_file(measured.run)
    )
    assert outcome.fit_seconds == 1.0
    assert outcome.persistence_seconds == 1.0
    assert now["t"] >= 200.0  # the monitor did run, outside both


def test_a_trial_keeps_its_fit_memory_reading(context: SearchContext) -> None:
    measured = replace(context, memory=memory_setup([300 * 1024]))
    outcome = run_trial(
        measured, configuration("LGB-u"), 1, CANDIDATES[0], candidate_file(measured.run)
    )
    assert outcome.fit_memory is not None
    assert outcome.fit_memory.host is not None
    assert outcome.fit_memory.host.peak_mib == 300.0
    assert outcome.fit_memory.scope.startswith("the trial's fit")


def test_a_trial_without_a_memory_setup_has_no_reading(context: SearchContext) -> None:
    outcome = run_trial(
        context, configuration("LGB-u"), 1, CANDIDATES[0], candidate_file(context.run)
    )
    assert outcome.fit_memory is None


def test_a_neural_fit_is_timed_between_two_synchronisations_of_the_device(
    context: SearchContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The synchronisations and the clock readings are one ordered sequence: device work is
    complete before the clock starts and before it stops (DR-015 §9)."""
    events: list[str] = []
    ticks = iter((10.0, 14.0))

    def now() -> float:
        events.append("clock")
        return next(ticks)

    class FakeModel:
        training_history = SimpleNamespace(
            best_objective=0.5, best_checkpoint=2, stopped_at=3, stopping_reason="patience"
        )

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def fit(self, *_args: object, **_kwargs: object) -> None:
            events.append("fit")

    neural = SimpleNamespace(
        artifact=SimpleNamespace(layout=None),
        artifact_sha256="a" * 64,
        plans={"resnet": {1: None}},
        train={configuration("RES-u").weighting: None},
        v1=None,
    )
    monkeypatch.setattr(trial_module, "_neural", lambda _context: neural)
    monkeypatch.setattr(trial_module, "NEURAL_MODELS", {"resnet": FakeModel})
    monkeypatch.setattr(
        trial_module,
        "gpu_clock",
        lambda _clock: Clock(now, lambda: events.append("synchronise")),
    )
    assert context.settings.training_device("resnet") == "gpu"

    fitted = trial_module._fit_neural(context, configuration("RES-u"), 1, CANDIDATES[0])

    assert events == ["synchronise", "clock", "fit", "synchronise", "clock"]
    assert fitted.seconds == 4.0
