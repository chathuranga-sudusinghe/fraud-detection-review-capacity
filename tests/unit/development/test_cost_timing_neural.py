"""The timing of the neural models and the reference (DR-015 §9; Issue #151): explicit device
synchronisation, batch-call warm-up, device-resident model-only inputs and measured thread pools,
on synthetic data on the CPU."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from review_capacity.development.cost_timing import (
    _boundary_calls,
    gpu_clock,
    library_threads,
    timed_on,
)
from review_capacity.development.runner import run_development
from review_capacity.development.scoring import ScoredPartition, ScoringContext, load_selected
from review_capacity.development.trial import TrialOutcome, trial_directory
from review_capacity.models.neural_training import NeuralModel
from review_capacity.models.registry import REFERENCE_SUBJECT, configuration
from review_capacity.training.timing import Clock
from support.search import FTT_CANDIDATES, mixed_setup


def test_the_gpu_clock_synchronises_cuda_device_zero_around_the_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))
    clock = gpu_clock(Clock())
    clock.synchronise()
    assert calls == [torch.device("cuda", 0)]


def test_without_cuda_there_is_nothing_to_synchronise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    gpu_clock(Clock()).synchronise()  # no error


def test_the_reference_records_the_thread_pools_of_its_libraries(
    tmp_path: Path,
) -> None:
    threads, pools = library_threads()
    assert threads >= 1
    assert all({"user_api", "num_threads"} <= set(pool) for pool in pools)
    setup = mixed_setup(tmp_path, ("LR",))
    who = timed_on(setup.settings, setup.session, REFERENCE_SUBJECT, "selected", "cpu")
    assert who.threads == threads
    assert who.extra["thread_pools"] == pools
    assert who.cpu_affinity == tuple(int(c) for c in setup.session["allowed_cpus"])


def test_a_neural_warmup_is_batch_calls_of_the_normal_and_the_final_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = mixed_setup(tmp_path, ("FTT-b",))
    assert setup.neural is not None
    run_development(setup)
    scoring = ScoringContext(
        run=setup.run,
        identity=setup.identity,
        declaration=setup.declaration,
        settings=setup.settings,
        features=setup.features,
        inputs=setup.scoring_inputs,
        session=setup.session,
        neural=setup.neural,
    )
    chosen = configuration("FTT-b")
    selected = TrialOutcome(
        "FTT-b", 1, dict(FTT_CANDIDATES[0]), 0.5, 1, 2, "maximum", 1.0, 0.1, None, 1,
        trial_directory(setup.run, "FTT-b", 1), {}, {}, {},
    )  # fmt: skip
    model = load_selected(scoring, chosen, selected)
    assert isinstance(model, NeuralModel)
    scored = ScoredPartition("FTT-b", "V1", setup.scoring_inputs.v1, {}, {}, 0.0)
    seen: list[int] = []
    original = model.predict_prepared

    def spy(prepared: Any) -> Any:
        seen.append(sum(numeric.shape[0] for numeric, _ in prepared))
        return original(prepared)

    monkeypatch.setattr(model, "predict_prepared", spy)
    one_pass, warmup = _boundary_calls(scoring, model, scored, "model_only")
    for call in warmup:
        call()
    batch = model.parameters["prediction"]["eval_batch_size"]
    rows = scored.rows.rows
    assert len(warmup) == setup.declaration.timing.warmup_neural
    assert seen == [batch] * (len(warmup) - 1) + [rows % batch or batch]
    one_pass()
    assert seen[-1] == rows  # the timed pass covers the whole partition
