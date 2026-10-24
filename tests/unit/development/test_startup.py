"""Shared start-up of the tree-based libraries and the neural models (DR-015 §9, #143, #124)."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from review_capacity.development.startup import (
    CATBOOST_SCOPE,
    LIGHTGBM_SCOPE,
    NEURAL_SCOPE,
    Startup,
    catboost_startup,
    measure_startups,
    neural_startup,
)
from review_capacity.models.registry import configuration
from review_capacity.models.run_record import load_settings_with_snapshot
from support.synthetic import REPO_PATHS


def settings() -> Any:
    return load_settings_with_snapshot(REPO_PATHS)[0]


class Ticks:
    """A clock that advances by one second at every reading."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


def test_the_catboost_warmup_is_one_iteration_on_the_gpu_and_is_timed() -> None:
    seen: list[Mapping[str, Any]] = []
    startup = catboost_startup(settings(), warm=seen.append, clock=Ticks())
    (params,) = seen
    assert params["iterations"] == 1
    assert params["task_type"] == "GPU"
    assert params["random_seed"] == 0
    assert params["allow_writing_files"] is False
    assert params["verbose"] is False
    assert startup == Startup(1.0, CATBOOST_SCOPE)


def test_a_failed_warmup_is_not_measured_and_does_not_stop_the_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail(_params: Mapping[str, Any]) -> None:
        raise RuntimeError("no GPU")

    startup = catboost_startup(settings(), warm=fail)
    assert startup == Startup(None, CATBOOST_SCOPE)
    assert "warm-up failed" in caplog.text


def test_each_library_of_the_run_has_its_shared_startup() -> None:
    chosen = (configuration("LGB-u"), configuration("Cat-b"))
    startups = measure_startups(
        chosen, settings(), lightgbm_seconds=0.7, catboost=lambda _s: Startup(2.0, CATBOOST_SCOPE)
    )
    assert startups == {
        "lightgbm": Startup(0.7, LIGHTGBM_SCOPE),
        "catboost": Startup(2.0, CATBOOST_SCOPE),
    }


def test_a_library_that_the_run_does_not_use_has_no_startup() -> None:
    def never(_settings: Any) -> Startup:
        raise AssertionError("CatBoost is not part of this run")

    startups = measure_startups(
        (configuration("LGB-u"),), settings(), lightgbm_seconds=None, catboost=never
    )
    assert set(startups) == {"lightgbm"}
    assert startups["lightgbm"].seconds is None


def test_the_neural_models_share_one_timed_cuda_startup() -> None:
    calls: list[int] = []

    def warm() -> None:
        calls.append(1)

    startup = neural_startup(warm=warm, clock=Ticks())
    assert calls == [1]
    assert startup == Startup(1.0, NEURAL_SCOPE)
    chosen = (configuration("RES-u"), configuration("FTT-b"), configuration("LGB-u"))
    startups = measure_startups(
        chosen, settings(), lightgbm_seconds=0.7, neural=lambda: Startup(3.0, NEURAL_SCOPE)
    )
    assert startups["resnet"] is startups["ft_transformer"]
    assert startups["resnet"] == Startup(3.0, NEURAL_SCOPE)
    assert set(startups) == {"lightgbm", "resnet", "ft_transformer"}


def test_a_failed_neural_warmup_is_not_measured(caplog: pytest.LogCaptureFixture) -> None:
    def fail() -> None:
        raise RuntimeError("no GPU")

    assert neural_startup(warm=fail) == Startup(None, NEURAL_SCOPE)
    assert "warm-up failed" in caplog.text


def test_a_run_without_neural_models_does_not_start_cuda() -> None:
    def never() -> Startup:
        raise AssertionError("the neural start-up is measured only for neural models")

    startups = measure_startups(
        (configuration("LGB-u"),), settings(), lightgbm_seconds=0.7, neural=never
    )
    assert set(startups) == {"lightgbm"}


def test_a_neural_model_trained_on_the_cpu_has_no_cuda_startup() -> None:
    def never() -> Startup:
        raise AssertionError("the CUDA start-up is measured only for models trained on the GPU")

    cpu = replace(settings(), training_devices={**settings().training_devices, "resnet": "cpu"})
    startups = measure_startups((configuration("RES-u"),), cpu, lightgbm_seconds=None, neural=never)
    assert startups == {}


def test_the_neural_scope_states_what_it_covers() -> None:
    assert "synchronised at its end" in NEURAL_SCOPE
    assert "only where they had not happened earlier" in NEURAL_SCOPE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_real_neural_warmup_leaves_the_random_state_alone_and_keeps_no_tensor() -> None:
    """On a GPU: the warm-up draws from no global generator. What it leaves allocated is the
    library's own workspace (about 16 MiB of cuBLAS on the measured machine), the same after the
    first call and after the second, so no tensor of the warm-up stays alive."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state(0)

    first = neural_startup()
    allocated = torch.cuda.memory_allocated(0)
    second = neural_startup()

    assert first.status == second.status == "measured"
    assert random.getstate() == python_state
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_state[0]
    assert np.array_equal(numpy_after[1], numpy_state[1])
    assert numpy_after[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.equal(torch.cuda.get_rng_state(0), cuda_state)
    assert torch.cuda.memory_allocated(0) == allocated
