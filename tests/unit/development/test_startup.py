"""Shared start-up of the tree-based libraries (DR-015 §9, Issue #143)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from review_capacity.development.startup import (
    CATBOOST_SCOPE,
    LIGHTGBM_SCOPE,
    Startup,
    catboost_startup,
    measure_startups,
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
