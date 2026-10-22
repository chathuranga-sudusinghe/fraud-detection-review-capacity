"""Shared start-up of the tree-based libraries (DR-015 §9, Issue #143).

Library and device start-up is a once-per-process cost that belongs to no trial: it is measured
before the first trial and recorded beside each fit, never inside the fit time. LightGBM is
imported by the build verification that every run starts with, so that step is timed. CatBoost
initialises its GPU training path inside its first fit; a one-iteration fit on 64 synthetic rows
(seed 0, nothing written, model and data released) is therefore run and timed once, and every real
trial builds its own Pool and model afresh with its declared seed. The warm-up is a stated
procedure, not a guarantee that no later lazy initialisation remains; if it fails, the failure
is logged and the start-up is recorded as not measured, and the run continues.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from review_capacity.config import ModelSettings
from review_capacity.models.registry import Configuration

logger = logging.getLogger(__name__)

LIGHTGBM_SCOPE = (
    "LightGBM library import and build verification, once per process before the first trial"
)
CATBOOST_SCOPE = (
    "CatBoost library import and one one-iteration GPU fit on 64 synthetic rows (seed 0; model and "
    "data released), once per process before the first trial"
)
WARMUP_ROWS = 64
WARMUP_SEED = 0


@dataclass(frozen=True)
class Startup:
    """The shared start-up of a library: its seconds (None if it could not be measured) and
    what it covers."""

    seconds: float | None
    scope: str

    @property
    def status(self) -> str:
        """``measured``, or ``not_measured``: a fit then ran without its start-up having been
        excluded, so its time may include the library and device initialisation."""
        return "not_measured" if self.seconds is None else "measured"


def _catboost_warmup(params: Mapping[str, Any]) -> None:
    import numpy as np
    from catboost import CatBoostClassifier, Pool

    rng = np.random.default_rng(WARMUP_SEED)
    features = rng.normal(size=(WARMUP_ROWS, 3))
    labels = np.tile([0, 1], WARMUP_ROWS // 2)
    model = CatBoostClassifier(**params)
    model.fit(Pool(features, labels))


def catboost_startup(
    settings: ModelSettings,
    *,
    warm: Callable[[Mapping[str, Any]], None] = _catboost_warmup,
    clock: Callable[[], float] = time.perf_counter,
) -> Startup:
    """Time the CatBoost import and the one-iteration GPU warm-up fit."""
    params = {
        **settings.catboost.params("gpu"),
        "iterations": 1,
        "learning_rate": 0.1,
        "random_seed": WARMUP_SEED,
        "verbose": False,
        "allow_writing_files": False,
    }
    started = clock()
    try:
        warm(params)
    except Exception:  # the documented case: a warm-up that fails does not stop the run
        logger.exception("The CatBoost start-up warm-up failed; its start-up is not measured")
        return Startup(None, CATBOOST_SCOPE)
    return Startup(clock() - started, CATBOOST_SCOPE)


def measure_startups(
    chosen: Sequence[Configuration],
    settings: ModelSettings,
    *,
    lightgbm_seconds: float | None,
    catboost: Callable[[ModelSettings], Startup] = catboost_startup,
) -> dict[str, Startup]:
    """The shared start-up of each tree-based library the run uses, by model name."""
    models = {c.model.name for c in chosen}
    startups: dict[str, Startup] = {}
    if "lightgbm" in models:
        startups["lightgbm"] = Startup(lightgbm_seconds, LIGHTGBM_SCOPE)
    if "catboost" in models:
        startups["catboost"] = catboost(settings)
    return startups
