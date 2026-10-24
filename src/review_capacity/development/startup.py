"""Shared start-up of the tree-based libraries and of the neural models (DR-015 §9, #143, #124).

Library and device start-up is a once-per-process cost that belongs to no trial: it is measured
before the first trial and recorded beside each fit, never inside the fit time. LightGBM is
imported by the build verification that every run starts with, so that step is timed. CatBoost
initialises its GPU training path inside its first fit; a one-iteration fit on 64 synthetic rows
(seed 0, nothing written, model and data released) is therefore run and timed once, and every real
trial builds its own Pool and model afresh with its declared seed. The warm-up is a stated
procedure, not a guarantee that no later lazy initialisation remains; if it fails, the failure
is logged and the start-up is recorded as not measured, and the run continues. The neural models
share one CUDA start-up: a tiny synthetic bfloat16 forward and backward step (seed 0, on device 0,
a private generator, released afterwards) is timed once per run before the first trial and
synchronised at its end, and recorded beside each neural fit under the same rules. Its scope
states what it covers: the PyTorch import and the CUDA context initialisation fall inside it only
if they had not happened earlier in the process (the import of the model modules, the CatBoost
warm-up), so the duration is the incremental start-up of the neural path.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from review_capacity.config import ModelSettings
from review_capacity.models.registry import Subject

logger = logging.getLogger(__name__)

LIGHTGBM_SCOPE = (
    "LightGBM library import and build verification, once per process before the first trial"
)
CATBOOST_SCOPE = (
    "CatBoost library import and one one-iteration GPU fit on 64 synthetic rows (seed 0; model and "
    "data released), once per process before the first trial"
)
NEURAL_SCOPE = (
    "one bfloat16 forward and backward step on 64 synthetic rows on CUDA device 0 (seed 0; "
    "released), synchronised at its end, once per run before the first trial; it includes the "
    "PyTorch import and the CUDA context initialisation only where they had not happened earlier "
    "in the process"
)
NEURAL_MODEL_NAMES = ("resnet", "ft_transformer")
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


def _neural_warmup() -> None:
    import torch

    generator = torch.Generator().manual_seed(WARMUP_SEED)
    device = torch.device("cuda", 0)
    inputs = torch.randn(WARMUP_ROWS, 8, generator=generator).to(device)
    weight = torch.randn(8, 8, generator=generator).to(device).requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = (inputs @ weight).float().square().mean()
    loss.backward()  # type: ignore[no-untyped-call]
    torch.cuda.synchronize(device)


def neural_startup(
    *,
    warm: Callable[[], None] = _neural_warmup,
    clock: Callable[[], float] = time.perf_counter,
) -> Startup:
    """Time the shared CUDA warm-up step of the neural models (its scope is ``NEURAL_SCOPE``)."""
    started = clock()
    try:
        warm()
    except Exception:  # the documented case: a warm-up that fails does not stop the run
        logger.exception("The neural start-up warm-up failed; its start-up is not measured")
        return Startup(None, NEURAL_SCOPE)
    return Startup(clock() - started, NEURAL_SCOPE)


def measure_startups(
    chosen: Sequence[Subject],
    settings: ModelSettings,
    *,
    lightgbm_seconds: float | None,
    catboost: Callable[[ModelSettings], Startup] = catboost_startup,
    neural: Callable[[], Startup] = neural_startup,
) -> dict[str, Startup]:
    """The shared start-up of each library the run uses, by model name."""
    models = {c.model.name for c in chosen}
    startups: dict[str, Startup] = {}
    if "lightgbm" in models:
        startups["lightgbm"] = Startup(lightgbm_seconds, LIGHTGBM_SCOPE)
    if "catboost" in models:
        startups["catboost"] = catboost(settings)
    named = [
        name
        for name in NEURAL_MODEL_NAMES
        if name in models and settings.training_device(name) == "gpu"
    ]
    if named:
        shared = neural()
        for name in named:
            startups[name] = shared
    return startups
