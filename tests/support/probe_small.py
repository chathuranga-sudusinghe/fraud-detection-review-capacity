"""Small workloads for the GPU acceptance probe on the CPU (unit tests only).

``small_workloads`` replaces the probe values with tiny ones, so the whole probe runs in seconds;
``main`` runs one probe on the CPU in its own process, so that a second run is a genuinely fresh
process (check C11): ``python -m support.probe_small OUTPUT [PREVIOUS]``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from review_capacity.probe import diagnostics, runner
from review_capacity.probe.synthetic import ProbeSizes
from support.synthetic import REPO_PATHS

SIZES = ProbeSizes(
    train=400, valid=200, score=200, neural_train=256, neural_valid=128, neural_score=128
)
SMALL: dict[str, Any] = {
    "NEURAL_VARIANTS": {
        "resnet": {"small": {"n_layers": 1, "d": 16, "d_hidden_factor": 1.0}},
        "ft_transformer": {
            "small": {"n_layers": 1, "d_token": 16, "batch_size": 64},
            "padded": {"n_layers": 1, "d_token": 24, "batch_size": 64},  # heads of 3 -> 8
        },
    },
    "NEURAL_SHARED": {
        "resnet": {
            "d_embedding": 4,
            "hidden_dropout": 0.1,
            "residual_dropout": 0.1,
            "batch_size": 64,
            "eval_batch_size": 64,
        },
        "ft_transformer": {
            "d_ffn_factor": 4 / 3,
            "attention_dropout": 0.1,
            "ffn_dropout": 0.1,
            "residual_dropout": 0.1,
            "eval_batch_size": 32,
        },
    },
    "NEURAL_TRAINING": {**runner.NEURAL_TRAINING, "max_epochs": 2},
    "TREE_HYPERPARAMETERS": {
        "lightgbm": {**runner.TREE_HYPERPARAMETERS["lightgbm"], "num_iterations": 40},
        "catboost": {**runner.TREE_HYPERPARAMETERS["catboost"], "iterations": 40},
    },
    "CPU_TIMING_ROWS": 32,
}


@contextmanager
def small_workloads() -> Iterator[None]:
    """The probe's workload values replaced by small ones, restored on exit."""
    saved = {name: getattr(runner, name) for name in SMALL}
    latency = diagnostics.LATENCY_ROWS
    try:
        for name, value in SMALL.items():
            setattr(runner, name, value)
        diagnostics.LATENCY_ROWS = 4
        yield
    finally:
        for name, value in saved.items():
            setattr(runner, name, value)
        diagnostics.LATENCY_ROWS = latency


def run_small(output: Path, previous: Path | None = None) -> dict[str, Any]:
    with small_workloads():
        return runner.run_probe(REPO_PATHS, output, device="cpu", sizes=SIZES, previous=previous)


def main(arguments: list[str]) -> int:
    previous = Path(arguments[1]) if len(arguments) > 1 else None
    run_small(Path(arguments[0]), previous)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
