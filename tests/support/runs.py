"""A synthetic development run directory for the output, summary and promotion tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from review_capacity.development.rendering import write_summary
from review_capacity.development.run_output import RunDirectory
from support.development import IDENTITY, PARTITIONS, example

RUN_ID: str = IDENTITY["run_id"]


def new_run(root: Path) -> RunDirectory:
    """A newly created run directory under ``root`` (the dated default)."""
    return RunDirectory.create(root, PARTITIONS, "development", RUN_ID)


SEED_RECALLS = {43: 0.3, 44: 0.35}


def write_records(run: RunDirectory, seed_recalls: Mapping[int, float] = SEED_RECALLS) -> None:
    """The ``run`` record first, then a trial, its configuration, V2 evaluations of the
    selected model and of the additional seeds ``seed_recalls`` (seed -> Recall@2%), and a
    timing."""
    run.write_record(example("run"))
    run.write_record(example("trial"), "LGB-u", 1)
    run.write_record(example("configuration"), "LGB-u")
    run.write_record(example("evaluation"), "LGB-u", "selected")
    for seed, recall in seed_recalls.items():
        refit = example("evaluation") | {
            "model_role": "seed_refit",
            "seed": seed,
            "metrics": example("evaluation")["metrics"] | {"recall": recall},
        }
        run.write_record(refit, "LGB-u", f"seed{seed}")
    run.write_record(example("timing"), "RES-u", "latency")


def finished_run(root: Path) -> RunDirectory:
    """A completed, closed run with its summary and log, as a run leaves it."""
    run = new_run(root)
    run.log_file.write_text("log of the run\n", encoding="utf-8")
    with run.closing(lambda path: write_summary(path, PARTITIONS)):
        write_records(run)
    return run
