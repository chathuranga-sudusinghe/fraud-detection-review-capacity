"""The search plan: the candidate lists and their predicted cost, as records and Markdown
(DR-015 §1 and §7; Issue #118).

``review-capacity plan-search`` writes, into one new directory and before any model is fitted:

* ``candidates/<model>.json``: the ordered candidate list of each model, a ``candidates`` record;
* ``cost_prediction.json``: the predicted cost of those lists, a ``cost_prediction`` record;
* ``search_plan.md``: both, generated from the records for reading;
* ``plan.log``: the log of the run (opened by the command line).

The command reads no data and fits no model. Whether the predicted cost is feasible is decided
explicitly before any run (DR-015 §7); the development search refuses to start from lists that
differ from the approved plan.
"""

from __future__ import annotations

import hashlib
import logging
import platform
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from review_capacity.development.records import (
    ALL_CONFIGURATIONS,
    MODEL_OF,
    MODELS_SEARCHED,
    RECORD_CONTRACTS,
    DevelopmentPartitions,
    dump_record,
    validate_record,
)
from review_capacity.development.summary import cell, code, record_markdown
from review_capacity.errors import FileAccessError, SpecificationError
from review_capacity.files import write_atomically
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import log_step
from review_capacity.models.run_record import SettingsSnapshot
from review_capacity.provenance import code_provenance
from review_capacity.training.cost import ModelCost, predict_cost
from review_capacity.training.declaration import TrainingDeclaration
from review_capacity.training.sampler import CandidateList, generate_candidates

logger = logging.getLogger(__name__)

RUN_TYPE = "search_plan"
OUTPUT_ROOT = Path("outputs") / RUN_TYPE
LOG_NAME = "plan.log"
PLAN_NAME = "search_plan.md"
COST_NAME = "cost_prediction.json"
CANDIDATES_DIR = "candidates"


def run_id(now: datetime | None = None) -> str:
    """The UTC start time as ``YYYY-MM-DDTHHMMSSZ``."""
    return (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H%M%SZ")


def check_output(output: Path) -> None:
    """A plan is written to a directory of its own: not one that holds anything but this run's
    log file."""
    try:
        occupied = output.exists() and (
            not output.is_dir() or bool({p.name for p in output.iterdir()} - {LOG_NAME})
        )
    except OSError as error:
        raise FileAccessError(f"{output}: cannot be inspected ({error.strerror})") from error
    if occupied:
        raise SpecificationError(f"{output} already exists; use a new directory")


def identity(
    root: Path,
    started: str,
    declaration: TrainingDeclaration,
    declaration_file: SettingsSnapshot,
    settings_file: SettingsSnapshot,
    main_seed: int,
    manifest: Path,
) -> dict[str, Any]:
    """The run identity of a plan's records (``development.records``): code, hashes and seeds."""
    code_state = code_provenance(root)
    return {
        "run_id": started,
        "run_type": RUN_TYPE,
        "code_commit": code_state["git_commit"],
        "uncommitted_changes": code_state["git_uncommitted_changes"],
        "source_sha256": code_state["package"]["source_sha256"],
        "settings_sha256": settings_file.sha256,
        "declaration_sha256": declaration_file.sha256,
        "dataset_manifest_sha256": file_sha256(manifest),
        "seeds": {
            "main": main_seed,
            "additional": list(declaration.additional_seeds),
            "sampler": declaration.sampler_seed,
            "preprocessing": declaration.preprocessing_seed,
        },
    }


def configurations_of(model: str) -> list[str]:
    """The configurations that share a model's list (the unweighted one first)."""
    return [name for name in ALL_CONFIGURATIONS if MODEL_OF[name] == model]


def sampler_software() -> dict[str, str]:
    """The software that draws the values: the generator constructor, its bit generator and the
    NumPy and Python versions (the lists are not assumed to regenerate bit for bit elsewhere)."""
    bit_generator = type(np.random.default_rng(0).bit_generator).__name__
    return {
        "generator": "numpy.random.default_rng",
        "bit_generator": bit_generator,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def candidates_record(run: Mapping[str, Any], listed: CandidateList, seed: int) -> dict[str, Any]:
    """The ``candidates`` record of one model's list."""
    return {
        "record": "candidates",
        "schema_version": 1,
        "identity": dict(run),
        "model": listed.model,
        "configurations": configurations_of(listed.model),
        "stream": {"index": listed.stream_index, "spawn_key": list(listed.spawn_key)},
        "sampler_seed": seed,
        "sampler": sampler_software(),
        "trials": [
            {"trial": number, "reference": number == 1, "parameters": dict(candidate)}
            for number, candidate in enumerate(listed.candidates, start=1)
        ],
    }


def _estimate(cost: ModelCost) -> dict[str, Any]:
    return {
        "model": cost.model,
        "estimated": cost.estimated,
        "basis": cost.basis,
        "epochs_low": cost.epochs_low,
        "epochs_high": cost.epochs_high,
        "candidate_epoch_minutes": list(cost.candidate_epoch_minutes or ()),
        "search_hours_low": cost.search_hours_low,
        "search_hours_high": cost.search_hours_high,
        "refit_hours": cost.refit_hours,
    }


def cost_record(
    run: Mapping[str, Any], costs: tuple[ModelCost, ...], files: Mapping[str, dict[str, str]]
) -> dict[str, Any]:
    """The ``cost_prediction`` record; ``files`` links each model's candidate record."""
    estimated = [c for c in costs if c.estimated]
    return {
        "record": "cost_prediction",
        "schema_version": RECORD_CONTRACTS["cost_prediction"].version,
        "identity": dict(run),
        "candidates": dict(files),
        "estimates": [_estimate(c) for c in costs],
        "totals": {
            "search_hours_low": sum(c.search_hours_low or 0.0 for c in estimated),
            "search_hours_high": sum(c.search_hours_high or 0.0 for c in estimated),
            "refit_hours": sum(c.refit_hours or 0.0 for c in estimated),
        },
        "decision": "pending_approval",
    }


def _table(header: list[str], body: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in body),
    ]


def plan_markdown(
    candidates: Mapping[str, Mapping[str, Any]], cost: Mapping[str, Any], files: Mapping[str, Any]
) -> str:
    """The plan for reading, generated from its records: the run identity once (it is the same in
    every record), then every field of every record, with one column per searched parameter in
    each candidate table, and a short cost summary."""
    run = cost["identity"]
    lines = [
        "# Search plan: candidate lists and predicted cost",
        "",
        "Generated by `review-capacity plan-search` from the records of this directory; not "
        "edited by hand. No model was fitted and no data were read. Whether the predicted cost "
        "is feasible is decided explicitly before any run (DR-015 §7).",
        "",
        "## Run identity",
        "",
        "The same in every record of this plan.",
        "",
        *_table(
            ["Field", "Value"],
            [[code(f"identity.{path}"), cell(value)] for path, value in _identity_rows(run)],
        ),
    ]
    for model in MODELS_SEARCHED:
        record = candidates[model]
        lines += [
            "",
            f"## {model}",
            "",
            f"Record {code(files[model]['path'])}, SHA-256 {code(files[model]['sha256'])}.",
            "",
            record_markdown(record, omit=("identity",)).rstrip("\n"),
        ]
    lines += [
        "",
        "## Predicted cost",
        "",
        "An estimate under the stated basis (DR-015 §7), not a bound. Hours are for both "
        "configurations of a model (the shared list is fitted twice); refit hours are for the "
        "costliest candidate of the list at the maximum length, because the selected candidate "
        "is unknown before the search. Models without a cost model in DR-015 §7 are not "
        "estimated.",
        "",
        *_table(
            ["Model", "Estimated", "Epochs", "Search h low", "Search h high", "Refit h"],
            [
                [
                    e["model"],
                    cell(e["estimated"]),
                    f"{cell(e['epochs_low'])} to {cell(e['epochs_high'])}",
                    cell(e["search_hours_low"]),
                    cell(e["search_hours_high"]),
                    cell(e["refit_hours"]),
                ]
                for e in cost["estimates"]
            ]
            + [
                [
                    "**Total**",
                    "",
                    "",
                    cell(cost["totals"]["search_hours_low"]),
                    cell(cost["totals"]["search_hours_high"]),
                    cell(cost["totals"]["refit_hours"]),
                ]
            ],
        ),
        "",
        "### Record `cost_prediction`",
        "",
        record_markdown(cost, omit=("identity",)).rstrip("\n"),
        "",
    ]
    return "\n".join(lines)


def _identity_rows(run: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """The run identity as dotted paths and values (the seeds spread over their names)."""
    rows: list[tuple[str, Any]] = []
    for name, value in run.items():
        if isinstance(value, Mapping):
            rows += [(f"{name}.{key}", item) for key, item in value.items()]
        else:
            rows.append((name, value))
    return rows


def _write(output: Path, relative: str, text: str) -> dict[str, str]:
    """Write ``text`` to ``output / relative``; its file reference for other records."""
    write_atomically(output / relative, lambda target: target.write_text(text, encoding="utf-8"))
    return {"path": relative, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def write_plan(
    output: Path,
    started: str,
    run: Mapping[str, Any],
    declaration: TrainingDeclaration,
    partitions: DevelopmentPartitions,
) -> dict[str, Any]:
    """Generate the lists and their cost, validate every record and write the plan into the new
    directory ``output``; returns the cost record."""
    with log_step(logger, f"Writing the search plan {started}"):
        check_output(output)
        lists = generate_candidates(declaration)
        records: dict[str, dict[str, Any]] = {}
        files: dict[str, dict[str, str]] = {}
        for model in MODELS_SEARCHED:
            record = candidates_record(run, lists[model], declaration.sampler_seed)
            text = dump_record(record, partitions)
            relative = f"{CANDIDATES_DIR}/{model}.json"
            files[model] = _write(output, relative, text)
            records[model] = record
        costs = predict_cost(declaration, lists)
        cost = cost_record(run, costs, files)
        validate_record(cost, partitions)
        _write(output, COST_NAME, dump_record(cost, partitions))
        _write(output, PLAN_NAME, plan_markdown(records, cost, files))
    for estimate in cost["estimates"]:
        if estimate["estimated"]:
            logger.info(
                "%s: predicted search %.1f to %.1f hours, refits up to %.1f hours",
                estimate["model"],
                estimate["search_hours_low"],
                estimate["search_hours_high"],
                estimate["refit_hours"],
            )
    logger.info("Decision pending: whether this cost is feasible is decided explicitly (DR-015 §7)")
    return cost
