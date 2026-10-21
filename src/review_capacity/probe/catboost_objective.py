"""CatBoost tree-objective acceptance: checks G1–G7 of ``docs/gpu-acceptance-probe.md``, Section 8
(Issue #123).

One CatBoost fit (``Cat-u``) on the synthetic tree inputs of the probe (Section 1), at the largest
depth of the search space and the maximum length of DR-015 §2, with the shared V1 objective as the
user-defined evaluation metric on the GPU learner and an overfitting detector that cannot stop
the fit before the maximum. No research data are read.

``run_acceptance`` writes ``acceptance.json`` and the generated ``acceptance.md`` before the fit,
after it and on any failure. A GPU run at depth 10 and 2,000 iterations is the acceptance run; a
CPU run (the unit tests of this module) runs the same code on small inputs and is marked as not
being acceptance evidence. There is no fallback: a failure of the GPU learner or of the metric is
recorded and raised.
"""

from __future__ import annotations

import gc
import logging
import os
import statistics
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from review_capacity.config import SECONDS_PER_DAY, Device, load_feature_spec
from review_capacity.data.catalog import read_catalog
from review_capacity.errors import FileAccessError, ModelError
from review_capacity.files import write_atomically, write_json
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import log_step
from review_capacity.models import catboost_model
from review_capacity.models.base import Keys
from review_capacity.models.catboost_model import CatBoostModel
from review_capacity.models.device import environment_record
from review_capacity.models.registry import Weighting, configuration
from review_capacity.models.run_record import jsonable, load_settings_with_snapshot
from review_capacity.models.v1_objective import OBJECTIVE_NAME, TrainingHistory, V1Objective
from review_capacity.paths import ProjectPaths
from review_capacity.probe import diagnostics
from review_capacity.probe.records import ACCEPTANCE_ENTRIES, CheckBook, check_acceptance_output
from review_capacity.probe.runner import FAILING_WARNING, MODEL_SEED, captured_warnings
from review_capacity.probe.synthetic import (
    DATA_SEED,
    ProbeSizes,
    category_levels,
    data_generator,
    tree_data,
)
from review_capacity.provenance import code_provenance

logger = logging.getLogger(__name__)

SPECIFICATION = "docs/gpu-acceptance-probe.md"
RECORD_NAME, SUMMARY_NAME = ACCEPTANCE_ENTRIES
SCHEMA_VERSION = 1
DEPTH = 10  # the largest depth of the search space (DR-015 §6)
ITERATIONS = 2000  # the maximum number of boosting rounds (DR-015 §2)
DAILY_SHARE = 0.02  # the share of V1 volume per day that sets the V1 budget (DR-015 §3)
NUMBA_NOTICE = "Failed to import numba for optimizing custom metrics and objectives"
CHECKS: dict[str, str] = {
    "G1": "Device and completion",
    "G2": "Length",
    "G3": "Objective correctness",
    "G4": "Resolved settings",
    "G5": "Persistence",
    "G6": "Warnings",
    "G7": "Resources and timing",
}


def hyperparameters(depth: int, iterations: int) -> dict[str, Any]:
    """The probe values of Section 8 [I]: the reference candidate of DR-015 §6 at ``depth``, with
    ``od_wait`` equal to ``iterations`` so that the overfitting detector cannot stop the fit."""
    return {
        "iterations": iterations,
        "od_wait": iterations,
        "eval_metric": OBJECTIVE_NAME,
        "boost_from_average": False,
        "metric_period": 1,
        "learning_rate": 0.03234,
        "depth": depth,
        "l2_leaf_reg": 3.0,
        "bagging_temperature": 1.0,
        "leaf_estimation_iterations": 10,
        "model_size_reg": 0.5,
        "fold_permutation_block": 1,
        "permutation_count": 4,
        "random_strength": 1.0,
    }


def evidence_blockers(
    device: Device,
    depth: int,
    iterations: int,
    sizes: ProbeSizes,
    code: dict[str, Any] | None,
    root: Path,
) -> list[str]:
    """Why a run is not the acceptance run of Section 8 (empty for the acceptance run): the GPU,
    depth 10, 2,000 rounds, the synthetic Train and V1 sizes of Section 1, and a clean committed
    tree whose code is the code that runs (the running package lies inside the inspected checkout
    ``root``, and its commit is recorded). Any other run is exploratory."""
    required = ProbeSizes()
    blockers = []
    if device != "gpu":
        blockers.append(f"device {device}, not gpu")
    if depth != DEPTH:
        blockers.append(f"depth {depth}, not {DEPTH}")
    if iterations != ITERATIONS:
        blockers.append(f"{iterations} rounds, not {ITERATIONS}")
    if (sizes.train, sizes.valid) != (required.train, required.valid):
        blockers.append(
            f"{sizes.train} Train and {sizes.valid} V1 rows, not {required.train} and "
            f"{required.valid}"
        )
    code = code or {}
    if code.get("git_uncommitted_changes") is not False:
        blockers.append("the working tree is not shown to be clean and committed")
    if not code.get("git_commit"):
        blockers.append("the commit is not recorded")
    location = (code.get("package") or {}).get("location")
    if not location or not Path(location).resolve().is_relative_to(root.resolve()):
        blockers.append("the running package is not the inspected checkout's")
    return blockers


def daily_budget(times: Keys, rows: int) -> tuple[int, int]:
    """The daily budget K of the V1 objective and the number of V1 days: ``round(0.02 × rows /
    days)`` (DR-015 §3), at least 1 (only the tiny partitions of the unit tests reach that)."""
    days = len(np.unique(times // SECONDS_PER_DAY))
    return max(1, round(DAILY_SHARE * rows / days)), days


def _check_resolved(model: CatBoostModel, fixed: dict[str, Any]) -> str | None:
    """The adapter's own comparison of every fixed setting with the resolved record and the
    model's system options; the reason it gives when they differ."""
    parameters = model.parameters
    try:
        catboost_model._check_resolved(parameters["resolved"], fixed)
        catboost_model._check_system(parameters["metadata"], fixed)
    except ModelError as error:
        return str(error)
    return None


def _length_item(
    book: CheckBook, model: CatBoostModel, history: TrainingHistory, rounds: int
) -> None:
    evaluated = model._fitted().get_evals_result()["validation"]
    lengths = sorted({len(values) for values in evaluated.values()})
    book.add(
        "G2",
        "Cat-u",
        history.stopped_at == rounds
        and len(history.checkpoints) == rounds
        and history.stopping_reason == "maximum"
        and lengths == [rounds]
        and model.fitted_iterations == history.best_checkpoint,
        rounds_requested=rounds,
        rounds_trained=history.stopped_at,
        v1_evaluations=len(history.checkpoints),
        stopping_reason=history.stopping_reason,
        catboost_evaluation_lengths=lengths,
        kept_trees=model.fitted_iterations,
        best_checkpoint=history.best_checkpoint,
    )


def _objective_item(
    book: CheckBook,
    model: CatBoostModel,
    history: TrainingHistory,
    objective: V1Objective,
    valid: Any,
) -> None:
    scores = model.predict(valid.features)
    recomputed = objective.queue.recall(scores, valid.labels, objective.k)
    recorded = history.checkpoints[history.best_checkpoint - 1].numerator
    book.add(
        "G3",
        "Cat-u",
        recomputed.numerator == recorded,
        recomputed_numerator=recomputed.numerator,
        recorded_numerator=recorded,
        frauds=recomputed.frauds,
    )
    book.add("G3", "Cat-u other evaluations", None, count=history.other_evaluations)


def _resolved_item(
    book: CheckBook, model: CatBoostModel, fixed: dict[str, Any], depth: int, device: Device
) -> None:
    parameters = model.parameters
    resolved = parameters["resolved"]
    difference = _check_resolved(model, fixed)
    expected_task = "GPU" if device == "gpu" else "CPU"
    task_type = resolved.get("task_type")
    book.add("G1", "Cat-u", task_type == expected_task, task_type=task_type)
    book.add(
        "G4",
        "Cat-u",
        resolved.get("depth") == depth
        and resolved.get("boost_from_average") is False
        and difference is None,
        depth=resolved.get("depth"),
        boost_from_average=resolved.get("boost_from_average"),
        fixed_settings_compared=len(fixed),
        difference=difference,
    )
    book.add(
        "G4",
        "Cat-u sampling_unit",
        None,
        note=(
            "CatBoost 1.2.10 does not serialise sampling_unit; the requested value Object is "
            "its documented default (Section 7, disposition of 15 October 2026)"
        ),
    )


def _persistence_item(
    book: CheckBook,
    model: CatBoostModel,
    settings: Any,
    features: Any,
    device: Device,
    valid: Any,
    identity: dict[str, Any],
) -> None:
    scores = model.predict(valid.features)
    with tempfile.TemporaryDirectory() as directory:
        model.save(Path(directory))
        loaded = CatBoostModel.load(
            Path(directory),
            model.configuration,
            model.seed,
            settings=settings,
            features=features,
            device=device,
        )
        reloaded = loaded.predict(valid.features)
    requested = loaded.parameters["requested"]
    book.add(
        "G5",
        "Cat-u",
        diagnostics.bitwise_equal(scores, reloaded)
        and requested.get(OBJECTIVE_NAME) == identity
        and requested.get("eval_metric") == OBJECTIVE_NAME,
        scores_identical=diagnostics.bitwise_equal(scores, reloaded),
        identity_recorded=requested.get(OBJECTIVE_NAME) == identity,
        eval_metric=requested.get("eval_metric"),
    )


def classify_warnings(capture: Any, book: CheckBook) -> None:
    """G6: the numba notice, by its text, concerns the speed of the metric only; any other warning
    of an ignored setting, an unsupported capability or a fallback fails; any other warning needs a
    recorded review."""
    for item in capture.records:
        text = item["text"]
        outcome: bool | str
        if NUMBA_NOTICE in text:
            outcome = True
        elif FAILING_WARNING.search(text):
            outcome = False
        else:
            outcome = "review"
        book.add("G6", item["phase"], outcome, logger=item["logger"], text=text)
    if not capture.records:
        book.add("G6", "whole run", True, warnings=0)


def _resources_item(
    book: CheckBook,
    sampler: diagnostics.GpuMemorySampler | None,
    history: TrainingHistory | None,
    fit_seconds: float | None,
) -> None:
    """G7: no out-of-memory error (the fit finished) and, on the GPU, valid whole-device memory
    monitoring over the fit; every measurement is recorded."""
    monitored: dict[str, Any] = {"monitored": False}
    valid = True
    if sampler is not None:
        snapshot = sampler.snapshot()
        valid = bool(
            not snapshot["problems"]
            and snapshot["samples"] > 1
            and snapshot["total_mib"] is not None
            and snapshot["baseline_used_mib"] is not None
        )
        monitored = {"monitored": True, **snapshot}
    increments: dict[str, float | None] = {"first": None, "median_increase": None, "last": None}
    if history is not None and history.checkpoints:
        seconds = [row.seconds for row in history.checkpoints]
        steps = [b - a for a, b in pairwise(seconds)]
        increments = {
            "first": seconds[0],
            "median_increase": statistics.median(steps) if steps else None,
            "last": seconds[-1],
        }
    book.add(
        "G7",
        "Cat-u",
        valid,
        fit_seconds=fit_seconds,
        checkpoint_seconds=increments,
        host_peak_bytes=diagnostics.host_peak_bytes(),
        **monitored,
    )


def run_acceptance(
    paths: ProjectPaths,
    output: Path,
    *,
    device: Device = "gpu",
    sizes: ProbeSizes | None = None,
    depth: int = DEPTH,
    iterations: int = ITERATIONS,
    log_opened: bool = False,
) -> dict[str, Any]:
    """Run the acceptance, write its records to ``output`` at every stage and on any failure, and
    return the final record."""
    sizes = sizes or ProbeSizes()
    output = output.resolve()
    check_acceptance_output(output, log_opened=log_opened)
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise FileAccessError(f"{output}: cannot be created ({error.strerror})") from error
    specification = paths.root / SPECIFICATION
    params = hyperparameters(depth, iterations)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "specification": {"path": SPECIFICATION, "sha256": None},
        "settings": None,
        "run": {
            "started": datetime.now(UTC).isoformat(timespec="seconds"),
            "finished": None,
            "stage": "start",
            "error": None,
            "device": device,
            "acceptance_evidence": False,  # decided once the code provenance is known
            "evidence_blockers": ["start-up not finished"],
            "pid": os.getpid(),
            "data_seed": DATA_SEED,
            "model_seed": MODEL_SEED,
            "sizes": asdict(sizes),
            "hyperparameters": params,
            "daily_k": None,
            "v1_days": None,
        },
        "code": None,
        "environment": None,
        "synthetic": {},
        "fit": None,
        "history": None,
        "gpu_memory": None,
        "host_peak_bytes": None,
        "checks": CheckBook(CHECKS).to_json(),
        "overall": "in progress",
    }
    book = CheckBook(CHECKS)
    sampler: diagnostics.GpuMemorySampler | None = None
    history: TrainingHistory | None = None
    fit_seconds: float | None = None
    classified = False

    def write(stage: str) -> dict[str, Any]:
        record["run"]["stage"] = stage
        record["checks"] = book.to_json()
        record["gpu_memory"] = None if sampler is None else sampler.snapshot()
        record["host_peak_bytes"] = diagnostics.host_peak_bytes()
        written: dict[str, Any] = jsonable(record)  # type: ignore[assignment]
        write_json(output / RECORD_NAME, written)
        link = Path(os.path.relpath(specification, output)).as_posix()
        text = summary_markdown(written, link)
        write_atomically(output / SUMMARY_NAME, lambda target: target.write_text(text, "utf-8"))
        return written

    with captured_warnings() as capture:
        try:
            record["specification"]["sha256"] = file_sha256(specification)
            settings, snapshot = load_settings_with_snapshot(paths)
            record["settings"] = {"path": snapshot.path, "sha256": snapshot.sha256}
            record["code"] = code_provenance(paths.root)
            blockers = evidence_blockers(
                device, depth, iterations, sizes, record["code"], paths.root
            )
            record["run"]["evidence_blockers"] = blockers
            record["run"]["acceptance_evidence"] = not blockers
            record["environment"] = environment_record()
            features = load_feature_spec(paths.features_spec)
            levels = category_levels(read_catalog(paths.feature_catalog), features)
            with diagnostics.sampled_gpu_memory(device == "gpu") as running:
                sampler = running
                with log_step(logger, "Acceptance synthetic inputs"):
                    tree = tree_data(features, levels, sizes, data_generator())
                train = tree.train[Weighting.UNWEIGHTED]
                valid = tree.valid[Weighting.UNWEIGHTED]
                if valid.times is None:
                    raise ModelError("The synthetic V1 partition has no row times")
                k, days = daily_budget(valid.times, len(valid.labels))
                record["run"]["daily_k"], record["run"]["v1_days"] = k, days
                record["synthetic"] = {"tree_sha256": tree.sha256, "tree": tree.diagnostics}
                objective = V1Objective(valid, k)
                model = CatBoostModel(
                    configuration("Cat-u"),
                    MODEL_SEED,
                    settings=settings,
                    features=features,
                    device=device,
                )
                capture.phase = f"Cat-u {device} fit"
                write(f"fitting Cat-u on {device}")  # a native abort leaves the running fit
                start = time.perf_counter()
                with log_step(logger, f"Acceptance fit: Cat-u, depth {depth}, {iterations} rounds"):
                    model.fit(train, valid, params, objective=objective)
                fit_seconds = time.perf_counter() - start
                history = model.training_history
                if history is None:
                    raise ModelError("The fit recorded no V1 history")
                record["fit"] = {
                    "device": device,
                    "kept": model.fitted_iterations,
                    "seconds": fit_seconds,
                    "parameters": jsonable(model.parameters),
                }
                record["history"] = jsonable(asdict(history))
                write("fitted")
                capture.phase = "checks"
                _resolved_item(book, model, model._fixed_parameters(), depth, device)
                _length_item(book, model, history, iterations)
                _objective_item(book, model, history, objective, valid)
                _persistence_item(
                    book, model, settings, features, device, valid, objective.identity
                )
                del model, train, tree
                gc.collect()
            _resources_item(book, sampler, history, fit_seconds)
            classify_warnings(capture, book)
            classified = True
            record["run"]["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
            record["overall"] = book.overall(compared=True)
            final = write("finished")
        except BaseException as error:
            if not classified:
                classify_warnings(capture, book)
            interrupted = isinstance(error, KeyboardInterrupt)
            record["run"]["error"] = f"{type(error).__name__}: {error}"
            record["overall"] = "interrupted" if interrupted else "failed"
            try:
                write("interrupted" if interrupted else "failed")
            except Exception:  # the original error is the one that is raised
                logger.exception("The acceptance record could not be written after the failure")
            raise
    logger.info("CatBoost tree-objective acceptance: %s (records in %s)", final["overall"], output)
    return final


def _cell(value: object) -> str:
    text = "—" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _scalar(value: object) -> str:
    """A value as text, with every digit of a float (its shortest round-trip form)."""
    return repr(value) if isinstance(value, float) else _cell(value)


def _flatten(value: object, prefix: str = "") -> list[tuple[str, str]]:
    """The leaves of a recorded value as (path, text) rows: dictionaries by key, lists of
    scalars as one row, other lists by index."""
    if isinstance(value, dict):
        if not value:
            return [(prefix, "{}")]
        rows: list[tuple[str, str]] = []
        for key, item in value.items():
            rows += _flatten(item, f"{prefix}.{key}" if prefix else str(key))
        return rows
    if isinstance(value, list):
        if all(not isinstance(item, dict | list) for item in value):
            return [(prefix, ", ".join(_scalar(item) for item in value) or "[]")]
        rows = []
        for index, item in enumerate(value):
            rows += _flatten(item, f"{prefix}[{index}]")
        return rows
    return [(prefix, _scalar(value))]


def _record_section(title: str, value: object) -> list[str]:
    """A section listing every recorded field of ``value``."""
    lines = ["", f"## {title}", ""]
    if value is None:
        return [*lines, "Not recorded."]
    lines += ["| Field | Value |", "|---|---|"]
    lines += [f"| `{path}` | {text} |" for path, text in _flatten(value)]
    return lines


def summary_markdown(record: dict[str, Any], specification_link: str) -> str:
    """The run summary generated from ``acceptance.json``, never written by hand: the checks and
    their items, every checkpoint of the V1 history, and every other recorded field, with floats
    in their full round-trip form."""
    run = record["run"]
    code: dict[str, Any] = record.get("code") or {}
    settings: dict[str, Any] = record.get("settings") or {}
    history: dict[str, Any] = record.get("history") or {}
    fit: dict[str, Any] = record.get("fit") or {}
    blockers = run.get("evidence_blockers") or []
    verdict = (
        ""
        if run["acceptance_evidence"]
        else " This run is not acceptance evidence: " + "; ".join(blockers) + "."
    )
    lines = [
        "# CatBoost tree-objective acceptance: run summary",
        "",
        "Generated by `review-capacity probe-catboost-objective` from `acceptance.json`; not "
        "edited by hand. The checks (G1–G7) and criteria are those of Section 8 of "
        f"[`docs/gpu-acceptance-probe.md`]({specification_link}).",
        "",
        f"**Overall result: {record['overall']}.**{verdict}",
        "",
        "## Run identity",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Record schema version | {record['schema_version']} |",
        f"| Code commit | `{_cell(code.get('git_commit'))}` |",
        f"| Uncommitted changes | {_cell(code.get('git_uncommitted_changes'))} |",
        f"| Specification SHA-256 | `{_cell(record['specification'].get('sha256'))}` |",
        f"| Settings SHA-256 | `{_cell(settings.get('sha256'))}` |",
        f"| Data seed; model seed | {run['data_seed']}; {run['model_seed']} |",
        f"| Device | {run['device']} |",
        f"| Depth; rounds; `od_wait` | {run['hyperparameters']['depth']}; "
        f"{run['hyperparameters']['iterations']}; {run['hyperparameters']['od_wait']} |",
        f"| V1 daily budget K; V1 days | {_cell(run.get('daily_k'))}; "
        f"{_cell(run.get('v1_days'))} |",
        f"| Started; finished | {run['started']}; {_cell(run.get('finished'))} |",
        f"| Error | {_cell(run.get('error'))} |",
        "",
        "## Checks",
        "",
        "| Check | Title | Status | Items | Failed | For review |",
        "|---|---|---|---|---|---|",
    ]
    for check, entry in record["checks"].items():
        items = entry["items"]
        failed = sum(item["outcome"] is False for item in items)
        review = sum(item["outcome"] == "review" for item in items)
        lines.append(
            f"| {check} | {entry['title']} | {entry['status']} | {len(items)} | {failed} | "
            f"{review} |"
        )
    lines += ["", "## Items of every check", "", "| Check | Subject | Outcome | Details |"]
    lines.append("|---|---|---|---|")
    for check, entry in record["checks"].items():
        for item in entry["items"]:
            details = {k: v for k, v in item.items() if k not in ("subject", "outcome")}
            outcome = "descriptive" if item["outcome"] is None else item["outcome"]
            lines.append(
                f"| {check} | {_cell(item['subject'])} | {_cell(outcome)} | {_cell(details)} |"
            )
    result = {k: v for k, v in history.items() if k != "checkpoints"}
    lines += _record_section("V1 history (result)", result or None)
    lines += ["", "## V1 objective after every round", ""]
    checkpoints = history.get("checkpoints")
    if checkpoints is None:
        lines.append("Not recorded.")
    else:
        lines += [
            "| Round | Numerator | V1 objective | Improved | Earliest best so far | Seconds |",
            "|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {row['checkpoint']} | {row['numerator']} | {_scalar(row['objective'])} | "
            f"{row['improved']} | {row['best_checkpoint']} | {_scalar(row['seconds'])} |"
            for row in checkpoints
        ]
    lines += _record_section("Fit", fit or None)
    lines += _record_section("Run record", {k: v for k, v in run.items() if k != "hyperparameters"})
    lines += _record_section("Hyperparameters", run["hyperparameters"])
    identity = {"specification": record["specification"], "settings": record.get("settings")}
    lines += _record_section("Specification and settings", identity)
    software = {"code": record.get("code"), "environment": record.get("environment")}
    lines += _record_section("Code, packages and environment", software)
    lines += _record_section("Synthetic inputs", record.get("synthetic") or None)
    lines += _record_section(
        "GPU memory and host memory",
        {"gpu_memory": record.get("gpu_memory"), "host_peak_bytes": record.get("host_peak_bytes")},
    )
    lines += [
        "",
        "## Symbols and notation",
        "",
        "| Symbol / abbreviation | Meaning |",
        "|---|---|",
        "| G1–G7 | Checks of the CatBoost tree-objective acceptance (Section 8 of the "
        "specification) |",
        "| K | Daily review budget of the V1 objective: round(0.02 × V1 rows / V1 days), at "
        "least 1 |",
        "| V1 | Validation 1 (here the synthetic early-stopping partition) |",
        "| `od_wait` | CatBoost's overfitting-detector wait (rounds without improvement) |",
        "| Numerator | Frauds inside the daily review queues after a round (the objective is "
        "numerator / frauds) |",
        "| MiB | Mebibyte (2^20 bytes) |",
        "| GPU | Graphics processing unit |",
        "| SHA-256 | Secure Hash Algorithm, 256-bit |",
        "",
    ]
    return "\n".join(lines)
