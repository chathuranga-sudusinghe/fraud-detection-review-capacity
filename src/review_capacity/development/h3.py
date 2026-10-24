"""The H3 configuration selection on the saved V2 results (DR-010; protocol Hypotheses and
Experimental Split Structure; Issue #159).

* ``select_primary`` reads, in the records of the open development run, each of the eight
  configurations' V2 R3 evaluation at the primary capacity and its V2 Average Precision, applies
  the rule of ``evaluation.h3`` and writes the **binding** ``h3_selection`` record. It is written
  and committed (promoted with the run's evidence) before the Test period is unsealed, and is
  final once recorded.
* ``select_latency`` repeats the rule on V2 days 127–147 only (label-latency sensitivity), from
  the saved V2 scores of the eight selected main-seed models: it writes, for each configuration,
  the R3 and ranking ``evaluation`` records of those days and then a descriptive
  ``h3_selection`` record (not binding; the H3 configuration is not changed).

Both work inside the open run that holds the evaluated records and the score files, so that the
records the selection links resolve in the run it is written to. Everything is read through
``load_run`` and ``read_scores``; no Test-period data is touched.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from review_capacity.development.evaluation import truth_of
from review_capacity.development.records import (
    CONFIGURATION_NAMES,
    DevelopmentPartitions,
)
from review_capacity.development.run_output import (
    RECORDS_DIR,
    LoadedRun,
    RunDirectory,
    load_run,
)
from review_capacity.development.scoring import (
    MODEL_ROLE,
    ScoredPartition,
    check_selected,
    read_scores,
)
from review_capacity.errors import ModelError, RecordContractError
from review_capacity.evaluation.h3 import (
    PRIMARY_CAPACITY,
    RULE_VERSION,
    H3Candidate,
    select_h3,
)
from review_capacity.evaluation.queue import DailyQueue
from review_capacity.evaluation.ranking import (
    Capacity,
    Subject,
    Truth,
    capacity_budgets,
    r3_outcome,
    r3_record,
    ranking_outcome,
    ranking_record,
)
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import log_step

logger = logging.getLogger(__name__)


def _selection_record(
    identity: Mapping[str, Any],
    period: str,
    days: Mapping[str, int],
    k: int,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    choice = select_h3(
        [
            H3Candidate(
                c["configuration"], c["frauds_captured"], c["frauds"], c["average_precision"]
            )
            for c in candidates
        ]
    )
    return {
        "record": "h3_selection",
        "schema_version": 1,
        "identity": dict(identity),
        "period": period,
        "selection_days": dict(days),
        "binding": period == "primary",
        "capacity": PRIMARY_CAPACITY,
        "k": k,
        "candidates": candidates,
        "selected": choice.selected,
        "tie_break_steps": list(choice.tie_break_steps),
        "rule_version": RULE_VERSION,
    }


def _candidate(
    configuration: str,
    r3: Mapping[str, Any],
    average_precision: float,
    evaluation: Mapping[str, str],
) -> dict[str, Any]:
    counts = r3["counts"]
    if counts["frauds"] < 1:
        raise RecordContractError(f"{configuration}: the V2 days of the selection have no fraud")
    return {
        "configuration": configuration,
        "frauds_captured": counts["true_positives"],
        "frauds": counts["frauds"],
        "recall": counts["true_positives"] / counts["frauds"],
        "average_precision": average_precision,
        "evaluation": dict(evaluation),
    }


def _only(
    source: LoadedRun, configuration: str, policy: str, **fields: Any
) -> tuple[str, dict[str, Any]]:
    matches = [
        (name, record)
        for name, record in source.records
        if record["record"] == "evaluation"
        and record["configuration"] == configuration
        and record["policy"] == policy
        and all(record[key] == value for key, value in fields.items())
    ]
    if len(matches) != 1:
        raise RecordContractError(
            f"{source.path}: {len(matches)} V2 {policy} evaluation records of the selected "
            f"{configuration}, not one"
        )
    return matches[0]


def select_primary(
    run: RunDirectory, partitions: DevelopmentPartitions, identity: Mapping[str, Any]
) -> Path:
    """Select the H3 configuration on all of V2 from the evaluation records already in ``run``
    (the open development run) and write the binding ``h3_selection`` record into it; returns its
    path. The candidates link the records they were taken from, which are in the same run."""
    v2 = partitions.v2
    source = load_run(run.path, partitions)
    capacity = _primary_capacity(partitions)
    with log_step(logger, "Selecting the H3 configuration on V2"):
        candidates = []
        for name in CONFIGURATION_NAMES:
            base = {"partition": "V2", "period": "primary", "model_role": MODEL_ROLE}
            r3_name, r3 = _only(source, name, "R3", capacity=PRIMARY_CAPACITY, **base)
            _, ranking = _only(source, name, "ranking", **base)
            average_precision = ranking["metrics"]["average_precision"]
            if average_precision is None:
                raise RecordContractError(f"{name}: the V2 ranking record has no Average Precision")
            if r3["k"] != capacity.k:
                raise RecordContractError(
                    f"{name}: the R3 evaluation uses K = {r3['k']}, not the {capacity.k} of the "
                    f"{PRIMARY_CAPACITY:.0%} capacity on V2"
                )
            if r3["counts"]["frauds"] < 1:
                raise RecordContractError(f"{name}: V2 has no fraud in its R3 evaluation record")
            reference = {
                "path": f"{RECORDS_DIR}/{r3_name}",
                "sha256": file_sha256(source.path / RECORDS_DIR / r3_name),
            }
            candidates.append(_candidate(name, r3, average_precision, reference))
        record = _selection_record(
            identity,
            "primary",
            {"first": v2.first_day, "last": v2.last_day},
            capacity.k,
            candidates,
        )
        path = run.write_record(record, "primary")
    logger.info("H3 configuration (binding): %s", record["selected"])
    return path


def _primary_capacity(partitions: DevelopmentPartitions) -> Capacity:
    return next(c for c in capacity_budgets(partitions.v2) if c.value == PRIMARY_CAPACITY)


def select_latency(
    run: RunDirectory,
    partitions: DevelopmentPartitions,
    identity: Mapping[str, Any],
    seed: int,
    scored: Mapping[str, ScoredPartition],
) -> Path:
    """Repeat the selection on V2 days 127–147 only, from the saved V2 scores ``scored`` (files in
    ``run``) of the eight selected main-seed models, writing their latency R3 and ranking
    ``evaluation`` records and the descriptive (not binding) ``h3_selection`` record into
    ``run``."""
    if list(scored) != list(CONFIGURATION_NAMES):
        raise ModelError(
            f"The latency H3 selection needs the eight configurations in order, not {list(scored)}"
        )
    fitting, _ = partitions.latency_ranges()
    capacity = _primary_capacity(partitions)
    candidates = []
    with log_step(logger, "Label-latency H3 selection on V2 days 127–147"):
        for name, v2 in scored.items():
            if v2.label != "V2" or v2.configuration != name:
                raise ModelError(f"{name}: the latency H3 selection needs its V2 scores")
            check_selected(v2, seed)
            full = truth_of(v2.rows)
            rows = (full.days >= fitting[0]) & (full.days <= fitting[1])
            truth = Truth.build(full.days[rows], full.labels[rows], full.amounts[rows])
            scores = read_scores(run, v2.file, v2.rows)[rows]
            queue = DailyQueue.build(v2.rows.ids[rows], truth.days)
            subject = Subject(name, MODEL_ROLE, seed, v2.record, "V2", "latency")
            outcome = r3_outcome(queue, truth, scores, capacity)
            r3_path = run.write_record(
                r3_record(identity, subject, truth, capacity, outcome),
                name,
                MODEL_ROLE,
                "V2",
                "R3-c02-latency",
            )
            ranking = ranking_outcome(truth, scores)
            run.write_record(
                ranking_record(identity, subject, truth, ranking),
                name,
                MODEL_ROLE,
                "V2",
                "ranking-latency",
            )
            reference = {
                "path": r3_path.relative_to(run.path).as_posix(),
                "sha256": file_sha256(r3_path),
            }
            average_precision = ranking["average_precision"]
            assert average_precision is not None
            candidates.append(
                _candidate(name, {"counts": outcome.counts}, average_precision, reference)
            )
        record = _selection_record(
            identity,
            "latency",
            {"first": fitting[0], "last": fitting[1]},
            capacity.k,
            candidates,
        )
        path = run.write_record(record, "latency")
    logger.info("H3 configuration on days %d–%d (descriptive): %s", *fitting, record["selected"])
    return path
