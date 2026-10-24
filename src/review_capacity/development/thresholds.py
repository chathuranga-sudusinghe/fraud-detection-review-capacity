"""The threshold policies R0, R1 and R2 on the saved scores of a selected model (protocol Decision
Policies; DR-003; DR-009; Issue #157).

For one configuration this reads the saved V1 and V2 scores through ``read_scores`` (which checks
the file against its record and the partition's identifier sequence) and writes

* ``r1_selection``: the R1 threshold selected on all of V2 (primary), and the same rule applied to
  the label-latency fitting days of V2 (protocol, Experimental Split Structure);
* ``evaluation`` records of R0 on V1 and V2, and of R1 and R2 on V2, each at the four capacities
  with the fixed daily budgets K_c of V2 (DR-002 §3), uncapped (DR-009);
* ``evaluation`` records of the R1 threshold of the fitting days on the held-out V2 days.

R1 and R2 thresholds are selected on V2 only, so they are not applied to V1. R2 uses scores only
and has no label-latency variant (DR-003). A selection that fails writes its ``r1_selection``
record with the recorded error first and then stops the run with a project exception.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from review_capacity.config import SECONDS_PER_DAY
from review_capacity.development.evaluation import truth_of
from review_capacity.development.inputs import ScoringInputs
from review_capacity.development.records import DevelopmentPartitions
from review_capacity.development.run_output import (
    RECORDS_DIR,
    LoadedRun,
    RunDirectory,
    source_file,
)
from review_capacity.development.scoring import (
    MODEL_ROLE,
    ScoredPartition,
    check_selected,
    read_scores,
)
from review_capacity.errors import DataValidationError, RecordContractError
from review_capacity.evaluation.ranking import Capacity, Subject, Truth, capacity_budgets
from review_capacity.evaluation.thresholds import (
    R0_THRESHOLD,
    R1Selection,
    select_r1,
    select_r2,
)
from review_capacity.evaluation.workload import (
    day_subset,
    r1_selection_record,
    threshold_outcome,
    threshold_record,
)
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import log_step
from review_capacity.models.base import Scores

logger = logging.getLogger(__name__)


def saved_partition(
    source: LoadedRun,
    inputs: ScoringInputs,
    configuration: str,
    label: str,
    *,
    rebase: bool = False,
) -> ScoredPartition:
    """The selected ``configuration``'s saved scoring of partition ``label``, from the records of
    a loaded run: its rows, the references of its score file and score record, and its time.
    With ``rebase``, the references are written under the ``source/`` link of an analysis run."""
    matches = [
        (name, record)
        for name, record in source.records
        if record["record"] == "score"
        and record["configuration"] == configuration
        and record["model_role"] == MODEL_ROLE
        and record["partition"] == label
    ]
    if len(matches) != 1:
        raise RecordContractError(
            f"{source.path}: {len(matches)} score records of the selected {configuration} on "
            f"{label}, not one"
        )
    name, record = matches[0]
    path = source.path / RECORDS_DIR / name
    reference = {"path": f"{RECORDS_DIR}/{name}", "sha256": file_sha256(path)}
    file = dict(record["file"])
    if rebase:
        file, reference = source_file(file), source_file(reference)
    return ScoredPartition(
        configuration, label, inputs.partition(label), file, reference, record["seconds"]
    )


def _subset(truth: Truth, scores: Scores, first: int, last: int) -> tuple[Truth, Scores]:
    keep = day_subset(truth.days, first, last)
    return Truth.build(truth.days[keep], truth.labels[keep], truth.amounts[keep]), scores[keep]


def _selection_failed(
    configuration: str, period: str, selection: R1Selection
) -> DataValidationError:
    return DataValidationError(
        f"{configuration}: the {period} R1 threshold cannot be selected ({selection.error}); "
        "the failure is recorded in the r1_selection record"
    )


def evaluate_thresholds(
    output: RunDirectory,
    source: LoadedRun | RunDirectory,
    partitions: DevelopmentPartitions,
    identity: Mapping[str, Any],
    seed: int,
    v1: ScoredPartition,
    v2: ScoredPartition,
) -> dict[str, Path]:
    """Write the R0, R1 and R2 records of the configuration whose saved scores are ``v1`` and
    ``v2`` (files read from ``source``) into ``output``; returns the path of each by name."""
    if (v1.label, v2.label) != ("V1", "V2") or v1.configuration != v2.configuration:
        raise DataValidationError("The threshold policies need the V1 and V2 scores of one model")
    name = v2.configuration
    check_selected(v1, seed)
    check_selected(v2, seed)
    written: dict[str, Path] = {}
    with log_step(logger, f"Evaluating the threshold policies of the selected {name}"):
        v2_scores = read_scores(source, v2.file, v2.rows)
        v1_scores = read_scores(source, v1.file, v1.rows)
        capacities = capacity_budgets(partitions.v2)

        def write(record: dict[str, Any], *parts: str) -> Path:
            return output.write_record(record, name, *parts)

        def selection_record(period: str, days: Mapping[str, int], selection: R1Selection) -> Path:
            record = r1_selection_record(identity, name, v2.record, period, days, selection)
            path = write(record, "V2", period)
            written[f"r1_selection-{period}"] = path
            return path

        def evaluate(
            subject: Subject,
            truth: Truth,
            scores: Scores,
            policy: str,
            threshold_for: Callable[[Capacity], float],
        ) -> None:
            for capacity in capacities:
                threshold = threshold_for(capacity)
                outcome = threshold_outcome(truth, scores, threshold, capacity)
                record = threshold_record(
                    identity, subject, truth, policy, threshold, capacity, outcome
                )
                part = f"{policy}-c{round(capacity.value * 100):02d}"
                if subject.period != "primary":
                    part += f"-{subject.period}"
                written[f"{subject.partition}-{part}"] = write(
                    record, MODEL_ROLE, subject.partition, part
                )

        # The primary R1 selection comes first and from the raw labels: a V2 without a fraud is a
        # recorded selection error (DR-009), not a failure to build the labels' daily counts.
        v2_days = v2.rows.times // SECONDS_PER_DAY
        primary = select_r1(v2_scores, v2.rows.labels)
        selection_record(
            "primary", {"first": int(v2_days.min()), "last": int(v2_days.max())}, primary
        )
        if primary.threshold is None:
            raise _selection_failed(name, "primary", primary)
        v1_truth, v2_truth = truth_of(v1.rows), truth_of(v2.rows)

        subject_v1 = Subject(name, MODEL_ROLE, seed, v1.record, "V1")
        subject_v2 = Subject(name, MODEL_ROLE, seed, v2.record, "V2")
        evaluate(subject_v1, v1_truth, v1_scores, "R0", lambda _: R0_THRESHOLD)
        evaluate(subject_v2, v2_truth, v2_scores, "R0", lambda _: R0_THRESHOLD)

        r1_threshold = primary.threshold
        evaluate(subject_v2, v2_truth, v2_scores, "R1", lambda _: r1_threshold)
        evaluate(
            subject_v2,
            v2_truth,
            v2_scores,
            "R2",
            lambda capacity: select_r2(v2_scores, capacity.share).threshold,
        )

        fitting, held_out = partitions.latency_ranges()
        fit_rows = day_subset(v2_truth.days, *fitting)
        latency = select_r1(v2_scores[fit_rows], v2_truth.labels[fit_rows])
        selection_record("latency", {"first": fitting[0], "last": fitting[1]}, latency)
        if latency.threshold is None:
            raise _selection_failed(name, "latency", latency)
        held_truth, held_scores = _subset(v2_truth, v2_scores, *held_out)
        subject_latency = Subject(name, MODEL_ROLE, seed, v2.record, "V2", "latency")
        latency_threshold = latency.threshold
        evaluate(subject_latency, held_truth, held_scores, "R1", lambda _: latency_threshold)
    return written
