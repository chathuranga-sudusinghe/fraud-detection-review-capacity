"""The outcome records of the Test evaluation, from the saved scores, the labels and the frozen
inputs (Issue #205; the functions of Issues #197, #199 and #201).

``build_outcome_records`` is pure: it returns the ``test_evaluation`` (the eight configurations
and the reference), ``test_workload``, ``test_calibration``, ``test_rq3`` and
``test_false_negative`` records in the order of their publication, each with the name parts of its
file. ``publish_outcomes`` writes them; a record that exists already must be the one that would be
written, so that the publication of an interrupted attempt is completed, never altered.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy.typing as npt

from review_capacity.confirmatory.frozen_policies import METHODS, FrozenPolicies
from review_capacity.confirmatory.records import contract_versions, to_json_value
from review_capacity.confirmatory.slices import SliceInputs
from review_capacity.development.records import ALL_CONFIGURATIONS, CONFIGURATION_NAMES
from review_capacity.development.run_output import RunDirectory
from review_capacity.errors import SpecificationError
from review_capacity.evaluation.declaration import EvaluationDeclaration
from review_capacity.evaluation.false_negatives import false_negative_analysis, misses_of
from review_capacity.evaluation.fidelity import fidelity
from review_capacity.evaluation.outcomes import (
    Partition,
    calibration_outcome,
    confirmatory_outcomes,
    secondary_resamples,
)
from review_capacity.evaluation.ranking import Capacity
from review_capacity.evaluation.rq3 import CostSubject, rq3_outcomes
from review_capacity.logging_setup import log_step

logger = logging.getLogger(__name__)

type Parts = tuple[str, ...]
type Named = tuple[Parts, dict[str, Any]]


@dataclass(frozen=True)
class OutcomeSetup:
    """What every outcome record repeats: the run identity, the Test days and, for each
    configuration, the SHA-256 of its saved score file."""

    declaration: EvaluationDeclaration
    identity: Mapping[str, Any]
    first_day: int
    last_day: int
    score_files: Mapping[str, str]
    reference_volume: float

    def envelope(self, kind: str) -> dict[str, Any]:
        return {
            "record": kind,
            "schema_version": contract_versions()[kind],
            "identity": dict(self.identity),
        }

    @property
    def days(self) -> dict[str, int]:
        return {"first": self.first_day, "last": self.last_day}


def capacities_of(declaration: EvaluationDeclaration) -> tuple[Capacity, ...]:
    return tuple(
        Capacity(share, budget)
        for share, budget in zip(
            declaration.capacity.shares, declaration.capacity.budgets, strict=True
        )
    )


def primary_capacity(declaration: EvaluationDeclaration) -> Capacity:
    primary = declaration.capacity.primary
    return next(c for c in capacities_of(declaration) if c.share == primary)


def _evaluation(
    setup: OutcomeSetup, name: str, policies: FrozenPolicies, outcome: Mapping[str, Any]
) -> Named:
    thresholds = policies.thresholds.get(name)
    return (name,), to_json_value(
        {
            **setup.envelope("test_evaluation"),
            "configuration": name,
            "partition": "Test",
            "days": setup.days,
            "scores_sha256": setup.score_files[name],
            "thresholds": None
            if thresholds is None
            else {
                "r0": thresholds.r0,
                "r1": thresholds.r1,
                "r2": [{"capacity": c, "threshold": t} for c, t in thresholds.r2.items()],
            },
            "ranking": outcome["ranking"],
            "r3": outcome["r3"],
            "policies": outcome["thresholds"],
        }
    )


def _workload(setup: OutcomeSetup, name: str, outcome: Mapping[str, Any]) -> Named:
    entries = [
        {
            "capacity": entry["capacity"],
            "k": entry["k"],
            "threshold": entry["threshold"],
            "fidelity": fidelity(
                entry["daily"], entry["capacity"], entry["k"], setup.reference_volume
            ),
        }
        for entry in outcome["thresholds"]
        if entry["policy"] == "R2"
    ]
    return (name,), to_json_value(
        {
            **setup.envelope("test_workload"),
            "configuration": name,
            "partition": "Test",
            "days": setup.days,
            "entries": entries,
        }
    )


def build_outcome_records(
    setup: OutcomeSetup,
    partition: Partition,
    scores: Mapping[str, npt.NDArray[Any]],
    policies: FrozenPolicies,
    slices: SliceInputs,
) -> list[Named]:
    """Every outcome record of the Test evaluation, in the order of their publication."""
    declaration = setup.declaration
    capacities = capacities_of(declaration)
    primary = primary_capacity(declaration)
    days = len(partition.truth.day_values)
    missing = sorted(set(ALL_CONFIGURATIONS) - set(scores))
    if missing:
        raise SpecificationError(f"The saved scores lack {missing}")
    with log_step(logger, "Test outcome records"):
        outcomes = confirmatory_outcomes(
            declaration,
            partition,
            {name: scores[name] for name in ALL_CONFIGURATIONS},
            dict(policies.thresholds),
            capacities,
        )
        records: list[Named] = [
            _evaluation(setup, name, policies, outcomes[name]) for name in ALL_CONFIGURATIONS
        ]
        records += [_workload(setup, name, outcomes[name]) for name in CONFIGURATION_NAMES]
        resamples = secondary_resamples(declaration, days)
        for name in CONFIGURATION_NAMES:
            for method in METHODS:
                result = calibration_outcome(
                    declaration,
                    partition,
                    scores[name],
                    policies.calibrators[(name, method)],
                    resamples,
                )
                intervals = result.pop("intervals")
                records.append(
                    (
                        (name, method),
                        to_json_value(
                            {
                                **setup.envelope("test_calibration"),
                                "configuration": name,
                                "method": method,
                                "role": "primary" if method == "platt" else "sensitivity",
                                "partition": "Test",
                                "days": setup.days,
                                "calibrator": dict(policies.calibrator_files[(name, method)]),
                                "diagnostic": result,
                                "intervals": intervals,
                            }
                        ),
                    )
                )
        subjects = [
            CostSubject(
                name,
                method,
                scores[name],
                policies.calibrators[(name, method)].predict(scores[name]),
            )
            for name in CONFIGURATION_NAMES
            for method in METHODS
        ]
        analyses = rq3_outcomes(declaration, partition, subjects, capacities)
        for name in CONFIGURATION_NAMES:
            for method in METHODS:
                records.append(
                    (
                        (name, method),
                        to_json_value(
                            {
                                **setup.envelope("test_rq3"),
                                "configuration": name,
                                "calibrator": method,
                                "partition": "Test",
                                "days": setup.days,
                                "scores_sha256": setup.score_files[name],
                                "calibrator_file": dict(policies.calibrator_files[(name, method)]),
                                "capacities": analyses[(name, method)],
                            }
                        ),
                    )
                )
        found = {name: misses_of(partition, scores[name], primary) for name in CONFIGURATION_NAMES}
        tables = false_negative_analysis(declaration, partition, slices.variables, found, primary)
        for name in CONFIGURATION_NAMES:
            table = tables[name]
            records.append(
                (
                    (name,),
                    to_json_value(
                        {
                            **setup.envelope("test_false_negative"),
                            "configuration": name,
                            "partition": "Test",
                            "days": setup.days,
                            "scores_sha256": setup.score_files[name],
                            **{k: v for k, v in table.items() if k != "slices"},
                            "boundaries": {
                                "amount_edges": list(slices.boundaries.amount_edges),
                                "kept": {
                                    variable: dict(levels)
                                    for variable, levels in slices.boundaries.kept.items()
                                },
                            },
                            "slices": table["slices"],
                        }
                    ),
                )
            )
    return records


def publish_outcomes(run: RunDirectory, records: Sequence[Named]) -> int:
    """Write the records not yet written; an existing record must equal the one that would be
    written. Returns the number written now."""
    written = 0
    for parts, record in records:
        path = run.record_path(str(record["record"]), *parts)
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != json.loads(json.dumps(record)):
                raise SpecificationError(f"{path.name} exists and differs from the computed record")
            continue
        run.write_record(record, *parts)
        written += 1
    logger.info("Outcome records: %d written, %d already present", written, len(records) - written)
    return written
