"""The outcome records built by the production code satisfy their contracts and, with the
scoring-phase records, the rules of a complete run (Issue #205; synthetic data)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from review_capacity.confirmatory.frozen_policies import FrozenPolicies
from review_capacity.confirmatory.outcome_records import (
    Named,
    OutcomeSetup,
    build_outcome_records,
    publish_outcomes,
)
from review_capacity.confirmatory.records import ConfirmatoryRecords, record_problems, run_problems
from review_capacity.development.records import ALL_CONFIGURATIONS, CONFIGURATION_NAMES
from review_capacity.development.run_output import RunDirectory
from review_capacity.errors import SpecificationError
from support import confirmatory as c

OUTCOME_KINDS = (
    "test_evaluation",
    "test_workload",
    "test_calibration",
    "test_rq3",
    "test_false_negative",
)


@pytest.fixture(scope="module")
def built() -> list[Named]:
    partition, scores = c.outcome_world()
    rng = np.random.default_rng(77)
    every = {
        name: np.round(
            np.clip(scores["LGB-b"] + rng.normal(0, 0.03, len(scores["LGB-b"])), 0, 1), 4
        )
        for name in ALL_CONFIGURATIONS
    }
    every["LGB-b"], every["LR"] = scores["LGB-b"], scores["LR"]
    calibrators = c._calibrators()
    policies = FrozenPolicies(
        dict.fromkeys(CONFIGURATION_NAMES, c.THRESHOLDS),
        {(n, m): calibrators[m] for n in CONFIGURATION_NAMES for m in ("platt", "isotonic")},
        {
            (n, m): c.file_ref(f"calibrator-{m}")
            for n in CONFIGURATION_NAMES
            for m in ("platt", "isotonic")
        },
    )
    ctx = c.outcome_context()
    setup = OutcomeSetup(
        c.outcome_declaration(),
        c.identity(),
        ctx.test.first_day,
        ctx.test.last_day,
        {name: c.scores(name)["file"]["sha256"] for name in ALL_CONFIGURATIONS},
        c.V2_VOLUME,
    )
    return build_outcome_records(setup, partition, every, policies, c.outcome_slices())


def test_every_kind_is_built_for_its_subjects_in_the_publication_order(built: list[Named]) -> None:
    kinds = [record["record"] for _, record in built]
    assert Counter(kinds) == {
        "test_evaluation": 9,
        "test_workload": 8,
        "test_calibration": 16,
        "test_rq3": 16,
        "test_false_negative": 8,
    }
    assert kinds == sorted(kinds, key=OUTCOME_KINDS.index)  # evaluation, workload, ... in order
    assert len({(record["record"], parts) for parts, record in built}) == len(built)


def test_every_built_record_satisfies_its_contract(built: list[Named]) -> None:
    for _, record in built:
        assert record_problems(record, c.outcome_context()) == [], record["record"]
        json.dumps(record)  # plain JSON values


def test_the_built_records_make_a_consistent_complete_run(built: list[Named]) -> None:
    kept = [r for r in c.run_records() if r["record"] not in OUTCOME_KINDS]
    records = [*kept, *(record for _, record in built)]
    assert run_problems(records) == []


def test_publication_writes_each_record_once_and_keeps_an_equal_record(
    built: list[Named], tmp_path: Path
) -> None:
    run = RunDirectory.create(
        tmp_path,
        None,
        "confirmatory",
        c.RUN_ID,
        records=ConfirmatoryRecords(c.outcome_context()),
    )
    run.write_record(c.test_run())
    subset = [
        item for item in built if item[1]["record"] in ("test_workload", "test_false_negative")
    ]
    assert publish_outcomes(run, subset) == len(subset)
    assert publish_outcomes(run, subset) == 0  # nothing is written twice
    first = subset[0]
    changed: Any = json.loads(json.dumps(first[1]))
    changed["configuration"] = "Cat-b"
    with pytest.raises(SpecificationError, match="differs from the computed record"):
        publish_outcomes(run, [(first[0], changed)])
