"""The outcome records built by the production code satisfy their contracts and, with the
scoring-phase records, the rules of a complete run (Issue #205; synthetic data)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from review_capacity.confirmatory.inference_record import (
    STREAMS,
    build_inference_record,
    inference_inputs,
)
from review_capacity.confirmatory.outcome_records import (
    Named,
    build_outcome_records,
    publish_outcomes,
)
from review_capacity.confirmatory.records import ConfirmatoryRecords, record_problems, run_problems
from review_capacity.development.run_output import RunDirectory
from review_capacity.errors import SpecificationError
from review_capacity.evaluation.resampling import (
    day_block_indices,
    naive_ratios,
    pigeonhole_ratios,
    stream_generator,
)
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
    partition, _ = c.outcome_world()
    return build_outcome_records(
        c.outcome_setup(),
        partition,
        c.outcome_scores_all(),
        c.outcome_policies(),
        c.outcome_slices(),
    )


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


def test_the_inference_inputs_are_drawn_from_the_scores_labels_and_declared_streams() -> None:
    declaration, (partition, _) = c.outcome_declaration(), c.outcome_world()
    scores, policies = c.outcome_scores_all(), c.outcome_policies()
    inputs = inference_inputs(declaration, partition, scores, policies)
    truth, order = partition.truth, declaration.configurations.order
    fraud = truth.labels == 1
    k = 56
    for i, name in enumerate(order):
        mask = partition.queue.select(scores[name], k).mask[fraud]
        assert np.array_equal(inputs.captures[i], mask)
        assert inputs.recalls[i] == mask.mean()
        threshold = c.THRESHOLDS.r2[0.02]
        assert inputs.rates[i] == pytest.approx((scores[name] >= threshold).sum() / truth.rows)
    reviewable = np.minimum(k, truth.frauds).sum() / int(fraud.sum())
    assert inputs.oracle_ceiling == pytest.approx(reviewable)
    assert (inputs.frauds, inputs.days, inputs.k) == (int(fraud.sum()), 28, 56)
    b = declaration.resampling.replicates
    assert inputs.pigeonhole.ratios.shape == (b, 8)
    assert inputs.naive.ratios.shape == (b, 8)
    assert set(inputs.h3_sets) == {3, 1, 7}
    direct = pigeonhole_ratios(
        stream_generator(declaration, "pigeonhole_h1_h2"),
        inputs.captures.astype(np.float64),
        partition.fraud_day_index,
        28,
        partition.entity_index,
        replicates=b,
    )
    assert np.array_equal(
        inputs.pigeonhole.ratios, direct.ratios
    )  # the stream restarts identically
    indices = day_block_indices(
        stream_generator(declaration, "day_block_s2"),
        28,
        declaration.resampling.block_length,
        replicates=b,
    )
    captured = np.vstack(
        [np.bincount(partition.fraud_day_index[row], minlength=28) for row in inputs.captures]
    )
    brute = captured[:, indices].sum(axis=2) / truth.frauds[indices].sum(axis=1)
    assert inputs.day_block.ratios == pytest.approx(brute.T)


def test_the_inference_record_satisfies_its_contract_and_names_every_estimand() -> None:
    record = c.outcome_inference()
    assert record_problems(record, c.outcome_context()) == []
    assert record["h3"]["configuration"] == "LGB-b"
    assert record["resampling"]["replicates"] == c.OUTCOME_B
    names = {unit["estimand"] for unit in record["effect_units"]}
    assert {"family", "interaction"} <= names
    assert {f"weighting:{m['model']}" for m in record["h2"]["models"]} <= names
    assert record["effect_basis"]["k"] == 56
    assert set(record["undefined_replicates"]) == set(STREAMS)


def test_the_inference_record_completes_the_run(built: list[Named]) -> None:
    kept = [r for r in c.run_records() if r["record"] not in (*OUTCOME_KINDS, "test_inference")]
    records = [*kept, *(record for _, record in built), c.outcome_inference()]
    assert run_problems(records) == []


def test_the_remaining_streams_are_reproduced_independently_and_the_record_is_deterministic() -> (
    None
):
    declaration, (partition, _) = c.outcome_declaration(), c.outcome_world()
    scores, policies = c.outcome_scores_all(), c.outcome_policies()
    inputs = inference_inputs(declaration, partition, scores, policies)
    b, truth = declaration.resampling.replicates, partition.truth
    direct = naive_ratios(
        stream_generator(declaration, "naive_s1"), inputs.captures.astype(np.float64), replicates=b
    )
    assert np.array_equal(inputs.naive.ratios, direct.ratios)
    # the three H3 block lengths are drawn one after the other from the one H3 stream
    generator = stream_generator(declaration, "day_block_h3")
    order = declaration.configurations.order
    threshold = c.THRESHOLDS.r2[0.02]
    alerts = np.vstack(
        [np.bincount(truth.day_index[scores[name] >= threshold], minlength=28) for name in order]
    ).astype(np.float64)
    volumes = truth.transactions.astype(np.float64)
    for length in (3, 1, 7):
        indices = day_block_indices(generator, 28, length, replicates=b)
        brute = alerts[:, indices].sum(axis=2) / volumes[indices].sum(axis=1)
        assert inputs.h3_sets[length].ratios == pytest.approx(brute.T)
    again = build_inference_record(c.outcome_setup(), inputs, "LGB-b")
    assert again == c.outcome_inference()  # a fresh build gives the identical record
