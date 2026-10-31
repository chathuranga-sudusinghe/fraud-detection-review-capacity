"""The contracts of the confirmatory Test run's records: structure, the rules between fields, the
recomputation of redundant values, and the rules between the phase records. The records are built
by ``support.confirmatory`` on synthetic data; no research data is read."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np
import pytest

from review_capacity.confirmatory.records import (
    RECORD_CONTRACTS,
    ConfirmatoryContext,
    contract_of,
    contract_versions,
    dump_record,
    phase_sequence_problems,
    read_record,
    record_problems,
    reference_markdown,
    run_problems,
    scores_bundle_sha256,
    to_json_value,
    validate_record,
)
from review_capacity.development.records import ALL_CONFIGURATIONS, PARTITION
from review_capacity.development.schema import loads
from review_capacity.errors import RecordContractError
from support import confirmatory as c

CTX = c.context()


def problems(record: Any) -> list[str]:
    return record_problems(record, CTX)


def refused(record: Any, fragment: str) -> None:
    found = problems(record)
    assert any(fragment in problem for problem in found), (fragment, found[:5])


@pytest.fixture(scope="module")
def inference() -> dict[str, Any]:
    return c.inference()


# --- valid records and the envelope --------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [c.test_run, c.phase, c.scores, c.inference, c.run_end],
    ids=["test_run", "test_phase", "test_scores", "test_inference", "test_run_end"],
)
def test_a_valid_record_of_each_kind_passes_and_round_trips(
    build: Callable[[], dict[str, Any]],
) -> None:
    record = build()
    assert problems(record) == []
    validate_record(record, CTX)
    assert loads(dump_record(record, CTX), "text") == record


def test_the_kinds_and_versions_are_those_of_the_run() -> None:
    assert list(RECORD_CONTRACTS) == [
        "test_run",
        "test_phase",
        "test_scores",
        "test_inference",
        "test_evaluation",
        "test_workload",
        "test_calibration",
        "test_rq3",
        "test_false_negative",
        "test_run_end",
    ]
    assert contract_versions() == dict.fromkeys(RECORD_CONTRACTS, 1)


def test_an_unknown_kind_and_a_wrong_expected_kind_are_refused() -> None:
    with pytest.raises(RecordContractError, match="Unknown confirmatory record kind"):
        contract_of({"record": "evaluation"})
    with pytest.raises(RecordContractError, match="Expected a test_phase record"):
        validate_record(c.scores(), CTX, kind="test_phase")


def test_structure_errors_are_all_reported() -> None:
    record = c.mutated(c.scores(), "rows", "many")
    record["extra"] = 1
    del record["device"]
    found = problems(record)
    assert any("expected an integer" in p for p in found)
    assert any("unknown field 'extra'" in p for p in found)
    assert any("missing field 'device'" in p for p in found)


def test_null_is_refused_where_a_value_is_required() -> None:
    refused(c.mutated(c.scores(), "identity.freeze.tag", None), "must not be null")


def test_a_non_finite_number_is_refused() -> None:
    refused(c.mutated(c.scores(), "timing.seconds", float("nan")), "not a finite number")
    refused(c.mutated(c.scores(), "timing.seconds", float("inf")), "not a finite number")


def test_a_file_is_read_back_and_a_bad_one_is_refused(tmp_path: Any) -> None:
    good = tmp_path / "test_scores.json"
    good.write_text(dump_record(c.scores(), CTX), encoding="utf-8")
    assert read_record(good, CTX, "test_scores")["configuration"] == "LGB-b"
    bad = tmp_path / "bad.json"
    bad.write_text(dump_record(c.scores(), CTX).replace('"LGB-b"', '"XGB-b"'), encoding="utf-8")
    with pytest.raises(RecordContractError, match=r"bad\.json"):
        read_record(bad, CTX)


# --- identity ------------------------------------------------------------------------------------


def test_the_evaluator_must_be_a_clean_tree_at_the_commit_of_the_tag() -> None:
    refused(
        c.mutated(c.test_run(), "identity.evaluator.uncommitted_changes", True), "must be clean"
    )
    refused(
        c.mutated(c.test_run(), "identity.evaluator.code_commit", "b" * 40),
        "commit of the freeze tag",
    )


def test_the_resampling_seed_is_the_declared_root_seed() -> None:
    refused(c.mutated(c.phase(), "identity.seeds.resampling", 7), "declared root seed")


# --- test_run ------------------------------------------------------------------------------------


def test_run_hashes_must_agree_with_the_identity() -> None:
    refused(
        c.mutated(c.test_run(), "settings.sha256", c.sha("other")),
        "differs from the run's own value",
    )


def test_run_frozen_state_identity_follows_the_frozen_inputs() -> None:
    refused(
        c.mutated(c.test_run(), "frozen_inputs.models.LGB-b.sha256", c.sha("tampered")),
        "frozen_state_sha256",
    )
    for path in (
        "frozen_inputs.model_manifests.RES-b.sha256",
        "frozen_inputs.model_records.Cat-u.sha256",
        "frozen_inputs.preprocessing_arrays.sha256",
    ):
        refused(c.mutated(c.test_run(), path, c.sha("tampered")), "frozen_state_sha256")


def test_run_needs_a_model_with_its_record_and_manifest_for_every_configuration() -> None:
    for name in ("models", "model_records", "model_manifests"):
        refused(c.without(c.test_run(), f"frozen_inputs.{name}.LR"), f"frozen_inputs.{name}")


def test_run_test_partition_is_the_declared_one() -> None:
    refused(c.mutated(c.test_run(), "test.first_day", 1), "test.first_day")
    refused(c.mutated(c.test_run(), "test.last_day", 154), "test.last_day")
    refused(c.mutated(c.test_run(), "test.rows", 77822), "test.rows")
    refused(
        c.mutated(c.test_run(), "test.transaction_id_sha256", c.sha("not the anchor")),
        "declared anchor",
    )


def test_run_contract_versions_are_the_current_ones() -> None:
    refused(c.mutated(c.test_run(), "contracts", {"test_run": 9}), "test_run.contracts")


# --- test_phase ----------------------------------------------------------------------------------


def test_the_scoring_phase_never_reaches_labels_and_a_completed_evaluation_does() -> None:
    refused(c.mutated(c.phase("scoring"), "labels_accessed", True), "never reaches")
    refused(c.mutated(c.phase("evaluation"), "labels_accessed", False), "reached the labels")
    # an evaluation attempt that failed before the consumption marker never reached the labels
    early = c.mutated(c.phase("evaluation", outcome="failed"), "labels_accessed", False)
    assert problems(early) == []
    assert problems(c.phase("evaluation", outcome="failed")) == []  # one that reached the marker


def test_a_failed_attempt_states_its_error_and_a_completed_one_has_none() -> None:
    refused(c.mutated(c.phase("scoring"), "error", "oops"), "null when the attempt completed")
    refused(c.mutated(c.phase("scoring", outcome="failed"), "error", None), "required when")
    assert problems(c.phase("scoring", outcome="interrupted")) == []


def test_the_scores_bundle_is_given_exactly_when_scoring_completed() -> None:
    refused(c.mutated(c.phase("scoring"), "scores_sha256", None), "given exactly when")
    refused(c.mutated(c.phase("scoring", outcome="failed"), "scores_sha256", c.sha("x")), "given")
    refused(c.mutated(c.phase("evaluation"), "scores_sha256", None), "names the scores")


def test_a_valid_phase_sequence_with_a_repeated_scoring_and_a_resumed_evaluation() -> None:
    bundle = c.sha("bundle")
    sequence = [
        c.phase("scoring", 1, "failed"),
        c.phase("scoring", 2, "completed", bundle),
        c.phase("evaluation", 1, "failed", bundle),
        c.phase("evaluation", 2, "completed", bundle),
    ]
    assert phase_sequence_problems(sequence) == []
    assert phase_sequence_problems([]) == []


def test_phase_sequence_rules() -> None:
    bundle = c.sha("bundle")
    done = c.phase("scoring", 1, "completed", bundle)
    evaluation = c.phase("evaluation", 1, "completed", bundle)
    cases = [
        ("no scoring attempt may follow", [done, c.phase("scoring", 2, "completed", bundle)]),
        (
            "no scoring attempt may follow",
            [
                c.phase("scoring", 1, "failed"),
                c.phase("scoring", 2, "completed", bundle),
                evaluation,
                c.phase("scoring", 3, "failed"),
            ],
        ),
        ("an evaluation attempt follows a completed scoring", [evaluation]),
        ("attempt: 2, expected 1", [c.phase("scoring", 2, "failed")]),
        (
            "differs from the completed scoring's",
            [done, c.phase("evaluation", 1, "completed", c.sha("other"))],
        ),
        (
            "no attempt may follow a completed evaluation",
            [done, evaluation, c.phase("evaluation", 2, "failed", bundle)],
        ),
    ]
    for fragment, sequence in cases:
        found = phase_sequence_problems(sequence)
        assert any(fragment in problem for problem in found), (fragment, found)


# --- test_scores ---------------------------------------------------------------------------------


def test_scores_describe_the_test_partition_and_never_a_development_one() -> None:
    refused(c.mutated(c.scores(), "days.first", 127), "days.first")
    refused(c.mutated(c.scores(), "days.last", 181), "days.last")
    refused(c.mutated(c.scores(), "rows", 77822), "rows")
    refused(c.mutated(c.scores(), "partition", "V2"), "is not one of Test")
    assert PARTITION.problems("Test", "x")  # the development contracts still refuse Test


def test_scores_rules_between_fields() -> None:
    refused(c.mutated(c.scores(), "min_score", 0.9999), "must not exceed")
    refused(c.mutated(c.scores(), "distinct_scores", c.context().test.rows + 1), "cannot exceed")
    refused(c.mutated(c.scores(), "timing.passes", 2), "is outside")
    refused(c.mutated(c.scores(), "timing.rows_per_second", 1.0), "rows divided by seconds")
    assert problems(c.scores("LR")) == []  # the logistic-regression reference
    refused(c.mutated(c.scores(), "configuration", "XGB-b"), "is not one of")


# --- test_inference: redundant values are recomputed ----------------------------------------------

CASES: list[tuple[str, Any, str]] = [
    ("holm.adjusted.0", 0.9999, "holm.adjusted"),
    ("holm.rejected.1", lambda r: not r["holm"]["rejected"][1], "holm.rejected"),
    ("decisions.h3_preserved", lambda r: not r["decisions"]["h3_preserved"], "h3_preserved"),
    ("decisions.h1_category", "inconclusive", "h1_category"),
    ("h1.position", "crossing", "h1.position"),
    ("h1.interaction_splits", lambda r: not r["h1"]["interaction_splits"], "interaction_splits"),
    ("h1.interval.level", 0.95, "h1.interval.level"),
    ("h1.cells.0.name", "cell:x", "h1.cells"),
    ("h1.by_weighting.0.name", "family:z", "h1.by_weighting"),
    ("h2.p.value", 0.00001, "h2.p"),
    ("h2.models.0.position", "crossing", "h2.models"),
    ("h2.models.0.model", "XGB", "h2.models"),
    ("h3.primary.p.mc_se", 0.5, "h3 (block length 3).p"),
    ("h3.primary.position", "crossing", "h3 (block length 3).position"),
    ("h3.smallest_tolerance", 0.5, "smallest_tolerance"),
    ("h3.tolerances.0.lower", 0.001, "h3.tolerances"),
    ("h3.tolerances.1.position", "crossing", "tolerances"),
    ("h3.others.0.configuration", "FTT-b", "h3.others"),
    ("h3.sensitivity.0.block_length", 5, "block lengths"),
    ("s1.quantities.0.se", 0.5, "se: not the root"),
    ("s1.p_h1", None, "given exactly when the family effect"),
    ("s1.comparison", None, "given exactly when"),
    ("s1.reasons", ["why"], "s1.reasons"),
    ("s2.comparison.differs", lambda r: not r["s2"]["comparison"]["differs"], "differs"),
    (
        "s2.comparison.holm.rejected.0",
        lambda r: not r["s2"]["comparison"]["holm"]["rejected"][0],
        "holm.rejected",
    ),
    ("s2.h1_position", "crossing", "s2.h1_position"),
    ("s2.comparison.p_h1", 0.4, "its p-values differ"),
    ("sesoi_views.0.sesoi", 0.5, "sesoi_views"),
    ("sesoi_views.0.h1_position", "inside", "h1_position"),
    ("sesoi_views.0.h2_positions.0", "beyond", "h2_positions"),
    ("mcnemar.0.p", 0.123456, "mcnemar"),
    ("mcnemar.0.first", "Cat-b", "mcnemar"),
    ("resampling.replicates", 5000, "test_inference.resampling"),
    ("undefined_replicates", {"no_such_stream": 1}, "must name exactly"),
    ("h1.interval.lower", 1.0, "lower exceeds upper"),
    ("h1.interval.replicates", 2000, "h1.interval.replicates"),
    ("h1.difference_interval.level", 0.9, "h1.difference_interval.level"),
    ("h1.cells.0.interval.level", 0.9, "h1.cells[cell:LGB-b-LGB-u].interval.level"),
    ("h1.cells.0.interval.undefined", 3, "h1.cells[cell:LGB-b-LGB-u].interval.undefined"),
    ("h2.models.0.interval.replicates", 2000, "h2.LGB.interval.replicates"),
    ("h3.others.0.interval.replicates", 2000, "h3.others"),
    ("s2.h1_interval.replicates", 5, "s2.h1_interval.replicates"),
    ("s1.interval_h1.replicates", 2000, "a Wald interval has no replicates"),
    ("undefined_replicates.naive_s1", 20000, "more than the 10000 replicates"),
    ("undefined_replicates.day_block_h3", 4, "largest count of the H3 intervals"),
    ("h1.p.value", 0.00031, "not an add-one proportion"),
    ("h1.p.mc_se", 0.5, "h1.p.mc_se"),
    ("h2.models.0.p_lower.value", 0.00123, "not an add-one proportion"),
    ("h3.primary.p_lower.mc_se", 0.3, "mc_se"),
    ("s2.p_h1.value", 0.00077, "not an add-one proportion"),
    ("s1.quantities.0.name", "family-effect", "s1.quantities: must be"),
    ("s1.quantities.0.estimate", 0.5, "s1.quantities[family].estimate"),
    ("s1.p_h1", 0.3, "differs from the Wald p-value"),
    ("s1.interval_h1.lower", -0.5, "differs from the Wald interval"),
    ("s1.position_h1", "crossing", "s1.position_h1"),
    ("s1.p_h2", 0.4, "differs from the largest Wald p-value"),
    ("s1.intervals_h2.0.upper", 0.4, "s1.intervals_h2"),
    ("s1.comparison.p_h1", 0.4, "its p-values differ from S1's"),
    ("effect_basis.days", 29, "effect_basis.days"),
    ("effect_basis.k", 55, "effect_basis.k"),
    ("effect_basis.frauds", 2700, "differs from the units of the estimate"),
    ("effect_units.0.percentage_points", 7.0, "effect_units[family].percentage_points"),
    ("effect_units.0.estimand", "no-such-estimand", "not an estimand of the record"),
]


@pytest.mark.parametrize(("path", "value", "fragment"), CASES, ids=[case[0] for case in CASES])
def test_a_redundant_value_that_differs_from_its_recomputation_is_refused(
    inference: dict[str, Any], path: str, value: Any, fragment: str
) -> None:
    new = value(inference) if callable(value) else value
    refused(c.mutated(inference, path, new), fragment)


def unestimable_family(record: dict[str, Any]) -> dict[str, Any]:
    """S1 with no standard error for the family effect, stated consistently."""
    result = c.mutated(record, "s1.quantities.0.se", None)
    result = c.mutated(result, "s1.quantities.0.v_pigeonhole", 0.0)
    result = c.mutated(result, "s1.quantities.0.v_naive", 0.0)
    for name in ("p_h1", "interval_h1", "position_h1", "comparison"):
        result = c.mutated(result, f"s1.{name}", None)
    return c.mutated(result, "s1.reasons", ["the corrected variance of delta_F is not positive"])


def test_a_consistently_unestimable_s1_is_accepted(inference: dict[str, Any]) -> None:
    assert problems(unestimable_family(inference)) == []


def test_an_s1_that_reports_conclusions_without_a_standard_error_is_refused(
    inference: dict[str, Any],
) -> None:
    only_se = c.mutated(inference, "s1.quantities.0.se", None)
    only_se = c.mutated(only_se, "s1.quantities.0.v_pigeonhole", 0.0)
    only_se = c.mutated(only_se, "s1.quantities.0.v_naive", 0.0)
    refused(only_se, "given exactly when the family effect has a standard error")
    refused(c.mutated(unestimable_family(inference), "s1.reasons", []), "s1.reasons")
    kept = c.mutated(unestimable_family(inference), "s1.comparison", inference["s1"]["comparison"])
    refused(kept, "s1.comparison")


def test_a_wald_interval_has_no_replicates_and_is_allowed(inference: dict[str, Any]) -> None:
    assert inference["s1"]["interval_h1"]["replicates"] == 0
    assert problems(inference) == []


# --- test_run_end --------------------------------------------------------------------------------


def test_run_end_rules() -> None:
    refused(c.mutated(c.run_end(), "error", "x"), "null when the run completed")
    refused(c.mutated(c.run_end(), "outcome", "failed"), "required when the run failed")
    two = [c.run_file("records/test_scores-A.json"), c.run_file("records/test_run.json")]
    refused(c.run_end(two), "increasing path order")
    refused(c.run_end([c.run_file("records/test_run_end.json")]), "not another record")
    refused(c.run_end([c.run_file("elsewhere/x.json")]), "not another record")


# --- converters and documentation ----------------------------------------------------------------


def test_to_json_value_converts_dataclasses_enums_tuples_and_numpy_scalars() -> None:
    class Colour(Enum):
        RED = "red"

    @dataclasses.dataclass
    class Part:
        values: tuple[int, ...]
        colour: Colour

    converted = to_json_value(
        {"part": Part((1, 2), Colour.RED), "n": np.int64(3), "x": np.float64(0.5)}
    )
    assert converted == {"part": {"values": [1, 2], "colour": "red"}, "n": 3, "x": 0.5}
    assert type(converted["n"]) is int
    assert type(converted["x"]) is float


def test_the_reference_documentation_lists_every_field_of_every_kind() -> None:
    text = reference_markdown()
    for kind, contract in RECORD_CONTRACTS.items():
        assert f"### `{kind}` (version {contract.version})" in text
        for path, *_ in contract.reference_rows():
            assert f"`{path}`" in text
    assert "not edited by hand" in text


def test_the_context_comes_from_the_dataset_declaration() -> None:
    assert isinstance(CTX, ConfirmatoryContext)
    assert (CTX.test.name, CTX.test.first_day, CTX.test.last_day) == ("test", 155, 182)


# --- the rules between the records of one run ----------------------------------------------------


def run_with(change: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]) -> list[str]:
    return run_problems(change(c.run_records()))


def has(found: list[str], fragment: str) -> bool:
    return any(fragment in problem for problem in found)


def test_the_records_of_a_completed_run_agree() -> None:
    assert run_problems(c.run_records()) == []


def test_the_phases_are_judged_in_the_order_of_their_start_not_of_their_position() -> None:
    records = c.run_records()
    assert run_problems([records[0], records[3], records[2], records[1], *records[4:]]) == []
    bundle = scores_bundle_sha256([c.scores(name)["file"]["sha256"] for name in ALL_CONFIGURATIONS])
    early = c.phase("evaluation", 1, "completed", bundle, started="2026-11-02T09:05:00.000000Z")
    found = run_problems([records[0], records[1], records[2], early, *records[4:]])
    assert has(found, "follows a completed scoring")


def first_scores(records: list[dict[str, Any]]) -> dict[str, Any]:
    return next(x for x in records if x["record"] == "test_scores")


def test_a_run_needs_every_score_record_once_and_the_frozen_model() -> None:
    assert has(run_with(lambda r: [*r[:-2], r[-1]]), "one test_scores record for every")
    assert has(
        run_with(lambda r: [*r, next(x for x in r if x["record"] == "test_scores")]),
        "more than one test_scores",
    )
    other_model = c.file_ref("another-model")
    assert has(
        run_with(
            lambda r: [c.mutated(x, "model", other_model) if x is first_scores(r) else x for x in r]
        ),
        "another model than the frozen one",
    )


def test_an_incomplete_run_is_judged_only_on_what_it_holds() -> None:
    partial = c.run_records()[:2]
    assert run_problems(partial, complete=False) == []
    assert has(run_problems(partial), "one test_scores record for every")


def test_the_completed_scoring_names_the_scores_that_exist() -> None:
    wrong = c.phase(
        "scoring", 2, "completed", c.sha("other"), started="2026-11-02T09:10:00.000000Z"
    )
    assert has(run_with(lambda r: [r[0], r[1], wrong, *r[3:]]), "names other scores")
    found = run_with(lambda r: [r[0], *r[3:]])
    assert has(found, "without a completed scoring")
    assert has(found, "needs a completed scoring")


def test_the_inference_follows_an_evaluation_and_is_of_the_frozen_configuration() -> None:
    assert has(run_with(lambda r: [*r[:3], *r[4:]]), "follows a completed evaluation")
    other = lambda r: [*r[:-1], c.mutated(r[-1], "h3.configuration", "Cat-b")]  # noqa: E731
    assert has(run_with(other), "not of the frozen H3 configuration")
    assert has(run_with(lambda r: [*r, r[-1]]), "more than one test_inference")


def test_a_run_has_exactly_one_run_record() -> None:
    assert run_problems([c.phase()]) == ["run: needs exactly one test_run record"]
    assert run_problems([c.test_run(), c.test_run()]) == ["run: needs exactly one test_run record"]
