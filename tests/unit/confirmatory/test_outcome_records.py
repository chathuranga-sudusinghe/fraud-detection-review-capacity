"""The Test outcome record kinds accept the output of the outcome functions and refuse each
altered redundant value (Issue #197; synthetic data, no data read)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest

from review_capacity.confirmatory.records import (
    RECORD_CONTRACTS,
    ConfirmatoryContext,
    record_problems,
    reference_markdown,
    run_problems,
    validate_record,
)
from review_capacity.development.schema import loads
from review_capacity.errors import RecordContractError
from support import confirmatory as c


def problems(record: dict[str, Any]) -> list[str]:
    return record_problems(record, c.outcome_context())


@pytest.mark.parametrize(
    "build",
    [
        c.evaluation,
        lambda: c.evaluation("LR"),
        c.workload,
        c.calibration,
        lambda: c.calibration("isotonic"),
        c.rq3_record,
        lambda: c.rq3_record("isotonic"),
        c.false_negative_record,
    ],
    ids=[
        "evaluation",
        "reference",
        "workload",
        "platt",
        "isotonic",
        "rq3-platt",
        "rq3-isotonic",
        "false-negative",
    ],
)
def test_the_output_of_the_functions_is_a_valid_record(build: Callable[[], dict[str, Any]]) -> None:
    record = build()
    assert problems(record) == []
    validate_record(record, c.outcome_context())
    assert loads(__import__("json").dumps(record), "text") == record


def refused(record: dict[str, Any], fragment: str) -> None:
    """The public validation refuses the record with a problem naming ``fragment``."""
    found = problems(record)
    assert found, "the altered record was accepted"
    assert any(fragment in problem for problem in found), found


def mutated_all(record: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    for path, value in changes.items():
        record = c.mutated(record, path, value)
    return record


def test_the_reference_has_r3_and_the_ranking_only() -> None:
    reference = c.evaluation("LR")
    assert reference["thresholds"] is None
    assert reference["policies"] == []
    refused(c.mutated(reference, "thresholds", {"r0": 0.5, "r1": 0.3, "r2": 0.2}), "thresholds")
    refused(c.mutated(c.evaluation(), "thresholds", None), "thresholds")


def test_counts_and_metrics_are_recomputed() -> None:
    record = c.evaluation()
    refused(c.mutated(record, "r3.1.counts.true_positives", 1), "true_positives")
    refused(c.mutated(record, "r3.1.metrics.recall", 0.999), "metrics.recall")
    refused(c.mutated(record, "policies.2.metrics.f1", 0.999), "metrics.f1")
    refused(c.mutated(record, "policies.2.metrics.total_overflow", 1), "total_overflow")
    refused(c.mutated(record, "r3.0.daily.0.overflow", 5), "overflow")
    refused(c.mutated(record, "policies.0.threshold", 0.4), "threshold")


def test_every_declared_capacity_is_evaluated() -> None:
    record = c.evaluation()
    last = record["r3"][-1]["capacity"]
    fewer = copy.deepcopy(record)
    fewer["r3"] = fewer["r3"][:-1]
    fewer["policies"] = [e for e in fewer["policies"] if e["capacity"] != last]
    fewer["thresholds"]["r2"] = fewer["thresholds"]["r2"][:-1]
    assert problems(fewer) != []
    refused(fewer, "every declared capacity")
    refused(c.mutated(record, "r3.0.k", 29), "must be")  # the daily table then disagrees
    refused(c.mutated(record, "r3.0.capacity", 0.03), "capacity_to_prevalence")


def test_r2_has_its_own_frozen_threshold_for_every_capacity() -> None:
    record = c.evaluation()
    r2 = record["thresholds"]["r2"]
    assert len({item["threshold"] for item in r2}) == len(r2)  # the fixture differs by capacity
    assert [e["threshold"] for e in record["policies"] if e["policy"] == "R2"] == [
        item["threshold"] for item in r2
    ]
    index = next(i for i, e in enumerate(record["policies"]) if e["policy"] == "R2")
    refused(c.mutated(record, f"policies.{index}.threshold", r2[1]["threshold"]), "R2 threshold")
    refused(c.mutated(record, "thresholds.r2.0.capacity", 0.03), "every R3 capacity")
    refused(c.mutated(record, "thresholds.r2", r2[:-1]), "every R3 capacity")


def test_intervals_follow_the_declaration_and_their_derivations() -> None:
    record = c.evaluation()
    refused(c.mutated(record, "r3.0.intervals.recall.level", 0.9), "level")
    refused(c.mutated(record, "r3.0.intervals.recall.replicates", 10_000), "declared B")
    refused(c.mutated(record, "r3.0.intervals.recall.undefined", c.OUTCOME_B), "undefined")
    refused(c.mutated(record, "r3.0.intervals.recall.lower", 0.99), "recall interval")
    lift = record["r3"][0]["intervals"]["lift"]
    refused(c.mutated(record, "r3.0.intervals.lift.lower", lift["lower"] * 1.5), "lift")
    refused(c.mutated(record, "r3.0.intervals.lift.undefined", 1), "recall interval's")
    refused(c.mutated(record, "ranking.intervals.roc_auc.replicates", 7), "declared B")
    refused(c.without(record, "r3.0.intervals.recall"), "intervals")


def test_the_days_must_be_the_test_partition() -> None:
    refused(c.mutated(c.evaluation(), "days.last", 181), "one entry per evaluated day")
    refused(c.mutated(c.workload(), "days.first", 154), "one entry per evaluated day")
    shifted = c.mutated(c.calibration(), "days.first", 154)
    refused(c.mutated(shifted, "diagnostic.days.first", 154), "Test partition")
    refused(c.mutated(c.calibration(), "diagnostic.rows", 1_000), "do not sum")


def test_the_decomposition_is_recomputed() -> None:
    record = c.workload()
    refused(c.mutated(record, "entries.1.fidelity.terms.0.rate", 0.5), "rate")
    refused(c.mutated(record, "entries.1.fidelity.terms.0.total", 0.5), "total")
    refused(c.mutated(record, "entries.1.fidelity.mean_total", 0.5), "mean_total")
    refused(c.mutated(record, "entries.1.fidelity.total_overflow", 1), "total_overflow")
    refused(c.mutated(record, "entries.1.fidelity.terms.0.day", 200), "one entry per evaluated day")
    refused(c.mutated(record, "entries.1.capacity", 0.03), "rate")


def test_the_reference_volume_is_the_declared_v2_volume() -> None:
    wrong = c.workload(reference_volume=2780.0)  # rounds to the same budgets, another volume
    assert any("V2 rows over V2 days" in p for p in problems(wrong))
    assert problems(c.workload()) == []
    fewer = copy.deepcopy(c.workload())
    fewer["entries"] = fewer["entries"][:-1]
    refused(fewer, "every declared capacity")


def test_every_metric_of_an_entry_is_present_and_recomputed() -> None:
    record = c.evaluation()
    refused(c.mutated(record, "ranking.metrics", {}), "missing")
    refused(c.mutated(record, "r3.0.metrics", {"recall": 0.5}), "missing")
    refused(c.without(record, "r3.0.metrics.amount_weighted_recall"), "missing")
    refused(c.mutated(record, "r3.0.metrics.unexpected", 1.0), "unexpected")
    index = next(i for i, e in enumerate(record["policies"]) if e["policy"] == "R1")
    load = record["policies"][index]["metrics"]["mean_load_ratio"]
    refused(c.mutated(record, f"policies.{index}.metrics.mean_load_ratio", load + 1), "load_ratio")
    refused(c.mutated(record, "r3.0.metrics.average_precision", 0.5), "unexpected")
    refused(c.mutated(record, "ranking.metrics.roc_auc", 1.5), "[0, 1]")


def test_a_null_in_a_daily_row_is_reported_and_never_raises() -> None:
    record = c.evaluation()
    for path in (
        "r3.0.daily.0.overflow",
        "r3.0.daily.0.oracle_reviewable",
        "policies.0.daily.0.unused",
        "policies.0.daily.0.load_ratio",
    ):
        found = problems(c.mutated(record, path, None))
        assert found
        validated = c.mutated(record, path, None)
        with pytest.raises(RecordContractError):
            validate_record(validated, c.outcome_context())


def test_the_entries_describe_the_same_transactions() -> None:
    record = c.evaluation()
    refused(c.mutated(record, "r3.1.daily.0.transactions", 1), "must be those of the first entry")
    refused(c.mutated(record, "policies.0.daily.0.frauds", 0), "first entry")


def test_the_calibration_method_decides_the_role() -> None:
    refused(c.mutated(c.calibration(), "role", "sensitivity"), "Platt is the primary")
    refused(c.mutated(c.calibration("isotonic"), "role", "primary"), "Platt is the primary")


def test_calibration_diagnostics_are_checked() -> None:
    record = c.calibration()
    refused(c.mutated(record, "diagnostic.rows_used", "fitting"), "held out")
    refused(c.mutated(record, "diagnostic.reliability.0.observed_rate", 0.9), "observed_rate")
    refused(c.mutated(record, "intervals.slope.replicates", 3), "declared B")
    refused(c.without(record, "intervals.slope"), "intervals")


def without_records(kind: str, **match: Any) -> list[dict[str, Any]]:
    return [
        r
        for r in c.run_records()
        if not (r["record"] == kind and all(r.get(k) == v for k, v in match.items()))
    ]


def replaced(kind: str, new: dict[str, Any]) -> list[dict[str, Any]]:
    """The run with the first record of ``kind`` of the same subject replaced by ``new``."""
    return [
        new
        if r["record"] == kind
        and r["configuration"] == new["configuration"]
        and r.get("method") == new.get("method")
        else r
        for r in c.run_records()
    ]


def test_a_complete_run_holds_every_outcome_record_once() -> None:
    assert run_problems(c.run_records()) == []
    assert any(
        "every subject (16)" in p
        for p in run_problems(
            without_records("test_calibration", method="isotonic", configuration="RES-u")
        )
    )
    assert any(
        "test_workload record for every" in p
        for p in run_problems(without_records("test_workload"))
    )
    assert any(
        "test_evaluation record for every" in p
        for p in run_problems(without_records("test_evaluation"))
    )
    assert any("once for each" in p for p in run_problems([*c.run_records(), c.evaluation()]))


def test_the_outcomes_follow_a_completed_evaluation() -> None:
    early = [r for r in c.run_records() if r.get("phase") != "evaluation"]
    assert any("completed evaluation" in p for p in run_problems(early, complete=False))


def test_an_evaluation_names_the_score_file_of_its_configuration() -> None:
    wrong = replaced("test_evaluation", c.mutated(c.evaluation(), "scores_sha256", "0" * 64))
    assert any("another score file" in p for p in run_problems(wrong))


def test_a_calibration_names_a_frozen_calibrator() -> None:
    other = c.mutated(c.calibration(), "calibrator.path", "outputs/elsewhere")
    assert any("not a frozen input" in p for p in run_problems(replaced("test_calibration", other)))


def test_a_workload_describes_the_r2_alerts_of_the_evaluation() -> None:
    record = c.workload()
    alerts = record["entries"][1]["fidelity"]["terms"][0]["alerts"]
    changed = c.mutated(record, "entries.1.fidelity.terms.0.alerts", alerts + 1)
    assert any("other R2 alerts" in p for p in run_problems(replaced("test_workload", changed)))
    other = c.mutated(record, "entries.1.threshold", 0.77)
    assert any("another R2 threshold" in p for p in run_problems(replaced("test_workload", other)))


def test_the_kinds_are_registered_and_documented() -> None:
    for kind in (
        "test_evaluation",
        "test_workload",
        "test_calibration",
        "test_rq3",
        "test_false_negative",
    ):
        assert kind in RECORD_CONTRACTS
        assert f"### `{kind}`" in reference_markdown()
    assert isinstance(c.outcome_context(), ConfirmatoryContext)


def test_the_cost_identities_are_recomputed() -> None:
    record = c.rq3_record()
    row = record["capacities"][1]["grid"][3]
    refused(
        c.mutated(record, "capacities.1.grid.3.cost_difference", row["cost_difference"] + 1),
        "cost_difference",
    )
    refused(c.mutated(record, "capacities.1.grid.3.cost_r3", row["cost_r3"] + 1), "cost_r3")
    refused(c.mutated(record, "capacities.1.grid.3.threshold", 0.5), "threshold")
    refused(c.mutated(record, "capacities.1.queue.captured", 0), "grid")
    refused(c.mutated(record, "capacities.1.break_even.classification", "none"), "break_even")
    refused(c.mutated(record, "capacities.1.friction.0.ratio", 1.5), "friction")


def test_the_regime_and_intervals_of_a_cell_are_checked() -> None:
    record = c.rq3_record()
    cells = [(i, j) for i, e in enumerate(record["capacities"]) for j, r in enumerate(e["grid"])]
    i, j = next((i, j) for i, j in cells if record["capacities"][i]["grid"][j]["unused"] > 0)
    base = f"capacities.{i}.grid.{j}"
    wrong = (
        "r3_cheaper"
        if record["capacities"][i]["grid"][j]["regime"] != "r3_cheaper"
        else "identical"
    )
    refused(c.mutated(record, f"{base}.regime", wrong), "regime")
    refused(
        c.mutated(record, f"{base}.intervals.cost_difference.lower", 1e12), "lower exceeds upper"
    )
    refused(c.mutated(record, f"{base}.intervals.cost_difference.level", 0.9), "level")
    refused(c.mutated(record, f"{base}.intervals.cost_difference.replicates", 7), "declared B")
    refused(c.mutated(record, f"{base}.frauds_forgone_pigeonhole.replicates", 7), "declared B")
    refused(c.without(record, f"{base}.intervals.unused"), "intervals")
    refused(c.mutated(record, f"{base}.intervals.unused", None), "null exactly")


def test_every_declared_capacity_and_ratio_is_analysed() -> None:
    record = c.rq3_record()
    fewer = copy.deepcopy(record)
    fewer["capacities"] = fewer["capacities"][:-1]
    refused(fewer, "every declared capacity")
    shorter = copy.deepcopy(record)
    shorter["capacities"][0]["grid"] = shorter["capacities"][0]["grid"][:-1]
    refused(shorter, "grid")


def test_a_cost_analysis_belongs_to_the_run() -> None:
    assert any("every subject (16)" in p for p in run_problems(without_records("test_rq3")))
    other = c.mutated(c.rq3_record(), "calibrator_file.path", "outputs/elsewhere")
    assert any("not a frozen input" in p for p in run_problems(replaced_rq3(other)))
    wrong = c.mutated(c.rq3_record(), "scores_sha256", "0" * 64)
    assert any("another score file" in p for p in run_problems(replaced_rq3(wrong)))
    queue = c.mutated(c.rq3_record(), "capacities.0.queue.captured", 0)
    assert any("another R3 queue" in p for p in run_problems(replaced_rq3(queue)))
    assert any("once for each" in p for p in run_problems([*c.run_records(), c.rq3_record()]))


def replaced_rq3(new: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        new
        if r["record"] == "test_rq3"
        and (r["configuration"], r["calibrator"]) == (new["configuration"], new["calibrator"])
        else r
        for r in c.run_records()
    ]


def fn_index(record: dict[str, Any], variable: str, level: str | None = None) -> int:
    return next(
        i
        for i, row in enumerate(record["slices"])
        if row["variable"] == variable and (level is None or row["level"] == level)
    )


def test_the_slice_statistics_are_recomputed() -> None:
    record = c.false_negative_record()
    i = fn_index(record, "identity", "present")
    row = record["slices"][i]
    refused(c.mutated(record, f"slices.{i}.miss_rate", 0.99), "miss_rate")
    refused(c.mutated(record, f"slices.{i}.share", 0.99), "share")
    refused(c.mutated(record, f"slices.{i}.misses", row["misses"] + 1), "sum")
    refused(c.mutated(record, f"slices.{i}.misses", row["frauds"] + 1), "exceed")
    refused(c.mutated(record, "false_negatives", record["false_negatives"] + 1), "sum")
    refused(c.mutated(record, "frauds", record["frauds"] + 1), "sum")


def test_the_interval_rule_follows_the_minimum_number_of_frauds() -> None:
    record = c.false_negative_record()
    i = fn_index(record, "identity", "present")
    refused(c.mutated(record, f"slices.{i}.interval", None), "interval")
    refused(c.mutated(record, f"slices.{i}.status", "fewer_than_minimum_frauds"), "status")
    refused(c.mutated(record, f"slices.{i}.interval.level", 0.9), "level")
    refused(c.mutated(record, f"slices.{i}.interval.replicates", 7), "declared B")
    refused(c.mutated(record, f"slices.{i}.interval.lower", 2.0), "lower exceeds upper")


def test_the_score_band_has_no_miss_rate_or_interval() -> None:
    record = c.false_negative_record()
    i = fn_index(record, "score_band")
    row = record["slices"][i]
    assert row["miss_rate"] is None
    assert row["interval"] is None
    refused(c.mutated(record, f"slices.{i}.miss_rate", 1.0), "score band")
    refused(c.mutated(record, f"slices.{i}.status", "interval"), "score band")
    refused(c.mutated(record, f"slices.{i}.frauds", row["frauds"] + 1), "missed frauds only")


def test_the_boundaries_follow_the_declaration() -> None:
    record = c.false_negative_record()
    refused(c.mutated(record, "boundaries.amount_edges", [1.0, 2.0, 3.0]), "amount edges")
    edges = record["boundaries"]["amount_edges"]
    refused(c.mutated(record, "boundaries.amount_edges", edges[::-1]), "increasing")
    refused(c.mutated(record, "boundaries.kept.card4.C", 99), "fewer than 100 V2 frauds")
    refused(c.mutated(record, "days.last", 181), "Test partition")
    refused(c.mutated(record, "capacity", 0.05), "primary capacity")
    short = copy.deepcopy(record)
    short["slices"] = [r for r in short["slices"] if r["level"] != "band 5"]
    refused(short, "amount_band")


def test_a_false_negative_table_belongs_to_the_run() -> None:
    assert any(
        "every subject (8)" in p for p in run_problems(without_records("test_false_negative"))
    )
    wrong = c.mutated(c.false_negative_record(), "false_negatives", 0)
    assert any(
        "other frauds or false negatives" in p
        for p in run_problems(
            [
                wrong
                if r["record"] == "test_false_negative" and r["configuration"] == "LGB-b"
                else r
                for r in c.run_records()
            ]
        )
    )
    assert any(
        "once for each" in p for p in run_problems([*c.run_records(), c.false_negative_record()])
    )


def test_the_declared_levels_of_every_variable_are_required() -> None:
    record = c.false_negative_record()
    week = fn_index(record, "week")
    refused(c.mutated(record, f"slices.{week}.level", "127-133"), "week levels")
    refused(
        c.mutated(record, f"slices.{fn_index(record, 'identity', 'present')}.level", "linked"),
        "identity levels",
    )
    refused(
        c.mutated(record, f"slices.{fn_index(record, 'entity', 'new')}.level", "fresh"),
        "entity levels",
    )
    refused(
        c.mutated(record, f"slices.{fn_index(record, 'score_band')}.level", "near"),
        "score_band levels",
    )
    refused(
        c.mutated(record, f"slices.{fn_index(record, 'amount_band')}.level", "q1"), "amount_band"
    )


def test_a_miss_rate_interval_lies_between_zero_and_one() -> None:
    record = c.false_negative_record()
    i = fn_index(record, "identity", "present")
    interval = record["slices"][i]["interval"]
    refused(c.mutated(record, f"slices.{i}.interval.upper", 1.5), "[0, 1]")
    low = c.mutated(record, f"slices.{i}.interval.lower", -0.1)
    refused(low, "[0, 1]")
    both = c.mutated(
        c.mutated(record, f"slices.{i}.interval.lower", 2.0), f"slices.{i}.interval.upper", 3.0
    )
    refused(both, "[0, 1]")
    assert 0 <= interval["lower"] <= interval["upper"] <= 1


def test_the_configurations_share_their_slice_boundaries() -> None:
    other = c.mutated(c.false_negative_record("Cat-b"), "boundaries.amount_edges.0", 0.5)
    assert any("other slice boundaries" in p for p in run_problems(replaced_fn(other)))
    i = fn_index(c.false_negative_record(), "identity", "present")
    moved = c.false_negative_record("Cat-b")
    moved["slices"][i]["frauds"] += 1
    assert any("other slice boundaries" in p for p in run_problems(replaced_fn(moved)))
    score = c.false_negative_record("Cat-b")
    j = fn_index(score, "score_band")
    score["slices"][j]["frauds"] += 1  # the score band differs by configuration
    assert not any("other slice boundaries" in p for p in run_problems(replaced_fn(score)))


def replaced_fn(new: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        new
        if r["record"] == "test_false_negative" and r["configuration"] == new["configuration"]
        else r
        for r in c.run_records()
    ]
