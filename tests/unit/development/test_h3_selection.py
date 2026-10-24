"""The H3 configuration selection on the saved V2 results of eight configurations: synthetic
scores and records in one run (no model is trained, no project data)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from review_capacity.development.evaluation import truth_of
from review_capacity.development.h3 import select_latency, select_primary
from review_capacity.development.inputs import PartitionRows
from review_capacity.development.records import CONFIGURATION_NAMES
from review_capacity.development.run_output import RunDirectory, load_run
from review_capacity.development.scoring import SCORE_COLUMNS, SCORES_DIR, ScoredPartition
from review_capacity.development.trial import run_file
from review_capacity.errors import ModelError, RecordContractError
from review_capacity.evaluation.queue import DailyQueue
from review_capacity.evaluation.ranking import (
    Capacity,
    Subject,
    capacity_budgets,
    r3_outcome,
    r3_record,
    ranking_outcome,
    ranking_record,
)
from review_capacity.hashing import file_sha256
from support.development import IDENTITY, example, file
from support.search import V2_ROWS, partition, synthetic_partitions

PARTITIONS = synthetic_partitions()


@dataclass
class Eight:
    run: RunDirectory
    rows: PartitionRows
    scores: dict[str, np.ndarray]
    scored: dict[str, ScoredPartition]


def build_scores(rows: PartitionRows, winner: str | None) -> dict[str, np.ndarray]:
    """Random scores for each configuration; the ``winner`` ranks the frauds first."""
    rng = np.random.default_rng(5)
    scores = {}
    for name in CONFIGURATION_NAMES:
        noise = rng.random(rows.rows)
        shift = 2.0 if name == winner else 0.4
        scores[name] = np.clip(0.4 * noise + shift * rows.labels * 0.3, 0.0, 1.0)
    return scores


def build_run(
    root: Path,
    winner: str | None = "Cat-b",
    scores: dict[str, np.ndarray] | None = None,
    budget: int | None = None,
) -> Eight:
    """A run with the V2 score files and the R3 and ranking records of the eight configurations;
    ``budget`` writes every R3 record with that K instead of the V2 budget of 2 %."""
    run = RunDirectory.create(root, PARTITIONS, "development", "2026-10-20T091530Z")
    run.write_record(example("run"))
    rows = partition("v2", V2_ROWS, 3, 3_000_000)
    chosen = scores or build_scores(rows, winner)
    truth = truth_of(rows)
    queue = DailyQueue.build(rows.ids, truth.days)
    budgets = capacity_budgets(PARTITIONS.v2)
    scored: dict[str, ScoredPartition] = {}
    for name in CONFIGURATION_NAMES:
        path = run.path / SCORES_DIR / f"{name}-selected-V2.parquet"
        path.parent.mkdir(exist_ok=True)
        frame = pd.DataFrame({SCORE_COLUMNS[0]: rows.ids, SCORE_COLUMNS[1]: chosen[name]})
        frame.to_parquet(path, index=False)
        reference = file(f"records/score-{name}-selected-V2.json")
        subject = Subject(name, "selected", 42, reference, "V2")
        for capacity in budgets:
            used = capacity if budget is None else Capacity(Fraction(capacity.share), budget)
            outcome = r3_outcome(queue, truth, chosen[name], used)
            record = r3_record(IDENTITY, subject, truth, used, outcome)
            part = f"R3-c{round(capacity.value * 100):02d}"
            run.write_record(record, name, "selected", "V2", part)
        ranking = ranking_record(IDENTITY, subject, truth, ranking_outcome(truth, chosen[name]))
        run.write_record(ranking, name, "selected", "V2", "ranking")
        scored[name] = ScoredPartition(name, "V2", rows, run_file(run, path), reference, 0.1)
    return Eight(run, rows, chosen, scored)


def selection(eight: Eight, period: str) -> dict:  # type: ignore[type-arg]
    return next(
        r
        for r in load_run(eight.run.path, PARTITIONS).of_kind("h3_selection")
        if r["period"] == period
    )


def captured(rows: PartitionRows, scores: np.ndarray, k: int) -> int:
    """Frauds inside the daily top-k, computed directly (continuous scores: no ties)."""
    days = rows.times // 86_400
    total = 0
    for day in np.unique(days):
        member = np.flatnonzero(days == day)
        top = member[np.argsort(-scores[member], kind="stable")[:k]]
        total += int(rows.labels[top].sum())
    return total


def test_the_binding_selection_picks_the_configuration_with_the_most_v2_frauds(
    tmp_path: Path,
) -> None:
    eight = build_run(tmp_path, winner="Cat-b")
    path = select_primary(eight.run, PARTITIONS, IDENTITY)
    record = selection(eight, "primary")
    assert path.name == "h3_selection-primary.json"
    assert (record["period"], record["binding"], record["selected"]) == ("primary", True, "Cat-b")
    assert record["selection_days"] == {"first": 1, "last": 3}
    assert record["capacity"] == 0.02
    assert record["k"] == 3
    assert [c["configuration"] for c in record["candidates"]] == list(CONFIGURATION_NAMES)
    for candidate in record["candidates"]:
        name = candidate["configuration"]
        assert candidate["frauds_captured"] == captured(eight.rows, eight.scores[name], 3)
        assert candidate["average_precision"] == pytest.approx(
            average_precision_score(eight.rows.labels, eight.scores[name])
        )
    # with only nine reviews on the synthetic V2 several configurations can capture the same
    # frauds; the rule then decides by Average Precision, which the winner leads
    assert set(record["tie_break_steps"]) <= {"average_precision"}


def test_the_linked_records_are_in_the_run_the_selection_is_written_to(tmp_path: Path) -> None:
    eight = build_run(tmp_path, winner="Cat-b")
    select_primary(eight.run, PARTITIONS, IDENTITY)
    select_latency(eight.run, PARTITIONS, IDENTITY, 42, eight.scored)
    for period in ("primary", "latency"):
        for candidate in selection(eight, period)["candidates"]:
            linked = eight.run.path / candidate["evaluation"]["path"]
            assert linked.is_file(), candidate["evaluation"]["path"]
            assert file_sha256(linked) == candidate["evaluation"]["sha256"]
            record = next(
                r for n, r in load_run(eight.run.path, PARTITIONS).records
                if f"records/{n}" == candidate["evaluation"]["path"]
            )  # fmt: skip
            assert (record["configuration"], record["policy"]) == (candidate["configuration"], "R3")
            assert record["period"] == period


def test_identical_best_configurations_are_decided_by_the_fixed_order(tmp_path: Path) -> None:
    rows = partition("v2", V2_ROWS, 3, 3_000_000)
    scores = build_scores(rows, winner="Cat-u")
    scores["RES-u"] = scores["Cat-u"].copy()  # the same scores: the same count and precision
    eight = build_run(tmp_path, scores=scores)
    select_primary(eight.run, PARTITIONS, IDENTITY)
    record = selection(eight, "primary")
    assert record["selected"] == "Cat-u"  # before RES-u in the fixed order
    assert record["tie_break_steps"] == ["average_precision", "fixed_order"]


def test_equal_counts_are_decided_by_average_precision(tmp_path: Path) -> None:
    rows = partition("v2", V2_ROWS, 3, 3_000_000)
    base = build_scores(rows, winner="LGB-u")["LGB-u"]
    days = rows.times // 86_400
    queue = DailyQueue.build(rows.ids, days)
    top = queue.select(base, 3).mask
    rng = np.random.default_rng(1)
    scores = build_scores(rows, winner=None)
    scores["LGB-u"] = np.where(top, 0.5 + 0.5 * base, 0.5 * base)  # same top-k, ordered below
    scores["LGB-b"] = np.where(top, 0.5 + 0.5 * base, 0.5 * rng.random(rows.rows))
    # the two share the daily top-3 sets, so they capture the same frauds
    assert captured(rows, scores["LGB-u"], 3) == captured(rows, scores["LGB-b"], 3)
    ap_u = average_precision_score(rows.labels, scores["LGB-u"])
    ap_b = average_precision_score(rows.labels, scores["LGB-b"])
    assert ap_u != ap_b
    for name in CONFIGURATION_NAMES:
        if name not in ("LGB-u", "LGB-b"):
            scores[name] = rng.random(rows.rows) * 0.01  # clearly worse
    eight = build_run(tmp_path, scores=scores)
    select_primary(eight.run, PARTITIONS, IDENTITY)
    record = selection(eight, "primary")
    assert record["tie_break_steps"] == ["average_precision"]
    assert record["selected"] == ("LGB-u" if ap_u > ap_b else "LGB-b")


def test_a_budget_other_than_the_v2_budget_of_two_percent_is_refused(tmp_path: Path) -> None:
    eight = build_run(tmp_path, budget=1)  # the V2 budget of 2 % is 3
    with pytest.raises(RecordContractError, match="uses K = 1, not the 3"):
        select_primary(eight.run, PARTITIONS, IDENTITY)
    assert load_run(eight.run.path, PARTITIONS).of_kind("h3_selection") == []


def records_of(eight: Eight, policy: str, configuration: str) -> list[str]:
    return [
        n
        for n, r in load_run(eight.run.path, PARTITIONS).records
        if r["record"] == "evaluation"
        and r["configuration"] == configuration
        and r["policy"] == policy
    ]


@pytest.mark.parametrize("policy", ["R3", "ranking"])
def test_a_missing_or_repeated_record_stops_the_selection(tmp_path: Path, policy: str) -> None:
    eight = build_run(tmp_path / "missing")
    for name in records_of(eight, policy, "FTT-b"):
        if policy == "R3" and "c02" not in name:
            continue
        (eight.run.path / "records" / name).unlink()
    with pytest.raises(RecordContractError, match=rf"0 V2 {policy} evaluation records.*FTT-b"):
        select_primary(eight.run, PARTITIONS, IDENTITY)
    twice = build_run(tmp_path / "twice")
    original = next(
        n for n in records_of(twice, policy, "Cat-u") if policy == "ranking" or "c02" in n
    )
    source = twice.run.path / "records" / original
    copy = source.with_name(original.replace("selected", "selected-again"))
    copy.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(RecordContractError, match=rf"2 V2 {policy} evaluation records.*Cat-u"):
        select_primary(twice.run, PARTITIONS, IDENTITY)


def test_the_latency_selection_uses_only_the_fitting_days(tmp_path: Path) -> None:
    eight = build_run(tmp_path, winner="Cat-b")
    path = select_latency(eight.run, PARTITIONS, IDENTITY, 42, eight.scored)
    run = load_run(eight.run.path, PARTITIONS)
    record = selection(eight, "latency")
    assert path.name == "h3_selection-latency.json"
    assert (record["period"], record["binding"]) == ("latency", False)
    assert record["selection_days"] == {"first": 1, "last": 2}  # synthetic fit days 1–2
    days = eight.rows.times // 86_400
    sub = np.flatnonzero(days <= 2)
    for candidate in record["candidates"]:
        name = candidate["configuration"]
        assert candidate["frauds"] == int(eight.rows.labels[sub].sum())
        sub_rows = PartitionRows(
            "v2",
            eight.rows.features.iloc[sub],
            eight.rows.labels[sub],
            eight.rows.ids[sub],
            eight.rows.times[sub],
            eight.rows.provenance,
        )
        assert candidate["frauds_captured"] == captured(sub_rows, eight.scores[name][sub], 3)
        assert candidate["average_precision"] == pytest.approx(
            average_precision_score(eight.rows.labels[sub], eight.scores[name][sub])
        )
    evaluations = [r for r in run.of_kind("evaluation") if r["period"] == "latency"]
    assert len(evaluations) == 16  # an R3 and a ranking record for each configuration
    assert {(r["days"]["first"], r["days"]["last"]) for r in evaluations} == {(1, 2)}


def test_the_latency_selection_refuses_other_models_or_incomplete_input(tmp_path: Path) -> None:
    eight = build_run(tmp_path)
    names = list(eight.scored)
    with pytest.raises(ModelError, match="eight configurations in order"):
        select_latency(eight.run, PARTITIONS, IDENTITY, 42, {n: eight.scored[n] for n in names[:7]})
    refit = {**eight.scored}
    refit["FTT-b"] = replace(refit["FTT-b"], role="seed_refit", seed=43, tag="seed_refit-s43")
    with pytest.raises(ModelError, match="not those of the selected main-seed model"):
        select_latency(eight.run, PARTITIONS, IDENTITY, 42, refit)
