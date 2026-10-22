"""The generated Markdown summary of a run: a pure function of the records (synthetic records;
no data read)."""

from __future__ import annotations

from pathlib import Path

import pytest

from review_capacity.development.rendering import render_summary, write_summary
from review_capacity.development.run_output import SUMMARY_NAME, load_run
from review_capacity.development.summary import cell, record_markdown
from review_capacity.errors import RecordContractError
from review_capacity.training.timing import Timed, search_record
from support.development import IDENTITY, PARTITIONS, example
from support.runs import finished_run, new_run, write_records


def summary_of(path: Path) -> str:
    return render_summary(load_run(path, PARTITIONS))


def test_the_written_summary_is_the_rendering_of_the_records(tmp_path: Path) -> None:
    run = finished_run(tmp_path)
    text = (run.path / SUMMARY_NAME).read_text(encoding="utf-8")
    assert text == summary_of(run.path)


def test_the_summary_can_be_regenerated_after_an_interruption(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    write_records(run)  # the process is killed here: no run_end
    first = write_summary(run.path, PARTITIONS).read_text(encoding="utf-8")
    assert "interrupted before closing (no run_end record)" in first
    assert write_summary(run.path, PARTITIONS).read_text(encoding="utf-8") == first


def test_every_record_is_presented_in_full(tmp_path: Path) -> None:
    run = finished_run(tmp_path)
    text = summary_of(run.path)
    loaded = load_run(run.path, PARTITIONS)
    assert loaded.end is not None
    for name, record in [*loaded.records, ("run_end.json", loaded.end)]:
        assert f"`{name}`" in text
        assert record_markdown(record, omit=("identity",)).rstrip() in text


def test_the_outcome_is_stated(tmp_path: Path) -> None:
    run = finished_run(tmp_path)
    assert f"| `outcome` | {cell('completed')} |" in summary_of(run.path)


def test_comparison_values_are_the_record_values(tmp_path: Path) -> None:
    run = finished_run(tmp_path)
    text = summary_of(run.path)
    trial, configuration = example("trial"), example("configuration")
    row = next(line for line in text.splitlines() if line.startswith("| `LGB-u trial 1`"))
    assert cell(trial["best_objective"]) in row
    assert cell(trial["hyperparameters"]["num_leaves"]) in row
    assert cell(trial["runtime_seconds"]) in row
    selected = next(
        line
        for line in text.splitlines()
        if line.startswith("| `LGB-u` |") and cell(configuration["selected_length"]) in line
    )
    assert cell(configuration["selected_objective"]) in selected


def test_columns_follow_the_parameters_present_in_the_records(tmp_path: Path) -> None:
    text = summary_of(finished_run(tmp_path).path)
    header = next(line for line in text.splitlines() if line.startswith("| Configuration | Trial"))
    assert "| learning_rate | num_leaves |" in header


def seed_row(text: str, recall: float) -> str:
    return next(
        line
        for line in text.splitlines()
        if line.startswith("| `LGB-u` |") and cell(recall) in line
    )


def test_seed_columns_range_and_sesoi_are_derived_from_the_records(tmp_path: Path) -> None:
    run = finished_run(tmp_path)
    text = summary_of(run.path)
    main = example("evaluation")["metrics"]["recall"]
    header = next(line for line in text.splitlines() if "V2 Recall@2% by seed (seed 42)" in line)
    assert "V2 Recall@2% by seed (seed 43)" in header
    assert "V2 Recall@2% by seed (seed 44)" in header
    row = seed_row(text, 0.3)
    assert cell(main) in row
    assert cell(0.35) in row
    assert cell(max(main, 0.3, 0.35) - min(main, 0.3, 0.35)) in row
    assert cell("not below") in row


def test_a_range_below_the_sesoi_is_stated(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    write_records(run, {43: 0.34, 44: 0.345})
    row = seed_row(summary_of(run.path), 0.345)
    assert cell("below") in row
    assert cell("not below") not in row


def test_the_range_needs_every_declared_seed(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    write_records(run, {43: 0.34})  # seed 44 has no result: two seeds could not decide
    main = example("evaluation")["metrics"]["recall"]
    row = seed_row(summary_of(run.path), 0.34)
    assert cell(main - 0.34) not in row
    assert cell("below") not in row
    assert cell("not below") not in row


def test_a_configuration_without_records_shows_dashes(tmp_path: Path) -> None:
    text = summary_of(finished_run(tmp_path).path)
    row = next(line for line in text.splitlines() if line.startswith("| `Cat-b` |"))
    assert set(row.replace("|", "").split()) == {"`Cat-b`", "—"}


def test_several_records_matching_one_row_are_an_error(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    write_records(run)
    run.write_record(example("evaluation"), "LGB-u", "duplicate")
    with pytest.raises(RecordContractError, match="must identify one"):
        write_summary(run.path, PARTITIONS)


def test_conflicting_trial_records_are_an_error(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    write_records(run)
    run.write_record(example("trial") | {"runtime_seconds": 99.0}, "LGB-u", "again")
    with pytest.raises(RecordContractError, match="same trial"):
        write_summary(run.path, PARTITIONS)


def test_the_summary_never_marks_a_timing_as_not_valid(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    write_records(run)
    assert "Not valid for the computational-cost outcome" not in summary_of(run.path)


def test_an_interrupted_search_is_flagged_as_partial_in_the_summary(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    write_records(run)
    assert "Partial search times" not in summary_of(run.path)
    partial = search_record(
        IDENTITY,
        Timed("LGB-u", None, "cpu", threads=24, cpu_affinity=(0, 1)),
        {"confirmed": True},
        seconds=100.0,
        trials_completed=3,
        trials_declared=10,
        interrupted=True,
    )
    run.write_record(partial, "LGB-u", "search")
    text = summary_of(run.path)
    assert "**Partial search times:**" in text
    assert "LGB-u (3 of 10 trials)" in text
