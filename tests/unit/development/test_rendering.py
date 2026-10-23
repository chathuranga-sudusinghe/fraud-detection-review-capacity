"""The generated Markdown summary of a run: a pure function of the records (synthetic records;
no data read)."""

from __future__ import annotations

from pathlib import Path

import pytest

from review_capacity.development.rendering import render_summary, write_summary
from review_capacity.development.run_output import SUMMARY_NAME, RunDirectory, load_run
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
    assert set(row.replace("|", "").split()) == {"`Cat-b`", "`tree`", "—"}


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


def test_memory_that_could_not_be_measured_is_listed_with_its_phase(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    write_records(run)
    assert "Memory not measured" not in summary_of(run.path)
    unmeasured = example("timing") | {"boundary": "end_to_end"}
    unmeasured["settings"] = {
        **unmeasured["settings"],
        "memory_problems": ["GPU memory not measured: sampling stalled"],
    }
    run.write_record(unmeasured, "RES-u", "latency-end-to-end")
    text = summary_of(run.path)
    assert "**Memory not measured**" in text
    assert "- RES-u latency end_to_end: GPU memory not measured: sampling stalled" in text


def test_memory_problems_are_listed_in_a_fixed_order(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    write_records(run)
    unmeasured = example("timing") | {"boundary": "end_to_end"}
    unmeasured["settings"] = {
        **unmeasured["settings"],
        "memory_problems": ["b problem", "a problem"],
    }
    run.write_record(unmeasured, "RES-u", "latency-end-to-end")
    text = summary_of(run.path)
    assert text.index("a problem") < text.index("b problem")


def test_a_fit_without_its_startup_excluded_is_flagged(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    write_records(run)
    assert "Fit time may include start-up" not in summary_of(run.path)
    fit = example("timing") | {"measurement": "fit", "model_role": "selected"}
    fit |= {"boundary": None, "latency_ms": None, "rows": None, "repetitions": None}
    fit |= {"warmup": None, "batch_rows": None, "rows_sha256": None, "row_selection": None}
    fit |= {"seconds": 12.5, "configuration": "Cat-u", "device": "gpu", "synchronised": True}
    fit["settings"] = {
        "procedure_version": 1,
        "session": {"load_average_1m": 0.25},
        "memory_problems": [],
        "completed_length": 10,
        "retained_length": 5,
        "startup_seconds": None,
        "startup_scope": "warm-up",
        "startup_status": "not_measured",
        "preprocessing_seconds": None,
        "persistence_seconds": 0.5,
        "host_memory_method": "clear_refs",
    }
    run.write_record(fit, "Cat-u", "fit")
    assert "Fit time may include start-up" in summary_of(run.path)


# --- the family column and the descriptive contrasts (Issue #153) --------------------------------

RECALLS = {
    "LGB-u": 0.30,
    "Cat-u": 0.34,
    "RES-u": 0.31,
    "FTT-u": 0.38,
    "LGB-b": 0.32,
    "Cat-b": 0.33,
    "RES-b": 0.35,
    "FTT-b": 0.36,
}


def write_recalls(run: RunDirectory, recalls: dict[str, float]) -> None:
    """The ``run`` record and, for each configuration, the V2 R3 evaluation at 2 % of its
    selected model with the given recall."""
    run.write_record(example("run"))
    for name, recall in recalls.items():
        record = example("evaluation") | {
            "configuration": name,
            "metrics": example("evaluation")["metrics"] | {"recall": recall},
        }
        run.write_record(record, name, "selected")


def contrast_rows(text: str) -> dict[str, list[str]]:
    """The cells of the contrast table, by contrast name (without the code span)."""
    start = text.index("### Descriptive contrasts of V2 Recall@2%")
    end = text.index("### ", start + 1)
    rows = {}
    for line in text[start:end].splitlines():
        if line.startswith("| `") and "---" not in line:
            cells = [part.strip() for part in line.strip("|").split("|")]
            rows[cells[0].strip("`")] = [c.strip("`") for c in cells[1:]]
    return rows


def numbers(cells: list[str]) -> list[float | None]:
    return [None if c == "—" else float(c) for c in cells]


def test_the_contrasts_are_exact_differences_of_the_recalls_of_the_records(
    tmp_path: Path,
) -> None:
    run = new_run(tmp_path)
    write_recalls(run, RECALLS)
    rows = contrast_rows(summary_of(run.path))
    assert list(rows) == [
        "CatBoost − LightGBM",
        "FT-Transformer − ResNet-style MLP",
        "Neural mean − tree-based mean",
    ]
    expected = {
        "CatBoost − LightGBM": [0.04, 0.01, 0.025],
        "FT-Transformer − ResNet-style MLP": [0.07, 0.01, 0.04],
        "Neural mean − tree-based mean": [0.025, 0.03, 0.0275],
    }
    for name, values in expected.items():
        assert numbers(rows[name]) == pytest.approx(values, abs=1e-12)


def test_the_contrast_table_states_that_it_is_descriptive(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    write_recalls(run, RECALLS)
    text = summary_of(run.path)
    assert "no interval, no test, no use in selection" in text
    assert "The logistic-regression reference enters no contrast." in text


def test_a_contrast_whose_configurations_were_not_scored_is_omitted_with_its_reason(
    tmp_path: Path,
) -> None:
    run = new_run(tmp_path)
    write_recalls(run, {"LGB-u": 0.30, "Cat-u": 0.34, "LGB-b": 0.32})  # no Cat-b
    text = summary_of(run.path)
    rows = contrast_rows(text)
    values = numbers(rows["CatBoost − LightGBM"])
    assert values[0] == pytest.approx(0.04, abs=1e-12)
    assert values[1:] == [None, None]
    assert numbers(rows["FT-Transformer − ResNet-style MLP"]) == [None, None, None]
    assert "**Contrasts not computed:**" in text
    assert "- CatBoost − LightGBM, balanced: not computed; no V2 Recall@2% record of" in text
    assert "Cat-b" in text.split("**Contrasts not computed:**")[1]


def test_the_reference_enters_no_contrast(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    write_recalls(run, RECALLS | {"LR": 0.99})
    rows = contrast_rows(summary_of(run.path))
    assert numbers(rows["Neural mean − tree-based mean"]) == pytest.approx(
        [0.025, 0.03, 0.0275], abs=1e-12
    )


def test_two_records_for_one_contrast_input_are_an_error(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    write_recalls(run, RECALLS)
    run.write_record(example("evaluation"), "LGB-u", "duplicate")
    with pytest.raises(RecordContractError, match="must identify one"):
        write_summary(run.path, PARTITIONS)


def test_every_configuration_row_names_its_model_family(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    write_recalls(run, RECALLS | {"LR": 0.2})
    text = summary_of(run.path)
    start = text.index("### Development metrics")
    section = text[start : text.index("### ", start + 1)]
    header = next(line for line in section.splitlines() if line.startswith("| Configuration"))
    assert header.startswith("| Configuration | Family | ")
    rows = {
        line.split("|")[1].strip().strip("`"): line.split("|")[2].strip().strip("`")
        for line in section.splitlines()
        if line.startswith("| `")
    }
    assert rows == {
        "LR": "reference",
        "LGB-u": "tree",
        "Cat-u": "tree",
        "RES-u": "neural",
        "FTT-u": "neural",
        "LGB-b": "tree",
        "Cat-b": "tree",
        "RES-b": "neural",
        "FTT-b": "neural",
    }
