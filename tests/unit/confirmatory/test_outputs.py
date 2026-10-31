"""The summary, the figures and the promotion of a closed confirmatory run (Issue #209), on the
synthetic run of the command tests. No research data is read."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from review_capacity.confirmatory.command import TestRunOptions, run_test
from review_capacity.confirmatory.figures import (
    COST_FIGURE,
    FIGURES_DIR,
    REFERENCE_RATIO,
    UTILISATION_FIGURE,
    _curve,
    _value_below,
    render_figures,
)
from review_capacity.confirmatory.outputs import (
    promote,
    render_outputs,
    verify_promoted,
    write_outputs,
)
from review_capacity.confirmatory.records import ConfirmatoryRecords
from review_capacity.confirmatory.summary import render_summary
from review_capacity.development.run_output import LoadedRun, load_run
from review_capacity.errors import RecordContractError, SpecificationError
from support import command_stubs as stubs
from support import confirmatory as c

RECORDS = ConfirmatoryRecords(c.outcome_context())


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    project = stubs.install_evaluation_stubs(tmp_path, monkeypatch)
    options = TestRunOptions(
        tag=stubs.TAG,
        development=Path("outputs/development/x"),
        analyses=(Path("outputs/analysis/y"),),
        started=c.RUN_ID,
        phase="all",
    )
    directory = run_test(project["paths"], options, release=lambda: None)
    # the command line opens the log files of the run directory; here they are written directly
    (directory / "run.log").write_text("run log\n", encoding="utf-8")
    (directory / "resume-2026-11-02T091000Z.log").write_text("resume log\n", encoding="utf-8")
    return {**project, "directory": directory, "root": tmp_path}


def records_of(directory: Path) -> list[dict[str, Any]]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((directory / "records").glob("*.json"))
    ]


def test_the_run_is_closed_and_its_summary_and_figures_are_written_after_it(
    world: dict[str, Any],
) -> None:
    directory = world["directory"]
    assert (directory / "summary.md").is_file()
    assert (directory / FIGURES_DIR / COST_FIGURE).is_file()
    assert (directory / FIGURES_DIR / UTILISATION_FIGURE).is_file()
    end = json.loads((directory / "records" / "test_run_end.json").read_text(encoding="utf-8"))
    listed = [entry["path"] for entry in end["records"]]
    assert all(path.startswith("records/") for path in listed)  # the outputs are not listed


def test_the_summary_is_a_pure_function_of_the_records(world: dict[str, Any]) -> None:
    run = load_run(world["directory"], records=RECORDS)
    first, second = render_summary(run), render_summary(run)
    assert first == second
    assert (world["directory"] / "summary.md").read_text(encoding="utf-8") == first
    for heading in (
        "## Run",
        "## Primary confirmatory results",
        "## Sensitivities",
        "## Secondary outcomes",
        "## RQ2",
        "## Calibration diagnostics",
        "## RQ3",
        "## False-negative slices",
        "## Test scoring time",
        "## Records",
    ):
        assert heading in first


def test_the_cells_are_the_exact_literals_of_the_record_fields(world: dict[str, Any]) -> None:
    run = load_run(world["directory"], records=RECORDS)
    text = render_summary(run)
    records = records_of(world["directory"])
    inference = next(r for r in records if r["record"] == "test_inference")
    assert f"`{json.dumps(inference['h1']['estimate'])}`" in text
    assert f"`{json.dumps(inference['holm']['adjusted'][2])}`" in text
    evaluation = next(
        r for r in records if r["record"] == "test_evaluation" and r["configuration"] == "LGB-b"
    )
    recall = evaluation["r3"][1]["metrics"]["recall"]  # the primary capacity
    assert f"`{json.dumps(recall)}`" in text
    fn = next(
        r for r in records if r["record"] == "test_false_negative" and r["configuration"] == "LGB-b"
    )
    rate = next(s["miss_rate"] for s in fn["slices"] if s["miss_rate"] is not None)
    assert f"`{json.dumps(rate)}`" in text
    scores = next(r for r in records if r["record"] == "test_scores")
    assert f"`{json.dumps(scores['timing']['seconds'])}`" in text


def test_the_h3_configuration_comes_first_among_the_false_negative_slices(
    world: dict[str, Any],
) -> None:
    text = render_summary(load_run(world["directory"], records=RECORDS))
    section = text.split("## False-negative slices")[1].split("## Test scoring time")[0]
    headings = [line for line in section.splitlines() if line.startswith("### ")]
    assert headings[0] == "### LGB-b (the H3 configuration)"
    assert len(headings) == 8


def test_the_summary_makes_no_causal_or_miscalibration_claim(world: dict[str, Any]) -> None:
    text = render_summary(load_run(world["directory"], records=RECORDS))
    assert "causal" in text  # only in the caution that the analysis is non-causal
    assert "non-causal" in text
    assert "No threshold for a material miscalibration is applied here" in text
    assert "miscalibrated" not in text
    assert "not a same-device comparison" in text


def test_the_figures_are_reproducible_and_agree_with_the_break_even_values(
    world: dict[str, Any],
) -> None:
    run = load_run(world["directory"], records=RECORDS)
    records = [r for _, r in run.records]
    first, second = render_figures(records), render_figures(records)
    assert first == second  # byte for byte
    assert set(first) == {f"{FIGURES_DIR}/{COST_FIGURE}", f"{FIGURES_DIR}/{UTILISATION_FIGURE}"}
    for content in first.values():
        assert b"<svg" in content[:600]
        assert b"<dc:date>" not in content  # no date: the file is reproducible
    assert (world["directory"] / FIGURES_DIR / COST_FIGURE).read_bytes() == first[
        f"{FIGURES_DIR}/{COST_FIGURE}"
    ]
    checked = 0
    for record in records:
        if record["record"] != "test_rq3" or record["calibrator"] != "platt":
            continue
        for entry in record["capacities"]:
            ratio = entry["break_even"]["ratio"]
            if ratio is None:
                continue
            assert _value_below(entry, ratio) < 0  # R4 is cheaper just below the break-even ratio
            checked += 1
    assert checked > 0


def test_writing_the_outputs_again_gives_the_same_files(world: dict[str, Any]) -> None:
    directory = world["directory"]
    before = {
        p: p.read_bytes()
        for p in [directory / "summary.md", *(directory / FIGURES_DIR).glob("*.svg")]
    }
    written = write_outputs(directory, RECORDS)
    assert {p: p.read_bytes() for p in written} == before
    assert render_outputs(load_run(directory, records=RECORDS)).keys() == {
        "summary.md",
        f"{FIGURES_DIR}/{COST_FIGURE}",
        f"{FIGURES_DIR}/{UTILISATION_FIGURE}",
    }


def test_promotion_writes_exactly_the_allowlisted_files_with_a_manifest(
    world: dict[str, Any],
) -> None:
    destination = promote(world["directory"], world["root"], RECORDS)
    assert destination == world["root"] / "evidence" / "confirmatory" / c.RUN_ID
    manifest = verify_promoted(destination)
    paths = sorted(entry["path"] for entry in manifest["files"])
    assert "summary.md" in paths
    assert f"{FIGURES_DIR}/{COST_FIGURE}" in paths
    assert {"run.log", "resume-2026-11-02T091000Z.log"} <= set(paths)
    assert "records/test_run_end.json" in paths
    assert not any(p.startswith(("scores/", "evaluation/")) for p in paths)  # row-level stays local
    assert (destination / "manifest.json").is_file()
    with pytest.raises(SpecificationError, match="already exists; evidence is never replaced"):
        promote(world["directory"], world["root"], RECORDS)


def test_promotion_refuses_a_run_that_never_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = stubs.install_evaluation_stubs(tmp_path, monkeypatch)
    project["plan"].fail_before_marker = True
    options = TestRunOptions(
        tag=stubs.TAG,
        development=Path("outputs/development/x"),
        analyses=(Path("outputs/analysis/y"),),
        started=c.RUN_ID,
        phase="all",
    )
    with pytest.raises(RuntimeError):
        run_test(project["paths"], options, release=lambda: None)
    (directory,) = (tmp_path / "outputs" / "confirmatory").glob("20*")
    with pytest.raises(SpecificationError, match="never closed"):
        promote(directory, tmp_path, RECORDS)
    assert not (tmp_path / "evidence").exists()


def test_promotion_refuses_a_summary_or_figure_that_differs_from_the_records(
    world: dict[str, Any],
) -> None:
    directory = world["directory"]
    summary = directory / "summary.md"
    original = summary.read_bytes()
    summary.write_bytes(original + b"edited\n")
    with pytest.raises(SpecificationError, match="differs from the records"):
        promote(directory, world["root"], RECORDS)
    summary.write_bytes(original)
    figure = directory / FIGURES_DIR / COST_FIGURE
    figure.write_bytes(figure.read_bytes() + b"<!-- edited -->")
    with pytest.raises(SpecificationError, match="differs from the records"):
        promote(directory, world["root"], RECORDS)
    assert not (world["root"] / "evidence").exists()


def test_promotion_refuses_a_link_a_missing_file_and_an_evidence_root_outside_the_project(
    world: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> None:
    directory = world["directory"]
    log = directory / "run.log"
    saved = log.read_bytes()
    log.unlink()
    with pytest.raises(SpecificationError, match="regular file"):
        promote(directory, world["root"], RECORDS)  # no run.log
    other = tmp_path_factory.mktemp("elsewhere")
    (other / "log").write_bytes(saved)
    log.symlink_to(other / "log")
    with pytest.raises(SpecificationError, match="regular file"):
        promote(directory, world["root"], RECORDS)  # a link
    log.unlink()
    log.write_bytes(saved)
    with pytest.raises(SpecificationError, match="inside the project root"):
        promote(directory, world["root"], RECORDS, evidence_root=other)
    assert not (world["root"] / "evidence").exists()


def test_a_source_that_changes_while_it_is_copied_is_not_promoted(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = shutil.copyfile

    def changing(source: Any, target: Any) -> Any:
        result = real(source, target)
        if str(target).endswith("test_inference.json"):
            Path(target).write_text(
                Path(target).read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
        return result

    monkeypatch.setattr(shutil, "copyfile", changing)
    with pytest.raises((SpecificationError, RecordContractError)):
        promote(world["directory"], world["root"], RECORDS)
    evidence = world["root"] / "evidence"
    assert not (evidence / "confirmatory" / c.RUN_ID).exists()
    assert not any(evidence.rglob(".*.tmp")) if evidence.exists() else True


def test_reference_ratio_is_the_v2_prevalence_inverse() -> None:
    assert REFERENCE_RATIO == 77_822 / 2_637


def test_curve_points_cover_every_segment_end(world: dict[str, Any]) -> None:
    run = load_run(world["directory"], records=RECORDS)
    for _, record in run.records:
        if record["record"] != "test_rq3" or record["calibrator"] != "platt":
            continue
        entry = record["capacities"][0]
        for cost in (True, False):
            xs, ys = _curve(entry, cost)
            assert len(xs) == len(ys)
            assert np.all(np.diff(xs) >= 0)
            for segment in entry["curve"]:
                assert float(segment["start"]) in xs
                assert float(segment["end"]) in xs
        return


def _tables(text: str) -> list[list[list[str]]]:
    """The Markdown tables of a text: each a list of rows of cells (the divider row dropped)."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in [*text.splitlines(), ""]:
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if not all(set(cell) <= {"-", ":", " "} for cell in cells):
                current.append(cells)
        elif current:
            tables.append(current)
            current = []
    return tables


def test_every_table_row_has_as_many_cells_as_its_header(world: dict[str, Any]) -> None:
    tables = _tables(render_summary(load_run(world["directory"], records=RECORDS)))
    assert tables
    for table in tables:
        assert {len(row) for row in table} == {len(table[0])}, table[0]


def test_a_table_cell_holds_the_record_value_of_its_row_and_column(world: dict[str, Any]) -> None:
    text = render_summary(load_run(world["directory"], records=RECORDS))
    records = records_of(world["directory"])
    scores = next(r for r in records if r["record"] == "test_scores")
    key = f"`{json.dumps(scores['configuration'])}`"
    table = next(t for t in _tables(text) if "Seconds" in t[0] and any(r[0] == key for r in t))
    row = next(r for r in table if r[0] == key)
    assert row[table[0].index("Seconds")] == f"`{json.dumps(scores['timing']['seconds'])}`"


def test_a_summary_of_a_run_without_some_rq3_records_is_still_written(
    world: dict[str, Any],
) -> None:
    run = load_run(world["directory"], records=RECORDS)
    kept = [
        (path, record)
        for path, record in run.records
        if not (record["record"] == "test_rq3" and record["configuration"] != "LGB-b")
    ]
    partial = LoadedRun(run.path, kept, run.end)
    text = render_summary(partial)
    assert "## RQ3" in text
    assert text == render_summary(partial)


def test_a_log_that_changes_while_it_is_copied_is_not_promoted(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = shutil.copyfile

    def changing(source: Any, target: Any) -> Any:
        if str(source).endswith("run.log"):
            Path(source).write_text("run log\nmore\n", encoding="utf-8")
        return real(source, target)

    monkeypatch.setattr(shutil, "copyfile", changing)
    with pytest.raises(SpecificationError, match="changed while the run was being copied"):
        promote(world["directory"], world["root"], RECORDS)
    assert not (world["root"] / "evidence" / "confirmatory" / c.RUN_ID).exists()


def test_a_stray_file_in_the_run_directory_is_not_promoted(world: dict[str, Any]) -> None:
    (world["directory"] / "notes.txt").write_text("stray\n", encoding="utf-8")
    destination = promote(world["directory"], world["root"], RECORDS)
    assert not (destination / "notes.txt").exists()
    manifest = verify_promoted(destination)
    assert "notes.txt" not in {entry["path"] for entry in manifest["files"]}


def test_the_command_line_dispatches_a_confirmatory_directory(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from review_capacity import cli
    from review_capacity.paths import ProjectPaths

    directory = world["directory"]
    monkeypatch.setattr(cli, "_confirmatory_records", lambda _paths: RECORDS)
    monkeypatch.setattr(
        ProjectPaths, "discover", classmethod(lambda _cls, _root=None: world["paths"])
    )
    summary = directory / "summary.md"
    summary.unlink()
    assert cli._summarise_run(argparse.Namespace(root=None, run_directory=directory)) == 0
    assert summary.is_file()
    code = cli._promote_evidence(
        argparse.Namespace(root=None, run_directory=directory, evidence_root=None)
    )
    assert code == 0
    assert (world["root"] / "evidence" / "confirmatory" / c.RUN_ID / "manifest.json").is_file()


def test_the_summary_ends_with_the_symbols_and_notation_section(world: dict[str, Any]) -> None:
    text = render_summary(load_run(world["directory"], records=RECORDS))
    assert text.rstrip().splitlines().count("## Symbols and notation") == 1
    assert text.index("## Symbols and notation") > text.index("## Records")
