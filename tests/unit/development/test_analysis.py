"""The analysis of a completed development run: stages, provenance and failure handling on a
synthetic source run (no model is trained, no project data is read)."""

from __future__ import annotations

from pathlib import Path

import pytest

from review_capacity.development.analysis import (
    DEFAULT_STAGES,
    STAGES,
    check_stages,
    run_analysis,
)
from review_capacity.development.records import CONFIGURATION_NAMES
from review_capacity.development.run_output import SOURCE_LINK, load_run
from review_capacity.development.summary import cell
from review_capacity.errors import RecordContractError, SpecificationError
from review_capacity.hashing import file_sha256
from support.analysis import PARTITIONS, build_setup
from support.development import PARTITIONS as REAL_PARTITIONS


def test_the_stages_are_checked_and_ordered() -> None:
    assert check_stages(["h3", "thresholds"]) == ("thresholds", "h3")
    assert check_stages(list(STAGES)) == STAGES
    assert "refits" not in DEFAULT_STAGES  # it trains models: named explicitly
    with pytest.raises(SpecificationError, match="Name at least one"):
        check_stages([])
    with pytest.raises(SpecificationError, match="Not stages"):
        check_stages(["thresholds", "bogus"])


def test_the_threshold_stage_writes_its_records_and_the_run_closes(tmp_path: Path) -> None:
    setup = build_setup(tmp_path, ["thresholds"])
    directory = run_analysis(setup)
    run = load_run(directory, PARTITIONS)
    assert run.end is not None
    assert run.end["outcome"] == "completed"
    assert run.identity["run_type"] == "analysis"
    assert (directory / "summary.md").is_file()
    assert (directory / SOURCE_LINK).is_symlink()
    evaluations = [r for r in run.of_kind("evaluation") if r["policy"] in ("R0", "R1", "R2")]
    assert len(evaluations) == 8 * 20  # R0 on V1 and V2, R1, R1 latency and R2, four capacities
    assert len(run.of_kind("r1_selection")) == 16
    assert run.of_kind("run")[0]["source"]["run_id"] == "2026-10-20T091530Z"


def test_every_reference_into_the_source_resolves_through_the_link(tmp_path: Path) -> None:
    setup = build_setup(tmp_path, ["thresholds"], names=("LGB-u",))
    directory = run_analysis(setup)
    run = load_run(directory, PARTITIONS)
    scored = [r for r in run.of_kind("evaluation") if r["policy"] == "R2"]
    assert scored
    for record in scored:
        reference = record["score"]
        assert reference["path"].startswith("source/records/score-")
        assert file_sha256(directory / reference["path"]) == reference["sha256"]


def test_the_calibration_stage_writes_calibrators_r4_and_the_cost_analysis(tmp_path: Path) -> None:
    setup = build_setup(tmp_path, ["calibration"], names=("LGB-u", "Cat-u"))
    run = load_run(run_analysis(setup), PARTITIONS)
    assert len(run.of_kind("calibration")) == 2 * 4
    assert len([r for r in run.of_kind("evaluation") if r["policy"] == "R4"]) == 2 * 144
    assert len(run.of_kind("cost_analysis")) == 2 * 16
    assert all(r["score"]["path"].startswith("source/") for r in run.of_kind("cost_analysis"))
    assert all(r["artifact"]["path"].startswith("calibrators/") for r in run.of_kind("calibration"))


def test_the_h3_stage_selects_from_the_source_records(tmp_path: Path) -> None:
    setup = build_setup(tmp_path, ["h3"], winner="Cat-b")
    run = load_run(run_analysis(setup), PARTITIONS)
    primary = next(r for r in run.of_kind("h3_selection") if r["period"] == "primary")
    latency = next(r for r in run.of_kind("h3_selection") if r["period"] == "latency")
    assert primary["binding"] is True
    assert latency["binding"] is False
    assert primary["selected"] == "Cat-b"
    assert all(c["evaluation"]["path"].startswith("source/records/") for c in primary["candidates"])
    assert all(
        not c["evaluation"]["path"].startswith("source/") for c in latency["candidates"]
    )  # the latency records are written into the analysis run


def test_the_h3_stage_needs_the_eight_configurations(tmp_path: Path) -> None:
    setup = build_setup(tmp_path, ["h3"], names=("LGB-u", "Cat-u"))
    with pytest.raises(SpecificationError, match="needs the eight configurations"):
        run_analysis(setup)
    run = load_run(setup.run.path, PARTITIONS)
    assert run.end is not None
    assert run.end["outcome"] == "failed"  # the run still closes, with its summary
    assert (setup.run.path / "summary.md").is_file()


def test_the_source_run_is_never_changed(tmp_path: Path) -> None:
    setup = build_setup(tmp_path, ["thresholds", "calibration", "h3"], names=CONFIGURATION_NAMES)
    source = setup.source.path

    def snapshot() -> dict[str, str]:
        return {
            str(p.relative_to(source)): file_sha256(p)
            for p in sorted(source.rglob("*"))
            if p.is_file()
        }

    before = snapshot()
    run_analysis(setup)
    assert snapshot() == before  # no file added, removed or changed in the source run
    assert load_run(source, PARTITIONS).end is not None  # and it is still a valid closed run


@pytest.mark.usefixtures("restore_lightgbm_logger")
def test_the_refits_stage_refits_the_selected_setting_and_links_the_source_records(
    tmp_path: Path,
) -> None:
    from support.analysis import build_refit_setup

    setup = build_refit_setup(tmp_path)
    run = load_run(run_analysis(setup), PARTITIONS)
    assert run.end is not None
    assert run.end["outcome"] == "completed"
    assert sorted(r["purpose"] for r in run.of_kind("fit")) == [
        "same_seed_repeat",
        "seed_refit",
        "seed_refit",
    ]
    refits = {(r["kind"], r["seed"]): r for r in run.of_kind("refit")}
    assert set(refits) == {("seed_refit", 43), ("seed_refit", 44), ("same_seed_repeat", 42)}
    repeat = refits[("same_seed_repeat", 42)]
    assert repeat["comparison"]["identical"] is True  # LightGBM on the CPU is deterministic
    assert repeat["reference_score"]["path"].startswith("source/records/score-LGB-u-selected-V2")
    assert repeat["fit"]["path"].startswith("records/fit-")  # written into the analysis run
    selected = next(c for c in load_run(setup.source.path, PARTITIONS).of_kind("configuration"))
    assert repeat["selected_trial"] == selected["selected_trial"]
    assert repeat["length"] == selected["selected_length"]
    assert run.of_kind("device_comparison") == []  # LightGBM has no CPU–GPU pair


def test_the_summary_merges_the_source_records_and_shows_the_analysis_tables(
    tmp_path: Path,
) -> None:
    setup = build_setup(tmp_path, ["thresholds", "calibration", "h3"])
    directory = run_analysis(setup)
    text = (directory / "summary.md").read_text(encoding="utf-8")
    assert text.startswith("# Analysis run 2026-10-21T091530Z")
    for heading in (
        "### Threshold policies R0–R2 at 2 % capacity (V2)",
        "### Threshold workload against the fixed daily budget (2 % capacity, V2)",
        "### Rate and volume components of the R2 demand (2 % capacity, V2)",
        "### Calibration diagnostics (V2)",
        "### H3 configuration selection",
        "### RQ3 saturation and break-even ratios",
    ):
        assert heading in text
    assert "**Binding selection (all of V2)**" in text
    assert "`Cat-b`" in text  # the winner of the synthetic run
    # the development tables read the main-seed V2 evaluations, which live in the source run
    section = text.split("### Development metrics", 1)[1].split("\n### ", 1)[0]
    source = load_run(setup.source.path, PARTITIONS)
    for name in ("LGB-u", "Cat-b"):
        r3 = next(
            r
            for r in source.of_kind("evaluation")
            if (r["configuration"], r["policy"], r["capacity"]) == (name, "R3", 0.02)
        )
        row = next(line for line in section.splitlines() if line.startswith(f"| `{name}`"))
        assert cell(r3["metrics"]["recall"]) in row  # a value only the source run's record holds
    run = load_run(directory, PARTITIONS)
    r2 = next(
        r
        for r in run.of_kind("evaluation")
        if (r["configuration"], r["policy"], r["capacity"]) == ("LGB-u", "R2", 0.02)
        and r["partition"] == "V2"
    )
    outcomes = text.split("### Threshold policies R0–R2 at 2 % capacity (V2)", 1)[1].split(
        "\n### ", 1
    )[0]
    lgb = next(line for line in outcomes.splitlines() if line.startswith("| `LGB-u`"))
    assert cell(r2["counts"]["selected"]) in lgb
    assert cell(r2["metrics"]["f1"]) in lgb
    primary = next(r for r in run.of_kind("h3_selection") if r["period"] == "primary")
    winner = next(
        line for line in text.split("**Binding selection (all of V2)**", 1)[1].splitlines()
        if line.endswith("| yes |")
    )  # fmt: skip
    assert f"`{primary['selected']}`" in winner


def test_a_source_run_that_differs_from_the_one_the_run_record_names_is_refused(
    tmp_path: Path,
) -> None:
    from review_capacity.development.rendering import load_source_of

    setup = build_setup(tmp_path, ["thresholds"], names=("LGB-u",))
    run_analysis(setup)
    own = load_run(setup.run.path, PARTITIONS)
    assert load_source_of(own, setup.run.path, PARTITIONS) is not None
    end = setup.source.path / "records" / "run_end.json"
    end.write_text(end.read_text(encoding="utf-8") + " ", encoding="utf-8")  # a changed source
    with pytest.raises(RecordContractError, match="differs from the one its run record names"):
        load_source_of(own, setup.run.path, PARTITIONS)


def test_an_analysis_run_is_promoted_with_its_summary_checked_against_its_source(
    tmp_path: Path,
) -> None:
    from review_capacity.development.promotion import promote, verify_promoted

    setup = build_setup(tmp_path, ["thresholds", "h3"], names=CONFIGURATION_NAMES)
    directory = run_analysis(setup)
    (directory / "run.log").write_text("log\n", encoding="utf-8")  # written by the command
    destination = promote(directory, tmp_path, PARTITIONS)
    manifest = verify_promoted(destination)
    assert manifest["run_type"] == "analysis"
    names = {entry["path"] for entry in manifest["files"]}
    assert "records/h3_selection-primary.json" in names
    assert not any(name.startswith("source") for name in names)  # the source run is not copied
    (directory / "summary.md").write_text("changed", encoding="utf-8")
    with pytest.raises(SpecificationError, match="differs from the records"):
        promote(directory, tmp_path, PARTITIONS, evidence_root=tmp_path / "other-evidence")


def test_only_a_closed_completed_development_run_can_be_analysed(tmp_path: Path) -> None:
    from review_capacity.development.analysis import load_source
    from support.runs import new_run, write_records

    open_run = new_run(tmp_path / "open")
    write_records(open_run)
    with pytest.raises(SpecificationError, match="not closed"):
        load_source(open_run.path, REAL_PARTITIONS)
    failed = new_run(tmp_path / "failed")
    write_records(failed)
    failed.finish("failed", "boom")
    with pytest.raises(SpecificationError, match="failed"):
        load_source(failed.path, REAL_PARTITIONS)
    done = new_run(tmp_path / "done")
    write_records(done)
    done.finish("completed")
    assert load_source(done.path, REAL_PARTITIONS).end is not None


def test_the_files_the_analysis_loads_must_be_those_the_source_run_used(tmp_path: Path) -> None:
    from review_capacity.development.analysis import check_source
    from support.runs import new_run, write_records

    run = new_run(tmp_path)
    write_records(run)
    run.finish("completed")
    source = load_run(run.path, REAL_PARTITIONS)
    identity = source.identity
    recorded = source.of_kind("run")[0]["dataset"]["processed"]["train"]["sha256"]
    check_source(
        source,
        identity["settings_sha256"],
        identity["declaration_sha256"],
        identity["dataset_manifest_sha256"],
        {"train": recorded},
    )
    ok = (identity["settings_sha256"], identity["declaration_sha256"])
    manifest = identity["dataset_manifest_sha256"]
    with pytest.raises(SpecificationError, match="model settings"):
        check_source(source, "0" * 64, ok[1], manifest, {})
    with pytest.raises(SpecificationError, match="training declaration"):
        check_source(source, ok[0], "0" * 64, manifest, {})
    with pytest.raises(SpecificationError, match="dataset manifest"):
        check_source(source, ok[0], ok[1], "0" * 64, {})
    with pytest.raises(SpecificationError, match="processed train file"):
        check_source(source, ok[0], ok[1], manifest, {"train": "0" * 64})


def test_an_analysis_run_writes_inside_the_ignored_outputs_tree(tmp_path: Path) -> None:
    from review_capacity.development.analysis import AnalysisOptions, analysis_directory
    from support.synthetic import REPO_PATHS

    default = analysis_directory(REPO_PATHS, AnalysisOptions(source=tmp_path), "2026-10-21T091530Z")
    assert default == REPO_PATHS.root / "outputs" / "analysis" / "2026-10-21T091530Z"
    inside = AnalysisOptions(source=tmp_path, output=Path("outputs/mine"))
    assert analysis_directory(REPO_PATHS, inside, "x") == REPO_PATHS.root / "outputs" / "mine"
    outside = AnalysisOptions(source=tmp_path, output=tmp_path / "elsewhere")
    with pytest.raises(SpecificationError, match="inside outputs/"):
        analysis_directory(REPO_PATHS, outside, "x")


def test_an_analysis_may_not_write_inside_its_source_run() -> None:
    from review_capacity.development.analysis import AnalysisOptions, analysis_directory
    from support.synthetic import REPO_PATHS

    source = REPO_PATHS.root / "outputs" / "development" / "x"
    inside = AnalysisOptions(source=source, output=Path("outputs/development/x/analysis"))
    with pytest.raises(SpecificationError, match="never writes inside its source run"):
        analysis_directory(REPO_PATHS, inside, "y")
    same = AnalysisOptions(source=source, output=Path("outputs/development/x"))
    with pytest.raises(SpecificationError, match="never writes inside its source run"):
        analysis_directory(REPO_PATHS, same, "y")
    beside = AnalysisOptions(source=source, output=Path("outputs/development/x-analysis"))
    assert analysis_directory(REPO_PATHS, beside, "y").name == "x-analysis"


def test_the_joined_file_is_compared_with_the_source_runs_too(tmp_path: Path) -> None:
    from review_capacity.development.analysis import check_source
    from support.runs import new_run, write_records

    run = new_run(tmp_path)
    write_records(run)
    run.finish("completed")
    source = load_run(run.path, REAL_PARTITIONS)
    identity = source.identity
    processed = source.of_kind("run")[0]["dataset"]["processed"]
    common = (
        identity["settings_sha256"],
        identity["declaration_sha256"],
        identity["dataset_manifest_sha256"],
    )
    key = next(iter(processed))
    check_source(source, *common, {key: processed[key]["sha256"]})
    with pytest.raises(SpecificationError, match=f"processed {key} file"):
        check_source(source, *common, {key: "0" * 64})


def test_the_source_link_must_lead_to_the_recorded_source_run(tmp_path: Path) -> None:
    import shutil

    from review_capacity.development.rendering import load_source_of

    setup = build_setup(tmp_path, ["thresholds"], names=("LGB-u",))
    run_analysis(setup)
    own = load_run(setup.run.path, PARTITIONS)
    assert load_source_of(own, setup.run.path, PARTITIONS, tmp_path) is not None
    outside = tmp_path.parent / f"{tmp_path.name}-copy"
    shutil.copytree(setup.source.path, outside)  # an identical copy outside the project
    link = setup.run.path / SOURCE_LINK
    link.unlink()
    link.symlink_to(outside)
    assert load_source_of(own, setup.run.path, PARTITIONS) is not None  # the hash still matches
    with pytest.raises(RecordContractError, match="does not lead to the source run"):
        load_source_of(own, setup.run.path, PARTITIONS, tmp_path)


def test_a_source_run_of_a_custom_directory_name_can_be_analysed(tmp_path: Path) -> None:
    from review_capacity.development.rendering import load_source_of
    from support.analysis import SOURCE_ID

    setup = build_setup(tmp_path, ["thresholds"], names=("LGB-u",))
    run_analysis(setup)
    own = load_run(setup.run.path, PARTITIONS)
    assert own.of_kind("run")[0]["source"]["run_id"] == SOURCE_ID
    assert load_source_of(own, setup.run.path, PARTITIONS, tmp_path) is not None
