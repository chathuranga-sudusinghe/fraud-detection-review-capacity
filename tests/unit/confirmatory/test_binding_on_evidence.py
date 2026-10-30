"""The record-level binding checks on the committed evidence of the development run and its analysis
runs (``evidence/``, aggregate records promoted by Issue #179). The model, calibrator and
preprocessing artifacts stay local and are not in the repository, so the file checks are tested on
synthetic files (``test_frozen_binding.py``); here the checks that read the records prove that the
field names and the coverage rules match the real runs. No research data is read."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from review_capacity.config import load_dataset_spec
from review_capacity.confirmatory.frozen import (
    binding_selections,
    check_analysis_runs,
    check_development_run,
    coverage_problems,
    model_references,
    record_ref,
)
from review_capacity.development.records import ALL_CONFIGURATIONS, DevelopmentPartitions
from review_capacity.development.run_output import LoadedRun, load_run
from review_capacity.evaluation.declaration import load_declaration
from review_capacity.hashing import file_sha256
from support.synthetic import REPO_PATHS

EVIDENCE = REPO_PATHS.root / "evidence"
DEVELOPMENT = EVIDENCE / "development" / "2026-10-25T134341Z"
ANALYSES = [
    EVIDENCE / "analysis" / "2026-10-29T013124Z",
    EVIDENCE / "analysis" / "2026-10-29T013414Z",
]
MODEL_FILE = re.compile(
    r"models/(?P<name>[A-Za-z-]+)/trial-\d{2}/(model\.(txt|cbm|pt)|arrays\.npz)"
)


@pytest.fixture(scope="module")
def partitions() -> DevelopmentPartitions:
    return DevelopmentPartitions.from_spec(load_dataset_spec(REPO_PATHS), REPO_PATHS)


@pytest.fixture(scope="module")
def development(partitions: DevelopmentPartitions) -> LoadedRun:
    return load_run(DEVELOPMENT, partitions)


@pytest.fixture(scope="module")
def analyses(partitions: DevelopmentPartitions) -> list[LoadedRun]:
    return [load_run(path, partitions) for path in ANALYSES]


@pytest.fixture(scope="module")
def capacities() -> list[float]:
    return [float(share) for share in load_declaration(REPO_PATHS)[0].capacity.shares]


def test_the_real_runs_pass_the_run_level_checks(
    development: LoadedRun, analyses: list[LoadedRun]
) -> None:
    check_development_run(development)
    assert check_analysis_runs(analyses, development)["selected"] == "LGB-b"
    assert [binding_selections(run)[0]["selected"] for run in analyses] == ["LGB-b", "LGB-b"]


def test_the_real_analysis_covers_the_frozen_policies(
    analyses: list[LoadedRun], capacities: list[float]
) -> None:
    assert capacities == [0.01, 0.02, 0.05, 0.1]
    assert coverage_problems(analyses, capacities) == []
    assert coverage_problems(analyses[:1], capacities) == []  # the first run alone holds it all
    assert len(coverage_problems(analyses[1:], capacities)) == 8 * (
        1 + 2 + 2 * 4
    )  # H3 and rolling only


def test_a_missing_or_wrong_record_is_found_in_the_real_set(
    analyses: list[LoadedRun], capacities: list[float]
) -> None:
    first = analyses[0]
    keep = [
        (n, r)
        for n, r in first.records
        if not (
            r["record"] == "evaluation"
            and r.get("policy") == "R2"
            and r.get("capacity") == 0.02
            and r.get("configuration") == "Cat-b"
            and r.get("partition") == "V2"
            and r.get("period") == "primary"
            and r.get("model_role") == "selected"
        )
    ]
    reduced = LoadedRun(first.path, keep, first.end)
    found = coverage_problems([reduced, *analyses[1:]], capacities)
    assert len(found) == 1
    assert "Cat-b" in found[0]
    assert "R2" in found[0]
    assert coverage_problems(analyses, [0.01, 0.02, 0.03]) != []  # a capacity that was never run


def test_every_selected_model_is_named_by_its_fit_and_run_record(development: LoadedRun) -> None:
    references = model_references(development)
    assert list(references) == list(ALL_CONFIGURATIONS)
    for name, reference in references.items():
        match = MODEL_FILE.fullmatch(reference.model["path"])
        assert match is not None, reference.model["path"]
        assert match["name"] == name
        assert (
            reference.record["path"]
            == f"models/{name}/{reference.model['path'].split('/')[2]}/model_run.json"
        )
        assert re.fullmatch(r"[0-9a-f]{64}", reference.model["sha256"])
        # the selected fit record is in the repository and is the one the configuration names
        assert file_sha256(DEVELOPMENT / reference.fit["path"]) == reference.fit["sha256"]


def test_a_record_of_a_closed_run_is_verified_against_its_closing_record(
    analyses: list[LoadedRun],
) -> None:
    reference = record_ref(analyses[0], "h3_selection-primary.json", REPO_PATHS.root)
    assert reference["path"] == (
        "evidence/analysis/2026-10-29T013124Z/records/h3_selection-primary.json"
    )
    path: Path = REPO_PATHS.root / reference["path"]
    assert file_sha256(path) == reference["sha256"]
