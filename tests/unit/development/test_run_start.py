"""The start of a development run: identity, the ``run`` record and the adopted candidate files
(synthetic processed files and records; no project data)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from review_capacity.data.processed import JOINED
from review_capacity.development.cost_gate import ApprovedPlan
from review_capacity.development.records import DevelopmentPartitions, dump_record
from review_capacity.development.run_output import RunDirectory
from review_capacity.development.run_start import (
    adopt_candidates,
    processed_files,
    project_file,
    run_identity,
    run_record,
)
from review_capacity.errors import FileAccessError, SpecificationError
from review_capacity.hashing import file_sha256
from review_capacity.models.run_record import load_settings_with_snapshot
from review_capacity.training.declaration import load_declaration
from support.development import PARTITIONS, example
from support.runs import RUN_ID
from support.synthetic import REPO_PATHS, Synthetic, build_processed

STEMS = (JOINED, "train", "v1", "v2")


@pytest.fixture
def processed(tmp_path: Path) -> Synthetic:
    return build_processed(tmp_path)


def _record(data: Synthetic) -> tuple[dict[str, object], DevelopmentPartitions]:
    _, settings_file = load_settings_with_snapshot(REPO_PATHS)
    declaration, declaration_file = load_declaration(REPO_PATHS)
    code = example("run")["code"]
    identity = run_identity(
        RUN_ID, code, declaration, declaration_file, settings_file, 42, data.paths.manifest
    )
    record = run_record(
        data.paths,
        identity,
        "2026-10-20T09:15:30.123456Z",
        code,
        {"gpu": {"name": "GPU"}},
        settings_file,
        declaration_file,
        STEMS,
        ["LGB-u", "LGB-b"],
        {"path": "outputs/search_plan/x/cost_prediction.json", "sha256": "a" * 64},
    )
    return record, DevelopmentPartitions.from_spec(data.dataset, data.paths)


def test_the_identity_names_the_code_the_hashes_and_the_seeds(processed: Synthetic) -> None:
    record, _ = _record(processed)
    identity = record["identity"]
    assert isinstance(identity, dict)
    declaration, declaration_file = load_declaration(REPO_PATHS)
    assert identity["run_id"] == RUN_ID
    assert identity["run_type"] == "development"
    assert identity["declaration_sha256"] == declaration_file.sha256
    assert identity["settings_sha256"] == file_sha256(REPO_PATHS.models_spec)
    assert identity["dataset_manifest_sha256"] == file_sha256(processed.paths.manifest)
    assert identity["seeds"] == {
        "main": 42,
        "additional": list(declaration.additional_seeds),
        "sampler": declaration.sampler_seed,
        "preprocessing": declaration.preprocessing_seed,
    }


def test_the_run_record_is_valid_and_names_every_input(processed: Synthetic) -> None:
    record, partitions = _record(processed)
    dump_record(record, partitions)  # validates the contract
    assert record["configurations"] == ["LGB-u", "LGB-b"]
    dataset = record["dataset"]
    assert isinstance(dataset, dict)
    assert set(dataset["processed"]) == set(STEMS)
    assert dataset["manifest"]["path"] == "data/datapackage.json"
    assert dataset["processed"]["train"]["path"] == "data/processed/train.parquet"


def test_processed_files_carry_the_hash_of_their_verified_record(processed: Synthetic) -> None:
    files = processed_files(processed.paths, STEMS)
    for stem, entry in files.items():
        assert entry["sha256"] == file_sha256(processed.paths.processed_file(stem))


def test_a_project_file_is_relative_to_the_project_root(processed: Synthetic) -> None:
    entry = project_file(processed.paths, processed.paths.manifest)
    assert entry["path"] == "data/datapackage.json"
    with pytest.raises(ValueError, match="not in the subpath"):
        project_file(processed.paths, Path("/elsewhere/file.json"), "a" * 64)


def _approved(tmp_path: Path) -> ApprovedPlan:
    plan = tmp_path / "plan"
    (plan / "candidates").mkdir(parents=True)
    content = json.dumps({"model": "lightgbm"}).encode("utf-8")
    (plan / "candidates" / "lightgbm.json").write_bytes(content)
    entry = {"path": "candidates/lightgbm.json", "sha256": hashlib.sha256(content).hexdigest()}
    return ApprovedPlan(plan, "d" * 64, {"lightgbm": entry})


def test_the_approved_candidate_files_are_copied_into_the_run(tmp_path: Path) -> None:
    approved = _approved(tmp_path)
    run = RunDirectory.create(tmp_path / "root", PARTITIONS, "development", RUN_ID)
    adopted = adopt_candidates(run, approved, ["lightgbm"])
    copy = run.path / adopted["lightgbm"]["path"]
    assert adopted["lightgbm"]["path"] == "candidates/lightgbm.json"
    assert copy.read_bytes() == (approved.directory / "candidates" / "lightgbm.json").read_bytes()
    assert adopted["lightgbm"]["sha256"] == approved.candidates["lightgbm"]["sha256"]


def test_a_candidate_file_changed_after_approval_is_refused(tmp_path: Path) -> None:
    approved = _approved(tmp_path)
    (approved.directory / "candidates" / "lightgbm.json").write_text("{}\n", encoding="utf-8")
    run = RunDirectory.create(tmp_path / "root", PARTITIONS, "development", RUN_ID)
    with pytest.raises(SpecificationError, match="differs from the file the approved cost"):
        adopt_candidates(run, approved, ["lightgbm"])
    assert not (run.path / "candidates" / "lightgbm.json").exists()


def test_a_candidate_file_that_disappeared_is_a_file_access_error(tmp_path: Path) -> None:
    approved = _approved(tmp_path)
    (approved.directory / "candidates" / "lightgbm.json").unlink()
    run = RunDirectory.create(tmp_path / "root", PARTITIONS, "development", RUN_ID)
    with pytest.raises(FileAccessError, match="cannot be read"):
        adopt_candidates(run, approved, ["lightgbm"])


def test_a_model_without_an_approved_list_is_refused(tmp_path: Path) -> None:
    run = RunDirectory.create(tmp_path / "root", PARTITIONS, "development", RUN_ID)
    with pytest.raises(SpecificationError, match="no candidates for catboost"):
        adopt_candidates(run, _approved(tmp_path), ["catboost"])
