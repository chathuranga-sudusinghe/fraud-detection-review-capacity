"""The candidate-cost gate: a search starts only from the plan whose cost was approved (the
project's real declaration; no data read, no model fitted)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from review_capacity.development.cost_gate import ApprovedPlan, check_plan
from review_capacity.development.records import MODELS_SEARCHED
from review_capacity.errors import SpecificationError
from review_capacity.hashing import file_sha256
from review_capacity.models.run_record import load_settings_with_snapshot
from review_capacity.training import plan as search_plan
from review_capacity.training.declaration import TrainingDeclaration, load_declaration
from review_capacity.training.plan import COST_NAME, write_plan
from review_capacity.training.sampler import CandidateList, generate_candidates
from support.development import PARTITIONS
from support.synthetic import REPO_PATHS

STARTED = "2026-10-21T120000Z"
DECLARATION_FILE = load_declaration(REPO_PATHS)[1]

pytestmark = pytest.mark.usefixtures("restore_logging")


def build() -> tuple[TrainingDeclaration, dict[str, Any]]:
    """The project's declaration and the identity of a plan written from it."""
    settings, settings_file = load_settings_with_snapshot(REPO_PATHS)
    declaration, declaration_file = load_declaration(REPO_PATHS)
    run = search_plan.identity(
        REPO_PATHS.root,
        STARTED,
        declaration,
        declaration_file,
        settings_file,
        settings.main_seed,
        REPO_PATHS.manifest,
    )
    return declaration, run


@pytest.fixture
def planned(tmp_path: Path) -> tuple[Path, TrainingDeclaration, dict[str, CandidateList]]:
    declaration, run = build()
    plan = tmp_path / STARTED
    write_plan(plan, STARTED, run, declaration, PARTITIONS)
    return plan, declaration, generate_candidates(declaration)


def approve(plan: Path) -> str:
    return file_sha256(plan / COST_NAME)


def test_the_approved_plan_passes(
    planned: tuple[Path, TrainingDeclaration, dict[str, CandidateList]],
) -> None:
    plan, declaration, lists = planned
    approved = check_plan(plan, approve(plan), declaration, DECLARATION_FILE, lists, PARTITIONS)
    assert isinstance(approved, ApprovedPlan)
    assert approved.cost_sha256 == approve(plan)
    assert set(approved.candidates) == set(MODELS_SEARCHED)
    entry = approved.candidates["lightgbm"]
    assert entry["path"] == "candidates/lightgbm.json"
    assert entry["sha256"] == file_sha256(plan / entry["path"])


def test_a_cost_prediction_other_than_the_approved_one_is_refused(
    planned: tuple[Path, TrainingDeclaration, dict[str, CandidateList]],
) -> None:
    plan, declaration, lists = planned
    with pytest.raises(SpecificationError, match="not the approved cost prediction"):
        check_plan(plan, "0" * 64, declaration, DECLARATION_FILE, lists, PARTITIONS)


def test_a_candidate_file_changed_after_the_plan_is_refused(
    planned: tuple[Path, TrainingDeclaration, dict[str, CandidateList]],
) -> None:
    plan, declaration, lists = planned
    path = plan / "candidates" / "lightgbm.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(SpecificationError, match="differs from the file the cost record lists"):
        check_plan(plan, approve(plan), declaration, DECLARATION_FILE, lists, PARTITIONS)


def test_lists_regenerated_under_another_declaration_are_refused(
    planned: tuple[Path, TrainingDeclaration, dict[str, CandidateList]],
) -> None:
    plan, declaration, _ = planned
    other = replace(declaration, sampler_seed=declaration.sampler_seed + 1)
    regenerated = generate_candidates(other)
    with pytest.raises(SpecificationError, match="sampler seed"):
        check_plan(plan, approve(plan), other, DECLARATION_FILE, regenerated, PARTITIONS)


def test_a_changed_candidate_value_is_refused(
    planned: tuple[Path, TrainingDeclaration, dict[str, CandidateList]],
) -> None:
    plan, declaration, lists = planned
    first = lists["lightgbm"].candidates[1]
    listed = lists["lightgbm"].candidates
    changed = (listed[0], {**first, "num_leaves": 1}, *listed[2:])
    altered = {**lists, "lightgbm": replace(lists["lightgbm"], candidates=changed)}
    with pytest.raises(SpecificationError, match="differ from those the training declaration"):
        check_plan(plan, approve(plan), declaration, DECLARATION_FILE, altered, PARTITIONS)


def test_a_cost_that_differs_from_the_lists_is_refused_even_if_approved(
    planned: tuple[Path, TrainingDeclaration, dict[str, CandidateList]],
) -> None:
    plan, declaration, lists = planned
    path = plan / COST_NAME
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    record["estimates"][0]["basis"] = "a different basis"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(SpecificationError, match="predicted cost differs"):
        check_plan(plan, approve(plan), declaration, DECLARATION_FILE, lists, PARTITIONS)


def test_a_changed_declaration_is_refused_even_if_candidates_and_cost_are_unchanged(
    planned: tuple[Path, TrainingDeclaration, dict[str, CandidateList]],
) -> None:
    """A fixed setting or the V1 budget changes neither the candidates nor the tree cost."""
    plan, declaration, lists = planned
    edited = replace(DECLARATION_FILE, sha256="1" * 64)
    with pytest.raises(SpecificationError, match="declaration differs from the approved plan"):
        check_plan(plan, approve(plan), declaration, edited, lists, PARTITIONS)
