"""The candidate-cost gate of the development search (DR-015 §1 and §7; Issue #133).

``review-capacity plan-search`` writes the candidate lists and their predicted cost with the
decision ``pending_owner_decision``: whether the predicted cost is feasible is the owner's explicit
decision, taken before any run. The decision itself is not written into the record. The gate
makes it verifiable: the owner approves one ``cost_prediction.json`` by its SHA-256, and the
search starts only if

* the file in the plan directory has exactly that SHA-256;
* the training declaration in use is the one the plan was written from (its SHA-256, so that a
  changed fixed setting or V1 budget, which candidates and cost do not show, is caught);
* the plan's records are valid and their candidate files have the hashes the cost record lists;
* the candidate lists regenerated from the training declaration equal the plan's lists, value
  for value, so that the declaration in use is the one that was costed;
* the predicted cost recomputed from those lists equals the plan's.

The gate reads no data and fits no model. A failed check raises ``SpecificationError``: the
search never starts from lists, or at a cost, that differ from the approved plan.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from review_capacity.development.records import DevelopmentPartitions, read_record
from review_capacity.errors import SpecificationError
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import log_step
from review_capacity.models.run_record import SettingsSnapshot
from review_capacity.training.cost import predict_cost
from review_capacity.training.declaration import TrainingDeclaration
from review_capacity.training.plan import COST_NAME, cost_record
from review_capacity.training.sampler import CandidateList

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApprovedPlan:
    """A plan that passed the gate: its directory, the approved cost file's SHA-256 and the
    candidate file of each model (``{path, sha256}``, the path relative to the plan directory)."""

    directory: Path
    cost_sha256: str
    candidates: Mapping[str, Mapping[str, str]]


def _check_candidates(
    plan: Path,
    model: str,
    listed: CandidateList,
    entry: Mapping[str, str],
    declaration: TrainingDeclaration,
    partitions: DevelopmentPartitions,
) -> None:
    path = plan / entry["path"]
    if file_sha256(path) != entry["sha256"]:
        raise SpecificationError(f"{path}: differs from the file the cost record lists")
    record = read_record(path, partitions, "candidates")
    if record["model"] != model or record["sampler_seed"] != declaration.sampler_seed:
        raise SpecificationError(f"{path}: not the list of {model} under the declared sampler seed")
    planned = [(t["trial"], t["parameters"]) for t in record["trials"]]
    regenerated = list(enumerate(listed.candidates, start=1))
    if planned != regenerated:
        raise SpecificationError(
            f"{path}: the candidates differ from those the training declaration generates"
        )


def check_plan(
    plan: Path,
    approved_cost_sha256: str,
    declaration: TrainingDeclaration,
    declaration_file: SettingsSnapshot,
    lists: Mapping[str, CandidateList],
    partitions: DevelopmentPartitions,
) -> ApprovedPlan:
    """Pass the plan in ``plan`` if it is the one approved by ``approved_cost_sha256`` and agrees
    with ``declaration`` (parsed from the bytes of ``declaration_file``) and ``lists``
    (``training.sampler.generate_candidates``); raise ``SpecificationError`` otherwise."""
    with log_step(logger, f"Checking the approved search plan {plan}"):
        cost_path = plan / COST_NAME
        digest = file_sha256(cost_path)
        if digest != approved_cost_sha256:
            raise SpecificationError(
                f"{cost_path}: SHA-256 {digest} is not the approved cost prediction "
                f"{approved_cost_sha256}"
            )
        cost: dict[str, Any] = read_record(cost_path, partitions, "cost_prediction")
        if cost["identity"]["declaration_sha256"] != declaration_file.sha256:
            raise SpecificationError(
                f"{cost_path}: the training declaration differs from the approved plan"
            )
        files: dict[str, dict[str, str]] = cost["candidates"]
        if set(files) != set(lists):
            raise SpecificationError(
                f"{cost_path}: lists {sorted(files)} differ from the declared streams "
                f"{sorted(lists)}"
            )
        for model, listed in lists.items():
            _check_candidates(plan, model, listed, files[model], declaration, partitions)
        recomputed = cost_record(cost["identity"], predict_cost(declaration, dict(lists)), files)
        if (recomputed["estimates"], recomputed["totals"]) != (cost["estimates"], cost["totals"]):
            raise SpecificationError(
                f"{cost_path}: the predicted cost differs from that of the candidate lists"
            )
    logger.info("Search plan %s matches the approved cost prediction %s", plan, digest[:12])
    return ApprovedPlan(plan, digest, files)
