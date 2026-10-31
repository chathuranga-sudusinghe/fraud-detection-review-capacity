"""Phase 2 of the single Test evaluation, up to the outcome records (Issue #205; protocol,
Authorised Test evaluation).

``evaluate_outcomes`` runs on an open run whose scoring phase completed, with the lock of the
frozen state held by the caller, in the order the protocol fixes:

1. the saved scores are verified: the run's records, the single completed scoring, every score
   file by hash and the bundle hash the phase names;
2. the Test features are read again (no target) and their identifiers must be those of every
   score file; the frozen thresholds and calibrators are loaded and the development partitions
   read: everything that needs no Test label;
3. the consumption marker is created atomically (or accepted, for the run that created it: a
   resumed evaluation). **Only then are the Test labels reached**, with the explicit unsealing
   that this module alone passes;
4. the slice inputs are assembled; none of them uses a Test label to set a threshold, a
   calibrator or a slice boundary;
5. the outcome records are computed from the saved scores and published.

The inference record, the evaluation attempts and their resume, the check of the whole run and the
closing record are added by the command (Issue 9c). This module is the only one that reads labels.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from review_capacity.config import DatasetSpec, FeatureSpec
from review_capacity.confirmatory.frozen import FrozenBinding
from review_capacity.confirmatory.frozen_policies import FrozenPolicies, load_policies
from review_capacity.confirmatory.inputs import read_test_raw
from review_capacity.confirmatory.outcome_records import (
    OutcomeSetup,
    build_outcome_records,
    publish_outcomes,
)
from review_capacity.confirmatory.records import (
    ConfirmatoryContext,
    run_problems,
    scores_bundle_sha256,
)
from review_capacity.confirmatory.slices import (
    SliceInputs,
    build_slice_inputs,
    development_boundaries,
)
from review_capacity.confirmatory.state import begin_consumption, lock_path
from review_capacity.data.loading import RawData, load_labels
from review_capacity.development.inputs import TRAIN, V2, read_partitions
from review_capacity.development.records import ALL_CONFIGURATIONS
from review_capacity.development.run_output import RunDirectory
from review_capacity.development.scoring import SCORE_COLUMNS, check_sequence
from review_capacity.errors import DataValidationError, SpecificationError
from review_capacity.evaluation.declaration import EvaluationDeclaration
from review_capacity.evaluation.outcomes import Partition
from review_capacity.evaluation.queue import DailyQueue
from review_capacity.evaluation.ranking import Truth
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import log_step
from review_capacity.models.base import Keys, Scores, checked_scores
from review_capacity.paths import ProjectPaths

logger = logging.getLogger(__name__)

AMOUNT = "TransactionAmt"
SCORING_KINDS = ("test_run", "test_phase", "test_scores")


@dataclass(frozen=True)
class EvaluationSetup:
    """What the evaluation phase needs besides the open run."""

    paths: ProjectPaths
    run: RunDirectory
    binding: FrozenBinding
    evaluation: EvaluationDeclaration
    dataset: DatasetSpec
    features: FeatureSpec
    before_consumption: Callable[[], None] = lambda: None  # the attempt's intent, durably


@dataclass(frozen=True)
class EvaluationResult:
    """What the evaluation phase leaves for the steps that follow it (the inference record): the
    number of records written now, the Test partition with its labels and pseudo-entities, the
    verified scores, the frozen policies and the slice inputs."""

    written: int
    partition: Partition
    saved: SavedScores
    policies: FrozenPolicies
    slices: SliceInputs


@dataclass(frozen=True)
class SavedScores:
    """The verified scores of the single scoring pass."""

    scores: dict[str, Scores]
    ids: dict[str, Keys]
    files: dict[str, str]  # the SHA-256 of each score file
    bundle_sha256: str


def verify_saved_scores(run: RunDirectory, binding: FrozenBinding) -> SavedScores:
    """The saved scores of the completed scoring phase of ``run``, after every check of the run
    (the records of the run agree with each other and with the frozen inputs, one scoring
    completed, the bundle hash is the one the phase names, every score file is the recorded one)."""
    records = [run.records.read(path) for path in sorted(run.records_dir.glob("test_*.json"))]
    if run.identity is None or run.identity["frozen_state_sha256"] != binding.frozen_state_sha256:
        raise SpecificationError("The run belongs to another frozen state")
    foreign = sorted({r["record"] for r in records if r["identity"] != run.identity})
    if foreign:
        raise SpecificationError(f"Records of another identity in the run: {foreign}")
    # an evaluation interrupted while publishing leaves outcome records without a completed
    # evaluation phase: only the records of the scoring phase are checked here, and the whole run
    # is checked when the evaluation completes
    scoring = [r for r in records if r["record"] in SCORING_KINDS]
    problems = run_problems(scoring, complete=False)
    if problems:
        raise SpecificationError("The run is not consistent: " + "; ".join(problems))
    completed = [
        r
        for r in records
        if r["record"] == "test_phase" and r["phase"] == "scoring" and r["outcome"] == "completed"
    ]
    if len(completed) != 1:
        raise SpecificationError(
            f"The run holds {len(completed)} completed scoring phases, not one"
        )
    by_name = {r["configuration"]: r for r in records if r["record"] == "test_scores"}
    if set(by_name) != set(ALL_CONFIGURATIONS):
        raise SpecificationError("The run lacks a score record for every configuration")
    files = {name: record["file"]["sha256"] for name, record in by_name.items()}
    bundle = scores_bundle_sha256(list(files.values()))
    if completed[0]["scores_sha256"] != bundle:
        raise SpecificationError("The completed scoring names other scores than the records do")
    scores: dict[str, Scores] = {}
    ids: dict[str, Keys] = {}
    for name, record in by_name.items():
        path = run.path / record["file"]["path"]
        if file_sha256(path) != record["file"]["sha256"]:
            raise SpecificationError(f"{name}: the score file differs from its record")
        frame = pd.read_parquet(path)
        if tuple(frame.columns) != SCORE_COLUMNS:
            raise DataValidationError(f"{path}: columns {tuple(frame.columns)}")
        ids[name] = frame[SCORE_COLUMNS[0]].to_numpy(dtype=np.int64)
        scores[name] = checked_scores(frame[SCORE_COLUMNS[1]].to_numpy(), record["rows"], name)
    return SavedScores(scores, ids, files, bundle)


def _labels(
    raw: RawData, features: FeatureSpec, dataset: DatasetSpec, rows: int
) -> npt.NDArray[np.int8]:
    """The Test labels, read with the explicit unsealing; binary, one for each Test row."""
    series = load_labels(raw, features, dataset, ["test"], unseal_test_labels=True)
    labels = series.to_numpy(dtype=np.int8)
    if labels.shape != (rows,) or not np.isin(labels, (0, 1)).all():
        raise DataValidationError(
            f"The Test labels are not one binary value for each of {rows} rows"
        )
    return labels


def evaluate_outcomes(setup: EvaluationSetup) -> EvaluationResult:
    """Run the phase up to the outcome records. The caller holds the lock of the frozen state
    (the file is checked to exist; owning it is the caller's part)."""
    paths, run, binding = setup.paths, setup.run, setup.binding
    root, state = paths.root, binding.frozen_state_sha256
    if not lock_path(root, state).exists():
        raise SpecificationError("The evaluation needs the lock of the frozen state to be held")
    with log_step(logger, "Test evaluation: outcome records"):
        saved = verify_saved_scores(run, binding)
        inputs, raw = read_test_raw(setup.features, setup.dataset, binding.test_anchor_sha256)
        for name, ids in saved.ids.items():
            check_sequence(inputs.ids, ids, f"{name} scores")
        capacities = [float(share) for share in setup.evaluation.capacity.shares]
        policies = load_policies(root, binding.frozen_inputs, capacities)
        parts = read_partitions(paths, setup.features, setup.dataset, (TRAIN, V2))
        boundaries = development_boundaries(setup.evaluation, parts[TRAIN], parts[V2])
        # everything that needs no Test label is done: the evaluation is consumed now, and only
        # then is the first label reached
        setup.before_consumption()
        begin_consumption(
            root, state, run_id=(run.identity or {})["run_id"], scores_sha256=saved.bundle_sha256
        )
        labels = _labels(raw, setup.features, setup.dataset, inputs.rows)
        truth = Truth.build(inputs.days, labels, inputs.features[AMOUNT].to_numpy(dtype=np.float64))
        slices = build_slice_inputs(
            setup.evaluation, setup.features, inputs, labels, parts[TRAIN], boundaries
        )
        context = ConfirmatoryContext.from_specs(setup.dataset, setup.evaluation)
        partition = Partition(
            truth,
            DailyQueue.build(inputs.ids, inputs.days),
            slices.entity_index,
            context.reference_volume,
        )
        outcome_setup = OutcomeSetup(
            setup.evaluation,
            run.identity or {},
            int(inputs.days.min()),
            int(inputs.days.max()),
            saved.files,
            context.reference_volume,
        )
        records = build_outcome_records(outcome_setup, partition, saved.scores, policies, slices)
        written = publish_outcomes(run, records)
    return EvaluationResult(written, partition, saved, policies, slices)
