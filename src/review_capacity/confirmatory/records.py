"""Versioned contracts of the confirmatory Test run's JSON records (Issue #195).

Every record is one JSON object with the envelope of the development records (``record``,
``schema_version``, ``identity``) and kind-specific fields, validated by the same declarative
primitives (``development.schema``): no unknown or missing field, no duplicate key, no NaN or
infinite number. A value that cannot be estimated is ``null`` together with an explicit reason, and
a record that rests on replicates carries the number of undefined replicates.

The kinds, in the order a run produces them: ``test_run`` (written first: the identity, the frozen
inputs and the Test partition), ``test_phase`` (one per attempt of the scoring or the evaluation
phase), ``test_scores`` (one per model, from the single scoring pass), ``test_inference`` (H1, H2,
H3, Holm and the sensitivities), ``test_evaluation`` (R0 to R3 and the ranking measures of one
configuration or the reference, with intervals), ``test_workload`` (the daily-budget fidelity of
RQ2) and ``test_calibration`` (the diagnostics of one frozen calibrator), and ``test_run_end``
(closes the run). The RQ3 and false-negative kinds are defined with the code that produces them.

These contracts own the structure of the records and the rules that hold between their fields, not
the values of the study. Redundant values must agree: the Holm decisions, interval positions,
p-values of intersection-union tests, tolerance views and the frozen-state identity are recomputed
from the record's own numbers and from the evaluation declaration, and a record in which they
differ is refused. No record admits a development partition: a score record names the Test
partition of ``config/dataset.yaml`` and its days and rows must match it.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from itertools import pairwise
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from review_capacity.config import DatasetSpec, PartitionSpec
from review_capacity.confirmatory.identity import (
    artifact_hashes,
    declaration_hashes,
    frozen_state_id,
)
from review_capacity.development.records import (
    ALL_CONFIGURATIONS,
    CONFIGURATION,
    COST_ANALYSIS,
    COUNT,
    COUNTS,
    CUTOFF_TIES,
    DAY_ROW,
    DEVICE,
    DIAGNOSTIC,
    GIT_COMMIT,
    GRID_ROW,
    METRICS,
    POSITIVE,
    PROJECT_FILE,
    REFERENCE,
    RUN_FILE,
    RUN_ID,
    SECONDS,
    SEED,
    SHA256,
    STUDY_CONFIGURATION,
    TIMESTAMP,
    _break_even_problems,
    _curve_problems,
    _daily,
    _grid_problems,
    _reliability,
)
from review_capacity.development.schema import (
    AnyJSON,
    Boolean,
    Check,
    Choice,
    Field,
    Integer,
    ListOf,
    MapOf,
    Number,
    Object,
    RecordContract,
    Text,
    dumps,
    loads,
)
from review_capacity.errors import FileAccessError, RecordContractError
from review_capacity.evaluation.contrasts import estimands
from review_capacity.evaluation.declaration import EvaluationDeclaration
from review_capacity.evaluation.false_negatives import (
    OTHER as FN_OTHER,
)
from review_capacity.evaluation.false_negatives import (
    POOLED_VARIABLES as FN_POOLED,
)
from review_capacity.evaluation.false_negatives import (
    STATUSES as FN_STATUSES,
)
from review_capacity.evaluation.false_negatives import (
    VARIABLES as FN_VARIABLES,
)
from review_capacity.evaluation.false_negatives import (
    declared_levels as fn_declared_levels,
)
from review_capacity.evaluation.inference import (
    H1Category,
    Interval,
    Position,
    effect_units,
    h1_category,
    holm,
    interaction_splits_family_effect,
    position_of,
    smallest_tolerance,
)
from review_capacity.evaluation.mcnemar import exact_p

logger = logging.getLogger(__name__)

RUN_KIND = "test_run"
END_KIND = "test_run_end"
PHASES = ("scoring", "evaluation")


@dataclass(frozen=True)
class ConfirmatoryContext:
    """What the cross-field rules need besides the record: the Test partition of DR-001 and the
    evaluation declaration; the V2 partition gives the reference volume of the fidelity
    decomposition."""

    test: PartitionSpec
    v2: PartitionSpec
    declaration: EvaluationDeclaration

    @classmethod
    def from_specs(
        cls, dataset: DatasetSpec, declaration: EvaluationDeclaration
    ) -> ConfirmatoryContext:
        return cls(dataset.partition("test"), dataset.partition("v2"), declaration)

    @property
    def reference_volume(self) -> float:
        """The mean daily V2 volume N̄_V2 that fixed the budgets K: V2 rows over V2 days."""
        return self.v2.rows / (self.v2.last_day - self.v2.first_day + 1)


type ContextCheck = Callable[[Mapping[str, Any], ConfirmatoryContext], list[str]]

# --- shared value types --------------------------------------------------------------------------

POSITION = Choice(tuple(position.value for position in Position))
H1_CATEGORY = Choice(tuple(category.value for category in H1Category))
UNIT_P = Number(0, 1)
PVALUE = Object(
    (
        Field("value", UNIT_P, "The add-one bootstrap p-value (DR-006)"),
        Field("mc_se", Number(minimum=0), "Its Monte Carlo standard error (DR-006)"),
    )
)
INTERVAL = Object(
    (
        Field("lower", Number(), "Lower endpoint"),
        Field("upper", Number(), "Upper endpoint"),
        Field("level", Number(0, 1, exclusive_minimum=True), "Interval level"),
        Field("undefined", COUNT, "Replicates excluded as undefined"),
        Field("replicates", COUNT, "Replicates drawn (0 for a Wald interval)"),
    )
)
ESTIMAND = Object(
    (
        Field("name", Text(), "Estimand name (evaluation.contrasts)"),
        Field("estimate", Number(), "Observed value"),
        Field("interval", INTERVAL, "95 % percentile interval"),
    )
)
HOLM = Object(
    (
        Field(
            "adjusted", ListOf(UNIT_P, min_items=3, max_items=3), "Adjusted p-values: H1, H2, H3"
        ),
        Field("rejected", ListOf(Boolean(), min_items=3, max_items=3), "Rejections: H1, H2, H3"),
    )
)
COMPARISON = Object(
    (
        Field("p_h1", UNIT_P, "p_H1 of the sensitivity analysis"),
        Field("p_h2", UNIT_P, "p_H2 of the sensitivity analysis"),
        Field("holm", HOLM, "Holm re-applied with p_H3 held at its primary value"),
        Field("h1_rejected", Boolean(), "H1 rejected under the sensitivity analysis"),
        Field("h2_rejected", Boolean(), "H2 rejected under the sensitivity analysis"),
        Field("h1_category", H1_CATEGORY, "H1 practical category under the sensitivity analysis"),
        Field("differs", Boolean(), "A rejection decision or the H1 category differs"),
    )
)
H3_VARIANT = Object(
    (
        Field("block_length", POSITIVE, "Block length of the bootstrap"),
        Field("p_lower", PVALUE, "H0a: r ≤ target − Δ"),
        Field("p_upper", PVALUE, "H0b: r ≥ target + Δ"),
        Field("p", PVALUE, "The larger of the two"),
        Field("interval", INTERVAL, "90 % interval of r"),
        Field("position", POSITION, "Position against the open band"),
    )
)
RUN_PHASE = Choice(PHASES)
CONFIGURATION_LIST = ListOf(CONFIGURATION, min_items=1, unique=True)


IDENTITY = Object(
    (
        Field("run_id", RUN_ID, "Run identifier: the UTC start time naming the run directory"),
        Field("run_type", Choice(("confirmatory",)), "Run type (the outputs/ subdirectory)"),
        Field(
            "evaluator",
            Object(
                (
                    Field("code_commit", GIT_COMMIT, "Git commit of the evaluating code"),
                    Field("uncommitted_changes", Boolean(), "Whether the tree had changes"),
                    Field("source_sha256", SHA256, "Fingerprint of the evaluating package"),
                )
            ),
            "The frozen evaluator: a clean tree at the commit of the freeze tag",
        ),
        Field(
            "freeze",
            Object(
                (
                    Field("tag", Text(), "Name of the annotated protocol freeze tag"),
                    Field("tag_commit", GIT_COMMIT, "Commit the tag points to"),
                )
            ),
            "The protocol freeze that authorises the evaluation",
        ),
        Field(
            "training",
            Object(
                (
                    Field("run_id", RUN_ID, "Development run that trained the frozen models"),
                    Field("code_commit", GIT_COMMIT, "Its code commit", nullable=True),
                    Field("source_sha256", SHA256, "Its package fingerprint"),
                )
            ),
            "The training identity, recorded separately from the evaluator's",
        ),
        Field("settings_sha256", SHA256, "SHA-256 of config/models.yaml as loaded"),
        Field("training_declaration_sha256", SHA256, "SHA-256 of the training declaration"),
        Field("evaluation_declaration_sha256", SHA256, "SHA-256 of the evaluation declaration"),
        Field("dataset_manifest_sha256", SHA256, "SHA-256 of the dataset manifest"),
        Field(
            "seeds",
            Object(
                (
                    Field("main", SEED, "Main training seed of the frozen models"),
                    Field("additional", ListOf(SEED, unique=True), "Additional training seeds"),
                    Field("sampler", SEED, "Seed of the candidate sampler"),
                    Field("preprocessing", SEED, "Seed of the neural preprocessing noise"),
                    Field("resampling", SEED, "Root seed of the bootstrap streams (DR-006)"),
                )
            ),
            "Seeds of the frozen training and of the Test resampling",
        ),
        Field(
            "frozen_state_sha256",
            SHA256,
            "Identity of the frozen state: tag commit, artifact and declaration hashes (not the "
            "tag name)",
        ),
    )
)


def _contract(
    kind: str, version: int, meaning: str, fields: tuple[Field, ...], *checks: Check
) -> RecordContract:
    envelope = (
        Field("record", Choice((kind,)), "Record kind"),
        Field("schema_version", Integer(version, version), "Contract version of this kind"),
        Field("identity", IDENTITY, "Run identity (identical in every record of a run)"),
    )
    return RecordContract(
        kind, version, meaning, Object((*envelope, *fields)), (_identity_rules, *checks)
    )


def _close(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-12)


def _near(first: float, second: float) -> bool:
    """Equal up to the rounding of two implementations of the same formula."""
    return math.isclose(first, second, rel_tol=1e-9, abs_tol=1e-12)


def _identity_rules(record: Mapping[str, Any]) -> list[str]:
    identity = record["identity"]
    evaluator, freeze = identity["evaluator"], identity["freeze"]
    problems = []
    if evaluator["uncommitted_changes"]:
        problems.append(f"{record['record']}.identity.evaluator: the working tree must be clean")
    if evaluator["code_commit"] != freeze["tag_commit"]:
        problems.append(
            f"{record['record']}.identity.evaluator.code_commit: "
            "must be the commit of the freeze tag"
        )
    return problems


# --- test_run ------------------------------------------------------------------------------------

FROZEN_INPUTS = Object(
    (
        Field(
            "development_run",
            Object(
                (
                    Field("run_id", RUN_ID, "Run identifier"),
                    Field("run_end", PROJECT_FILE, "The closed run's closing record"),
                )
            ),
            "The closed, completed development run that trained and selected the models",
        ),
        Field(
            "analysis_runs",
            ListOf(
                Object(
                    (
                        Field("run_id", RUN_ID, "Run identifier"),
                        Field("run_end", PROJECT_FILE, "The closed run's closing record"),
                    )
                ),
                min_items=1,
                unique=True,
            ),
            "The closed analysis runs that fixed the thresholds, calibrators and H3 selection",
        ),
        Field("h3_selection", PROJECT_FILE, "The binding H3 selection record (DR-010)"),
        Field("h3_configuration", STUDY_CONFIGURATION, "The configuration that record selects"),
        Field("preprocessing", PROJECT_FILE, "The shared neural preprocessing artifact manifest"),
        Field("preprocessing_arrays", PROJECT_FILE, "The arrays file beside that manifest"),
        Field("models", MapOf(PROJECT_FILE), "The frozen main-seed model of each configuration"),
        Field(
            "model_records",
            MapOf(PROJECT_FILE),
            "The model-run record of each model, as its fit recorded it",
        ),
        Field(
            "model_manifests",
            MapOf(PROJECT_FILE),
            "The manifest.json beside each saved model (frozen when the inputs are bound)",
        ),
        Field(
            "thresholds",
            ListOf(PROJECT_FILE, min_items=1, unique=True),
            "The records fixing the V2 R1 and R2 thresholds",
        ),
        Field(
            "calibrators",
            ListOf(PROJECT_FILE, min_items=1, unique=True),
            "The records and artifacts of the frozen V2 calibrators",
        ),
    )
)
TEST_PARTITION = Object(
    (
        Field("first_day", POSITIVE, "First relative day of the Test period"),
        Field("last_day", POSITIVE, "Last relative day of the Test period"),
        Field("rows", POSITIVE, "Test transactions"),
        Field(
            "transaction_id_sha256",
            SHA256,
            "Identity of the Test partition from the dataset manifest, verified before the "
            "labels are read",
        ),
    )
)


def _run_consistency(record: Mapping[str, Any]) -> list[str]:
    identity = record["identity"]
    pairs = (
        ("settings_sha256", record["settings"]["sha256"]),
        ("training_declaration_sha256", record["training_declaration"]["sha256"]),
        ("evaluation_declaration_sha256", record["evaluation_declaration"]["sha256"]),
        ("dataset_manifest_sha256", record["dataset_manifest"]["sha256"]),
    )
    problems = [
        f"test_run.identity.{name}: {identity[name]!r} differs from the run's own value {value!r}"
        for name, value in pairs
        if identity[name] != value
    ]
    expected = frozen_state_id(
        commit=identity["freeze"]["tag_commit"],
        declarations=declaration_hashes(identity),
        artifacts=artifact_hashes(record["frozen_inputs"]),
    )
    if identity["frozen_state_sha256"] != expected:
        problems.append(
            "test_run.identity.frozen_state_sha256: differs from the identity of the frozen "
            f"inputs ({expected})"
        )
    problems += [
        f"test_run.frozen_inputs.{name}: must hold exactly {sorted(ALL_CONFIGURATIONS)}"
        for name in ("models", "model_records", "model_manifests")
        if set(record["frozen_inputs"][name]) != set(ALL_CONFIGURATIONS)
    ]
    return problems


def _run_contracts(record: Mapping[str, Any]) -> list[str]:
    expected = contract_versions()
    if record["contracts"] != expected:
        return [f"test_run.contracts: {record['contracts']} differs from the current {expected}"]
    return []


def _run_partition(record: Mapping[str, Any], context: ConfirmatoryContext) -> list[str]:
    block, test = record["test"], context.test
    problems = [
        f"test_run.test.{name}: must be {expected}"
        for name, expected in (
            ("first_day", test.first_day),
            ("last_day", test.last_day),
            ("rows", test.rows),
        )
        if block[name] != expected
    ]
    if not block["transaction_id_sha256"].startswith(test.transaction_id_sha256_prefix):
        problems.append("test_run.test.transaction_id_sha256: differs from the declared anchor")
    return problems


def _seed_rule(record: Mapping[str, Any], context: ConfirmatoryContext) -> list[str]:
    seed = record["identity"]["seeds"]["resampling"]
    root = context.declaration.resampling.root_seed
    if seed != root:
        return [
            f"{record['record']}.identity.seeds.resampling: must be the declared root seed {root}"
        ]
    return []


TEST_RUN = _contract(
    "test_run",
    1,
    "The confirmatory run's provenance, written first: the frozen evaluator and training "
    "identities, the freeze tag, the frozen inputs and the Test partition; every other record "
    "repeats its identity.",
    (
        Field("started", TIMESTAMP, "When the run started"),
        Field("environment", MapOf(AnyJSON()), "Run environment"),
        Field("settings", PROJECT_FILE, "The loaded config/models.yaml"),
        Field("training_declaration", PROJECT_FILE, "The loaded training declaration"),
        Field("evaluation_declaration", PROJECT_FILE, "The loaded evaluation declaration"),
        Field("dataset_manifest", PROJECT_FILE, "The dataset manifest"),
        Field("frozen_inputs", FROZEN_INPUTS, "Every frozen artifact the evaluation uses"),
        Field("test", TEST_PARTITION, "The Test partition as declared"),
        Field("contracts", MapOf(POSITIVE), "Contract version of every record kind written"),
    ),
    _run_consistency,
    _run_contracts,
)

# --- test_phase ----------------------------------------------------------------------------------


def _phase_rules(record: Mapping[str, Any]) -> list[str]:
    phase, outcome = record["phase"], record["outcome"]
    problems = []
    if phase == "scoring" and record["labels_accessed"]:
        problems.append("test_phase.labels_accessed: the scoring phase never reaches the labels")
    if phase == "evaluation" and not record["labels_accessed"]:
        problems.append("test_phase.labels_accessed: the evaluation phase reaches the labels")
    if outcome == "completed" and record["error"] is not None:
        problems.append("test_phase.error: must be null when the attempt completed")
    if outcome != "completed" and record["error"] is None:
        problems.append(f"test_phase.error: required when the attempt {outcome}")
    bundle = record["scores_sha256"]
    if phase == "evaluation" and bundle is None:
        problems.append("test_phase.scores_sha256: the evaluation phase names the scores it used")
    if phase == "scoring" and (bundle is not None) != (outcome == "completed"):
        problems.append("test_phase.scores_sha256: given exactly when the scoring completed")
    return problems


TEST_PHASE = _contract(
    "test_phase",
    1,
    "One attempt of a phase of the single evaluation: phase 1 scores the Test features without "
    "labels, phase 2 reads the labels and computes every outcome from the saved scores. A failed "
    "scoring attempt may be repeated; once the labels are read the evaluation is consumed and a "
    "failed evaluation attempt resumes from the saved scores.",
    (
        Field("phase", RUN_PHASE, "Scoring (phase 1) or evaluation (phase 2)"),
        Field("attempt", POSITIVE, "Attempt number within the phase, from 1"),
        Field("started", TIMESTAMP, "When the attempt started"),
        Field("finished", TIMESTAMP, "When the attempt ended"),
        Field("outcome", Choice(("completed", "failed", "interrupted")), "How the attempt ended"),
        Field("error", Text(), "Failure or interruption description", nullable=True),
        Field(
            "labels_accessed",
            Boolean(),
            "Whether the attempt reached label access: recorded before the labels are read, so "
            "that a crash cannot allow a fresh scoring",
        ),
        Field(
            "scores_sha256",
            SHA256,
            "SHA-256 of the sorted score-file hashes of the scoring pass",
            nullable=True,
        ),
    ),
    _phase_rules,
)


def phase_sequence_problems(phases: Sequence[Mapping[str, Any]]) -> list[str]:
    """The rules between the ``test_phase`` records of one run, in the order of their start:
    attempts are numbered from 1 within a phase; the scoring phase completes at most once and no
    attempt of it follows a completed one or any evaluation attempt; an evaluation attempt
    follows a completed scoring attempt, uses its scores and none follows a completed
    evaluation."""
    problems: list[str] = []
    scoring_done: str | None = None
    evaluation_started = False
    evaluation_done = False
    counts = dict.fromkeys(PHASES, 0)
    for index, record in enumerate(phases):
        phase, where = record["phase"], f"test_phase[{index}]"
        counts[phase] += 1
        if record["attempt"] != counts[phase]:
            problems.append(f"{where}.attempt: {record['attempt']}, expected {counts[phase]}")
        if phase == "scoring":
            if scoring_done is not None or evaluation_started:
                problems.append(
                    f"{where}: no scoring attempt may follow a completed scoring or any evaluation"
                )
            if record["outcome"] == "completed":
                scoring_done = record["scores_sha256"]
        else:
            if scoring_done is None:
                problems.append(f"{where}: an evaluation attempt follows a completed scoring")
            elif record["scores_sha256"] != scoring_done:
                problems.append(f"{where}.scores_sha256: differs from the completed scoring's")
            if evaluation_done:
                problems.append(f"{where}: no attempt may follow a completed evaluation")
            evaluation_started = True
            evaluation_done = evaluation_done or record["outcome"] == "completed"
    return problems


# --- test_scores ---------------------------------------------------------------------------------


def _scores_rules(record: Mapping[str, Any]) -> list[str]:
    problems = []
    if record["min_score"] > record["max_score"]:
        problems.append("test_scores.min_score: must not exceed max_score")
    if record["distinct_scores"] > record["rows"]:
        problems.append("test_scores.distinct_scores: cannot exceed the rows")
    if record["days"]["first"] > record["days"]["last"]:
        problems.append("test_scores.days: first must not exceed last")
    timing = record["timing"]
    if timing["seconds"] <= 0:
        problems.append("test_scores.timing.seconds: a scoring pass takes time")
    elif not _near(timing["rows_per_second"], record["rows"] / timing["seconds"]):
        problems.append("test_scores.timing.rows_per_second: must be rows divided by seconds")
    return problems


def _scores_partition(record: Mapping[str, Any], context: ConfirmatoryContext) -> list[str]:
    test = context.test
    return [
        f"test_scores.{name}: must be {expected} (the Test partition)"
        for name, actual, expected in (
            ("days.first", record["days"]["first"], test.first_day),
            ("days.last", record["days"]["last"], test.last_day),
            ("rows", record["rows"], test.rows),
        )
        if actual != expected
    ]


TEST_SCORES = _contract(
    "test_scores",
    1,
    "The identity of one model's scores on the Test period, from the single scoring pass; the "
    "row-level scores stay local and are never committed.",
    (
        Field(
            "configuration", CONFIGURATION, "Configuration (or the logistic-regression reference)"
        ),
        Field("partition", Choice(("Test",)), "The partition scored"),
        Field(
            "days",
            Object(
                (
                    Field("first", POSITIVE, "First relative day scored"),
                    Field("last", POSITIVE, "Last relative day scored"),
                )
            ),
            "Relative days scored",
        ),
        Field("rows", POSITIVE, "Transactions scored"),
        Field("model", PROJECT_FILE, "The frozen model that scored"),
        Field("file", RUN_FILE, "Row-level score file (local only)"),
        Field("format", Choice(("parquet",)), "File format"),
        Field("min_score", Number(0, 1), "Smallest score"),
        Field("max_score", Number(0, 1), "Largest score"),
        Field("distinct_scores", POSITIVE, "Number of distinct scores"),
        Field("device", DEVICE, "Scoring device"),
        Field(
            "timing",
            Object(
                (
                    Field(
                        "passes", Integer(1, 1), "Scoring passes timed: the single authorised one"
                    ),
                    Field("seconds", SECONDS, "Wall-clock scoring time of the pass"),
                    Field(
                        "rows_per_second",
                        Number(0, exclusive_minimum=True),
                        "Rows scored per second",
                    ),
                )
            ),
            "Timing of the single scoring pass (never repeated for timing)",
        ),
    ),
    _scores_rules,
)

# --- test_inference ------------------------------------------------------------------------------

TEST_INFERENCE = _contract(
    "test_inference",
    1,
    "The confirmatory inference on the Test period: H1, H2 and H3 with the Holm decision, the "
    "S1 and S2 sensitivities, the SESOI views, the cell contrasts with their McNemar tests and "
    "the effect units (DR-003, DR-005, DR-006, DR-010).",
    (
        Field(
            "resampling",
            Object(
                (
                    Field("replicates", POSITIVE, "Bootstrap replicates B"),
                    Field("root_seed", SEED, "Root seed of the streams"),
                    Field("block_length", POSITIVE, "Primary day-block length"),
                    Field(
                        "block_length_sensitivity",
                        ListOf(POSITIVE, unique=True),
                        "Sensitivity block lengths",
                    ),
                )
            ),
            "The resampling design used",
        ),
        Field(
            "undefined_replicates",
            MapOf(COUNT),
            "Replicates without a defined value, by stream",
        ),
        Field(
            "h1",
            Object(
                (
                    Field("estimate", Number(), "Family effect δ_F (neural minus tree-based)"),
                    Field("p", PVALUE, "Two-sided p-value p_H1"),
                    Field("interval", INTERVAL, "90 % interval"),
                    Field("position", POSITION, "Position against (−SESOI, +SESOI)"),
                    Field("difference_interval", INTERVAL, "95 % interval"),
                    Field(
                        "within_family", ListOf(ESTIMAND, min_items=2), "Within-family contrasts"
                    ),
                    Field("interaction", ESTIMAND, "Family × weighting interaction"),
                    Field(
                        "interaction_splits",
                        Boolean(),
                        "The family effect is also reported per weighting",
                    ),
                    Field(
                        "by_weighting",
                        ListOf(ESTIMAND, min_items=2),
                        "Family effect at each weighting",
                    ),
                    Field("cells", ListOf(ESTIMAND, min_items=1), "Cell contrasts"),
                )
            ),
            "H1: the model-family effect",
        ),
        Field(
            "h2",
            Object(
                (
                    Field(
                        "models",
                        ListOf(
                            Object(
                                (
                                    Field("model", Text(), "Model"),
                                    Field("estimate", Number(), "Weighting effect δ_W,m"),
                                    Field("p_lower", PVALUE, "H0a: δ ≤ −SESOI"),
                                    Field("p_upper", PVALUE, "H0b: δ ≥ +SESOI"),
                                    Field("interval", INTERVAL, "90 % interval"),
                                    Field(
                                        "position", POSITION, "Position against (−SESOI, +SESOI)"
                                    ),
                                    Field("difference_interval", INTERVAL, "95 % interval"),
                                )
                            ),
                            min_items=1,
                        ),
                        "The models, tree-based then neural",
                    ),
                    Field("p", PVALUE, "p_H2: the largest of the one-sided p-values"),
                )
            ),
            "H2: the class-weighting equivalence test",
        ),
        Field(
            "h3",
            Object(
                (
                    Field("configuration", STUDY_CONFIGURATION, "The H3 configuration"),
                    Field("rate", Number(0, 1), "Observed Test alert rate r"),
                    Field("primary", H3_VARIANT, "TOST at the primary block length"),
                    Field("sensitivity", ListOf(H3_VARIANT), "The other block lengths"),
                    Field(
                        "tolerances",
                        ListOf(
                            Object(
                                (
                                    Field(
                                        "tolerance",
                                        Number(0, 1, exclusive_minimum=True),
                                        "Relative tolerance",
                                    ),
                                    Field("lower", Number(), "Lower bound of the band"),
                                    Field("upper", Number(), "Upper bound of the band"),
                                    Field("position", POSITION, "Position of the 90 % interval"),
                                )
                            )
                        ),
                        "The 90 % interval against other tolerance bands",
                    ),
                    Field("smallest_tolerance", Number(minimum=0), "t_min (DR-003)"),
                    Field(
                        "others",
                        ListOf(
                            Object(
                                (
                                    Field("configuration", STUDY_CONFIGURATION, "Configuration"),
                                    Field("rate", Number(0, 1), "Observed alert rate"),
                                    Field("interval", INTERVAL, "90 % interval"),
                                    Field("position", POSITION, "Position against the open band"),
                                )
                            )
                        ),
                        "The other configurations, descriptive",
                    ),
                )
            ),
            "H3: alert-rate transfer of the H3 configuration",
        ),
        Field("holm", HOLM, "Holm over H1, H2 and H3"),
        Field(
            "decisions",
            Object(
                (
                    Field("h1_rejected", Boolean(), "H1 rejected by the Holm-adjusted p-value"),
                    Field("h2_equivalent", Boolean(), "H2₁ concluded by the Holm-adjusted p-value"),
                    Field("h3_preserved", Boolean(), "H3₁ concluded by the Holm-adjusted p-value"),
                    Field("h1_category", H1_CATEGORY, "H1 practical category"),
                )
            ),
            "The confirmatory decisions",
        ),
        Field(
            "s1",
            Object(
                (
                    Field(
                        "quantities",
                        ListOf(
                            Object(
                                (
                                    Field("name", Text(), "Estimand"),
                                    Field("estimate", Number(), "Observed value"),
                                    Field("v_pigeonhole", Number(minimum=0), "Pigeonhole variance"),
                                    Field("v_naive", Number(minimum=0), "Fraud-level variance"),
                                    Field(
                                        "se",
                                        Number(0, exclusive_minimum=True),
                                        "Corrected standard error",
                                        nullable=True,
                                    ),
                                )
                            ),
                            min_items=1,
                        ),
                        "The Wald quantities: δ_F, then the weighting effects",
                    ),
                    Field("p_h1", UNIT_P, "S1 p_H1", nullable=True),
                    Field("interval_h1", INTERVAL, "S1 90 % interval of δ_F", nullable=True),
                    Field("position_h1", POSITION, "Its position against ±SESOI", nullable=True),
                    Field("p_h2", UNIT_P, "S1 p_H2", nullable=True),
                    Field(
                        "intervals_h2",
                        ListOf(INTERVAL),
                        "S1 90 % intervals of δ_W,m",
                        nullable=True,
                    ),
                    Field("comparison", COMPARISON, "Comparison with the primary", nullable=True),
                    Field("reasons", ListOf(Text()), "Why a value is null"),
                )
            ),
            "S1: the Owen-corrected sensitivity",
        ),
        Field(
            "s2",
            Object(
                (
                    Field("p_h1", PVALUE, "S2 p_H1"),
                    Field("p_h2", PVALUE, "S2 p_H2"),
                    Field("h1_interval", INTERVAL, "S2 90 % interval of δ_F"),
                    Field("h1_position", POSITION, "Its position against ±SESOI"),
                    Field("comparison", COMPARISON, "Comparison with the primary"),
                )
            ),
            "S2: the day-block sensitivity",
        ),
        Field(
            "sesoi_views",
            ListOf(
                Object(
                    (
                        Field("sesoi", Number(0, 1, exclusive_minimum=True), "Alternative SESOI"),
                        Field("h1_position", POSITION, "H1 interval position"),
                        Field("h1_category", H1_CATEGORY, "H1 category"),
                        Field("h2_positions", ListOf(POSITION), "H2 interval positions by model"),
                    )
                )
            ),
            "H1 and H2 read against the alternative SESOIs (descriptive)",
        ),
        Field(
            "mcnemar",
            ListOf(
                Object(
                    (
                        Field("first", CONFIGURATION, "Configuration a of the contrast a − b"),
                        Field("second", CONFIGURATION, "Configuration b"),
                        Field("only_first", COUNT, "Frauds captured by a only"),
                        Field("only_second", COUNT, "Frauds captured by b only"),
                        Field("p", UNIT_P, "Exact two-sided p-value (independence-based)"),
                    )
                ),
                min_items=1,
            ),
            "Exact McNemar tests of the cell contrasts (descriptive)",
        ),
        Field(
            "effect_basis",
            Object(
                (
                    Field("frauds", POSITIVE, "Frauds in the Test period"),
                    Field("days", POSITIVE, "Days in the Test period"),
                    Field("k", POSITIVE, "Fixed daily review budget K at the primary capacity"),
                    Field(
                        "oracle_ceiling",
                        Number(0, 1, exclusive_minimum=True),
                        "Oracle recall ceiling on the Test period",
                    ),
                )
            ),
            "The Test quantities behind the effect units",
        ),
        Field(
            "effect_units",
            ListOf(
                Object(
                    (
                        Field("estimand", Text(), "Estimand"),
                        Field("percentage_points", Number(), "Difference in percentage points"),
                        Field("frauds_per_partition", Number(), "Frauds per Test period"),
                        Field("frauds_per_day", Number(), "Frauds per day"),
                        Field("precision_at_k_points", Number(), "Precision@K equivalent, pp"),
                        Field("normalised_recall", Number(), "Normalised recall difference"),
                    )
                ),
                min_items=1,
            ),
            "Effect units of the confirmatory estimands",
        ),
    ),
)


def _interval(value: Mapping[str, Any]) -> Interval:
    return Interval(
        value["lower"], value["upper"], value["level"], value["undefined"], value["replicates"]
    )


def _walk_intervals(value: object, where: str) -> list[str]:
    """Every interval in the record has lower ≤ upper and no more undefined replicates than
    replicates."""
    problems: list[str] = []
    if isinstance(value, dict):
        if {"lower", "upper", "level", "undefined", "replicates"} <= value.keys():
            if value["lower"] > value["upper"]:
                problems.append(f"{where}: lower exceeds upper")
            if value["replicates"] and value["undefined"] > value["replicates"]:
                problems.append(f"{where}: more undefined replicates than replicates")
        for key, item in value.items():
            problems += _walk_intervals(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            problems += _walk_intervals(item, f"{where}[{index}]")
    return problems


def _maximum_problems(
    where: str, parent: Mapping[str, Any], parts: Sequence[Mapping[str, Any]]
) -> list[str]:
    """``parent`` is the largest of the one-sided p-values ``parts``, with the standard error of
    the component that attains it."""
    largest = max(parts, key=lambda part: part["value"])
    if not (
        _close(parent["value"], largest["value"]) and _close(parent["mc_se"], largest["mc_se"])
    ):
        return [f"{where}: must be the largest one-sided p-value and its standard error"]
    return []


def _comparison_problems(
    where: str,
    comparison: Mapping[str, Any],
    record: Mapping[str, Any],
    position: str,
    alpha: float,
) -> list[str]:
    decisions = record["decisions"]
    p_h3 = record["h3"]["primary"]["p"]["value"]
    expected = holm((comparison["p_h1"], comparison["p_h2"], p_h3), alpha)
    problems = []
    if not all(
        _close(a, b) for a, b in zip(comparison["holm"]["adjusted"], expected.adjusted, strict=True)
    ):
        problems.append(f"{where}.holm.adjusted: Holm of p_H1, p_H2 and the primary p_H3 differs")
    if tuple(comparison["holm"]["rejected"]) != expected.rejected:
        problems.append(f"{where}.holm.rejected: differs from Holm of the p-values")
    rejected_h1, rejected_h2, _ = expected.rejected
    if comparison["h1_rejected"] != rejected_h1 or comparison["h2_rejected"] != rejected_h2:
        problems.append(f"{where}: the rejections differ from the Holm decisions")
    category = h1_category(rejected_h1, Position(position)).value
    if comparison["h1_category"] != category:
        problems.append(f"{where}.h1_category: the rule gives {category!r}")
    differs = (
        rejected_h1 != decisions["h1_rejected"]
        or rejected_h2 != decisions["h2_equivalent"]
        or category != decisions["h1_category"]
    )
    if comparison["differs"] != differs:
        problems.append(f"{where}.differs: the comparison with the primary gives {differs}")
    return problems


RESAMPLED_STREAMS = ("pigeonhole_h1_h2", "naive_s1", "day_block_s2", "day_block_h3")
SQRT2 = math.sqrt(2.0)


def _upper_tail(x: float) -> float:
    """The standard-normal upper tail, 1 − Φ(x)."""
    return 0.5 * math.erfc(x / SQRT2)


def _lower_tail(x: float) -> float:
    """The standard-normal distribution function Φ(x)."""
    return 0.5 * math.erfc(-x / SQRT2)


def _wald_bounds(estimate: float, se: float, level: float) -> tuple[float, float]:
    z = NormalDist().inv_cdf(0.5 + level / 2)
    return estimate - z * se, estimate + z * se


def _add_one_problems(
    where: str, p: Mapping[str, Any], replicates: int, *, two_sided: bool = False
) -> list[str]:
    """``p`` is an add-one proportion (1 + count)/(B + 1) with the Monte Carlo standard error of
    its tail proportion (twice that of the smaller tail for a two-sided p-value below 1)."""
    value = p["value"]
    if two_sided and value >= 1:
        return []
    tail = value / 2 if two_sided else value
    scaled = tail * (replicates + 1)
    problems = []
    if scaled < 1 - 1e-6 or abs(scaled - round(scaled)) > 1e-6:
        problems.append(f"{where}.value: not an add-one proportion (1 + count)/(B + 1)")
    expected = (2 if two_sided else 1) * math.sqrt(tail * (1 - tail) / replicates)
    if not _near(p["mc_se"], expected):
        problems.append(f"{where}.mc_se: the Monte Carlo standard error is {expected}")
    return problems


def _interval_problems(record: Mapping[str, Any], declaration: EvaluationDeclaration) -> list[str]:
    """Every interval has the declared level and comes from the declared resampling: a percentile
    interval has B replicates and the undefined count of its stream; a Wald interval has none."""
    levels, replicates = declaration.levels, declaration.resampling.replicates
    counts = record["undefined_replicates"]
    h1, h2, h3, s1, s2 = record["h1"], record["h2"], record["h3"], record["s1"], record["s2"]
    pigeonhole = "pigeonhole_h1_h2"
    entries: list[tuple[str, Mapping[str, Any], float, str | None]] = [
        ("h1.interval", h1["interval"], levels.interpretation, pigeonhole),
        ("h1.difference_interval", h1["difference_interval"], levels.difference, pigeonhole),
        ("h1.interaction.interval", h1["interaction"]["interval"], levels.difference, pigeonhole),
        ("s2.h1_interval", s2["h1_interval"], levels.interpretation, "day_block_s2"),
        ("h3.primary.interval", h3["primary"]["interval"], levels.interpretation, "day_block_h3"),
    ]
    for field in ("within_family", "by_weighting", "cells"):
        entries += [
            (
                f"h1.{field}[{item['name']}].interval",
                item["interval"],
                levels.difference,
                pigeonhole,
            )
            for item in h1[field]
        ]
    for model in h2["models"]:
        entries += [
            (f"h2.{model['model']}.interval", model["interval"], levels.interpretation, pigeonhole),
            (
                f"h2.{model['model']}.difference_interval",
                model["difference_interval"],
                levels.difference,
                pigeonhole,
            ),
        ]
    entries += [
        (
            f"h3 (block length {v['block_length']}).interval",
            v["interval"],
            levels.interpretation,
            "day_block_h3",
        )
        for v in h3["sensitivity"]
    ]
    entries += [
        (
            f"h3.others[{o['configuration']}].interval",
            o["interval"],
            levels.interpretation,
            "day_block_h3",
        )
        for o in h3["others"]
    ]
    if s1["interval_h1"] is not None:
        entries.append(("s1.interval_h1", s1["interval_h1"], levels.interpretation, None))
    entries += [
        (f"s1.intervals_h2[{i}]", interval, levels.interpretation, None)
        for i, interval in enumerate(s1["intervals_h2"] or [])
    ]

    problems = []
    if set(counts) != set(RESAMPLED_STREAMS):
        problems.append(
            f"test_inference.undefined_replicates: must name exactly {list(RESAMPLED_STREAMS)}"
        )
    problems += [
        f"test_inference.undefined_replicates.{stream}: more than the {replicates} replicates"
        for stream, count in counts.items()
        if count > replicates
    ]
    undefined_h3: list[int] = []
    for where, interval, level, stream in entries:
        if interval["level"] != level:
            problems.append(f"test_inference.{where}.level: must be {level}")
        if stream is None:
            if interval["replicates"] != 0 or interval["undefined"] != 0:
                problems.append(f"test_inference.{where}: a Wald interval has no replicates")
            continue
        if interval["replicates"] != replicates:
            problems.append(f"test_inference.{where}.replicates: must be the declared {replicates}")
        if stream == "day_block_h3":
            undefined_h3.append(interval["undefined"])
        elif interval["undefined"] != counts.get(stream, -1):
            problems.append(
                f"test_inference.{where}.undefined: must be the count of the {stream} stream"
            )
    if counts.get("day_block_h3", -1) != max(undefined_h3, default=0):
        problems.append(
            "test_inference.undefined_replicates.day_block_h3: must be the largest count of the "
            "H3 intervals"
        )
    return problems


def _p_value_problems(record: Mapping[str, Any], declaration: EvaluationDeclaration) -> list[str]:
    """Every bootstrap p-value is an add-one proportion with its Monte Carlo standard error."""
    replicates = declaration.resampling.replicates
    h1, h2, h3, s2 = record["h1"], record["h2"], record["h3"], record["s2"]
    problems = _add_one_problems("test_inference.h1.p", h1["p"], replicates, two_sided=True)
    problems += _add_one_problems("test_inference.h2.p", h2["p"], replicates)
    problems += _add_one_problems("test_inference.s2.p_h1", s2["p_h1"], replicates, two_sided=True)
    problems += _add_one_problems("test_inference.s2.p_h2", s2["p_h2"], replicates)
    for model in h2["models"]:
        for side in ("p_lower", "p_upper"):
            problems += _add_one_problems(
                f"test_inference.h2.{model['model']}.{side}", model[side], replicates
            )
    for variant in [h3["primary"], *h3["sensitivity"]]:
        for name in ("p_lower", "p_upper", "p"):
            problems += _add_one_problems(
                f"test_inference.h3 (block length {variant['block_length']}).{name}",
                variant[name],
                replicates,
            )
    return problems


def _s1_problems(record: Mapping[str, Any], declaration: EvaluationDeclaration) -> list[str]:
    """S1 is available exactly when the corrected variance of its quantities is positive, and its
    Wald intervals, p-values, positions and comparison are those of the quantities."""
    s1, inference, levels = record["s1"], declaration.inference, declaration.levels
    sesoi = inference.sesoi
    models = (*declaration.configurations.tree, *declaration.configurations.neural)
    names = ["family", *(f"weighting:{model}" for model in models)]
    quantities = s1["quantities"]
    if [q["name"] for q in quantities] != names:
        return [f"test_inference.s1.quantities: must be {names}"]
    problems = []
    estimates = [record["h1"]["estimate"], *(m["estimate"] for m in record["h2"]["models"])]
    for quantity, estimate in zip(quantities, estimates, strict=True):
        name = quantity["name"]
        if not _close(quantity["estimate"], estimate):
            problems.append(f"test_inference.s1.quantities[{name}].estimate: differs from H1/H2")
        corrected = quantity["v_pigeonhole"] - 2 * quantity["v_naive"]
        if (quantity["se"] is None) != (corrected <= 0):
            problems.append(
                f"test_inference.s1.quantities[{name}].se: "
                "null exactly when the corrected variance is not positive"
            )
        elif quantity["se"] is not None and not _close(quantity["se"] ** 2, corrected):
            problems.append(
                f"test_inference.s1.quantities[{name}].se: not the root of the corrected variance"
            )
    family, weighting = quantities[0], quantities[1:]
    family_ok = family["se"] is not None
    weighting_ok = all(q["se"] is not None for q in weighting)
    h1_given = [s1[name] is not None for name in ("p_h1", "interval_h1", "position_h1")]
    h2_given = [s1[name] is not None for name in ("p_h2", "intervals_h2")]
    if not (all(h1_given) if family_ok else not any(h1_given)):
        problems.append(
            "test_inference.s1: p_h1, interval_h1 and position_h1 are given exactly when the "
            "family effect has a standard error"
        )
    if not (all(h2_given) if weighting_ok else not any(h2_given)):
        problems.append(
            "test_inference.s1: p_h2 and intervals_h2 are given exactly when every weighting "
            "effect has a standard error"
        )
    if (s1["comparison"] is not None) != (family_ok and weighting_ok):
        problems.append("test_inference.s1.comparison: given exactly when p_h1 and p_h2 are given")
    if bool(s1["reasons"]) != (not (family_ok and weighting_ok)):
        problems.append("test_inference.s1.reasons: stated exactly when a value is null")
    if family_ok and all(h1_given):
        estimate, se = family["estimate"], family["se"]
        if not _near(s1["p_h1"], math.erfc(abs(estimate) / (se * SQRT2))):
            problems.append("test_inference.s1.p_h1: differs from the Wald p-value")
        low, high = _wald_bounds(estimate, se, levels.interpretation)
        interval = s1["interval_h1"]
        if not (_near(interval["lower"], low) and _near(interval["upper"], high)):
            problems.append("test_inference.s1.interval_h1: differs from the Wald interval")
        if s1["position_h1"] != position_of(_interval(interval), -sesoi, sesoi).value:
            problems.append("test_inference.s1.position_h1: differs from the interval")
    if weighting_ok and all(h2_given):
        components = []
        for quantity in weighting:
            estimate, se = quantity["estimate"], quantity["se"]
            components += [
                _upper_tail((estimate + sesoi) / se),
                _lower_tail((estimate - sesoi) / se),
            ]
        if not _near(s1["p_h2"], max(components)):
            problems.append("test_inference.s1.p_h2: differs from the largest Wald p-value")
        expected = [_wald_bounds(q["estimate"], q["se"], levels.interpretation) for q in weighting]
        given = [(i["lower"], i["upper"]) for i in s1["intervals_h2"]]
        if len(given) != len(expected) or not all(
            _near(a, c) and _near(b, d) for (a, b), (c, d) in zip(given, expected, strict=False)
        ):
            problems.append("test_inference.s1.intervals_h2: differ from the Wald intervals")
    comparison = s1["comparison"]
    if comparison is not None and all(h1_given) and all(h2_given):
        if not (_near(comparison["p_h1"], s1["p_h1"]) and _near(comparison["p_h2"], s1["p_h2"])):
            problems.append("test_inference.s1.comparison: its p-values differ from S1's")
        problems += _comparison_problems(
            "test_inference.s1.comparison", comparison, record, s1["position_h1"], inference.alpha
        )
    return problems


def _effect_problems(record: Mapping[str, Any], context: ConfirmatoryContext) -> list[str]:
    """The effect units are those of the estimates, recomputed from the stated Test quantities."""
    declaration, basis = context.declaration, record["effect_basis"]
    capacity = declaration.capacity
    days = context.test.last_day - context.test.first_day + 1
    k = capacity.budgets[capacity.shares.index(capacity.primary)]
    problems = []
    if basis["days"] != days:
        problems.append(f"test_inference.effect_basis.days: must be {days} (the Test days)")
    if basis["k"] != k:
        problems.append(f"test_inference.effect_basis.k: must be the declared budget {k}")
    h1, h2 = record["h1"], record["h2"]
    estimates = {"family": h1["estimate"], "interaction": h1["interaction"]["estimate"]}
    for field in ("by_weighting", "within_family", "cells"):
        estimates |= {item["name"]: item["estimate"] for item in h1[field]}
    estimates |= {f"weighting:{m['model']}": m["estimate"] for m in h2["models"]}
    names = [unit["estimand"] for unit in record["effect_units"]]
    if len(set(names)) != len(names):
        problems.append("test_inference.effect_units: an estimand is listed once")
    for unit in record["effect_units"]:
        name = unit["estimand"]
        if name not in estimates:
            problems.append(f"test_inference.effect_units[{name}]: not an estimand of the record")
            continue
        expected = effect_units(
            estimates[name],
            frauds=basis["frauds"],
            days=basis["days"],
            k=basis["k"],
            oracle_ceiling=basis["oracle_ceiling"],
        )
        problems += [
            f"test_inference.effect_units[{name}].{field}: differs from the units of the estimate"
            for field in (
                "percentage_points",
                "frauds_per_partition",
                "frauds_per_day",
                "precision_at_k_points",
                "normalised_recall",
            )
            if not _near(unit[field], getattr(expected, field))
        ]
    return problems


def _inference_rules(record: Mapping[str, Any], context: ConfirmatoryContext) -> list[str]:
    declaration = context.declaration
    inference, levels, resampling = (
        declaration.inference,
        declaration.levels,
        declaration.resampling,
    )
    order = declaration.configurations.order
    sesoi, target, delta = inference.sesoi, inference.h3_target, inference.h3_delta
    low, high = target - delta, target + delta
    h1, h2, h3, decisions = record["h1"], record["h2"], record["h3"], record["decisions"]
    problems = _walk_intervals(record, "test_inference")

    expected_resampling = {
        "replicates": resampling.replicates,
        "root_seed": resampling.root_seed,
        "block_length": resampling.block_length,
        "block_length_sensitivity": list(resampling.block_length_sensitivity),
    }
    if record["resampling"] != expected_resampling:
        problems.append(f"test_inference.resampling: must be {expected_resampling}")

    # H1
    names = estimands(declaration).names
    for field, prefix in (
        ("within_family", "within:"),
        ("by_weighting", "family:"),
        ("cells", "cell:"),
    ):
        expected = [name for name in names if name.startswith(prefix)]
        if [item["name"] for item in h1[field]] != expected:
            problems.append(f"test_inference.h1.{field}: must be {expected}")
    if h1["interaction"]["name"] != "interaction":
        problems.append("test_inference.h1.interaction.name: must be 'interaction'")
    for name, level in (
        ("interval", levels.interpretation),
        ("difference_interval", levels.difference),
    ):
        if h1[name]["level"] != level:
            problems.append(f"test_inference.h1.{name}.level: must be {level}")
    if h1["position"] != position_of(_interval(h1["interval"]), -sesoi, sesoi).value:
        problems.append("test_inference.h1.position: differs from the interval against ±SESOI")
    interaction = h1["interaction"]
    if h1["interaction_splits"] != interaction_splits_family_effect(
        _interval(interaction["interval"]), interaction["estimate"], sesoi
    ):
        problems.append("test_inference.h1.interaction_splits: differs from the qualification rule")

    # H2
    models = (*declaration.configurations.tree, *declaration.configurations.neural)
    if [item["model"] for item in h2["models"]] != list(models):
        problems.append(f"test_inference.h2.models: must be {list(models)}")
    for item in h2["models"]:
        if item["position"] != position_of(_interval(item["interval"]), -sesoi, sesoi).value:
            problems.append(
                f"test_inference.h2.models[{item['model']}].position: differs from the interval"
            )
    problems += _maximum_problems(
        "test_inference.h2.p",
        h2["p"],
        [part for item in h2["models"] for part in (item["p_lower"], item["p_upper"])],
    )

    # H3
    if h3["configuration"] not in order:
        problems.append("test_inference.h3.configuration: not a configuration of the study")
    variants = [h3["primary"], *h3["sensitivity"]]
    lengths = [resampling.block_length, *resampling.block_length_sensitivity]
    if [variant["block_length"] for variant in variants] != lengths:
        problems.append(f"test_inference.h3: block lengths must be {lengths}")
    for variant in variants:
        problems += _maximum_problems(
            f"test_inference.h3 (block length {variant['block_length']}).p",
            variant["p"],
            [variant["p_lower"], variant["p_upper"]],
        )
        if variant["position"] != position_of(_interval(variant["interval"]), low, high).value:
            problems.append(
                f"test_inference.h3 (block length {variant['block_length']}).position: "
                "differs from the interval"
            )
    primary = _interval(h3["primary"]["interval"])
    if primary.level != levels.interpretation:
        problems.append(
            f"test_inference.h3.primary.interval.level: must be {levels.interpretation}"
        )
    if [view["tolerance"] for view in h3["tolerances"]] != list(inference.h3_tolerance_sensitivity):
        problems.append("test_inference.h3.tolerances: must follow the declared tolerances")
    for view in h3["tolerances"]:
        band = (target * (1 - view["tolerance"]), target * (1 + view["tolerance"]))
        if not (_close(view["lower"], band[0]) and _close(view["upper"], band[1])):
            problems.append(f"test_inference.h3.tolerances[{view['tolerance']}]: wrong band")
        if view["position"] != position_of(primary, *band).value:
            problems.append(
                f"test_inference.h3.tolerances[{view['tolerance']}].position: "
                "differs from the interval"
            )
    if not _close(h3["smallest_tolerance"], smallest_tolerance(primary, target)):
        problems.append("test_inference.h3.smallest_tolerance: differs from the interval")
    others = [item["configuration"] for item in h3["others"]]
    if others != [name for name in order if name != h3["configuration"]]:
        problems.append("test_inference.h3.others: must be the other configurations in order")
    for item in h3["others"]:
        if item["position"] != position_of(_interval(item["interval"]), low, high).value:
            problems.append(
                f"test_inference.h3.others[{item['configuration']}].position: "
                "differs from the interval"
            )

    # Holm and the decisions
    p_values = (h1["p"]["value"], h2["p"]["value"], h3["primary"]["p"]["value"])
    expected_holm = holm(p_values, inference.alpha)
    if not all(
        _close(a, b)
        for a, b in zip(record["holm"]["adjusted"], expected_holm.adjusted, strict=True)
    ):
        problems.append("test_inference.holm.adjusted: differs from Holm of p_H1, p_H2 and p_H3")
    if tuple(record["holm"]["rejected"]) != expected_holm.rejected:
        problems.append("test_inference.holm.rejected: differs from Holm of the p-values")
    for name, rejected in zip(
        ("h1_rejected", "h2_equivalent", "h3_preserved"), expected_holm.rejected, strict=True
    ):
        if decisions[name] != rejected:
            problems.append(
                f"test_inference.decisions.{name}: must be {rejected} (Holm-adjusted p)"
            )
    category = h1_category(expected_holm.rejected[0], Position(h1["position"])).value
    if decisions["h1_category"] != category:
        problems.append(f"test_inference.decisions.h1_category: the rule gives {category!r}")

    # S1
    problems += _s1_problems(record, declaration)
    problems += _interval_problems(record, declaration)
    problems += _p_value_problems(record, declaration)
    problems += _effect_problems(record, context)

    # S2
    s2 = record["s2"]
    problems += _comparison_problems(
        "test_inference.s2.comparison", s2["comparison"], record, s2["h1_position"], inference.alpha
    )
    if s2["h1_position"] != position_of(_interval(s2["h1_interval"]), -sesoi, sesoi).value:
        problems.append("test_inference.s2.h1_position: differs from the interval")
    if not (
        _close(s2["comparison"]["p_h1"], s2["p_h1"]["value"])
        and _close(s2["comparison"]["p_h2"], s2["p_h2"]["value"])
    ):
        problems.append("test_inference.s2.comparison: its p-values differ from the S2 p-values")

    # SESOI views, McNemar
    if [view["sesoi"] for view in record["sesoi_views"]] != list(inference.sesoi_sensitivity):
        problems.append("test_inference.sesoi_views: must follow the declared sensitivities")
    for view in record["sesoi_views"]:
        s = view["sesoi"]
        if view["h1_position"] != position_of(_interval(h1["interval"]), -s, s).value:
            problems.append(
                f"test_inference.sesoi_views[{s}].h1_position: differs from the interval"
            )
        if (
            view["h1_category"]
            != h1_category(decisions["h1_rejected"], Position(view["h1_position"])).value
        ):
            problems.append(f"test_inference.sesoi_views[{s}].h1_category: differs from the rule")
        if view["h2_positions"] != [
            position_of(_interval(m["interval"]), -s, s).value for m in h2["models"]
        ]:
            problems.append(
                f"test_inference.sesoi_views[{s}].h2_positions: differ from the intervals"
            )
    if [(m["first"], m["second"]) for m in record["mcnemar"]] != list(declaration.contrasts):
        problems.append("test_inference.mcnemar: must follow the declared cell contrasts")
    for m in record["mcnemar"]:
        if not _close(m["p"], exact_p(m["only_first"], m["only_second"])):
            problems.append(
                f"test_inference.mcnemar[{m['first']}-{m['second']}].p: differs from the exact test"
            )
    return problems


# --- test_run_end --------------------------------------------------------------------------------

_RECORD_FILE = re.compile(r"records/[A-Za-z0-9_.-]+\.json")


def _run_end_rules(record: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if record["outcome"] == "completed":
        if record["error"] is not None:
            problems.append("test_run_end.error: must be null when the run completed")
    elif record["error"] is None:
        problems.append(f"test_run_end.error: required when the run {record['outcome']}")
    paths = [entry["path"] for entry in record["records"]]
    if len(set(paths)) != len(paths):
        problems.append("test_run_end.records: a path is listed once")
    if paths != sorted(paths):
        problems.append("test_run_end.records: must be listed in increasing path order")
    problems += [
        f"test_run_end.records: {path} is not another record of the run"
        for path in paths
        if not _RECORD_FILE.fullmatch(path) or path == f"records/{END_KIND}.json"
    ]
    return problems


TEST_RUN_END = _contract(
    "test_run_end",
    1,
    "Written last: how the run ended and the SHA-256 of every other record. A run directory "
    "without it was interrupted before it could close; its completed records are preserved.",
    (
        Field("finished", TIMESTAMP, "When the run ended"),
        Field("outcome", Choice(("completed", "failed", "interrupted")), "How the run ended"),
        Field("error", Text(), "Failure or interruption description", nullable=True),
        Field(
            "records",
            ListOf(RUN_FILE, min_items=1, unique=True),
            "Every other record of the run, in increasing path order",
        ),
    ),
    _run_end_rules,
)

# --- test_evaluation, test_workload and test_calibration ------------------------------------------

THRESHOLD_POLICY = Choice(("R0", "R1", "R2"))
R3_MEASURES = (
    "recall",
    "precision",
    "frauds_per_1000_reviews",
    "normalised_recall",
    "lift",
    "amount_weighted_recall",
    "false_negatives",
)
THRESHOLD_MEASURES = ("recall", "precision", "f1")
RANKING_MEASURES = ("average_precision", "roc_auc")
CALIBRATION_MEASURES = ("brier", "log_loss", "intercept", "slope")
CALIBRATION_METHODS = ("platt", "isotonic")
CALIBRATION_ROLES = {"platt": "primary", "isotonic": "sensitivity"}
CAPACITY_SHARE = Number(0, 1, exclusive_minimum=True)
METRIC_MAP = MapOf(Number(), keys=Choice(tuple(METRICS)), nullable_values=True)


def _intervals(names: tuple[str, ...], meaning: str) -> Field:
    return Field(
        "intervals",
        MapOf(INTERVAL, keys=Choice(names), nullable_values=True),
        f"{meaning} (null: the measure is not estimable)",
    )


R3_ENTRY = Object(
    (
        Field("capacity", CAPACITY_SHARE, "Review-capacity share c"),
        Field("k", POSITIVE, "Fixed daily review budget K"),
        Field("counts", COUNTS, "Pooled counts"),
        Field("daily", ListOf(DAY_ROW, min_items=1), "Per-day values, one entry per Test day"),
        Field("cutoff_ties", CUTOFF_TIES, "Score ties crossing the daily cutoff"),
        Field("metrics", METRIC_MAP, "Outcome measures"),
        _intervals(R3_MEASURES, "95 % intervals: pigeonhole recall-based ratios"),
    )
)
POLICY_ENTRY = Object(
    (
        Field("policy", THRESHOLD_POLICY, "Threshold policy"),
        Field(
            "capacity", CAPACITY_SHARE, "Review-capacity share c the workload is measured against"
        ),
        Field("k", POSITIVE, "Fixed daily review budget K"),
        Field("threshold", Number(0, 1), "The frozen threshold (R0: 0.5)"),
        Field("counts", COUNTS, "Pooled counts of the uncapped alerts"),
        Field("daily", ListOf(DAY_ROW, min_items=1), "Per-day values, one entry per Test day"),
        Field("metrics", METRIC_MAP, "Outcome measures"),
        _intervals(THRESHOLD_MEASURES, "95 % day-block intervals"),
    )
)
RANKING_ENTRY = Object(
    (
        Field("metrics", METRIC_MAP, "Average Precision and ROC-AUC, score ties intact"),
        _intervals(RANKING_MEASURES, "95 % day-block intervals"),
    )
)
THRESHOLDS = Object(
    (
        Field("r0", Number(0.5, 0.5), "R0: the default 0.5 threshold"),
        Field("r1", Number(0, 1), "R1: the frozen F1-optimal threshold"),
        Field(
            "r2",
            ListOf(
                Object(
                    (
                        Field("capacity", CAPACITY_SHARE, "Review-capacity share c"),
                        Field("threshold", Number(0, 1), "The frozen threshold tau_c at c"),
                    )
                ),
                min_items=1,
            ),
            "R2: the frozen capacity-rate threshold of every capacity",
        ),
    )
)
RECORD_DAYS = Object(
    (
        Field("first", POSITIVE, "First relative day evaluated"),
        Field("last", POSITIVE, "Last relative day evaluated"),
    )
)


WORKLOAD_FIELDS = ("oracle_reviewable", "overflow", "unused", "load_ratio")
R3_METRIC_KEYS = frozenset(
    (
        "recall",
        "precision",
        "frauds_per_1000_reviews",
        "oracle_ceiling",
        "normalised_recall",
        "random_review_recall",
        "lift",
        "utilisation",
        "capacity_to_prevalence",
        "amount_weighted_recall",
        "fraud_value_exposed",
    )
)
POLICY_METRIC_KEYS = frozenset(
    (
        "recall",
        "precision",
        "f1",
        "capacity_to_prevalence",
        "total_overflow",
        "total_unused",
        "mean_load_ratio",
        "days_above_budget",
        "days_below_budget",
        "zero_alert_days",
    )
)
RANKING_METRIC_KEYS = frozenset(RANKING_MEASURES)
UNIT_METRICS = frozenset(
    (
        "recall",
        "precision",
        "f1",
        "oracle_ceiling",
        "normalised_recall",
        "random_review_recall",
        "amount_weighted_recall",
        "average_precision",
        "roc_auc",
    )
)


def _count_problems(where: str, counts: Mapping[str, Any]) -> list[str]:
    problems = []
    if counts["true_positives"] + counts["false_negatives"] != counts["frauds"]:
        problems.append(f"{where}.counts: true_positives + false_negatives ≠ frauds")
    if counts["true_positives"] + counts["false_positives"] != counts["selected"]:
        problems.append(f"{where}.counts: true_positives + false_positives ≠ selected")
    if max(counts["selected"], counts["frauds"]) > counts["transactions"]:
        problems.append(f"{where}.counts: selected and frauds are at most the transactions")
    return problems


def _expected(
    where: str, metrics: Mapping[str, Any], expected: Mapping[str, float | None]
) -> list[str]:
    problems = []
    for name, value in expected.items():
        actual = metrics.get(name)
        if value is None or actual is None:
            same = value is None and actual is None
        else:
            same = _near(actual, value)
        if not same:
            problems.append(f"{where}.metrics.{name}: must be {value!r}, not {actual!r}")
    return problems


def _metric_set_problems(
    where: str, metrics: Mapping[str, Any], required: frozenset[str]
) -> list[str]:
    """Exactly the metrics of the entry's kind, each share inside [0, 1]."""
    problems = []
    if set(metrics) != required:
        missing, extra = sorted(required - set(metrics)), sorted(set(metrics) - required)
        problems.append(f"{where}.metrics: missing {missing}, unexpected {extra}")
    problems += [
        f"{where}.metrics.{name}: a share lies in [0, 1]"
        for name in UNIT_METRICS & set(metrics)
        if metrics[name] is not None and not 0 <= metrics[name] <= 1
    ]
    if required == RANKING_METRIC_KEYS:
        problems += [
            f"{where}.metrics.{name}: required"
            for name in sorted(required & set(metrics))
            if metrics[name] is None
        ]
    return problems


def _entry_frame(
    record: Mapping[str, Any], entry: Mapping[str, Any], policy: str
) -> dict[str, Any]:
    """The entry as a development ``evaluation`` record, for the per-day rules they share."""
    return {
        "policy": policy,
        "days": record["days"],
        "k": entry["k"],
        "counts": entry["counts"],
        "daily": entry["daily"],
        "cutoff_ties": entry.get("cutoff_ties"),
    }


def _workload_given(daily: Sequence[Mapping[str, Any]]) -> bool:
    """Whether every day states its workload; ``_daily`` reports the days that do not."""
    return all(row[name] is not None for row in daily for name in WORKLOAD_FIELDS)


def _r3_problems(where: str, entry: Mapping[str, Any]) -> list[str]:
    counts, metrics, daily = entry["counts"], entry["metrics"], entry["daily"]
    problems = _count_problems(where, counts)
    problems += _metric_set_problems(where, metrics, R3_METRIC_KEYS)
    frauds, selected, hits = counts["frauds"], counts["selected"], counts["true_positives"]
    if counts["selected"] > entry["k"] * counts["days"]:
        problems.append(f"{where}.counts.selected: R3 reviews at most K a day")
    if not _workload_given(daily):
        return problems
    achievable = sum(row["oracle_reviewable"] for row in daily)
    if frauds and selected and achievable:
        problems += _expected(
            where,
            metrics,
            {
                "recall": hits / frauds,
                "precision": hits / selected,
                "frauds_per_1000_reviews": 1000 * hits / selected,
                "oracle_ceiling": achievable / frauds,
                "normalised_recall": hits / achievable,
                "random_review_recall": selected / counts["transactions"],
                "lift": hits * counts["transactions"] / (frauds * selected),
                "utilisation": selected / (entry["k"] * counts["days"]),
                "capacity_to_prevalence": entry["capacity"] * counts["transactions"] / frauds,
            },
        )
    return problems


def _policy_problems(where: str, entry: Mapping[str, Any]) -> list[str]:
    counts, daily = entry["counts"], entry["daily"]
    problems = _count_problems(where, counts)
    problems += _metric_set_problems(where, entry["metrics"], POLICY_METRIC_KEYS)
    alerts, frauds, hits = counts["selected"], counts["frauds"], counts["true_positives"]
    if frauds and _workload_given(daily):
        problems += _expected(
            where,
            entry["metrics"],
            {
                "recall": hits / frauds,
                "precision": hits / alerts if alerts else None,
                "f1": 2 * hits / (frauds + alerts),
                "capacity_to_prevalence": entry["capacity"] * counts["transactions"] / frauds,
                "total_overflow": sum(row["overflow"] for row in daily),
                "total_unused": sum(row["unused"] for row in daily),
                "mean_load_ratio": math.fsum(row["load_ratio"] for row in daily) / len(daily),
                "days_above_budget": sum(row["overflow"] > 0 for row in daily),
                "days_below_budget": sum(row["unused"] > 0 for row in daily),
                "zero_alert_days": sum(row["selected"] == 0 for row in daily),
            },
        )
    return problems


def _population_problems(label: str, entries: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every entry of a record describes the same transactions and frauds, day by day."""
    if not entries:
        return []
    first = [(row["day"], row["transactions"], row["frauds"]) for row in entries[0]["daily"]]
    return [
        f"{label}[{i}]: the days, transactions and frauds must be those of the first entry"
        for i, entry in enumerate(entries[1:], start=1)
        if [(row["day"], row["transactions"], row["frauds"]) for row in entry["daily"]] != first
        or any(entry["counts"][n] != entries[0]["counts"][n] for n in ("transactions", "frauds"))
    ]


def _interval_set_problems(
    where: str,
    entry: Mapping[str, Any],
    names: tuple[str, ...],
    declaration: EvaluationDeclaration,
) -> list[str]:
    intervals, metrics = entry["intervals"], entry["metrics"]
    if sorted(intervals) != sorted(names):
        return [f"{where}.intervals: must hold exactly {list(names)}"]
    problems = []
    for name, interval in intervals.items():
        point_missing = name in metrics and metrics[name] is None
        if (interval is None) != point_missing:
            problems.append(f"{where}.intervals.{name}: null exactly when the measure is null")
        if interval is None:
            continue
        label = f"{where}.intervals.{name}"
        if interval["lower"] > interval["upper"]:
            problems.append(f"{label}: lower exceeds upper")
        if not _close(interval["level"], declaration.levels.difference):
            problems.append(f"{label}.level: must be the declared {declaration.levels.difference}")
        if interval["replicates"] != declaration.resampling.replicates:
            problems.append(f"{label}.replicates: must be the declared B")
        if interval["undefined"] >= interval["replicates"]:
            problems.append(f"{label}.undefined: fewer than the replicates")
    return problems


def _derived_problems(where: str, entry: Mapping[str, Any]) -> list[str]:
    """The intervals of precision, per-1,000, lift, normalised recall and false negatives are
    fixed linear maps of the recall interval at fixed review counts (outcomes.py), over the same
    replicates: their endpoints follow the recall interval's and so do their undefined counts."""
    counts, intervals, daily = entry["counts"], entry["intervals"], entry["daily"]
    recall = intervals.get("recall")  # a missing interval is reported by the interval-set rule
    if recall is None or not _workload_given(daily):
        return []
    achievable = sum(row["oracle_reviewable"] for row in daily)
    if not counts["selected"] or not achievable:
        return []
    frauds, selected = counts["frauds"], counts["selected"]
    scale = {
        "precision": frauds / selected,
        "frauds_per_1000_reviews": 1000 * frauds / selected,
        "normalised_recall": frauds / achievable,
        "lift": counts["transactions"] / selected,
    }
    problems = []
    for name, factor in scale.items():
        got = intervals.get(name)
        if got is None or not (
            _near(got["lower"], factor * recall["lower"])
            and _near(got["upper"], factor * recall["upper"])
        ):
            problems.append(f"{where}.intervals.{name}: must be {factor!r} × the recall interval")
    misses = intervals.get("false_negatives")
    if misses is None or not (
        _near(misses["lower"], frauds * (1 - recall["upper"]))
        and _near(misses["upper"], frauds * (1 - recall["lower"]))
    ):
        problems.append(
            f"{where}.intervals.false_negatives: must be F (1 − recall) with swapped ends"
        )
    problems += [
        f"{where}.intervals.{name}.undefined: must be the recall interval's"
        for name in (*scale, "false_negatives")
        if intervals.get(name) is not None and intervals[name]["undefined"] != recall["undefined"]
    ]
    return problems


def _evaluation_rules(record: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    reference = record["configuration"] == REFERENCE
    if reference != (record["thresholds"] is None):
        problems.append(
            "test_evaluation.thresholds: given for the configurations, not the reference"
        )
    if reference != (not record["policies"]):
        problems.append("test_evaluation.policies: the reference has R3 and the ranking only")
    first = record["days"]["first"]
    entries = record["r3"]
    capacities = [entry["capacity"] for entry in entries]
    if len(set(capacities)) != len(capacities):
        problems.append("test_evaluation.r3: a capacity is evaluated once")
    for i, entry in enumerate(entries):
        where = f"test_evaluation.r3[{i}]"
        problems += _daily(_entry_frame(record, entry, "R3"))
        problems += _r3_problems(where, entry)
        problems += _derived_problems(where, entry)
    problems += _metric_set_problems(
        "test_evaluation.ranking", record["ranking"]["metrics"], RANKING_METRIC_KEYS
    )
    thresholds = record["thresholds"]
    expected = [(p, c) for p in ("R0", "R1", "R2") for c in capacities] if thresholds else []
    if [(e["policy"], e["capacity"]) for e in record["policies"]] != expected:
        problems.append(
            "test_evaluation.policies: R0, R1 and R2 at every R3 capacity, in that order"
        )
    if thresholds:
        r2 = [(t["capacity"], t["threshold"]) for t in thresholds["r2"]]
        if [c for c, _ in r2] != capacities:
            problems.append("test_evaluation.thresholds.r2: one threshold for every R3 capacity")
    for i, entry in enumerate(record["policies"]):
        where = f"test_evaluation.policies[{i}]"
        problems += _daily(_entry_frame(record, entry, entry["policy"]))
        problems += _policy_problems(where, entry)
        if thresholds and entry["threshold"] != _frozen_threshold(thresholds, entry):
            problems.append(f"{where}.threshold: must be the record's {entry['policy']} threshold")
    problems += _population_problems("test_evaluation.r3", entries)
    problems += _population_problems("test_evaluation.policies", record["policies"])
    if entries and record["policies"]:
        problems += _population_problems("test_evaluation", [entries[0], record["policies"][0]])
    if not all(e["counts"]["days"] == len(e["daily"]) for e in (*entries, *record["policies"])):
        problems.append("test_evaluation: counts.days must count the daily entries")
    if entries and entries[0]["daily"] and entries[0]["daily"][0]["day"] != first:
        problems.append("test_evaluation.days: the daily entries start at days.first")
    return problems


def _frozen_threshold(thresholds: Mapping[str, Any], entry: Mapping[str, Any]) -> float | None:
    """The threshold the record freezes for the policy (and, for R2, the capacity) of ``entry``."""
    if entry["policy"] != "R2":
        return float(thresholds[entry["policy"].lower()])
    for item in thresholds["r2"]:
        if _close(item["capacity"], entry["capacity"]):
            return float(item["threshold"])
    return None


def _declared_pair(declaration: EvaluationDeclaration, capacity: float, k: int) -> bool:
    budgets = zip(declaration.capacity.shares, declaration.capacity.budgets, strict=True)
    return any(_close(float(share), capacity) and budget == k for share, budget in budgets)


def _declared_capacities(declaration: EvaluationDeclaration, capacities: Sequence[float]) -> bool:
    shares = declaration.capacity.shares
    return len(capacities) == len(shares) and all(
        _close(float(share), value) for share, value in zip(shares, capacities, strict=True)
    )


def _evaluation_context(record: Mapping[str, Any], context: ConfirmatoryContext) -> list[str]:
    declaration, test = context.declaration, context.test
    problems = [
        f"test_evaluation.days.{name}: must be {expected} (the Test partition)"
        for name, expected in (("first", test.first_day), ("last", test.last_day))
        if record["days"][name] != expected
    ]
    if not _declared_capacities(declaration, [e["capacity"] for e in record["r3"]]):
        problems.append("test_evaluation.r3: every declared capacity, in the declared order")
    for kind in ("r3", "policies"):
        for i, entry in enumerate(record[kind]):
            where = f"test_evaluation.{kind}[{i}]"
            if not _declared_pair(declaration, entry["capacity"], entry["k"]):
                problems.append(f"{where}: capacity and K must be a declared pair")
            if entry["counts"]["transactions"] != test.rows:
                problems.append(f"{where}.counts.transactions: must be the {test.rows} Test rows")
            names = R3_MEASURES if kind == "r3" else THRESHOLD_MEASURES
            problems += _interval_set_problems(where, entry, names, declaration)
    problems += _interval_set_problems(
        "test_evaluation.ranking", record["ranking"], RANKING_MEASURES, declaration
    )
    return problems


TEST_EVALUATION = _contract(
    "test_evaluation",
    1,
    "The Test outcomes of one configuration (or the logistic-regression reference) from the single "
    "scoring pass: R3 at the four V2-derived budgets, the uncapped R0, R1 and R2 against each "
    "budget, and the ranking measures, with the 95 % bootstrap intervals chosen by each outcome's "
    "dependence (protocol, Statistical Plan; DR-006).",
    (
        Field(
            "configuration", CONFIGURATION, "Configuration (or the logistic-regression reference)"
        ),
        Field("partition", Choice(("Test",)), "The partition evaluated"),
        Field("days", RECORD_DAYS, "Relative days evaluated"),
        Field("scores_sha256", SHA256, "SHA-256 of the score file of the single scoring pass"),
        Field(
            "thresholds",
            THRESHOLDS,
            "The thresholds of R0, R1 and R2 (null for the reference)",
            nullable=True,
        ),
        Field("ranking", RANKING_ENTRY, "Whole-ranking measures"),
        Field("r3", ListOf(R3_ENTRY, min_items=1), "R3 at each declared capacity"),
        Field(
            "policies",
            ListOf(POLICY_ENTRY),
            "R0, R1 and R2 at each declared capacity (empty for the reference)",
        ),
    ),
    _evaluation_rules,
)

FIDELITY_TERM = Object(
    (
        Field("day", POSITIVE, "Relative day d"),
        Field("alerts", COUNT, "Threshold-generated alerts n_d"),
        Field("transactions", POSITIVE, "Transactions N_d"),
        Field("rate", Number(), "ln((n_d / N_d) / c); null without an alert", nullable=True),
        Field("volume", Number(), "ln(N_d / N̄_V2); null without an alert", nullable=True),
        Field(
            "budget",
            Number(),
            "ln(c N̄_V2 / K), the rounding of K; null without an alert",
            nullable=True,
        ),
        Field("total", Number(), "ln(n_d / K); null without an alert", nullable=True),
    )
)
FIDELITY = Object(
    (
        Field("days", POSITIVE, "Days evaluated"),
        Field("mean_alerts", Number(minimum=0), "Mean daily alerts"),
        Field("mean_deviation", Number(), "Mean of n_d − K"),
        Field("mean_absolute_deviation", Number(minimum=0), "Mean of |n_d − K|"),
        Field("max_absolute_deviation", COUNT, "Largest |n_d − K|"),
        Field("mean_relative_deviation", Number(), "Mean of (n_d − K) / K"),
        Field("days_above_budget", COUNT, "Days with n_d > K"),
        Field("days_below_budget", COUNT, "Days with n_d < K"),
        Field("share_above_budget", UNIT_P, "Share of days above K"),
        Field("share_below_budget", UNIT_P, "Share of days below K"),
        Field("total_overflow", COUNT, "Sum of max(n_d − K, 0)"),
        Field("total_unused", COUNT, "Sum of max(K − n_d, 0)"),
        Field("mean_load_ratio", Number(minimum=0), "Mean of n_d / K"),
        Field("zero_alert_days", COUNT, "Days without an alert"),
        Field("reference_volume", Number(0, exclusive_minimum=True), "Mean daily V2 volume N̄_V2"),
        Field(
            "mean_rate", Number(), "Mean rate component over the days with an alert", nullable=True
        ),
        Field("mean_volume", Number(), "Mean volume component", nullable=True),
        Field("mean_budget", Number(), "Mean budget component", nullable=True),
        Field("mean_total", Number(), "Mean of ln(n_d / K)", nullable=True),
        Field("terms", ListOf(FIDELITY_TERM, min_items=1), "The decomposition of each day"),
    )
)
WORKLOAD_ENTRY = Object(
    (
        Field("capacity", CAPACITY_SHARE, "Review-capacity share c"),
        Field("k", POSITIVE, "Fixed daily review budget K"),
        Field("threshold", Number(0, 1), "The frozen R2 threshold"),
        Field("fidelity", FIDELITY, "Daily-budget fidelity of R2 against K"),
    )
)


def _mean_close(values: Sequence[float], stated: float | None) -> bool:
    return (
        (stated is None)
        if not values
        else (stated is not None and _near(math.fsum(values) / len(values), stated))
    )


def _fidelity_problems(where: str, entry: Mapping[str, Any]) -> list[str]:
    fidelity, k, capacity = entry["fidelity"], entry["k"], entry["capacity"]
    terms = fidelity["terms"]
    days = len(terms)
    alerts = [t["alerts"] for t in terms]
    deviation = [n - k for n in alerts]
    problems = []
    checks = {
        "days": days,
        "mean_alerts": math.fsum(alerts) / days,
        "mean_deviation": math.fsum(deviation) / days,
        "mean_absolute_deviation": math.fsum(abs(d) for d in deviation) / days,
        "max_absolute_deviation": max(abs(d) for d in deviation),
        "mean_relative_deviation": math.fsum(deviation) / days / k,
        "days_above_budget": sum(d > 0 for d in deviation),
        "days_below_budget": sum(d < 0 for d in deviation),
        "share_above_budget": sum(d > 0 for d in deviation) / days,
        "share_below_budget": sum(d < 0 for d in deviation) / days,
        "total_overflow": sum(max(d, 0) for d in deviation),
        "total_unused": sum(max(-d, 0) for d in deviation),
        "mean_load_ratio": math.fsum(n / k for n in alerts) / days,
        "zero_alert_days": sum(n == 0 for n in alerts),
    }
    problems += [
        f"{where}.fidelity.{name}: must be {value!r}"
        for name, value in checks.items()
        if not _near(fidelity[name], value)
    ]
    reference = fidelity["reference_volume"]
    parts: dict[str, list[float]] = {"rate": [], "volume": [], "budget": [], "total": []}
    for i, term in enumerate(terms):
        label = f"{where}.fidelity.terms[{i}]"
        values = [term[name] for name in parts]
        if term["alerts"] == 0:
            if any(v is not None for v in values):
                problems.append(f"{label}: a day without an alert has no logarithm")
            continue
        if any(v is None for v in values):
            problems.append(f"{label}: a day with an alert has all four terms")
            continue
        expected = {
            "rate": math.log(term["alerts"] / term["transactions"] / capacity),
            "volume": math.log(term["transactions"] / reference),
            "budget": math.log(capacity * reference / k),
            "total": math.log(term["alerts"] / k),
        }
        problems += [
            f"{label}.{name}: must be {value!r}"
            for name, value in expected.items()
            if not _near(term[name], value)
        ]
        if not _near(term["rate"] + term["volume"] + term["budget"], term["total"]):
            problems.append(f"{label}: rate + volume + budget must equal ln(n_d / K)")
        for name in parts:
            parts[name].append(term[name])
    problems += [
        f"{where}.fidelity.mean_{name}: must be the mean over the days with an alert"
        for name, values in parts.items()
        if not _mean_close(values, fidelity[f"mean_{name}"])
    ]
    return problems


def _workload_rules(record: Mapping[str, Any]) -> list[str]:
    problems = []
    entries = record["entries"]
    capacities = [e["capacity"] for e in entries]
    if len(set(capacities)) != len(capacities):
        problems.append("test_workload.entries: a capacity is evaluated once")
    days = list(range(record["days"]["first"], record["days"]["last"] + 1))
    for i, entry in enumerate(entries):
        where = f"test_workload.entries[{i}]"
        if [t["day"] for t in entry["fidelity"]["terms"]] != days:
            problems.append(f"{where}.fidelity.terms: one entry per evaluated day, in day order")
            continue
        problems += _fidelity_problems(where, entry)
    return problems


def _workload_context(record: Mapping[str, Any], context: ConfirmatoryContext) -> list[str]:
    declaration, test = context.declaration, context.test
    problems = [
        f"test_workload.days.{name}: must be {expected} (the Test partition)"
        for name, expected in (("first", test.first_day), ("last", test.last_day))
        if record["days"][name] != expected
    ]
    entries = record["entries"]
    if not _declared_capacities(declaration, [e["capacity"] for e in entries]):
        problems.append("test_workload.entries: every declared capacity, in the declared order")
    for i, entry in enumerate(entries):
        where = f"test_workload.entries[{i}]"
        if not _declared_pair(declaration, entry["capacity"], entry["k"]):
            problems.append(f"{where}: capacity and K must be a declared pair")
        if not _near(entry["fidelity"]["reference_volume"], context.reference_volume):
            problems.append(
                f"{where}.fidelity.reference_volume: must be the V2 mean daily volume "
                f"{context.reference_volume!r} (V2 rows over V2 days)"
            )
    return problems


TEST_WORKLOAD = _contract(
    "test_workload",
    1,
    "The daily-budget fidelity of R2 for one configuration on the Test period (RQ2): daily "
    "alerts against K, the deviation statistics and the rate/volume decomposition against the V2 "
    "reference volume, at each declared capacity. Point values only (DR-003, DR-009).",
    (
        Field("configuration", STUDY_CONFIGURATION, "Configuration"),
        Field("partition", Choice(("Test",)), "The partition evaluated"),
        Field("days", RECORD_DAYS, "Relative days evaluated"),
        Field("entries", ListOf(WORKLOAD_ENTRY, min_items=1), "Fidelity at each declared capacity"),
    ),
    _workload_rules,
)

CALIBRATION_FILE = Object(
    (
        Field("path", Text(), "Path of the frozen calibrator, relative to the project root"),
        Field("sha256", SHA256, "SHA-256 of the calibrator file"),
    )
)


def _calibration_rules(record: Mapping[str, Any]) -> list[str]:
    diagnostic = record["diagnostic"]
    problems = _reliability(diagnostic, "test_calibration.diagnostic.reliability")
    if diagnostic["rows_used"] != "held_out":
        problems.append("test_calibration.diagnostic.rows_used: the Test rows are held out")
    if diagnostic["frauds"] > diagnostic["rows"]:
        problems.append("test_calibration.diagnostic.frauds: exceed the rows")
    if record["days"] != diagnostic["days"]:
        problems.append("test_calibration.diagnostic.days: must be the record's days")
    if record["role"] != CALIBRATION_ROLES[record["method"]]:
        problems.append(
            "test_calibration.role: Platt is the primary method, isotonic a sensitivity"
        )
    if sorted(record["intervals"]) != sorted(CALIBRATION_MEASURES):
        problems.append(
            f"test_calibration.intervals: must hold exactly {list(CALIBRATION_MEASURES)}"
        )
    return problems


def _calibration_context(record: Mapping[str, Any], context: ConfirmatoryContext) -> list[str]:
    declaration, test = context.declaration, context.test
    problems = [
        f"test_calibration.days.{name}: must be {expected} (the Test partition)"
        for name, expected in (("first", test.first_day), ("last", test.last_day))
        if record["days"][name] != expected
    ]
    if record["diagnostic"]["rows"] != test.rows:
        problems.append(f"test_calibration.diagnostic.rows: must be the {test.rows} Test rows")
    for name, interval in record["intervals"].items():
        label = f"test_calibration.intervals.{name}"
        if interval["lower"] > interval["upper"]:
            problems.append(f"{label}: lower exceeds upper")
        if not _close(interval["level"], declaration.levels.difference):
            problems.append(f"{label}.level: must be the declared {declaration.levels.difference}")
        if interval["replicates"] != declaration.resampling.replicates:
            problems.append(f"{label}.replicates: must be the declared B")
        if interval["undefined"] >= interval["replicates"]:
            problems.append(f"{label}.undefined: fewer than the replicates")
    return problems


TEST_CALIBRATION = _contract(
    "test_calibration",
    1,
    "The calibration diagnostics of one configuration and one frozen calibrator (Platt, "
    "primary; isotonic, sensitivity) on the Test period: Brier score, clipped log loss, "
    "calibration intercept and slope, the reliability table, with 95 % day-block intervals "
    "(protocol, Calibration; DR-006).",
    (
        Field("configuration", STUDY_CONFIGURATION, "Configuration"),
        Field("method", Choice(CALIBRATION_METHODS), "Calibration method"),
        Field(
            "role", Choice(("primary", "sensitivity")), "Platt is primary, isotonic a sensitivity"
        ),
        Field("partition", Choice(("Test",)), "The partition evaluated"),
        Field("days", RECORD_DAYS, "Relative days evaluated"),
        Field("calibrator", CALIBRATION_FILE, "The frozen calibrator that was applied"),
        Field("diagnostic", DIAGNOSTIC, "The diagnostics on the Test rows (held out)"),
        Field(
            "intervals",
            MapOf(INTERVAL, keys=Choice(CALIBRATION_MEASURES)),
            "95 % day-block intervals; a measure undefined on every resample stops the run",
        ),
    ),
    _calibration_rules,
)


# --- test_rq3 -------------------------------------------------------------------------------------

RQ3_REGIMES = ("identical", "r4_cheaper", "r3_cheaper", "inconclusive")
RQ3_INTERVALS = (
    "unused",
    "frauds_forgone",
    "utilisation",
    "cost_r3",
    "cost_r4",
    "cost_difference",
    "relative_difference",
    "declined_fraud_rate",
    "declined_mean_probability",
    "fraud_value_exposed_r4",
    "fraud_value_forgone",
    "forgone_per_unused",
    "amount_weighted_recall_r4",
)
RQ3_NULLABLE = (
    "declined_fraud_rate",
    "declined_mean_probability",
    "forgone_per_unused",
    "amount_weighted_recall_r4",
)
RQ3_GRID_ROW = Object(
    (
        *GRID_ROW.fields,
        Field(
            "intervals",
            MapOf(INTERVAL, keys=Choice(RQ3_INTERVALS), nullable_values=True),
            "95 % day-block intervals (null where the point value is null)",
        ),
        Field(
            "frauds_forgone_pigeonhole",
            INTERVAL,
            "Two-way pigeonhole interval of the frauds forgone (sensitivity, stream 6)",
        ),
        Field("regime", Choice(RQ3_REGIMES), "Regime of the cell from the interval of ΔJ"),
    )
)
RQ3_CAPACITY = Object(
    (
        Field("capacity", CAPACITY_SHARE, "Review-capacity share c"),
        Field("k", POSITIVE, "Fixed daily review budget K"),
        *(COST_ANALYSIS.body.field(name) for name in ("queue", "saturation", "break_even")),
        COST_ANALYSIS.body.field("friction"),
        COST_ANALYSIS.body.field("curve"),
        Field("grid", ListOf(RQ3_GRID_ROW, min_items=1), "The ratio grid of DR-007 §2"),
    )
)


def _regime_of(unused: int, interval: Mapping[str, Any]) -> str:
    if unused == 0:
        return "identical"
    if interval["upper"] < 0:
        return "r4_cheaper"
    if interval["lower"] > 0:
        return "r3_cheaper"
    return "inconclusive"


def _rq3_cell_problems(where: str, row: Mapping[str, Any]) -> list[str]:
    problems = []
    intervals = row["intervals"]
    if sorted(intervals) != sorted(RQ3_INTERVALS):
        return [f"{where}.intervals: must hold exactly {list(RQ3_INTERVALS)}"]
    for name, interval in intervals.items():
        if (interval is None) != (name in RQ3_NULLABLE and row[name] is None):
            problems.append(f"{where}.intervals.{name}: null exactly when the point value is null")
        elif interval is not None and interval["lower"] > interval["upper"]:
            problems.append(f"{where}.intervals.{name}: lower exceeds upper")
    delta = intervals["cost_difference"]
    if delta is not None and row["regime"] != _regime_of(row["unused"], delta):
        problems.append(f"{where}.regime: must follow the interval of the cost difference")
    pigeonhole = row["frauds_forgone_pigeonhole"]
    if pigeonhole["lower"] > pigeonhole["upper"]:
        problems.append(f"{where}.frauds_forgone_pigeonhole: lower exceeds upper")
    return problems


def _rq3_rules(record: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    capacities = [entry["capacity"] for entry in record["capacities"]]
    if len(set(capacities)) != len(capacities):
        problems.append("test_rq3.capacities: a capacity is analysed once")
    frauds = {entry["queue"]["frauds"] for entry in record["capacities"]}
    if len(frauds) > 1:
        problems.append("test_rq3.capacities: every capacity counts the same frauds")
    for i, entry in enumerate(record["capacities"]):
        where = f"test_rq3.capacities[{i}]"
        frame = {**entry, "days": record["days"]}
        for check in (_grid_problems, _break_even_problems, _curve_problems):
            problems += [p.replace("cost_analysis", where) for p in check(frame)]
        if entry["queue"]["reviews"] > entry["k"] * (
            record["days"]["last"] - record["days"]["first"] + 1
        ):
            problems.append(f"{where}.queue.reviews: at most K a day")
        for j, row in enumerate(entry["grid"]):
            problems += _rq3_cell_problems(f"{where}.grid[{j}]", row)
    return problems


def _rq3_context(record: Mapping[str, Any], context: ConfirmatoryContext) -> list[str]:
    declaration, test = context.declaration, context.test
    problems = [
        f"test_rq3.days.{name}: must be {expected} (the Test partition)"
        for name, expected in (("first", test.first_day), ("last", test.last_day))
        if record["days"][name] != expected
    ]
    entries = record["capacities"]
    if not _declared_capacities(declaration, [e["capacity"] for e in entries]):
        problems.append("test_rq3.capacities: every declared capacity, in the declared order")
    for i, entry in enumerate(entries):
        where = f"test_rq3.capacities[{i}]"
        if not _declared_pair(declaration, entry["capacity"], entry["k"]):
            problems.append(f"{where}: capacity and K must be a declared pair")
        if [float(r) for r in declaration.rq3.ratio_grid] != [
            row["ratio"] for row in entry["grid"]
        ]:
            problems.append(f"{where}.grid: the declared ratio grid, in order")
        for j, row in enumerate(entry["grid"]):
            cells = [*row["intervals"].values(), row["frauds_forgone_pigeonhole"]]
            for interval in (c for c in cells if c is not None):
                if not _close(interval["level"], declaration.levels.difference):
                    problems.append(f"{where}.grid[{j}]: interval level must be the declared one")
                if interval["replicates"] != declaration.resampling.replicates:
                    problems.append(
                        f"{where}.grid[{j}]: interval replicates must be the declared B"
                    )
                if interval["undefined"] >= interval["replicates"]:
                    problems.append(f"{where}.grid[{j}]: undefined replicates fewer than drawn")
    return problems


TEST_RQ3 = _contract(
    "test_rq3",
    1,
    "The R4 outcomes and the RQ3 cost analysis of one configuration under one frozen calibrator "
    "on the Test period: for each capacity the queue totals, the exact curve, the saturation and "
    "break-even ratios, the C_FP conversions and the ratio grid, each grid row with 95 % "
    "day-block intervals, a pigeonhole sensitivity interval of the frauds forgone and the regime "
    "of the cell. Descriptive; scenario costs are assumptions (DR-007 §7 and §9).",
    (
        Field("configuration", STUDY_CONFIGURATION, "Configuration"),
        Field("calibrator", Choice(CALIBRATION_METHODS), "Calibrator giving the probabilities"),
        Field("partition", Choice(("Test",)), "The partition evaluated"),
        Field("days", RECORD_DAYS, "Relative days evaluated"),
        Field("scores_sha256", SHA256, "SHA-256 of the score file of the single scoring pass"),
        Field("calibrator_file", CALIBRATION_FILE, "The frozen calibrator that was applied"),
        Field("capacities", ListOf(RQ3_CAPACITY, min_items=1), "The analysis at each capacity"),
    ),
    _rq3_rules,
)


# --- test_false_negative --------------------------------------------------------------------------

FN_SLICE = Object(
    (
        Field("variable", Choice(FN_VARIABLES), "Slicing variable"),
        Field("level", Text(), "Level of the variable"),
        Field("frauds", COUNT, "Test frauds in the slice (the missed frauds, for the score band)"),
        Field("misses", COUNT, "False negatives in the slice"),
        Field(
            "miss_rate", UNIT_P, "misses / frauds (null: no fraud; the score band has none)", True
        ),
        Field("share", UNIT_P, "misses / total false negatives (null: none)", nullable=True),
        Field("status", Choice(FN_STATUSES), "Whether the slice has an interval and why not"),
        Field("interval", INTERVAL, "95 % pigeonhole interval of the miss rate", nullable=True),
    )
)
FN_BOUNDARIES = Object(
    (
        Field("amount_edges", ListOf(Number(), min_items=1), "Interior Train amount-band edges"),
        Field(
            "kept",
            Object(
                tuple(
                    Field(name, MapOf(COUNT), f"{name}: levels kept with their V2 fraud counts")
                    for name in FN_POOLED
                )
            ),
            "Levels kept by the V2 pooling rule; every other level is pooled into other",
        ),
    )
)


def _fn_slice_problems(where: str, row: Mapping[str, Any], total: int) -> list[str]:
    problems = []
    if row["misses"] > row["frauds"]:
        problems.append(f"{where}.misses: exceed the frauds")
    expected_rate = row["misses"] / row["frauds"] if row["frauds"] else None
    expected_share = row["misses"] / total if total else None
    if row["variable"] == "score_band":
        if row["frauds"] != row["misses"]:
            problems.append(f"{where}: the score band holds missed frauds only")
        if row["miss_rate"] is not None or row["interval"] is not None:
            problems.append(f"{where}: the score band has no miss rate and no interval")
        if row["status"] != "not_applicable":
            problems.append(f"{where}.status: not_applicable for the score band")
    else:
        if (row["miss_rate"] is None) != (expected_rate is None) or (
            expected_rate is not None and not _near(row["miss_rate"], expected_rate)
        ):
            problems.append(f"{where}.miss_rate: must be misses / frauds")
        if (row["interval"] is None) != (row["status"] != "interval"):
            problems.append(f"{where}: an interval exactly when the status is interval")
        if row["status"] == "not_applicable":
            problems.append(f"{where}.status: not_applicable is for the score band")
    if (row["share"] is None) != (expected_share is None) or (
        expected_share is not None and not _near(row["share"], expected_share)
    ):
        problems.append(f"{where}.share: must be misses / the false negatives")
    if row["interval"] is not None and row["interval"]["lower"] > row["interval"]["upper"]:
        problems.append(f"{where}.interval: lower exceeds upper")
    return problems


def _fn_rules(record: Mapping[str, Any]) -> list[str]:
    problems = []
    total, frauds = record["false_negatives"], record["frauds"]
    if total > frauds:
        problems.append("test_false_negative.false_negatives: exceed the frauds")
    slices = record["slices"]
    order = list(dict.fromkeys(row["variable"] for row in slices))
    if order != list(FN_VARIABLES):
        problems.append(f"test_false_negative.slices: the variables must be {list(FN_VARIABLES)}")
    for i, row in enumerate(slices):
        problems += _fn_slice_problems(f"test_false_negative.slices[{i}]", row, total)
    for variable in order:
        rows = [row for row in slices if row["variable"] == variable]
        levels = [row["level"] for row in rows]
        if len(set(levels)) != len(levels):
            problems.append(f"test_false_negative.slices: a {variable} level is listed once")
        sums = (sum(r["frauds"] for r in rows), sum(r["misses"] for r in rows))
        if variable == "score_band":
            if sums[1] != total:
                problems.append(
                    "test_false_negative.slices: score bands sum to the false negatives"
                )
        elif sums != (frauds, total):
            problems.append(
                f"test_false_negative.slices: the {variable} levels must sum to the frauds and "
                "the false negatives"
            )
    kept = record["boundaries"]["kept"]
    for variable in FN_POOLED:
        levels = [r["level"] for r in slices if r["variable"] == variable]
        if levels != [*kept[variable], FN_OTHER]:
            problems.append(
                f"test_false_negative.slices: {variable} levels are the kept ones, then other"
            )
    edges = record["boundaries"]["amount_edges"]
    if any(b <= a for a, b in pairwise(edges)):
        problems.append("test_false_negative.boundaries.amount_edges: increasing")
    return problems


def _fn_context(record: Mapping[str, Any], context: ConfirmatoryContext) -> list[str]:
    declaration, test = context.declaration, context.test
    rule = declaration.false_negative
    problems = [
        f"test_false_negative.days.{name}: must be {expected} (the Test partition)"
        for name, expected in (("first", test.first_day), ("last", test.last_day))
        if record["days"][name] != expected
    ]
    if not _close(record["capacity"], float(declaration.capacity.primary)) or not _declared_pair(
        declaration, record["capacity"], record["k"]
    ):
        problems.append("test_false_negative.capacity: the declared primary capacity and its K")
    if len(record["boundaries"]["amount_edges"]) != rule.amount_quantiles - 1:
        problems.append(f"test_false_negative.boundaries: {rule.amount_quantiles - 1} amount edges")
    problems += [
        f"test_false_negative.boundaries.kept: {variable} level {level} has fewer than "
        f"{rule.minimum_frauds} V2 frauds"
        for variable, levels in record["boundaries"]["kept"].items()
        for level, n in levels.items()
        if n < rule.minimum_frauds
    ]
    for variable, expected in fn_declared_levels(declaration).items():
        found = [row["level"] for row in record["slices"] if row["variable"] == variable]
        if found != list(expected):
            problems.append(
                f"test_false_negative.slices: the {variable} levels must be {list(expected)}"
            )
    for i, row in enumerate(record["slices"]):
        where = f"test_false_negative.slices[{i}]"
        if row["variable"] == "score_band":
            continue
        enough = row["frauds"] >= rule.minimum_frauds
        if enough != (row["status"] == "interval"):
            problems.append(
                f"{where}.status: an interval exactly from {rule.minimum_frauds} frauds"
            )
        interval = row["interval"]
        if interval is not None:
            if not _close(interval["level"], rule.interval_level):
                problems.append(f"{where}.interval.level: must be the declared level")
            if interval["replicates"] != declaration.resampling.replicates:
                problems.append(f"{where}.interval.replicates: must be the declared B")
            if interval["undefined"] >= interval["replicates"]:
                problems.append(f"{where}.interval.undefined: fewer than the replicates")
            if not 0 <= interval["lower"] <= interval["upper"] <= 1:
                problems.append(f"{where}.interval: a miss rate lies in [0, 1]")
    return problems


TEST_FALSE_NEGATIVE = _contract(
    "test_false_negative",
    1,
    "The false-negative analysis of one configuration on the Test period: the frauds outside the "
    "daily top-K of R3 at the primary capacity, by the pre-declared slices, with the miss rate, "
    "the share of the false negatives and the two-way pigeonhole interval of the miss rate. "
    "Exploratory and non-causal; the slice boundaries are fixed without Test labels.",
    (
        Field("configuration", STUDY_CONFIGURATION, "Configuration"),
        Field("partition", Choice(("Test",)), "The partition analysed"),
        Field("days", RECORD_DAYS, "Relative days analysed"),
        Field("scores_sha256", SHA256, "SHA-256 of the score file of the single scoring pass"),
        Field("capacity", CAPACITY_SHARE, "The primary capacity share c"),
        Field("k", POSITIVE, "Fixed daily review budget K"),
        Field("frauds", POSITIVE, "Test frauds"),
        Field("false_negatives", COUNT, "Frauds outside the daily top-K"),
        Field("boundaries", FN_BOUNDARIES, "The label-free slice boundaries"),
        Field("slices", ListOf(FN_SLICE, min_items=1), "Every slice, in the declared order"),
    ),
    _fn_rules,
)


RECORD_CONTRACTS: dict[str, RecordContract] = {
    contract.kind: contract
    for contract in (
        TEST_RUN,
        TEST_PHASE,
        TEST_SCORES,
        TEST_INFERENCE,
        TEST_EVALUATION,
        TEST_WORKLOAD,
        TEST_CALIBRATION,
        TEST_RQ3,
        TEST_FALSE_NEGATIVE,
        TEST_RUN_END,
    )
}

# The rules that need the Test partition or the evaluation declaration, by record kind.
CONTEXT_CHECKS: dict[str, tuple[ContextCheck, ...]] = {
    "test_run": (_run_partition, _seed_rule),
    "test_phase": (_seed_rule,),
    "test_scores": (_scores_partition, _seed_rule),
    "test_inference": (_inference_rules, _seed_rule),
    "test_evaluation": (_evaluation_context, _seed_rule),
    "test_workload": (_workload_context, _seed_rule),
    "test_calibration": (_calibration_context, _seed_rule),
    "test_rq3": (_rq3_context, _seed_rule),
    "test_false_negative": (_fn_context, _seed_rule),
    "test_run_end": (_seed_rule,),
}


def to_json_value(value: object) -> Any:
    """``value`` as a JSON value for a record: dataclasses become objects, enums their value,
    tuples lists and numpy scalars Python numbers."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def scores_bundle_sha256(file_hashes: Sequence[str]) -> str:
    """The SHA-256 of the sorted score-file hashes of the scoring pass, joined by newlines: the
    identity of the scores that the evaluation phase uses."""
    return hashlib.sha256("\n".join(sorted(file_hashes)).encode("utf-8")).hexdigest()


def run_problems(records: Sequence[Mapping[str, Any]], *, complete: bool = True) -> list[str]:
    """The rules between the records of one run (each already valid): the phases in order of
    their start follow ``phase_sequence_problems``; the score records name the frozen model of
    their configuration, and a completed scoring names the bundle of their files; the inference
    is of the frozen H3 configuration and follows a completed evaluation. For a ``complete`` run
    there is exactly one score record for every configuration and the reference, a completed
    scoring and, with an inference record, a completed evaluation."""
    runs = [record for record in records if record["record"] == "test_run"]
    if len(runs) != 1:
        return ["run: needs exactly one test_run record"]
    frozen = runs[0]["frozen_inputs"]
    phases = sorted(
        (record for record in records if record["record"] == "test_phase"),
        key=lambda record: (record["started"], PHASES.index(record["phase"]), record["attempt"]),
    )
    problems = phase_sequence_problems(phases)
    scores = [record for record in records if record["record"] == "test_scores"]
    configurations = [record["configuration"] for record in scores]
    if len(set(configurations)) != len(configurations):
        problems.append("run: a configuration has more than one test_scores record")
    if complete and set(configurations) != set(ALL_CONFIGURATIONS):
        problems.append(
            "run: needs one test_scores record for every configuration and the reference"
        )
    problems += [
        f"run: test_scores of {record['configuration']} names another model than the frozen one"
        for record in scores
        if record["model"] != frozen["models"].get(record["configuration"])
    ]
    scoring = [p for p in phases if p["phase"] == "scoring" and p["outcome"] == "completed"]
    bundle = scores_bundle_sha256([record["file"]["sha256"] for record in scores])
    if scores and not scoring:
        problems.append("run: score records exist without a completed scoring")
    if complete and not scoring:
        problems.append("run: needs a completed scoring")
    problems += [
        "run: the completed scoring names other scores than the score records"
        for phase in scoring
        if phase["scores_sha256"] != bundle
    ]
    inferences = [record for record in records if record["record"] == "test_inference"]
    evaluated = any(p["phase"] == "evaluation" and p["outcome"] == "completed" for p in phases)
    if len(inferences) > 1:
        problems.append("run: more than one test_inference record")
    for inference in inferences:
        if not evaluated:
            problems.append("run: the inference follows a completed evaluation")
        if inference["h3"]["configuration"] != frozen["h3_configuration"]:
            problems.append("run: the inference is not of the frozen H3 configuration")
    problems += _outcome_problems(records, evaluated, scores, frozen, complete=complete)
    return problems


def _outcome_problems(
    records: Sequence[Mapping[str, Any]],
    evaluated: bool,
    scores: Sequence[Mapping[str, Any]],
    frozen: Mapping[str, Any],
    *,
    complete: bool,
) -> list[str]:
    """The rules between the outcome records and the rest of the run: one record per
    configuration (the reference only in ``test_evaluation``) or per configuration and method, each
    after a completed evaluation, the evaluations naming the score file of their configuration, the
    workload records agreeing with the R2 evaluation of their configuration and the calibration
    records naming a frozen calibrator. A complete run with a completed evaluation holds the whole
    set of every kind."""
    study = tuple(ALL_CONFIGURATIONS[:-1])
    kinds: dict[
        str, tuple[Callable[[Mapping[str, Any]], tuple[str, ...]], set[tuple[str, ...]]]
    ] = {
        "test_evaluation": (
            lambda r: (r["configuration"],),
            {(name,) for name in ALL_CONFIGURATIONS},
        ),
        "test_workload": (lambda r: (r["configuration"],), {(name,) for name in study}),
        "test_calibration": (
            lambda r: (r["configuration"], r["method"]),
            {(name, method) for name in study for method in CALIBRATION_METHODS},
        ),
        "test_rq3": (
            lambda r: (r["configuration"], r["calibrator"]),
            {(name, method) for name in study for method in CALIBRATION_METHODS},
        ),
        "test_false_negative": (lambda r: (r["configuration"],), {(name,) for name in study}),
    }
    problems: list[str] = []
    score_hash = {r["configuration"]: r["file"]["sha256"] for r in scores}
    for kind, (key, wanted) in kinds.items():
        found = [record for record in records if record["record"] == kind]
        keys = [key(record) for record in found]
        if len(set(keys)) != len(keys):
            problems.append(f"run: a {kind} record is written once for each of its subjects")
        if found and not evaluated:
            problems.append(f"run: the {kind} records follow a completed evaluation")
        if complete and evaluated and set(keys) != wanted:
            problems.append(f"run: needs a {kind} record for every subject ({len(wanted)})")
    problems += [
        f"run: {record['record']} of {record['configuration']} names another score file"
        for record in records
        if record["record"] in ("test_evaluation", "test_rq3", "test_false_negative")
        and record["scores_sha256"] != score_hash.get(record["configuration"])
    ]
    problems += [
        f"run: test_rq3 of {record['configuration']} ({record['calibrator']}) names "
        "a calibrator that is not a frozen input"
        for record in records
        if record["record"] == "test_rq3" and record["calibrator_file"] not in frozen["calibrators"]
    ]
    problems += [
        f"run: test_calibration of {record['configuration']} ({record['method']}) names "
        "a calibrator that is not a frozen input"
        for record in records
        if record["record"] == "test_calibration"
        and record["calibrator"] not in frozen["calibrators"]
    ]
    evaluations = {r["configuration"]: r for r in records if r["record"] == "test_evaluation"}
    for workload in (r for r in records if r["record"] == "test_workload"):
        problems += _workload_agreement(workload, evaluations.get(workload["configuration"]))
    for analysis in (r for r in records if r["record"] == "test_rq3"):
        problems += _rq3_agreement(analysis, evaluations.get(analysis["configuration"]))
    tables = [r for r in records if r["record"] == "test_false_negative"]
    for table in tables:
        problems += _fn_agreement(table, evaluations.get(table["configuration"]))
    problems += _fn_shared_boundaries(tables)
    return problems


def _fn_shared_boundaries(tables: Sequence[Mapping[str, Any]]) -> list[str]:
    """The slice boundaries and the slices they define are those of all configurations: the
    boundaries, and the fraud count of every level of every variable except the score band, are
    identical in all the false-negative records of a run."""
    if not tables:
        return []

    def shared(table: Mapping[str, Any]) -> tuple[Any, Any]:
        slices = [
            (row["variable"], row["level"], row["frauds"])
            for row in table["slices"]
            if row["variable"] != "score_band"
        ]
        return table["boundaries"], slices

    first = shared(tables[0])
    return [
        f"run: test_false_negative of {table['configuration']} has other slice boundaries or "
        "slice fraud counts than the first configuration's"
        for table in tables[1:]
        if shared(table) != first
    ]


def _fn_agreement(table: Mapping[str, Any], evaluation: Mapping[str, Any] | None) -> list[str]:
    """The false negatives are those of the evaluation record's R3 at the same capacity."""
    if evaluation is None:
        return []
    r3 = next((e for e in evaluation["r3"] if _close(e["capacity"], table["capacity"])), None)
    if r3 is None or (r3["counts"]["frauds"], r3["counts"]["false_negatives"]) != (
        table["frauds"],
        table["false_negatives"],
    ):
        return [
            f"run: test_false_negative of {table['configuration']} counts other frauds or "
            "false negatives than the evaluation"
        ]
    return []


def _rq3_agreement(analysis: Mapping[str, Any], evaluation: Mapping[str, Any] | None) -> list[str]:
    """The R3 queue of the cost analysis is the R3 result of the evaluation record."""
    if evaluation is None:
        return []
    r3 = {entry["capacity"]: entry["counts"] for entry in evaluation["r3"]}
    return [
        f"run: test_rq3 of {analysis['configuration']} ({analysis['calibrator']}) counts another "
        f"R3 queue than the evaluation at capacity {entry['capacity']}"
        for entry in analysis["capacities"]
        if (entry["capacity"] not in r3)
        or (
            r3[entry["capacity"]]["selected"],
            r3[entry["capacity"]]["true_positives"],
            r3[entry["capacity"]]["frauds"],
        )
        != (entry["queue"]["reviews"], entry["queue"]["captured"], entry["queue"]["frauds"])
    ]


def _workload_agreement(
    workload: Mapping[str, Any], evaluation: Mapping[str, Any] | None
) -> list[str]:
    """The workload record describes the R2 alerts of the evaluation record of its configuration."""
    if evaluation is None:
        return []
    name = workload["configuration"]
    r2 = {e["capacity"]: e for e in evaluation["policies"] if e["policy"] == "R2"}
    problems = []
    for entry in workload["entries"]:
        match = r2.get(entry["capacity"])
        terms = entry["fidelity"]["terms"]
        if match is None or (match["k"], match["threshold"]) != (entry["k"], entry["threshold"]):
            problems.append(f"run: test_workload of {name} names another R2 threshold or K")
        elif [(t["day"], t["alerts"], t["transactions"]) for t in terms] != [
            (row["day"], row["selected"], row["transactions"]) for row in match["daily"]
        ]:
            problems.append(
                f"run: test_workload of {name} counts other R2 alerts than the evaluation"
            )
    return problems


def contract_versions() -> dict[str, int]:
    """The contract version of every record kind (the run record's ``contracts``)."""
    return {kind: contract.version for kind, contract in RECORD_CONTRACTS.items()}


def contract_of(record: object) -> RecordContract:
    """The contract named by a record's ``record`` field; ``RecordContractError`` if none."""
    kind = record.get("record") if isinstance(record, Mapping) else None
    if not isinstance(kind, str) or kind not in RECORD_CONTRACTS:
        raise RecordContractError(f"Unknown confirmatory record kind {kind!r}")
    return RECORD_CONTRACTS[kind]


def record_problems(record: object, context: ConfirmatoryContext) -> list[str]:
    """Every problem of ``record``: its contract, then (on a valid structure) the rules that need
    the Test partition and the evaluation declaration."""
    contract = contract_of(record)
    problems = contract.problems(record)
    if problems or not isinstance(record, Mapping):
        return problems
    return [
        problem
        for check in CONTEXT_CHECKS.get(contract.kind, ())
        for problem in check(record, context)
    ]


def validate_record(record: object, context: ConfirmatoryContext, kind: str | None = None) -> None:
    """Validate ``record`` against its contract and the context (and check that it is of
    ``kind``, if given)."""
    contract = contract_of(record)
    if kind is not None and contract.kind != kind:
        raise RecordContractError(f"Expected a {kind} record, got a {contract.kind} record")
    problems = record_problems(record, context)
    if problems:
        raise RecordContractError(
            f"{contract.kind} record (version {contract.version}) does not satisfy its "
            "contract:\n- " + "\n- ".join(problems)
        )


def dump_record(record: Mapping[str, Any], context: ConfirmatoryContext) -> str:
    """The record as JSON text, after validating it."""
    validate_record(record, context)
    return dumps(record)


def read_record(
    path: Path, context: ConfirmatoryContext, kind: str | None = None
) -> dict[str, Any]:
    """Read and validate one record file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise FileAccessError(f"{path}: cannot be read ({error})") from error
    record = loads(text, str(path))
    try:
        validate_record(record, context, kind)
    except RecordContractError as error:
        raise RecordContractError(f"{path}: {error}") from error
    logger.debug("Read %s record %s", record["record"], path)
    return record


@dataclass(frozen=True)
class ConfirmatoryRecords:
    """The record set of the confirmatory run, for ``development.run_output.RunDirectory`` and
    ``load_run``."""

    context: ConfirmatoryContext

    @property
    def contracts(self) -> Mapping[str, RecordContract]:
        return RECORD_CONTRACTS

    @property
    def run_kind(self) -> str:
        return RUN_KIND

    @property
    def end_kind(self) -> str:
        return END_KIND

    def dump(self, record: Mapping[str, Any]) -> str:
        return dump_record(record, self.context)

    def read(self, path: Path, kind: str | None = None) -> dict[str, Any]:
        return read_record(path, self.context, kind)


def reference_markdown() -> str:
    """Reference documentation of every confirmatory record contract, generated from them."""

    def escape(text: str) -> str:
        return text.replace("|", "\\|")

    lines = [
        "# Confirmatory records: contract reference",
        "",
        "Generated from `review_capacity.confirmatory.records`; not edited by hand. Every record",
        "carries `record`, `schema_version` and the run `identity`; paths are relative to the",
        "project root unless stated.",
        "",
        "## JSON records",
    ]
    for contract in RECORD_CONTRACTS.values():
        lines += [
            "",
            f"### `{contract.kind}` (version {contract.version})",
            "",
            escape(contract.meaning),
            "",
            "| Field | Type | Null | Meaning |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| `{path}` | {escape(kind)} | {'yes' if nullable else 'no'} | {escape(meaning)} |"
            for path, kind, nullable, meaning in contract.reference_rows()
        ]
    return "\n".join(lines) + "\n"
