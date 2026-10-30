"""Versioned contracts of the confirmatory Test run's JSON records (Issue #195).

Every record is one JSON object with the envelope of the development records (``record``,
``schema_version``, ``identity``) and kind-specific fields, validated by the same declarative
primitives (``development.schema``): no unknown or missing field, no duplicate key, no NaN or
infinite number. A value that cannot be estimated is ``null`` together with an explicit reason, and
a record that rests on replicates carries the number of undefined replicates.

The kinds, in the order a run produces them: ``test_run`` (written first: the identity, the frozen
inputs and the Test partition), ``test_phase`` (one per attempt of the scoring or the evaluation
phase), ``test_scores`` (one per model, from the single scoring pass), ``test_inference`` (H1, H2,
H3, Holm and the sensitivities) and ``test_run_end`` (closes the run). The outcome, RQ2, RQ3 and
false-negative kinds are defined with the code that produces them.

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
    COUNT,
    DEVICE,
    GIT_COMMIT,
    POSITIVE,
    PROJECT_FILE,
    RUN_FILE,
    RUN_ID,
    SECONDS,
    SEED,
    SHA256,
    STUDY_CONFIGURATION,
    TIMESTAMP,
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
    evaluation declaration."""

    test: PartitionSpec
    declaration: EvaluationDeclaration

    @classmethod
    def from_specs(
        cls, dataset: DatasetSpec, declaration: EvaluationDeclaration
    ) -> ConfirmatoryContext:
        return cls(dataset.partition("test"), declaration)


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

RECORD_CONTRACTS: dict[str, RecordContract] = {
    contract.kind: contract
    for contract in (TEST_RUN, TEST_PHASE, TEST_SCORES, TEST_INFERENCE, TEST_RUN_END)
}

# The rules that need the Test partition or the evaluation declaration, by record kind.
CONTEXT_CHECKS: dict[str, tuple[ContextCheck, ...]] = {
    "test_run": (_run_partition, _seed_rule),
    "test_phase": (_seed_rule,),
    "test_scores": (_scores_partition, _seed_rule),
    "test_inference": (_inference_rules, _seed_rule),
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
