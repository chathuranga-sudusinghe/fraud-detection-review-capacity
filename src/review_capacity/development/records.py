"""Versioned contracts of the development pipeline's JSON records (Issue #117; Issue #96).

Every record is one JSON object with an envelope and kind-specific fields:

* ``record`` (its kind) and ``schema_version`` (the version of that kind's contract);
* ``identity``: the run identity, the same in every record of a run: run identifier, code commit
  and source fingerprint, the SHA-256 of the model settings (``config/models.yaml``), of the
  training declaration and of the dataset manifest, and the seeds.

The kinds, in the order a run produces them: ``candidates`` and ``cost_prediction`` (the search
plan, written before any fit), ``run``, ``preprocessing``, ``preflight``, ``fit``,
``checkpoint``, ``trial``, ``configuration``, ``refit``, ``score``, ``evaluation``, ``timing``,
``device_comparison``, ``calibration``, ``cost_analysis``, ``r1_selection`` and ``h3_selection``;
``run_end`` closes the run (Issue #120). Records link to one another and to the
model-run records of ``models.run_record`` by path (relative to the run directory) and SHA-256.
JSON is the pipeline's machine-readable output and has no CSV companion: long tables live inside
the record they belong to (a trial's checkpoint history, an evaluation's per-day values, a
calibrator's reliability bins). ``development.summary`` specifies the human-readable Markdown.

These contracts own the structure of the records and the rules that hold between their fields,
not the values of the study: the search budget, seeds, capacities and other protocol values come
from the training declaration and the protocol, and each calculation is owned by the code that
produces it.

No record admits the sealed Test period. Every record is validated together with the
development partitions of ``config/dataset.yaml`` (``DevelopmentPartitions``, DR-001): a
partition is named V1 or V2 only, and every relative day, day range and row count of a record
must match its partition (Train for fits and preprocessing). A full-V2 (``primary``) range covers
all of V2; a label-latency (``latency``) fitting or selection range starts on the first V2 day
and ends before the last, and its held-out evaluation range ends on the last V2 day.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any

from review_capacity.config import DEVICES, SEED_LIMIT, DatasetSpec, PartitionSpec
from review_capacity.data.processed import stems
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
    RelativePath,
    Text,
    dumps,
    loads,
)
from review_capacity.errors import FileAccessError, RecordContractError
from review_capacity.models.registry import CONFIGURATIONS, Family
from review_capacity.paths import ProjectPaths

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DevelopmentPartitions:
    """The development partitions a record may describe (Train, V1 and V2 of DR-001) and the
    processed development files a run may read, by stem, with their root-relative paths."""

    train: PartitionSpec
    v1: PartitionSpec
    v2: PartitionSpec
    processed: tuple[tuple[str, str], ...]
    latency_fitting_last_day: int | None = None  # protocol: label-latency fit on V2 days 127–147

    @classmethod
    def from_spec(cls, spec: DatasetSpec, paths: ProjectPaths) -> DevelopmentPartitions:
        """From the loaded ``config/dataset.yaml``; a sealed partition is never admitted."""
        processed = tuple(
            (stem, paths.processed_file(stem).relative_to(paths.root).as_posix())
            for stem in stems(spec)
        )
        partitions = cls(
            spec.partition("train"),
            spec.partition("v1"),
            spec.partition("v2"),
            processed,
            spec.label_latency_fitting_last_day,
        )
        if any(p.sealed for p in (partitions.train, partitions.v1, partitions.v2)):
            raise RecordContractError("A development partition is declared sealed")
        return partitions

    def named(self, label: str) -> PartitionSpec:
        """The partition a record names: ``"V1"`` or ``"V2"``."""
        return {"V1": self.v1, "V2": self.v2}[label]

    def latency_ranges(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """The label-latency fitting and held-out day ranges of V2, as (first, last) pairs."""
        last = self.latency_fitting_last_day
        if last is None:
            raise RecordContractError("The dataset declaration has no label-latency range")
        return (self.v2.first_day, last), (last + 1, self.v2.last_day)


type DayCheck = Callable[[Mapping[str, Any], DevelopmentPartitions], list[str]]


def _range_problems(
    where: str, days: Mapping[str, int], partition: PartitionSpec, role: str
) -> list[str]:
    """``role``: ``full`` (the whole partition), ``fitting`` (a label-latency fit: from the first
    day, ending before the last) or ``held_out`` (ending on the last day)."""
    first, last = days["first"], days["last"]
    span = f"{partition.label} (days {partition.first_day}–{partition.last_day})"
    if not partition.first_day <= first <= last <= partition.last_day:
        return [f"{where}: days {first}–{last} lie outside {span}"]
    if role == "full" and (first, last) != (partition.first_day, partition.last_day):
        return [f"{where}: days {first}–{last} must cover all of {span}"]
    if role == "fitting" and (first != partition.first_day or last >= partition.last_day):
        return [f"{where}: a label-latency fit starts on day {partition.first_day} and ends early"]
    if role == "held_out" and (first == partition.first_day or last != partition.last_day):
        return [f"{where}: a held-out range starts late and ends on day {partition.last_day}"]
    whole = (first, last) == (partition.first_day, partition.last_day)
    inside = partition.first_day < first and last < partition.last_day
    if role == "latency" and (whole or inside):
        return [f"{where}: a label-latency range is a fitting or a held-out part of {span}"]
    return []


def _period_role(period: str) -> str:
    return "full" if period == "primary" else "fitting"


# The logistic-regression reference of DR-011 §5: outside the eight configurations, reported
# descriptively under the ranking measures and R3 only.
REFERENCE = "LR"
REFERENCE_MODEL = "logistic_regression"
CONFIGURATION_NAMES: tuple[str, ...] = tuple(c.name for c in CONFIGURATIONS)
ALL_CONFIGURATIONS: tuple[str, ...] = (*CONFIGURATION_NAMES, REFERENCE)
MODEL_OF: dict[str, str] = {c.name: c.model.name for c in CONFIGURATIONS} | {
    REFERENCE: REFERENCE_MODEL
}
FAMILY_OF: dict[str, Family | None] = {c.name: c.family for c in CONFIGURATIONS} | {REFERENCE: None}
WEIGHTING_OF: dict[str, str] = {c.name: c.weighting.value for c in CONFIGURATIONS} | {
    REFERENCE: "u"
}

# DR-015 §2: a tree-based trial stops after ``patience`` consecutive rounds without improvement
# (last round = best + patience); a neural trial stops at the (patience + 1)-th consecutive epoch
# without improvement (last epoch = best + patience + 1).
STOP_OFFSET = {"round": 0, "epoch": 1}
UNIT_OF_FAMILY = {Family.TREE: "round", Family.NEURAL: "epoch"}

# Measures an evaluation record may carry; producers own their calculation (protocol Outcomes).
METRICS: dict[str, str] = {
    "recall": "Share of frauds selected: Recall@K under R3 and R4; recall of R0–R2",
    "precision": "Share of selected transactions that are fraud (Precision@K under R3)",
    "f1": "F1 of a threshold policy (R0–R2), uncapped",
    "frauds_per_1000_reviews": "Frauds selected per 1,000 selected transactions",
    "oracle_ceiling": "Highest attainable recall under the budget: sum min(K, F_d) / sum F_d",
    "normalised_recall": "Pooled normalised recall: sum TP_d / sum min(K, F_d) (DR-002)",
    "random_review_recall": "Expected recall of random review with the same number of reviews",
    "lift": "Recall divided by the random-review recall",
    "utilisation": "Reviews divided by K × days (R3, R4)",
    "capacity_to_prevalence": "Capacity share c divided by fraud prevalence",
    "amount_weighted_recall": "Share of fraud TransactionAmt selected (exposure proxy)",
    "fraud_value_exposed": "Sum of TransactionAmt of frauds not selected (exposure proxy)",
    "average_precision": "Average Precision (PR-AUC), score ties intact",
    "roc_auc": "ROC-AUC, score ties intact",
    "total_overflow": "Sum over days of max(A_d − K, 0) (DR-009)",
    "total_unused": "Sum over days of max(K − A_d, 0) (DR-009)",
    "mean_load_ratio": "Mean over days of A_d / K (DR-009)",
    "days_above_budget": "Days with A_d > K",
    "days_below_budget": "Days with A_d < K",
    "zero_alert_days": "Days with A_d = 0",
    "scenario_cost": "Scenario cost: reviews + rho × missed frauds, in review-cost units (R3, R4)",
}

# Value types used across the contracts.
SHA256 = Text(r"[0-9a-f]{64}", "SHA-256 hexadecimal digest")
GIT_COMMIT = Text(r"[0-9a-f]{40}", "Git commit (40 hexadecimal digits)")
RUN_ID = Text(r"\d{4}-\d{2}-\d{2}T\d{6}Z", "UTC start time YYYY-MM-DDTHHMMSSZ")
TIMESTAMP = Text(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z", "UTC timestamp (ISO 8601)")
SEED = Integer(0, SEED_LIMIT - 1)
COUNT = Integer(minimum=0)
POSITIVE = Integer(minimum=1)
SHARE = Number(0, 1)
SECONDS = Number(minimum=0)
MIB = Number(minimum=0)
MILLISECONDS = Number(minimum=0)
CONFIGURATION = Choice(ALL_CONFIGURATIONS)
STUDY_CONFIGURATION = Choice(CONFIGURATION_NAMES)
DEVICE = Choice(DEVICES)
PARTITION = Choice(("V1", "V2"))
MODEL_ROLE = Choice(("selected", "seed_refit", "same_seed_repeat", "cpu_refit"))
PERIOD = Choice(("primary", "latency"))
STATUS = Choice(("completed", "failed"))


def _file(base: str) -> Object:
    return Object(
        (
            Field("path", RelativePath(base), "File path"),
            Field("sha256", SHA256, "SHA-256 of the file's bytes"),
        )
    )


RUN_FILE = _file("the run directory")
PROJECT_FILE = _file("the project root")
DAYS = Object(
    (
        Field("first", POSITIVE, "First relative day"),
        Field("last", POSITIVE, "Last relative day"),
    )
)

IDENTITY = Object(
    (
        Field("run_id", RUN_ID, "Run identifier: the UTC start time naming the run directory"),
        Field(
            "run_type",
            Choice(("development", "search_plan")),
            "Run type (the outputs/ subdirectory)",
        ),
        Field("code_commit", GIT_COMMIT, "Git commit of the project root", nullable=True),
        Field(
            "uncommitted_changes",
            Boolean(),
            "Whether the working tree had uncommitted changes",
            nullable=True,
        ),
        Field("source_sha256", SHA256, "Fingerprint of the running package's source files"),
        Field("settings_sha256", SHA256, "SHA-256 of config/models.yaml as loaded"),
        Field("declaration_sha256", SHA256, "SHA-256 of the training declaration as loaded"),
        Field("dataset_manifest_sha256", SHA256, "SHA-256 of the dataset manifest"),
        Field(
            "seeds",
            Object(
                (
                    Field("main", SEED, "Main training seed of every trial and main result"),
                    Field(
                        "additional",
                        ListOf(SEED, unique=True),
                        "Additional training seeds of the seed refits",
                    ),
                    Field("sampler", SEED, "Seed of the candidate sampler"),
                    Field("preprocessing", SEED, "Seed of the neural preprocessing noise"),
                )
            ),
            "Seeds of the run, from the training declaration",
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
    return RecordContract(kind, version, meaning, Object((*envelope, *fields)), tuple(checks))


def _required(kind: str, record: Mapping[str, Any], names: tuple[str, ...], when: str) -> list[str]:
    return [f"{kind}.{name}: required when {when}" for name in names if record[name] is None]


def _absent(kind: str, record: Mapping[str, Any], names: tuple[str, ...], when: str) -> list[str]:
    given = [name for name in names if record[name] is not None]
    return [f"{kind}.{name}: must be null when {when}" for name in given]


def _absent_list(kind: str, name: str, items: list[Any], when: str) -> list[str]:
    return [f"{kind}.{name}: must be empty when {when}"] if items else []


def _status(kind: str, done: str, required: tuple[str, ...]) -> Check:
    """``status == done``: no error and every ``required`` field given; otherwise an error."""

    def check(record: Mapping[str, Any]) -> list[str]:
        if record["status"] == done:
            when = f"status is {done}"
            return _absent(kind, record, ("error",), when) + _required(kind, record, required, when)
        return _required(kind, record, ("error",), f"status is {record['status']}")

    return check


def _seed_rule(kind: str, role_field: str | None = None) -> Check:
    """The training seed of each kind of fit (DR-015 §4): the main seed, except for the seed
    refits; ``role_field`` names the field holding the kind of fit (None: a search trial)."""

    def check(record: Mapping[str, Any]) -> list[str]:
        seeds = record["identity"]["seeds"]
        role = "search_trial" if role_field is None else record[role_field]
        seed = record["seed"]
        if role == "seed_refit":
            ok = seed in seeds["additional"]
            expected = f"one of the additional seeds {seeds['additional']}"
        else:
            ok = seed == seeds["main"]
            expected = f"the main seed {seeds['main']}"
        return [] if ok else [f"{kind}.seed: {seed} must be {expected} for {role}"]

    return check


def _cpu_refit(kind: str, role_field: str) -> Check:
    """The CPU refits of DR-013 §7 are CatBoost refits on the CPU."""

    def check(record: Mapping[str, Any]) -> list[str]:
        if record[role_field] != "cpu_refit":
            return []
        problems: list[str] = []
        if MODEL_OF[record["configuration"]] != "catboost":
            problems.append(f"{kind}.configuration: a CPU refit is a CatBoost refit")
        if record["device"] != "cpu":
            problems.append(f"{kind}.device: a CPU refit runs on the cpu")
        return problems

    return check


# --- search plan: candidates and predicted cost --------------------------------------------------

MODELS_SEARCHED: tuple[str, ...] = tuple(dict.fromkeys(MODEL_OF.values()))  # DR-011, then LR
SEARCH_MODEL = Choice(MODELS_SEARCHED)
CANDIDATE = Object(
    (
        Field("trial", POSITIVE, "Trial number"),
        Field("reference", Boolean(), "Trial 1 is the reference configuration (no random draw)"),
        Field("parameters", MapOf(Number()), "The searched parameters, in declaration order"),
    )
)


def _candidates_rules(record: Mapping[str, Any]) -> list[str]:
    trials = record["trials"]
    problems: list[str] = []
    if [t["trial"] for t in trials] != list(range(1, len(trials) + 1)):
        problems.append("candidates.trials: numbered 1, 2, ... in trial order")
    if [t["reference"] for t in trials] != [True] + [False] * (len(trials) - 1):
        problems.append("candidates.trials: only trial 1 is the reference configuration")
    names = {tuple(t["parameters"]) for t in trials}
    if len(names) != 1:
        problems.append("candidates.trials: every trial declares the same parameters in order")
    stream = record["stream"]
    if stream["spawn_key"] != [stream["index"]]:
        problems.append("candidates.stream: the spawn key is the stream index")
    if stream["index"] != MODELS_SEARCHED.index(record["model"]):
        problems.append(
            f"candidates.stream.index: {record['model']} uses stream "
            f"{MODELS_SEARCHED.index(record['model'])} (DR-011 order, then LR as child (4,))"
        )
    if record["sampler_seed"] != record["identity"]["seeds"]["sampler"]:
        problems.append("candidates.sampler_seed: must be the declared sampler seed")
    problems += [
        f"candidates.configurations: {name} does not belong to {record['model']}"
        for name in record["configurations"]
        if MODEL_OF[name] != record["model"]
    ]
    return problems


CANDIDATES = _contract(
    "candidates",
    1,
    "One model's ordered candidate list (DR-015 §1), generated before any fit; the model's "
    "configurations (unweighted and balanced) share it.",
    (
        Field("model", SEARCH_MODEL, "Model"),
        Field(
            "configurations",
            ListOf(CONFIGURATION, min_items=1, unique=True),
            "Configurations that use this list",
        ),
        Field(
            "stream",
            Object(
                (
                    Field("index", COUNT, "Child stream of the sampler (DR-015 §1, §8)"),
                    Field("spawn_key", ListOf(COUNT, min_items=1), "SeedSequence spawn key"),
                )
            ),
            "The model's child stream",
        ),
        Field("sampler_seed", SEED, "Seed of the sampler's parent SeedSequence"),
        Field(
            "sampler",
            Object(
                (
                    Field("generator", Text(), "Generator constructor"),
                    Field("bit_generator", Text(), "Bit generator of the stream"),
                    Field("numpy", Text(), "NumPy version that drew the values"),
                    Field("python", Text(), "Python version"),
                )
            ),
            "The software that generated the values; bit-for-bit regeneration in another "
            "environment is not assumed (DR-015 §6)",
        ),
        Field("trials", ListOf(CANDIDATE, min_items=2), "The candidates, in trial order"),
    ),
    _candidates_rules,
)

COST_ESTIMATE = Object(
    (
        Field("model", SEARCH_MODEL, "Model"),
        Field("estimated", Boolean(), "Whether DR-015 §7 gives a cost model for it"),
        Field("basis", Text(), "How the estimate is made, or why there is none"),
        Field("epochs_low", POSITIVE, "Epochs per trial at the earliest stop", nullable=True),
        Field("epochs_high", POSITIVE, "Epochs per trial at the maximum", nullable=True),
        Field(
            "candidate_epoch_minutes",
            ListOf(Number(0, exclusive_minimum=True)),
            "Minutes per epoch of each candidate, with the V1 evaluation (empty: not estimated)",
        ),
        Field("search_hours_low", SECONDS, "Search hours at the earliest stops", nullable=True),
        Field("search_hours_high", SECONDS, "Search hours at the maximum", nullable=True),
        Field(
            "refit_hours",
            SECONDS,
            "Hours of the refits if the costliest candidate of the list is selected, at the "
            "maximum length (the selected candidate is unknown before the search)",
            nullable=True,
        ),
    )
)
COST_NUMBERS = ("epochs_low", "epochs_high", "search_hours_low", "search_hours_high")


def _cost_rules(record: Mapping[str, Any]) -> list[str]:
    estimates = record["estimates"]
    problems: list[str] = []
    if [e["model"] for e in estimates] != list(MODELS_SEARCHED):
        problems.append("cost_prediction.estimates: one per model, in DR-011 order then LR")
    if set(record["candidates"]) != set(MODELS_SEARCHED):
        problems.append("cost_prediction.candidates: one candidate record per model")
    for e in estimates:
        where = f"cost_prediction.estimates[{e['model']}]"
        given = (*COST_NUMBERS, "refit_hours")
        if e["estimated"]:
            problems += [f"{where}.{n}: required when estimated" for n in given if e[n] is None]
            if not e["candidate_epoch_minutes"]:
                problems.append(f"{where}.candidate_epoch_minutes: required when estimated")
            elif None not in (e["epochs_low"], e["epochs_high"]) and not (
                e["epochs_low"] <= e["epochs_high"]
            ):
                problems.append(f"{where}: epochs_low must not exceed epochs_high")
            elif None not in (e["search_hours_low"], e["search_hours_high"]) and not (
                e["search_hours_low"] <= e["search_hours_high"]
            ):
                problems.append(f"{where}: search_hours_low must not exceed search_hours_high")
        else:
            problems += [
                f"{where}.{n}: must be null when not estimated" for n in given if e[n] is not None
            ]
            if e["candidate_epoch_minutes"]:
                problems.append(f"{where}.candidate_epoch_minutes: empty when not estimated")
    totals = record["totals"]
    for name in ("search_hours_low", "search_hours_high", "refit_hours"):
        values = [e[name] for e in estimates if e["estimated"]]
        if None in values:
            continue  # reported above as required when estimated
        expected = sum(values)
        if totals[name] != expected:
            problems.append(f"cost_prediction.totals.{name}: must be the sum {expected!r}")
    return problems


COST_PREDICTION = _contract(
    "cost_prediction",
    2,
    "The predicted cost of the actual candidate lists (DR-015 §7), recorded before any fit. An "
    "estimate under the stated basis, not a bound; no list is redrawn and no candidate rejected "
    "for its cost.",
    (
        Field(
            "candidates",
            MapOf(RUN_FILE, keys=SEARCH_MODEL),
            "The candidate record of each model that the prediction is computed from",
        ),
        Field("estimates", ListOf(COST_ESTIMATE, min_items=1), "One estimate per model"),
        Field(
            "totals",
            Object(
                (
                    Field("search_hours_low", SECONDS, "Search hours at the earliest stops"),
                    Field("search_hours_high", SECONDS, "Search hours at the maximum"),
                    Field("refit_hours", SECONDS, "Refit hours, as in the estimates"),
                )
            ),
            "Sums over the estimated models",
        ),
        Field(
            "decision",
            Choice(("pending_approval",)),
            "Whether the predicted cost is feasible is decided explicitly before any run "
            "(DR-015 §7)",
        ),
    ),
    _cost_rules,
)

# --- run -----------------------------------------------------------------------------------------

CODE = Object(
    (
        Field(
            "package",
            Object(
                (
                    Field("name", Text(), "Package name"),
                    Field("version", Text(), "Package version"),
                    Field("location", Text(), "Directory of the running package"),
                    Field("source_sha256", SHA256, "Fingerprint of its source files"),
                )
            ),
            "The running package",
        ),
        Field("git_commit", GIT_COMMIT, "Git commit of the project root", nullable=True),
        Field("git_uncommitted_changes", Boolean(), "Uncommitted changes", nullable=True),
        Field("python", Text(), "Python version"),
        Field("packages", MapOf(Text(), nullable_values=True), "Recorded package versions"),
    )
)


def _run_identity(record: Mapping[str, Any]) -> list[str]:
    identity, code = record["identity"], record["code"]
    pairs = (
        ("code_commit", code["git_commit"]),
        ("uncommitted_changes", code["git_uncommitted_changes"]),
        ("source_sha256", code["package"]["source_sha256"]),
        ("settings_sha256", record["settings"]["sha256"]),
        ("declaration_sha256", record["declaration"]["sha256"]),
        ("dataset_manifest_sha256", record["dataset"]["manifest"]["sha256"]),
    )
    return [
        f"run.identity.{name}: {identity[name]!r} differs from the run's own value {value!r}"
        for name, value in pairs
        if identity[name] != value
    ]


def _run_inputs(record: Mapping[str, Any], partitions: DevelopmentPartitions) -> list[str]:
    """A run reads processed development files only, each at its own path."""
    allowed = dict(partitions.processed)
    problems: list[str] = []
    for stem, entry in record["dataset"]["processed"].items():
        if stem not in allowed:
            problems.append(f"run.dataset.processed: {stem!r} is not a development file")
        elif entry["path"] != allowed[stem]:
            problems.append(f"run.dataset.processed.{stem}.path: must be {allowed[stem]}")
    return problems


def _run_contracts(record: Mapping[str, Any]) -> list[str]:
    expected = contract_versions()
    actual = record["contracts"]
    if actual != expected:
        return [f"run.contracts: {actual} differs from the current versions {expected}"]
    return []


RUN = _contract(
    "run",
    2,
    "The run's full provenance, written first; every other record repeats its identity.",
    (
        Field("started", TIMESTAMP, "When the run started"),
        Field("code", CODE, "Code provenance (provenance.code_provenance)"),
        Field("environment", MapOf(AnyJSON()), "Run environment (models.device)"),
        Field("settings", PROJECT_FILE, "The loaded config/models.yaml"),
        Field("declaration", PROJECT_FILE, "The loaded training declaration"),
        Field(
            "plan",
            PROJECT_FILE,
            "The approved search plan's cost prediction (its SHA-256 is the approved one)",
        ),
        Field(
            "dataset",
            Object(
                (
                    Field("manifest", PROJECT_FILE, "The dataset manifest"),
                    Field("processed", MapOf(PROJECT_FILE), "Processed files read, by name"),
                )
            ),
            "Data read by the run",
        ),
        Field(
            "configurations",
            ListOf(CONFIGURATION, min_items=1, unique=True),
            "Configurations run, in run order",
        ),
        Field("contracts", MapOf(POSITIVE), "Contract version of every record kind written"),
    ),
    _run_identity,
    _run_contracts,
)

# --- preprocessing -------------------------------------------------------------------------------


def _preprocessing_counts(record: Mapping[str, Any]) -> list[str]:
    inputs = record["inputs"]
    problems: list[str] = []
    if inputs["total"] != inputs["numeric"] + inputs["categorical"] + inputs["indicators"]:
        problems.append("preprocessing.inputs.total: must equal numeric + categorical + indicators")
    if inputs["indicators"] > inputs["numeric"]:
        problems.append("preprocessing.inputs.indicators: at most one per numeric input")
    if record["seed"] != record["identity"]["seeds"]["preprocessing"]:
        problems.append("preprocessing.seed: must be the declared preprocessing seed")
    return problems


def _train_rows(kind: str) -> DayCheck:
    """A record fitted on Train covers every Train row."""

    def check(record: Mapping[str, Any], partitions: DevelopmentPartitions) -> list[str]:
        rows, expected = record["train"]["rows"], partitions.train.rows
        if rows != expected:
            return [f"{kind}.train.rows: {rows}, not the {expected} Train rows"]
        return []

    return check


PREPROCESSING = _contract(
    "preprocessing",
    2,
    "The shared Train-fitted neural preprocessing artifact (DR-012), reused by every neural fit "
    "and by the logistic-regression reference; the artifact itself stays local.",
    (
        Field("artifact", PROJECT_FILE, "The artifact's manifest file (its bytes' SHA-256)"),
        Field(
            "artifact_sha256",
            SHA256,
            "The artifact's identity: the SHA-256 every neural model and the reference record",
        ),
        Field("seed", SEED, "Seed of the quantile-fitting noise"),
        Field(
            "train",
            Object(
                (
                    Field("rows", POSITIVE, "Train rows fitted on"),
                    Field("transaction_ids_sha256", SHA256, "Ordered Train identity"),
                    Field("file_sha256", SHA256, "SHA-256 of the processed Train file"),
                )
            ),
            "The Train data fitted on",
        ),
        Field(
            "inputs",
            Object(
                (
                    Field("numeric", COUNT, "Quantile-transformed numeric inputs"),
                    Field("categorical", COUNT, "Categorical inputs (vocabularies)"),
                    Field("indicators", COUNT, "Missingness indicators"),
                    Field("total", POSITIVE, "All neural inputs"),
                    Field("order_sha256", SHA256, "SHA-256 of the ordered input names"),
                )
            ),
            "Input layout",
        ),
        Field("settings", MapOf(AnyJSON()), "Fitting settings"),
        Field("software", MapOf(Text()), "Software versions"),
    ),
    _preprocessing_counts,
)

# --- preflight -----------------------------------------------------------------------------------

ATTEMPT = Object(
    (
        Field("phase", Choice(("training", "evaluation")), "What was tried"),
        Field("rows", POSITIVE, "Micro-batch (training) or evaluation batch rows"),
        Field("outcome", Choice(("pass", "out_of_memory", "below_margin")), "Result"),
        Field("peak_used_mib", MIB, "Sampled peak device memory used", nullable=True),
        Field("free_mib", Number(), "Device memory free at the sampled peak", nullable=True),
    )
)


def _preflight_rules(record: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    attempts = record["attempts"]
    training = [a for a in attempts if a["phase"] == "training"]
    sizes = [a["rows"] for a in training]
    if any(b >= a for a, b in pairwise(sizes)):
        problems.append("preflight.attempts: training micro-batches must be tried largest first")
    if any(size > record["effective_batch"] for size in sizes):
        problems.append("preflight.attempts: a micro-batch exceeds the effective batch")
    for i, attempt in enumerate(attempts):
        where = f"preflight.attempts[{i}]"
        measured = attempt["peak_used_mib"] is not None and attempt["free_mib"] is not None
        if attempt["outcome"] != "out_of_memory" and not measured:
            problems.append(f"{where}: memory must be measured for outcome {attempt['outcome']}")
        if measured and (attempt["free_mib"] >= record["margin_mib"]) != (
            attempt["outcome"] == "pass"
        ):
            problems.append(f"{where}: outcome contradicts the free-memory margin")
    passed = [a["rows"] for a in training if a["outcome"] == "pass"]
    if record["status"] == "pass":
        if not passed or record["micro_batch"] != passed[0]:
            problems.append("preflight.micro_batch: must be the largest passing micro-batch")
        if record["model"] == "resnet" and record["micro_batch"] != record["effective_batch"]:
            problems.append("preflight.micro_batch: the ResNet-style MLP runs the effective batch")
        evaluation = [a for a in attempts if a["phase"] == "evaluation"]
        if not evaluation or evaluation[-1]["outcome"] != "pass":
            problems.append("preflight.attempts: the evaluation batch must pass")
        if any(a["rows"] != record["evaluation_batch"] for a in evaluation):
            problems.append("preflight.attempts: evaluation attempts use the evaluation batch")
    else:
        problems += _absent("preflight", record, ("micro_batch",), "status is fail")
    return problems


PREFLIGHT = _contract(
    "preflight",
    1,
    "The pre-search micro-batch preflight of one neural candidate (DR-015 §7).",
    (
        Field("model", Choice(("resnet", "ft_transformer")), "Neural model"),
        Field("trial", POSITIVE, "Candidate (trial) number"),
        Field("candidate_list", RUN_FILE, "The recorded candidate list"),
        Field("effective_batch", POSITIVE, "Train rows per optimiser update"),
        Field("evaluation_batch", POSITIVE, "Evaluation batch rows"),
        Field("margin_mib", MIB, "Required free device memory at the sampled peak"),
        Field("attempts", ListOf(ATTEMPT, min_items=1), "Attempts in the order made"),
        Field("micro_batch", POSITIVE, "Selected micro-batch rows", nullable=True),
        Field("status", Choice(("pass", "fail")), "Outcome; a failure stops execution"),
        Field("error", Text(), "Failure description", nullable=True),
    ),
    _status("preflight", "pass", ("micro_batch",)),
    _preflight_rules,
)

# --- fit -----------------------------------------------------------------------------------------


def _fit_rules(record: Mapping[str, Any]) -> list[str]:
    name, purpose = record["configuration"], record["purpose"]
    family = FAMILY_OF[name]
    train = record["train"]
    problems: list[str] = []
    if WEIGHTING_OF[name] == "u":
        if record["positive_weight"] != 1:
            problems.append("fit.positive_weight: must be 1 for an unweighted configuration")
    elif train["frauds"] == 0 or record["positive_weight"] != (
        (train["rows"] - train["frauds"]) / train["frauds"]
    ):
        problems.append("fit.positive_weight: must be n_neg / n_pos of the Train data")
    if train["frauds"] > train["rows"]:
        problems.append("fit.train.frauds: exceeds the rows")
    neural_inputs = ("preprocessing", "preflight")
    if family is Family.TREE:
        problems += _absent("fit", record, neural_inputs, "the model is tree-based")
    elif family is Family.NEURAL:
        problems += _required("fit", record, neural_inputs, "the model is neural")
    else:
        problems += _required("fit", record, ("preprocessing",), "fitting the reference")
        problems += _absent("fit", record, ("preflight",), "fitting the reference")
        if purpose != "search_trial" or record["early_stopping"]:
            problems.append("fit: the reference is fitted to convergence in its search only")
    if purpose == "search_trial":
        problems += _absent("fit", record, ("length",), "fitting a search trial")
        if family is not None and not record["early_stopping"]:
            problems.append("fit.early_stopping: every search trial stops early on V1")
    else:
        problems += _required("fit", record, ("length",), f"the purpose is {purpose}")
        if record["early_stopping"]:
            problems.append("fit.early_stopping: refits run at a fixed length")
        if (
            record["status"] == "completed"
            and record["fitted_iterations"] is not None
            and record["fitted_iterations"] != record["length"]
        ):
            problems.append("fit.fitted_iterations: a refit grows exactly its length")
    if purpose == "same_seed_repeat" and not record["fresh_process"]:
        problems.append("fit.fresh_process: a same-seed repeat runs in a fresh process")
    return problems


FIT = _contract(
    "fit",
    1,
    "One model fit on Train: a search trial or a fixed-length refit, linked to its model-run "
    "record (models.run_record) and saved model.",
    (
        Field("configuration", CONFIGURATION, "Configuration fitted"),
        Field(
            "purpose",
            Choice(("search_trial", "seed_refit", "same_seed_repeat", "cpu_refit")),
            "Why the model was fitted",
        ),
        Field("trial", POSITIVE, "Search trial (for a refit: the selected trial)"),
        Field("seed", SEED, "Training seed"),
        Field("device", DEVICE, "Training device"),
        Field("fresh_process", Boolean(), "Fitted in a fresh process"),
        Field("early_stopping", Boolean(), "Early stopping on the V1 objective"),
        Field("length", POSITIVE, "Fixed length of a refit (rounds or epochs)", nullable=True),
        Field(
            "fitted_iterations",
            POSITIVE,
            "Trees, epochs or solver iterations kept",
            nullable=True,
        ),
        Field(
            "train",
            Object(
                (
                    Field("rows", POSITIVE, "Train rows"),
                    Field("frauds", COUNT, "Train frauds"),
                    Field("transaction_ids_sha256", SHA256, "Train identity"),
                    Field("file_sha256", SHA256, "SHA-256 of the processed Train file"),
                )
            ),
            "The Train data",
        ),
        Field(
            "positive_weight",
            Number(0, exclusive_minimum=True),
            "Weight of each fraud row (legitimate rows weigh 1)",
        ),
        Field("preprocessing", RUN_FILE, "Preprocessing record used", nullable=True),
        Field("preflight", RUN_FILE, "Preflight record of the candidate", nullable=True),
        Field("model", RUN_FILE, "Saved model", nullable=True),
        Field("model_record", RUN_FILE, "Model-run record (models.run_record)", nullable=True),
        Field("fit_seconds", SECONDS, "Wall-clock fit time", nullable=True),
        Field("status", STATUS, "Outcome"),
        Field("error", Text(), "Failure description", nullable=True),
    ),
    _status("fit", "completed", ("fitted_iterations", "model", "model_record", "fit_seconds")),
    _fit_rules,
    _seed_rule("fit", "purpose"),
    _cpu_refit("fit", "purpose"),
)

# --- checkpoint ----------------------------------------------------------------------------------


def _checkpoint_unit(record: Mapping[str, Any]) -> list[str]:
    family = FAMILY_OF[record["configuration"]]
    if family is None:
        return ["checkpoint.configuration: the reference has no checkpoints"]
    if record["unit"] != UNIT_OF_FAMILY[family]:
        return [f"checkpoint.unit: a {family.value} model checkpoints per {UNIT_OF_FAMILY[family]}"]
    return []


CHECKPOINT = _contract(
    "checkpoint",
    1,
    "The retained (earliest best) checkpoint of a search trial, saved as a model.",
    (
        Field("configuration", CONFIGURATION, "Configuration"),
        Field("trial", POSITIVE, "Search trial"),
        Field("unit", Choice(("round", "epoch")), "Checkpoint unit"),
        Field("index", POSITIVE, "Round or epoch of the checkpoint"),
        Field("objective", SHARE, "V1 objective at the checkpoint"),
        Field("model", RUN_FILE, "Saved model at the checkpoint"),
        Field("model_record", RUN_FILE, "Model-run record (models.run_record)"),
    ),
    _checkpoint_unit,
)

# --- trial ---------------------------------------------------------------------------------------

STOPPING_REASON = Choice(("early_stopping", "maximum", "converged"))
CHECKPOINT_ROW = Object(
    (
        Field("checkpoint", POSITIVE, "Round or epoch"),
        Field("objective", SHARE, "V1 objective after it"),
        Field("improved", Boolean(), "Strictly above every earlier checkpoint"),
        Field("best_checkpoint", POSITIVE, "Earliest best checkpoint so far"),
        Field("seconds", SECONDS, "Elapsed trial time at the checkpoint"),
    )
)


def _history(record: Mapping[str, Any]) -> list[str]:
    """The V1 objective after every round or epoch: numbered 1, 2, ...; strict improvement and
    earliest-best retention (DR-015 §5); a completed trial's history ends at its last checkpoint
    and agrees with its retained checkpoint."""
    history = record["checkpoints"]
    if record["unit"] == "convergence":
        return _absent_list("trial", "checkpoints", history, "the reference converges")
    if [row["checkpoint"] for row in history] != list(range(1, len(history) + 1)):
        return ["trial.checkpoints: numbered 1, 2, ... in order"]
    problems: list[str] = []
    best_value, best = -1.0, 0
    patience = record["patience"]
    offset = STOP_OFFSET[record["unit"]]
    for row in history:
        if best and patience is not None and row["checkpoint"] > best + patience + offset:
            problems.append(
                f"trial.checkpoints[{row['checkpoint'] - 1}]: training stops when patience runs "
                f"out after checkpoint {best}"
            )
            break
        improved = row["objective"] > best_value
        if improved:
            best_value, best = row["objective"], row["checkpoint"]
        if row["improved"] != improved or row["best_checkpoint"] != best:
            problems.append(f"trial.checkpoints[{row['checkpoint'] - 1}]: improvement is strict")
    if record["status"] == "completed" and record["stopped_at"] is not None:
        if len(history) != record["stopped_at"]:
            problems.append("trial.checkpoints: one entry per round or epoch trained")
        elif (best, best_value) != (record["best_checkpoint"], record["best_objective"]):
            problems.append("trial.checkpoints: the history's best differs from the trial's result")
    return problems


def _trial_rules(record: Mapping[str, Any]) -> list[str]:
    family = FAMILY_OF[record["configuration"]]
    unit = "convergence" if family is None else UNIT_OF_FAMILY[family]
    if record["unit"] != unit:
        return [f"trial.unit: must be {unit} for {record['configuration']}"]
    stepwise = ("patience", "maximum", "best_checkpoint", "stopped_at", "checkpoint")
    if unit == "convergence":
        problems = _absent("trial", record, stepwise, "the reference converges")
        if record["stopping_reason"] not in (None, "converged"):
            problems.append("trial.stopping_reason: the reference stops on convergence")
        return problems
    problems = _required("trial", record, ("patience", "maximum"), f"training per {unit}")
    if problems or record["status"] != "completed":
        return problems
    problems = _required("trial", record, stepwise, "a stepwise trial completed")
    if problems:
        return problems
    best, stopped, maximum = record["best_checkpoint"], record["stopped_at"], record["maximum"]
    patience_end = best + record["patience"] + STOP_OFFSET[unit]
    if not best <= stopped <= maximum:
        problems.append("trial: best checkpoint ≤ stopped_at ≤ maximum must hold")
    reason = record["stopping_reason"]
    if reason == "early_stopping" and stopped != patience_end:
        problems.append(f"trial.stopped_at: early stopping after {best} ends at {patience_end}")
    elif reason == "maximum" and not (stopped == maximum < patience_end):
        problems.append("trial.stopping_reason: maximum applies only when patience did not run out")
    elif reason == "converged":
        problems.append("trial.stopping_reason: only the reference stops on convergence")
    return problems


TRIAL = _contract(
    "trial",
    1,
    "One search trial: its candidate, V1 objective, retained checkpoint and stopping (DR-015).",
    (
        Field("configuration", CONFIGURATION, "Configuration"),
        Field("trial", POSITIVE, "Trial number (1 = reference configuration)"),
        Field("candidate_list", RUN_FILE, "The recorded candidate list"),
        Field("hyperparameters", MapOf(AnyJSON()), "The candidate's searched values"),
        Field("seed", SEED, "Training seed"),
        Field("device", DEVICE, "Training device"),
        Field("unit", Choice(("round", "epoch", "convergence")), "Training unit"),
        Field("objective_k", POSITIVE, "Daily review count of the V1 objective (K_V1)"),
        Field("patience", COUNT, "Early-stopping patience", nullable=True),
        Field("maximum", POSITIVE, "Maximum rounds or epochs", nullable=True),
        Field("best_checkpoint", POSITIVE, "Retained (earliest best) checkpoint", nullable=True),
        Field("best_objective", SHARE, "V1 objective of the trial's result", nullable=True),
        Field("stopped_at", POSITIVE, "Last round or epoch trained", nullable=True),
        Field("stopping_reason", STOPPING_REASON, "Why training stopped", nullable=True),
        Field("runtime_seconds", SECONDS, "Wall-clock time of the trial", nullable=True),
        Field(
            "checkpoints",
            ListOf(CHECKPOINT_ROW),
            "V1 objective after every round or epoch (empty for the reference)",
        ),
        Field("fit", RUN_FILE, "Fit record", nullable=True),
        Field("checkpoint", RUN_FILE, "Retained checkpoint record", nullable=True),
        Field("status", STATUS, "Outcome"),
        Field("error", Text(), "Failure description", nullable=True),
    ),
    _status("trial", "completed", ("best_objective", "stopping_reason", "runtime_seconds", "fit")),
    _trial_rules,
    _history,
    _seed_rule("trial"),
)

# --- configuration -------------------------------------------------------------------------------

TRIAL_SUMMARY = Object(
    (
        Field("trial", POSITIVE, "Trial number"),
        Field("objective", SHARE, "V1 objective of the trial's result"),
        Field("stopping_reason", STOPPING_REASON, "Why training stopped"),
        Field("record", RUN_FILE, "Trial record"),
    )
)


def _selection(record: Mapping[str, Any]) -> list[str]:
    trials = record["trials"]
    problems: list[str] = []
    if [t["trial"] for t in trials] != list(range(1, len(trials) + 1)):
        problems.append("configuration.trials: numbered 1, 2, ... in trial order")
    best = max(t["objective"] for t in trials)
    tied = [t["trial"] for t in trials if t["objective"] == best]
    selected = next(t for t in trials if t["objective"] == best)
    if record["selected_trial"] != selected["trial"]:
        problems.append(f"configuration.selected_trial: the rule selects trial {selected['trial']}")
    if record["selected_objective"] != best:
        problems.append("configuration.selected_objective: differs from the best trial objective")
    if record["tied_trials"] != tied:
        problems.append(f"configuration.tied_trials: the trials with the best value are {tied}")
    if record["selected_stopping_reason"] != selected["stopping_reason"]:
        problems.append("configuration.selected_stopping_reason: differs from the selected trial")
    reference = FAMILY_OF[record["configuration"]] is None
    stepwise = ("selected_length", "selected_checkpoint")
    if reference:
        problems += _absent("configuration", record, stepwise, "selecting the reference")
    else:
        problems += _required("configuration", record, stepwise, "selecting a configuration")
    return problems


CONFIGURATION_RECORD = _contract(
    "configuration",
    1,
    "The search result of one configuration: every trial and the selected trial (DR-015 §1, §5).",
    (
        Field("configuration", CONFIGURATION, "Configuration"),
        Field("candidate_list", RUN_FILE, "The recorded candidate list"),
        Field("trials", ListOf(TRIAL_SUMMARY, min_items=1), "Every completed trial, in order"),
        Field("selected_trial", POSITIVE, "Highest V1 objective, lowest trial number on ties"),
        Field("selected_objective", SHARE, "V1 objective of the selected trial"),
        Field(
            "tied_trials",
            ListOf(POSITIVE, min_items=1, unique=True),
            "Trials whose objective equals the best exactly",
        ),
        Field("selected_hyperparameters", MapOf(AnyJSON()), "The selected candidate's values"),
        Field("selected_length", POSITIVE, "Retained rounds or epochs", nullable=True),
        Field("selected_stopping_reason", STOPPING_REASON, "Stopping of the selected trial"),
        Field("selected_fit", RUN_FILE, "Fit record of the selected trial"),
        Field("selected_checkpoint", RUN_FILE, "Retained checkpoint record", nullable=True),
    ),
    _selection,
)

# --- refit ---------------------------------------------------------------------------------------

COMPARISON = Object(
    (
        Field("rows", POSITIVE, "V2 rows compared"),
        Field("identical", Boolean(), "Whether every score is bitwise identical"),
        Field("differing_rows", COUNT, "Rows whose scores differ"),
        Field("max_abs_difference", Number(minimum=0), "Largest absolute score difference"),
    )
)


def _refit_rules(record: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    comparison = record["comparison"]
    compared = record["kind"] in ("same_seed_repeat", "cpu_refit")
    if record["status"] == "completed":
        if compared:
            problems += _required("refit", record, ("comparison",), f"a {record['kind']} ends")
        else:
            problems += _absent("refit", record, ("comparison",), "the seed differs")
    if comparison is not None:
        identical = comparison["identical"]
        if identical != (comparison["differing_rows"] == 0):
            problems.append("refit.comparison: identical exactly when no row differs")
        if identical != (comparison["max_abs_difference"] == 0):
            problems.append("refit.comparison: identical exactly when the largest difference is 0")
        if comparison["differing_rows"] > comparison["rows"]:
            problems.append("refit.comparison.differing_rows: exceeds the rows compared")
    return problems


REFIT = _contract(
    "refit",
    1,
    "A fixed-length refit of a selected setting: an additional seed, a same-seed repeat or a "
    "CatBoost CPU refit, evaluated on V2 only (DR-015 §4; DR-013 §6–§7).",
    (
        Field("configuration", STUDY_CONFIGURATION, "Configuration (the reference has none)"),
        Field("kind", Choice(("seed_refit", "same_seed_repeat", "cpu_refit")), "Kind of refit"),
        Field("seed", SEED, "Training seed"),
        Field("device", DEVICE, "Training device"),
        Field("selected_trial", POSITIVE, "The selected trial refitted"),
        Field("length", POSITIVE, "Fixed length (the retained rounds or epochs)"),
        Field("fit", RUN_FILE, "Fit record", nullable=True),
        Field("score", RUN_FILE, "V2 score record of the refit", nullable=True),
        Field("reference_score", RUN_FILE, "V2 score record of the selected main-seed model"),
        Field("comparison", COMPARISON, "Score comparison with the selected model", nullable=True),
        Field("status", STATUS, "Outcome"),
        Field("error", Text(), "Failure description", nullable=True),
    ),
    _status("refit", "completed", ("fit", "score")),
    _refit_rules,
    _seed_rule("refit", "kind"),
    _cpu_refit("refit", "kind"),
)

# --- device comparison ---------------------------------------------------------------------------

BOOTSTRAP = Object(
    (
        Field("capacity", Number(0, 1, exclusive_minimum=True), "Capacity share c of the recall"),
        Field("k", POSITIVE, "Fixed daily review budget K"),
        Field("entity_key", Choice(("fine",)), "Pseudo-entity key of the frauds (DR-006)"),
        Field("estimate", Number(-1, 1), "Recall(CPU) − recall(GPU) on V2"),
        Field("lower", Number(-1, 1), "Lower end of the percentile interval"),
        Field("upper", Number(-1, 1), "Upper end of the percentile interval"),
        Field("level", Number(0, 1, exclusive_minimum=True), "Interval level"),
        Field("replicates", POSITIVE, "Bootstrap replicates B"),
        Field("undefined_replicates", COUNT, "Replicates with zero total weight (left out)"),
        Field("seed", SEED, "Seed of the bootstrap generator"),
        Field("days", POSITIVE, "Relative days drawn"),
        Field("entities", POSITIVE, "Distinct fraud pseudo-entities drawn"),
        Field("frauds", POSITIVE, "Frauds in the partition"),
    )
)
DEVICE_CAPACITY = Object(
    (
        Field("capacity", Number(0, 1, exclusive_minimum=True), "Capacity share c"),
        Field("k", POSITIVE, "Fixed daily review budget K"),
        Field(
            "differences",
            MapOf(Number(), keys=Choice(tuple(METRICS))),
            "CPU − GPU point differences of the R3 measures",
        ),
    )
)


def _device_rules(record: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if MODEL_OF[record["configuration"]] != "catboost":
        problems.append("device_comparison.configuration: only CatBoost has a CPU–GPU pair")
    bootstrap = record["bootstrap"]
    if bootstrap["lower"] > bootstrap["upper"]:
        problems.append("device_comparison.bootstrap: lower is above upper")
    if bootstrap["undefined_replicates"] >= bootstrap["replicates"]:
        problems.append("device_comparison.bootstrap: no replicate is defined")
    capacities = [(c["capacity"], c["k"]) for c in record["capacities"]]
    if len(set(capacities)) != len(capacities):
        problems.append("device_comparison.capacities: a capacity is listed once")
    if (bootstrap["capacity"], bootstrap["k"]) not in capacities:
        problems.append("device_comparison.bootstrap: its capacity is one of the capacities")
    return problems


def _device_days(record: Mapping[str, Any], partitions: DevelopmentPartitions) -> list[str]:
    v2 = partitions.v2
    days = v2.last_day - v2.first_day + 1
    if record["bootstrap"]["days"] != days:
        return [f"device_comparison.bootstrap.days: V2 has {days} relative days"]
    return []


DEVICE_COMPARISON = _contract(
    "device_comparison",
    1,
    "The supportive CPU–GPU comparison of one CatBoost configuration on V2: R3 point differences "
    "(CPU − GPU) at each capacity and the paired two-way bootstrap interval of the Recall@K "
    "difference (DR-013 §7; DR-006). Descriptive only.",
    (
        Field("configuration", STUDY_CONFIGURATION, "Configuration (CatBoost)"),
        Field("seed", SEED, "Training seed of both models (the main seed)"),
        Field("gpu_score", RUN_FILE, "V2 score record of the selected GPU model"),
        Field("cpu_score", RUN_FILE, "V2 score record of the CPU refit"),
        Field("refit", RUN_FILE, "Refit record of the CPU refit"),
        Field("capacities", ListOf(DEVICE_CAPACITY, min_items=1), "R3 differences by capacity"),
        Field("bootstrap", BOOTSTRAP, "Paired bootstrap of the Recall@K difference"),
    ),
    _device_rules,
    _seed_rule("device_comparison"),
)

# --- score ---------------------------------------------------------------------------------------


def _score_rules(record: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if record["first_day"] > record["last_day"]:
        problems.append("score: first_day after last_day")
    if record["min_score"] > record["max_score"]:
        problems.append("score: min_score above max_score")
    if record["distinct_scores"] > record["rows"]:
        problems.append("score.distinct_scores: exceeds the rows")
    if (record["distinct_scores"] == 1) != (record["min_score"] == record["max_score"]):
        problems.append("score: one distinct score exactly when min_score equals max_score")
    if record["configuration"] == REFERENCE and record["model_role"] != "selected":
        problems.append("score.model_role: the reference has only its selected model")
    return problems


def _score_days(record: Mapping[str, Any], partitions: DevelopmentPartitions) -> list[str]:
    """Scores cover their whole partition."""
    partition = partitions.named(record["partition"])
    days = {"first": record["first_day"], "last": record["last_day"]}
    problems = _range_problems("score", days, partition, "full")
    if record["rows"] != partition.rows:
        rows = record["rows"]
        problems.append(f"score.rows: {rows}, not the {partition.rows} of {partition.label}")
    return problems


SCORE = _contract(
    "score",
    1,
    "The identity of one model's scores on a development partition; the row-level scores stay "
    "local and are never committed.",
    (
        Field("configuration", CONFIGURATION, "Configuration"),
        Field("model_role", MODEL_ROLE, "Which model scored"),
        Field("seed", SEED, "Training seed of the model"),
        Field("fit", RUN_FILE, "Fit record of the model"),
        Field("partition", PARTITION, "Development partition scored"),
        Field("first_day", POSITIVE, "First relative day scored"),
        Field("last_day", POSITIVE, "Last relative day scored"),
        Field("rows", POSITIVE, "Transactions scored"),
        Field("file", RUN_FILE, "Row-level score file (local only)"),
        Field("format", Choice(("parquet",)), "File format"),
        Field("columns", ListOf(Text(), min_items=1, unique=True), "Columns of the file"),
        Field("score_dtype", Choice(("float32", "float64")), "Stored score type"),
        Field("device", DEVICE, "Scoring device"),
        Field("batch_rows", POSITIVE, "Evaluation batch rows (neural models)", nullable=True),
        Field("min_score", SHARE, "Smallest score"),
        Field("max_score", SHARE, "Largest score"),
        Field("distinct_scores", POSITIVE, "Number of distinct scores"),
        Field("seconds", SECONDS, "Wall-clock scoring time"),
    ),
    _score_rules,
    _seed_rule("score", "model_role"),
)

# --- evaluation ----------------------------------------------------------------------------------

COUNTS = Object(
    (
        Field("days", POSITIVE, "Relative days evaluated"),
        Field("transactions", POSITIVE, "Transactions"),
        Field("frauds", COUNT, "Frauds"),
        Field("selected", COUNT, "Alerts (R0–R2) or reviews (R3, R4)"),
        Field("true_positives", COUNT, "Selected frauds"),
        Field("false_positives", COUNT, "Selected legitimate transactions"),
        Field("false_negatives", COUNT, "Frauds not selected"),
    )
)
CUTOFF_TIES = Object(
    (
        Field("days", COUNT, "Days whose review cutoff splits a group of equal scores"),
        Field("group_sizes", ListOf(Integer(minimum=2)), "Size of each such group"),
    )
)
POLICY_FIELDS = ("capacity", "k", "threshold", "cost_ratio", "calibrator", "calibration")
# For each policy: the fields it requires; every other field of POLICY_FIELDS must be null. R0
# and R1 have one threshold each and may be compared with any budget, so capacity and k are
# optional for them (both or neither).
POLICY_REQUIRES: dict[str, tuple[str, ...]] = {
    "ranking": (),
    "R0": ("threshold",),
    "R1": ("threshold",),
    "R2": ("threshold", "capacity", "k"),
    "R3": ("capacity", "k"),
    "R4": ("capacity", "k", "threshold", "cost_ratio", "calibrator", "calibration"),
}
POLICY_OPTIONAL: dict[str, tuple[str, ...]] = {"R0": ("capacity", "k"), "R1": ("capacity", "k")}


def _evaluation_rules(record: Mapping[str, Any]) -> list[str]:
    policy = record["policy"]
    when = f"the policy is {policy}"
    required = POLICY_REQUIRES[policy]
    optional = POLICY_OPTIONAL.get(policy, ())
    rest = tuple(n for n in POLICY_FIELDS if n not in required and n not in optional)
    problems = _required("evaluation", record, required, when)
    problems += _absent("evaluation", record, rest, when)
    if (record["capacity"] is None) != (record["k"] is None):
        problems.append("evaluation: capacity and k are given together")
    capped = policy in ("R3", "R4")
    if policy == "ranking":
        problems += _absent("evaluation", record, ("counts", "cutoff_ties"), when)
    else:
        problems += _required("evaluation", record, ("counts",), when)
        if capped:
            problems += _required("evaluation", record, ("cutoff_ties",), when)
        else:
            problems += _absent("evaluation", record, ("cutoff_ties",), when)
    ratio, threshold = record["cost_ratio"], record["threshold"]
    if (
        policy == "R4"
        and ratio is not None
        and threshold is not None
        and (ratio <= 1 or threshold != 1 / ratio)
    ):
        problems.append("evaluation.threshold: R4 reviews p_hat ≥ p* = 1/rho with rho > 1")
    counts = record["counts"]
    if counts is not None:
        if counts["true_positives"] + counts["false_negatives"] != counts["frauds"]:
            problems.append("evaluation.counts: true_positives + false_negatives ≠ frauds")
        if counts["true_positives"] + counts["false_positives"] != counts["selected"]:
            problems.append("evaluation.counts: true_positives + false_positives ≠ selected")
        if max(counts["selected"], counts["frauds"]) > counts["transactions"]:
            problems.append("evaluation.counts: selected and frauds are at most the transactions")
        if capped and record["k"] is not None and counts["selected"] > record["k"] * counts["days"]:
            problems.append("evaluation.counts.selected: a capped policy reviews at most K a day")
        if counts["days"] != record["days"]["last"] - record["days"]["first"] + 1:
            problems.append("evaluation.counts.days: must count the evaluated days")
    if record["configuration"] == REFERENCE and policy not in ("ranking", "R3"):
        problems.append("evaluation.policy: the reference is evaluated under ranking and R3 only")
    primary = record["period"] == "primary"
    if record["model_role"] != "selected" and (
        policy != "R3" or record["partition"] != "V2" or not primary
    ):
        problems.append("evaluation: refits are evaluated under R3 on all of V2 only")
    if not primary and record["partition"] != "V2":
        problems.append("evaluation.period: the label-latency analysis uses V2 only")
    return problems


DAY_ROW = Object(
    (
        Field("day", POSITIVE, "Relative day d"),
        Field("transactions", POSITIVE, "Transactions N_d"),
        Field("frauds", COUNT, "Frauds F_d"),
        Field("selected", COUNT, "Alerts A_d (R0–R2) or reviews (R3, R4)"),
        Field("true_positives", COUNT, "Selected frauds TP_d"),
        Field("oracle_reviewable", COUNT, "min(K, F_d)", nullable=True),
        Field("overflow", COUNT, "max(A_d − K, 0)", nullable=True),
        Field("unused", COUNT, "max(K − A_d, 0)", nullable=True),
        Field("load_ratio", Number(minimum=0), "A_d / K", nullable=True),
        Field(
            "cutoff_tied_group",
            Integer(minimum=2),
            "Size of the group of equal scores split by the day's cutoff",
            nullable=True,
        ),
    )
)
DAY_TOTALS = ("transactions", "frauds", "selected", "true_positives")


def _daily(record: Mapping[str, Any]) -> list[str]:
    """One entry per evaluated day, in day order, whose totals are the pooled counts; the
    workload measures follow the budget K (DR-009)."""
    daily, k = record["daily"], record["k"]
    if record["policy"] == "ranking":
        return _absent_list("evaluation", "daily", daily, "the policy is ranking")
    days = record["days"]
    problems: list[str] = []
    if [row["day"] for row in daily] != list(range(days["first"], days["last"] + 1)):
        problems.append("evaluation.daily: one entry per evaluated day, in day order")
    counts = record["counts"]
    if counts is not None:
        problems += [
            f"evaluation.daily: {name} do not sum to counts.{name}"
            for name in DAY_TOTALS
            if sum(row[name] for row in daily) != counts[name]
        ]
    capped = record["policy"] in ("R3", "R4")
    for i, row in enumerate(daily):
        where = f"evaluation.daily[{i}]"
        selected, frauds = row["selected"], row["frauds"]
        if row["true_positives"] > min(selected, frauds):
            problems.append(f"{where}: true_positives exceed the selected or the frauds")
        if max(selected, frauds) > row["transactions"]:
            problems.append(f"{where}: selected and frauds are at most the transactions")
        workload = ("oracle_reviewable", "overflow", "unused", "load_ratio")
        if k is None:
            problems += _absent(where, row, workload, "there is no budget k")
            continue
        expected = {
            "oracle_reviewable": min(k, frauds),
            "overflow": max(selected - k, 0),
            "unused": max(k - selected, 0),
            "load_ratio": selected / k,
        }
        problems += [
            f"{where}.{name}: must be {value!r}"
            for name, value in expected.items()
            if row[name] != value
        ]
        if capped and selected > k:
            problems.append(f"{where}: a capped policy reviews at most k")
        if record["policy"] == "R3" and selected != min(k, row["transactions"]):
            problems.append(f"{where}: R3 reviews min(K, N_d) transactions")
    ties = record["cutoff_ties"]
    if ties is not None:
        groups = [row["cutoff_tied_group"] for row in daily if row["cutoff_tied_group"] is not None]
        if (ties["days"], ties["group_sizes"]) != (len(groups), groups):
            problems.append("evaluation.cutoff_ties: must summarise the daily tied groups")
    elif any(row["cutoff_tied_group"] is not None for row in daily):
        problems.append("evaluation.daily: cutoff ties are recorded for capped policies only")
    return problems


def _evaluation_days(record: Mapping[str, Any], partitions: DevelopmentPartitions) -> list[str]:
    role = "full" if record["period"] == "primary" else "latency"
    partition = partitions.named(record["partition"])
    problems = _range_problems("evaluation.days", record["days"], partition, role)
    counts = record["counts"]
    if role == "full" and counts is not None and counts["transactions"] != partition.rows:
        problems.append(
            f"evaluation.counts.transactions: {counts['transactions']}, not the "
            f"{partition.rows} of {partition.label}"
        )
    return problems


EVALUATION = _contract(
    "evaluation",
    1,
    "The development outcomes of one model's scores under one policy setting (protocol "
    "Outcomes), with its per-day values.",
    (
        Field("configuration", CONFIGURATION, "Configuration"),
        Field("model_role", MODEL_ROLE, "Which model was evaluated"),
        Field("seed", SEED, "Training seed of the model"),
        Field("score", RUN_FILE, "Score record evaluated"),
        Field("partition", PARTITION, "Development partition"),
        Field("period", PERIOD, "Whole partition (primary) or a label-latency part of V2"),
        Field("days", DAYS, "Relative days evaluated"),
        Field("policy", Choice(tuple(POLICY_REQUIRES)), "Decision policy, or whole ranking"),
        Field("capacity", Number(0, 1, exclusive_minimum=True), "Capacity share c", nullable=True),
        Field("k", POSITIVE, "Fixed daily review budget K", nullable=True),
        Field("threshold", Number(), "Score or probability threshold", nullable=True),
        Field("cost_ratio", Number(), "Miss-to-review cost ratio rho (R4)", nullable=True),
        Field("calibrator", Choice(("platt", "isotonic")), "Calibrator (R4)", nullable=True),
        Field("calibration", RUN_FILE, "Calibration record (R4)", nullable=True),
        Field("counts", COUNTS, "Pooled counts", nullable=True),
        Field(
            "metrics",
            MapOf(Number(), keys=Choice(tuple(METRICS)), nullable_values=True),
            "Outcome measures (null: not estimable)",
        ),
        Field("daily", ListOf(DAY_ROW), "Per-day values (empty for the whole ranking)"),
        Field("cutoff_ties", CUTOFF_TIES, "Score ties crossing the daily cutoff", nullable=True),
    ),
    _evaluation_rules,
    _daily,
    _seed_rule("evaluation", "model_role"),
)

# --- timing --------------------------------------------------------------------------------------

LATENCY = Object(
    (
        Field("median", MILLISECONDS, "Median single-transaction latency (ms)"),
        Field("p95", MILLISECONDS, "95th percentile (ms)"),
        Field("p99", MILLISECONDS, "99th percentile (ms)"),
    )
)


TIMING_PROCEDURE_VERSION = 1  # the computational-cost procedure of DR-015 §9
TIMING_FIT_DEVICE = {"lightgbm": "cpu", "catboost": "gpu", "resnet": "gpu", "ft_transformer": "gpu"}
HOST_MEMORY_METHODS = ("clear_refs", "sampled")
STUDY_ATTENTION_BACKEND = "efficient"  # config/models.yaml gpu_attention_backend (DR-013 §4)


def _timing_device(measurement: str, configuration: str, role: str | None) -> str:
    """The study device of a measurement (DR-013 §1, §5, §7): fits and searches run on the
    model's training device (the CatBoost CPU refit on the cpu), scoring and latency on the
    cpu for the tree-based models and the reference and on the gpu for the neural models."""
    if measurement in ("fit", "search"):
        if role == "cpu_refit":
            return "cpu"
        return TIMING_FIT_DEVICE.get(MODEL_OF[configuration], "cpu")
    return "gpu" if FAMILY_OF[configuration] is Family.NEURAL else "cpu"


def _is_count(value: object, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_seconds(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value >= 0


def _timing_settings(record: Mapping[str, Any]) -> list[str]:
    """The settings every timing carries, and those of its measurement (DR-015 §9)."""
    settings, measurement = record["settings"], record["measurement"]
    problems: list[str] = []

    def need(name: str, ok: bool, what: str) -> None:
        if name not in settings:
            problems.append(f"timing.settings.{name}: required when measuring {measurement}")
        elif not ok:
            problems.append(f"timing.settings.{name}: must be {what}")

    value = settings.get
    need("procedure_version", value("procedure_version") == TIMING_PROCEDURE_VERSION, "1")
    need("session", isinstance(value("session"), dict), "the session conditions (context)")
    problem_list = value("memory_problems")
    need(
        "memory_problems",
        isinstance(problem_list, list) and all(isinstance(p, str) and p for p in problem_list),
        "a list of the memory components that could not be measured",
    )
    if measurement == "fit":
        completed, retained = value("completed_length"), value("retained_length")
        need("completed_length", _is_count(completed, 1), "a positive integer")
        need("retained_length", _is_count(retained, 1), "a positive integer")
        if _is_count(completed, 1) and _is_count(retained, 1) and retained > completed:
            problems.append("timing.settings.retained_length: at most the completed length")
        for name in ("startup_seconds", "preprocessing_seconds", "persistence_seconds"):
            need(name, value(name) is None or _is_seconds(value(name)), "seconds, or null")
        scope = value("startup_scope")
        scoped = scope is None or (isinstance(scope, str) and bool(scope))
        need("startup_scope", scoped, "text, or null")
        status = value("startup_status")
        known = status in (None, "measured", "not_measured")
        need("startup_status", known, "measured, not_measured or null")
        if status == "measured" and not _is_seconds(value("startup_seconds")):
            problems.append("timing.settings.startup_seconds: required when measured")
    elif measurement == "search":
        done, declared = value("trials_completed"), value("trials_declared")
        need("trials_completed", _is_count(done), "a count")
        need("trials_declared", _is_count(declared, 1), "the declared number of trials")
        need("interrupted", isinstance(value("interrupted"), bool), "true or false")
        if _is_count(done) and _is_count(declared, 1) and isinstance(value("interrupted"), bool):
            if value("interrupted") and done >= declared:
                problems.append("timing.settings.interrupted: an interrupted search is partial")
            elif not value("interrupted") and done != declared:
                problems.append(
                    "timing.settings.trials_completed: a search that was not interrupted "
                    "completed every declared trial"
                )
    elif measurement == "scoring":
        passes = value("pass_seconds")
        listed = isinstance(passes, list) and all(_is_seconds(x) for x in passes) and bool(passes)
        need("pass_seconds", listed, "the seconds of every timed pass")
        for name, pick in (("pass_min_seconds", min), ("pass_max_seconds", max)):
            agrees = listed and _is_seconds(value(name)) and value(name) == pick(passes)
            need(name, agrees, f"the {name[5:8]} of settings.pass_seconds")
        if listed and record["repetitions"] is not None:
            if len(passes) != record["repetitions"]:
                problems.append("timing.settings.pass_seconds: one entry per repetition")
            elif record["seconds"] is not None and record["seconds"] != median(passes):
                problems.append("timing.seconds: must be the median of settings.pass_seconds")
    else:
        rounds = value("round_latency_ms")
        listed = (
            isinstance(rounds, list)
            and bool(rounds)
            and all(
                isinstance(r, dict) and LATENCY.problems(r, "round") == [] and _ordered(r)
                for r in rounds
            )
        )
        need("round_latency_ms", listed, "the median, p95 and p99 of every round, in order")
        need("percentile_method", value("percentile_method") == "linear", "linear")
        calls = value("timed_calls")
        need("timed_calls", _is_count(calls, 1), "a positive integer")
        if listed and record["repetitions"] is not None and len(rounds) != record["repetitions"]:
            problems.append("timing.settings.round_latency_ms: one entry per round")
        counted = _is_count(calls, 1) and None not in (record["rows"], record["repetitions"])
        if counted and calls != record["rows"] * record["repetitions"]:
            problems.append("timing.settings.timed_calls: must be rows x repetitions")
    if record["peak_host_mib"] is not None:
        method = settings.get("host_memory_method")
        if method not in HOST_MEMORY_METHODS:
            problems.append(f"timing.settings.host_memory_method: one of {HOST_MEMORY_METHODS}")
        elif method == "sampled" and not _is_count(settings.get("host_sample_interval_ms"), 1):
            problems.append("timing.settings.host_sample_interval_ms: required when sampled")
    if record["peak_gpu_mib"] is not None:
        problems += _gpu_memory_problems(record)
    return problems


def _gpu_memory_problems(record: Mapping[str, Any]) -> list[str]:
    """The sampled GPU memory is stored with its baseline, increase and sampling quality, and
    agrees with the record's sampled maximum."""
    memory = record["settings"].get("gpu_memory")
    if not isinstance(memory, dict):
        return ["timing.settings.gpu_memory: required when peak_gpu_mib is reported"]
    problems: list[str] = []
    maximum, baseline = memory.get("sampled_max_mib"), memory.get("baseline_mib")
    if maximum != record["peak_gpu_mib"]:
        problems.append("timing.settings.gpu_memory.sampled_max_mib: must equal peak_gpu_mib")
    if (
        isinstance(maximum, int | float)
        and isinstance(baseline, int | float)
        and memory.get("increase_above_baseline_mib") != maximum - baseline
    ):
        problems.append("timing.settings.gpu_memory: the increase is the maximum minus baseline")
    problems += [
        f"timing.settings.gpu_memory.{name}: required (a positive integer)"
        for name in ("samples", "interval_ms")
        if not _is_count(memory.get(name), 1)
    ]
    return problems


def _ordered(latency: Mapping[str, Any]) -> bool:
    return bool(latency["median"] <= latency["p95"] <= latency["p99"])


def _timing_rows(record: Mapping[str, Any], partitions: DevelopmentPartitions) -> list[str]:
    """A scoring time covers the whole partition (a latency sample is a sample)."""
    if record["measurement"] != "scoring" or record["partition"] is None:
        return []
    rows = partitions.named(record["partition"]).rows
    if record["rows"] is not None and record["rows"] != rows:
        return [f"timing.rows: scoring {record['partition']} covers all {rows} of its rows"]
    return []


def _timing_rules(record: Mapping[str, Any]) -> list[str]:
    measurement, configuration = record["measurement"], record["configuration"]
    when = f"measuring {measurement}"
    if measurement in ("fit", "search"):
        problems = _required("timing", record, ("seconds",), when)
        problems += _absent("timing", record, ("partition", "latency_ms", "boundary"), when)
    elif measurement == "scoring":
        scoring = (
            "partition",
            "boundary",
            "rows",
            "seconds",
            "rows_per_second",
            "batch_rows",
            "warmup",
            "repetitions",
        )
        problems = _required("timing", record, scoring, when)
        problems += _absent("timing", record, ("latency_ms",), when)
    else:
        latency_fields = ("boundary", "rows", "warmup", "repetitions", "latency_ms")
        problems = _required("timing", record, latency_fields, when)
        latency = record["latency_ms"]
        if latency is not None and not _ordered(latency):
            problems.append("timing.latency_ms: median ≤ p95 ≤ p99 must hold")
        if record["batch_rows"] != 1:
            problems.append("timing.batch_rows: single-transaction latency is a batch of 1")
        problems += _required("timing", record, ("rows_sha256", "row_selection"), when)
    if measurement == "scoring" and None not in (record["rows"], record["seconds"]):
        expected_rate = record["rows"] / record["seconds"] if record["seconds"] > 0 else 0.0
        if record["rows_per_second"] is not None and record["rows_per_second"] != expected_rate:
            problems.append("timing.rows_per_second: must be rows divided by the seconds")
    if measurement == "search":
        problems += _absent("timing", record, ("model_role",), when)
    else:
        problems += _required("timing", record, ("model_role",), when)
    role = record["model_role"]
    if role == "cpu_refit" and MODEL_OF[configuration] != "catboost":
        problems.append("timing.model_role: a CPU refit is a CatBoost refit")
    expected = _timing_device(measurement, configuration, role)
    if record["device"] != expected:
        problems.append(f"timing.device: {measurement} of {configuration} runs on the {expected}")
    device = f"the device is {record['device']}"
    if record["device"] == "cpu":
        problems += _required("timing", record, ("cpu_affinity", "threads"), device)
    elif not record["synchronised"]:
        problems.append("timing.synchronised: GPU work is synchronised before each clock reading")
    if MODEL_OF[configuration] == "ft_transformer":
        problems += _required("timing", record, ("attention_backend",), "timing FT-Transformer")
        if record["attention_backend"] not in (None, STUDY_ATTENTION_BACKEND):
            problems.append(
                f"timing.attention_backend: study timing uses the {STUDY_ATTENTION_BACKEND} "
                "backend (DR-013 §4)"
            )
    else:
        problems += _absent("timing", record, ("attention_backend",), "the model has no attention")
    if (record["peak_gpu_mib"], record["peak_host_mib"]) != (None, None):
        problems += _required("timing", record, ("memory_scope",), "memory is reported")
    return problems + _timing_settings(record)


TIMING = _contract(
    "timing",
    4,
    "One computational-cost measurement on the study device (protocol Outcomes, computational "
    "cost; DR-015 §9).",
    (
        Field("configuration", CONFIGURATION, "Configuration"),
        Field("measurement", Choice(("fit", "search", "scoring", "latency")), "What was timed"),
        Field(
            "model_role",
            Choice(("search_trial", *MODEL_ROLE.values)),
            "Which model was timed: a search trial, the selected model or a refit (null: the "
            "whole search)",
            nullable=True,
        ),
        Field("subject", RUN_FILE, "Record of what was timed (fit or score)", nullable=True),
        Field("partition", PARTITION, "Partition scored", nullable=True),
        Field("device", DEVICE, "Device of the measurement (its study device, DR-013 §1, §5)"),
        Field(
            "boundary",
            Choice(("model_only", "end_to_end")),
            "Model-only inference or end-to-end scoring",
            nullable=True,
        ),
        Field("threads", POSITIVE, "CPU threads", nullable=True),
        Field(
            "cpu_affinity",
            ListOf(Integer(minimum=0), min_items=1, unique=True),
            "Effective CPU affinity of the process (CPU timing)",
            nullable=True,
        ),
        Field("synchronised", Boolean(), "GPU work synchronised before each clock reading"),
        Field("attention_backend", Text(), "Attention backend (FT-Transformer)", nullable=True),
        Field("row_selection", Text(), "Rule selecting the latency rows", nullable=True),
        Field("rows_sha256", SHA256, "Identity of the timed rows (latency)", nullable=True),
        Field("batch_rows", POSITIVE, "Batch rows", nullable=True),
        Field("warmup", COUNT, "Untimed warm-up calls", nullable=True),
        Field("repetitions", POSITIVE, "Timed passes (scoring) or rounds (latency)", nullable=True),
        Field(
            "rows", POSITIVE, "Rows per pass (scoring) or distinct rows (latency)", nullable=True
        ),
        Field(
            "seconds", SECONDS, "Wall-clock seconds (scoring: median of the passes)", nullable=True
        ),
        Field("rows_per_second", Number(minimum=0), "Throughput", nullable=True),
        Field(
            "latency_ms", LATENCY, "Single-transaction latency over all timed calls", nullable=True
        ),
        Field(
            "peak_gpu_mib",
            MIB,
            "Sampled whole-device GPU memory maximum (not a true peak)",
            nullable=True,
        ),
        Field(
            "peak_host_mib",
            MIB,
            "Peak resident memory of the process during the phase",
            nullable=True,
        ),
        Field("memory_scope", Text(), "What the peak memory covers", nullable=True),
        Field("settings", MapOf(AnyJSON()), "Further settings of the timing procedure"),
    ),
    _timing_rules,
)

# --- calibration ---------------------------------------------------------------------------------

PLATT_FIELDS = ("clipping", "targets", "parameters", "solver")
RELIABILITY_BIN = Object(
    (
        Field("bin", POSITIVE, "Bin number, by increasing probability"),
        Field("min_probability", SHARE, "Smallest calibrated probability"),
        Field("max_probability", SHARE, "Largest calibrated probability"),
        Field("mean_probability", SHARE, "Mean calibrated probability"),
        Field("transactions", POSITIVE, "Transactions"),
        Field("frauds", COUNT, "Frauds"),
        Field("observed_rate", SHARE, "frauds / transactions"),
    )
)
DIAGNOSTIC = Object(
    (
        Field(
            "rows_used",
            Choice(("fitting", "held_out")),
            "The calibrator's fitting rows (in-sample) or the held-out V2 days after them",
        ),
        Field("days", DAYS, "Relative days of the diagnostic rows"),
        Field("rows", POSITIVE, "Transactions"),
        Field("frauds", COUNT, "Frauds"),
        Field("brier", SHARE, "Brier score"),
        Field("log_loss", Number(minimum=0), "Epsilon-clipped log loss"),
        Field("metric_clipped", COUNT, "Probabilities clipped for the metrics"),
        Field("intercept", Number(), "Calibration intercept (null: not estimable)", nullable=True),
        Field("slope", Number(), "Calibration slope (null: not estimable)", nullable=True),
        Field(
            "out_of_range",
            Object(
                (
                    Field("below", COUNT, "Scores below the fitting range"),
                    Field("above", COUNT, "Scores above the fitting range"),
                )
            ),
            "Scores outside the fitting range",
        ),
        Field(
            "reliability",
            ListOf(RELIABILITY_BIN, min_items=1),
            "Equal-count reliability-diagram bins (identical probabilities share a bin)",
        ),
    )
)


def _diagnostics(record: Mapping[str, Any]) -> list[str]:
    """A completed calibrator has one in-sample diagnostic on its fitting days and, when fitted
    for the label-latency analysis, one held-out diagnostic on the days after them."""
    diagnostics = record["diagnostics"]
    expected = ["fitting"] if record["period"] == "primary" else ["fitting", "held_out"]
    problems: list[str] = []
    if record["status"] == "completed" and [d["rows_used"] for d in diagnostics] != expected:
        problems.append(f"calibration.diagnostics: expected {expected}, in that order")
    fitting_days = record["fitting_days"]
    for i, diagnostic in enumerate(diagnostics):
        where = f"calibration.diagnostics[{i}]"
        days = diagnostic["days"]
        if diagnostic["frauds"] > diagnostic["rows"]:
            problems.append(f"{where}.frauds: exceed the rows")
        if diagnostic["rows_used"] == "fitting":
            counted = (diagnostic["rows"], diagnostic["frauds"])
            fitted = counted == (record["rows"], record["frauds"])
            if days != fitting_days or not fitted:
                problems.append(f"{where}: in-sample diagnostics use exactly the fitting rows")
        elif days["first"] != fitting_days["last"] + 1:
            problems.append(f"{where}.days: held-out days start after the fitting days")
        problems += _reliability(diagnostic, f"{where}.reliability")
    return problems


def _reliability(diagnostic: Mapping[str, Any], where: str) -> list[str]:
    bins = diagnostic["reliability"]
    if [b["bin"] for b in bins] != list(range(1, len(bins) + 1)):
        return [f"{where}: bins numbered 1, 2, ... in order"]
    problems: list[str] = []
    for b in bins:
        if not b["min_probability"] <= b["mean_probability"] <= b["max_probability"]:
            problems.append(f"{where}, bin {b['bin']}: min ≤ mean ≤ max must hold")
        if b["frauds"] > b["transactions"]:
            problems.append(f"{where}, bin {b['bin']}: frauds exceed transactions")
        elif b["observed_rate"] != b["frauds"] / b["transactions"]:
            problems.append(f"{where}, bin {b['bin']}: observed_rate ≠ frauds/transactions")
    if any(a["max_probability"] >= b["min_probability"] for a, b in pairwise(bins)):
        problems.append(f"{where}: equal probabilities share one bin; bins do not overlap")
    for name in ("transactions", "frauds"):
        total = "rows" if name == "transactions" else "frauds"
        if sum(b[name] for b in bins) != diagnostic[total]:
            problems.append(f"{where}: bin {name} do not sum to the diagnostic's {total}")
    return problems


def _calibration_rules(record: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if record["rows"] != record["frauds"] + record["legitimate"]:
        problems.append("calibration.rows: must equal frauds + legitimate")
    if record["fitting_days"]["first"] > record["fitting_days"]["last"]:
        problems.append("calibration.fitting_days: first after last")
    platt = record["method"] == "platt"
    method = f"the method is {record['method']}"
    if platt:
        problems += _absent("calibration", record, ("knots",), method)
    else:
        problems += _absent("calibration", record, PLATT_FIELDS, method)
    if record["status"] != "completed":
        return problems
    needed = PLATT_FIELDS if platt else ("knots",)
    problems += _required("calibration", record, needed, f"{method} and completed")
    if record["frauds"] == 0 or record["legitimate"] == 0:
        problems.append("calibration: both classes are required")
    if record["reload_verified"] is not True:
        problems.append("calibration.reload_verified: predictions are verified after reloading")
    if platt and not problems:
        targets = record["targets"]
        if targets["positive"] != (record["frauds"] + 1) / (record["frauds"] + 2):
            problems.append("calibration.targets.positive: must be (N_pos + 1)/(N_pos + 2)")
        if targets["negative"] != 1 / (record["legitimate"] + 2):
            problems.append("calibration.targets.negative: must be 1/(N_neg + 2)")
        if not record["parameters"]["slope"] > 0:
            problems.append("calibration.parameters.slope: kappa_1 must be positive")
        if not record["solver"]["converged"]:
            problems.append("calibration.solver.converged: a completed fit converged")
    return problems


CALIBRATION = _contract(
    "calibration",
    1,
    "One calibrator of a configuration's selected main-seed model, fitted on V2 scores (DR-007 "
    "§8; protocol Calibration).",
    (
        Field("configuration", STUDY_CONFIGURATION, "Configuration (the reference has none)"),
        Field("method", Choice(("platt", "isotonic")), "Calibration method"),
        Field("period", PERIOD, "Full V2 (primary) or label-latency fitting days"),
        Field("score", RUN_FILE, "V2 score record fitted on"),
        Field("fitting_days", DAYS, "Relative days of the fitting rows"),
        Field("rows", POSITIVE, "Fitting rows"),
        Field("frauds", COUNT, "Fraud rows N_pos"),
        Field("legitimate", COUNT, "Legitimate rows N_neg"),
        Field(
            "clipping",
            Object(
                (
                    Field("epsilon", Number(0, exclusive_minimum=True), "Clipping constant"),
                    Field("clipped_low", COUNT, "Scores raised to epsilon"),
                    Field("clipped_high", COUNT, "Scores lowered to 1 − epsilon"),
                )
            ),
            "Input clipping (Platt)",
            nullable=True,
        ),
        Field(
            "targets",
            Object(
                (
                    Field("positive", SHARE, "Smoothed fraud target (N_pos + 1)/(N_pos + 2)"),
                    Field("negative", SHARE, "Smoothed legitimate target 1/(N_neg + 2)"),
                )
            ),
            "Platt's smoothed targets",
            nullable=True,
        ),
        Field(
            "parameters",
            Object(
                (
                    Field("intercept", Number(), "Intercept kappa_0"),
                    Field("slope", Number(), "Slope kappa_1"),
                )
            ),
            "Platt parameters on the log-odds",
            nullable=True,
        ),
        Field(
            "solver",
            Object(
                (
                    Field("name", Text(), "Solver"),
                    Field("settings", MapOf(AnyJSON()), "Initialisation, tolerance, limits"),
                    Field("converged", Boolean(), "Convergence"),
                    Field("iterations", COUNT, "Iterations used"),
                )
            ),
            "Platt solver",
            nullable=True,
        ),
        Field("knots", POSITIVE, "Isotonic knots (stored in the artifact)", nullable=True),
        Field("artifact", RUN_FILE, "Saved calibrator", nullable=True),
        Field("software", MapOf(Text()), "Software versions"),
        Field("reload_verified", Boolean(), "Predictions equal after reloading", nullable=True),
        Field(
            "diagnostics",
            ListOf(DIAGNOSTIC),
            "Diagnostics on the fitting rows and, for a label-latency calibrator, on the "
            "held-out V2 days",
        ),
        Field("status", STATUS, "Outcome; a failure stops the analysis for an issue"),
        Field("error", Text(), "Failure description", nullable=True),
    ),
    _status("calibration", "completed", ("artifact", "reload_verified")),
    _calibration_rules,
    _diagnostics,
)


def _calibration_days(record: Mapping[str, Any], partitions: DevelopmentPartitions) -> list[str]:
    v2 = partitions.v2
    problems = _range_problems(
        "calibration.fitting_days", record["fitting_days"], v2, _period_role(record["period"])
    )
    if record["period"] == "primary" and record["rows"] != v2.rows:
        problems.append(f"calibration.rows: a full-V2 calibrator fits all {v2.rows} V2 rows")
    diagnostics = record["diagnostics"]
    covered = sum(d["rows"] for d in diagnostics)
    if record["period"] == "latency" and len(diagnostics) == 2 and covered != v2.rows:
        problems.append(f"calibration.diagnostics: fitting and held-out rows cover {v2.rows}")
    for i, diagnostic in enumerate(record["diagnostics"]):
        if diagnostic["rows_used"] == "held_out":
            where = f"calibration.diagnostics[{i}].days"
            problems += _range_problems(where, diagnostic["days"], v2, "held_out")
    return problems


# --- cost analysis -------------------------------------------------------------------------------

GRID_ROW = Object(
    (
        Field("ratio", Number(1, exclusive_minimum=True), "Miss-to-review cost ratio rho"),
        Field("threshold", Number(0, 1, exclusive_minimum=True), "R4's threshold p* = 1/rho"),
        Field("reviews", COUNT, "R4 reviews N_R4"),
        Field("unused", COUNT, "Unused reviews u: R3's reviews that R4 declines"),
        Field("frauds_forgone", COUNT, "Frauds forgone f: frauds among the unused reviews"),
        Field("days_with_unused", COUNT, "Days on which R4 reviews fewer than K"),
        Field("utilisation", SHARE, "N_R4 / (K × days)"),
        Field("declined_fraud_rate", SHARE, "f / u (null: u = 0)", nullable=True),
        Field(
            "declined_mean_probability",
            SHARE,
            "Mean calibrated probability of the unused reviews (null: u = 0)",
            nullable=True,
        ),
        Field("cost_r3", Number(minimum=0), "J_R3 = N_R3 + rho FN_R3"),
        Field("cost_r4", Number(minimum=0), "J_R4 = N_R4 + rho FN_R4"),
        Field("cost_difference", Number(), "ΔJ = J_R4 − J_R3 = rho f − u"),
        Field("relative_difference", Number(), "ΔJ / J_R3"),
        Field("cost_no_review", Number(minimum=0), "No-review reference J = rho F"),
        Field(
            "fixed_status",
            Choice(("finite", "infinite", "identical")),
            "Fixed-policy break-even: finite, infinite (f = 0 < u) or identical (u = 0)",
        ),
        Field("fixed_ratio", Number(minimum=0), "rho_fixed = u / f", nullable=True),
        Field("fraud_value_exposed_r3", Number(minimum=0), "Amount of frauds R3 does not review"),
        Field("fraud_value_exposed_r4", Number(minimum=0), "Amount of frauds R4 does not review"),
        Field(
            "fraud_value_forgone", Number(minimum=0), "Amount of frauds among the unused reviews"
        ),
        Field("forgone_per_unused", Number(minimum=0), "Forgone amount per unused review", True),
        Field("amount_weighted_recall_r3", SHARE, "Amount-weighted recall of R3", nullable=True),
        Field("amount_weighted_recall_r4", SHARE, "Amount-weighted recall of R4", nullable=True),
    )
)
BREAK_EVEN_INTERVAL = Object(
    (
        Field("start", Number(1), "Start of an interval of ratios with ΔJ < 0 (closed; open at 1)"),
        Field("end", Number(1, 1000), "End of the interval"),
        Field("end_closed", Boolean(), "True when the end is the range end 1000 (included)"),
    )
)
CASES = ("coincidence", "crossing", "none", "right_censored")
# DR-007 §2 and §1: the ratio grid of the tabulated RQ3 results and the C_FP sensitivities
# phi = C_FP / C_R; the continuous range is (1, RQ3_RATIO_MAX].
RQ3_RATIO_GRID: tuple[int, ...] = (2, 5, 10, 20, 50, 100, 200, 500, 1000)
RQ3_FRICTION: tuple[int, ...] = (1, 5)
RQ3_RATIO_MAX = 1000


def _grid_problems(record: Mapping[str, Any]) -> list[str]:
    queue, grid = record["queue"], record["grid"]
    problems: list[str] = []
    ratios = [row["ratio"] for row in grid]
    if ratios != [float(r) for r in RQ3_RATIO_GRID]:
        problems.append(f"cost_analysis.grid: the ratios must be {list(RQ3_RATIO_GRID)}, in order")
    if queue["captured"] > queue["frauds"]:
        problems.append("cost_analysis.queue: captured frauds exceed the frauds")
    misses = queue["frauds"] - queue["captured"]
    for i, row in enumerate(grid):
        where = f"cost_analysis.grid[{i}]"
        if row["reviews"] + row["unused"] != queue["reviews"]:
            problems.append(f"{where}: reviews + unused ≠ the R3 reviews")
        if row["frauds_forgone"] > min(row["unused"], queue["captured"]):
            problems.append(f"{where}.frauds_forgone: exceeds the unused reviews or R3's frauds")
        if abs(row["threshold"] - 1 / row["ratio"]) > 1e-15:
            problems.append(f"{where}.threshold: must be 1/ratio")
        none = row["unused"] == 0
        evaluated_days = record["days"]["last"] - record["days"]["first"] + 1
        if abs(row["utilisation"] - row["reviews"] / (record["k"] * evaluated_days)) > 1e-12:
            problems.append(f"{where}.utilisation: must be reviews / (k × days)")
        rate, mean = row["declined_fraud_rate"], row["declined_mean_probability"]
        if (
            rate is not None
            and not none
            and abs(rate - row["frauds_forgone"] / row["unused"]) > 1e-12
        ):
            problems.append(f"{where}.declined_fraud_rate: must be frauds_forgone / unused")
        if mean is not None and mean > row["threshold"] + 1e-12:
            problems.append(f"{where}.declined_mean_probability: the unused reviews lie below p*")
        problems.extend(
            f"{where}.{name}: null exactly when there are no unused reviews"
            for name in ("declined_fraud_rate", "declined_mean_probability", "forgone_per_unused")
            if (row[name] is None) != none
        )
        expected = {
            "cost_r3": queue["reviews"] + row["ratio"] * misses,
            "cost_r4": row["reviews"] + row["ratio"] * (misses + row["frauds_forgone"]),
            "cost_difference": row["ratio"] * row["frauds_forgone"] - row["unused"],
            "cost_no_review": row["ratio"] * queue["frauds"],
        }
        for name, value in expected.items():
            if abs(row[name] - value) > 1e-9 * max(1.0, abs(value)):
                problems.append(f"{where}.{name}: must be {value!r}")
        if abs(row["relative_difference"] * row["cost_r3"] - row["cost_difference"]) > 1e-9 * max(
            1.0, abs(row["cost_difference"])
        ):
            problems.append(f"{where}.relative_difference: must be cost_difference / cost_r3")
        status = "identical" if none else "infinite" if row["frauds_forgone"] == 0 else "finite"
        if row["fixed_status"] != status:
            problems.append(f"{where}.fixed_status: must be {status}")
        if (row["fixed_ratio"] is None) != (status != "finite"):
            problems.append(f"{where}.fixed_ratio: given exactly when the status is finite")
        elif (
            row["fixed_ratio"] is not None
            and abs(row["fixed_ratio"] - row["unused"] / row["frauds_forgone"]) > 1e-9
        ):
            problems.append(f"{where}.fixed_ratio: must be unused / frauds_forgone")
        exposed = row["fraud_value_exposed_r3"] + row["fraud_value_forgone"]
        if abs(row["fraud_value_exposed_r4"] - exposed) > 1e-6 * max(1.0, exposed):
            problems.append(f"{where}.fraud_value_exposed_r4: R3's exposure plus the value forgone")
        recall_r3, recall_r4 = row["amount_weighted_recall_r3"], row["amount_weighted_recall_r4"]
        if recall_r3 is not None and recall_r4 is not None and recall_r4 > recall_r3 + 1e-12:
            problems.append(f"{where}.amount_weighted_recall_r4: above R3's")
    reviews = [row["reviews"] for row in grid]
    if reviews != sorted(reviews):
        problems.append("cost_analysis.grid: R4's reviews rise with the ratio")
    return problems


def _break_even_problems(record: Mapping[str, Any]) -> list[str]:
    saturation, case = record["saturation"], record["break_even"]
    problems: list[str] = []
    if (saturation["ratio"] is None) != saturation["infinite"]:
        problems.append("cost_analysis.saturation: the ratio is null exactly when infinite")
    intervals, kind, ratio = case["intervals"], case["classification"], case["ratio"]
    if case["single_interval"] != (len(intervals) <= 1):
        problems.append("cost_analysis.break_even.single_interval: must be len(intervals) ≤ 1")
    spans = [(i["start"], i["end"]) for i in intervals]
    if any(
        i["start"] > i["end"] or (i["start"] == i["end"] and not i["end_closed"]) for i in intervals
    ):
        problems.append("cost_analysis.break_even.intervals: start < end (a single ratio: closed)")
    if any(a[1] > b[0] for a, b in pairwise(spans)):
        problems.append("cost_analysis.break_even.intervals: increasing and disjoint")
    if any(i["end_closed"] and i["end"] != RQ3_RATIO_MAX for i in intervals):
        problems.append("cost_analysis.break_even.intervals: only the range end 1000 is closed")
    if any(i["end_closed"] for i in intervals[:-1]):
        problems.append("cost_analysis.break_even.intervals: only the last may reach 1000")
    if (kind == "none") != (not intervals):
        problems.append("cost_analysis.break_even: none exactly when no interval is negative")
    closed = bool(intervals) and intervals[-1]["end_closed"]
    if (kind == "right_censored") != closed:
        problems.append(
            "cost_analysis.break_even: right-censored exactly when the last reaches 1000"
        )
    if kind == "right_censored" and not (
        saturation["infinite"] or (saturation["ratio"] or 0) > RQ3_RATIO_MAX
    ):
        problems.append("cost_analysis.break_even: right-censoring needs saturation above 1000")
    if (ratio is None) != (kind in ("none", "right_censored")):
        problems.append("cost_analysis.break_even.ratio: null for none and right-censored")
    if ratio is not None and spans and ratio != spans[-1][1]:
        problems.append("cost_analysis.break_even.ratio: the end of the last interval")
    if kind == "coincidence" and ratio != saturation["ratio"]:
        problems.append("cost_analysis.break_even: coincidence needs ratio = saturation ratio")
    if (
        kind == "crossing"
        and saturation["ratio"] is not None
        and not (ratio is not None and ratio < saturation["ratio"])
    ):
        problems.append("cost_analysis.break_even: crossing needs ratio < saturation ratio")
    if case["zero_crossings"] != sorted(case["zero_crossings"]):
        problems.append("cost_analysis.break_even.zero_crossings: increasing order")
    if [item["phi"] for item in record["friction"]] != [float(p) for p in RQ3_FRICTION]:
        problems.append(f"cost_analysis.friction: the values phi must be {list(RQ3_FRICTION)}")
    for item in record["friction"]:
        expected = None if ratio is None else (1 + item["phi"]) * ratio - item["phi"]
        if (item["ratio"] is None) != (expected is None) or (
            expected is not None and abs(item["ratio"] - expected) > 1e-9 * max(1.0, abs(expected))
        ):
            problems.append("cost_analysis.friction: ratio = (1 + phi) × break-even − phi")
    return problems


COST_ANALYSIS = _contract(
    "cost_analysis",
    1,
    "The RQ3 cost analysis of one configuration, capacity and calibrator: R4 against R3 over the "
    "ratio grid, the exact break-even and saturation ratios, the declined-set calibration "
    "diagnostic, the amount proxies and the C_FP sensitivity (DR-007 §4–§9). Descriptive; "
    "scenario costs are assumptions, never institutional costs.",
    (
        Field("configuration", STUDY_CONFIGURATION, "Configuration"),
        Field("calibrator", Choice(("platt", "isotonic")), "Calibrator giving the probabilities"),
        Field("period", PERIOD, "Full V2 (primary) or the held-out label-latency days"),
        Field("capacity", Number(0, 1, exclusive_minimum=True), "Capacity share c"),
        Field("k", POSITIVE, "Fixed daily review budget K"),
        Field("days", DAYS, "Relative days analysed"),
        Field("score", RUN_FILE, "V2 score record analysed"),
        Field("calibration", RUN_FILE, "Calibration record of the calibrator"),
        Field(
            "queue",
            Object(
                (
                    Field("reviews", POSITIVE, "R3's reviews N_R3"),
                    Field("frauds", POSITIVE, "Frauds of the analysed days F"),
                    Field("captured", COUNT, "Frauds R3 reviews"),
                )
            ),
            "R3's queues over the analysed days",
        ),
        Field(
            "saturation",
            Object(
                (
                    Field("ratio", Number(1), "rho_sat = 1 / smallest probability", nullable=True),
                    Field("infinite", Boolean(), "A queue member has probability 0"),
                )
            ),
            "Saturation ratio: R4 fills the budget on every day from this ratio",
        ),
        Field(
            "break_even",
            Object(
                (
                    Field("classification", Choice(CASES), "Case of DR-007 §4"),
                    Field("ratio", Number(1, 1000), "rho_BE (null: none or right-censored)", True),
                    Field("single_interval", Boolean(), "ΔJ < 0 on one interval only"),
                    Field("intervals", ListOf(BREAK_EVEN_INTERVAL), "Every interval with ΔJ < 0"),
                    Field(
                        "zero_crossings",
                        ListOf(Number(1, 1000)),
                        "Ratios where ΔJ = 0 with unused reviews (ties)",
                    ),
                )
            ),
            "Break-even ratio over (1, 1000]",
        ),
        Field(
            "friction",
            ListOf(
                Object(
                    (
                        Field("phi", Number(0, exclusive_minimum=True), "phi = C_FP / C_R"),
                        Field("ratio", Number(), "(1 + phi) rho_BE − phi", nullable=True),
                    )
                ),
                min_items=1,
            ),
            "C_FP-converted break-even ratios (sensitivity)",
        ),
        Field("grid", ListOf(GRID_ROW, min_items=1), "The ratio grid of DR-007 §2"),
    ),
    _grid_problems,
    _break_even_problems,
)


def _cost_days(record: Mapping[str, Any], partitions: DevelopmentPartitions) -> list[str]:
    role = "full" if record["period"] == "primary" else "held_out"
    return _range_problems("cost_analysis.days", record["days"], partitions.v2, role)


# --- R1 selection --------------------------------------------------------------------------------

R1_SELECTED = (
    "distinct_scores",
    "threshold",
    "true_positives",
    "false_positives",
    "false_negatives",
    "alerts",
    "f1_numerator",
    "f1_denominator",
)


def _r1_rules(record: Mapping[str, Any]) -> list[str]:
    if record["status"] == "error":
        problems = _required("r1_selection", record, ("error",), "selection failed")
        problems += _absent("r1_selection", record, R1_SELECTED, "selection failed")
        error = record["error"]
        if error == "empty_selection_set" and record["transactions"] != 0:
            problems.append("r1_selection.error: the selection set is not empty")
        if error == "no_fraud" and (record["transactions"] == 0 or record["frauds"] != 0):
            problems.append("r1_selection.error: no_fraud needs transactions without frauds")
        return problems
    problems = _absent("r1_selection", record, ("error",), "a threshold is selected")
    problems += _required("r1_selection", record, R1_SELECTED, "a threshold is selected")
    if problems:
        return problems
    tp, fp, fn = record["true_positives"], record["false_positives"], record["false_negatives"]
    frauds, alerts, n = record["frauds"], record["alerts"], record["transactions"]
    if frauds < 1:
        problems.append("r1_selection.frauds: selection needs at least one fraud")
    if tp + fn != frauds:
        problems.append("r1_selection: true_positives + false_negatives ≠ frauds")
    if tp + fp != alerts:
        problems.append("r1_selection: true_positives + false_positives ≠ alerts")
    if not 1 <= alerts <= n or frauds > n:
        problems.append("r1_selection: 1 ≤ alerts ≤ transactions and frauds ≤ transactions")
    if not 1 <= record["distinct_scores"] <= n:
        problems.append("r1_selection.distinct_scores: between 1 and the transactions")
    if record["f1_numerator"] != 2 * tp:
        problems.append("r1_selection.f1_numerator: must be 2 TP")
    if record["f1_denominator"] != frauds + alerts:
        problems.append("r1_selection.f1_denominator: must be P + A (= 2 TP + FP + FN)")
    return problems


def _selection_days(kind: str) -> DayCheck:
    """A selection on all of V2 (primary) or on its label-latency fitting days."""

    def check(record: Mapping[str, Any], partitions: DevelopmentPartitions) -> list[str]:
        role = _period_role(record["period"])
        days = record["selection_days"]
        return _range_problems(f"{kind}.selection_days", days, partitions.v2, role)

    return check


def _r1_rows(record: Mapping[str, Any], partitions: DevelopmentPartitions) -> list[str]:
    rows = partitions.v2.rows
    if record["period"] == "primary" and record["transactions"] != rows:
        return [f"r1_selection.transactions: a full-V2 selection uses all {rows} V2 rows"]
    return []


R1_SELECTION = _contract(
    "r1_selection",
    1,
    "One R1 (F1-optimal) threshold selection on V2 scores of a selected main-seed model "
    "(DR-009, R1 threshold selection; Issue #115).",
    (
        Field("configuration", STUDY_CONFIGURATION, "Configuration"),
        Field("score", RUN_FILE, "V2 score record (model identity)"),
        Field("period", PERIOD, "Full V2 (primary) or label-latency selection days"),
        Field("selection_days", DAYS, "Relative days of the selection set"),
        Field("transactions", COUNT, "Transactions in the selection set"),
        Field("frauds", COUNT, "Frauds P in the selection set"),
        Field("distinct_scores", POSITIVE, "Candidate thresholds", nullable=True),
        Field("threshold", Number(), "Selected threshold at full precision", nullable=True),
        Field("true_positives", COUNT, "TP at the threshold", nullable=True),
        Field("false_positives", COUNT, "FP at the threshold", nullable=True),
        Field("false_negatives", COUNT, "FN at the threshold", nullable=True),
        Field("alerts", COUNT, "Alerts A at the threshold", nullable=True),
        Field("f1_numerator", COUNT, "2 TP", nullable=True),
        Field("f1_denominator", COUNT, "P + A", nullable=True),
        Field("rule_version", Text(), "Version of the selection rule"),
        Field("status", Choice(("selected", "error")), "Outcome"),
        Field(
            "error",
            Choice(("empty_selection_set", "no_fraud", "non_finite_scores")),
            "Recorded error",
            nullable=True,
        ),
    ),
    _r1_rules,
)

# --- H3 selection --------------------------------------------------------------------------------

H3_CANDIDATE = Object(
    (
        Field("configuration", STUDY_CONFIGURATION, "Configuration"),
        Field("frauds_captured", COUNT, "V2 frauds reviewed under R3"),
        Field("frauds", POSITIVE, "V2 frauds"),
        Field("recall", SHARE, "frauds_captured / frauds"),
        Field("average_precision", SHARE, "V2 Average Precision (first tie-break)"),
        Field("evaluation", RUN_FILE, "R3 evaluation record"),
    )
)


def _h3_rules(record: Mapping[str, Any]) -> list[str]:
    candidates = record["candidates"]
    problems: list[str] = []
    names = [c["configuration"] for c in candidates]
    if names != list(CONFIGURATION_NAMES):
        problems.append(f"h3_selection.candidates: {names} are not the eight in DR-011 order")
    problems += [
        f"h3_selection: {c['configuration']} recall ≠ captured / frauds"
        for c in candidates
        if c["frauds_captured"] > c["frauds"] or c["recall"] != c["frauds_captured"] / c["frauds"]
    ]
    if len({c["frauds"] for c in candidates}) > 1:
        problems.append("h3_selection.candidates: every configuration counts the same V2 frauds")
    if record["binding"] != (record["period"] == "primary"):
        problems.append("h3_selection.binding: only the primary selection is binding")
    if problems:
        return problems
    best = max(c["frauds_captured"] for c in candidates)
    tied = [c for c in candidates if c["frauds_captured"] == best]
    steps: list[str] = []
    if len(tied) > 1:
        steps.append("average_precision")
        best_ap = max(c["average_precision"] for c in tied)
        tied = [c for c in tied if c["average_precision"] == best_ap]
        if len(tied) > 1:
            steps.append("fixed_order")
    selected = tied[0]["configuration"]
    if record["selected"] != selected:
        problems.append(f"h3_selection.selected: the rule selects {selected}")
    if record["tie_break_steps"] != steps:
        problems.append(f"h3_selection.tie_break_steps: the rule used {steps}")
    return problems


H3_SELECTION = _contract(
    "h3_selection",
    1,
    "The H3-configuration selection on V2 (DR-010): committed before the Test period is unsealed.",
    (
        Field("period", PERIOD, "Full V2 (primary, binding) or label-latency days (descriptive)"),
        Field("selection_days", DAYS, "Relative days of the V2 evaluations compared"),
        Field("binding", Boolean(), "Whether this selection fixes the H3 configuration"),
        Field("capacity", Number(0, 1, exclusive_minimum=True), "Capacity share c"),
        Field("k", POSITIVE, "Fixed daily review budget K"),
        Field("candidates", ListOf(H3_CANDIDATE), "The eight configurations, DR-011 order"),
        Field("selected", STUDY_CONFIGURATION, "Selected configuration"),
        Field(
            "tie_break_steps",
            ListOf(Choice(("average_precision", "fixed_order"))),
            "Tie-break steps used, in order",
        ),
        Field("rule_version", Text(), "Version of the selection rule"),
    ),
    _h3_rules,
)

# --- run end -------------------------------------------------------------------------------------


_RECORD_FILE = re.compile(r"records/[A-Za-z0-9_.-]+\.json")


def _run_end_rules(record: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if record["outcome"] == "completed":
        problems += _absent("run_end", record, ("error",), "the run completed")
    else:
        problems += _required("run_end", record, ("error",), f"the run {record['outcome']}")
    paths = [entry["path"] for entry in record["records"]]
    if len(set(paths)) != len(paths):
        problems.append("run_end.records: a path is listed once")
    if paths != sorted(paths):
        problems.append("run_end.records: must be listed in increasing path order")
    problems += [
        f"run_end.records: {path} is not another record of the run"
        for path in paths
        if not _RECORD_FILE.fullmatch(path) or path == "records/run_end.json"
    ]
    return problems


RUN_END = _contract(
    "run_end",
    1,
    "Written last: how the run ended and the SHA-256 of every other record. A run directory "
    "without it was interrupted before it could close (a killed process); its completed records "
    "are preserved and its summary can be regenerated (Issue #96, no resume).",
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
    for contract in (
        CANDIDATES,
        COST_PREDICTION,
        RUN,
        PREPROCESSING,
        PREFLIGHT,
        FIT,
        CHECKPOINT,
        TRIAL,
        CONFIGURATION_RECORD,
        REFIT,
        SCORE,
        EVALUATION,
        TIMING,
        DEVICE_COMPARISON,
        CALIBRATION,
        COST_ANALYSIS,
        R1_SELECTION,
        H3_SELECTION,
        RUN_END,
    )
}


# The rules that need the development partitions (DR-001), by record kind.
DAY_CHECKS: dict[str, tuple[DayCheck, ...]] = {
    "run": (_run_inputs,),
    "preprocessing": (_train_rows("preprocessing"),),
    "fit": (_train_rows("fit"),),
    "score": (_score_days,),
    "evaluation": (_evaluation_days,),
    "timing": (_timing_rows,),
    "device_comparison": (_device_days,),
    "cost_analysis": (_cost_days,),
    "calibration": (_calibration_days,),
    "r1_selection": (_selection_days("r1_selection"), _r1_rows),
    "h3_selection": (_selection_days("h3_selection"),),
}


def contract_versions() -> dict[str, int]:
    """The contract version of every record kind (the run record's ``contracts``)."""
    return {kind: contract.version for kind, contract in RECORD_CONTRACTS.items()}


def contract_of(record: object) -> RecordContract:
    """The contract named by a record's ``record`` field; ``RecordContractError`` if none."""
    kind = record.get("record") if isinstance(record, Mapping) else None
    if not isinstance(kind, str) or kind not in RECORD_CONTRACTS:
        raise RecordContractError(f"Unknown record kind {kind!r}")
    return RECORD_CONTRACTS[kind]


def record_problems(record: object, partitions: DevelopmentPartitions) -> list[str]:
    """Every problem of ``record``: its contract, then (on a valid structure) its day rules."""
    contract = contract_of(record)
    problems = contract.problems(record)
    if problems or not isinstance(record, Mapping):
        return problems
    checks = DAY_CHECKS.get(contract.kind, ())
    return [problem for check in checks for problem in check(record, partitions)]


def validate_record(
    record: object, partitions: DevelopmentPartitions, kind: str | None = None
) -> None:
    """Validate ``record`` against its contract and the development partitions (and check that it
    is of ``kind``, if given)."""
    contract = contract_of(record)
    if kind is not None and contract.kind != kind:
        raise RecordContractError(f"Expected a {kind} record, got a {contract.kind} record")
    problems = record_problems(record, partitions)
    if problems:
        raise RecordContractError(
            f"{contract.kind} record (version {contract.version}) does not satisfy its "
            "contract:\n- " + "\n- ".join(problems)
        )


def dump_record(record: Mapping[str, Any], partitions: DevelopmentPartitions) -> str:
    """The record as JSON text, after validating it."""
    validate_record(record, partitions)
    return dumps(record)


def read_record(
    path: Path, partitions: DevelopmentPartitions, kind: str | None = None
) -> dict[str, Any]:
    """Read and validate one record file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise FileAccessError(f"{path}: cannot be read ({error})") from error
    record = loads(text, str(path))
    try:
        validate_record(record, partitions, kind)
    except RecordContractError as error:
        raise RecordContractError(f"{path}: {error}") from error
    logger.debug("Read %s record %s", record["record"], path)
    return record
