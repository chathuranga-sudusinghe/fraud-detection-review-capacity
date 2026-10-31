"""Builders of valid confirmatory records on synthetic data (no data is read). The inference
record is produced by the real inference functions on simulated replicates, so a valid record is
exactly what the production code would write."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
from functools import cache
from typing import Any

import numpy as np

from review_capacity.config import load_dataset_spec
from review_capacity.confirmatory.identity import (
    artifact_hashes,
    declaration_hashes,
    frozen_state_id,
)
from review_capacity.confirmatory.records import (
    ConfirmatoryContext,
    contract_versions,
    scores_bundle_sha256,
    to_json_value,
)
from review_capacity.confirmatory.slices import SliceInputs
from review_capacity.development.records import ALL_CONFIGURATIONS
from review_capacity.evaluation.calibration import fit_isotonic, fit_platt
from review_capacity.evaluation.contrasts import estimands
from review_capacity.evaluation.declaration import EvaluationDeclaration, load_declaration
from review_capacity.evaluation.false_negatives import (
    Boundaries,
    SliceVariable,
    amount_codes,
    amount_edges,
    declared_levels,
    false_negative_analysis,
    misses_of,
    pooled_variable,
    week_codes,
)
from review_capacity.evaluation.fidelity import fidelity
from review_capacity.evaluation.hypotheses import analyse_h1, analyse_h2, analyse_h3, decide
from review_capacity.evaluation.inference import effect_units
from review_capacity.evaluation.mcnemar import cell_mcnemar
from review_capacity.evaluation.outcomes import (
    Partition,
    Thresholds,
    calibration_outcome,
    confirmatory_outcomes,
    secondary_resamples,
)
from review_capacity.evaluation.queue import DailyQueue
from review_capacity.evaluation.ranking import Capacity, Truth
from review_capacity.evaluation.resampling import Replicates
from review_capacity.evaluation.rq3 import CostSubject, rq3_outcomes
from review_capacity.evaluation.sensitivity import analyse_s1, analyse_s2, sesoi_views
from support.synthetic import REPO_PATHS

RUN_ID = "2026-11-02T090000Z"
TAG_COMMIT = "a" * 40
B = load_declaration(REPO_PATHS)[0].resampling.replicates  # the declared replicate count
RECALLS = np.array([0.40, 0.39, 0.34, 0.33, 0.40, 0.39, 0.34, 0.33])
RATES = np.array([0.0195, 0.0205, 0.0135, 0.0260, 0.0200, 0.0190, 0.0210, 0.0200])


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def declaration() -> EvaluationDeclaration:
    return load_declaration(REPO_PATHS)[0]


def context() -> ConfirmatoryContext:
    return ConfirmatoryContext.from_specs(load_dataset_spec(REPO_PATHS), declaration())


def file_ref(name: str) -> dict[str, str]:
    return {"path": f"outputs/{name}", "sha256": sha(name)}


def run_file(name: str) -> dict[str, str]:
    return {"path": name, "sha256": sha(name)}


def frozen_inputs() -> dict[str, Any]:
    return {
        "development_run": {"run_id": "2026-10-25T134341Z", "run_end": file_ref("dev/run_end")},
        "analysis_runs": [
            {"run_id": "2026-10-29T013124Z", "run_end": file_ref("analysis-1/run_end")},
            {"run_id": "2026-10-29T013414Z", "run_end": file_ref("analysis-2/run_end")},
        ],
        "h3_selection": file_ref("analysis-1/h3_selection-primary"),
        "h3_configuration": "LGB-b",
        "preprocessing": file_ref("preprocessing"),
        "preprocessing_arrays": file_ref("preprocessing-arrays"),
        "models": {name: file_ref(f"model-{name}") for name in ALL_CONFIGURATIONS},
        "model_records": {name: file_ref(f"model-run-{name}") for name in ALL_CONFIGURATIONS},
        "model_manifests": {name: file_ref(f"manifest-{name}") for name in ALL_CONFIGURATIONS},
        "thresholds": [file_ref("analysis-1/r1_selection-LGB-b"), file_ref("analysis-1/eval-R2")],
        "calibrators": [
            file_ref("analysis-1/calibration-LGB-b"),
            file_ref("analysis-1/cal.bin"),
            file_ref("calibrator-platt"),
            file_ref("calibrator-isotonic"),
        ],
    }


def identity(inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    inputs = inputs if inputs is not None else frozen_inputs()
    decl = declaration()
    hashes = {
        "settings_sha256": sha("settings"),
        "training_declaration_sha256": sha("training"),
        "evaluation_declaration_sha256": sha("evaluation"),
        "dataset_manifest_sha256": sha("manifest"),
    }
    base = {
        "run_id": RUN_ID,
        "run_type": "confirmatory",
        "evaluator": {
            "code_commit": TAG_COMMIT,
            "uncommitted_changes": False,
            "source_sha256": sha("evaluator-source"),
        },
        "freeze": {"tag": "protocol-freeze", "tag_commit": TAG_COMMIT},
        "training": {
            "run_id": "2026-10-25T134341Z",
            "code_commit": "c" * 40,
            "source_sha256": sha("training-source"),
        },
        **hashes,
        "seeds": {
            "main": 42,
            "additional": [43, 44],
            "sampler": 42,
            "preprocessing": 0,
            "resampling": decl.resampling.root_seed,
        },
    }
    base["frozen_state_sha256"] = frozen_state_id(
        commit=TAG_COMMIT,
        declarations=declaration_hashes(base),
        artifacts=artifact_hashes(inputs),
    )
    return base


def envelope(kind: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "record": kind,
        "schema_version": contract_versions()[kind],
        "identity": identity(inputs),
    }


def test_run() -> dict[str, Any]:
    inputs = frozen_inputs()
    ctx = context()
    prefix = ctx.test.transaction_id_sha256_prefix
    return {
        **envelope("test_run", inputs),
        "started": "2026-11-02T09:00:00.000000Z",
        "environment": {"python": "3.12"},
        "settings": file_ref("config/models.yaml") | {"sha256": sha("settings")},
        "training_declaration": file_ref("config/training.yaml") | {"sha256": sha("training")},
        "evaluation_declaration": file_ref("config/evaluation.yaml")
        | {"sha256": sha("evaluation")},
        "dataset_manifest": file_ref("data/datapackage.json") | {"sha256": sha("manifest")},
        "frozen_inputs": inputs,
        "test": {
            "first_day": ctx.test.first_day,
            "last_day": ctx.test.last_day,
            "rows": ctx.test.rows,
            "transaction_id_sha256": prefix + "0" * (64 - len(prefix)),
        },
        "contracts": contract_versions(),
    }


def phase(
    name: str = "scoring",
    attempt: int = 1,
    outcome: str = "completed",
    scores: str | None = None,
    started: str = "2026-11-02T09:01:00.000000Z",
) -> dict[str, Any]:
    scoring = name == "scoring"
    if scores is None and (not scoring or outcome == "completed"):
        scores = sha("scores-bundle")
    return {
        **envelope("test_phase"),
        "phase": name,
        "attempt": attempt,
        "started": started,
        "finished": "2026-11-02T09:29:00.000000Z",
        "outcome": outcome,
        "error": None if outcome == "completed" else "RuntimeError: out of memory",
        "labels_accessed": not scoring,
        "scores_sha256": scores if (not scoring or outcome == "completed") else None,
    }


def scores(configuration: str = "LGB-b") -> dict[str, Any]:
    ctx = context()
    return {
        **envelope("test_scores"),
        "configuration": configuration,
        "partition": "Test",
        "days": {"first": ctx.test.first_day, "last": ctx.test.last_day},
        "rows": ctx.test.rows,
        "model": file_ref(f"model-{configuration}"),
        "file": run_file(f"scores/{configuration}.parquet"),
        "format": "parquet",
        "min_score": 0.0001,
        "max_score": 0.9987,
        "distinct_scores": ctx.test.rows - 11,
        "device": "cpu",
        "timing": {"passes": 1, "seconds": 3.5, "rows_per_second": ctx.test.rows / 3.5},
    }


def around(base: np.ndarray, sigma: float, seed: int) -> Replicates:
    rng = np.random.default_rng(seed)
    return Replicates(base + rng.normal(0.0, sigma, (B, 8)))


def inference() -> dict[str, Any]:
    """The inference record that the production functions give on simulated replicates."""
    decl = declaration()
    table = estimands(decl)
    pigeonhole = around(RECALLS, 0.004, 1)
    h1 = analyse_h1(decl, table, RECALLS, pigeonhole)
    h2 = analyse_h2(decl, table, RECALLS, pigeonhole)
    sets = {length: around(RATES, 0.0008, 10 + length) for length in (3, 1, 7)}
    h3 = analyse_h3(decl, "LGB-b", RATES, sets)
    primary = decide(decl, h1, h2, h3)
    s1 = analyse_s1(decl, table, RECALLS, pigeonhole, around(RECALLS, 0.0012, 5), primary)
    s2 = analyse_s2(decl, table, RECALLS, around(RECALLS, 0.004, 2), primary)
    rng = np.random.default_rng(31)
    captures = rng.random((8, 400)) < np.linspace(0.30, 0.45, 8)[:, np.newaxis]
    family = float(table.apply(RECALLS)[table.names.index("family")])
    units = effect_units(family, frauds=2660, days=28, k=56, oracle_ceiling=0.6)
    return {
        **envelope("test_inference"),
        "resampling": {
            "replicates": decl.resampling.replicates,
            "root_seed": decl.resampling.root_seed,
            "block_length": decl.resampling.block_length,
            "block_length_sensitivity": list(decl.resampling.block_length_sensitivity),
        },
        "undefined_replicates": dict.fromkeys(
            ("pigeonhole_h1_h2", "naive_s1", "day_block_s2", "day_block_h3"), 0
        ),
        "h1": to_json_value(h1),
        "h2": to_json_value(h2),
        "h3": to_json_value(h3),
        "holm": to_json_value(primary.holm),
        "decisions": {
            "h1_rejected": primary.h1_rejected,
            "h2_equivalent": primary.h2_equivalent,
            "h3_preserved": primary.h3_preserved,
            "h1_category": primary.h1_category.value,
        },
        "s1": {
            "quantities": [{**to_json_value(q), "se": q.se} for q in s1.quantities],
            "p_h1": s1.p_h1,
            "interval_h1": to_json_value(s1.interval_h1),
            "position_h1": to_json_value(s1.position_h1),
            "p_h2": s1.p_h2,
            "intervals_h2": to_json_value(s1.intervals_h2),
            "comparison": to_json_value(s1.comparison),
            "reasons": [],
        },
        "s2": {
            "p_h1": to_json_value(s2.h1.p),
            "p_h2": to_json_value(s2.h2.p),
            "h1_interval": to_json_value(s2.h1.interval),
            "h1_position": to_json_value(s2.h1.position),
            "comparison": to_json_value(s2.comparison),
        },
        "sesoi_views": to_json_value(sesoi_views(decl, primary)),
        "mcnemar": to_json_value(cell_mcnemar(decl, captures)),
        "effect_basis": {"frauds": 2660, "days": 28, "k": 56, "oracle_ceiling": 0.6},
        "effect_units": [{"estimand": "family", **to_json_value(units)}],
    }


def run_end(records: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        **envelope("test_run_end"),
        "finished": "2026-11-02T09:30:00.000000Z",
        "outcome": "completed",
        "error": None,
        "records": records or [run_file("records/test_run.json")],
    }


def run_records() -> list[dict[str, Any]]:
    """The valid records of a completed run: the run, a failed and a completed scoring, a
    completed evaluation, the nine score records and the inference."""
    files = [scores(name)["file"]["sha256"] for name in ALL_CONFIGURATIONS]
    bundle = scores_bundle_sha256(files)
    return [
        test_run(),
        phase("scoring", 1, "failed", started="2026-11-02T09:01:00.000000Z"),
        phase("scoring", 2, "completed", bundle, started="2026-11-02T09:10:00.000000Z"),
        phase("evaluation", 1, "completed", bundle, started="2026-11-02T09:20:00.000000Z"),
        *outcome_records(),
        *(scores(name) for name in ALL_CONFIGURATIONS),
        inference(),
    ]


def mutated(record: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    """A deep copy of ``record`` with the field at the dotted ``path`` set to ``value``
    (list items by index: ``h1.cells.0.estimate``)."""
    result = copy.deepcopy(record)
    target: Any = result
    *parents, last = path.split(".")
    for part in parents:
        target = target[int(part)] if isinstance(target, list) else target[part]
    if isinstance(target, list):
        target[int(last)] = value
    else:
        target[last] = value
    return result


def without(record: dict[str, Any], path: str) -> dict[str, Any]:
    result = copy.deepcopy(record)
    target: Any = result
    *parents, last = path.split(".")
    for part in parents:
        target = target[int(part)] if isinstance(target, list) else target[part]
    del target[last]
    return result


# --- the outcome records on a Test-shaped synthetic partition -------------------------------------

OUTCOME_B = 150  # replicates of the outcome records: the contract takes the declared count
V2_VOLUME = 77_822 / 28  # the declared V2 volume; K = round(c x N) is 28, 56, 139 and 278
THRESHOLDS = Thresholds(0.5, 0.3, {0.01: 0.45, 0.02: 0.35, 0.05: 0.25, 0.1: 0.15})


@cache
def outcome_declaration() -> EvaluationDeclaration:
    base = declaration()
    return dataclasses.replace(
        base, resampling=dataclasses.replace(base.resampling, replicates=OUTCOME_B)
    )


@cache
def outcome_context() -> ConfirmatoryContext:
    return ConfirmatoryContext.from_specs(load_dataset_spec(REPO_PATHS), outcome_declaration())


@cache
def outcome_world() -> tuple[Partition, dict[str, np.ndarray]]:
    """The 28 Test days and 78,542 rows with a few percent frauds, a pseudo-entity for each fraud,
    and the scores of one configuration and of the reference (ties from rounding)."""
    ctx = outcome_context()
    days_count = ctx.test.last_day - ctx.test.first_day + 1
    rng = np.random.default_rng(21)
    sizes = np.full(days_count, ctx.test.rows // days_count)
    sizes[: ctx.test.rows - sizes.sum()] += 1
    days = np.repeat(np.arange(ctx.test.first_day, ctx.test.last_day + 1), sizes).astype(np.int64)
    labels = (rng.random(len(days)) < 0.03).astype(np.int8)
    amounts = np.round(rng.gamma(2.0, 60.0, len(days)), 2)
    _, entities = np.unique(rng.integers(0, 400, int(labels.sum())), return_inverse=True)
    partition = Partition(
        Truth.build(days, labels, amounts),
        DailyQueue.build(np.arange(len(days), dtype=np.int64), days),
        entities.astype(np.intp),
        V2_VOLUME,
    )
    scores = {
        "LGB-b": np.round(np.clip(rng.normal(0.1 + 0.6 * labels, 0.2), 0, 1), 4),
        "LR": np.round(np.clip(rng.normal(0.08 + 0.4 * labels, 0.2), 0, 1), 4),
    }
    return partition, scores


@cache
def outcome_capacities() -> tuple[Capacity, ...]:
    decl = outcome_declaration()
    return tuple(
        Capacity(s, k) for s, k in zip(decl.capacity.shares, decl.capacity.budgets, strict=True)
    )


@cache
def _outcomes() -> dict[str, dict[str, Any]]:
    partition, scores = outcome_world()
    return confirmatory_outcomes(
        outcome_declaration(), partition, scores, {"LGB-b": THRESHOLDS}, outcome_capacities()
    )


def evaluation(configuration: str = "LGB-b") -> dict[str, Any]:
    ctx = outcome_context()
    outcome = copy.deepcopy(_outcomes()["LR" if configuration == "LR" else "LGB-b"])
    record: dict[str, Any] = to_json_value(
        {
            **envelope("test_evaluation"),
            "configuration": configuration,
            "partition": "Test",
            "days": {"first": ctx.test.first_day, "last": ctx.test.last_day},
            "scores_sha256": scores(configuration)["file"]["sha256"],
            "thresholds": (
                None
                if configuration == "LR"
                else {
                    "r0": THRESHOLDS.r0,
                    "r1": THRESHOLDS.r1,
                    "r2": [{"capacity": c, "threshold": t} for c, t in THRESHOLDS.r2.items()],
                }
            ),
            "ranking": outcome["ranking"],
            "r3": outcome["r3"],
            "policies": outcome["thresholds"],
        }
    )
    return record


def workload(configuration: str = "LGB-b", reference_volume: float = V2_VOLUME) -> dict[str, Any]:
    ctx = outcome_context()
    entries = [
        {
            "capacity": entry["capacity"],
            "k": entry["k"],
            "threshold": entry["threshold"],
            "fidelity": fidelity(entry["daily"], entry["capacity"], entry["k"], reference_volume),
        }
        for entry in _outcomes()["LGB-b"]["thresholds"]
        if entry["policy"] == "R2"
    ]
    record: dict[str, Any] = to_json_value(
        {
            **envelope("test_workload"),
            "configuration": configuration,
            "partition": "Test",
            "days": {"first": ctx.test.first_day, "last": ctx.test.last_day},
            "entries": entries,
        }
    )
    return record


@cache
def _calibrators() -> dict[str, Any]:
    partition, scores_by = outcome_world()
    labels, fitting = partition.truth.labels, scores_by["LGB-b"]
    return {"platt": fit_platt(fitting, labels)[0], "isotonic": fit_isotonic(fitting, labels)}


@cache
def _calibration(method: str) -> dict[str, Any]:
    partition, scores_by = outcome_world()
    resamples = secondary_resamples(outcome_declaration(), len(partition.truth.day_values))
    return calibration_outcome(
        outcome_declaration(), partition, scores_by["LGB-b"], _calibrators()[method], resamples
    )


def calibration(method: str = "platt", configuration: str = "LGB-b") -> dict[str, Any]:
    ctx = outcome_context()
    result = copy.deepcopy(_calibration(method))
    intervals = result.pop("intervals")
    record: dict[str, Any] = to_json_value(
        {
            **envelope("test_calibration"),
            "configuration": configuration,
            "method": method,
            "role": "primary" if method == "platt" else "sensitivity",
            "partition": "Test",
            "days": {"first": ctx.test.first_day, "last": ctx.test.last_day},
            "calibrator": file_ref(f"calibrator-{method}"),
            "diagnostic": result,
            "intervals": intervals,
        }
    )
    return record


@cache
def _rq3() -> dict[tuple[str, str], list[dict[str, Any]]]:
    partition, scores_by = outcome_world()
    raw = scores_by["LGB-b"]
    subjects = [
        CostSubject("LGB-b", method, raw, _calibrators()[method].predict(raw))
        for method in ("platt", "isotonic")
    ]
    return rq3_outcomes(outcome_declaration(), partition, subjects, outcome_capacities())


def rq3_record(method: str = "platt", configuration: str = "LGB-b") -> dict[str, Any]:
    ctx = outcome_context()
    record: dict[str, Any] = to_json_value(
        {
            **envelope("test_rq3"),
            "configuration": configuration,
            "calibrator": method,
            "partition": "Test",
            "days": {"first": ctx.test.first_day, "last": ctx.test.last_day},
            "scores_sha256": scores(configuration)["file"]["sha256"],
            "calibrator_file": file_ref(f"calibrator-{method}"),
            "capacities": copy.deepcopy(_rq3()[("LGB-b", method)]),
        }
    )
    return record


@cache
def outcome_slices() -> SliceInputs:
    """The slice inputs of the outcome world: boundaries, variables and the entity index."""
    partition, _ = outcome_world()
    rule = outcome_declaration().false_negative
    rng = np.random.default_rng(23)
    truth = partition.truth
    fraud = truth.labels == 1
    frauds = int(fraud.sum())
    edges = amount_edges(truth.amounts, rule.amount_quantiles)
    kept = {
        name: {"A": 4000, "B": 2500, "C": 150}
        for name in ("ProductCD", "card4", "card6", "DeviceType")
    }
    boundaries = Boundaries(edges, kept)
    levels = declared_levels(outcome_declaration())
    variables = [
        SliceVariable("identity", levels["identity"], rng.integers(0, 2, frauds).astype(np.intp)),
        *(
            pooled_variable(
                name, np.array(rng.choice(["A", "B", "C", "D"], frauds), dtype=np.str_), boundaries
            )
            for name in ("ProductCD", "card4", "card6", "DeviceType")
        ),
        SliceVariable(
            "amount_band", levels["amount_band"], amount_codes(truth.amounts[fraud], edges)
        ),
        SliceVariable("week", levels["week"], week_codes(truth.days[fraud], rule.test_weeks)),
        SliceVariable("entity", levels["entity"], rng.integers(0, 2, frauds).astype(np.intp)),
    ]
    return SliceInputs(boundaries, variables, partition.entity_index)


@cache
def _false_negatives() -> tuple[Boundaries, dict[str, dict[str, Any]]]:
    partition, scores_by = outcome_world()
    slices = outcome_slices()
    capacity = outcome_capacities()[1]  # the primary capacity, 2 %
    found = misses_of(partition, scores_by["LGB-b"], capacity)
    return slices.boundaries, false_negative_analysis(
        outcome_declaration(), partition, slices.variables, {"LGB-b": found}, capacity
    )


def false_negative_record(configuration: str = "LGB-b") -> dict[str, Any]:
    ctx = outcome_context()
    boundaries, tables = _false_negatives()
    record: dict[str, Any] = to_json_value(
        {
            **envelope("test_false_negative"),
            "configuration": configuration,
            "partition": "Test",
            "days": {"first": ctx.test.first_day, "last": ctx.test.last_day},
            "scores_sha256": scores(configuration)["file"]["sha256"],
            **copy.deepcopy({k: v for k, v in tables["LGB-b"].items() if k != "slices"}),
            "boundaries": {
                "amount_edges": list(boundaries.amount_edges),
                "kept": {name: dict(levels) for name, levels in boundaries.kept.items()},
            },
            "slices": copy.deepcopy(tables["LGB-b"]["slices"]),
        }
    )
    return record


def outcome_records() -> list[dict[str, Any]]:
    """The valid outcome records of a completed run: an evaluation for every configuration and the
    reference, a workload for every configuration, and both calibrations and both cost analyses of
    every configuration.
    The configurations other than the one computed share its numbers (the contracts name no
    model)."""
    study = ALL_CONFIGURATIONS[:-1]
    return [
        *(evaluation(name) for name in ALL_CONFIGURATIONS),
        *(workload(name) for name in study),
        *(calibration(method, name) for name in study for method in ("platt", "isotonic")),
        *(rq3_record(method, name) for name in study for method in ("platt", "isotonic")),
        *(false_negative_record(name) for name in study),
    ]
