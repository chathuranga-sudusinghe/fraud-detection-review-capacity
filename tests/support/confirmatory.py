"""Builders of valid confirmatory records on synthetic data (no data is read). The inference
record is produced by the real inference functions on simulated replicates, so a valid record is
exactly what the production code would write."""

from __future__ import annotations

import copy
import hashlib
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
from review_capacity.development.records import ALL_CONFIGURATIONS
from review_capacity.evaluation.contrasts import estimands
from review_capacity.evaluation.declaration import EvaluationDeclaration, load_declaration
from review_capacity.evaluation.hypotheses import analyse_h1, analyse_h2, analyse_h3, decide
from review_capacity.evaluation.inference import effect_units
from review_capacity.evaluation.mcnemar import cell_mcnemar
from review_capacity.evaluation.resampling import Replicates
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
        "calibrators": [file_ref("analysis-1/calibration-LGB-b"), file_ref("analysis-1/cal.bin")],
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
