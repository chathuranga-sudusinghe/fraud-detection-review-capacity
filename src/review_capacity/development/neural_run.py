"""The neural inputs of a development run: the shared preprocessing artifact, the network inputs of
Train, V1 and V2, the recorded batch plans and their records (DR-012; DR-015 §7; Issue #151).

A run that fits a neural model or the logistic-regression reference loads the shared Train-only
preprocessing artifact (``fit-neural-preprocessing``) once and applies it to the verified Train,
V1 and V2 rows. The artifact is checked against the declared preprocessing seed, the feature
declaration and the ordered Train identity before any row is transformed. A neural run also reads
the micro-batch preflight (``preflight-neural``) with ``training.preflight.read_preflight``, which
refuses a plan made for another artifact, declaration, settings, package source or GPU, or for
other candidates. The artifact and each candidate's plan are written into the run as
``preprocessing`` and ``preflight`` records, which every neural ``fit`` record links to.

The Train inputs are built once and shared by the weightings (their weights differ, the inputs do
not). V1 carries the identifiers and times of the V1 objective, never as network inputs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from review_capacity.config import FeatureSpec
from review_capacity.development.inputs import ScoringInputs, SearchInputs
from review_capacity.development.records import RECORD_CONTRACTS
from review_capacity.development.run_output import RunDirectory
from review_capacity.development.run_start import project_file
from review_capacity.errors import ModelError
from review_capacity.features.neural_preprocessing import (
    MANIFEST_FILE,
    NeuralPreprocessing,
    load_neural_preprocessing,
)
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import log_step
from review_capacity.models.neural_training import BatchPlan, NeuralData, NeuralInputs
from review_capacity.models.registry import Weighting
from review_capacity.paths import ProjectPaths
from review_capacity.training.declaration import TrainingDeclaration
from review_capacity.training.preflight import MODELS as NEURAL_MODELS
from review_capacity.training.preflight import read_preflight
from review_capacity.training.sampler import Candidate

logger = logging.getLogger(__name__)

PREPROCESSING_RECORD = "preprocessing"
PREFLIGHT_RECORD = "preflight"


@dataclass(frozen=True)
class NeuralRun:
    """What the neural models and the reference share in one run."""

    artifact: NeuralPreprocessing
    directory: Path
    artifact_sha256: str
    train: Mapping[Weighting, NeuralData]
    v1: NeuralData
    v1_inputs: NeuralInputs
    v2_inputs: NeuralInputs
    plans: Mapping[str, Mapping[int, BatchPlan]]
    preflight_path: Path | None

    def scoring_inputs(self, label: str) -> NeuralInputs:
        """The network inputs of ``"V1"`` or ``"V2"``."""
        return {"V1": self.v1_inputs, "V2": self.v2_inputs}[label]


def neural_models_of(configurations: Sequence[Any]) -> tuple[str, ...]:
    """The neural models (``resnet``, ``ft_transformer``) among ``configurations``, in DR-011
    order, each once."""
    named = {c.model.name for c in configurations}
    return tuple(model for model in NEURAL_MODELS if model in named)


def preflight_identity(
    artifact_sha256: str,
    declaration_sha256: str,
    settings_sha256: str,
    code: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """The values a preflight must have been made under for this run: the artifact, the training
    declaration, the model settings, the package source and the GPU."""
    return {
        "artifact_sha256": artifact_sha256,
        "declaration_sha256": declaration_sha256,
        "settings_sha256": settings_sha256,
        "source_sha256": code["package"]["source_sha256"],
        "gpu_name": (environment.get("gpu") or {}).get("name"),
    }


def build_neural_run(
    features: FeatureSpec,
    declaration: TrainingDeclaration,
    search: SearchInputs,
    scoring: ScoringInputs,
    preprocessing: Path,
    *,
    preflight: Path | None,
    candidates: Mapping[str, Sequence[Candidate]],
    models: Sequence[str],
    declaration_sha256: str,
    settings_sha256: str,
    code: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> NeuralRun:
    """Load the artifact, transform Train, V1 and V2 once and read the recorded batch plans of
    ``models`` (none: the reference alone needs no preflight). The preflight must have been made
    for this artifact, declaration, settings, package source and GPU (``preflight_identity``)."""
    train = search.train
    with log_step(logger, "Neural inputs of the run"):
        artifact = load_neural_preprocessing(
            preprocessing,
            features,
            declaration.preprocessing_seed,
            train_ids_sha256=train.ordered_ids_sha256,
        )
        sha256 = artifact.artifact_sha256
        train_inputs = artifact.to_neural_inputs(train.features)
        v1_inputs = artifact.to_neural_inputs(scoring.v1.features)
        v2_inputs = artifact.to_neural_inputs(scoring.v2.features)
        by_weighting = {
            weighting: NeuralData(train_inputs, train.labels, search.train_data(weighting).weights)
            for weighting in Weighting
        }
        v1_data = search.v1_data()
        valid = NeuralData(v1_inputs, v1_data.labels, v1_data.weights, v1_data.ids, v1_data.times)
    plans: dict[str, dict[int, BatchPlan]] = {}
    if models:
        if preflight is None:
            raise ModelError("A neural run needs the micro-batch preflight (--preflight)")
        identity = preflight_identity(
            sha256, declaration_sha256, settings_sha256, code, environment
        )
        plans = read_preflight(
            preflight, declaration, candidates, identity=identity, models=tuple(models)
        )
    return NeuralRun(
        artifact,
        preprocessing,
        sha256,
        by_weighting,
        valid,
        v1_inputs,
        v2_inputs,
        plans,
        preflight if models else None,
    )


# --- Records ------------------------------------------------------------------------------


def _order_sha256(artifact: NeuralPreprocessing) -> str:
    layout = artifact.layout
    names = "\n".join([*layout.numeric, *layout.categorical])
    return hashlib.sha256(names.encode("utf-8")).hexdigest()


def preprocessing_record(
    paths: ProjectPaths, identity: Mapping[str, Any], neural: NeuralRun, search: SearchInputs
) -> dict[str, Any]:
    """The ``preprocessing`` record of the artifact in use (contract ``preprocessing``)."""
    artifact, train = neural.artifact, search.train
    layout = artifact.layout
    return {
        "record": PREPROCESSING_RECORD,
        "schema_version": RECORD_CONTRACTS[PREPROCESSING_RECORD].version,
        "identity": dict(identity),
        "artifact": project_file(paths, neural.directory / MANIFEST_FILE),
        "artifact_sha256": neural.artifact_sha256,
        "seed": artifact.seed,
        "train": {
            "rows": train.rows,
            "transaction_ids_sha256": train.ordered_ids_sha256,
            "file_sha256": train.provenance.sha256,
        },
        "inputs": {
            "numeric": len(artifact.numeric_inputs),
            "categorical": len(artifact.categorical_inputs),
            "indicators": len(artifact.indicators),
            "total": len(layout.numeric) + len(layout.categorical),
            "order_sha256": _order_sha256(artifact),
        },
        "settings": {
            "requested_quantiles": artifact.requested_quantiles,
            "actual_quantiles": artifact.actual_quantiles,
            "missing_label": artifact.missing_label,
            "category_dropout": layout.category_dropout,
        },
        "software": dict(artifact.versions),
    }


def _attempt(phase: Mapping[str, Any], name: str, rows: int, margin: int) -> dict[str, Any]:
    """One recorded attempt of the preflight in the form of the ``preflight`` contract."""
    if phase["error"] is not None:
        return {"phase": name, "rows": rows, "outcome": "out_of_memory",
                "peak_used_mib": None, "free_mib": None}  # fmt: skip
    free = phase["free_at_peak_mib"]
    if not phase["passes"] and (free is None or free >= margin):
        raise ModelError(
            f"The preflight attempt of {name} at {rows} rows failed without a memory shortfall "
            f"({'; '.join(phase['problems']) or 'no reading'}): it cannot be recorded"
        )
    return {
        "phase": name,
        "rows": rows,
        "outcome": "pass" if phase["passes"] else "below_margin",
        "peak_used_mib": phase["sampled_max_used_mib"],
        "free_mib": free,
    }


def preflight_records(
    preflight: Mapping[str, Any],
    identity: Mapping[str, Any],
    candidate_files: Mapping[str, Mapping[str, str]],
    models: Sequence[str],
) -> dict[tuple[str, int], dict[str, Any]]:
    """One ``preflight`` record per candidate of ``models``, from the preflight JSON the run has
    already validated with ``read_preflight``. ``candidate_files`` are the run-directory
    references of the models' candidate lists."""
    margin = preflight["margin_mib"]
    version = RECORD_CONTRACTS[PREFLIGHT_RECORD].version
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for model in models:
        entry = preflight["models"][model]
        for item in entry["candidates"]:
            attempts = [
                _attempt(a["training"], "training", a["micro_batch"], margin)
                for a in item["attempts"]
            ]
            attempts.append(
                _attempt(item["evaluation"], "evaluation", entry["evaluation_batch"], margin)
            )
            records[(model, item["trial"])] = {
                "record": PREFLIGHT_RECORD,
                "schema_version": version,
                "identity": dict(identity),
                "model": model,
                "trial": item["trial"],
                "candidate_list": dict(candidate_files[model]),
                "effective_batch": entry["effective_batch"],
                "evaluation_batch": entry["evaluation_batch"],
                "margin_mib": margin,
                "attempts": attempts,
                "micro_batch": item["micro_batch"],
                "status": "pass",
                "error": None,
            }
    return records


def _run_file(run: RunDirectory, path: Path) -> dict[str, str]:
    """A ``{path, sha256}`` reference to a file inside the run directory."""
    return {"path": path.relative_to(run.path).as_posix(), "sha256": file_sha256(path)}


@dataclass(frozen=True)
class NeuralRecords:
    """The run-directory references of the records the neural fits link to."""

    preprocessing: dict[str, str]
    preflights: Mapping[tuple[str, int], dict[str, str]]


def write_neural_records(
    run: RunDirectory,
    paths: ProjectPaths,
    identity: Mapping[str, Any],
    neural: NeuralRun,
    search: SearchInputs,
    candidate_files: Mapping[str, Mapping[str, str]],
    models: Sequence[str],
) -> NeuralRecords:
    """Write the ``preprocessing`` record and, for each neural candidate, its ``preflight``
    record; return their references."""
    path = run.write_record(preprocessing_record(paths, identity, neural, search))
    preprocessing = _run_file(run, path)
    preflights: dict[tuple[str, int], dict[str, str]] = {}
    if models:
        if neural.preflight_path is None:
            raise ModelError("The run holds no micro-batch preflight")
        recorded = json.loads(neural.preflight_path.read_text(encoding="utf-8"))
        for (model, trial), record in preflight_records(
            recorded, identity, candidate_files, models
        ).items():
            preflights[(model, trial)] = _run_file(
                run, run.write_record(record, model, f"trial{trial}")
            )
    return NeuralRecords(preprocessing, preflights)
