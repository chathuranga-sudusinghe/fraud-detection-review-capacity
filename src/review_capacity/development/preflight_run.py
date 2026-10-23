"""The ``preflight-neural`` run: the micro-batch preflight of DR-015 §7 on the project GPU
(Issue #151, Part 2).

Reads no research data: the vocabularies and column counts come from the shared preprocessing
artifact (DR-012), the candidate lists from the training declaration, and the inputs are
synthetic. Writes ``preflight.json`` and the run's log inside ``outputs/``. The record names the
artifact's SHA-256, the declaration and settings hashes and the environment, so that a
development run can check the plan against them (``training.preflight.read_preflight``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from review_capacity.config import FeatureSpec
from review_capacity.development.run_output import LOG_NAME as RUN_LOG_NAME
from review_capacity.development.run_output import check_new_directory, default_directory
from review_capacity.errors import EnvironmentMismatchError
from review_capacity.features.neural_preprocessing import load_neural_preprocessing
from review_capacity.models.device import environment_record
from review_capacity.models.run_record import load_settings_with_snapshot
from review_capacity.paths import ProjectPaths
from review_capacity.provenance import code_provenance
from review_capacity.training.declaration import load_declaration
from review_capacity.training.preflight import (
    MODELS,
    RUN_TYPE,
    GpuMeter,
    build_record,
    run_preflight,
    write_preflight,
)
from review_capacity.training.sampler import generate_candidates

logger = logging.getLogger(__name__)

LOG_NAME = RUN_LOG_NAME  # run.log, as for every run of this package


def output_directory(paths: ProjectPaths, started: str, output: Path | None) -> Path:
    """The run's new directory: ``output``, or ``outputs/neural_preflight/<started>``."""
    chosen = output if output is not None else default_directory(paths.root, RUN_TYPE, started)
    return check_new_directory(chosen, paths.root)


def run_preflight_command(
    paths: ProjectPaths,
    features: FeatureSpec,
    directory: Path,
    started: str,
    preprocessing: Path,
) -> dict[str, Any]:
    """Run the preflight of both neural models and write ``preflight.json`` to ``directory``."""
    check_new_directory(directory, paths.root, log_opened=True)
    declaration, declaration_file = load_declaration(paths)
    settings, settings_file = load_settings_with_snapshot(paths)
    artifact = load_neural_preprocessing(preprocessing, features, declaration.preprocessing_seed)
    devices = {settings.training_device(model) for model in MODELS}
    if devices != {"gpu"}:
        raise EnvironmentMismatchError(
            f"The preflight measures GPU memory; the neural models train on {sorted(devices)}"
        )
    lists = generate_candidates(declaration)
    candidates = {model: lists[model].candidates for model in MODELS}
    models = run_preflight(
        declaration,
        settings,
        artifact.layout,
        candidates,
        device="gpu",
        meter=GpuMeter(),
        seed=settings.main_seed,
    )
    identity = {
        "run_id": started,
        "run_type": RUN_TYPE,
        "code": code_provenance(paths.root),
        "declaration_sha256": declaration_file.sha256,
        "settings_sha256": settings_file.sha256,
        "artifact_sha256": artifact.artifact_sha256,
        "environment": environment_record(),
    }
    record = build_record(declaration, identity, models)
    path = write_preflight(directory, record)
    logger.info("Neural preflight written to %s", path)
    return record
