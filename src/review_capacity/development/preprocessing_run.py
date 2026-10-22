"""The ``fit-neural-preprocessing`` run: fit the shared neural preprocessing once on Train, write
the artifact and its Validation 1 and Validation 2 diagnostics (DR-012; Issue #149).

Reads the verified processed development files only (Train, Validation 1, Validation 2; the Test
period is never read) and writes inside ``outputs/``. The artifact is reloaded and compared with
the fitted one before the run ends, so the file that later runs link to is the file that was
checked. The run record names the artifact's SHA-256, which every neural trial and refit will
record.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from review_capacity.config import DatasetSpec, FeatureSpec
from review_capacity.development.inputs import TRAIN, V1, V2, read_partitions
from review_capacity.development.run_output import LOG_NAME as RUN_LOG_NAME
from review_capacity.development.run_output import check_new_directory, default_directory
from review_capacity.errors import ModelError
from review_capacity.features.neural_preprocessing import (
    fit_neural_preprocessing,
    load_neural_preprocessing,
)
from review_capacity.files import write_json
from review_capacity.logging_setup import log_step
from review_capacity.paths import ProjectPaths
from review_capacity.provenance import code_provenance
from review_capacity.training.declaration import load_declaration

logger = logging.getLogger(__name__)

RUN_TYPE = "neural_preprocessing"
LOG_NAME = RUN_LOG_NAME  # run.log, as for every run of this package
RESULT_FILE = "result.json"


def output_directory(paths: ProjectPaths, started: str, output: Path | None) -> Path:
    """The run's new directory: ``output``, or ``outputs/neural_preprocessing/<started>``."""
    chosen = output if output is not None else default_directory(paths.root, RUN_TYPE, started)
    return check_new_directory(chosen, paths.root)


def run_fit(
    paths: ProjectPaths,
    features: FeatureSpec,
    dataset: DatasetSpec,
    directory: Path,
    started: str,
) -> dict[str, Any]:
    """Fit, write, reload and check the artifact; write the diagnostics and the run record."""
    check_new_directory(directory, paths.root, log_opened=True)
    declaration, declaration_file = load_declaration(paths)
    seed = declaration.preprocessing_seed
    parts = read_partitions(paths, features, dataset, (TRAIN, V1, V2))
    train = parts[TRAIN]
    artifact = fit_neural_preprocessing(train.features, train.ids, train.times, features, seed)
    sha256 = artifact.save(directory)
    with log_step(logger, "Checking the written artifact"):
        reloaded = load_neural_preprocessing(
            directory, features, seed, train_ids_sha256=train.ordered_ids_sha256
        )
        if reloaded.artifact_sha256 != sha256:
            raise ModelError("The reloaded artifact differs from the fitted one")
    layout = artifact.layout
    observed = train.features[list(artifact.numeric_inputs)].notna().sum()
    if int(observed.min()) < artifact.requested_quantiles:
        logger.warning(
            "%s has %d non-missing Train values, fewer than the %d requested quantiles",
            observed.idxmin(),
            int(observed.min()),
            artifact.requested_quantiles,
        )
    result: dict[str, Any] = {
        "run_id": started,
        "code": code_provenance(paths.root),
        "declaration_sha256": declaration_file.sha256,
        "artifact_sha256": sha256,
        "seed": seed,
        "train": {"rows": train.rows, "ordered_ids_sha256": train.ordered_ids_sha256},
        "inputs": {
            "numeric": len(artifact.numeric_inputs),
            "indicators": len(artifact.indicators),
            "categorical": len(artifact.categorical_inputs),
            "network_numeric": len(layout.numeric),
            "total": len(layout.numeric) + len(layout.categorical),
        },
        "quantiles": {
            "requested": artifact.requested_quantiles,
            "actual": artifact.actual_quantiles,
            "fewest_non_missing_train_values": int(observed.min()),
            "input_with_fewest": str(observed.idxmin()),
        },
        "diagnostics": {},
    }
    for stem in (V1, V2):
        part = parts[stem]
        with log_step(logger, f"Applying the artifact to {stem}"):
            inputs = artifact.to_neural_inputs(part.features)
            diagnostics = artifact.diagnostics(stem, part.features)
        file_name = f"diagnostics_{stem}.json"
        write_json(directory / file_name, diagnostics)
        outside = [e for e in diagnostics["numeric"] if e["below"] + e["above"] > 0]
        result["diagnostics"][stem] = {
            "file": file_name,
            "rows": inputs.rows,
            "ordered_ids_sha256": part.ordered_ids_sha256,
            "numeric_inputs_with_values_outside_the_fitted_range": len(outside),
            "missing_without_indicator": diagnostics["missing_without_indicator_total"],
        }
    write_json(directory / RESULT_FILE, result)
    logger.info(
        "Neural preprocessing: %d numeric + %d indicators + %d categorical = %d inputs; sha256 %s",
        result["inputs"]["numeric"],
        result["inputs"]["indicators"],
        result["inputs"]["categorical"],
        result["inputs"]["total"],
        sha256,
    )
    return result
