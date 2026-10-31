"""The ``declare-fn-boundaries`` run: compute the data-derived false-negative slice boundaries from
the development data, for entry into ``config/evaluation.yaml`` (Issue #211; protocol,
False-Negative Analysis).

Reads the verified Train and Validation 2 files only (``read_partitions`` cannot name the Test
period) and no Test row, label or score. It writes the ``boundaries`` block of the declaration,
with the exact values, to a new directory inside ``outputs/``; the values are entered into the
configuration file by a reviewed change, and the Test evaluation then compares what it recomputes
with the frozen values.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from review_capacity.config import DatasetSpec, FeatureSpec
from review_capacity.confirmatory.slices import AMOUNT
from review_capacity.development.inputs import TRAIN, V2, read_partitions
from review_capacity.development.run_output import check_new_directory, default_directory
from review_capacity.evaluation.declaration import load_declaration
from review_capacity.evaluation.false_negatives import (
    POOLED_VARIABLES,
    Boundaries,
    compute_boundaries,
)
from review_capacity.files import write_new_atomically
from review_capacity.logging_setup import log_step
from review_capacity.paths import ProjectPaths

logger = logging.getLogger(__name__)

RUN_TYPE = "fn_boundaries"
RESULT_FILE = "false_negative_boundaries.yaml"


def output_directory(paths: ProjectPaths, started: str, output: Path | None) -> Path:
    """The run's new directory: ``output``, or ``outputs/fn_boundaries/<started>``."""
    chosen = output if output is not None else default_directory(paths.root, RUN_TYPE, started)
    return check_new_directory(chosen, paths.root)


def render_yaml(boundaries: Boundaries) -> str:
    """The ``boundaries`` entry of ``false_negative`` as YAML, at the indentation of the file,
    with the exact edge values (shortest round-trip decimals) and quoted level names."""
    lines = ["  boundaries:", "    amount_edges:"]
    lines += [f"      - {edge!r}" for edge in boundaries.amount_edges]
    lines.append("    kept:")
    for name in POOLED_VARIABLES:
        lines.append(f"      {name}:")
        lines += [f"        {json.dumps(level)}: {n}" for level, n in boundaries.kept[name].items()]
    return "\n".join(lines) + "\n"


def run_declare(
    paths: ProjectPaths,
    features: FeatureSpec,
    dataset: DatasetSpec,
    directory: Path,
) -> Boundaries:
    """Compute and write the boundaries into the new ``directory``."""
    check_new_directory(directory, paths.root, log_opened=True)
    declaration, _ = load_declaration(paths)
    parts = read_partitions(paths, features, dataset, (TRAIN, V2))
    v2_frauds = parts[V2].features[parts[V2].labels == 1]
    with log_step(logger, "Computing the false-negative slice boundaries"):
        boundaries = compute_boundaries(
            declaration,
            parts[TRAIN].features[AMOUNT].to_numpy(dtype=np.float64),
            {
                name: np.array(v2_frauds[name].to_numpy(dtype=object), dtype=np.str_)
                for name in POOLED_VARIABLES
            },
        )
    text = render_yaml(boundaries)
    write_new_atomically(directory / RESULT_FILE, text)
    logger.info("Boundaries (%d V2 frauds):\n%s", len(v2_frauds), text)
    logger.info("Written to %s", directory / RESULT_FILE)
    return boundaries
