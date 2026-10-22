"""Scoring the selected tree-based model on V1 and V2 (DR-013 §5; DR-015 §9; Issue #134).

The selected main-seed model is loaded from its saved trial directory (``development.trial``) and
scores the whole of a development partition in one call, on the CPU prediction backend. The
row-level scores are written to a local Parquet file inside the run directory (``scores/``;
gitignored, never promoted) with the ``TransactionID`` of every row, and a ``score`` record
describes the file: its rows, days, score range and distinct scores, device and scoring time. The
scores are the model's raw fraud probabilities; calibration is separate.

**Row alignment.** The scores, the labels and the transaction amounts of a partition describe the
same rows only if their ordered ``TransactionID`` sequences are identical. ``read_scores`` reads a
score file back and ``check_sequence`` stops the run on any difference, so that no evaluation can
pair a score with another transaction's label or amount.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from review_capacity.config import SECONDS_PER_DAY, FeatureSpec, ModelSettings
from review_capacity.development.inputs import PartitionRows, ScoringInputs
from review_capacity.development.run_output import RunDirectory
from review_capacity.development.trial import TREE_MODELS, TrialOutcome, run_file
from review_capacity.errors import DataValidationError, SpecificationError
from review_capacity.files import write_atomically
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import log_step
from review_capacity.models.base import Keys, Scores, checked_scores
from review_capacity.models.catboost_model import CatBoostModel
from review_capacity.models.lightgbm_model import LightGBMModel
from review_capacity.models.registry import Configuration, Family
from review_capacity.training.declaration import TrainingDeclaration

logger = logging.getLogger(__name__)

SCORES_DIR = "scores"
SCORE_COLUMNS = ("TransactionID", "score")
SCORING_DEVICE = "cpu"  # the tree-based models score on the CPU (DR-013 §5)
MODEL_ROLE = "selected"


@dataclass(frozen=True)
class ScoringContext:
    """What scoring and evaluation share: the run directory and its identity, the declaration and
    model settings as loaded, the verified V1 and V2 rows and the timing session conditions."""

    run: RunDirectory
    identity: Mapping[str, Any]
    declaration: TrainingDeclaration
    settings: ModelSettings
    features: FeatureSpec
    inputs: ScoringInputs
    session: Mapping[str, Any]

    @property
    def seed(self) -> int:
        """The main training seed of the selected model (DR-015 §4)."""
        seed: int = self.identity["seeds"]["main"]
        return seed


@dataclass(frozen=True)
class ScoredPartition:
    """A scored development partition: its rows and the references of its score file and record."""

    configuration: str
    label: str
    rows: PartitionRows
    file: dict[str, str]
    record: dict[str, str]
    seconds: float


def load_selected(
    context: ScoringContext, configuration: Configuration, selected: TrialOutcome
) -> LightGBMModel | CatBoostModel:
    """The selected trial's saved model, reloaded on its training device."""
    entry = TREE_MODELS.get(configuration.model.name)
    if entry is None or configuration.family is not Family.TREE:
        raise SpecificationError(
            f"{configuration.name}: only the tree-based models are scored here"
        )
    model_class, _ = entry
    with log_step(logger, f"Loading the selected {configuration.name} (trial {selected.trial})"):
        return model_class.load(
            selected.model_directory,
            configuration,
            context.seed,
            settings=context.settings,
            features=context.features,
            device=context.settings.training_device(configuration.model.name),
        )


def check_sequence(expected: Keys, actual: Keys, what: str) -> None:
    """Stop unless ``actual`` is exactly the ordered ``TransactionID`` sequence ``expected``."""
    if expected.shape != actual.shape:
        raise DataValidationError(
            f"{what}: {len(actual)} identifiers, the partition has {len(expected)}"
        )
    differing = np.flatnonzero(expected != actual)
    if len(differing):
        row = int(differing[0])
        raise DataValidationError(
            f"{what}: the identifier sequence differs from the partition's at row {row} "
            f"({int(actual[row])} instead of {int(expected[row])}); {len(differing)} rows differ"
        )


def score_path(run: RunDirectory, configuration: str, label: str) -> Path:
    return run.path / SCORES_DIR / f"{configuration}-{MODEL_ROLE}-{label}.parquet"


def read_scores(run: RunDirectory, file: Mapping[str, str], rows: PartitionRows) -> Scores:
    """The scores of a score file, after checking its columns, its SHA-256 and that its ordered
    ``TransactionID`` sequence is that of ``rows`` (and so of the labels and amounts)."""
    path = run.path / file["path"]
    if file_sha256(path) != file["sha256"]:
        raise DataValidationError(f"{path}: differs from the score file the record lists")
    frame = pd.read_parquet(path)
    if tuple(frame.columns) != SCORE_COLUMNS:
        raise DataValidationError(f"{path}: columns {tuple(frame.columns)}, not {SCORE_COLUMNS}")
    check_sequence(rows.ids, frame["TransactionID"].to_numpy(dtype=np.int64), f"{path.name} scores")
    return checked_scores(frame["score"].to_numpy(), rows.rows, path.name)


def score_record(
    context: ScoringContext,
    configuration: Configuration,
    selected: TrialOutcome,
    label: str,
    rows: PartitionRows,
    scores: Scores,
    file: Mapping[str, str],
    seconds: float,
) -> dict[str, Any]:
    days = rows.times // SECONDS_PER_DAY
    return {
        "record": "score",
        "schema_version": 1,
        "identity": dict(context.identity),
        "configuration": configuration.name,
        "model_role": MODEL_ROLE,
        "seed": context.seed,
        "fit": dict(selected.fit),
        "partition": label,
        "first_day": int(days.min()),
        "last_day": int(days.max()),
        "rows": rows.rows,
        "file": dict(file),
        "format": "parquet",
        "columns": list(SCORE_COLUMNS),
        "score_dtype": "float64",
        "device": SCORING_DEVICE,
        "batch_rows": None,
        "min_score": float(scores.min()),
        "max_score": float(scores.max()),
        "distinct_scores": len(np.unique(scores)),
        "seconds": seconds,
    }


def score_partition(
    context: ScoringContext,
    configuration: Configuration,
    selected: TrialOutcome,
    model: LightGBMModel | CatBoostModel,
    label: str,
) -> ScoredPartition:
    """Score the whole partition ``label`` (``"V1"`` or ``"V2"``) in one call, write the local
    score file and the ``score`` record. ``seconds`` is this call's wall-clock time; the
    benchmarked scoring time is measured separately (``development.cost_timing``)."""
    run, name = context.run, configuration.name
    rows = context.inputs.partition(label)
    with log_step(logger, f"Scoring {label} with the selected {name} ({rows.rows} rows)"):
        started = time.perf_counter()
        scores = model.predict(rows.features)
        seconds = time.perf_counter() - started
    frame = pd.DataFrame({SCORE_COLUMNS[0]: rows.ids, SCORE_COLUMNS[1]: scores})
    path = score_path(run, name, label)
    write_atomically(path, lambda target: frame.to_parquet(target, index=False))
    file = run_file(run, path)
    read_scores(run, file, rows)  # the file as written pairs with the partition's rows
    record = score_record(context, configuration, selected, label, rows, scores, file, seconds)
    record_path = run.write_record(record, name, MODEL_ROLE, label)
    logger.info("%s scored on %s: %d distinct scores", name, label, record["distinct_scores"])
    return ScoredPartition(name, label, rows, file, run_file(run, record_path), seconds)
