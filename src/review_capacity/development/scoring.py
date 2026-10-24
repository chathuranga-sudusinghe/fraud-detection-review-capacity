"""Scoring the selected model on V1 and V2 (DR-013 §5; DR-015 §9; Issues #134 and #151).

The selected main-seed model is loaded from its saved trial directory (``development.trial``) and
scores the whole of a development partition in one call: a tree-based model and the reference on
the CPU, a neural model on the GPU through its float32 prediction path with the evaluation batch
of its hyperparameters. A reloaded neural model is first checked against the declaration and the
recorded batch plan: its saved hyperparameters must be those of the selected candidate with the
declared fixed values, and its saved batching must be the plan of the preflight. The
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
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

from review_capacity.config import SECONDS_PER_DAY, Device, FeatureSpec, ModelSettings
from review_capacity.development.inputs import PartitionRows, ScoringInputs
from review_capacity.development.memory import MemorySetup
from review_capacity.development.run_output import RunDirectory
from review_capacity.development.subjects import (
    NEURAL,
    REFERENCE,
    kind_of,
    scoring_device,
)
from review_capacity.development.trial import (
    NEURAL_MODELS,
    TREE_MODELS,
    TrialOutcome,
    run_file,
)
from review_capacity.errors import DataValidationError, ModelError, SpecificationError
from review_capacity.files import write_atomically
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import log_step
from review_capacity.models.base import Keys, Scores, checked_scores
from review_capacity.models.catboost_model import CatBoostModel
from review_capacity.models.lightgbm_model import LightGBMModel
from review_capacity.models.logistic_regression_model import LogisticRegressionReference
from review_capacity.models.neural_training import NeuralModel
from review_capacity.models.registry import Configuration, Subject
from review_capacity.models.run_record import jsonable
from review_capacity.training.declaration import REFERENCE_MODEL, TrainingDeclaration
from review_capacity.training.preflight import fit_hyperparameters

if TYPE_CHECKING:
    from review_capacity.development.neural_run import NeuralRun

logger = logging.getLogger(__name__)

SCORES_DIR = "scores"
SCORE_COLUMNS = ("TransactionID", "score")
SCORING_DEVICE = "cpu"  # the tree-based models and the reference score on the CPU (DR-013 §5)
MODEL_ROLE = "selected"

type ScoredModel = LightGBMModel | CatBoostModel | NeuralModel | LogisticRegressionReference


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
    memory: MemorySetup | None = None
    neural: NeuralRun | None = None

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
    role: str = MODEL_ROLE  # which model: the selected one, or a refit (DR-015 §4)
    seed: int | None = None  # its training seed; None: the run's main seed
    tag: str = MODEL_ROLE  # names its files: "selected", or e.g. "seed_refit-s43"


def _neural(context: ScoringContext) -> NeuralRun:
    if context.neural is None:
        raise SpecificationError("The neural models and the reference need the neural inputs")
    return context.neural


def verify_neural_model(
    context: ScoringContext, configuration: Subject, selected: TrialOutcome, model: NeuralModel
) -> None:
    """Stop unless the reloaded neural model is the selected trial as declared: its saved
    hyperparameters are the candidate's with the declared fixed values, early stopping and
    batches, and its saved batching is the recorded plan of that candidate."""
    name = configuration.model.name
    expected = jsonable(fit_hyperparameters(context.declaration, name, selected.hyperparameters))
    parameters = model.parameters
    if parameters["requested"] != expected:
        raise ModelError(
            f"{configuration.name}: the saved hyperparameters differ from the selected candidate "
            "and the declaration"
        )
    plan = _neural(context).plans[name][selected.trial]
    batching = parameters["resolved"]["batching"]
    if (batching["effective_batch"], batching["micro_batch"]) != (plan.effective, plan.micro):
        raise ModelError(
            f"{configuration.name}: the saved batching differs from the recorded batch plan "
            f"({plan.effective} rows in micro-batches of {plan.micro})"
        )


def load_selected(
    context: ScoringContext,
    configuration: Subject,
    selected: TrialOutcome,
    *,
    directory: Path | None = None,
    seed: int | None = None,
    device: Device | None = None,
) -> ScoredModel:
    """The selected trial's saved model, reloaded on its training device. A refit of the selected
    setting (``directory``, its ``seed`` and ``device``) is reloaded in the same way."""
    kind, name = kind_of(configuration), configuration.name
    where = selected.model_directory if directory is None else directory
    with log_step(logger, f"Loading the {name} saved in {where.name} (trial {selected.trial})"):
        if kind == REFERENCE:
            fixed = dict(context.declaration.space(REFERENCE_MODEL).fixed)
            return LogisticRegressionReference.load(where, _neural(context).artifact, fixed)
        assert isinstance(configuration, Configuration)
        training = device or context.settings.training_device(configuration.model.name)
        trained_with = context.seed if seed is None else seed
        if kind == NEURAL:
            neural_model = NEURAL_MODELS[configuration.model.name].load(
                where,
                configuration,
                trained_with,
                settings=context.settings,
                layout=_neural(context).artifact.layout,
                device=training,
                artifact_sha256=_neural(context).artifact_sha256,
            )
            verify_neural_model(context, configuration, selected, neural_model)
            return neural_model
        model_class, _ = TREE_MODELS[configuration.model.name]
        return model_class.load(
            where,
            configuration,
            trained_with,
            settings=context.settings,
            features=context.features,
            device=training,
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


def score_path(run: RunDirectory, configuration: str, label: str, tag: str = MODEL_ROLE) -> Path:
    return run.path / SCORES_DIR / f"{configuration}-{tag}-{label}.parquet"


class HoldsRunFiles(Protocol):
    """A run directory, open (``RunDirectory``) or loaded (``LoadedRun``): it names its path."""

    @property
    def path(self) -> Path: ...


def read_scores(run: HoldsRunFiles, file: Mapping[str, str], rows: PartitionRows) -> Scores:
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
    configuration: Subject,
    fit: Mapping[str, str],
    label: str,
    rows: PartitionRows,
    scores: Scores,
    file: Mapping[str, str],
    seconds: float,
    batch_rows: int | None = None,
    *,
    role: str = MODEL_ROLE,
    seed: int | None = None,
) -> dict[str, Any]:
    days = rows.times // SECONDS_PER_DAY
    return {
        "record": "score",
        "schema_version": 1,
        "identity": dict(context.identity),
        "configuration": configuration.name,
        "model_role": role,
        "seed": context.seed if seed is None else seed,
        "fit": dict(fit),
        "partition": label,
        "first_day": int(days.min()),
        "last_day": int(days.max()),
        "rows": rows.rows,
        "file": dict(file),
        "format": "parquet",
        "columns": list(SCORE_COLUMNS),
        "score_dtype": "float64",
        "device": scoring_device(context.settings, configuration),
        "batch_rows": batch_rows,
        "min_score": float(scores.min()),
        "max_score": float(scores.max()),
        "distinct_scores": len(np.unique(scores)),
        "seconds": seconds,
    }


def score_partition(
    context: ScoringContext,
    configuration: Subject,
    selected: TrialOutcome,
    model: ScoredModel,
    label: str,
) -> ScoredPartition:
    """Score the whole partition ``label`` (``"V1"`` or ``"V2"``) with the selected model in one
    call, write the local score file and the ``score`` record. ``seconds`` is this call's
    wall-clock time; the benchmarked scoring time is measured separately
    (``development.cost_timing``)."""
    return score_model(context, configuration, selected.fit, model, label)


def score_model(
    context: ScoringContext,
    configuration: Subject,
    fit: Mapping[str, str],
    model: ScoredModel,
    label: str,
    *,
    role: str = MODEL_ROLE,
    seed: int | None = None,
    tag: str = MODEL_ROLE,
) -> ScoredPartition:
    """Score ``label`` with a model fitted by the fit record ``fit``: the selected model, or a
    refit (``role``, ``seed``; ``tag`` names its score file and record)."""
    run, name = context.run, configuration.name
    rows = context.inputs.partition(label)
    with log_step(logger, f"Scoring {label} with the {tag} {name} ({rows.rows} rows)"):
        batch_rows: int | None = None
        started = time.perf_counter()
        if isinstance(model, NeuralModel):
            scores = model.predict(_neural(context).scoring_inputs(label))
            batch_rows = model.parameters["prediction"]["eval_batch_size"]
        else:
            scores = model.predict(rows.features)
        seconds = time.perf_counter() - started
    frame = pd.DataFrame({SCORE_COLUMNS[0]: rows.ids, SCORE_COLUMNS[1]: scores})
    path = score_path(run, name, label, tag)
    write_atomically(path, lambda target: frame.to_parquet(target, index=False))
    file = run_file(run, path)
    read_scores(run, file, rows)  # the file as written pairs with the partition's rows
    record = score_record(
        context, configuration, fit, label, rows, scores, file, seconds, batch_rows,
        role=role, seed=seed,
    )  # fmt: skip
    record_path = run.write_record(record, name, tag, label)
    logger.info(
        "%s (%s) scored on %s: %d distinct scores", name, tag, label, record["distinct_scores"]
    )
    return ScoredPartition(
        name, label, rows, file, run_file(run, record_path), seconds, role, seed, tag
    )
