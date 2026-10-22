"""One search trial of a tree-based configuration (DR-015 §1, §2, §3 and §5; Issue #133).

A trial fits one candidate on Train with early stopping on the shared V1 objective (LightGBM and
CatBoost through the adapters of Issue #123), keeps the earliest best checkpoint, saves the model
and writes four files into the run directory:

* ``models/<configuration>/trial-<n>/``: the model, its manifest and its model-run record;
* a ``fit`` record, a ``checkpoint`` record (the retained model: a tree model's saved trees are
  its earliest best checkpoint) and a ``trial`` record with the V1 objective after every round.

A trial that fails is recorded as failed (its ``fit`` and ``trial`` records say so) and the error
is raised: a search never continues past a failed trial. The neural models and the
logistic-regression reference connect through their own children of Issue #96.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from review_capacity.config import FeatureSpec, ModelSettings
from review_capacity.development.inputs import SearchInputs
from review_capacity.development.run_output import RunDirectory
from review_capacity.errors import (
    FileAccessError,
    ModelError,
    ReviewCapacityError,
    SpecificationError,
)
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import log_step
from review_capacity.models import catboost_model, lightgbm_model
from review_capacity.models.catboost_model import CatBoostModel
from review_capacity.models.device import LightGBMIdentity
from review_capacity.models.lightgbm_model import LightGBMModel
from review_capacity.models.registry import Configuration, Family
from review_capacity.models.run_record import (
    SettingsSnapshot,
    model_run_record,
    write_model_run_record,
)
from review_capacity.models.v1_objective import TrainingHistory, V1Objective
from review_capacity.training.declaration import TrainingDeclaration
from review_capacity.training.sampler import Candidate

logger = logging.getLogger(__name__)

MODELS_DIR = "models"
MODEL_RECORD_NAME = "model_run.json"
UNIT = "round"  # the training unit of the tree-based models

# The tree-based models of DR-011 and the file each saves its model in.
TREE_MODELS: dict[str, tuple[type[LightGBMModel] | type[CatBoostModel], str]] = {
    "lightgbm": (LightGBMModel, lightgbm_model.MODEL_FILE),
    "catboost": (CatBoostModel, catboost_model.MODEL_FILE),
}


@dataclass(frozen=True)
class SearchContext:
    """Everything a search shares: the run directory and its identity, the declaration and the
    model settings as loaded, the verified Train and V1 inputs, the V1 objective built from V1
    once, the environment and code records, the verified LightGBM build and the conditions of
    the timing session."""

    run: RunDirectory
    identity: Mapping[str, Any]
    declaration: TrainingDeclaration
    settings: ModelSettings
    settings_snapshot: SettingsSnapshot
    features: FeatureSpec
    inputs: SearchInputs
    objective: V1Objective
    environment: Mapping[str, Any]
    code: Mapping[str, Any]
    lightgbm_build: LightGBMIdentity | None
    session: Mapping[str, Any]

    @property
    def seed(self) -> int:
        """The main training seed, used by every search trial (DR-015 §4)."""
        seed: int = self.identity["seeds"]["main"]
        return seed


@dataclass(frozen=True)
class TrialOutcome:
    """A completed trial: what the selection and the ``configuration`` record need."""

    configuration: str
    trial: int
    hyperparameters: dict[str, Any]
    objective: float
    best_checkpoint: int
    stopped_at: int
    stopping_reason: str
    fit_seconds: float
    fitted_iterations: int
    model_directory: Path
    fit: dict[str, str]
    checkpoint: dict[str, str]
    record: dict[str, str]


def run_file(run: RunDirectory, path: Path) -> dict[str, str]:
    """A ``{path, sha256}`` reference to a file inside the run directory."""
    return {"path": path.relative_to(run.path).as_posix(), "sha256": file_sha256(path)}


def trial_directory(run: RunDirectory, configuration: str, trial: int) -> Path:
    return run.path / MODELS_DIR / configuration / f"trial-{trial:02d}"


def hyperparameters(
    declaration: TrainingDeclaration, configuration: Configuration, candidate: Candidate
) -> dict[str, Any]:
    """The candidate's searched values with the model's fixed values of the declaration."""
    fixed = dict(declaration.space(configuration.model.name).fixed)
    clash = sorted(fixed.keys() & candidate.keys())
    if clash:
        raise SpecificationError(
            f"{configuration.name}: a candidate sets fixed declaration values {clash}"
        )
    return {**fixed, **candidate}


def tree_model(
    context: SearchContext, configuration: Configuration
) -> LightGBMModel | CatBoostModel:
    """A new, unfitted tree model on its training device (DR-013 §1)."""
    entry = TREE_MODELS.get(configuration.model.name)
    if entry is None or configuration.family is not Family.TREE:
        raise SpecificationError(
            f"{configuration.name}: the search core runs the tree-based models only"
        )
    model_class, _ = entry
    return model_class(
        configuration,
        context.seed,
        settings=context.settings,
        features=context.features,
        device=context.settings.training_device(configuration.model.name),
    )


def _train_summary(context: SearchContext) -> dict[str, Any]:
    train = context.inputs.train
    return {
        "rows": train.rows,
        "frauds": train.frauds,
        "transaction_ids_sha256": train.ordered_ids_sha256,
        "file_sha256": train.provenance.sha256,
    }


def _fit_record(
    context: SearchContext,
    configuration: Configuration,
    trial: int,
    *,
    model: LightGBMModel | CatBoostModel | None,
    files: tuple[dict[str, str], dict[str, str]] | None,
    seconds: float | None,
    error: str | None,
) -> dict[str, Any]:
    weights = context.inputs.weights(configuration.weighting)
    return {
        "record": "fit",
        "schema_version": 1,
        "identity": dict(context.identity),
        "configuration": configuration.name,
        "purpose": "search_trial",
        "trial": trial,
        "seed": context.seed,
        "device": context.settings.training_device(configuration.model.name),
        "fresh_process": False,
        "early_stopping": True,
        "length": None,
        "fitted_iterations": None if model is None else model.fitted_iterations,
        "train": _train_summary(context),
        "positive_weight": weights.positive,
        "preprocessing": None,
        "preflight": None,
        "model": None if files is None else files[0],
        "model_record": None if files is None else files[1],
        "fit_seconds": seconds,
        "status": "completed" if error is None else "failed",
        "error": error,
    }


def _checkpoint_record(
    context: SearchContext,
    configuration: Configuration,
    trial: int,
    history: TrainingHistory,
    files: tuple[dict[str, str], dict[str, str]],
) -> dict[str, Any]:
    return {
        "record": "checkpoint",
        "schema_version": 1,
        "identity": dict(context.identity),
        "configuration": configuration.name,
        "trial": trial,
        "unit": UNIT,
        "index": history.best_checkpoint,
        "objective": history.best_objective,
        "model": files[0],
        "model_record": files[1],
    }


def _trial_record(
    context: SearchContext,
    configuration: Configuration,
    trial: int,
    candidate: Candidate,
    candidate_file: Mapping[str, str],
    *,
    history: TrainingHistory | None,
    fit: dict[str, str],
    checkpoint: dict[str, str] | None,
    seconds: float | None,
    error: str | None,
) -> dict[str, Any]:
    stopping = context.declaration.tree_stopping
    patience = stopping.patience if history is None else history.patience
    maximum = stopping.maximum if history is None else history.maximum
    done = history is not None
    return {
        "record": "trial",
        "schema_version": 1,
        "identity": dict(context.identity),
        "configuration": configuration.name,
        "trial": trial,
        "candidate_list": dict(candidate_file),
        "hyperparameters": dict(candidate),
        "seed": context.seed,
        "device": context.settings.training_device(configuration.model.name),
        "unit": UNIT,
        "objective_k": context.declaration.v1_daily_k,
        "patience": patience,
        "maximum": maximum,
        "best_checkpoint": history.best_checkpoint if history else None,
        "best_objective": history.best_objective if history else None,
        "stopped_at": history.stopped_at if history else None,
        "stopping_reason": history.stopping_reason if history else None,
        "runtime_seconds": seconds,
        "checkpoints": []
        if history is None
        else [
            {
                "checkpoint": row.checkpoint,
                "objective": row.objective,
                "improved": row.improved,
                "best_checkpoint": row.best_checkpoint,
                "seconds": row.seconds,
            }
            for row in history.checkpoints
        ],
        "fit": fit,
        "checkpoint": checkpoint,
        "status": "completed" if done and error is None else "failed",
        "error": error,
    }


def _record_failure(
    context: SearchContext,
    configuration: Configuration,
    trial: int,
    candidate: Candidate,
    candidate_file: Mapping[str, str],
    error: ReviewCapacityError,
) -> None:
    """Write the failed ``fit`` and ``trial`` records of a trial that raised ``error``."""
    run, name = context.run, configuration.name
    message = f"{type(error).__name__}: {error}"
    failed = _fit_record(
        context, configuration, trial, model=None, files=None, seconds=None, error=message
    )
    fit_path = run.write_record(failed, name, f"trial{trial}")
    run.write_record(
        _trial_record(
            context,
            configuration,
            trial,
            candidate,
            candidate_file,
            history=None,
            fit=run_file(run, fit_path),
            checkpoint=None,
            seconds=None,
            error=message,
        ),
        name,
        trial,
    )


def run_trial(
    context: SearchContext,
    configuration: Configuration,
    trial: int,
    candidate: Candidate,
    candidate_file: Mapping[str, str],
) -> TrialOutcome:
    """Fit, save and record trial ``trial`` (1 = the reference configuration) of ``configuration``.

    ``candidate_file`` is the run-directory reference of the recorded candidate list.
    """
    run, name = context.run, configuration.name
    model = tree_model(context, configuration)
    settings = hyperparameters(context.declaration, configuration, candidate)
    train = context.inputs.train_data(configuration.weighting)
    valid = context.inputs.v1_data()
    directory = trial_directory(run, name, trial)
    model_file = TREE_MODELS[configuration.model.name][1]
    with log_step(logger, f"Trial {trial} of {name}"):
        try:
            started = time.perf_counter()
            model.fit(train, valid, settings, objective=context.objective)
            seconds = time.perf_counter() - started
            history = model.training_history
            if history is None:
                raise ModelError(f"{name}: the fit returned no V1 history")
            try:
                directory.mkdir(parents=True)
            except OSError as error:
                raise FileAccessError(
                    f"{directory}: cannot be created ({error.strerror})"
                ) from error
            model.save(directory)
            build = context.lightgbm_build if configuration.model.name == "lightgbm" else None
            write_model_run_record(
                directory / MODEL_RECORD_NAME,
                model_run_record(
                    model=model,
                    settings=context.settings_snapshot,
                    hyperparameters=settings,
                    lightgbm_build=build,
                    environment=context.environment,
                    code=context.code,
                ),
            )
            files = (
                run_file(run, directory / model_file),
                run_file(run, directory / MODEL_RECORD_NAME),
            )
        except ReviewCapacityError as error:
            _record_failure(context, configuration, trial, candidate, candidate_file, error)
            raise
        completed = _fit_record(
            context, configuration, trial, model=model, files=files, seconds=seconds, error=None
        )
        fit_path = run.write_record(completed, name, f"trial{trial}")
        checkpoint_path = run.write_record(
            _checkpoint_record(context, configuration, trial, history, files), name, f"trial{trial}"
        )
        fit, checkpoint = run_file(run, fit_path), run_file(run, checkpoint_path)
        record_path = run.write_record(
            _trial_record(
                context,
                configuration,
                trial,
                candidate,
                candidate_file,
                history=history,
                fit=fit,
                checkpoint=checkpoint,
                seconds=seconds,
                error=None,
            ),
            name,
            trial,
        )
    logger.info(
        "%s trial %d: V1 objective %.6f at round %d (%s after %d rounds)",
        name,
        trial,
        history.best_objective,
        history.best_checkpoint,
        history.stopping_reason,
        history.stopped_at,
    )
    return TrialOutcome(
        configuration=name,
        trial=trial,
        hyperparameters=dict(candidate),
        objective=history.best_objective,
        best_checkpoint=history.best_checkpoint,
        stopped_at=history.stopped_at,
        stopping_reason=history.stopping_reason,
        fit_seconds=seconds,
        fitted_iterations=model.fitted_iterations,
        model_directory=directory,
        fit=fit,
        checkpoint=checkpoint,
        record=run_file(run, record_path),
    )
