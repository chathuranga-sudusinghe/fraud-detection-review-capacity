"""One search trial of a tree-based configuration (DR-015 §1, §2, §3 and §5; Issue #133).

A trial fits one candidate on Train with early stopping on the shared V1 objective (LightGBM and
CatBoost through the adapters of Issue #123), keeps the earliest best checkpoint, saves the model
and writes four files into the run directory:

* ``models/<configuration>/trial-<n>/``: the model, its manifest and its model-run record;
* a ``fit`` record, a ``checkpoint`` record (the retained model: a tree model's saved trees are
  its earliest best checkpoint) and a ``trial`` record with the V1 objective after every round.

The neural models fit through the neural loop (the shared preprocessing artifact, the recorded
batch plan of the candidate, epochs) and keep their best epoch the same way; their ``fit`` records
link to the ``preprocessing`` and ``preflight`` records. The logistic-regression reference fits
once to convergence on the artifact's matrix: its V1 objective is computed after the fit, its
trial has no checkpoints and its unit is ``convergence`` (Issue #151).

A trial that fails is recorded as failed (its ``fit`` and ``trial`` records say so) and the error
is raised: a search never continues past a failed trial.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from review_capacity.config import FeatureSpec, ModelSettings
from review_capacity.development.inputs import SearchInputs
from review_capacity.development.memory import MemorySetup, PhaseMemory, run_measured
from review_capacity.development.run_output import RunDirectory
from review_capacity.development.startup import Startup
from review_capacity.development.subjects import (
    MODEL_FILES,
    NEURAL,
    REFERENCE,
    TREE,
    kind_of,
    training_device,
)
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
from review_capacity.models.ft_transformer_model import FTTransformerModel
from review_capacity.models.lightgbm_model import LightGBMModel
from review_capacity.models.logistic_regression_model import LogisticRegressionReference
from review_capacity.models.neural_training import NeuralModel
from review_capacity.models.registry import Configuration, Family, Subject, Weighting
from review_capacity.models.resnet_model import ResNetModel
from review_capacity.models.run_record import (
    RECORD_SCHEMA_VERSION,
    SettingsSnapshot,
    jsonable,
    model_run_record,
    write_model_run_record,
)
from review_capacity.models.v1_objective import TrainingHistory, V1Objective
from review_capacity.training.declaration import REFERENCE_MODEL, TrainingDeclaration
from review_capacity.training.preflight import fit_hyperparameters
from review_capacity.training.sampler import Candidate
from review_capacity.training.timing import Clock, gpu_clock, timed

if TYPE_CHECKING:
    from review_capacity.development.neural_run import NeuralRecords, NeuralRun

logger = logging.getLogger(__name__)

MODELS_DIR = "models"
MODEL_RECORD_NAME = "model_run.json"
UNIT = "round"  # the training unit of the tree-based models
UNITS = {TREE: "round", NEURAL: "epoch", REFERENCE: "convergence"}
FIT_MEMORY_SCOPE = (
    "the trial's fit, from its fit-specific data containers to the fitted model, with the Train "
    "and V1 inputs and the earlier trials' retained results resident; model saving is outside"
)
NEURAL_FIT_MEMORY_SCOPE = (
    "the trial's fit, from its first epoch to the best-epoch model, with the Train, V1 and V2 "
    "network inputs and the earlier trials' retained results resident; the V1 evaluations of the "
    "epochs are inside, model saving is outside"
)
REFERENCE_FIT_MEMORY_SCOPE = (
    "the trial's fit: the construction of the Train matrix, the solver's convergence and the "
    "V1 scoring that gives its objective excluded; the Train and V1 inputs resident; model "
    "saving is outside"
)
NEURAL_MODELS: dict[str, type[NeuralModel]] = {
    "resnet": ResNetModel,
    "ft_transformer": FTTransformerModel,
}

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
    memory: MemorySetup | None = None
    startups: Mapping[str, Startup] = field(default_factory=dict)
    neural: NeuralRun | None = None
    neural_records: NeuralRecords | None = None

    @property
    def seed(self) -> int:
        """The main training seed, used by every search trial (DR-015 §4)."""
        seed: int = self.identity["seeds"]["main"]
        return seed


@dataclass(frozen=True)
class TrialOutcome:
    """A completed trial: what the selection and the ``configuration`` record need. A trial of
    the reference has the unit ``convergence``: its lengths are the solver's iterations and it has
    no retained-checkpoint record."""

    configuration: str
    trial: int
    hyperparameters: dict[str, Any]
    objective: float
    best_checkpoint: int
    stopped_at: int
    stopping_reason: str
    fit_seconds: float
    persistence_seconds: float
    fit_memory: PhaseMemory | None
    fitted_iterations: int
    model_directory: Path
    fit: dict[str, str]
    checkpoint: dict[str, str] | None
    record: dict[str, str]
    unit: str = UNIT


@dataclass
class _Fit:
    """A fitted trial before it is saved: the model, the hyperparameters it was fitted with and
    the V1 result."""

    model: Any
    settings: dict[str, Any]
    seconds: float
    memory: PhaseMemory | None
    history: TrainingHistory | None
    objective: float
    best_checkpoint: int
    stopped_at: int
    stopping_reason: str


def run_file(run: RunDirectory, path: Path) -> dict[str, str]:
    """A ``{path, sha256}`` reference to a file inside the run directory."""
    return {"path": path.relative_to(run.path).as_posix(), "sha256": file_sha256(path)}


def trial_directory(run: RunDirectory, configuration: str, trial: int) -> Path:
    return run.path / MODELS_DIR / configuration / f"trial-{trial:02d}"


def hyperparameters(
    declaration: TrainingDeclaration, configuration: Subject, candidate: Candidate
) -> dict[str, Any]:
    """The candidate's searched values with the model's fixed values of the declaration."""
    fixed = dict(declaration.space(configuration.model.name).fixed)
    clash = sorted(fixed.keys() & candidate.keys())
    if clash:
        raise SpecificationError(
            f"{configuration.name}: a candidate sets fixed declaration values {clash}"
        )
    return {**fixed, **candidate}


def tree_model(context: SearchContext, configuration: Subject) -> LightGBMModel | CatBoostModel:
    """A new, unfitted tree model on its training device (DR-013 §1)."""
    entry = TREE_MODELS.get(configuration.model.name)
    if entry is None or configuration.family is not Family.TREE:
        raise SpecificationError(f"{configuration.name}: not a tree-based model")
    assert isinstance(configuration, Configuration)
    model_class, _ = entry
    return model_class(
        configuration,
        context.seed,
        settings=context.settings,
        features=context.features,
        device=context.settings.training_device(configuration.model.name),
    )


def _neural(context: SearchContext) -> NeuralRun:
    if context.neural is None:
        raise SpecificationError("The neural models and the reference need the neural inputs")
    return context.neural


def _train_summary(context: SearchContext) -> dict[str, Any]:
    train = context.inputs.train
    return {
        "rows": train.rows,
        "frauds": train.frauds,
        "transaction_ids_sha256": train.ordered_ids_sha256,
        "file_sha256": train.provenance.sha256,
    }


def _fit_tree(
    context: SearchContext, configuration: Subject, trial: int, candidate: Candidate
) -> _Fit:
    del trial
    assert isinstance(configuration, Configuration)
    name = configuration.name
    model = tree_model(context, configuration)
    settings = hyperparameters(context.declaration, configuration, candidate)
    train = context.inputs.train_data(configuration.weighting)
    valid = context.inputs.v1_data()

    def timed_fit() -> float:
        started = time.perf_counter()
        model.fit(train, valid, settings, objective=context.objective)
        return time.perf_counter() - started

    seconds, memory = run_measured(
        context.memory,
        FIT_MEMORY_SCOPE,
        timed_fit,
        gpu=context.settings.training_device(configuration.model.name) == "gpu",
    )
    history = model.training_history
    if history is None:
        raise ModelError(f"{name}: the fit returned no V1 history")
    return _Fit(
        model,
        settings,
        seconds,
        memory,
        history,
        history.best_objective,
        history.best_checkpoint,
        history.stopped_at,
        history.stopping_reason,
    )


def _fit_neural(
    context: SearchContext, configuration: Subject, trial: int, candidate: Candidate
) -> _Fit:
    assert isinstance(configuration, Configuration)
    name, model_name = configuration.name, configuration.model.name
    neural = _neural(context)
    device = context.settings.training_device(model_name)
    model = NEURAL_MODELS[model_name](
        configuration,
        context.seed,
        settings=context.settings,
        layout=neural.artifact.layout,
        device=device,
        artifact_sha256=neural.artifact_sha256,
    )
    settings = fit_hyperparameters(context.declaration, model_name, candidate)
    plan = neural.plans[model_name][trial]
    train = neural.train[configuration.weighting]

    clock = gpu_clock(Clock()) if device == "gpu" else Clock()

    def timed_fit() -> float:
        # CUDA device 0 is synchronised before the clock starts and after the work completes
        return timed(
            lambda: model.fit(train, neural.v1, settings, objective=context.objective, plan=plan),
            clock,
        )

    seconds, memory = run_measured(
        context.memory,
        NEURAL_FIT_MEMORY_SCOPE,
        timed_fit,
        gpu=device == "gpu",
        allocator=device == "gpu",
    )
    history = model.training_history
    if history is None:
        raise ModelError(f"{name}: the fit returned no V1 history")
    return _Fit(
        model,
        settings,
        seconds,
        memory,
        history,
        history.best_objective,
        history.best_checkpoint,
        history.stopped_at,
        history.stopping_reason,
    )


def _fit_reference(
    context: SearchContext, configuration: Subject, trial: int, candidate: Candidate
) -> _Fit:
    del trial
    neural = _neural(context)
    fixed = dict(context.declaration.space(REFERENCE_MODEL).fixed)
    model = LogisticRegressionReference(neural.artifact, fixed)
    train = context.inputs.train_data(Weighting.UNWEIGHTED)
    valid = context.inputs.v1_data()
    settings = dict(candidate)

    def timed_fit() -> float:
        started = time.perf_counter()
        model.fit(train, settings)
        return time.perf_counter() - started

    seconds, memory = run_measured(context.memory, REFERENCE_FIT_MEMORY_SCOPE, timed_fit)
    iterations = model.fitted_iterations
    if iterations < 1:
        raise ModelError(
            f"{configuration.name}: the solver reports {iterations} iterations; a fit record "
            "holds a positive count"
        )
    objective = context.objective
    recall = objective.queue.recall(model.predict(valid.features), valid.labels, objective.k)
    return _Fit(
        model, settings, seconds, memory, None, recall.value, iterations, iterations, "converged"
    )


_FITTERS = {TREE: _fit_tree, NEURAL: _fit_neural, REFERENCE: _fit_reference}


def _device(context: SearchContext, configuration: Subject) -> str:
    return training_device(context.settings, configuration)


def _fit_record(
    context: SearchContext,
    configuration: Subject,
    trial: int,
    *,
    model: Any | None,
    files: tuple[dict[str, str], dict[str, str]] | None,
    seconds: float | None,
    error: str | None,
) -> dict[str, Any]:
    kind = kind_of(configuration)
    weights = context.inputs.weights(configuration.weighting)
    records = context.neural_records
    preprocessing = preflight = None
    if kind != TREE:
        if records is None:
            raise SpecificationError(f"{configuration.name}: the neural records are missing")
        preprocessing = records.preprocessing
    if kind == NEURAL:
        assert records is not None
        preflight = records.preflights[(configuration.model.name, trial)]
    return {
        "record": "fit",
        "schema_version": 1,
        "identity": dict(context.identity),
        "configuration": configuration.name,
        "purpose": "search_trial",
        "trial": trial,
        "seed": context.seed,
        "device": _device(context, configuration),
        "fresh_process": False,
        "early_stopping": kind != REFERENCE,
        "length": None,
        "fitted_iterations": None if model is None else model.fitted_iterations,
        "train": _train_summary(context),
        "positive_weight": weights.positive,
        "preprocessing": preprocessing,
        "preflight": preflight,
        "model": None if files is None else files[0],
        "model_record": None if files is None else files[1],
        "fit_seconds": seconds,
        "status": "completed" if error is None else "failed",
        "error": error,
    }


def _checkpoint_record(
    context: SearchContext,
    configuration: Subject,
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
        "unit": history.unit,
        "index": history.best_checkpoint,
        "objective": history.best_objective,
        "model": files[0],
        "model_record": files[1],
    }


def _trial_record(
    context: SearchContext,
    configuration: Subject,
    trial: int,
    candidate: Candidate,
    candidate_file: Mapping[str, str],
    *,
    fitted: _Fit | None,
    fit: dict[str, str],
    checkpoint: dict[str, str] | None,
    error: str | None,
) -> dict[str, Any]:
    kind = kind_of(configuration)
    stopping = (
        context.declaration.neural_stopping if kind == NEURAL else context.declaration.tree_stopping
    )
    stepwise = kind != REFERENCE
    done = fitted is not None
    history = None if fitted is None else fitted.history
    return {
        "record": "trial",
        "schema_version": 1,
        "identity": dict(context.identity),
        "configuration": configuration.name,
        "trial": trial,
        "candidate_list": dict(candidate_file),
        "hyperparameters": dict(candidate),
        "seed": context.seed,
        "device": _device(context, configuration),
        "unit": UNITS[kind],
        "objective_k": context.declaration.v1_daily_k,
        "patience": stopping.patience if stepwise else None,
        "maximum": stopping.maximum if stepwise else None,
        "best_checkpoint": fitted.best_checkpoint if fitted and stepwise else None,
        "best_objective": fitted.objective if fitted else None,
        "stopped_at": fitted.stopped_at if fitted and stepwise else None,
        "stopping_reason": fitted.stopping_reason if fitted else None,
        "runtime_seconds": fitted.seconds if fitted else None,
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
    configuration: Subject,
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
            fitted=None,
            fit=run_file(run, fit_path),
            checkpoint=None,
            error=message,
        ),
        name,
        trial,
    )


def _reference_record(context: SearchContext, fitted: _Fit) -> dict[str, Any]:
    """The model-run record of the reference (it has no ``Configuration``)."""
    settings = context.settings_snapshot
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "configuration": "LR",
        "model": REFERENCE_MODEL,
        "family": None,
        "weighting": Weighting.UNWEIGHTED.value,
        "device": "cpu",
        "seed": None,
        "hyperparameters": jsonable(fitted.settings),
        "parameters": jsonable(fitted.model.parameters),
        "fitted_iterations": fitted.model.fitted_iterations,
        "artifact_sha256": _neural(context).artifact_sha256,
        "settings": {"path": settings.path, "sha256": settings.sha256, "content": settings.content},
        "environment": jsonable(context.environment),
        "code": jsonable(context.code),
    }


def _save(
    context: SearchContext, configuration: Subject, fitted: _Fit, directory: Path
) -> tuple[float, tuple[dict[str, str], dict[str, str]]]:
    """Save the model and its model-run record; return the seconds taken and their references."""
    run, kind = context.run, kind_of(configuration)
    saving = time.perf_counter()
    try:
        directory.mkdir(parents=True)
    except OSError as error:
        raise FileAccessError(f"{directory}: cannot be created ({error.strerror})") from error
    fitted.model.save(directory)
    if kind == REFERENCE:
        record = _reference_record(context, fitted)
    else:
        build = context.lightgbm_build if configuration.model.name == "lightgbm" else None
        record = model_run_record(
            model=fitted.model,
            settings=context.settings_snapshot,
            hyperparameters=fitted.settings,
            lightgbm_build=build,
            environment=context.environment,
            code=context.code,
        )
    write_model_run_record(directory / MODEL_RECORD_NAME, record)
    seconds = time.perf_counter() - saving
    files = (
        run_file(run, directory / MODEL_FILES[configuration.model.name]),
        run_file(run, directory / MODEL_RECORD_NAME),
    )
    return seconds, files


def run_trial(
    context: SearchContext,
    configuration: Subject,
    trial: int,
    candidate: Candidate,
    candidate_file: Mapping[str, str],
) -> TrialOutcome:
    """Fit, save and record trial ``trial`` (1 = the reference configuration) of ``configuration``.

    ``candidate_file`` is the run-directory reference of the recorded candidate list.
    """
    run, name = context.run, configuration.name
    kind = kind_of(configuration)
    directory = trial_directory(run, name, trial)
    if kind != TREE:
        _neural(context)  # a missing input is a set-up error, not a failed trial
    with log_step(logger, f"Trial {trial} of {name}"):
        try:
            fitted = _FITTERS[kind](context, configuration, trial, candidate)
            persistence, files = _save(context, configuration, fitted, directory)
        except ReviewCapacityError as error:
            _record_failure(context, configuration, trial, candidate, candidate_file, error)
            raise
        completed = _fit_record(
            context,
            configuration,
            trial,
            model=fitted.model,
            files=files,
            seconds=fitted.seconds,
            error=None,
        )
        fit_path = run.write_record(completed, name, f"trial{trial}")
        fit = run_file(run, fit_path)
        checkpoint: dict[str, str] | None = None
        if fitted.history is not None:
            checkpoint_path = run.write_record(
                _checkpoint_record(context, configuration, trial, fitted.history, files),
                name,
                f"trial{trial}",
            )
            checkpoint = run_file(run, checkpoint_path)
        record_path = run.write_record(
            _trial_record(
                context,
                configuration,
                trial,
                candidate,
                candidate_file,
                fitted=fitted,
                fit=fit,
                checkpoint=checkpoint,
                error=None,
            ),
            name,
            trial,
        )
    logger.info(
        "%s trial %d: V1 objective %.6f at %s %d (%s after %d)",
        name,
        trial,
        fitted.objective,
        UNITS[kind],
        fitted.best_checkpoint,
        fitted.stopping_reason,
        fitted.stopped_at,
    )
    return TrialOutcome(
        configuration=name,
        trial=trial,
        hyperparameters=dict(candidate),
        objective=fitted.objective,
        best_checkpoint=fitted.best_checkpoint,
        stopped_at=fitted.stopped_at,
        stopping_reason=fitted.stopping_reason,
        fit_seconds=fitted.seconds,
        persistence_seconds=persistence,
        fit_memory=fitted.memory,
        fitted_iterations=fitted.model.fitted_iterations,
        model_directory=directory,
        fit=fit,
        checkpoint=checkpoint,
        record=run_file(run, record_path),
        unit=UNITS[kind],
    )
