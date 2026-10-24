"""The fresh process of the same-seed repeat (DR-013 §6, item 1; DR-015 §4; Issue #156).

A same-seed repeat measures numerical and backend non-determinism, so its fit runs in a fresh
process with full model reinitialisation: the library, CUDA and allocator state of the run's
process do not carry over. ``FreshProcess`` is the callable that ``development.refits`` calls: it
writes a job file into the run directory, starts ``review-capacity refit-fresh --job FILE`` as a
child process of the same interpreter (its log goes to ``refit-<configuration>-<tag>.log`` beside
the job, its terminal output to the parent's terminal) and reads the seconds the fit took from the
result file. The child rebuilds what the fit needs from the project files, exactly as a run does
(the verified Train, V1 and V2 inputs, the training declaration, the model settings and, for a
neural model, the shared preprocessing artifact and the selected trial's recorded batch plan),
checks that its package source is the run's, fits for exactly the selected length, and saves the
model in the directory it is given. The parent reloads it and scores it like any other model.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from review_capacity.config import load_dataset_spec, load_feature_spec
from review_capacity.development.inputs import (
    TRAIN,
    V1,
    V2,
    ScoringInputs,
    SearchInputs,
    read_partitions,
)
from review_capacity.development.neural_run import NeuralRun, build_neural_run
from review_capacity.development.records import DevelopmentPartitions
from review_capacity.development.refits import RefitJob, RefitSpec, fit_refit
from review_capacity.development.run_output import RunDirectory
from review_capacity.development.subjects import NEURAL, kind_of
from review_capacity.development.trial import SearchContext
from review_capacity.errors import FileAccessError, ModelError, SpecificationError
from review_capacity.files import write_atomically
from review_capacity.logging_setup import log_step
from review_capacity.models.device import environment_record, verify_lightgbm_build
from review_capacity.models.neural_training import BatchPlan
from review_capacity.models.registry import configuration as configuration_named
from review_capacity.models.run_record import jsonable, load_settings_with_snapshot
from review_capacity.models.v1_objective import V1Objective
from review_capacity.paths import ProjectPaths
from review_capacity.provenance import code_provenance
from review_capacity.training.declaration import load_declaration
from review_capacity.training.sampler import Candidate

logger = logging.getLogger(__name__)

JOBS_DIR = "refit-jobs"
COMMAND = "refit-fresh"

type ContextLoader = Callable[[Mapping[str, Any]], SearchContext]


def job_document(
    root: Path,
    run: RunDirectory,
    identity: Mapping[str, Any],
    job: RefitJob,
    directory: Path,
    result: Path,
    neural: NeuralRun | None,
) -> dict[str, Any]:
    """The JSON document that describes ``job`` for the child process."""
    plan = None
    if neural is not None and kind_of(configuration_named(job.configuration)) == NEURAL:
        batch = neural.plans[configuration_named(job.configuration).model.name][job.trial]
        plan = {"effective": batch.effective, "micro": batch.micro}
    return {
        "root": str(root),
        "run_directory": str(run.path),
        "started": run.started,
        "identity": jsonable(dict(identity)),
        "configuration": job.configuration,
        "kind": job.spec.kind,
        "seed": job.spec.seed,
        "device": job.spec.device,
        "trial": job.trial,
        "length": job.length,
        "hyperparameters": jsonable(dict(job.hyperparameters)),
        "directory": str(directory),
        "result": str(result),
        "preprocessing": None if neural is None else str(neural.directory),
        "batch_plan": plan,
    }


class FreshProcess:
    """Fits and saves a refit in a child process; see the module description."""

    def __init__(
        self,
        root: Path,
        run: RunDirectory,
        identity: Mapping[str, Any],
        neural: NeuralRun | None = None,
        *,
        launch: Callable[[list[str]], int] | None = None,
    ) -> None:
        self.root = root
        self.run = run
        self.identity = identity
        self.neural = neural
        self.launch = launch or (lambda command: subprocess.run(command, check=False).returncode)

    def __call__(self, job: RefitJob, directory: Path) -> float:
        name, tag = job.configuration, job.spec.tag
        folder = self.run.path / JOBS_DIR
        try:
            folder.mkdir(exist_ok=True)
        except OSError as error:
            raise FileAccessError(f"{folder}: cannot be created ({error.strerror})") from error
        job_file, result = folder / f"{name}-{tag}.json", folder / f"{name}-{tag}.result.json"
        log = self.run.path / f"refit-{name}-{tag}.log"
        document = job_document(
            self.root, self.run, self.identity, job, directory, result, self.neural
        )
        text = json.dumps(document, indent=2, sort_keys=True)
        write_atomically(job_file, lambda target: target.write_text(text, encoding="utf-8"))
        command = [
            sys.executable,
            "-m",
            "review_capacity",
            "--log-file",
            str(log),
            COMMAND,
            "--root",
            str(self.root),
            "--job",
            str(job_file),
        ]
        with log_step(logger, f"Fresh process for the {tag} of {name}"):
            code = self.launch(command)
        if code != 0:
            raise ModelError(
                f"{name}: the fresh process of the {tag} ended with exit code {code}; "
                f"see {log.name} in the run directory"
            )
        return read_result(result)


def read_result(path: Path) -> float:
    """The fit seconds that the child process wrote."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        seconds = document["seconds"]
    except (OSError, ValueError, KeyError) as error:
        raise FileAccessError(f"{path}: the fresh process left no readable result") from error
    if isinstance(seconds, bool) or not isinstance(seconds, int | float) or seconds < 0:
        raise ModelError(f"{path}: the fit seconds are not a non-negative number")
    return float(seconds)


# --- The child process ---------------------------------------------------------------------------


def check_identity(
    identity: Mapping[str, Any], source_sha256: str, settings_sha256: str, declaration_sha256: str
) -> None:
    """Stop unless the child's package source, model settings and training declaration are those
    of the run (its identity): a refit repeats the run's selected setting."""
    for what, found, expected in (
        ("package source", source_sha256, identity["source_sha256"]),
        ("model settings (config/models.yaml)", settings_sha256, identity["settings_sha256"]),
        (
            "training declaration (config/training.yaml)",
            declaration_sha256,
            identity["declaration_sha256"],
        ),
    ):
        if found != expected:
            raise SpecificationError(
                f"The {what} differ from the run's: the refit would not repeat its setting"
            )


def load_context(job: Mapping[str, Any]) -> SearchContext:
    """The search state a refit needs, rebuilt from the project files as a run builds it."""
    paths = ProjectPaths(Path(job["root"]))
    subject = configuration_named(job["configuration"])
    settings, settings_file = load_settings_with_snapshot(paths)
    declaration, declaration_file = load_declaration(paths)
    features = load_feature_spec(paths.features_spec)
    dataset = load_dataset_spec(paths)
    partitions = DevelopmentPartitions.from_spec(dataset, paths)
    build = (
        verify_lightgbm_build(settings.lightgbm_build) if subject.model.name == "lightgbm" else None
    )
    parts = read_partitions(paths, features, dataset, (TRAIN, V1, V2))
    search, scoring = SearchInputs(parts[TRAIN], parts[V1]), ScoringInputs(parts[V1], parts[V2])
    code = code_provenance(paths.root)
    identity = job["identity"]
    check_identity(
        identity, code["package"]["source_sha256"], settings_file.sha256, declaration_file.sha256
    )
    environment = environment_record()
    neural: NeuralRun | None = None
    if kind_of(subject) == NEURAL:
        neural = build_neural_run(
            features,
            declaration,
            search,
            scoring,
            Path(job["preprocessing"]),
            preflight=None,
            candidates={},
            models=(),
            declaration_sha256=declaration_file.sha256,
            settings_sha256=settings_file.sha256,
            code=code,
            environment=environment,
        )
        plan = job["batch_plan"]
        neural = replace(
            neural,
            plans={subject.model.name: {job["trial"]: BatchPlan(plan["effective"], plan["micro"])}},
        )
    run = RunDirectory(Path(job["run_directory"]), partitions, identity["run_type"], job["started"])
    return SearchContext(
        run=run,
        identity=identity,
        declaration=declaration,
        settings=settings,
        settings_snapshot=settings_file,
        features=features,
        inputs=search,
        objective=V1Objective(search.v1_data(), declaration.v1_daily_k),
        environment=environment,
        code=code,
        lightgbm_build=build,
        session={},
        neural=neural,
    )


def run_job(job: Mapping[str, Any], *, load: ContextLoader = load_context) -> float:
    """Fit and save the refit that ``job`` describes, write its result file and return the fit
    seconds. ``load`` rebuilds the shared state (tests substitute synthetic inputs)."""
    subject = configuration_named(job["configuration"])
    spec = RefitSpec(job["kind"], job["seed"], job["device"], fresh_process=True)
    candidate: Candidate = dict(job["hyperparameters"])
    refit = RefitJob(subject.name, spec, job["trial"], candidate, job["length"])
    with log_step(logger, f"Fresh-process {spec.tag} of {subject.name}"):
        context = load(job)
        seconds, _ = fit_refit(context, subject, refit, Path(job["directory"]))
    text = json.dumps({"seconds": seconds}, sort_keys=True)
    result = Path(job["result"])
    write_atomically(result, lambda target: target.write_text(text, encoding="utf-8"))
    return seconds


def run_job_file(path: Path, *, load: ContextLoader | None = None) -> float:
    """``run_job`` for the job file written by ``FreshProcess``."""
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise FileAccessError(f"{path}: the job file cannot be read") from error
    return run_job(job, load=load_context if load is None else load)
