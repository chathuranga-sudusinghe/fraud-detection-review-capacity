"""The development run: search, selection, scoring, evaluation and timing of the eight
configurations and the logistic-regression reference, recorded in one run directory (DR-015;
Issues #135 and #151).

``prepare_development`` turns the command's options into a ``DevelopmentSetup``. Everything that
can be checked without writing is checked first: it loads the settings and the training
declaration with their snapshots, verifies the LightGBM build, passes the cost gate
(``development.cost_gate``), reads the verified Train, V1 and V2 inputs once, loads the shared
preprocessing artifact and the recorded batch plans when neural models or the reference run
(``development.neural_run``), collects the code, environment and session records and builds the
``run`` record, which names the approved plan; only then does it create the run directory. A
failure up to that point leaves at most the run's log. ``run_development`` then writes the ``run``
record first, adopts the candidate files of the approved plan, writes the ``preprocessing`` and
``preflight`` records, caps PyTorch's allocator when a neural model trains on the GPU, and runs the
setup: for each configuration in order (the reference first, so that a convergence failure shows
before the long searches; then DR-011 order) it

1. searches the declared candidates and selects a trial (``development.search``);
2. reloads the selected model, scores V1 and V2, evaluates R3 and the ranking and times the
   scoring at both boundaries (``development.scoring``, ``development.evaluation``,
   ``development.cost_timing``);
3. times single-transaction latency on the V1 sample.

Once the ``run`` record is published, the run closes with its ``run_end`` record and a generated
``summary.md`` whatever the outcome: completed, failed or interrupted. If publishing the ``run``
record itself fails, the run has no identity to close with: the error is raised and the run's log
keeps it. The setup holds everything the run uses, so a test can run the whole pipeline on
synthetic inputs.

The conditions of the timing session are recorded with every timing as context (DR-015 §9).
Timings are descriptive and never enter selection.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from review_capacity.config import (
    DatasetSpec,
    FeatureSpec,
    ModelSettings,
    load_dataset_spec,
    load_feature_spec,
)
from review_capacity.data.processed import JOINED
from review_capacity.development.cost_gate import ApprovedPlan, check_plan
from review_capacity.development.cost_timing import time_latency, time_scoring
from review_capacity.development.evaluation import evaluate_partition
from review_capacity.development.inputs import (
    TRAIN,
    V1,
    V2,
    ScoringInputs,
    SearchInputs,
    read_partitions,
)
from review_capacity.development.memory import MemorySetup
from review_capacity.development.neural_run import (
    NeuralRun,
    build_neural_run,
    neural_models_of,
    write_neural_records,
)
from review_capacity.development.records import DevelopmentPartitions
from review_capacity.development.rendering import write_summary
from review_capacity.development.run_output import (
    OUTPUT_ROOT,
    RunDirectory,
    default_directory,
    run_id,
    timestamp,
)
from review_capacity.development.run_start import (
    RUN_TYPE,
    adopt_candidates,
    project_file,
    run_identity,
    run_record,
)
from review_capacity.development.scoring import (
    ScoredPartition,
    ScoringContext,
    load_selected,
    score_partition,
)
from review_capacity.development.search import search_configuration
from review_capacity.development.startup import Startup, measure_startups
from review_capacity.development.subjects import NEURAL, REFERENCE, kind_of, training_device
from review_capacity.development.trial import SearchContext
from review_capacity.errors import SpecificationError
from review_capacity.logging_setup import log_step
from review_capacity.models.device import (
    LightGBMIdentity,
    environment_record,
    verify_lightgbm_build,
)
from review_capacity.models.registry import (
    CONFIGURATIONS,
    REFERENCE_SUBJECT,
    RUN_CONFIGURATIONS,
    Subject,
    configuration,
)
from review_capacity.models.run_record import (
    SettingsSnapshot,
    load_settings_with_snapshot,
)
from review_capacity.models.v1_objective import V1Objective
from review_capacity.paths import ProjectPaths
from review_capacity.provenance import code_provenance
from review_capacity.training.declaration import TrainingDeclaration, load_declaration
from review_capacity.training.plan import COST_NAME
from review_capacity.training.preflight import cap_allocator, release_gpu_memory
from review_capacity.training.sampler import CandidateList, generate_candidates
from review_capacity.training.timing import Clock, session_conditions

logger = logging.getLogger(__name__)


REFERENCE_NAME = REFERENCE_SUBJECT.name
DEFAULT_CONFIGURATIONS = RUN_CONFIGURATIONS


@dataclass(frozen=True)
class DevelopmentOptions:
    """What the command line chooses. ``preprocessing`` is the shared preprocessing artifact
    (needed by the neural models and the reference) and ``preflight`` the micro-batch preflight
    (needed by the neural models)."""

    plan: Path
    approved_cost_sha256: str
    configurations: tuple[str, ...] = DEFAULT_CONFIGURATIONS
    output: Path | None = None
    started: str | None = None
    preprocessing: Path | None = None
    preflight: Path | None = None


@dataclass(frozen=True)
class DevelopmentSetup:
    """Everything a development run uses."""

    run: RunDirectory
    identity: Mapping[str, Any]
    run_record: Mapping[str, Any]
    declaration: TrainingDeclaration
    settings: ModelSettings
    settings_snapshot: SettingsSnapshot
    features: FeatureSpec
    search_inputs: SearchInputs
    scoring_inputs: ScoringInputs
    lists: Mapping[str, CandidateList]
    adopt: Callable[[RunDirectory], Mapping[str, Mapping[str, str]]]
    configurations: tuple[Subject, ...]
    environment: Mapping[str, Any]
    code: Mapping[str, Any]
    lightgbm_build: LightGBMIdentity | None
    session: Mapping[str, Any]
    clock: Clock = field(default_factory=Clock)
    sleep: Callable[[float], None] = time.sleep
    memory: MemorySetup | None = None
    startups: Mapping[str, Startup] = field(default_factory=dict)
    neural: NeuralRun | None = None
    paths: ProjectPaths | None = None
    cap_allocator: Callable[[int], None] = cap_allocator
    release: Callable[[], None] = release_gpu_memory


def chosen_configurations(names: Sequence[str]) -> tuple[Subject, ...]:
    """The configurations named, each once: the reference (``LR``) first, then the eight in
    DR-011 order."""
    if not names:
        raise SpecificationError("Name at least one configuration")
    known = [c.name for c in CONFIGURATIONS]
    unknown = sorted(set(names) - {*known, REFERENCE_NAME})
    if unknown:
        raise SpecificationError(
            f"Not configurations of this run: {unknown}; the choices are "
            f"{list(DEFAULT_CONFIGURATIONS)}"
        )
    chosen: list[Subject] = [REFERENCE_SUBJECT] if REFERENCE_NAME in names else []
    chosen += [configuration(name) for name in known if name in names]
    return tuple(chosen)


def run_directory_for(paths: ProjectPaths, options: DevelopmentOptions, started: str) -> Path:
    """The new run directory: ``--output``, or by default ``outputs/development/<started>``.

    A run writes row-level scores and models, so its directory must be inside the ignored
    ``outputs/`` tree of the project: anything else could be committed by mistake.
    """
    if options.output is None:
        return default_directory(paths.root, RUN_TYPE, started)
    target = options.output if options.output.is_absolute() else paths.root / options.output
    resolved = target.resolve()
    if not resolved.is_relative_to((paths.root / OUTPUT_ROOT).resolve()):
        raise SpecificationError(
            f"{resolved}: a development run writes inside {OUTPUT_ROOT}/ of the project "
            "(ignored by Git), where its row-level scores and models stay local"
        )
    return resolved


def prepare_development(
    paths: ProjectPaths, options: DevelopmentOptions, *, now: datetime | None = None
) -> DevelopmentSetup:
    """Load, verify and gate everything a run needs; create its directory and read its inputs."""
    chosen = chosen_configurations(options.configurations)
    neural_models = neural_models_of(chosen)
    needs_artifact = bool(neural_models) or any(kind_of(c) == REFERENCE for c in chosen)
    if needs_artifact and options.preprocessing is None:
        raise SpecificationError(
            "The neural models and the reference need the shared preprocessing artifact "
            "(--preprocessing)"
        )
    if neural_models and options.preflight is None:
        raise SpecificationError("The neural models need the micro-batch preflight (--preflight)")
    started = options.started or run_id(now)
    with log_step(logger, f"Preparing the development run {started}"):
        settings, settings_file = load_settings_with_snapshot(paths)
        declaration, declaration_file = load_declaration(paths)
        features = load_feature_spec(paths.features_spec)
        dataset: DatasetSpec = load_dataset_spec(paths)
        partitions = DevelopmentPartitions.from_spec(dataset, paths)
        build: LightGBMIdentity | None = None
        build_seconds: float | None = None
        if any(c.model.name == "lightgbm" for c in chosen):
            build_started = time.perf_counter()
            build = verify_lightgbm_build(settings.lightgbm_build)
            build_seconds = time.perf_counter() - build_started
        lists = generate_candidates(declaration)
        approved: ApprovedPlan = check_plan(
            options.plan,
            options.approved_cost_sha256,
            declaration,
            declaration_file,
            lists,
            partitions,
        )
        memory = MemorySetup.detect(
            declaration.timing.host_sample_interval_ms,
            gpu_monitoring=any(training_device(settings, c) == "gpu" for c in chosen),
        )
        startups = measure_startups(chosen, settings, lightgbm_seconds=build_seconds)
        directory = run_directory_for(paths, options, started)
        models = tuple(dict.fromkeys(c.model.name for c in chosen))
        parts = read_partitions(paths, features, dataset, (TRAIN, V1, V2))
        code = code_provenance(paths.root)
        environment = environment_record()
        neural: NeuralRun | None = None
        if needs_artifact and options.preprocessing is not None:
            neural = build_neural_run(
                features,
                declaration,
                SearchInputs(parts[TRAIN], parts[V1]),
                ScoringInputs(parts[V1], parts[V2]),
                options.preprocessing,
                preflight=options.preflight,
                candidates={model: listed.candidates for model, listed in lists.items()},
                models=neural_models,
                declaration_sha256=declaration_file.sha256,
                settings_sha256=settings_file.sha256,
                code=code,
                environment=environment,
            )
        identity = run_identity(
            started,
            code,
            declaration,
            declaration_file,
            settings_file,
            settings.main_seed,
            paths.manifest,
        )
        record = run_record(
            paths,
            identity,
            timestamp(now),
            code,
            environment,
            settings_file,
            declaration_file,
            (JOINED, TRAIN, V1, V2),
            [c.name for c in chosen],
            project_file(paths, options.plan / COST_NAME, approved.cost_sha256),
        )
        session = session_conditions()
        run = RunDirectory.create(
            paths.root, partitions, RUN_TYPE, started, directory, log_opened=True
        )
    return DevelopmentSetup(
        run=run,
        identity=identity,
        run_record=record,
        declaration=declaration,
        settings=settings,
        settings_snapshot=settings_file,
        features=features,
        search_inputs=SearchInputs(parts[TRAIN], parts[V1]),
        scoring_inputs=ScoringInputs(parts[V1], parts[V2]),
        lists=lists,
        adopt=lambda target: adopt_candidates(target, approved, models),
        configurations=chosen,
        environment=environment,
        code=code,
        lightgbm_build=build,
        session=session,
        memory=memory,
        startups=startups,
        neural=neural,
        paths=paths,
    )


def _run_configuration(
    setup: DevelopmentSetup,
    search: SearchContext,
    scoring: ScoringContext,
    chosen: Subject,
    candidate_files: Mapping[str, Mapping[str, str]],
) -> None:
    name, model_name = chosen.name, chosen.model.name
    result = search_configuration(
        search, chosen, setup.lists[model_name], candidate_files[model_name]
    )
    model = load_selected(scoring, chosen, result.selected)
    scored: dict[str, ScoredPartition] = {}
    for label in ("V1", "V2"):
        scored[label] = score_partition(scoring, chosen, result.selected, model, label)
        evaluate_partition(scoring, scored[label])
        time_scoring(scoring, chosen, model, scored[label], clock=setup.clock, sleep=setup.sleep)
    time_latency(scoring, chosen, model, scored["V1"], clock=setup.clock, sleep=setup.sleep)
    logger.info("%s done: selected trial %d", name, result.selected.trial)


def run_development(setup: DevelopmentSetup) -> Path:
    """Run every configuration of ``setup`` and close the run; returns the run directory.

    The ``run`` record is published first. From then on the run closes with ``run_end`` and
    ``summary.md`` also after a failure or an interrupt (including a failure while adopting the
    candidate files or building the search state), and the error or interrupt is raised again.
    If the ``run`` record cannot be published, the error is raised and nothing else is written.
    """
    run = setup.run
    declaration = setup.declaration
    partitions = run.partitions
    with run.closing(lambda path: write_summary(path, partitions)):
        run.write_record(setup.run_record)
        candidate_files = setup.adopt(run)
        neural_records = None
        if setup.neural is not None:
            if setup.paths is None:
                raise SpecificationError("A neural run needs the project paths")
            neural_models = neural_models_of(setup.configurations)
            neural_records = write_neural_records(
                run,
                setup.paths,
                setup.identity,
                setup.neural,
                setup.search_inputs,
                candidate_files,
                neural_models,
            )
            if any(
                training_device(setup.settings, c) == "gpu"
                for c in setup.configurations
                if kind_of(c) == NEURAL
            ):
                setup.cap_allocator(declaration.neural.preflight_free_margin_mib)
        search = SearchContext(
            run=run,
            identity=setup.identity,
            declaration=declaration,
            settings=setup.settings,
            settings_snapshot=setup.settings_snapshot,
            features=setup.features,
            inputs=setup.search_inputs,
            objective=V1Objective(setup.search_inputs.v1_data(), declaration.v1_daily_k),
            environment=setup.environment,
            code=setup.code,
            lightgbm_build=setup.lightgbm_build,
            session=setup.session,
            memory=setup.memory,
            startups=setup.startups,
            neural=setup.neural,
            neural_records=neural_records,
        )
        scoring = ScoringContext(
            run=run,
            identity=setup.identity,
            declaration=declaration,
            settings=setup.settings,
            features=setup.features,
            inputs=setup.scoring_inputs,
            session=setup.session,
            memory=setup.memory,
            neural=setup.neural,
        )
        with log_step(logger, f"Development run {run.started}"):
            for chosen in setup.configurations:
                _run_configuration(setup, search, scoring, chosen, candidate_files)
                # The configuration's model, optimiser and cached GPU blocks are released before
                # the next one starts, which may use another library (FTT-u is followed by Cat-b).
                setup.release()
    return run.path
