"""The analysis of a completed development run: the post-selection stages (Issue #166; DR-009,
DR-007, DR-010, DR-015 §4, DR-013 §7).

A development run searches, selects, scores V1 and V2 and evaluates R3 for each configuration.
Everything that follows from those results is done here, in a **new analysis run** that reads the
closed development run (its *source*) and never changes it:

* ``thresholds``: R0 on V1 and V2 and R1 and R2 on V2 at the four capacities, with the R1
  selections, including the label-latency one (``development.thresholds``);
* ``calibration``: the Platt and isotonic calibrators of each configuration (primary and
  label-latency), R4 and the RQ3 cost analysis (``development.calibration``, ``development.rq3``);
* ``h3``: the binding H3 configuration selection and its label-latency check
  (``development.h3``);
* ``refits``: the fixed-length refits and the CatBoost CPU–GPU comparison
  (``development.refits``); they train models, need the shared preprocessing artifact and the
  micro-batch preflight and are never part of the default stages.

The analysis run is a directory of its own (``outputs/analysis/<UTC start time>/``) with its
``run`` record (naming the source run by the SHA-256 of its ``run_end`` record), its records, a
generated ``summary.md`` and ``run.log``. Its ``source/`` link points to the source run, so a
reference such as ``source/scores/<file>`` resolves and its SHA-256 can be checked. A stage can be
repeated in a new analysis run without touching the source run or the other stages' outputs.

Nothing reads the Test period: the V1 and V2 files are read through the verified readers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
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
from review_capacity.development.calibration import fit_calibrators
from review_capacity.development.h3 import select_latency, select_primary
from review_capacity.development.inputs import (
    TRAIN,
    V1,
    V2,
    ScoringInputs,
    SearchInputs,
    read_partitions,
)
from review_capacity.development.neural_run import (
    NeuralRecords,
    NeuralRun,
    build_neural_run,
    neural_models_of,
)
from review_capacity.development.records import CONFIGURATION_NAMES, DevelopmentPartitions
from review_capacity.development.refit_process import FreshProcess
from review_capacity.development.refits import RefitJob, run_refits
from review_capacity.development.rendering import write_summary
from review_capacity.development.rq3 import evaluate_r4
from review_capacity.development.run_output import (
    OUTPUT_ROOT,
    LoadedRun,
    RunDirectory,
    default_directory,
    load_run,
    run_id,
    source_file,
    timestamp,
)
from review_capacity.development.run_start import (
    processed_files,
    project_file,
    run_identity,
    run_record,
)
from review_capacity.development.scoring import ScoredPartition, ScoringContext
from review_capacity.development.subjects import NEURAL, kind_of, training_device
from review_capacity.development.thresholds import evaluate_thresholds, saved_partition
from review_capacity.development.trial import SearchContext, TrialOutcome
from review_capacity.errors import RecordContractError, SpecificationError
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import log_step
from review_capacity.models.device import (
    LightGBMIdentity,
    environment_record,
    verify_lightgbm_build,
)
from review_capacity.models.neural_training import BatchPlan
from review_capacity.models.registry import configuration as configuration_named
from review_capacity.models.run_record import SettingsSnapshot, load_settings_with_snapshot
from review_capacity.models.v1_objective import V1Objective
from review_capacity.paths import ProjectPaths
from review_capacity.provenance import code_provenance
from review_capacity.training.declaration import TrainingDeclaration, load_declaration
from review_capacity.training.preflight import cap_allocator
from review_capacity.training.timing import session_conditions

logger = logging.getLogger(__name__)

RUN_TYPE = "analysis"
THRESHOLDS = "thresholds"
CALIBRATION = "calibration"
H3 = "h3"
REFITS = "refits"
STAGES = (THRESHOLDS, CALIBRATION, H3, REFITS)
DEFAULT_STAGES = (THRESHOLDS, CALIBRATION, H3)  # refits train models: named explicitly


@dataclass(frozen=True)
class AnalysisOptions:
    """What the command line chooses: the source run, the stages and where to write."""

    source: Path
    stages: tuple[str, ...] = DEFAULT_STAGES
    output: Path | None = None
    started: str | None = None


@dataclass(frozen=True)
class AnalysisSetup:
    """Everything an analysis run uses."""

    run: RunDirectory
    identity: Mapping[str, Any]
    run_record: Mapping[str, Any]
    source: LoadedRun
    declaration: TrainingDeclaration
    settings: ModelSettings
    features: FeatureSpec
    scoring_inputs: ScoringInputs
    configurations: tuple[str, ...]
    stages: tuple[str, ...]
    paths: ProjectPaths
    settings_snapshot: SettingsSnapshot | None = None
    environment: Mapping[str, Any] = field(default_factory=dict)
    code: Mapping[str, Any] = field(default_factory=dict)
    # What the refits stage adds: the Train and V1 inputs, the neural inputs and records, the
    # LightGBM build, the session conditions and the way a fresh process is started.
    search_inputs: SearchInputs | None = None
    neural: NeuralRun | None = None
    neural_records: NeuralRecords | None = None
    lightgbm_build: LightGBMIdentity | None = None
    session: Mapping[str, Any] = field(default_factory=dict)
    fresh: Callable[[RefitJob, Path], float] | None = None
    cap_allocator: Callable[[int], None] = cap_allocator

    @property
    def seed(self) -> int:
        seed: int = self.identity["seeds"]["main"]
        return seed

    @property
    def partitions(self) -> DevelopmentPartitions:
        return self.run.partitions

    def saved(self, configuration: str, label: str) -> ScoredPartition:
        """The selected ``configuration``'s saved scoring of V1 or V2, under the source link."""
        return saved_partition(self.source, self.scoring_inputs, configuration, label, rebase=True)


def check_stages(stages: Sequence[str]) -> tuple[str, ...]:
    """The stages named, each once, in the order they run."""
    if not stages:
        raise SpecificationError("Name at least one stage")
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        raise SpecificationError(f"Not stages of the analysis: {unknown}; the choices are {STAGES}")
    return tuple(stage for stage in STAGES if stage in stages)


def analysis_directory(paths: ProjectPaths, options: AnalysisOptions, started: str) -> Path:
    """The new run directory: ``--output``, or ``outputs/analysis/<started>``; always inside the
    ignored ``outputs/`` tree, where the row-level files stay local."""
    if options.output is None:
        return default_directory(paths.root, RUN_TYPE, started)
    target = options.output if options.output.is_absolute() else paths.root / options.output
    resolved = target.resolve()
    if not resolved.is_relative_to((paths.root / OUTPUT_ROOT).resolve()):
        raise SpecificationError(
            f"{resolved}: an analysis run writes inside {OUTPUT_ROOT}/ of the project "
            "(ignored by Git), where its row-level files stay local"
        )
    if resolved.is_relative_to(options.source.resolve()):
        raise SpecificationError(
            f"{resolved}: an analysis run never writes inside its source run, which it does not "
            "change"
        )
    return resolved


def load_source(directory: Path, partitions: DevelopmentPartitions) -> LoadedRun:
    """The closed, completed development run at ``directory``, with every record validated."""
    source = load_run(directory, partitions)
    if source.end is None:
        raise SpecificationError(f"{directory}: the run was not closed; it cannot be analysed")
    if source.end["outcome"] != "completed":
        raise SpecificationError(
            f"{directory}: the run {source.end['outcome']}; only a completed run is analysed"
        )
    if source.identity["run_type"] != "development":
        raise SpecificationError(f"{directory}: not a development run")
    return source


def check_source(
    source: LoadedRun,
    settings_sha256: str,
    declaration_sha256: str,
    manifest_sha256: str,
    processed: Mapping[str, str],
) -> None:
    """Stop unless the files the analysis loads are those the source run used: the model
    settings, the training declaration, the dataset manifest and the processed files."""
    identity = source.identity
    for what, found, expected in (
        ("model settings", settings_sha256, identity["settings_sha256"]),
        ("training declaration", declaration_sha256, identity["declaration_sha256"]),
        ("dataset manifest", manifest_sha256, identity["dataset_manifest_sha256"]),
    ):
        if found != expected:
            raise SpecificationError(f"The {what} differ from those of the source run")
    recorded = source.of_kind("run")[0]["dataset"]["processed"]
    for stem, sha256 in processed.items():
        if recorded.get(stem, {}).get("sha256") != sha256:
            raise SpecificationError(f"The processed {stem} file differs from the source run's")


def prepare_analysis(
    paths: ProjectPaths, options: AnalysisOptions, *, now: datetime | None = None
) -> AnalysisSetup:
    """Load and verify everything an analysis run uses, then create its directory."""
    stages = check_stages(options.stages)
    refits = REFITS in stages
    started = options.started or run_id(now)
    with log_step(logger, f"Preparing the analysis run {started}"):
        dataset: DatasetSpec = load_dataset_spec(paths)
        partitions = DevelopmentPartitions.from_spec(dataset, paths)
        source_directory = options.source.resolve()
        source = load_source(source_directory, partitions)
        settings, settings_file = load_settings_with_snapshot(paths)
        declaration, declaration_file = load_declaration(paths)
        features = load_feature_spec(paths.features_spec)
        manifest_sha256 = file_sha256(paths.manifest)
        stems = (TRAIN, V1, V2) if refits else (V1, V2)
        parts = read_partitions(paths, features, dataset, stems)
        check_source(
            source,
            settings_file.sha256,
            declaration_file.sha256,
            manifest_sha256,
            {stem: ref["sha256"] for stem, ref in processed_files(paths, (JOINED, *stems)).items()},
        )
        if source.identity["seeds"]["main"] != settings.main_seed:
            raise SpecificationError("The main seed differs from that of the source run")
        run_record_of_source = source.of_kind("run")[0]
        configurations = tuple(
            name for name in run_record_of_source["configurations"] if name in CONFIGURATION_NAMES
        )
        if not configurations:
            raise RecordContractError(f"{source_directory}: the source run holds no configuration")
        code = code_provenance(paths.root)
        environment = environment_record()
        identity = run_identity(
            started,
            code,
            declaration,
            declaration_file,
            settings_file,
            settings.main_seed,
            paths.manifest,
            run_type=RUN_TYPE,
        )
        record = run_record(
            paths,
            identity,
            timestamp(now),
            code,
            environment,
            settings_file,
            declaration_file,
            (JOINED, *stems),
            configurations,
            dict(run_record_of_source["plan"]),
            {
                "run_id": source.identity["run_id"],
                "run_end": project_file(paths, source_directory / "records" / "run_end.json"),
            },
        )
        directory = analysis_directory(paths, options, started)
        run = RunDirectory.create(
            paths.root,
            partitions,
            RUN_TYPE,
            started,
            directory,
            log_opened=True,
            source=source_directory,
        )
    setup = AnalysisSetup(
        run=run,
        identity=identity,
        run_record=record,
        source=source,
        declaration=declaration,
        settings=settings,
        features=features,
        scoring_inputs=ScoringInputs(parts[V1], parts[V2]),
        configurations=configurations,
        stages=stages,
        paths=paths,
        settings_snapshot=settings_file,
        environment=environment,
        code=code,
    )
    return with_refit_inputs(setup, parts, dataset) if refits else setup


def _thresholds(setup: AnalysisSetup) -> None:
    run = setup.run
    for name in setup.configurations:
        evaluate_thresholds(
            run,
            run,
            setup.partitions,
            setup.identity,
            setup.seed,
            setup.saved(name, "V1"),
            setup.saved(name, "V2"),
        )


def _calibration(setup: AnalysisSetup) -> None:
    run = setup.run
    for name in setup.configurations:
        v2 = setup.saved(name, "V2")
        fitted = fit_calibrators(run, run, setup.partitions, setup.identity, v2)
        evaluate_r4(run, run, setup.partitions, setup.identity, setup.seed, v2, fitted)


def _h3(setup: AnalysisSetup) -> None:
    missing = [name for name in CONFIGURATION_NAMES if name not in setup.configurations]
    if missing:
        raise SpecificationError(
            f"The H3 selection needs the eight configurations; the source run lacks {missing}"
        )
    run = setup.run
    select_primary(run, setup.partitions, setup.identity, source=setup.source)
    scored = {name: setup.saved(name, "V2") for name in CONFIGURATION_NAMES}
    select_latency(run, setup.partitions, setup.identity, setup.seed, scored)


def _one(records: Sequence[Mapping[str, Any]], what: str) -> Mapping[str, Any]:
    if len(records) != 1:
        raise RecordContractError(f"{len(records)} {what}, not one")
    return records[0]


def selected_outcome(source: LoadedRun, configuration: str) -> TrialOutcome:
    """The selected trial of ``configuration`` rebuilt from the records of the source run: its
    searched values, its retained length, its saved model and its records (under ``source/``)."""
    chosen = _one(
        [r for r in source.of_kind("configuration") if r["configuration"] == configuration],
        f"configuration records of {configuration}",
    )
    number = chosen["selected_trial"]
    trial = _one(
        [
            r
            for r in source.of_kind("trial")
            if r["configuration"] == configuration and r["trial"] == number
        ],
        f"trial records of {configuration} trial {number}",
    )
    fits = [r for n, r in source.records if f"records/{n}" == chosen["selected_fit"]["path"]]
    fit = _one(fits, f"fit records of {configuration} trial {number}")
    if fit["status"] != "completed" or fit["model"] is None:
        raise RecordContractError(f"{configuration} trial {number}: the fit holds no saved model")
    summary = _one([t for t in chosen["trials"] if t["trial"] == number], "trial summaries")
    return TrialOutcome(
        configuration=configuration,
        trial=number,
        hyperparameters=dict(chosen["selected_hyperparameters"]),
        objective=chosen["selected_objective"],
        best_checkpoint=trial["best_checkpoint"],
        stopped_at=trial["stopped_at"],
        stopping_reason=trial["stopping_reason"],
        fit_seconds=fit["fit_seconds"],
        persistence_seconds=0.0,
        fit_memory=None,
        fitted_iterations=fit["fitted_iterations"],
        model_directory=(source.path / fit["model"]["path"]).parent,
        fit=source_file(chosen["selected_fit"]),
        checkpoint=None,
        record=source_file(summary["record"]),
        unit=trial["unit"],
    )


def _source_reference(source: LoadedRun, kind: str, **fields: Any) -> tuple[str, dict[str, str]]:
    """The one record of ``kind`` with the given field values, and its file reference in the
    source run."""
    matches = [
        (name, record)
        for name, record in source.records
        if record["record"] == kind and all(record[k] == v for k, v in fields.items())
    ]
    if len(matches) != 1:
        raise RecordContractError(f"{len(matches)} {kind} records {fields} in the source run")
    name = matches[0][0]
    path = source.path / "records" / name
    return name, {"path": f"records/{name}", "sha256": file_sha256(path)}


def with_refit_inputs(
    setup: AnalysisSetup, parts: Mapping[str, Any], dataset: DatasetSpec
) -> AnalysisSetup:
    """``setup`` with what the refits stage needs: the Train and V1 inputs, the LightGBM build,
    and, for neural configurations, the shared preprocessing artifact and the batch plans that
    the source run recorded (the artifact must be the one the source run used)."""
    del dataset
    search = SearchInputs(parts[TRAIN], parts[V1])
    subjects = [configuration_named(name) for name in setup.configurations]
    build = (
        verify_lightgbm_build(setup.settings.lightgbm_build)
        if any(c.model.name == "lightgbm" for c in subjects)
        else None
    )
    neural: NeuralRun | None = None
    records: NeuralRecords | None = None
    models = neural_models_of(subjects)
    if models:
        source, paths = setup.source, setup.paths
        _, artifact = _source_reference(source, "preprocessing")
        record = _one(source.of_kind("preprocessing"), "preprocessing records")
        directory = paths.root / Path(record["artifact"]["path"]).parent
        if setup.settings_snapshot is None:
            raise SpecificationError("The refits need the loaded model settings")
        neural = build_neural_run(
            setup.features,
            setup.declaration,
            search,
            setup.scoring_inputs,
            directory,
            preflight=None,
            candidates={},
            models=(),
            declaration_sha256=setup.identity["declaration_sha256"],
            settings_sha256=setup.settings_snapshot.sha256,
            code=setup.code,
            environment=setup.environment,
        )
        if neural.artifact_sha256 != record["artifact_sha256"]:
            raise SpecificationError("The preprocessing artifact differs from the source run's")
        plans: dict[str, dict[int, BatchPlan]] = {}
        preflights: dict[tuple[str, int], dict[str, str]] = {}
        for name, entry in source.records:
            if entry["record"] != "preflight":
                continue
            plans.setdefault(entry["model"], {})[entry["trial"]] = BatchPlan(
                entry["effective_batch"], entry["micro_batch"]
            )
            reference = {
                "path": f"records/{name}",
                "sha256": file_sha256(source.path / "records" / name),
            }
            preflights[(entry["model"], entry["trial"])] = source_file(reference)
        neural = replace(neural, plans=plans)
        records = NeuralRecords(source_file(artifact), preflights)
    return replace(
        setup,
        search_inputs=search,
        neural=neural,
        neural_records=records,
        lightgbm_build=build,
        session=session_conditions(),
    )


def refit_contexts(setup: AnalysisSetup) -> tuple[SearchContext, ScoringContext]:
    """The search and scoring state of the refits, writing into the analysis run."""
    if setup.search_inputs is None or setup.settings_snapshot is None:
        raise SpecificationError("The refits stage needs the Train and V1 inputs and the settings")
    run, declaration = setup.run, setup.declaration
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
        neural=setup.neural,
        neural_records=setup.neural_records,
    )
    scoring = ScoringContext(
        run=run,
        identity=setup.identity,
        declaration=declaration,
        settings=setup.settings,
        features=setup.features,
        inputs=setup.scoring_inputs,
        session=setup.session,
        neural=setup.neural,
    )
    return search, scoring


def _refits(setup: AnalysisSetup) -> None:
    search, scoring = refit_contexts(setup)
    run, declaration = setup.run, setup.declaration
    subjects = [configuration_named(name) for name in setup.configurations]
    if any(kind_of(c) == NEURAL and training_device(setup.settings, c) == "gpu" for c in subjects):
        setup.cap_allocator(declaration.neural.preflight_free_margin_mib)
    fresh = setup.fresh or FreshProcess(setup.paths.root, run, setup.identity, setup.neural)
    for subject in subjects:
        run_refits(
            search,
            scoring,
            subject,
            selected_outcome(setup.source, subject.name),
            setup.saved(subject.name, "V2"),
            fresh=fresh,
        )


STAGE_FUNCTIONS: dict[str, Callable[[AnalysisSetup], None]] = {
    THRESHOLDS: _thresholds,
    CALIBRATION: _calibration,
    H3: _h3,
    REFITS: _refits,
}


def run_analysis(setup: AnalysisSetup) -> Path:
    """Run the stages of ``setup`` in order and close the run; returns the run directory.

    The ``run`` record is written first. From then on the run closes with ``run_end`` and
    ``summary.md`` also after a failure or an interrupt, and the error is raised again.
    """
    run = setup.run
    with run.closing(lambda path: write_summary(path, run.partitions, setup.paths.root)):
        run.write_record(setup.run_record)
        with log_step(logger, f"Analysis run {run.started}: stages {', '.join(setup.stages)}"):
            for stage in setup.stages:
                function = STAGE_FUNCTIONS.get(stage)
                if function is None:
                    raise SpecificationError(f"The stage {stage!r} is not available")
                with log_step(logger, f"Stage {stage}"):
                    function(setup)
    return run.path
