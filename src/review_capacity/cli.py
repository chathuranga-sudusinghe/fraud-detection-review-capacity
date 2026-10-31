"""Command-line interface: ``review-capacity <command>`` or ``python -m review_capacity <command>``.

Commands:

    prepare-data   validate the raw data; write the manifest and the development files, once
                   (--check-only: validate and report, write nothing). The only command that
                   reads the raw files before the confirmatory Test run.
    migrate-provenance
                   verify the processed development files (SHA-256, rows, columns, types,
                   raw-source hashes, DR-001 anchors) and write their provenance records again
                   with the feature-declaration fingerprint; the data files are not rewritten
                   and no raw file is read (--check-only: verify and report, write nothing)
    profile-data   recompute the feature-catalog statistics from the verified processed
                   development file (no labels, no raw file) and report every difference from
                   docs/feature-catalog.md (--save: also write the profile and its provenance
                   record to the processed directory)
    audit-duplicates
                   count repeated input vectors (identical model inputs under different
                   TransactionID values) within and across the development partitions, from
                   the verified processed development file (no labels, no raw file); report
                   only, nothing is removed (--save: also write the counts with their provenance
                   to the processed directory)
    probe-gpu      the GPU acceptance probe of DR-013 §8 (docs/gpu-acceptance-probe.md), on
                   synthetic data only: all eight configurations on their training devices
                   (DR-013 §1), the checks C1-C13 and the tolerances T1-T4; writes probe.json,
                   the generated probe.md, the comparison files and its log, always probe.log
                   (--log-file is refused), to a new directory: --output, or by default
                   outputs/gpu_acceptance_probe/<UTC start time, YYYY-MM-DDTHHMMSSZ>/ (ignored
                   by Git). A first run ends as "pending second run"; a second run, in a fresh
                   process and its own directory, with --previous (the first run's
                   probe.json), completes the comparison

    probe-catboost-objective
                   the CatBoost tree-objective acceptance (docs/gpu-acceptance-probe.md,
                   Section 8, checks G1-G7), on synthetic data only: one CatBoost fit on the GPU
                   at depth 10 and 2,000 rounds with the shared V1 objective as the
                   user-defined metric; writes acceptance.json, the generated acceptance.md and
                   its log, always acceptance.log (--log-file is refused), to a new directory:
                   --output, or by default
                   outputs/catboost_objective_acceptance/<UTC start time, YYYY-MM-DDTHHMMSSZ>/

    probe-neural-batching
                   the neural batching acceptance (docs/gpu-acceptance-probe.md, Section 9,
                   checks B1-B5), on synthetic data only: accumulated against unsplit gradients
                   (unequal micro-batches, non-uniform weights, the final batch of 1,016 rows,
                   padded attention heads), same-seed repeatability under a micro-batch plan,
                   and the corners of the neural search spaces through the preflight; writes
                   batching.json, the generated batching.md and its log, always batching.log
                   (--log-file is refused), to a new directory: --output, or by default
                   outputs/neural_batching_acceptance/<UTC start time, YYYY-MM-DDTHHMMSSZ>/

    plan-search    the search plan of DR-015 §1 and §7, before any fit: the candidate list of each
                   model and the predicted cost of the lists, as records and a generated
                   search_plan.md, and its log, always plan.log (--log-file is refused), in a
                   new directory: --output, or by default
                   outputs/search_plan/<UTC start time, YYYY-MM-DDTHHMMSSZ>/. Reads no data and
                   fits no model; whether the predicted cost is feasible is decided explicitly
                   before any run

    fit-neural-preprocessing
                   the shared Train-only preprocessing of the neural models (DR-012), fitted once
                   with the declared preprocessing seed: the artifact (manifest.json, arrays.npz
                   and its SHA-256), the Validation 1 and Validation 2 diagnostics, result.json
                   and its log, always run.log (--log-file is refused), in a new
                   directory: --output, or by default
                   outputs/neural_preprocessing/<UTC start time, YYYY-MM-DDTHHMMSSZ>/. Reads the
                   verified processed development files only; the Test period is never read.

    preflight-neural
                   the micro-batch preflight of the neural models (DR-015 §7) on the project GPU:
                   for each of the ten candidates of the ResNet-style MLP and FT-Transformer, the
                   largest micro-batch whose training epoch of two effective batches and whose
                   float32 evaluation inside the training loop (one evaluation batch, optimiser
                   state alive) leave at least the declared free GPU memory at the sampled peak;
                   synthetic inputs of the vocabularies of the shared preprocessing artifact
                   (--preprocessing), no research data. Writes preflight.json and its
                   log, always run.log (--log-file is refused), to a new directory: --output, or
                   by default outputs/neural_preflight/<UTC start time, YYYY-MM-DDTHHMMSSZ>/

    run-development
                   the development run of the eight configurations (LightGBM, CatBoost,
                   ResNet-style MLP and FT-Transformer, each unweighted and balanced) and the
                   logistic-regression reference, which runs first (--configurations chooses a
                   subset, the reference is named LR): the declared search for each
                   configuration (DR-015 §1), selection on V1, scoring of V1 and V2, R3 and
                   ranking outcomes, and the computational-cost timings; records, a generated
                   summary.md and its log, always run.log (--log-file is refused), in a new
                   directory inside outputs/: --output, or by default
                   outputs/development/<UTC start time, YYYY-MM-DDTHHMMSSZ>/. Starts only from
                   the search plan --plan whose cost_prediction.json has the SHA-256 given in
                   --approved-cost (the approval of the predicted cost). The neural models and
                   the reference need the shared preprocessing artifact (--preprocessing) and
                   the neural models the micro-batch preflight (--preflight), made under the
                   same artifact, declaration, settings, package source and GPU. The timing
                   session conditions are recorded as context. Reads the verified processed
                   development files only; the Test period is never read. Consequential: run
                   only after its scope is approved

    run-test-evaluation
                   the single authorised Test evaluation, phase 1 (--phase scoring): from a clean
                   tree at the commit of the annotated freeze tag (--tag), with the closed
                   development run (--development) and analysis runs (--analysis), it verifies
                   every frozen input by hash, reads the Test features without the target and
                   scores them once with the eight frozen models and the reference; the scores,
                   the run record and the phase records are written to a new directory inside
                   outputs/confirmatory/ (--output) or, to repeat a failed scoring attempt,
                   into the unfinished run --resume. The Test labels are never read in this
                   phase. A lock and a consumption marker (outputs/confirmatory/state/) keep
                   the evaluation to one pass. Consequential: run only after the protocol freeze
                   and an explicit approval

    analyse-development
                   the post-selection analyses of a completed development run (--source), in a
                   new analysis run that never changes the source run: stages thresholds
                   (R0–R2 and their workload, R1 selections), calibration (Platt and isotonic
                   calibrators, R4 and the RQ3 cost analysis), h3 (the H3 configuration selection
                   and its label-latency check), refits (the fixed-length refits and the
                   CatBoost CPU–GPU comparison; they train models, using the preprocessing
                   artifact and the batch plans that the source run recorded) and rolling (the
                   rolling-origin supplement for the H3 configuration: refits on the days before
                   each origin and the transfer of the R2 threshold; it needs the h3 stage in
                   the same run). --stages chooses
                   the stages (default: thresholds, calibration, h3); each stage can be repeated
                   in a new analysis run. Writes records, a generated summary.md and its log,
                   always run.log (--log-file is refused), to a new directory inside outputs/:
                   --output, or by
                   default outputs/analysis/<UTC start time, YYYY-MM-DDTHHMMSSZ>/. Reads the
                   verified processed development files only; the Test period is never read

    refit-fresh    one fixed-length refit of a selected setting, fitted and saved in a fresh
                   process (DR-013 §6; DR-015 §4); an internal step of the refit stage, not run
                   by hand

    summarise-run  regenerate summary.md of a development run directory from its records alone
                   (also after an interrupted run); reads no data and fits no model

    promote-evidence
                   copy a completed development run's aggregate evidence (records, summary.md,
                   run.log and a manifest with SHA-256 values) to evidence/<run type>/<run id>/,
                   a new directory; never row-level scores, fitted preprocessing artifacts or
                   models. A separate, explicit step; no run calls it

Every message is logged with a timestamp to the terminal and, with --log-file PATH, also to a
file kept with the run's evidence. --verbose adds detail; --quiet shows warnings and errors only.

Exit codes: 0 success (for probe-gpu: pass, or pending second run; for probe-catboost-objective
and probe-neural-batching: pass); 1 profile-data found differences, or a probe did not pass; 2
configuration or project-root error; 3 data-validation error; 4 sealed Test period; 5 file
access (a required file cannot be read or an output or log file cannot be written); 6
environment mismatch (for example, a LightGBM library that is not the recorded build); 7
model failure (a model failed to fit or produced invalid output); 130 interrupted. An
expected failure is summarised in one line with its exit code (steps that were running also
log that they failed); the log file always keeps its traceback and cause, the terminal only
with --debug. Any other error is a defect: it is logged with its traceback and raised.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from review_capacity.config import load_dataset_spec, load_feature_spec
from review_capacity.data.catalog import read_catalog
from review_capacity.data.duplicates import audit_duplicates, audit_summary
from review_capacity.data.migration import migrate_provenance
from review_capacity.data.preparation import prepare_data
from review_capacity.data.processed import JOINED, read_development_features
from review_capacity.data.profiling import compare_with_catalog
from review_capacity.errors import (
    DataValidationError,
    EnvironmentMismatchError,
    FileAccessError,
    ModelError,
    ProjectRootNotFoundError,
    ReviewCapacityError,
    SealedTestError,
    SpecificationError,
)
from review_capacity.files import write_atomically, write_json
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import configure_logging
from review_capacity.models.registry import RUN_CONFIGURATIONS
from review_capacity.paths import ProjectPaths
from review_capacity.probe.records import (
    ACCEPTANCE_LOG_NAME,
    ACCEPTANCE_ROOT,
    BATCHING_LOG_NAME,
    BATCHING_ROOT,
    LOG_NAME,
    check_acceptance_output,
    check_output,
)
from review_capacity.provenance import code_provenance, write_record

if TYPE_CHECKING:
    from review_capacity.confirmatory.command import TestRunOptions
    from review_capacity.development.analysis import AnalysisOptions
    from review_capacity.development.runner import DevelopmentOptions

logger = logging.getLogger(__name__)

EXIT_DIFFERENCES = 1
# Run outputs stay inside the repository, in Git-ignored, dated directories (Issue #96)
PROBE_OUTPUT_ROOT = Path("outputs") / "gpu_acceptance_probe"
EXIT_CODES: tuple[tuple[type[ReviewCapacityError], int], ...] = (
    (SpecificationError, 2),
    (ProjectRootNotFoundError, 2),
    (DataValidationError, 3),
    (SealedTestError, 4),
    (FileAccessError, 5),
    (EnvironmentMismatchError, 6),
    (ModelError, 7),
)
EXIT_INTERRUPTED = 130


def _prepare_data(args: argparse.Namespace) -> int:
    paths = ProjectPaths.discover(args.root)
    logger.info("Project root: %s", paths.root)
    result = prepare_data(
        load_feature_spec(paths.features_spec),
        load_dataset_spec(paths),
        paths,
        write_processed=not args.check_only,
    )
    for name, digest in result.checksums.items():
        logger.info("Raw file %s: SHA-256 %s (matches the recorded value)", name, digest)
    for s in result.summaries:
        logger.info(
            "Partition %s, days %d-%d: %d rows, TransactionDT %d-%d, TransactionID hash %s",
            s.name,
            s.first_day,
            s.last_day,
            s.rows,
            s.transaction_dt_min,
            s.transaction_dt_max,
            s.transaction_id_sha256[:16],
        )
    logger.info("Validation contract: passed")
    for path in result.written:
        logger.info("Written: %s", path)
    return 0


def _migrate_provenance(args: argparse.Namespace) -> int:
    paths = ProjectPaths.discover(args.root)
    logger.info("Project root: %s", paths.root)
    result = migrate_provenance(
        paths,
        load_feature_spec(paths.features_spec),
        load_dataset_spec(paths),
        check_only=args.check_only,
    )
    for stem in result.current:
        logger.info("%s: already records the current fingerprint", stem)
    verb = "would be migrated" if result.check_only else "migrated"
    for stem in result.migrated:
        logger.info("%s: record %s; the data file is unchanged", stem, verb)
    return 0


def _profile_data(args: argparse.Namespace) -> int:
    paths = ProjectPaths.discover(args.root)
    logger.info("Project root: %s", paths.root)
    features = load_feature_spec(paths.features_spec)
    dataset = load_dataset_spec(paths)
    catalog = read_catalog(paths.feature_catalog)
    data = read_development_features(paths, features, dataset)
    report = compare_with_catalog(data, features, dataset, catalog)
    logger.info(
        "Development rows only (relative days 1-%d); no labels read",
        dataset.last_development_day,
    )
    logger.info(
        "Checked %d catalog statistics: %d differences", report.checks, len(report.differences)
    )
    for reason in report.excluded:
        logger.info("Not checked by design: %s", reason)
    for d in report.differences:
        logger.warning(
            "%s: %s: catalog %r, observed %r", d.column, d.statistic, d.catalog, d.observed
        )
    if args.save:
        path = paths.processed_dir / "feature_profile.csv"
        record_path = paths.processed_dir / "feature_profile.provenance.json"
        try:
            record_path.unlink(missing_ok=True)  # never an old record beside a new profile
        except OSError as error:
            raise FileAccessError(f"{record_path}: cannot be removed ({error.strerror})") from error
        write_atomically(path, lambda target: report.profile.to_csv(target))
        write_record(
            record_path,
            {
                "file": path.name,
                "sha256": file_sha256(path),
                "rows": len(report.profile),
                "inputs": {
                    paths.processed_file(JOINED).name: data.record.sha256,
                    paths.feature_catalog.name: file_sha256(paths.feature_catalog),
                },
                "configuration": {
                    config.name: file_sha256(config)
                    for config in (paths.features_spec, paths.dataset_spec)
                },
                "checks": report.checks,
                "differences": len(report.differences),
                "code": code_provenance(paths.root),
            },
        )
        logger.info("Written: %s and %s", path, record_path.name)
    return EXIT_DIFFERENCES if report.differences else 0


def _audit_duplicates(args: argparse.Namespace) -> int:
    paths = ProjectPaths.discover(args.root)
    logger.info("Project root: %s", paths.root)
    features = load_feature_spec(paths.features_spec)
    dataset = load_dataset_spec(paths)
    data = read_development_features(paths, features, dataset)
    audit = audit_duplicates(data, features, dataset)
    logger.info(
        "Development rows only (relative days 1-%d); no labels read; nothing removed",
        dataset.last_development_day,
    )
    for d in audit.definitions:
        logger.info("Definition %s: %d compared columns", d.definition, len(d.columns))
        for name, c in d.within.items():
            logger.info(
                "  %s: %d rows, %d repeated vectors, %d rows in them, %d excess rows",
                name,
                c.rows,
                c.groups,
                c.rows_in_groups,
                c.excess_rows,
            )
        logger.info(
            "  across partitions: %d vectors in more than one partition, on %d rows",
            d.vectors_in_several_partitions,
            d.rows_with_such_vectors,
        )
        for s in d.shared:
            logger.info("  shared by %s: %d vectors", " and ".join(s.partitions), s.vectors)
    if args.save:
        path = paths.processed_dir / "duplicate_audit.json"
        write_json(
            path,
            {
                "audit": audit_summary(audit),
                "inputs": {paths.processed_file(JOINED).name: data.record.sha256},
                "configuration": {
                    config.name: file_sha256(config)
                    for config in (paths.features_spec, paths.dataset_spec)
                },
                "code": code_provenance(paths.root),
            },
        )
        logger.info("Written: %s", path)
    return 0


def _probe_paths(args: argparse.Namespace, paths: ProjectPaths) -> tuple[Path, Path | None]:
    """The output directory and the first run's record, relative paths taken from the root.
    Without ``--output``, a new directory named by the UTC start time under
    ``PROBE_OUTPUT_ROOT``; it is fixed on ``args`` at the first call, so that the log file and
    the run use the same directory."""
    if args.output is None:
        args.output = PROBE_OUTPUT_ROOT / datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    output: Path = args.output if args.output.is_absolute() else paths.root / args.output
    previous: Path | None = args.previous
    if previous is not None and not previous.is_absolute():
        previous = paths.root / previous
    return output, previous


def _probe_gpu(args: argparse.Namespace) -> int:
    from review_capacity.probe.runner import run_probe  # PyTorch and both tree libraries

    paths = ProjectPaths.discover(args.root)
    logger.info("Project root: %s", paths.root)
    output, previous = _probe_paths(args, paths)
    record = run_probe(paths, output, previous=previous, log_opened=True)
    logger.info("Probe result: %s", record["overall"])
    return 0 if record["overall"] in ("pass", "pending second run") else EXIT_DIFFERENCES


def _acceptance_output(args: argparse.Namespace, paths: ProjectPaths) -> Path:
    """The acceptance run's output directory: ``--output``, or by default a new directory named
    by the UTC start time under ``ACCEPTANCE_ROOT``, fixed on ``args`` at the first call."""
    if args.output is None:
        args.output = ACCEPTANCE_ROOT / datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    output: Path = args.output if args.output.is_absolute() else paths.root / args.output
    return output


def _probe_catboost_objective(args: argparse.Namespace) -> int:
    from review_capacity.probe.catboost_objective import run_acceptance  # CatBoost and PyTorch

    paths = ProjectPaths.discover(args.root)
    logger.info("Project root: %s", paths.root)
    record = run_acceptance(paths, _acceptance_output(args, paths), log_opened=True)
    logger.info("Acceptance result: %s", record["overall"])
    return 0 if record["overall"] == "pass" else EXIT_DIFFERENCES


def _batching_output(args: argparse.Namespace, paths: ProjectPaths) -> Path:
    """The neural batching acceptance's output directory: ``--output``, or by default a new
    directory named by the UTC start time under ``BATCHING_ROOT``, fixed on ``args`` at the first
    call."""
    if args.output is None:
        args.output = BATCHING_ROOT / datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    output: Path = args.output if args.output.is_absolute() else paths.root / args.output
    return output


def _probe_neural_batching(args: argparse.Namespace) -> int:
    from review_capacity.probe.neural_batching import run_acceptance  # PyTorch

    paths = ProjectPaths.discover(args.root)
    logger.info("Project root: %s", paths.root)
    record = run_acceptance(paths, _batching_output(args, paths), log_opened=True)
    logger.info("Acceptance result: %s", record["overall"])
    return 0 if record["overall"] == "pass" else EXIT_DIFFERENCES


def _plan_output(args: argparse.Namespace, paths: ProjectPaths) -> Path:
    """The plan's output directory. Without ``--output``, a new directory named by the UTC start
    time under ``plan.OUTPUT_ROOT``; the start time is fixed on ``args`` at the first call, so that
    the log file, the records and the directory agree."""
    from review_capacity.training import plan

    if getattr(args, "started", None) is None:
        args.started = plan.run_id()
    if args.output is None:
        args.output = plan.OUTPUT_ROOT / args.started
    output: Path = args.output if args.output.is_absolute() else paths.root / args.output
    return output


def _plan_search(args: argparse.Namespace) -> int:
    from review_capacity.development.records import DevelopmentPartitions
    from review_capacity.models.run_record import load_settings_with_snapshot
    from review_capacity.training import plan
    from review_capacity.training.declaration import load_declaration

    paths = ProjectPaths.discover(args.root)
    logger.info("Project root: %s", paths.root)
    output = _plan_output(args, paths)
    settings, settings_file = load_settings_with_snapshot(paths)
    declaration, declaration_file = load_declaration(paths)
    partitions = DevelopmentPartitions.from_spec(load_dataset_spec(paths), paths)
    run = plan.identity(
        paths.root,
        args.started,
        declaration,
        declaration_file,
        settings_file,
        settings.main_seed,
        paths.manifest,
    )
    plan.write_plan(output, args.started, run, declaration, partitions)
    logger.info("Search plan written to %s", output)
    return 0


def _preprocessing_directory(args: argparse.Namespace, paths: ProjectPaths) -> Path:
    """The new directory of ``fit-neural-preprocessing``; the start time is fixed on ``args`` at
    the first call (shown to be free then; the log file is opened in it afterwards), so that the
    log file, the records and the directory agree."""
    from review_capacity.development.preprocessing_run import output_directory
    from review_capacity.development.run_output import run_id

    if getattr(args, "directory", None) is None:
        if getattr(args, "started", None) is None:
            args.started = run_id()
        args.directory = output_directory(paths, args.started, args.output)
    directory: Path = args.directory
    return directory


def _fit_neural_preprocessing(args: argparse.Namespace) -> int:
    from review_capacity.development.preprocessing_run import run_fit

    paths = ProjectPaths.discover(args.root)
    logger.info("Project root: %s", paths.root)
    directory = _preprocessing_directory(args, paths)
    features, dataset = load_feature_spec(paths.features_spec), load_dataset_spec(paths)
    run_fit(paths, features, dataset, directory, args.started)
    logger.info("Neural preprocessing written to %s", directory)
    return 0


def _preflight_directory(args: argparse.Namespace, paths: ProjectPaths) -> Path:
    """The new directory of ``preflight-neural``, fixed on ``args`` at the first call (shown to
    be free then; the log file is opened in it afterwards)."""
    from review_capacity.development.preflight_run import output_directory
    from review_capacity.development.run_output import run_id

    if getattr(args, "directory", None) is None:
        if getattr(args, "started", None) is None:
            args.started = run_id()
        args.directory = output_directory(paths, args.started, args.output)
    directory: Path = args.directory
    return directory


def _preflight_neural(args: argparse.Namespace) -> int:
    from review_capacity.development.preflight_run import run_preflight_command

    paths = ProjectPaths.discover(args.root)
    logger.info("Project root: %s", paths.root)
    directory = _preflight_directory(args, paths)
    features = load_feature_spec(paths.features_spec)
    run_preflight_command(paths, features, directory, args.started, args.preprocessing.resolve())
    logger.info("Neural preflight written to %s", directory)
    return 0


def _development_options(args: argparse.Namespace) -> DevelopmentOptions:
    """The options of ``run-development``; the start time is fixed on ``args`` at the first call,
    so that the log file, the records and the directory agree."""
    from review_capacity.development.run_output import run_id
    from review_capacity.development.runner import DevelopmentOptions

    if getattr(args, "started", None) is None:
        args.started = run_id()
    return DevelopmentOptions(
        plan=args.plan.resolve(),
        approved_cost_sha256=args.approved_cost,
        configurations=tuple(args.configurations or RUN_CONFIGURATIONS),
        preprocessing=args.preprocessing,
        preflight=args.preflight,
        output=args.output,
        started=args.started,
    )


def _development_directory(args: argparse.Namespace, paths: ProjectPaths) -> Path:
    """The new run directory of ``run-development``, shown to be free for a new run."""
    from review_capacity.development.run_output import check_new_directory
    from review_capacity.development.runner import run_directory_for

    options = _development_options(args)
    return check_new_directory(run_directory_for(paths, options, args.started), paths.root)


def _run_development(args: argparse.Namespace) -> int:
    from review_capacity.development.runner import prepare_development, run_development

    paths = ProjectPaths.discover(args.root)
    logger.info("Project root: %s", paths.root)
    setup = prepare_development(paths, _development_options(args))
    directory = run_development(setup)
    logger.info("Development run written to %s", directory)
    return 0


def _refit_fresh(args: argparse.Namespace) -> int:
    from review_capacity.development.refit_process import run_job_file

    seconds = run_job_file(args.job)
    logger.info("Refit fitted and saved in %.1f s", seconds)
    return 0


def _analysis_options(args: argparse.Namespace) -> AnalysisOptions:
    """The options of ``analyse-development``; the start time is fixed on ``args`` at the first
    call, so that the log file, the records and the directory agree."""
    from review_capacity.development.analysis import DEFAULT_STAGES, AnalysisOptions
    from review_capacity.development.run_output import run_id

    if getattr(args, "started", None) is None:
        args.started = run_id()
    return AnalysisOptions(
        source=args.source.resolve(),
        stages=tuple(args.stages or DEFAULT_STAGES),
        output=args.output,
        started=args.started,
    )


def _analysis_directory(args: argparse.Namespace, paths: ProjectPaths) -> Path:
    """The new run directory of ``analyse-development``, shown to be free for a new run."""
    from review_capacity.development.analysis import analysis_directory
    from review_capacity.development.run_output import check_new_directory

    options = _analysis_options(args)
    return check_new_directory(analysis_directory(paths, options, args.started), paths.root)


def _analyse_development(args: argparse.Namespace) -> int:
    from review_capacity.development.analysis import prepare_analysis, run_analysis

    paths = ProjectPaths.discover(args.root)
    logger.info("Project root: %s", paths.root)
    setup = prepare_analysis(paths, _analysis_options(args))
    directory = run_analysis(setup)
    logger.info("Analysis run written to %s", directory)
    return 0


def _test_options(args: argparse.Namespace) -> TestRunOptions:
    """The options of ``run-test-evaluation``; the start time is fixed on ``args`` at the first
    call, so that the log file, the records and the directory agree."""
    from review_capacity.confirmatory.command import TestRunOptions
    from review_capacity.development.run_output import run_id

    if getattr(args, "started", None) is None:
        args.started = run_id()
    return TestRunOptions(
        tag=args.tag,
        development=args.development.resolve(),
        analyses=tuple(path.resolve() for path in args.analysis),
        phase=args.phase,
        output=args.output,
        resume=None if args.resume is None else args.resume.resolve(),
        started=args.started,
    )


def _test_log(args: argparse.Namespace, paths: ProjectPaths) -> Path:
    """The log file of ``run-test-evaluation``: ``run.log`` of a new run directory (shown to be
    free) or a file of the resumed run's directory named by this start time."""
    from review_capacity.confirmatory.command import RUN_TYPE, run_log_name
    from review_capacity.development.run_output import check_new_directory, default_directory

    options = _test_options(args)
    if options.resume is not None:
        return options.resume / run_log_name(args.started, resume=True)
    target = options.output or default_directory(paths.root, RUN_TYPE, args.started)
    return check_new_directory(target, paths.root) / run_log_name(args.started, resume=False)


def _run_test_evaluation(args: argparse.Namespace) -> int:
    from review_capacity.confirmatory.command import run_test_scoring

    paths = ProjectPaths.discover(args.root)
    logger.info("Project root: %s", paths.root)
    directory = run_test_scoring(paths, _test_options(args))
    logger.info("Test evaluation, scoring phase written to %s", directory)
    return 0


def _summarise_run(args: argparse.Namespace) -> int:
    from review_capacity.development.records import DevelopmentPartitions
    from review_capacity.development.rendering import write_summary

    paths = ProjectPaths.discover(args.root)
    partitions = DevelopmentPartitions.from_spec(load_dataset_spec(paths), paths)
    write_summary(args.run_directory.resolve(), partitions, paths.root)
    return 0


def _promote_evidence(args: argparse.Namespace) -> int:
    from review_capacity.development.promotion import promote, verify_promoted
    from review_capacity.development.records import DevelopmentPartitions

    paths = ProjectPaths.discover(args.root)
    partitions = DevelopmentPartitions.from_spec(load_dataset_spec(paths), paths)
    destination = promote(
        args.run_directory, paths.root, partitions, evidence_root=args.evidence_root
    )
    verify_promoted(destination)
    logger.info("Evidence promoted to %s", destination)
    return 0


def _log_file(args: argparse.Namespace) -> Path | None:
    """The log file: ``--log-file``; for ``probe-gpu`` always ``probe.log`` and for ``plan-search``
    ``plan.log`` in its output directory (Section 5 of the probe specification), and only once the
    directory is shown to be free for a new run, so that no earlier run's log is ever written to.
    Raises before any file is opened."""
    if args.command == "plan-search":
        from review_capacity.training import plan

        if args.log_file is not None:
            raise SpecificationError(
                f"plan-search writes its log to {plan.LOG_NAME} in its output directory; "
                "omit --log-file"
            )
        output = _plan_output(args, ProjectPaths.discover(args.root))
        plan.check_output(output)
        return output / plan.LOG_NAME
    if args.command == "fit-neural-preprocessing":
        from review_capacity.development.preprocessing_run import LOG_NAME as FIT_LOG_NAME

        if args.log_file is not None:
            raise SpecificationError(
                f"fit-neural-preprocessing writes its log to {FIT_LOG_NAME} in its output "
                "directory; omit --log-file"
            )
        return _preprocessing_directory(args, ProjectPaths.discover(args.root)) / FIT_LOG_NAME
    if args.command == "preflight-neural":
        from review_capacity.development.preflight_run import LOG_NAME as PREFLIGHT_LOG_NAME

        if args.log_file is not None:
            raise SpecificationError(
                f"preflight-neural writes its log to {PREFLIGHT_LOG_NAME} in its output "
                "directory; omit --log-file"
            )
        return _preflight_directory(args, ProjectPaths.discover(args.root)) / PREFLIGHT_LOG_NAME
    if args.command == "run-development":
        from review_capacity.development.run_output import LOG_NAME as RUN_LOG_NAME

        if args.log_file is not None:
            raise SpecificationError(
                f"run-development writes its log to {RUN_LOG_NAME} in its output directory; "
                "omit --log-file"
            )
        return _development_directory(args, ProjectPaths.discover(args.root)) / RUN_LOG_NAME
    if args.command == "run-test-evaluation":
        if args.log_file is not None:
            raise SpecificationError(
                "run-test-evaluation writes its log into its run directory; omit --log-file"
            )
        return _test_log(args, ProjectPaths.discover(args.root))
    if args.command == "analyse-development":
        from review_capacity.development.run_output import LOG_NAME as ANALYSIS_LOG_NAME

        if args.log_file is not None:
            raise SpecificationError(
                f"analyse-development writes its log to {ANALYSIS_LOG_NAME} in its output "
                "directory; omit --log-file"
            )
        return _analysis_directory(args, ProjectPaths.discover(args.root)) / ANALYSIS_LOG_NAME
    if args.command == "probe-catboost-objective":
        if args.log_file is not None:
            raise SpecificationError(
                f"probe-catboost-objective writes its log to {ACCEPTANCE_LOG_NAME} in its output "
                "directory; omit --log-file"
            )
        output = _acceptance_output(args, ProjectPaths.discover(args.root))
        check_acceptance_output(output)
        return output / ACCEPTANCE_LOG_NAME
    if args.command == "probe-neural-batching":
        if args.log_file is not None:
            raise SpecificationError(
                f"probe-neural-batching writes its log to {BATCHING_LOG_NAME} in its output "
                "directory; omit --log-file"
            )
        output = _batching_output(args, ProjectPaths.discover(args.root))
        check_acceptance_output(output, log_name=BATCHING_LOG_NAME)
        return output / BATCHING_LOG_NAME
    if args.command != "probe-gpu":
        log_file: Path | None = args.log_file
        return log_file
    if args.log_file is not None:
        raise SpecificationError(
            f"probe-gpu writes its log to {LOG_NAME} in its output directory; omit --log-file"
        )
    paths = ProjectPaths.discover(args.root)
    output, previous = _probe_paths(args, paths)
    check_output(output, previous)
    return output / LOG_NAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-capacity",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", help="also log detailed messages")
    verbosity.add_argument("--quiet", action="store_true", help="log warnings and errors only")
    parser.add_argument("--log-file", type=Path, default=None, help="also log to this file")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="also show the traceback of project errors on the terminal (the log file has it)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare-data",
        help="validate the raw data; write the manifest and the development files",
    )
    prepare.add_argument("--root", type=Path, default=None, help="project root directory")
    prepare.add_argument(
        "--check-only", action="store_true", help="validate and report; write nothing"
    )
    prepare.set_defaults(handler=_prepare_data)
    migrate = commands.add_parser(
        "migrate-provenance",
        help="record the feature-declaration fingerprint in the processed files' provenance",
    )
    migrate.add_argument("--root", type=Path, default=None, help="project root directory")
    migrate.add_argument(
        "--check-only", action="store_true", help="verify and report; write nothing"
    )
    migrate.set_defaults(handler=_migrate_provenance)
    profile = commands.add_parser(
        "profile-data",
        help="recompute the feature-catalog statistics from the processed development file",
    )
    profile.add_argument("--root", type=Path, default=None, help="project root directory")
    profile.add_argument(
        "--save", action="store_true", help="also write the profile to data/processed/"
    )
    profile.set_defaults(handler=_profile_data)
    duplicates = commands.add_parser(
        "audit-duplicates",
        help="count repeated input vectors in the development partitions (report only)",
    )
    duplicates.add_argument("--root", type=Path, default=None, help="project root directory")
    duplicates.add_argument(
        "--save", action="store_true", help="also write the counts to data/processed/"
    )
    duplicates.set_defaults(handler=_audit_duplicates)
    probe = commands.add_parser(
        "probe-gpu",
        help="run the GPU acceptance probe of DR-013 §8 on synthetic data",
    )
    probe.add_argument("--root", type=Path, default=None, help="project root directory")
    probe.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"new directory for this run's records (default: {PROBE_OUTPUT_ROOT}/<UTC time>)",
    )
    probe.add_argument(
        "--previous",
        type=Path,
        default=None,
        help="the first run's probe.json, to compare a second run with it (check C11)",
    )
    probe.set_defaults(handler=_probe_gpu)
    acceptance = commands.add_parser(
        "probe-catboost-objective",
        help="run the CatBoost tree-objective acceptance (probe specification, Section 8)",
    )
    acceptance.add_argument("--root", type=Path, default=None, help="project root directory")
    acceptance.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"new directory for this run's records (default: {ACCEPTANCE_ROOT}/<UTC time>)",
    )
    acceptance.set_defaults(handler=_probe_catboost_objective)
    batching = commands.add_parser(
        "probe-neural-batching",
        help="run the neural batching acceptance (probe specification, Section 9)",
    )
    batching.add_argument("--root", type=Path, default=None, help="project root directory")
    batching.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"new directory for this run's records (default: {BATCHING_ROOT}/<UTC time>)",
    )
    batching.set_defaults(handler=_probe_neural_batching)
    plan_search = commands.add_parser(
        "plan-search",
        help="generate the candidate lists and predict their cost, before any fit (DR-015)",
    )
    plan_search.add_argument("--root", type=Path, default=None, help="project root directory")
    plan_search.add_argument(
        "--output",
        type=Path,
        default=None,
        help="new directory for the plan (default: outputs/search_plan/<UTC time>)",
    )
    plan_search.set_defaults(handler=_plan_search)
    fit_preprocessing = commands.add_parser(
        "fit-neural-preprocessing",
        help="fit the shared Train-only neural preprocessing artifact (DR-012)",
    )
    fit_preprocessing.add_argument("--root", type=Path, default=None, help="project root directory")
    fit_preprocessing.add_argument(
        "--output",
        type=Path,
        default=None,
        help="new directory for the artifact (default: outputs/neural_preprocessing/<UTC time>)",
    )
    fit_preprocessing.set_defaults(handler=_fit_neural_preprocessing)
    preflight = commands.add_parser(
        "preflight-neural",
        help="micro-batch preflight of the neural models on the project GPU (DR-015 §7)",
    )
    preflight.add_argument("--root", type=Path, default=None, help="project root directory")
    preflight.add_argument(
        "--preprocessing",
        type=Path,
        required=True,
        help="directory of the shared preprocessing artifact (fit-neural-preprocessing)",
    )
    preflight.add_argument(
        "--output",
        type=Path,
        default=None,
        help="new directory for the preflight (default: outputs/neural_preflight/<UTC time>)",
    )
    preflight.set_defaults(handler=_preflight_neural)
    development = commands.add_parser(
        "run-development",
        help="the development run: search, selection, scoring, evaluation and timing",
    )
    development.add_argument("--root", type=Path, default=None, help="project root directory")
    development.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="the search plan directory written by plan-search",
    )
    development.add_argument(
        "--approved-cost",
        required=True,
        metavar="SHA256",
        help="SHA-256 of the plan's cost_prediction.json that was approved",
    )
    development.add_argument(
        "--configurations",
        nargs="+",
        choices=RUN_CONFIGURATIONS,
        default=None,
        help="the configurations to run (default: the reference LR and all eight configurations)",
    )
    development.add_argument(
        "--preprocessing",
        type=Path,
        default=None,
        help="directory of the shared preprocessing artifact (needed by the neural models and LR)",
    )
    development.add_argument(
        "--preflight",
        type=Path,
        default=None,
        help="the preflight.json of preflight-neural (needed by the neural models)",
    )
    development.add_argument(
        "--output",
        type=Path,
        default=None,
        help="new directory for this run (default: outputs/development/<UTC time>)",
    )
    development.set_defaults(handler=_run_development)
    test_evaluation = commands.add_parser(
        "run-test-evaluation",
        help="the single authorised Test evaluation, phase 1: the frozen models score the Test "
        "features once, without labels",
    )
    test_evaluation.add_argument("--root", type=Path, default=None, help="project root directory")
    test_evaluation.add_argument(
        "--tag", required=True, help="the annotated protocol freeze tag; HEAD must be its commit"
    )
    test_evaluation.add_argument(
        "--development",
        type=Path,
        required=True,
        help="the closed development run that trained the frozen models",
    )
    test_evaluation.add_argument(
        "--analysis",
        type=Path,
        nargs="+",
        required=True,
        help="the closed analysis runs that fixed the thresholds, calibrators and H3 selection",
    )
    test_evaluation.add_argument(
        "--phase",
        default="scoring",
        choices=["scoring"],
        help="the phase to run (default: scoring)",
    )
    test_evaluation.add_argument(
        "--output",
        type=Path,
        default=None,
        help="new directory for this run (default: outputs/confirmatory/<UTC time>)",
    )
    test_evaluation.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="the unfinished run directory in which to repeat a failed scoring attempt",
    )
    test_evaluation.set_defaults(handler=_run_test_evaluation)
    analysis = commands.add_parser(
        "analyse-development",
        help="the post-selection analyses of a completed development run, in a new run",
    )
    analysis.add_argument("--root", type=Path, default=None, help="project root directory")
    analysis.add_argument(
        "--source",
        type=Path,
        required=True,
        help="the completed development run to analyse (a directory under outputs/development/)",
    )
    analysis.add_argument(
        "--stages",
        nargs="+",
        default=None,
        metavar="STAGE",
        help="thresholds, calibration, h3, refits and rolling (default: thresholds, calibration, "
        "h3; refits and rolling train models and are named explicitly; the neural refits use "
        "the preprocessing artifact and the batch plans recorded in the source run; rolling is "
        "the rolling-origin supplement for the H3 configuration and needs h3 in the same run)",
    )
    analysis.add_argument(
        "--output",
        type=Path,
        default=None,
        help="new directory for this run (default: outputs/analysis/<UTC time>)",
    )
    analysis.set_defaults(handler=_analyse_development)
    refit_fresh = commands.add_parser(
        "refit-fresh",
        help="one fixed-length refit in a fresh process (internal step of the refit stage)",
    )
    refit_fresh.add_argument("--root", type=Path, default=None, help="project root directory")
    refit_fresh.add_argument("--job", type=Path, required=True, help="the job file of the refit")
    refit_fresh.set_defaults(handler=_refit_fresh)
    summarise = commands.add_parser(
        "summarise-run",
        help="regenerate summary.md of a development run directory from its records",
    )
    summarise.add_argument("run_directory", type=Path, help="the run's directory under outputs/")
    summarise.add_argument("--root", type=Path, default=None, help="project root directory")
    summarise.set_defaults(handler=_summarise_run)
    promote = commands.add_parser(
        "promote-evidence",
        help="copy a completed development run's aggregate evidence into evidence/",
    )
    promote.add_argument("run_directory", type=Path, help="the run's directory under outputs/")
    promote.add_argument("--root", type=Path, default=None, help="project root directory")
    promote.add_argument(
        "--evidence-root",
        type=Path,
        default=None,
        help="directory receiving evidence/<run type>/<run id>/ (default: evidence)",
    )
    promote.set_defaults(handler=_promote_evidence)
    return parser


def _exit_code(error: ReviewCapacityError) -> int:
    for kind, code in EXIT_CODES:
        if isinstance(error, kind):
            return code
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    handler: Callable[[argparse.Namespace], int] = args.handler
    own_log = args.command in (
        "probe-gpu",
        "probe-catboost-objective",
        "probe-neural-batching",
        "plan-search",
        "fit-neural-preprocessing",
        "preflight-neural",
        "run-development",
        "analyse-development",
    )
    try:
        configure_logging(level, terminal_tracebacks=args.debug)  # until the log file is known
        configure_logging(
            level,
            _log_file(args),
            terminal_tracebacks=args.debug,
            # these logs are part of the run's evidence: complete even with --quiet, and new
            file_level=min(level, logging.INFO) if own_log else None,
            new_file=own_log,
        )
        logger.info("review-capacity %s", args.command)
        return handler(args)
    except ReviewCapacityError as error:
        logger.exception("%s: %s", type(error).__name__, error)  # traceback: log file only
        return _exit_code(error)
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return EXIT_INTERRUPTED
    except Exception:
        logger.critical("Unexpected error (a defect in the code)", exc_info=True)
        raise
