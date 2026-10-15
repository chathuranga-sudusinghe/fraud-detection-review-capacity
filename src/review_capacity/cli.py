"""Command-line interface: ``review-capacity <command>`` or ``python -m review_capacity <command>``.

Commands:

    prepare-data   validate the raw data; write the manifest and the development files, once
                   (--check-only: validate and report, write nothing). The only command that
                   reads the raw files before the confirmatory Test run.
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

Every message is logged with a timestamp to the terminal and, with --log-file PATH, also to a
file kept with the run's evidence. --verbose adds detail; --quiet shows warnings and errors only.

Exit codes: 0 success (for probe-gpu: pass, or pending second run); 1 profile-data found
differences, or probe-gpu did not pass; 2 configuration or project-root error; 3 data-validation
error; 4 sealed Test period; 5 file access (a required file cannot be read or an output or log file
cannot be written); 6 environment mismatch (for example, a LightGBM library that is not the
recorded build); 7 model failure (a model failed to fit or produced invalid output); 130
interrupted. An expected failure is summarised in one line with its exit code (steps that were
running also log that they failed); the log file always keeps its traceback and cause, the terminal
only with --debug. Any other error is a defect: it is logged with its traceback and raised.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from review_capacity.config import load_dataset_spec, load_feature_spec
from review_capacity.data.catalog import read_catalog
from review_capacity.data.duplicates import audit_duplicates, audit_summary
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
from review_capacity.paths import ProjectPaths
from review_capacity.probe.records import LOG_NAME, check_output
from review_capacity.provenance import code_provenance, write_record

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


def _log_file(args: argparse.Namespace) -> Path | None:
    """The log file: ``--log-file``; for ``probe-gpu`` always ``probe.log`` in its output directory
    (Section 5 of the probe specification), and only once the directory is shown to be free for a
    new run, so that no earlier run's log is ever written to. Raises before any file is opened."""
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
    probe = args.command == "probe-gpu"
    try:
        configure_logging(level, terminal_tracebacks=args.debug)  # until the log file is known
        configure_logging(
            level,
            _log_file(args),
            terminal_tracebacks=args.debug,
            # the probe's log is part of its evidence: complete even with --quiet, and new
            file_level=min(level, logging.INFO) if probe else None,
            new_file=probe,
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
