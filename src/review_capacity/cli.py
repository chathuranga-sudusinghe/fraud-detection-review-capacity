"""Command-line interface: ``review-capacity <command>`` or ``python -m review_capacity <command>``.

Commands:

    prepare-data   validate the raw data; write the manifest and the development files, once
                   (--check-only: validate and report, write nothing). The only command that
                   reads the raw files before the confirmatory Test run.
    profile-data   recompute the feature-catalog statistics from the verified processed
                   development file (no labels, no raw file) and report every difference from
                   docs/feature-catalog.md (--save: also write the profile and its provenance
                   record to the processed directory)

Every message is logged with a timestamp to the terminal and, with --log-file PATH, also to a
file kept with the run's evidence. --verbose adds detail; --quiet shows warnings and errors only.

Exit codes: 0 success; 1 profile-data found differences; 2 configuration or project-root error;
3 data-validation error; 4 sealed Test period; 5 file access (a required file cannot be read or an
output or log file cannot be written); 6 environment mismatch (for example, a LightGBM library
that is not the recorded build); 130 interrupted. An expected failure is summarised in one line
with its exit code (steps that were running also log that they failed); the log file always keeps
its traceback and cause, the terminal only with --debug. Any other error is a defect: it is logged
with its traceback and raised.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from review_capacity.config import load_dataset_spec, load_feature_spec
from review_capacity.data.catalog import read_catalog
from review_capacity.data.preparation import prepare_data
from review_capacity.data.processed import JOINED, read_development_features
from review_capacity.data.profiling import compare_with_catalog
from review_capacity.errors import (
    DataValidationError,
    EnvironmentMismatchError,
    FileAccessError,
    ProjectRootNotFoundError,
    ReviewCapacityError,
    SealedTestError,
    SpecificationError,
)
from review_capacity.files import write_atomically
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import configure_logging
from review_capacity.paths import ProjectPaths
from review_capacity.provenance import code_provenance, write_record

logger = logging.getLogger(__name__)

EXIT_DIFFERENCES = 1
EXIT_CODES: tuple[tuple[type[ReviewCapacityError], int], ...] = (
    (SpecificationError, 2),
    (ProjectRootNotFoundError, 2),
    (DataValidationError, 3),
    (SealedTestError, 4),
    (FileAccessError, 5),
    (EnvironmentMismatchError, 6),
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
    try:
        configure_logging(level, args.log_file, terminal_tracebacks=args.debug)
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
