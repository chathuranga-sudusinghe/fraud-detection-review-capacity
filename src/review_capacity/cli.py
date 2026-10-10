"""Command-line interface: ``review-capacity <command>`` or ``python -m review_capacity <command>``.

Commands:

    prepare-data   validate the raw data; write the manifest and the development files
                   (--check-only: validate and print, write nothing)

Every command accepts ``--root PATH`` when the project root is not found automatically
(see ``review_capacity.paths``).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from review_capacity.config import load_dataset_spec, load_feature_spec
from review_capacity.data.preparation import prepare_data
from review_capacity.paths import ProjectPaths


def _prepare_data(args: argparse.Namespace) -> int:
    paths = ProjectPaths.discover(args.root)
    print(f"Project root: {paths.root}")
    result = prepare_data(
        load_feature_spec(paths.features_spec),
        load_dataset_spec(paths),
        paths,
        write_processed=not args.check_only,
    )
    print("Raw files: SHA-256 match the recorded values")
    for name, digest in result.checksums.items():
        print(f"  {name}: {digest}")
    print("Partitions (label-free): rows, TransactionDT bounds, TransactionID hash prefix")
    for s in result.summaries:
        print(
            f"  {s.name:5s} days {s.first_day}-{s.last_day}: {s.rows:,} rows, "
            f"{s.transaction_dt_min:,}-{s.transaction_dt_max:,}, "
            f"{s.transaction_id_sha256[:16]}"
        )
    print("Validation contract: passed")
    for path in result.written:
        print(f"Written: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-capacity",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare-data",
        help="validate the raw data; write the manifest and the development files",
    )
    prepare.add_argument("--root", type=Path, default=None, help="project root directory")
    prepare.add_argument(
        "--check-only", action="store_true", help="validate and print the summary; write nothing"
    )
    prepare.set_defaults(handler=_prepare_data)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = args.handler
    return int(handler(args))
