"""Re-recording the provenance of the processed development files under the feature-declaration
fingerprint (Issue #125).

The processed files were built before ``config.features_fingerprint`` existed; their records
carry only the whole-file hash of ``config/features.yaml``. ``migrate_provenance`` verifies each
file with every check of ``verify_processed`` (SHA-256, rows, columns, types, raw-source hashes,
DR-001 anchors; the recorded whole-file hash must still equal the current file's) and then writes
its record again with the fingerprint added. The data files are never rewritten, so their
SHA-256 values and the DR-001 anchors are unchanged by construction; the original record is
hashed into the new one, and no raw file is read.

All files are verified before any record is written, and a record that already has the current
fingerprint is left as it is, so an interrupted migration is completed by running it again.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from review_capacity.config import DatasetSpec, FeatureSpec, features_fingerprint
from review_capacity.data.processed import FINGERPRINT, read_record, stems, verify_processed
from review_capacity.errors import DataValidationError, FileAccessError
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import log_step
from review_capacity.paths import ProjectPaths
from review_capacity.provenance import code_provenance, write_record

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MigrationResult:
    """The stems whose record was (or, with ``check_only``, would be) written, and those that
    already carried the current fingerprint."""

    migrated: tuple[str, ...]
    current: tuple[str, ...]
    check_only: bool


def _is_current(paths: ProjectPaths, stem: str, fingerprint: str) -> bool:
    record = read_record(paths.provenance_file(stem))
    recorded = record.configuration.get(FINGERPRINT)
    if recorded is None:
        return False
    if recorded != fingerprint:
        raise DataValidationError(
            f"{stem}: recorded {FINGERPRINT} differs from the current declaration; the file was "
            "built under another declaration and cannot be migrated"
        )
    return True


def migrate_provenance(
    paths: ProjectPaths,
    features: FeatureSpec,
    dataset: DatasetSpec,
    *,
    check_only: bool = False,
) -> MigrationResult:
    """Verify every processed development file and record the current fingerprint for each.

    With ``check_only`` the files are verified and nothing is written."""
    fingerprint = features_fingerprint(features)
    pending: list[str] = []
    current: list[str] = []
    for stem in stems(dataset):
        with log_step(logger, f"Verifying {stem}"):
            if _is_current(paths, stem, fingerprint):
                verify_processed(paths, stem, features, dataset)
                current.append(stem)
                continue
            verify_processed(paths, stem, features, dataset, migrating=True)
            pending.append(stem)
    if check_only:
        logger.info("Check only: nothing written; %d record(s) would be migrated", len(pending))
        return MigrationResult(tuple(pending), tuple(current), True)
    code = code_provenance(paths.root)
    for stem in pending:
        _rewrite(paths, stem, fingerprint, code)
    return MigrationResult(tuple(pending), tuple(current), False)


def _rewrite(paths: ProjectPaths, stem: str, fingerprint: str, code: dict[str, Any]) -> None:
    path = paths.provenance_file(stem)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FileAccessError(f"{path}: cannot be read ({error.strerror})") from error
    record["configuration"] = {**record["configuration"], FINGERPRINT: fingerprint}
    record["migration"] = {
        "issue": 125,
        "reason": "feature-declaration fingerprint added to a record built before it existed",
        "previous_record_sha256": file_sha256(path),
        "data_file_unchanged_sha256": record["sha256"],
        "code": code,
    }
    write_record(path, record)
    logger.info("%s: record migrated; the data file is unchanged", path.name)
