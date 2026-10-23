"""The data-preparation stage: validate the raw data, write the development files and manifest.

Run on the local raw data under the execution rules of AGENTS.md (experiment execution
responsibility) through the command line (``review_capacity.cli``):

    review-capacity prepare-data               # validate, processed files and manifest
    review-capacity prepare-data --check-only  # validate and print; write nothing

This is the one step that reads the raw files before the confirmatory Test run; every later step
reads the processed files through ``data.processed`` (Issue #70 data-access rule). The processed
files are built once: the stage refuses to write over processed files that already exist.

Test-period labels are never read: the stage refuses to start unless the Test partition is
declared sealed. Only development partitions are written to the processed directory; no
Test-period file is written before the protocol freeze.

Write order, so that an interrupted run never leaves a data file beside a record that does not
describe it: for each processed file, the old provenance record is removed, the data file is
replaced atomically, and the new record is written last; the manifest is written after all
processed files.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd

from review_capacity.config import DatasetSpec, FeatureSpec, features_fingerprint
from review_capacity.data.loading import RawData, check_file_checksums, load_labels, load_raw
from review_capacity.data.manifest import build_datapackage, write_datapackage
from review_capacity.data.processed import (
    DAY_COLUMN,
    FINGERPRINT,
    JOINED,
    PARTITION_COLUMN,
    stems,
)
from review_capacity.data.validation import PartitionSummary, validate
from review_capacity.errors import FileAccessError, SealedTestError
from review_capacity.features.model_inputs import build_canonical_frame
from review_capacity.files import write_atomically
from review_capacity.hashing import file_sha256, id_set_sha256
from review_capacity.logging_setup import log_step
from review_capacity.paths import ProjectPaths
from review_capacity.provenance import code_provenance, write_record

PARQUET_ENGINE: Final[Literal["pyarrow"]] = "pyarrow"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparationResult:
    checksums: dict[str, str]
    summaries: list[PartitionSummary]
    written: list[Path]


def _development_frames(
    raw: RawData, labels: pd.Series, features: FeatureSpec, dataset: DatasetSpec
) -> Iterator[tuple[str, str, pd.DataFrame]]:
    """Yield (file stem, scope, frame): the joined development rows, then one model-input
    frame per development partition, built one at a time."""
    development = [p.name for p in dataset.development]
    dev_index = raw.joined.index[raw.partition.isin(development).to_numpy()]
    meta = pd.DataFrame(
        {
            features.key: raw.joined.loc[dev_index, features.key],
            DAY_COLUMN: raw.relative_day.loc[dev_index],
            PARTITION_COLUMN: raw.partition.loc[dev_index].astype(str),
            features.target: labels.loc[dev_index],
        }
    )
    joined = pd.concat([meta, raw.joined.loc[dev_index].drop(columns=[features.key])], axis=1)
    yield JOINED, "development", joined
    del joined
    for name in development:
        index = meta.index[(meta[PARTITION_COLUMN] == name).to_numpy()]
        inputs = build_canonical_frame(raw.joined.loc[index], features)
        frame = pd.concat(
            [meta.loc[index, [features.key, DAY_COLUMN, features.target]], inputs], axis=1
        )
        yield name, name, frame


def _write_processed(
    frame: pd.DataFrame, path: Path, record_path: Path, provenance: dict[str, Any]
) -> None:
    """Replace the data file and its provenance record in the safe order."""
    try:
        record_path.unlink(missing_ok=True)
    except OSError as error:
        raise FileAccessError(f"{record_path}: cannot be removed ({error.strerror})") from error
    with log_step(logger, f"Writing {path.name} ({len(frame)} rows, {len(frame.columns)} columns)"):
        write_atomically(
            path, lambda target: frame.to_parquet(target, index=False, engine=PARQUET_ENGINE)
        )
    record = {
        "file": path.name,
        "sha256": file_sha256(path),
        "rows": len(frame),
        "columns": len(frame.columns),
        "parquet_engine": PARQUET_ENGINE,
        **provenance,
    }
    write_record(record_path, record)
    logger.info("%s: provenance record written", record_path.name)


def prepare_data(
    features: FeatureSpec,
    dataset: DatasetSpec,
    paths: ProjectPaths,
    *,
    write_processed: bool = True,
    config_files: Sequence[Path] | None = None,
) -> PreparationResult:
    """Validate the raw data; optionally write the processed files and the manifest.

    ``config_files`` are hashed into the provenance records; by default the two
    declarations in ``paths.config_dir``.
    """
    dataset.validate()
    features.validate()
    if not any(p.sealed for p in dataset.partitions):
        raise SealedTestError("Data preparation requires the Test partition to be sealed")
    if config_files is None:
        config_files = (paths.features_spec, paths.dataset_spec)
    if write_processed:
        existing = [
            path.name
            for stem in stems(dataset)
            for path in (paths.processed_file(stem), paths.provenance_file(stem))
            if path.exists()
        ]
        if existing:
            raise FileAccessError(
                f"Processed files already exist in {paths.processed_dir}: {existing}. They are "
                "built once; remove them deliberately only to rebuild from the raw data"
            )

    checksums = check_file_checksums(dataset)
    raw = load_raw(features, dataset)
    summaries = validate(raw, features, dataset)
    labels = load_labels(raw, features, dataset, [p.name for p in dataset.development])
    written: list[Path] = []
    if not write_processed:
        logger.info("Check only: nothing written")
        return PreparationResult(checksums, summaries, written)

    common: dict[str, Any] = {
        "sources": checksums,
        "configuration": {
            **{path.name: file_sha256(path) for path in config_files},
            FINGERPRINT: features_fingerprint(features),
        },
        "code": code_provenance(paths.root),
    }
    for stem, scope, frame in _development_frames(raw, labels, features, dataset):
        path = paths.processed_file(stem)
        _write_processed(
            frame,
            path,
            paths.provenance_file(stem),
            {
                "scope": scope,
                "transaction_id_sha256": id_set_sha256(frame[features.key]),
                **common,
            },
        )
        written.append(path)

    rows = {
        dataset.transaction.path.name: len(raw.joined),
        dataset.identity.path.name: raw.identity_rows,
    }
    write_datapackage(
        paths.manifest, build_datapackage(dataset, checksums, summaries, rows, paths.data_dir)
    )
    logger.info("Dataset manifest written: %s", paths.manifest)
    written.append(paths.manifest)
    return PreparationResult(checksums, summaries, written)
