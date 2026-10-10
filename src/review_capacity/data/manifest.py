"""The dataset manifest: a Frictionless Data Package v2 descriptor, ``data/datapackage.json``.

It describes the two raw files (path relative to the data directory, size, SHA-256) and, under
the private property ``_split``, the label-free partition summary of DR-001. The standard
requires the name ``datapackage.json`` at the top level of the data directory; private
properties are prefixed with an underscore by convention. No timestamp is written, so
regenerating from the same inputs gives the same descriptor.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from review_capacity import __version__
from review_capacity.config import SECONDS_PER_DAY, DatasetSpec
from review_capacity.data.validation import PartitionSummary
from review_capacity.files import write_json

DATAPACKAGE_SCHEMA = "https://datapackage.org/profiles/2.0/datapackage.json"
SOURCE_URL = "https://www.kaggle.com/competitions/ieee-fraud-detection/data"
RULES_URL = "https://www.kaggle.com/competitions/ieee-fraud-detection/rules"


def build_datapackage(
    dataset: DatasetSpec,
    checksums: dict[str, str],
    summaries: list[PartitionSummary],
    rows: dict[str, int],
    data_dir: Path,
) -> dict[str, Any]:
    """Data Package descriptor; resource paths are relative to ``data_dir``.

    ``rows`` holds the observed data rows of each raw file, keyed by file name.
    """
    resources = []
    for name, spec in (
        ("train_transaction", dataset.transaction),
        ("train_identity", dataset.identity),
    ):
        resources.append(
            {
                "name": name,
                "path": spec.path.resolve().relative_to(data_dir.resolve()).as_posix(),
                "format": "csv",
                "mediatype": "text/csv",
                "bytes": spec.path.stat().st_size,
                "hash": f"sha256:{checksums[spec.path.name]}",
                "_rows": rows[spec.path.name],
                "_columns": spec.columns,
            }
        )
    sealed = {p.name for p in dataset.partitions if p.sealed}
    return {
        "$schema": DATAPACKAGE_SCHEMA,
        "name": "ieee-cis-fraud-detection-labelled-training-data",
        "title": "IEEE-CIS Fraud Detection: labelled training data (local copy)",
        "description": (
            "The two labelled training files of the IEEE-CIS Fraud Detection competition, as "
            "used by this study. The files stay local and are never committed or redistributed "
            "(competition rules). Partition statistics are label-free."
        ),
        "sources": [
            {"title": "IEEE-CIS Fraud Detection, Kaggle competition data", "path": SOURCE_URL}
        ],
        "licenses": [
            {
                "title": "Competition Rules: non-commercial academic use, no redistribution",
                "path": RULES_URL,
            }
        ],
        "resources": resources,
        "_split": {
            "decision_record": "docs/decisions/DR-001-temporal-split-strategy.md",
            "relative_day": f"{dataset.time} // {SECONDS_PER_DAY}",
            "transaction_id_sha256": (
                "SHA-256 of the partition's TransactionIDs, sorted ascending, "
                "as little-endian 64-bit integers"
            ),
            "joined_rows": sum(s.rows for s in summaries),
            "partitions": [{**asdict(s), "sealed": s.name in sealed} for s in summaries],
        },
        "_generator": {"package": "review_capacity", "version": __version__},
    }


def write_datapackage(path: Path, descriptor: dict[str, Any]) -> None:
    """Write the descriptor as indented UTF-8 JSON, atomically."""
    write_json(path, descriptor)
