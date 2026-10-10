"""Synthetic raw data for unit tests (no project data is read).

A miniature dataset with the same structure as the IEEE-CIS files: a transaction file with the
target, an identity file joined on ``TransactionID``, four partitions of one or two relative days
(the last one sealed), text and numeric-coded categoricals, numeric columns and missing values.
Sealed-period labels are written as the invalid value ``"X"``; whether the label read stops
before the sealed rows is tested directly (``tests/unit/data/test_loading.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from review_capacity.config import DatasetSpec, FeatureSpec, PartitionSpec, RawFileSpec
from review_capacity.hashing import file_sha256, id_set_sha256
from review_capacity.paths import ProjectPaths

REPO_PATHS = ProjectPaths.discover(start=Path(__file__).parent)  # the real config/ files
DAY = 86_400
OFFSETS = (0, 6 * 3_600, 12 * 3_600, DAY - 1)  # start, a quarter and half of the day, last second
PARTITION_DAYS = {"train": (1, 2), "v1": (3, 3), "v2": (4, 4), "test": (5, 5)}

FEATURES = FeatureSpec(
    key="TransactionID",
    time="TransactionDT",
    target="isFraud",
    excluded=("TransactionID", "TransactionDT"),
    categorical=("ProductCD", "card1", "DeviceType", "id_13"),
    numeric_coded=frozenset({"card1", "id_13"}),
    numeric=("TransactionAmt", "V1", "id_01"),
    derived=("tod_sin", "tod_cos"),
)


@dataclass(frozen=True)
class Synthetic:
    features: FeatureSpec
    dataset: DatasetSpec
    paths: ProjectPaths
    transaction: pd.DataFrame
    identity: pd.DataFrame


def _transactions(test_label: str) -> pd.DataFrame:
    rows = []
    for name, (first, last) in PARTITION_DAYS.items():
        for day in range(first, last + 1):
            for offset in OFFSETS:
                rows.append({"TransactionDT": day * DAY + offset, "partition": name})
    frame = pd.DataFrame(rows)
    n = len(frame)
    frame.insert(0, "TransactionID", range(3_000_000, 3_000_000 + n))
    labels = [str(i % 2) for i in range(n)]
    frame.insert(
        1,
        "isFraud",
        [
            test_label if p == "test" else lab
            for p, lab in zip(frame["partition"], labels, strict=True)
        ],
    )
    frame["TransactionAmt"] = [10.5 + i for i in range(n)]
    frame["ProductCD"] = ["W", " C", None, "H"] * (n // 4)  # leading space kept exactly
    frame["card1"] = [100.0, None, -660.0, 1000.0] * (n // 4)
    frame["V1"] = [1.0, None, 0.0, 2.0] * (n // 4)
    return frame.drop(columns="partition")


def _identity(transaction: pd.DataFrame) -> pd.DataFrame:
    ids = transaction["TransactionID"].iloc[::2].to_numpy()
    return pd.DataFrame(
        {
            "TransactionID": ids,
            "id_01": [-5.0 + i for i in range(len(ids))],
            "id_13": [49.0, None] * (len(ids) // 2) + [49.0] * (len(ids) % 2),
            "DeviceType": ["desktop", "mobile"] * (len(ids) // 2) + ["desktop"] * (len(ids) % 2),
        }
    )


def build_synthetic(
    root: Path,
    *,
    test_label: str = "X",
    edit_transaction: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    edit_identity: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> Synthetic:
    """Write the synthetic raw files in the raw-data directory under ``root``; return specs.

    The partition anchors are computed from the unedited data, so an edit that changes rows,
    bounds or identifiers is detected by the validation contract.
    """
    transaction = _transactions(test_label)
    identity = _identity(transaction)
    days = transaction["TransactionDT"] // DAY
    partitions = []
    for name, (first, last) in PARTITION_DAYS.items():
        rows = transaction[(days >= first) & (days <= last)]
        partitions.append(
            PartitionSpec(
                name=name,
                label=name,
                first_day=first,
                last_day=last,
                rows=len(rows),
                transaction_dt_min=int(rows["TransactionDT"].min()),
                transaction_dt_max=int(rows["TransactionDT"].max()),
                transaction_id_sha256_prefix=id_set_sha256(rows["TransactionID"])[:16],
                sealed=name == "test",
            )
        )
    if edit_transaction is not None:
        transaction = edit_transaction(transaction)
    if edit_identity is not None:
        identity = edit_identity(identity)
    paths = ProjectPaths(root)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    tx_path = paths.raw_dir / "train_transaction.csv"
    id_path = paths.raw_dir / "train_identity.csv"
    transaction.to_csv(tx_path, index=False)
    identity.to_csv(id_path, index=False)
    dataset = DatasetSpec(
        transaction=RawFileSpec(tx_path, file_sha256(tx_path), len(transaction.columns)),
        identity=RawFileSpec(id_path, file_sha256(id_path), len(identity.columns)),
        key="TransactionID",
        time="TransactionDT",
        seconds_per_day=DAY,
        total_rows=sum(p.rows for p in partitions),
        partitions=tuple(partitions),
    )
    dataset.validate()
    FEATURES.validate()
    return Synthetic(FEATURES, dataset, paths, transaction, identity)


def with_partition(dataset: DatasetSpec, name: str, **changes: object) -> DatasetSpec:
    """Copy of ``dataset`` with one partition's anchors changed."""
    parts = tuple(
        replace(p, **changes) if p.name == name else p  # type: ignore[arg-type]
        for p in dataset.partitions
    )
    return replace(dataset, partitions=parts)
