"""Reading and joining the raw files, with the Test period sealed in code.

The only module that opens the raw files (checked by ``tests/unit/test_architecture.py``); only
the Parquet build (``data.preparation``) and the raw-data contract (``data.validation``) use it.

Feature columns are read for all rows; the target is never read with them. Labels are read
separately and only for the leading rows of the requested development partitions: the file is
ordered by ``TransactionDT`` (checked), so the CSV parser stops before the first Test row and
Test-period labels are never parsed unless ``unseal_test_labels=True`` is passed explicitly.

Missing values are recognised by column role. In text categoricals only an empty field is
missing, so text such as ``"NA"`` or ``"None"`` is kept exactly (DR-008 §5). Numeric columns and
the key use the missing-value tokens that pandas recognises by default.
"""

from __future__ import annotations

import csv
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from review_capacity.config import DatasetSpec, FeatureSpec
from review_capacity.data.partitions import (
    assign_partitions,
    relative_day,
    validate_transaction_dt,
)
from review_capacity.errors import DataValidationError, FileAccessError, SealedTestError
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import log_step

logger = logging.getLogger(__name__)

# The default missing-value tokens of pandas.read_csv, as documented for pandas 3.0
# (https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html, parameter na_values).
NUMERIC_MISSING_TOKENS = (
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
)  # fmt: skip
TEXT_MISSING_TOKENS = ("",)


def check_file_checksums(dataset: DatasetSpec) -> dict[str, str]:
    """SHA-256 of both raw files; they must equal the recorded values."""
    observed: dict[str, str] = {}
    for spec in (dataset.transaction, dataset.identity):
        with log_step(logger, f"Computing SHA-256 of {spec.path.name}"):
            digest = file_sha256(spec.path)
        if digest != spec.sha256:
            raise DataValidationError(
                f"{spec.path.name}: SHA-256 {digest} differs from the recorded {spec.sha256}"
            )
        observed[spec.path.name] = digest
        logger.info("%s: SHA-256 matches the recorded value", spec.path.name)
    return observed


def read_header(path: Path) -> list[str]:
    """Column names in the first line of a CSV file."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return next(csv.reader(handle))
    except OSError as error:
        raise FileAccessError(f"{path}: cannot be read ({error.strerror})") from error
    except StopIteration as error:
        raise DataValidationError(f"{path.name}: the file is empty") from error


def check_headers(features: FeatureSpec, dataset: DatasetSpec) -> None:
    """The two headers must join to exactly the columns declared in ``config/features.yaml``."""
    tx_cols = read_header(dataset.transaction.path)
    id_cols = read_header(dataset.identity.path)
    problems: list[str] = []
    for name, cols, expected in (
        ("transaction", tx_cols, dataset.transaction.columns),
        ("identity", id_cols, dataset.identity.columns),
    ):
        repeated = sorted(c for c, n in Counter(cols).items() if n > 1)
        if repeated:
            problems.append(f"{name} file repeats {repeated}")
        if len(cols) != expected:
            problems.append(f"{name} file has {len(cols)} columns, {expected} expected")
        if features.key not in cols:
            problems.append(f"{name} file lacks the key {features.key}")
    shared = (set(tx_cols) & set(id_cols)) - {features.key}
    if shared:
        problems.append(f"columns in both files besides the key: {sorted(shared)}")
    if set(id_cols) - {features.key} != set(dataset.identity_columns):
        problems.append("identity columns differ from columns_besides_key in config/dataset.yaml")
    if features.target not in tx_cols:
        problems.append(f"target {features.target} not in the transaction file")
    joined = set(tx_cols) | set(id_cols)
    declared = set(features.joined_columns)
    if joined != declared:
        problems.append(
            f"undeclared columns {sorted(joined - declared)}; "
            f"declared but absent {sorted(declared - joined)}"
        )
    if problems:
        raise DataValidationError(
            "Raw headers do not match the declaration: " + "; ".join(problems)
        )
    logger.info("Raw headers match the declaration: %d joined columns", len(joined))


def _dtypes(features: FeatureSpec, columns: Sequence[str]) -> dict[str, str]:
    """Read text categoricals as text, the key as nullable integer and all else as float64."""
    text = set(features.text_categorical)
    dtypes: dict[str, str] = {}
    for col in columns:
        if col == features.key:
            dtypes[col] = "Int64"
        elif col in text:
            dtypes[col] = "str"
        else:
            dtypes[col] = "float64"
    return dtypes


def _missing_tokens(features: FeatureSpec, columns: Sequence[str]) -> dict[str, list[str]]:
    """Missing-value tokens per column: only the empty field for text categoricals."""
    text = set(features.text_categorical)
    return {
        col: list(TEXT_MISSING_TOKENS if col in text else NUMERIC_MISSING_TOKENS) for col in columns
    }


def _read_csv(path: Path, **options: object) -> pd.DataFrame:
    """``pandas.read_csv`` with parsing failures reported as ``DataValidationError``."""
    try:
        frame: pd.DataFrame = pd.read_csv(path, keep_default_na=False, **options)  # type: ignore[call-overload]
    except OSError as error:
        raise FileAccessError(f"{path}: cannot be read ({error.strerror})") from error
    except (ValueError, pd.errors.ParserError) as error:
        raise DataValidationError(f"{path.name}: cannot be parsed as declared") from error
    return frame


def _check_key(frame: pd.DataFrame, key: str, name: str) -> None:
    if frame[key].isna().any():
        raise DataValidationError(f"{name} file has missing {key} values")
    if frame[key].duplicated().any():
        raise DataValidationError(f"{name} file has repeated {key} values")


def read_feature_table(path: Path, features: FeatureSpec) -> pd.DataFrame:
    """Read every column except the target, with declared types."""
    columns = [c for c in read_header(path) if c != features.target]
    with log_step(logger, f"Reading {path.name} (features only, no target)"):
        frame = _read_csv(
            path,
            usecols=columns,
            dtype=_dtypes(features, columns),
            na_values=_missing_tokens(features, columns),
        )
    _check_key(frame, features.key, path.name)
    frame[features.key] = frame[features.key].astype("int64")
    logger.info("%s: %d rows, %d columns", path.name, len(frame), len(frame.columns))
    return frame


def join_tables(transaction: pd.DataFrame, identity: pd.DataFrame, key: str) -> pd.DataFrame:
    """Left join transaction <- identity on the key, validated one-to-one and row-preserving."""
    unmatched = ~identity[key].isin(transaction[key])
    if unmatched.any():
        raise DataValidationError(f"{int(unmatched.sum())} identity rows match no transaction")
    joined = transaction.merge(identity, on=key, how="left", validate="one_to_one", sort=False)
    if len(joined) != len(transaction) or not joined[key].equals(transaction[key]):
        raise DataValidationError("The join changed the transaction rows or their order")
    logger.info(
        "Joined %d transaction rows with %d identity rows (one-to-one, row-preserving)",
        len(transaction),
        len(identity),
    )
    return joined


@dataclass(frozen=True)
class RawData:
    """The joined raw data of all rows, without the target, in file order."""

    joined: pd.DataFrame
    relative_day: pd.Series
    partition: pd.Series
    identity_rows: int
    has_identity: pd.Series  # True where the transaction has a row in the identity file


def load_raw(features: FeatureSpec, dataset: DatasetSpec) -> RawData:
    """Read, join and time-validate the raw files; assign relative days and partitions."""
    check_headers(features, dataset)
    transaction = read_feature_table(dataset.transaction.path, features)
    identity = read_feature_table(dataset.identity.path, features)
    joined = join_tables(transaction, identity, features.key)
    time = joined[features.time]
    validate_transaction_dt(time, first_day=dataset.first_day, last_day=dataset.last_day)
    if not time.is_monotonic_increasing:
        raise DataValidationError(f"{features.time} is not in non-decreasing file order")
    joined[features.time] = time.astype("int64")
    logger.info("%s valid and in file order on all %d rows", features.time, len(joined))
    days = relative_day(joined[features.time])
    partition = assign_partitions(days, dataset)
    logger.info(
        "Rows per partition: %s",
        ", ".join(f"{name} {count}" for name, count in partition.value_counts(sort=False).items()),
    )
    return RawData(
        joined=joined,
        relative_day=days,
        partition=partition,
        identity_rows=len(identity),
        has_identity=joined[features.key].isin(identity[features.key]),
    )


def load_labels(
    raw: RawData,
    features: FeatureSpec,
    dataset: DatasetSpec,
    partitions: Sequence[str],
    *,
    unseal_test_labels: bool = False,
) -> pd.Series:
    """Labels of the requested partitions, aligned to ``raw.joined``.

    Raises ``SealedTestError`` if a sealed partition is requested without explicit unsealing.
    """
    requested = [dataset.partition(name) for name in partitions]
    sealed = [p.name for p in requested if p.sealed]
    if sealed and not unseal_test_labels:
        raise SealedTestError(f"Labels of sealed partition(s) {sealed} cannot be read")
    mask = raw.partition.isin([p.name for p in requested]).to_numpy()
    if not mask.any():
        raise DataValidationError(f"No rows in partitions {list(partitions)}")
    n_rows = int(np.flatnonzero(mask).max()) + 1
    parsed_partitions = set(raw.partition.iloc[:n_rows].astype(str))
    sealed_names = {p.name for p in dataset.partitions if p.sealed}
    if parsed_partitions & sealed_names and not unseal_test_labels:
        raise SealedTestError("Reading these labels would parse sealed rows")
    scope = "sealed rows explicitly unsealed" if unseal_test_labels else "no sealed row"
    with log_step(
        logger,
        f"Reading labels of {[p.name for p in requested]} (first {n_rows} rows; {scope})",
    ):
        labels = _read_csv(
            dataset.transaction.path,
            usecols=[features.key, features.target],
            dtype={features.key: "int64", features.target: "str"},
            na_values={features.key: [""], features.target: [""]},
            nrows=n_rows,
        )
    if not labels[features.key].equals(
        raw.joined[features.key].iloc[:n_rows].reset_index(drop=True)
    ):
        raise DataValidationError("Label rows do not align with the feature rows")
    selected = labels[features.target].iloc[np.flatnonzero(mask)]
    if not selected.isin(["0", "1"]).all():
        raise DataValidationError(f"{features.target} is not binary and complete")
    return pd.Series(
        selected.astype("int64").to_numpy(), index=raw.joined.index[mask], name=features.target
    )
