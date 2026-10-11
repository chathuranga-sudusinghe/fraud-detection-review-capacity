"""Profiling the development data and checking every statistic stated in the feature catalog.

Recomputes, from development rows only (Train, Validation 1 and Validation 2; relative days 1-154),
the statements that ``data.catalog`` reads from ``docs/feature-catalog.md``. The rows come from the
verified processed file ``development_joined.parquet`` (``data.processed``); no raw file is read.
The target is never read: only the declared source columns besides ``isFraud`` are read from
storage. Table membership comes from ``config/dataset.yaml``. Differences are reported, never
written back to the catalog.

Checked, per column row: table membership, interpretation and model-input role; for joined columns
other than the target, the kind (text or numeric), the observed range or the most frequent values
(any order among equally frequent values), the missing share, the levels in Train, integer-valued
(``V`` appendix) and the statistics stated in the Caution column (Validation 2 values unseen in
Train, negative values, amounts with more than two decimals, rare-level shares in Train, agreement
with the hour of the relative day). Also checked: the role counts, the ``V`` summary (table, kind,
interpretation, model input, missing and level ranges, two-valued and integer-valued counts, number
of blocks) and the ``V`` missingness blocks (every ``V`` column in exactly one block, stated sizes,
identical missing rows within a block, distinct patterns between blocks, missing share in Train).

Not checked by design (``EXCLUDED``): descriptive prose, the int64 / float64 storage label, the
target and derived-input statistics, partition totals in prose and sample-based correlations.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from review_capacity.config import SECONDS_PER_DAY, DatasetSpec, FeatureSpec
from review_capacity.data.catalog import Catalog, CatalogEntry
from review_capacity.data.processed import DevelopmentFeatures
from review_capacity.logging_setup import ProgressLogger, log_step

TOP_VALUES = 5
# D9 is stored to six decimal places (1/24 as 0.041666001): agreement with hour / 24 within one
# unit of the sixth decimal; the largest observed difference on development rows is 6.95e-7.
HOUR_TOLERANCE = 1e-6

EXCLUDED = (
    "Official meaning, Reason for treatment and descriptive Caution text: prose, not statistics",
    "Exact int64 / float64 storage label: pandas type inference when the catalog was written; "
    "checked as numeric versus text",
    "Target (isFraud) statistics: the target is never read; derived-input (tod_sin, tod_cos) "
    "statistics: theoretical bounds guaranteed by the TransactionDT validation rule",
    "Partition totals and boundaries in prose: anchored in config/dataset.yaml and checked by "
    "prepare-data and the integration tests",
    "Spearman rank correlations: computed on a 100,000-row random sample",
)

logger = logging.getLogger(__name__)
_V_COLUMN = re.compile(r"V\d+")
ROLE_NAMES = (
    "Target",
    "Excluded (identifier / raw time)",
    "Categorical input",
    "Numeric input",
    "Total",
)


@dataclass(frozen=True)
class Difference:
    column: str
    statistic: str
    catalog: str
    observed: str


@dataclass
class ProfileReport:
    profile: pd.DataFrame
    checks: int = 0
    differences: list[Difference] = field(default_factory=list)
    excluded: tuple[str, ...] = EXCLUDED

    def check(self, ok: bool, column: str, statistic: str, catalog: str, observed: str) -> None:
        self.checks += 1
        if not ok:
            self.differences.append(Difference(column, statistic, catalog, observed))


@dataclass(frozen=True)
class _Frames:
    dev: pd.DataFrame
    train: pd.DataFrame
    v2: pd.DataFrame


def _frames(data: DevelopmentFeatures) -> _Frames:
    return _Frames(
        dev=data.joined,
        train=data.joined[(data.partition == "train").to_numpy()],
        v2=data.joined[(data.partition == "v2").to_numpy()],
    )


def _counts(values: pd.Series) -> pd.Series:
    """Frequencies of the non-missing values, most frequent first (ties: by value)."""
    counts = values.dropna().astype(str).value_counts()
    order = sorted(counts.index, key=lambda v: (-int(counts[v]), v))
    return counts.loc[order]


def profile_columns(data: DevelopmentFeatures, features: FeatureSpec) -> pd.DataFrame:
    """One row per declared joined column except the target, from development rows only."""
    frames = _frames(data)
    text = set(features.text_categorical)
    categorical = set(features.categorical)
    records = []
    for column in (c for c in features.joined_columns if c != features.target):
        values = frames.dev[column]
        record: dict[str, object] = {
            "column": column,
            "kind": "text" if column in text else "numeric",
            "missing_dev": float(values.isna().mean()),
            "missing_dev_rows": int(values.isna().sum()),
            "levels_train": int(frames.train[column].nunique(dropna=True)),
        }
        if column in text:
            record["top_values"] = " | ".join(_counts(values).index[:TOP_VALUES])
        else:
            present = values.dropna().to_numpy(dtype="float64")
            record["min_dev"] = float(present.min()) if present.size else np.nan
            record["max_dev"] = float(present.max()) if present.size else np.nan
            record["negatives_dev"] = int((present < 0).sum())
            record["integer_valued"] = bool((np.floor(present) == present).all())
        if column in categorical:
            seen = set(frames.train[column].dropna().astype(str))
            later = frames.v2[column].dropna().astype(str)
            record["v2_unseen"] = float((~later.isin(seen)).mean()) if len(later) else 0.0
        records.append(record)
    return pd.DataFrame.from_records(records).set_index("column")


def _share_text(share: float) -> str:
    return f"{100 * share:.3f} %"


def _check_roles(
    entry: CatalogEntry, features: FeatureSpec, dataset: DatasetSpec, report: ProfileReport
) -> None:
    column = entry.column
    if column == features.key:
        table, interpretation = "both", "identifier"
    elif column in features.derived:
        table, interpretation = "derived", "numeric"
    else:
        table = dataset.source_table(column)
        if column == features.time:
            interpretation = "temporal"
        elif column == features.target:
            interpretation = "target"
        elif column in features.categorical:
            interpretation = "categorical"
        else:
            interpretation = "numeric"
    model_input = "include" if column in features.model_inputs else "exclude"
    for statistic, stated, expected in (
        ("table", entry.table, table),
        ("interpretation", entry.interpretation, interpretation),
        ("model input", entry.model_input, model_input),
    ):
        if stated is not None:
            report.check(stated == expected, column, statistic, stated, expected)


def _check_top_values(
    column: str, listed: tuple[str, ...], counts: pd.Series, report: ProfileReport
) -> None:
    """The listed values must be the most frequent, in an order consistent with frequency."""
    listed_counts = [int(counts.get(v, 0)) for v in listed]
    top_counts = [int(c) for c in counts.iloc[: len(listed)]]
    ok = all(v in counts.index for v in listed) and listed_counts == top_counts
    report.check(
        ok,
        column,
        "most frequent values",
        ", ".join(listed),
        ", ".join(f"{v} ({int(c)})" for v, c in counts.iloc[: len(listed) + 1].items()),
    )


def _check_caution(
    entry: CatalogEntry,
    row: pd.Series,
    frames: _Frames,
    features: FeatureSpec,
    report: ProfileReport,
) -> None:
    column, caution = entry.column, entry.caution
    if caution.unseen is not None:
        share = float(row["v2_unseen"])
        stated = caution.unseen
        report.check(
            stated.matches(share, 0),
            column,
            "Validation 2 unseen in Train",
            f"{'< ' if stated.less_than else ''}{stated.value.value:g} %",
            _share_text(share),
        )
    if caution.negatives is not None:
        observed = int(row["negatives_dev"])
        report.check(
            caution.negatives == observed,
            column,
            "negative values",
            str(caution.negatives),
            str(observed),
        )
    if caution.decimals is not None:
        cents = frames.dev[column].dropna().to_numpy(dtype="float64") * 100
        share = float((np.abs(cents - np.round(cents)) > 1e-6).mean())
        report.check(
            caution.decimals.matches(100 * share),
            column,
            "more than two decimals",
            f"{caution.decimals.value:g} %",
            _share_text(share),
        )
    if caution.rare_rows is not None:
        stated_share, below = caution.rare_rows
        counts = frames.train[column].value_counts()
        share = float(counts[counts < below].sum() / len(frames.train))
        report.check(
            stated_share.matches(100 * share),
            column,
            "Train rows in rare levels",
            f"{stated_share.value:g} % (fewer than {below} rows)",
            _share_text(share),
        )
    if caution.rare_levels is not None:
        stated_share, below = caution.rare_levels
        counts = frames.train[column].value_counts()
        share = float((counts < below).mean())
        report.check(
            stated_share.matches(100 * share),
            column,
            "rare Train levels",
            f"{stated_share.value:g} % (fewer than {below} rows)",
            _share_text(share),
        )
    if caution.hour is not None:
        present = frames.dev[[column, features.time]].dropna()
        hours = (present[features.time].to_numpy(dtype="int64") % SECONDS_PER_DAY) // 3_600
        agree = np.abs(present[column].to_numpy(dtype="float64") - hours / 24) <= HOUR_TOLERANCE
        share = float(agree.mean())
        report.check(
            caution.hour.matches(100 * share),
            column,
            "hour of the relative day / 24",
            f"{caution.hour.value:g} %",
            _share_text(share),
        )


def _check_statistics(
    entry: CatalogEntry,
    row: pd.Series,
    frames: _Frames,
    features: FeatureSpec,
    report: ProfileReport,
) -> None:
    column = entry.column
    if entry.dtype is not None:
        kind = "text" if entry.dtype == "text" else "numeric"
        report.check(kind == row["kind"], column, "kind", entry.dtype, str(row["kind"]))
    if entry.top_values is not None:
        _check_top_values(column, entry.top_values, _counts(frames.dev[column]), report)
    if entry.numeric_range is not None:
        if row["kind"] != "numeric":
            report.check(False, column, "range", "numeric range", "text column")
        else:
            low, high = entry.numeric_range
            ok = low.matches(float(row["min_dev"])) and high.matches(float(row["max_dev"]))
            report.check(
                ok,
                column,
                "range",
                f"{low.value:g} to {high.value:g}",
                f"{row['min_dev']:g} to {row['max_dev']:g}",
            )
    if entry.missing is not None:
        ok = entry.missing.matches(float(row["missing_dev"]), int(row["missing_dev_rows"]))
        report.check(
            ok,
            column,
            "missing (dev)",
            f"{entry.missing.value.value:g} %",
            _share_text(float(row["missing_dev"])),
        )
    if entry.levels is not None:
        observed = int(row["levels_train"])
        report.check(
            entry.levels == observed, column, "levels in Train", str(entry.levels), str(observed)
        )
    if entry.integer_valued is not None:
        observed_integer = bool(row["integer_valued"])
        report.check(
            entry.integer_valued == observed_integer,
            column,
            "integer-valued",
            str(entry.integer_valued),
            str(observed_integer),
        )
    _check_caution(entry, row, frames, features, report)


def _check_role_counts(catalog: Catalog, features: FeatureSpec, report: ProfileReport) -> None:
    expected = dict(
        zip(
            ROLE_NAMES,
            (
                1,
                len(features.excluded),
                len(features.categorical),
                len(features.numeric),
                len(features.joined_columns),
            ),
            strict=True,
        )
    )
    for role in sorted(set(expected) | set(catalog.role_counts)):
        stated = catalog.role_counts.get(role)
        report.check(
            stated == expected.get(role),
            f"role {role}",
            "count",
            str(stated),
            str(expected.get(role)),
        )


def _common(values: set[str]) -> str:
    """The one value shared by every ``V`` column, or ``mixed``."""
    return values.pop() if len(values) == 1 else "mixed"


def _check_v_summary(
    catalog: Catalog,
    profile: pd.DataFrame,
    v_columns: list[str],
    features: FeatureSpec,
    dataset: DatasetSpec,
    report: ProfileReport,
) -> None:
    summary = catalog.v_summary
    if summary is None:
        report.check(not v_columns, "V columns", "property table", "absent", "V columns exist")
        return
    v = profile.loc[v_columns]
    kind = _common(set(v["kind"].astype(str)))
    for statistic, stated, expected in (
        ("table", summary.table, _common({dataset.source_table(c) for c in v_columns})),
        ("kind", "text" if summary.dtype == "text" else "numeric", kind),
        ("interpretation", summary.interpretation,
         "categorical" if set(v_columns) & set(features.categorical) else "numeric"),
        ("model input", summary.model_input,
         _common({"include" if c in features.model_inputs else "exclude" for c in v_columns})),
        ("missingness blocks", str(summary.blocks), str(len(catalog.blocks))),
    ):  # fmt: skip
        report.check(stated == expected, "V columns", statistic, stated, expected)
    low, high = summary.missing_range
    report.check(
        low.value.matches(100 * float(v["missing_dev"].min())),
        "V columns",
        "lowest missing (dev)",
        f"{low.value.value:g} %",
        _share_text(float(v["missing_dev"].min())),
    )
    report.check(
        high.value.matches(100 * float(v["missing_dev"].max())),
        "V columns",
        "highest missing (dev)",
        f"{high.value.value:g} %",
        _share_text(float(v["missing_dev"].max())),
    )
    levels = (int(v["levels_train"].min()), int(v["levels_train"].max()))
    report.check(
        summary.levels_range == levels,
        "V columns",
        "levels range",
        str(summary.levels_range),
        str(levels),
    )
    two_valued = int((v["levels_train"] == 2).sum())
    report.check(
        summary.two_valued == two_valued,
        "V columns",
        "two-valued columns",
        str(summary.two_valued),
        str(two_valued),
    )
    integer_valued = int(v["integer_valued"].astype(bool).sum())
    report.check(
        summary.integer_valued == integer_valued,
        "V columns",
        "integer-valued columns",
        str(summary.integer_valued),
        str(integer_valued),
    )


def _check_blocks(
    catalog: Catalog,
    frames: _Frames,
    v_columns: list[str],
    report: ProfileReport,
    progress: Callable[[], None],
) -> None:
    numbers = [b.number for b in catalog.blocks]
    report.check(
        len(numbers) == len(set(numbers)),
        "V blocks",
        "unique block numbers",
        "unique",
        str(numbers),
    )
    listed = [c for b in catalog.blocks for c in b.columns]
    repeated = sorted({c for c in listed if listed.count(c) > 1})
    omitted = sorted(set(v_columns) - set(listed))
    unknown = sorted(set(listed) - set(v_columns))
    report.check(
        not (repeated or omitted or unknown),
        "V blocks",
        "every V column in exactly one block",
        "exactly once",
        f"repeated {repeated}, omitted {omitted}, unknown {unknown}",
    )
    patterns: dict[bytes, int] = {}
    for block in catalog.blocks:
        name = f"V block {block.number}"
        report.check(
            len(block.columns) == block.stated_columns,
            name,
            "stated column count",
            str(block.stated_columns),
            str(len(block.columns)),
        )
        present = [c for c in block.columns if c in frames.train.columns]
        if len(present) != len(block.columns):
            progress()
            continue
        masks = frames.train[present].isna().to_numpy()
        identical = bool((masks == masks[:, :1]).all())
        report.check(
            identical,
            name,
            "identical missing rows in Train",
            "identical",
            "identical" if identical else "differ",
        )
        first = masks[:, 0]
        key = np.packbits(first).tobytes()
        report.check(
            key not in patterns,
            name,
            "pattern distinct from other blocks",
            "distinct",
            f"same as block {patterns.get(key)}",
        )
        patterns.setdefault(key, block.number)
        share, rows = float(first.mean()), int(first.sum())
        report.check(
            block.missing_in_train.matches(share, rows),
            name,
            "missing in Train",
            f"{block.missing_in_train.value.value:g} %",
            _share_text(share),
        )
        progress()


def compare_with_catalog(
    data: DevelopmentFeatures, features: FeatureSpec, dataset: DatasetSpec, catalog: Catalog
) -> ProfileReport:
    """Profile the development rows and check every interpreted catalog statement."""
    with log_step(logger, "Profiling development rows (no labels)"):
        profile = profile_columns(data, features)
    frames = _frames(data)
    report = ProfileReport(profile)
    documented = set(features.joined_columns) | set(features.derived)
    v_columns = [c for c in features.numeric if _V_COLUMN.fullmatch(c)]
    with log_step(logger, "Comparing with the feature catalog"):
        progress = ProgressLogger(
            logger,
            "Catalog column rows and V-column missingness blocks",
            len(catalog.entries) + len(catalog.blocks),
        )
        for column, entry in catalog.entries.items():
            if column not in documented:
                report.check(False, column, "column known", "in catalog", "not declared")
            else:
                _check_roles(entry, features, dataset, report)
                if column in profile.index:
                    row = profile.loc[column]
                    if not isinstance(row, pd.Series):
                        raise TypeError(f"Profile has more than one row for column {column}")
                    _check_statistics(entry, row, frames, features, report)
            progress.advance()
        for column in sorted(documented - set(catalog.entries)):
            report.check(False, column, "column documented", "missing", "declared")
        _check_role_counts(catalog, features, report)
        _check_v_summary(catalog, profile, v_columns, features, dataset, report)
        _check_blocks(catalog, frames, v_columns, report, progress.advance)
    return report
