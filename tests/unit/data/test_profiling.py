"""Profiling development rows and checking catalog statements, on synthetic data.

The synthetic dataset (``tests/support/synthetic.py``) has 20 transactions: Train 8, Validation 1
4, Validation 2 4 and Test 4 rows. Every test profiles the processed development file after the
raw files have been removed, so profiling provably reads no raw file. Expected values below are
calculated by hand from the synthetic rows.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from review_capacity.data.catalog import read_catalog
from review_capacity.data.processed import DevelopmentFeatures, read_development_features
from review_capacity.data.profiling import compare_with_catalog, profile_columns
from support.catalog import catalog_text
from support.synthetic import Synthetic, build_processed


@pytest.fixture
def processed(tmp_path: Path) -> Synthetic:
    return build_processed(tmp_path)


def _development(data: Synthetic) -> DevelopmentFeatures:
    return read_development_features(data.paths, data.features, data.dataset)


def _report(data: Synthetic, tmp_path: Path, edit: Callable[[str], str] = lambda t: t,
            cautions: dict[str, str] | None = None) -> list[tuple[str, str]]:  # fmt: skip
    development = _development(data)
    text = edit(catalog_text(profile_columns(development, data.features), cautions))
    path = tmp_path / "catalog.md"
    path.write_text(text, encoding="utf-8")
    report = compare_with_catalog(development, data.features, data.dataset, read_catalog(path))
    return [(d.column, d.statistic) for d in report.differences]


def test_profile_values_calculated_by_hand(processed: Synthetic) -> None:
    assert not processed.dataset.transaction.path.exists()  # no raw file to read
    profile = profile_columns(_development(processed), processed.features)
    assert "isFraud" not in profile.index
    # ProductCD: [W, " C", missing, H] repeated; 16 development rows, 4 missing; 3 Train levels.
    assert profile.loc["ProductCD", "missing_dev"] == pytest.approx(0.25)
    assert profile.loc["ProductCD", "levels_train"] == 3
    # TransactionAmt = 10.5 + i for development rows i = 0..15.
    assert profile.loc["TransactionAmt", "min_dev"] == 10.5
    assert profile.loc["TransactionAmt", "max_dev"] == 25.5
    # card1 = [100, missing, -660, 1000] repeated: -660 on 4 of the 16 development rows.
    assert profile.loc["card1", "negatives_dev"] == 4
    assert profile.loc["card1", "v2_unseen"] == 0.0
    assert profile["max_dev"].loc["TransactionDT"] < 5 * 86_400  # no Test-period row
    assert bool(profile.loc["V1", "integer_valued"])


def test_unseen_share_calculated_by_hand(tmp_path: Path) -> None:
    def new_codes(frame: pd.DataFrame) -> pd.DataFrame:
        codes = list(frame["card1"])
        codes[12:16] = [7.0, None, 7.0, 1000.0]  # Validation 2 rows: 7 is unseen in Train
        return frame.assign(card1=codes)

    data = build_processed(tmp_path, edit_transaction=new_codes)
    profile = profile_columns(_development(data), data.features)
    assert profile.loc["card1", "v2_unseen"] == pytest.approx(2 / 3)  # missing value not counted


def test_matching_catalog_has_no_difference(processed: Synthetic, tmp_path: Path) -> None:
    assert _report(processed, tmp_path) == []


def test_tied_values_may_be_listed_in_any_order(processed: Synthetic, tmp_path: Path) -> None:
    # W, " C" and H each occur 4 times in the development rows.
    def reorder(text: str) -> str:
        assert "` C`, `H`, `W`" in text
        return text.replace("` C`, `H`, `W`", "`W`, `H`, ` C`")

    assert _report(processed, tmp_path, reorder) == []


@pytest.mark.parametrize(
    ("old", "new", "difference"),
    [
        ("| 25.0 % | 3 |", "| 30.0 % | 3 |", ("ProductCD", "missing (dev)")),
        ("`V1` (1 column) | 25.0 %", "`V1` (1 column) | 50.0 %", ("V block 1", "missing in Train")),
        ("1 are integer-valued", "0 are integer-valued", ("V columns", "integer-valued columns")),
        ("4 negative values", "5 negative values", ("card1", "negative values")),
        ("| Numeric input | n | 3 |", "| Numeric input | n | 4 |", ("role Numeric input", "count")),
        ("| `DeviceType` | identity |", "| `DeviceType` | transaction |", ("DeviceType", "table")),
        ("`V1` (1 column)", "`V1` (2 columns)", ("V block 1", "stated column count")),
    ],
)
def test_changed_statement_is_reported(
    processed: Synthetic, tmp_path: Path, old: str, new: str, difference: tuple[str, str]
) -> None:
    def change(text: str) -> str:
        assert old in text
        return text.replace(old, new, 1)

    assert _report(processed, tmp_path, change) == [difference]


def test_v_column_in_two_blocks_is_reported(processed: Synthetic, tmp_path: Path) -> None:
    def duplicate(text: str) -> str:
        return text.replace(
            "| 1 | `V1` (1 column) | 25.0 % |",
            "| 1 | `V1` (1 column) | 25.0 % |\n| 2 | `V1` (1 column) | 25.0 % |",
        )

    differences = _report(processed, tmp_path, duplicate)
    assert ("V blocks", "every V column in exactly one block") in differences
    assert ("V block 2", "pattern distinct from other blocks") in differences


def test_undocumented_column_is_reported(processed: Synthetic, tmp_path: Path) -> None:
    def drop(text: str) -> str:
        return "\n".join(
            line for line in text.splitlines() if not line.startswith("| `DeviceType`")
        )

    assert ("DeviceType", "column documented") in _report(processed, tmp_path, drop)


def test_caution_statistics_calculated_by_hand(processed: Synthetic, tmp_path: Path) -> None:
    cautions = {
        # TransactionAmt = 10.5 + i has one decimal: 0 % with more than two decimals.
        "TransactionAmt": "0.0 % of amounts have more than two decimal places",
        # card1 in Train (8 rows): levels 100, -660, 1000, 2 rows each; 6 of 8 rows in rare levels.
        "card1": "75.0 % of Train rows are in levels with fewer than 100 rows; "
        "100.0 % of its Train levels have fewer than 100 rows",
    }
    assert _report(processed, tmp_path, cautions=cautions) == []
    wrong = {"TransactionAmt": "5.0 % of amounts have more than two decimal places"}
    assert _report(processed, tmp_path, cautions=wrong) == [
        ("TransactionAmt", "more than two decimals")
    ]


def test_hour_agreement_calculated_by_hand(tmp_path: Path) -> None:
    # Offsets 0, 6 h, 12 h, 86,399 s: hours 0, 6, 12, 23. Amount = hour / 24, one row disagreeing:
    # 15 of the 16 development rows agree, 93.75 %.
    def hours(frame: pd.DataFrame) -> pd.DataFrame:
        values = list(((frame["TransactionDT"] % 86_400) // 3_600) / 24)
        values[1] = 0.9
        return frame.assign(TransactionAmt=values)

    data = build_processed(tmp_path, edit_transaction=hours)
    stated = "Equals (hour of the relative day) ÷ 24 on {} % of non-missing development rows"
    assert _report(data, tmp_path, cautions={"TransactionAmt": stated.format("93.8")}) == []
    assert _report(data, tmp_path, cautions={"TransactionAmt": stated.format("90.0")}) == [
        ("TransactionAmt", "hour of the relative day / 24")
    ]
