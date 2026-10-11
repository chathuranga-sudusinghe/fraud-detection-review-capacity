"""Strict reading of the statistics stated in the feature catalog."""

from __future__ import annotations

from pathlib import Path

import pytest

from review_capacity.config import load_feature_spec
from review_capacity.data.catalog import Number, Percent, read_catalog
from review_capacity.errors import FileAccessError, SpecificationError
from support.synthetic import REPO_PATHS

FIELD = (
    "| Column | Table | Official meaning | Observed dtype "
    "| Observed range / most frequent values | Interpretation | Model input "
    "| Reason for treatment | Missing (dev) | Levels in Train | Caution |"
)
CARD1 = (
    "| `card1` | transaction | Masked | int64 | 1,000 to 18,396 | categorical (numeric-coded) "
    "| include | Declared | 0.0 % | 11,870 | Validation 2 values unseen in Train: 1.3 % |"
)
CARD4 = (
    "| `card4` | transaction | Masked | text | `visa`, `mastercard` | categorical | include "
    "| Declared | 0.2 % | 4 | — |"
)
MISSING = "| Missing (dev) | 0.0 %"  # start of the V property row
CATALOG_LINES = [
    "# Catalog",
    "",
    FIELD,
    "|---|---|---|---|---|---|---|---|---|---|---|",
    CARD1,
    CARD4,
    "",
    "| Role | Columns | Count |",
    "|---|---|---|",
    "| Target | t | 1 |",
    "",
    "| Property | Value |",
    "|---|---|",
    "| Table | transaction |",
    "| Official meaning | Masked |",
    "| Observed dtype | float64 (all 339) |",
    "| Interpretation | numeric |",
    "| Model input | include (all 339) |",
    "| Missing (dev) | 0.0 % to 85.7 % per column |",
    "| Levels in Train | 1 to 25,530 per column; 7 columns are two-valued; "
    "263 are integer-valued |",
    "| Structure | 1 blocks of columns share an identical missing-row pattern in Train |",
    "",
    "| Block | Columns | Missing in Train |",
    "|---|---|---|",
    "| 1 | `V1`–`V3`, `V5` (4 columns) | < 0.01 % (12 rows) |",
    "",
    "| Column | Missing (dev) | Levels in Train | Observed range | Integer-valued |",
    "|---|---|---|---|---|",
    "| `V1` | 50.1 % | 2 | 0 to 1 | yes |",
    "",
]


def _write(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "catalog.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_numbers_match_within_half_a_unit_of_the_shown_precision() -> None:
    assert Number.parse("0.958").matches(23 / 24)
    assert Number.parse("31,937.391").matches(31937.391)
    assert Number.parse("−660").matches(-660)
    assert Number.parse("0.2").matches(0.25)  # boundary is inclusive
    assert not Number.parse("0.2").matches(0.26)


def test_percent_bounds_and_row_counts() -> None:
    small = Percent(Number.parse("0.01"), less_than=True, rows=12)
    assert small.matches(12 / 368_632, 12)
    assert not small.matches(13 / 368_632, 13)
    assert Percent(Number.parse("0.1"), less_than=True, rows=None).matches(0.0, 0)
    assert Percent(Number.parse("57.5"), less_than=False, rows=None).matches(0.57462, 0)


def test_reads_typed_statements(tmp_path: Path) -> None:
    catalog = read_catalog(_write(tmp_path, CATALOG_LINES))
    card1, card4, v1 = (catalog.entries[c] for c in ("card1", "card4", "V1"))
    assert card1.line == 5
    assert card1.dtype == "int64"
    assert card1.interpretation == "categorical"
    assert card1.numeric_range is not None
    assert card1.numeric_range[1].value == 18_396
    assert card1.levels == 11_870
    assert card4.top_values == ("visa", "mastercard")
    assert v1.integer_valued is True
    assert catalog.blocks[0].columns == ("V1", "V2", "V3", "V5")
    assert catalog.blocks[0].stated_columns == 4
    assert catalog.blocks[0].missing_in_train.rows == 12
    assert catalog.v_summary is not None
    assert catalog.v_summary.levels_range == (1, 25_530)
    assert (catalog.v_summary.two_valued, catalog.v_summary.integer_valued) == (7, 263)
    assert (catalog.v_summary.table, catalog.v_summary.model_input) == ("transaction", "include")
    assert catalog.v_summary.blocks == 1


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("1,000 to 18,396", "about one thousand", "is not 'a to b'"),
        ("`visa`, `mastercard`", "visa, mastercard", "no value in backticks"),
        ("| int64 |", "| int32 |", "Observed dtype"),
        ("| 0.2 % |", "| some |", "percentage"),
        ("| 11,870 |", "| many |", "integer"),
        ("| include |", "| maybe |", "Model input"),
        ("(4 columns)", "", "cannot be interpreted"),
        ("`V1`–`V3`", "`V3`–`V1`", "empty range"),
        ("| yes |", "| perhaps |", "Integer-valued"),
        ("| Declared | 0.2 % | 4 | — |", "| Declared | 0.2 % | 4 |", "row has 10 cells"),
        ("| Block | Columns | Missing in Train |", "| Block | Cols | Missing |", "unrecognised"),
        ("`visa`, `mastercard`", "`visa`, `visa`", "a listed value is repeated"),
        ("1,000 to 18,396", "1,,000 to 18,396", "is not 'a to b'"),
        ("| 11,870 |", "| 11,87 |", "integer"),
        ("1,000 to 18,396", "0 to 1e9", "is not 'a to b'"),
        ("| — |", "| 1,,004 negative values |", "'negative value' cannot be"),
        ("| — |", "| 1e4 negative values |", "'negative value' cannot be"),
        ("7 columns are two-valued", "1.7 columns are two-valued", "V levels statement"),
        ("< 0.01 % (12 rows)", "< 0.01 % (unknown rows)", "row count"),
        ("| Levels in Train | 1 to", "| Levels in Train | 1.5 to", "V levels statement"),
        ("unseen in Train: 1.3 %", "unseen in Train: unknown", "'unseen in Train' cannot be"),
        ("unseen in Train: 1.3 % |", "unseen in Train: 1.3 %; unseen in Train: 2 % |",
         "stated more than once"),
        ("| — |", "| some negative values |", "'negative value' cannot be"),
        ("| Target | t | 1 |", "| Target | 1 |", "row has 2 cells"),
        ("| Target | t | 1 |", "| Target | t | 1 |\n| Target | t | 2 |", "'Target' appears twice"),
        (MISSING, f"| Missing (dev) | 1.0 % to 2.0 % per column |\n{MISSING}",
         "'Missing \\(dev\\)' appears twice"),
        (MISSING, f"| Spread | 3 |\n{MISSING}", "unknown V property"),
        (f"{MISSING} to 85.7 % per column |\n", "", r"lacks \['Missing \(dev\)'\]"),
        ("| Structure | 1 blocks", "| Structure | One block", "structure statement"),
        ("| Model input | include (all 339) |", "| Model input | some |", "Model input"),
        ("| 1 | `V1`–`V3`, `V5` (4 columns) | < 0.01 % (12 rows) |\n", "", "has no rows"),
    ],
)
def test_uninterpretable_statement_is_an_error_with_its_line(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    text = "\n".join(CATALOG_LINES)
    assert old in text
    path = tmp_path / "catalog.md"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(SpecificationError, match=rf"catalog\.md:\d+: .*{message}"):
        read_catalog(path)


def test_missing_catalog_is_a_file_access_error(tmp_path: Path) -> None:
    with pytest.raises(FileAccessError):
        read_catalog(tmp_path / "absent.md")


def test_project_catalog_is_fully_interpretable() -> None:
    features = load_feature_spec(REPO_PATHS.features_spec)
    catalog = read_catalog(REPO_PATHS.feature_catalog)
    assert set(catalog.entries) == set(features.joined_columns) | set(features.derived)
    assert sorted(catalog.role_counts) == sorted(
        [
            "Target",
            "Excluded (identifier / raw time)",
            "Categorical input",
            "Numeric input",
            "Total",
        ]
    )
    listed = [c for block in catalog.blocks for c in block.columns]
    v_columns = [f"V{n}" for n in range(1, 340)]
    assert sorted(listed, key=lambda c: int(c[1:])) == v_columns  # every V column exactly once
    assert len(catalog.blocks) == 14
    assert all(len(b.columns) == b.stated_columns for b in catalog.blocks)
    assert catalog.v_summary is not None
    # Quantitative Caution statements become typed values, so each is checked.
    assert catalog.entries["TransactionAmt"].caution.decimals is not None
    assert catalog.entries["D9"].caution.hour is not None
    assert catalog.entries["card1"].caution.unseen is not None
    assert catalog.entries["card1"].caution.rare_rows is not None
