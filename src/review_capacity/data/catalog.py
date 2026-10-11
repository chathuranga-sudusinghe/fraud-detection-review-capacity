"""Reading the statistics stated in ``docs/feature-catalog.md``, strictly.

The catalog is the human-readable companion of ``config/features.yaml``. This module turns the
statements it makes into typed values, so that ``data.profiling`` can recompute and compare them.

Parsing is strict: a table with an unknown header layout, an interpreted table without rows, a row
with the wrong number of cells, a repeated column, role or property, a listed value repeated, or a
statistic cell that cannot be interpreted raises ``SpecificationError`` with its line number. In the
Caution column, a quantitative statement is recognised by its phrase (``CAUTION_STATEMENTS``); a
recognised statement that cannot be interpreted, or that appears twice, is an error, so no
statement escapes its check. No statement is skipped silently. Tables that only explain the
catalog (legend) are ignored by design.

Recognised tables:

* column tables: the key-target-time table, the eleven-column field tables and the ``V``
  appendix (headers ``KEY_HEADERS``, ``FIELD_HEADERS``, ``APPENDIX_HEADERS``);
* the summary of roles (counts per role);
* the ``V`` property table (table, dtype, interpretation and model input of the ``V`` columns,
  missing-share and level ranges, two-valued and integer-valued counts, number of missingness
  blocks; ``V_PROSE`` rows are descriptive);
* the ``V`` missingness blocks (columns, stated column count, missing share in Train).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from review_capacity.errors import FileAccessError, SpecificationError

FIELD_HEADERS = (
    "Column", "Table", "Official meaning", "Observed dtype",
    "Observed range / most frequent values", "Interpretation", "Model input",
    "Reason for treatment", "Missing (dev)", "Levels in Train", "Caution",
)  # fmt: skip
KEY_HEADERS = (
    "Column", "Table", "Official meaning", "Observed dtype", "Observed range / values",
    "Interpretation", "Model input", "Reason for treatment", "Missing (dev)", "Caution",
)  # fmt: skip
APPENDIX_HEADERS = (
    "Column", "Missing (dev)", "Levels in Train", "Observed range", "Integer-valued",
)  # fmt: skip
ROLE_HEADERS = ("Role", "Columns", "Count")
PROPERTY_HEADERS = ("Property", "Value")
BLOCK_HEADERS = ("Block", "Columns", "Missing in Train")
INTERPRETED_HEADERS = (
    FIELD_HEADERS, KEY_HEADERS, APPENDIX_HEADERS, ROLE_HEADERS, PROPERTY_HEADERS, BLOCK_HEADERS,
)  # fmt: skip
# Tables that explain the catalog and state no statistic: the column legend and the symbol table.
IGNORED_HEADERS = (("Field", "Values"), ("Symbol / abbreviation", "Meaning"))
# Rows of the V property table: interpreted statements, then descriptive prose.
V_PROPERTIES = (
    "Table", "Observed dtype", "Interpretation", "Model input", "Missing (dev)",
    "Levels in Train", "Structure",
)  # fmt: skip
V_PROSE = ("Official meaning", "Reason for treatment", "Caution")

DTYPES = ("int64", "float64", "text")
TABLES = ("transaction", "identity", "both", "derived")
INTERPRETATIONS = ("identifier", "temporal", "target", "categorical", "numeric")
MODEL_INPUTS = ("include", "exclude")

_DIGITS = r"(?:\d{1,3}(?:,\d{3})+|\d+)"  # 18,396 or 18396; never 1,,000 or 18,39
# A whole token: no letter, digit, underscore, decimal point, comma or sign directly before it and
# no letter, digit, decimal point or comma directly after it, so that 1e9, 1e4, 1,,004, 1.7 or 2.5
# are never read as 1, 4, 4, 7 or 2.
_EDGE_BEFORE, _EDGE_AFTER = r"(?<![\w.,+\-−])", r"(?![\w.,])"
_NUMBER = rf"{_EDGE_BEFORE}[-−]?{_DIGITS}(?:\.\d+)?{_EDGE_AFTER}"
_COUNT = rf"{_EDGE_BEFORE}{_DIGITS}{_EDGE_AFTER}"  # a whole number, never truncated
_RANGE = re.compile(rf"^\s*({_NUMBER})\s+to\s+({_NUMBER})")
_COUNT_RANGE = re.compile(rf"^\s*({_COUNT})\s+to\s+({_COUNT})")
_PERCENT = re.compile(rf"^\s*(<\s*)?({_NUMBER})\s*%(?:\s*\(({_COUNT}) rows?\))?")
_ROWS_WORD = re.compile(r"\brows?\b")
_INTEGER = re.compile(rf"^{_DIGITS}$")
_BACKTICKED = re.compile(r"`([^`]*)`")
_V_RANGE = re.compile(r"`V(\d+)`(?:–`V(\d+)`)?")
_COLUMN_COUNT = re.compile(r"\((\d+) columns?\)")

# Quantitative statements of the Caution column: (name, recognising phrase, full pattern).
CAUTION_STATEMENTS = (
    ("unseen", "unseen in Train",
     re.compile(rf"Validation 2 values unseen in Train: (<\s*)?({_NUMBER})\s*%")),
    ("negatives", "negative value", re.compile(rf"({_COUNT}) negative values?\b")),
    ("decimals", "more than two decimal",
     re.compile(rf"({_NUMBER}) % of amounts have more than two decimal places")),
    ("rare_rows", "of Train rows are in levels",
     re.compile(rf"({_NUMBER}) % of Train rows are in levels with fewer than ({_COUNT}) rows")),
    ("rare_levels", "of its Train levels",
     re.compile(rf"({_NUMBER}) % of its Train levels have fewer than ({_COUNT}) rows")),
    ("hour", "hour of the relative day",
     re.compile(rf"hour of the relative day\) ÷ 24 on ({_NUMBER}) % of non-missing "
                "development rows")),
)  # fmt: skip


@dataclass(frozen=True)
class Number:
    """A number as written in the catalog, with the decimals it shows.

    ``matches`` accepts an observed value within half a unit of the last shown digit, inclusive;
    this tolerates either rounding convention at the boundary.
    """

    value: float
    decimals: int

    @classmethod
    def parse(cls, text: str) -> Number:
        cleaned = text.replace(",", "").replace("−", "-")
        decimals = len(cleaned.split(".")[1]) if "." in cleaned else 0
        return cls(float(cleaned), decimals)

    def matches(self, observed: float) -> bool:
        return abs(observed - self.value) <= 0.5 * 10.0**-self.decimals + 1e-9


@dataclass(frozen=True)
class Percent:
    """A share written as ``x %``, ``< x %`` or ``< x % (n rows)``."""

    value: Number
    less_than: bool
    rows: int | None

    def matches(self, share: float, rows: int) -> bool:
        percent = 100 * share
        if self.rows is not None and rows != self.rows:
            return False
        if self.less_than:
            return 0 <= percent < self.value.value
        return self.value.matches(percent)


@dataclass(frozen=True)
class Caution:
    """The quantitative statements of one Caution cell (``None`` where not stated)."""

    unseen: Percent | None = None  # Validation 2 values unseen in Train
    negatives: int | None = None  # negative values in the development rows
    decimals: Number | None = None  # % of amounts with more than two decimal places
    rare_rows: tuple[Number, int] | None = None  # % of Train rows in levels below n rows
    rare_levels: tuple[Number, int] | None = None  # % of Train levels below n rows
    hour: Number | None = None  # % of rows equal to (hour of the relative day) / 24


@dataclass(frozen=True)
class CatalogEntry:
    """The interpreted statements of one column row (``None`` where the table has no cell)."""

    column: str
    line: int
    table: str | None
    dtype: str | None
    interpretation: str | None
    model_input: str | None
    numeric_range: tuple[Number, Number] | None
    top_values: tuple[str, ...] | None
    missing: Percent | None
    levels: int | None
    integer_valued: bool | None
    caution: Caution


@dataclass(frozen=True)
class CatalogBlock:
    number: int
    line: int
    columns: tuple[str, ...]
    stated_columns: int
    missing_in_train: Percent


@dataclass(frozen=True)
class VSummary:
    """Statements of the ``V`` property table."""

    line: int
    table: str
    dtype: str
    interpretation: str
    model_input: str
    missing_range: tuple[Percent, Percent]
    levels_range: tuple[int, int]
    two_valued: int
    integer_valued: int
    blocks: int  # blocks with an identical missing-row pattern in Train


@dataclass(frozen=True)
class Catalog:
    path: Path
    entries: dict[str, CatalogEntry]
    role_counts: dict[str, int]
    blocks: tuple[CatalogBlock, ...]
    v_summary: VSummary | None


class _Source:
    """Raises ``SpecificationError`` that name the file and line."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def error(self, line: int, message: str) -> SpecificationError:
        return SpecificationError(f"{self.path.name}:{line}: {message}")


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _tables(lines: list[str]) -> Iterator[tuple[int, tuple[str, ...], list[tuple[int, list[str]]]]]:
    """Every Markdown table: header line number, header cells and numbered body rows."""
    i = 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|---"):
            header_line, header = i + 1, tuple(_cells(lines[i]))
            rows = []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append((i + 1, _cells(lines[i])))
                i += 1
            yield header_line, header, rows
        else:
            i += 1


def _first_word(text: str) -> str:
    words = text.replace("*", "").split()
    return words[0] if words else ""


def _choice(src: _Source, line: int, column: str, header: str, text: str, allowed: tuple[str, ...]
            ) -> str:  # fmt: skip
    word = _first_word(text)
    if word not in allowed:
        raise src.error(line, f"{column}: {header} {text!r} is not one of {allowed}")
    return word


def _percent(src: _Source, line: int, where: str, text: str) -> Percent:
    match = _PERCENT.match(text)
    if match is None:
        raise src.error(line, f"{where}: percentage {text!r} cannot be interpreted")
    if match.group(3) is None and _ROWS_WORD.search(text):
        raise src.error(line, f"{where}: row count in {text!r} cannot be interpreted")
    rows = int(match.group(3).replace(",", "")) if match.group(3) else None
    return Percent(Number.parse(match.group(2)), bool(match.group(1)), rows)


def _integer(src: _Source, line: int, where: str, text: str) -> int:
    cleaned = text.replace("*", "").strip()
    if not _INTEGER.match(cleaned):
        raise src.error(line, f"{where}: integer {text!r} cannot be interpreted")
    return int(cleaned.replace(",", ""))


def _caution(src: _Source, line: int, column: str, text: str) -> Caution:
    caution = Caution()
    for name, phrase, pattern in CAUTION_STATEMENTS:
        stated = text.count(phrase)
        if not stated:
            continue
        matches = list(pattern.finditer(text))
        if stated > 1 or len(matches) != 1:
            problem = "is stated more than once" if stated > 1 else "cannot be interpreted"
            raise src.error(line, f"{column}: Caution statement '{phrase}' {problem}")
        groups = matches[0].groups()
        if name == "unseen":
            unseen = Percent(Number.parse(groups[1]), less_than=bool(groups[0]), rows=None)
            caution = replace(caution, unseen=unseen)
        elif name == "negatives":
            caution = replace(caution, negatives=int(groups[0].replace(",", "")))
        elif name == "decimals":
            caution = replace(caution, decimals=Number.parse(groups[0]))
        elif name == "rare_rows":
            caution = replace(caution, rare_rows=_share_below(groups))
        elif name == "rare_levels":
            caution = replace(caution, rare_levels=_share_below(groups))
        else:
            caution = replace(caution, hour=Number.parse(groups[0]))
    return caution


def _share_below(groups: tuple[str | Any, ...]) -> tuple[Number, int]:
    """A share and its row threshold: ``x % of ... fewer than n rows``."""
    return Number.parse(groups[0]), int(groups[1].replace(",", ""))


def _entry(src: _Source, line: int, header: tuple[str, ...], row: list[str], target: str
           ) -> CatalogEntry:  # fmt: skip
    if len(row) != len(header):
        raise src.error(line, f"row has {len(row)} cells, the table has {len(header)} columns")
    names = _BACKTICKED.findall(row[0])
    if len(names) != 1 or row[0] != f"`{names[0]}`":
        raise src.error(line, f"first cell {row[0]!r} must be one column name in backticks")
    column = names[0]
    cell = dict(zip(header, row, strict=True))

    def get(name: str) -> str | None:
        return cell.get(name)

    dtype_text = get("Observed dtype")
    is_target = column == target
    dtype = None
    if dtype_text is not None and not is_target:
        dtype = _choice(src, line, column, "Observed dtype", dtype_text, DTYPES)
    range_text = get(FIELD_HEADERS[4]) or get(KEY_HEADERS[4]) or get("Observed range")
    numeric_range = top_values = None
    if range_text is not None and not is_target:
        if dtype == "text":
            top_values = tuple(_BACKTICKED.findall(range_text))
            if not top_values:
                raise src.error(line, f"{column}: no value in backticks in {range_text!r}")
            if len(set(top_values)) != len(top_values):
                raise src.error(line, f"{column}: a listed value is repeated in {range_text!r}")
        else:
            match = _RANGE.match(range_text)
            if match is None:
                raise src.error(line, f"{column}: range {range_text!r} is not 'a to b'")
            numeric_range = (Number.parse(match.group(1)), Number.parse(match.group(2)))
    missing_text = get("Missing (dev)")
    levels_text = get("Levels in Train")
    integer_text = get("Integer-valued")
    if integer_text is not None and integer_text not in ("yes", "no"):
        raise src.error(line, f"{column}: Integer-valued {integer_text!r} is not yes or no")
    table, interpretation, model_input = get("Table"), get("Interpretation"), get("Model input")
    return CatalogEntry(
        column=column,
        line=line,
        table=None if table is None else _choice(src, line, column, "Table", table, TABLES),
        dtype=dtype,
        interpretation=None
        if interpretation is None
        else _choice(src, line, column, "Interpretation", interpretation, INTERPRETATIONS),
        model_input=None
        if model_input is None
        else _choice(src, line, column, "Model input", model_input, MODEL_INPUTS),
        numeric_range=numeric_range,
        top_values=top_values,
        missing=None
        if missing_text is None or is_target
        else _percent(src, line, column, missing_text),
        levels=None if levels_text is None else _integer(src, line, column, levels_text),
        integer_valued=None if integer_text is None else integer_text == "yes",
        caution=_caution(src, line, column, get("Caution") or ""),
    )


def _block(src: _Source, line: int, row: list[str]) -> CatalogBlock:
    if len(row) != len(BLOCK_HEADERS):
        raise src.error(line, f"block row has {len(row)} cells, expected {len(BLOCK_HEADERS)}")
    number = _integer(src, line, "block number", row[0])
    columns: list[str] = []
    for first, last in _V_RANGE.findall(row[1]):
        start, end = int(first), int(last) if last else int(first)
        if end < start:
            raise src.error(line, f"block {number}: empty range V{start}–V{end}")
        columns.extend(f"V{n}" for n in range(start, end + 1))
    count = _COLUMN_COUNT.search(row[1])
    if not columns or count is None:
        raise src.error(line, f"block {number}: columns or '(N columns)' cannot be interpreted")
    return CatalogBlock(
        number=number,
        line=line,
        columns=tuple(columns),
        stated_columns=int(count.group(1)),
        missing_in_train=_percent(src, line, f"block {number}", row[2]),
    )


def _named_rows(src: _Source, header: tuple[str, ...], rows: list[tuple[int, list[str]]]
                ) -> list[tuple[int, list[str]]]:  # fmt: skip
    """Rows of a summary table, each with exactly the table's number of cells."""
    for line, row in rows:
        if len(row) != len(header):
            raise src.error(line, f"row has {len(row)} cells, the table has {len(header)} columns")
    return rows


def _v_summary(src: _Source, header_line: int, line_of: dict[str, int], values: dict[str, str]
               ) -> VSummary:  # fmt: skip
    absent = [name for name in V_PROPERTIES if name not in values]
    if absent:
        raise src.error(header_line, f"V property table lacks {absent}")
    missing_text, levels_text = values["Missing (dev)"], values["Levels in Train"]
    structure = re.match(r"(\d+) blocks of columns share an identical missing-row pattern in "
                         r"Train", values["Structure"])  # fmt: skip
    if structure is None:
        raise src.error(line_of["Structure"], "V structure statement cannot be interpreted")

    def choice(name: str, allowed: tuple[str, ...]) -> str:
        return _choice(src, line_of[name], "V columns", name, values[name], allowed)

    low, _, high = missing_text.partition(" to ")
    levels = _COUNT_RANGE.match(levels_text)
    two = re.search(rf"({_COUNT}) columns are two-valued", levels_text)
    integer = re.search(rf"({_COUNT}) are integer-valued", levels_text)
    line = line_of["Levels in Train"]
    if levels is None or two is None or integer is None:
        raise src.error(line, f"V levels statement {levels_text!r} cannot be interpreted")
    return VSummary(
        line=line,
        table=choice("Table", TABLES),
        dtype=choice("Observed dtype", DTYPES),
        interpretation=choice("Interpretation", INTERPRETATIONS),
        model_input=choice("Model input", MODEL_INPUTS),
        missing_range=(
            _percent(src, line_of["Missing (dev)"], "V missing range", low),
            _percent(src, line_of["Missing (dev)"], "V missing range", high),
        ),
        levels_range=(
            int(levels.group(1).replace(",", "")),
            int(levels.group(2).replace(",", "")),
        ),
        two_valued=int(two.group(1).replace(",", "")),
        integer_valued=int(integer.group(1).replace(",", "")),
        blocks=int(structure.group(1)),
    )


def read_catalog(path: Path, *, target: str = "isFraud") -> Catalog:
    """Read and interpret every statistic the catalog states; raise on anything uninterpretable."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise FileAccessError(f"{path}: cannot be read") from error
    src = _Source(path)
    entries: dict[str, CatalogEntry] = {}
    role_counts: dict[str, int] = {}
    blocks: list[CatalogBlock] = []
    v_summary: VSummary | None = None
    for header_line, header, rows in _tables(lines):
        if header in IGNORED_HEADERS:
            continue
        if header not in INTERPRETED_HEADERS:
            raise src.error(header_line, f"unrecognised table header {header}")
        if not rows:
            raise src.error(header_line, f"table {header[0]!r} has no rows")
        if header in (FIELD_HEADERS, KEY_HEADERS, APPENDIX_HEADERS):
            for line, row in rows:
                entry = _entry(src, line, header, row, target)
                if entry.column in entries:
                    raise src.error(line, f"column {entry.column} appears twice")
                entries[entry.column] = entry
        elif header == ROLE_HEADERS:
            for line, row in _named_rows(src, header, rows):
                role = row[0].replace("*", "")
                if role in role_counts:
                    raise src.error(line, f"role {role!r} appears twice")
                role_counts[role] = _integer(src, line, "role count", row[2])
        elif header == PROPERTY_HEADERS:
            if v_summary is not None:
                raise src.error(header_line, "a second V property table")
            named = _named_rows(src, header, rows)
            line_of: dict[str, int] = {}
            for line, row in named:
                if row[0] not in V_PROPERTIES + V_PROSE:
                    raise src.error(line, f"unknown V property {row[0]!r}")
                if row[0] in line_of:
                    raise src.error(line, f"V property {row[0]!r} appears twice")
                line_of[row[0]] = line
            v_summary = _v_summary(src, header_line, line_of, {row[0]: row[1] for _, row in named})
        else:
            blocks.extend(_block(src, line, row) for line, row in rows)
    if not entries:
        raise src.error(1, "no column table found")
    return Catalog(path, entries, role_counts, tuple(blocks), v_summary)
