"""The generated Markdown summary of a development run (Issue #120; Issue #96).

``render_summary`` builds the summary from the validated records of a run directory alone, so
that its values cannot differ from them (``development.summary`` specifies the comparison tables
and the record sections; ``development.run_output`` loads the records). It is a pure function of
the records: regenerating it after an interruption, or years later, gives the same text.

The summary has the run's identity and outcome, the comparison tables of every model across each
of its parameters, and one section per record. Every value is written exactly as in its record
(``summary.cell``); a value that no record holds is ``—``.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from review_capacity.development.records import (
    ALL_CONFIGURATIONS,
    MODEL_OF,
    DevelopmentPartitions,
    contract_of,
)
from review_capacity.development.run_output import SUMMARY_NAME, LoadedRun, load_run
from review_capacity.development.summary import (
    COMPARISON_TABLES,
    ComparisonColumn,
    ComparisonTable,
    Source,
    cell,
    code,
    record_markdown,
)
from review_capacity.errors import RecordContractError
from review_capacity.files import write_atomically

logger = logging.getLogger(__name__)

# The smallest effect size of interest, 2 percentage points of Recall@2% (DR-005 §1).
SESOI = 0.02

SEED_RANGE = "max − min of the seed columns"
SEED_VS_SESOI = "whether the seed range is below the SESOI"

type Record = Mapping[str, Any]
type Getter = Callable[[Record], object]


def _order(configuration: str) -> int:
    return ALL_CONFIGURATIONS.index(configuration)


def _value(record: Record, path: str) -> object:
    """The field at the dotted ``path`` of ``record``; ``None`` if a segment is absent."""
    value: object = record
    for segment in path.split("."):
        if not isinstance(value, Mapping) or segment not in value:
            return None
        value = value[segment]
    return value


def _selected(record: Record, source: Source) -> bool:
    return record["record"] == source.kind and all(
        record.get(name) in values for name, values in source.select
    )


def _header(text: str) -> str:
    return text.replace("|", "\\|")


class _Table:
    """The records a comparison table reads, and its columns resolved over them."""

    def __init__(self, run: LoadedRun, table: ComparisonTable, rows: Sequence[Record]) -> None:
        self.run = run
        self.table = table
        self.rows = rows
        self.columns: list[tuple[str, Getter]] = []
        self._seed_source: Source | None = None
        self._range: Getter | None = None
        for column in table.columns:
            self._add(column)

    def _candidates(self, source: Source) -> list[Record]:
        names = {row.get("configuration") for row in self.rows}
        return [
            r
            for r in self.run.of_kind(source.kind)
            if _selected(r, source) and r.get("configuration") in names
        ]

    def _match(
        self, row: Record, source: Source, extra: tuple[str, object] | None
    ) -> Record | None:
        if self.table.rows == "trial":
            found = [row] if _selected(row, source) else []
        else:
            found = [
                r
                for r in self._candidates(source)
                if r.get("configuration") == row["configuration"]
                and (extra is None or r.get(extra[0]) == extra[1])
            ]
        if len(found) > 1:
            raise RecordContractError(
                f"{self.table.name}: {len(found)} {source.kind} records match "
                f"{row['configuration']} {source.select}; the selection must identify one"
            )
        return found[0] if found else None

    def _add(self, column: ComparisonColumn) -> None:
        source = column.source
        if source is None:
            self._add_derived(column)
            return
        if source.keys:
            keys: dict[str, None] = {}
            for row in self.rows:
                record = self._match(row, source, None)
                spread = _value(record, source.field) if record is not None else None
                if isinstance(spread, Mapping):
                    keys.update(dict.fromkeys(spread))
            for key in keys:
                self.columns.append((_header(key), self._reader(source, None, key)))
        elif source.per is not None:
            values = sorted({r[source.per] for r in self._candidates(source)})
            for value in values:
                header = _header(f"{column.header} ({source.per} {value})")
                getter = self._reader(source, (source.per, value), None)
                self.columns.append((header, getter))
                if source.per == "seed":
                    self._seed_source = source
        else:
            self.columns.append((_header(column.header), self._reader(source, None, None)))

    def _reader(self, source: Source, extra: tuple[str, object] | None, key: str | None) -> Getter:
        def read(row: Record) -> object:
            record = self._match(row, source, extra)
            if record is None:
                return None
            found = _value(record, source.field)
            return found.get(key) if key is not None and isinstance(found, Mapping) else found

        return read

    def _add_derived(self, column: ComparisonColumn) -> None:

        def seed_range(row: Record) -> object:
            """The range over the main and every additional seed of the run (DR-015 §4); a
            seed without a result leaves the range, and the comparison with the SESOI, open."""
            source = self._seed_source
            if source is None:
                return None
            seeds = self.run.identity["seeds"]
            declared = [seeds["main"], *seeds["additional"]]
            values = [self._reader(source, ("seed", seed), None)(row) for seed in declared]
            numbers = [v for v in values if isinstance(v, int | float)]
            return max(numbers) - min(numbers) if len(numbers) == len(declared) else None

        if column.derivation == SEED_RANGE:
            self._range = seed_range
            self.columns.append((_header(column.header), seed_range))
        elif column.derivation == SEED_VS_SESOI:
            spread = self._range

            def below(row: Record) -> object:
                width = spread(row) if spread is not None else None
                if not isinstance(width, int | float):
                    return None
                return "below" if width < SESOI else "not below"

            self.columns.append((_header(column.header), below))
        else:
            raise RecordContractError(
                f"{self.table.name}: unknown derivation {column.derivation!r}"
            )

    def markdown(self, label: Callable[[Record], str]) -> list[str]:
        header = ["Configuration", *(name for name, _ in self.columns)]
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for row in self.rows:
            cells = [label(row), *(cell(get(row)) for _, get in self.columns)]
            lines.append("| " + " | ".join(cells) + " |")
        return lines


def _configurations(run: LoadedRun) -> list[str]:
    names = run.of_kind("run")[0]["configurations"]
    return sorted(names, key=_order)


def _tables(run: LoadedRun) -> list[str]:
    lines = ["## Comparison tables"]
    configurations = _configurations(run)
    identities = Counter((t["configuration"], t["trial"]) for t in run.of_kind("trial"))
    repeated = sorted(f"{c} trial {n}" for (c, n), count in identities.items() if count > 1)
    if repeated:
        raise RecordContractError(f"Several trial records describe the same trial: {repeated}")
    for table in COMPARISON_TABLES:
        lines += ["", f"### {table.title}", "", table.meaning]
        if table.rows == "configuration":
            groups: list[tuple[str, list[dict[str, Any]]]] = [
                ("", [{"configuration": name} for name in configurations])
            ]
        elif table.rows == "model":
            models = dict.fromkeys(MODEL_OF[name] for name in configurations)
            groups = [
                (model, [{"configuration": n} for n in configurations if MODEL_OF[n] == model])
                for model in models
            ]
        else:
            groups = [
                (name, sorted(trials, key=lambda r: r["trial"]))
                for name in configurations
                if (trials := [t for t in run.of_kind("trial") if t["configuration"] == name])
            ]
        if not groups:
            lines += ["", "No records yet."]
        for group, rows in groups:
            lines += ["", f"**{code(group)}**", ""] if group else [""]
            lines += _Table(run, table, rows).markdown(_row_label(table.rows))
        if table.name == "computational_cost":
            lines += _cost_notes(run)
    return lines


def _cost_notes(run: LoadedRun) -> list[str]:
    """Notes under the computational-cost table: search times of interrupted searches
    (DR-015 §9)."""
    timings = run.of_kind("timing")
    partial = sorted(
        f"{t['configuration']} ({t['settings']['trials_completed']} of "
        f"{t['settings']['trials_declared']} trials)"
        for t in timings
        if t["measurement"] == "search" and t["settings"].get("interrupted") is True
    )
    lines: list[str] = []
    if partial:
        lines += [
            "",
            "**Partial search times:** the search of "
            f"{', '.join(partial)} was interrupted; its time is not a completed-search time.",
        ]
    return lines


def _row_label(scope: str) -> Callable[[Record], str]:
    if scope == "trial":
        return lambda row: code(f"{row['configuration']} trial {row['trial']}")
    return lambda row: code(str(row["configuration"]))


def _outcome(run: LoadedRun) -> list[str]:
    start = run.of_kind("run")[0]["started"]
    rows: list[tuple[str, object]] = [
        ("run_type", run.identity["run_type"]),
        ("run_id", run.identity["run_id"]),
        ("started", start),
    ]
    if run.end is None:
        rows += [
            ("outcome", "interrupted before closing (no run_end record)"),
            ("records", len(run.records)),
        ]
    else:
        rows += [
            ("outcome", run.end["outcome"]),
            ("finished", run.end["finished"]),
            ("error", run.end["error"]),
            ("records", len(run.records) + 1),
        ]
    return ["| Field | Value |", "|---|---|", *(f"| {code(k)} | {cell(v)} |" for k, v in rows)]


def render_summary(run: LoadedRun) -> str:
    """The Markdown summary of ``run``, from its records alone."""
    first = run.records[0][1]
    shared = tuple(f.name for f in contract_of(first).body.fields if f.name != "identity")
    lines = [
        f"# Development run {run.identity['run_id']}",
        "",
        "Generated from the JSON records of this run directory; not edited by hand.",
        "",
        *_outcome(run),
        "",
        "## Run identity",
        "",
        record_markdown(first, omit=shared).rstrip("\n"),
        "",
        *_tables(run),
        "",
        "## Records",
    ]
    sections = [*run.records, *([("run_end.json", run.end)] if run.end is not None else [])]
    for name, record in sections:
        lines += ["", f"### {code(name)}", "", record_markdown(record, omit=("identity",)).rstrip()]
    return "\n".join(lines) + "\n"


def write_summary(directory: Path, partitions: DevelopmentPartitions) -> Path:
    """Regenerate ``summary.md`` of a run directory from its records, atomically."""
    text = render_summary(load_run(directory, partitions))
    path = directory / SUMMARY_NAME
    write_atomically(path, lambda target: target.write_text(text, encoding="utf-8"))
    logger.info("Summary written to %s", path)
    return path
