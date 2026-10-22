"""Specification of the human-readable Markdown output of the development pipeline (Issue #117).

The pipeline writes two formats only. JSON records (``development.records``) are the
machine-readable output; Markdown is the human-readable output. The Markdown is generated from
the JSON records alone, never written by hand, so that its values cannot differ from them, and it
presents every output of a run with its related information:

* **Comparison tables** (``COMPARISON_TABLES``), each with one of three row scopes:
  ``configuration`` (one row per configuration, in DR-011 order, then the logistic-regression
  reference), ``model`` (one table per model, one row per weighting) and ``trial`` (one table per
  configuration, one row per search trial). A column reads one record field, chosen by the record
  values it selects; ``keys`` spreads a free-key object (the searched hyperparameters) over one
  column per parameter, so every model is compared across each of its parameters.
* **Record sections**: ``record_markdown`` presents every field of a record. Nested fields are
  written as dotted paths, and lists of objects (a trial's checkpoint history, an evaluation's
  per-day values, a calibrator's reliability bins) as tables. Every value is written as its
  exact JSON literal in a literal code span (``cell``), so that no string is reinterpreted as
  Markdown or HTML and no number is rounded.

``comparison_problems`` checks every comparison-table reference against the record contracts, and
``reference_markdown`` renders the contract reference documentation from the contracts.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from review_capacity.development.records import RECORD_CONTRACTS, contract_of
from review_capacity.development.schema import Field, ListOf, MapOf, Object, RecordContract

# The primary capacity share of the protocol (Review Capacity: 2 %), at which the seed refits are
# evaluated (DR-015 §4).
PRIMARY_CAPACITY = 0.02

type Selected = str | int | float | bool
type Selection = tuple[tuple[str, tuple[Selected, ...]], ...]

ROW_SCOPES = ("configuration", "model", "trial")
FORMS = ("integer", "decimal", "seconds", "text", "boolean", "list", "mapping")


@dataclass(frozen=True)
class Source:
    """Where a column's values come from: field ``field`` of the records of kind ``kind`` that
    belong to the row and whose ``select`` fields hold one of the listed values.

    With ``per``, the column repeats once per distinct value of that field, in increasing order;
    with ``keys``, once per key of the free-key object ``field``, in the order the keys first
    appear in the records. The selection identifies the record: at most one record may match a
    row (and a ``per`` value); several matches are an error of the run, never resolved by
    choosing one."""

    kind: str
    field: str
    select: Selection = ()
    per: str | None = None
    keys: bool = False


@dataclass(frozen=True)
class ComparisonColumn:
    """A column of a comparison table: read from records (``source``) or derived from the other
    columns of its row (``derivation``)."""

    header: str
    meaning: str
    form: str
    source: Source | None = None
    derivation: str | None = None


@dataclass(frozen=True)
class ComparisonTable:
    """One comparison table of the generated Markdown; ``rows`` is its row scope."""

    name: str
    title: str
    meaning: str
    rows: str
    columns: tuple[ComparisonColumn, ...]


SELECTED: Selection = (("model_role", ("selected",)),)
V2_SELECTED: Selection = (*SELECTED, ("partition", ("V2",)), ("period", ("primary",)))
V2_R3: Selection = (*V2_SELECTED, ("policy", ("R3",)))
V2_RANKING: Selection = (*V2_SELECTED, ("policy", ("ranking",)))
SELECTED_FIT: Selection = (*SELECTED, ("measurement", ("fit",)))
STUDY_SCORING: Selection = (
    *SELECTED,
    ("measurement", ("scoring",)),
    ("partition", ("V2",)),
    ("boundary", ("end_to_end",)),
)


def _column(header: str, meaning: str, form: str, source: Source) -> ComparisonColumn:
    return ComparisonColumn(header, meaning, form, source)


COMPARISON_TABLES: tuple[ComparisonTable, ...] = (
    ComparisonTable(
        "selected_settings",
        "Selected settings",
        "For each model, the unweighted and balanced configurations side by side: the selected "
        "trial, its V1 objective and length, and every searched parameter (DR-015 §1).",
        "model",
        (
            _column(
                "Selected trial",
                "Trial with the highest V1 objective (lowest number on ties)",
                "integer",
                Source("configuration", "selected_trial"),
            ),
            _column(
                "Trials tied at the best",
                "Trials whose V1 objective equals the selected one",
                "list",
                Source("configuration", "tied_trials"),
            ),
            _column(
                "V1 objective",
                "V1 Recall@2% of the selected trial",
                "decimal",
                Source("configuration", "selected_objective"),
            ),
            _column(
                "Retained length",
                "Retained rounds or epochs",
                "integer",
                Source("configuration", "selected_length"),
            ),
            _column(
                "Stopping",
                "Early stopping, the maximum or convergence",
                "text",
                Source("configuration", "selected_stopping_reason"),
            ),
            _column(
                "Parameters",
                "One column per searched parameter of the model",
                "decimal",
                Source("configuration", "selected_hyperparameters", keys=True),
            ),
        ),
    ),
    ComparisonTable(
        "search_trials",
        "Search trials",
        "For each configuration, every search trial with every searched parameter, its V1 "
        "objective, retained checkpoint, stopping and runtime (DR-015 §1–§2).",
        "trial",
        (
            _column("Trial", "Trial number (1: reference)", "integer", Source("trial", "trial")),
            _column("Status", "Completed or failed", "text", Source("trial", "status")),
            _column(
                "Parameters",
                "One column per searched parameter of the model",
                "decimal",
                Source("trial", "hyperparameters", keys=True),
            ),
            _column(
                "V1 objective",
                "V1 Recall@2% of the trial's result",
                "decimal",
                Source("trial", "best_objective"),
            ),
            _column(
                "Best checkpoint",
                "Retained (earliest best) round or epoch",
                "integer",
                Source("trial", "best_checkpoint"),
            ),
            _column(
                "Stopped at",
                "Last round or epoch trained",
                "integer",
                Source("trial", "stopped_at"),
            ),
            _column(
                "Stopping reason",
                "Early stopping, the maximum or convergence",
                "text",
                Source("trial", "stopping_reason"),
            ),
            _column(
                "Runtime (s)",
                "Wall-clock time of the trial",
                "seconds",
                Source("trial", "runtime_seconds"),
            ),
        ),
    ),
    ComparisonTable(
        "development_metrics",
        "Development metrics",
        "The V1 objective and the V2 outcomes of every selected main-seed model.",
        "configuration",
        (
            _column(
                "V1 objective",
                "V1 Recall@2% of the selected trial",
                "decimal",
                Source("configuration", "selected_objective"),
            ),
            _column(
                "V2 recall (R3)",
                "Recall@K under daily top-K review, one column per capacity",
                "decimal",
                Source("evaluation", "metrics.recall", V2_R3, "capacity"),
            ),
            _column(
                "V2 normalised recall (R3)",
                "Pooled normalised recall, one column per capacity",
                "decimal",
                Source("evaluation", "metrics.normalised_recall", V2_R3, "capacity"),
            ),
            _column(
                "V2 Average Precision",
                "Average Precision (PR-AUC) of the whole V2 ranking",
                "decimal",
                Source("evaluation", "metrics.average_precision", V2_RANKING),
            ),
            _column(
                "V2 ROC-AUC",
                "ROC-AUC of the whole V2 ranking",
                "decimal",
                Source("evaluation", "metrics.roc_auc", V2_RANKING),
            ),
        ),
    ),
    ComparisonTable(
        "seed_variation",
        "Seed variation",
        "V2 Recall@2% under R3 for the main and additional seeds, and the same-seed repeat "
        "(DR-015 §4; descriptive only).",
        "configuration",
        (
            _column(
                "V2 Recall@2% by seed",
                "One column per training seed (main seed: the selected model)",
                "decimal",
                Source(
                    "evaluation",
                    "metrics.recall",
                    (
                        ("partition", ("V2",)),
                        ("period", ("primary",)),
                        ("policy", ("R3",)),
                        ("capacity", (PRIMARY_CAPACITY,)),
                        ("model_role", ("selected", "seed_refit")),
                    ),
                    "seed",
                ),
            ),
            ComparisonColumn(
                "Seed range",
                "Largest minus smallest Recall@2% over the seeds",
                "decimal",
                derivation="max − min of the seed columns",
            ),
            ComparisonColumn(
                "Range vs SESOI",
                "The range compared with the SESOI of DR-005 (descriptive)",
                "text",
                derivation="whether the seed range is below the SESOI",
            ),
            _column(
                "Same-seed repeat identical",
                "Whether the fresh-process main-seed refit reproduces every V2 score",
                "boolean",
                Source("refit", "comparison.identical", (("kind", ("same_seed_repeat",)),)),
            ),
            _column(
                "Same-seed largest score difference",
                "Largest absolute V2 score difference of the same-seed repeat",
                "decimal",
                Source(
                    "refit", "comparison.max_abs_difference", (("kind", ("same_seed_repeat",)),)
                ),
            ),
        ),
    ),
    ComparisonTable(
        "computational_cost",
        "Computational cost",
        "Training, scoring and latency on the project machine (protocol Outcomes; descriptive).",
        "configuration",
        (
            _column(
                "Training device",
                "Device of the selected fit",
                "text",
                Source("timing", "device", SELECTED_FIT),
            ),
            _column(
                "Search (s)",
                "Wall-clock time of the whole search",
                "seconds",
                Source("timing", "seconds", (("measurement", ("search",)),)),
            ),
            _column(
                "Selected fit (s)",
                "Wall-clock time of one fit including early stopping",
                "seconds",
                Source("timing", "seconds", SELECTED_FIT),
            ),
            _column(
                "Peak GPU memory (MiB)",
                "Peak GPU memory of the selected fit",
                "decimal",
                Source("timing", "peak_gpu_mib", SELECTED_FIT),
            ),
            _column(
                "V2 scoring (s)",
                "End-to-end V2 scoring time on the study device",
                "seconds",
                Source("timing", "seconds", STUDY_SCORING),
            ),
            _column(
                "Throughput (rows/s)",
                "End-to-end V2 scoring throughput on the study device",
                "decimal",
                Source("timing", "rows_per_second", STUDY_SCORING),
            ),
            _column(
                "Latency median / p95 / p99 (ms)",
                "Model-only single-transaction latency on the study device",
                "decimal",
                Source(
                    "timing",
                    "latency_ms",
                    (*SELECTED, ("measurement", ("latency",)), ("boundary", ("model_only",))),
                ),
            ),
            _column(
                "End-to-end latency median / p95 / p99 (ms)",
                "End-to-end single-transaction latency on the study device",
                "decimal",
                Source(
                    "timing",
                    "latency_ms",
                    (*SELECTED, ("measurement", ("latency",)), ("boundary", ("end_to_end",))),
                ),
            ),
            _column(
                "Session confirmed",
                "Whether the operator confirmed the timing session conditions (DR-015 §9)",
                "boolean",
                Source("timing", "settings.session_confirmed", (("measurement", ("search",)),)),
            ),
        ),
    ),
)


def _field(contract: RecordContract, path: str) -> Field | None:
    try:
        return contract.resolve(path)
    except KeyError:
        return None


def _source_problems(table: ComparisonTable, source: Source, where: str) -> list[str]:
    contract = RECORD_CONTRACTS.get(source.kind)
    if contract is None:
        return [f"{where}: unknown record kind {source.kind!r}"]
    if table.rows == "trial" and source.kind != "trial":
        return [f"{where}: a trial-row table reads trial records"]
    problems = [
        f"{where}: {source.kind}.{name} is not a field"
        for name in ("configuration", source.field, source.per)
        if name is not None and _field(contract, name) is None
    ]
    field = _field(contract, source.field)
    if source.keys and field is not None and not isinstance(field.kind, MapOf):
        problems.append(f"{where}: keys needs a free-key object, not {source.kind}.{source.field}")
    for name, values in source.select:
        selected = _field(contract, name)
        if selected is None:
            problems.append(f"{where}: {source.kind}.{name} is not a field")
            continue
        for value in values:
            problems += selected.kind.problems(value, f"{where} select {name}")
    return problems


def comparison_problems() -> list[str]:
    """Every reference of the comparison tables that does not match the record contracts."""
    problems: list[str] = []
    for table in COMPARISON_TABLES:
        if table.rows not in ROW_SCOPES:
            problems.append(f"{table.name}: unknown row scope {table.rows!r}")
        for column in table.columns:
            where = f"{table.name}.{column.header}"
            if column.form not in FORMS:
                problems.append(f"{where}: unknown form {column.form!r}")
            if (column.source is None) == (column.derivation is None):
                problems.append(f"{where}: needs exactly one of a source and a derivation")
            if column.source is not None:
                problems += _source_problems(table, column.source, where)
    return problems


# --- record sections -----------------------------------------------------------------------------


def code(text: str) -> str:
    """``text`` as a literal Markdown code span for a table cell: never interpreted as Markdown
    or HTML; the fence is longer than any backtick run inside, and pipes are escaped for the
    table (GitHub-flavoured Markdown removes the escape inside the span)."""
    longest = max((len(run) for run in re.findall("`+", text)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}".replace("|", "\\|")


def cell(value: object) -> str:
    """A JSON value as a Markdown table cell, written exactly as in the record: its compact JSON
    literal (integers, shortest round-trip decimals, ``true``/``false``, quoted strings with their
    escapes, lists and objects) as a literal code span; ``—`` for null."""
    if value is None:
        return "—"
    return code(json.dumps(value, ensure_ascii=False, separators=(", ", ": "), allow_nan=False))


def _flatten(obj: Object, value: Mapping[str, Any], prefix: str) -> list[tuple[str, object]]:
    """(dotted path, value) of every scalar, list and free-key entry of an object."""
    rows: list[tuple[str, object]] = []
    for f in obj.fields:
        path, item = f"{prefix}{f.name}", value[f.name]
        if isinstance(f.kind, Object) and item is not None:
            rows += _flatten(f.kind, item, f"{path}.")
        elif isinstance(f.kind, MapOf) and item:
            rows += [(f"{path}.{key}", entry) for key, entry in item.items()]
        else:
            rows.append((path, item))
    return rows


def _table(header: list[str], body: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in body),
    ]


def record_markdown(record: Mapping[str, Any], *, omit: tuple[str, ...] = ()) -> str:
    """Every field of ``record`` as Markdown: a field/value table, then one table per list of
    objects (one row per entry). The caller validates the record and adds the heading. A page
    that shows several records of one run may present their shared fields once and ``omit``
    them from the others (for example ``identity``)."""
    contract = contract_of(record)
    scalars: list[tuple[str, object]] = []
    lists: list[tuple[str, Object, list[Mapping[str, Any]]]] = []
    for f in contract.body.fields:
        if f.name in omit:
            continue
        item = record[f.name]
        if isinstance(f.kind, ListOf) and isinstance(f.kind.item, Object):
            lists.append((f.name, f.kind.item, item))
        else:
            scalars += _flatten(Object((f,)), {f.name: item}, "")
    lines = _table(["Field", "Value"], [[code(path), cell(value)] for path, value in scalars])
    for name, item, entries in lists:
        lines += ["", f"**{code(name)}** ({len(entries)} entries)", ""]
        if not entries:
            lines.append("None.")
            continue
        header = [code(path) for path, _ in _flatten(item, entries[0], "")]
        body = [[cell(value) for _, value in _flatten(item, entry, "")] for entry in entries]
        lines += _table(header, body)
    return "\n".join(lines) + "\n"


# --- reference documentation ---------------------------------------------------------------------


def _escape(text: str) -> str:
    return text.replace("|", "\\|")


def reference_markdown() -> str:
    """Reference documentation of every record contract and comparison table, generated from
    them."""
    lines = [
        "# Development records: contract reference",
        "",
        "Generated from `review_capacity.development` (records, summary); not edited by hand.",
        "JSON records are the machine-readable output and Markdown the human-readable output; no",
        "CSV files are written. Every record carries `record`, `schema_version` and the run",
        "`identity`; paths are relative to the run directory unless stated.",
        "",
        "## JSON records",
    ]
    for contract in RECORD_CONTRACTS.values():
        lines += [
            "",
            f"### `{contract.kind}` (version {contract.version})",
            "",
            _escape(contract.meaning),
            "",
            "| Field | Type | Null | Meaning |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| `{path}` | {_escape(kind)} | {'yes' if nullable else 'no'} | {_escape(meaning)} |"
            for path, kind, nullable, meaning in contract.reference_rows()
        ]
    lines += ["", "## Comparison tables of the Markdown summary"]
    for comparison in COMPARISON_TABLES:
        lines += [
            "",
            f"### {comparison.title}",
            "",
            f"{_escape(comparison.meaning)} Rows: one per `{comparison.rows}`.",
            "",
            "| Column | Meaning | Source |",
            "|---|---|---|",
        ]
        for column in comparison.columns:
            source = column.source
            if source is None:
                origin = f"derived: {column.derivation}"
            else:
                select = "; ".join(
                    f"{name} in {{{', '.join(map(str, values))}}}" for name, values in source.select
                )
                origin = f"`{source.kind}.{source.field}`" + (f" where {select}" if select else "")
                if source.per is not None:
                    origin += f"; one column per `{source.per}`"
                if source.keys:
                    origin += "; one column per parameter"
            lines.append(f"| {column.header} | {_escape(column.meaning)} | {_escape(origin)} |")
    lines.append("")
    return "\n".join(lines)
