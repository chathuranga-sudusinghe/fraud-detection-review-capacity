"""The Markdown summary specification: comparison tables, record sections and the generated
reference (synthetic records; no data read)."""

from __future__ import annotations

import json

import pytest

from review_capacity.development import summary
from review_capacity.development.records import (
    RECORD_CONTRACTS,
    contract_versions,
    validate_record,
)
from review_capacity.development.summary import (
    COMPARISON_TABLES,
    ComparisonColumn,
    ComparisonTable,
    Source,
    cell,
    comparison_problems,
    record_markdown,
    reference_markdown,
)
from support.development import PARTITIONS, example


def test_comparison_tables_match_the_record_contracts() -> None:
    assert [(t.name, t.rows) for t in COMPARISON_TABLES] == [
        ("selected_settings", "model"),
        ("search_trials", "trial"),
        ("development_metrics", "configuration"),
        ("seed_variation", "configuration"),
        ("device_comparison", "configuration"),
        ("break_even", "configuration"),
        ("family_contrasts", "contrast"),
        ("computational_cost", "configuration"),
    ]
    assert comparison_problems() == []


def test_parameter_tables_spread_every_searched_parameter() -> None:
    spread = [
        (table.name, column.source.kind, column.source.field)
        for table in COMPARISON_TABLES
        for column in table.columns
        if column.source is not None and column.source.keys
    ]
    assert spread == [
        ("selected_settings", "configuration", "selected_hyperparameters"),
        ("search_trials", "trial", "hyperparameters"),
    ]


def test_comparison_problems_are_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = ComparisonTable(
        "broken",
        "Broken",
        "References that do not exist",
        "configuration",
        (
            ComparisonColumn("A", "Unknown kind", "text", Source("summary", "x")),
            ComparisonColumn("B", "Unknown field", "text", Source("evaluation", "metrics.x.y")),
            ComparisonColumn(
                "C", "Invalid selection", "text", Source("evaluation", "k", (("policy", ("R9",)),))
            ),
            ComparisonColumn("D", "No configuration", "text", Source("h3_selection", "k")),
            ComparisonColumn("E", "Neither source nor derivation", "number"),
            ComparisonColumn("F", "Keys of a scalar", "text", Source("trial", "seed", keys=True)),
        ),
    )
    other_kind = (ComparisonColumn("G", "Not a trial", "text", Source("fit", "seed")),)
    trials = ComparisonTable("trials", "Trials", "Reads another kind", "trial", other_kind)
    scope = ComparisonTable("scope", "Scope", "Unknown rows", "day", ())
    monkeypatch.setattr(summary, "COMPARISON_TABLES", (broken, trials, scope))
    assert comparison_problems() == [
        "broken.A: unknown record kind 'summary'",
        "broken.B: evaluation.metrics.x.y is not a field",
        "broken.C select policy: 'R9' is not one of ranking, R0, R1, R2, R3, R4",
        "broken.D: h3_selection.configuration is not a field",
        "broken.E: unknown form 'number'",
        "broken.E: needs exactly one of a source and a derivation",
        "broken.F: keys needs a free-key object, not trial.seed",
        "trials.G: a trial-row table reads trial records",
        "scope: unknown row scope 'day'",
    ]


def decode(text: str) -> object:
    """The JSON value of a cell written by ``cell``: GitHub-flavoured Markdown's reading of the
    code span (fence removed, table pipe escapes removed)."""
    if text == "—":
        return None
    fence = len(text) - len(text.lstrip("`"))
    inner = text[fence:-fence].replace("\\|", "|")
    if inner.startswith(" ") and inner.endswith(" ") and inner.strip(" "):
        inner = inner[1:-1]
    return json.loads(inner)


def test_cells_write_values_exactly_as_in_the_record() -> None:
    assert cell(0.1 + 0.2) == "`0.30000000000000004`"
    assert cell(5e-324) == "`5e-324`"
    assert cell(None) == "—"
    assert cell(True) == "`true`"
    assert cell(56) == "`56`"
    assert cell([2, 3]) == "`[2, 3]`"
    assert cell({"a": None}) == '`{"a": null}`'
    assert cell("a|b\nc") == '`"a\\|b\\nc"`'
    assert cell("x`y") == '``"x`y"``'


@pytest.mark.parametrize(
    "value",
    [
        "line 1\nline *2* | <b>bold</b> `code` ``fence``",
        "",
        "—",
        0.1 + 0.2,
        1e-300,
        -0.0,
        2**80,
        [1, "a|b", None, {"k": [True, 0.5]}],
        {"nested": {"x": "<script>"}},
    ],
)
def test_cells_decode_to_the_recorded_value(value: object) -> None:
    assert decode(cell(value)) == value


@pytest.mark.parametrize("kind", list(RECORD_CONTRACTS))
def test_field_tables_hold_exact_values(kind: str) -> None:
    lines = record_markdown(example(kind)).split("\n\n")[0].splitlines()[2:]
    for line in lines:
        value = line.rsplit(" | ", 1)[1].removesuffix(" |")
        if value != "—":
            decode(value)


def test_error_text_is_presented_unchanged() -> None:
    error = "fit failed:\n  *CUDA* out of memory | <stop>"
    record = example("fit") | {"status": "failed", "error": error}
    rows = record_markdown(record).splitlines()
    (row,) = [line for line in rows if line.startswith("| `error` |")]
    assert decode(row.removeprefix("| `error` | ").removesuffix(" |")) == error


@pytest.mark.parametrize("kind", list(RECORD_CONTRACTS))
def test_record_sections_present_every_field(kind: str) -> None:
    record = example(kind)
    text = record_markdown(record)
    assert text.startswith("| Field | Value |\n|---|---|\n")
    for path, _, _, _ in RECORD_CONTRACTS[kind].reference_rows():
        top = path.split("[]")[0].split(".")[0]
        assert f"`{top}" in text, path


def test_lists_of_objects_become_tables() -> None:
    trial = record_markdown(example("trial"))
    assert "**`checkpoints`** (170 entries)" in trial
    assert "| `checkpoint` | `objective` | `improved` | `best_checkpoint` | `seconds` |" in trial
    assert "| `120` | `0.41` | `true` | `120` | `12.0` |" in trial
    assert "| `hyperparameters.num_leaves` | `31` |" in trial
    assert "| `identity.seeds.additional` | `[43, 44]` |" in trial
    evaluation = record_markdown(example("evaluation"))
    assert "**`daily`** (28 entries)" in evaluation
    calibration = record_markdown(example("calibration"))
    assert "| `rows_used` | `days.first` | `days.last` |" in calibration
    assert "| `reliability` |" in calibration.split("**`diagnostics`**")[1].splitlines()[2]


def test_empty_lists_are_stated() -> None:
    reference = example("trial") | {"configuration": "LR", "checkpoints": []}
    assert "**`checkpoints`** (0 entries)\n\nNone." in record_markdown(reference)


def test_reference_documents_every_contract() -> None:
    text = reference_markdown()
    for kind, contract in RECORD_CONTRACTS.items():
        assert f"### `{kind}` (version {contract.version})" in text
    for comparison in COMPARISON_TABLES:
        assert f"### {comparison.title}" in text
    assert "no\nCSV files are written" in text
    assert "| `identity.seeds.additional` |" in text
    assert "| `checkpoints[].objective` |" in text
    assert "| `diagnostics[].reliability` |" in text
    assert "; one column per `capacity`" in text
    assert "; one column per parameter" in text


def test_run_records_list_the_contract_versions() -> None:
    assert contract_versions() == {kind: c.version for kind, c in RECORD_CONTRACTS.items()}
    validate_record(example("run") | {"contracts": contract_versions()}, PARTITIONS)


def test_a_long_curve_shows_its_first_and_last_segments_only() -> None:
    record = example("cost_analysis")
    segments = [
        {
            "start": 1.0 + i,
            "end": 2.0 + i,
            "end_closed": False,
            "unused": 9000 - i,
            "frauds_forgone": 0,
        }
        for i in range(5000)
    ]
    text = record_markdown(record | {"curve": segments})
    assert (
        "**`curve`** (5000 entries; the first 10 and the last 10 are shown, all are in the record)"
        in text
    )
    curve_lines = text.split("**`curve`**")[1].split("**`grid`**")[0].splitlines()
    rows = [
        line for line in curve_lines if line.startswith("| `") and not line.startswith("| `start`")
    ]
    assert len(rows) == 20  # not 5,000
    assert rows[0].startswith("| `1.0` |")
    assert rows[-1].startswith("| `5000.0` |")
    assert len(text) < 20_000  # a bounded page however long the curve
    # a short curve is shown in full, as every other list
    short = record_markdown(example("cost_analysis"))
    assert "**`curve`** (4 entries)" in short
