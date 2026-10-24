"""The tables of the analysis summary: threshold policies and their workload, the rate and volume
components of the daily demand, the calibration diagnostics and the H3 configuration selection
(Issue #166; DR-009, DR-003, DR-007 §8, DR-010).

``analysis_markdown`` builds them from the records of an analysis run together with those of its
source development run (the merged view of ``rendering``), and nothing else: every value is one
that a record holds or is computed from the ``daily`` rows of a record, so regenerating the summary
gives the same text. Each section states what it shows; a value that no record holds is ``—``.

The rate/volume decomposition of the daily demand (DR-003) is derived here: with ``n_d`` the alerts
of the R2 threshold on day ``d``, ``N_d`` the transactions of day ``d`` and ``N̄`` their mean over
the evaluated days, for the days with ``n_d > 0``

$$
\\ln\\frac{n_d}{K} \\approx \\ln\\frac{n_d / N_d}{c} + \\ln\\frac{N_d}{\\bar N}
$$

The first term on the right is the rate component (model and data drift) and the second the volume
component; the approximation is exact up to the rounding of ``K``. The table reports the means of
both components, the mean of the left side and the days without an alert, which are left out of
the logarithms and counted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from review_capacity.development.records import ALL_CONFIGURATIONS, CONFIGURATION_NAMES
from review_capacity.development.run_output import LoadedRun
from review_capacity.development.summary import cell, code, family_label
from review_capacity.errors import RecordContractError

type Record = Mapping[str, Any]

PRIMARY_CAPACITY = 0.02
POLICIES = ("R0", "R1", "R2")


@dataclass(frozen=True)
class AnalysisSection:
    """One section of the analysis summary: its name, title and what it shows."""

    name: str
    title: str
    meaning: str


SECTIONS: tuple[AnalysisSection, ...] = (
    AnalysisSection(
        "threshold_outcomes",
        "Threshold policies R0–R2 at 2 % capacity (V2)",
        "Alerts and the alert-set recall, precision and F1 of the uncapped threshold policies on "
        "V2 (DR-009): R0 (the default 0.5 threshold on the raw score), R1 (the F1-optimal "
        "threshold selected on V2) and R2 (the 2 % capacity-rate threshold). The measures "
        "describe the whole alert set; the policies generate different workloads, so recalls are "
        "read together with the alerts. A frauds-in-excess count is not part of this issue.",
    ),
    AnalysisSection(
        "threshold_workload",
        "Threshold workload against the fixed daily budget (2 % capacity, V2)",
        "Overflow (alerts above K), unused capacity (K above the alerts) and the mean load ratio "
        "(alerts / K) of R0, R1 and R2 over the V2 days, with the days above and below the "
        "budget (DR-009). Alerts above K are neither missed nor false negatives.",
    ),
    AnalysisSection(
        "rate_volume",
        "Rate and volume components of the R2 demand (2 % capacity, V2)",
        "The realised V2 alert rate of R2 and the decomposition of its daily demand (DR-003): "
        "the mean over the days with an alert of the rate component and of the volume "
        "component, and of their sum's left side ln(n_d / K); days without an alert are "
        "counted. In-sample for the threshold, which is selected on V2.",
    ),
    AnalysisSection(
        "calibration",
        "Calibration diagnostics (V2)",
        "The primary calibrators fitted on all of V2 (Platt scaling and the isotonic check; "
        "DR-007 §8), diagnosed in-sample: the Platt parameters, the Brier score, the "
        "ε-clipped log loss and the calibration intercept and slope (— when not estimable); "
        "and the label-latency Platt calibrator on its held-out days.",
    ),
    AnalysisSection(
        "h3_selection",
        "H3 configuration selection",
        "The eight configurations ranked by the V2 frauds captured under R3 at 2 % capacity "
        "(DR-010): the binding selection on all of V2 and the label-latency check on the "
        "fitting days (descriptive; it does not change the H3 configuration).",
    ),
    AnalysisSection(
        "rolling_origin",
        "Rolling-origin supplement (primary configuration)",
        "The exploratory transfer of the R2 threshold across origins (protocol; DR-001, DR-003): "
        "at each origin the primary configuration is refitted on the days before the gap, the "
        "R2 threshold (2 % capacity) is set on the scores of the next 28 days (no labels) and "
        "applied to the following 28 days against the origin's fixed daily budget K. "
        "Retrospective: the configuration was selected on V1 and V2, which lie after the early "
        "origins. Descriptive only; the origins overlap and share training data, so no interval "
        "or test is given. The band count is the origins whose pooled transfer rate lies "
        "strictly inside (0.016, 0.024). Days without an alert are left out of the rate and "
        "volume components (DR-003) and counted.",
    ),
)


def _matches(view: LoadedRun, kind: str, **fields: Any) -> list[Record]:
    return [
        record
        for record in view.of_kind(kind)
        if all(record.get(key) == value for key, value in fields.items())
    ]


def _one(view: LoadedRun, kind: str, **fields: Any) -> Record | None:
    found = _matches(view, kind, **fields)
    if len(found) > 1:
        raise RecordContractError(f"{len(found)} {kind} records match {fields}; one is expected")
    return found[0] if found else None


def _configurations(view: LoadedRun) -> list[str]:
    names = view.of_kind("run")[0]["configurations"]
    return sorted((n for n in names if n in CONFIGURATION_NAMES), key=ALL_CONFIGURATIONS.index)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _threshold(view: LoadedRun, name: str, policy: str) -> Record | None:
    return _one(
        view,
        "evaluation",
        configuration=name,
        policy=policy,
        partition="V2",
        period="primary",
        model_role="selected",
        capacity=PRIMARY_CAPACITY,
    )


def _metric(record: Record | None, name: str) -> object:
    return None if record is None else record["metrics"].get(name)


def _count(record: Record | None, name: str) -> object:
    return None if record is None else record["counts"][name]


def _row_label(name: str) -> list[str]:
    return [code(name), code(family_label(name))]


def threshold_outcomes(view: LoadedRun) -> list[str]:
    header = ["Configuration", "Family"]
    for policy in POLICIES:
        header += [f"{policy} alerts", f"{policy} recall", f"{policy} precision", f"{policy} F1"]
    rows = []
    for name in _configurations(view):
        cells = _row_label(name)
        for policy in POLICIES:
            record = _threshold(view, name, policy)
            cells += [
                cell(_count(record, "selected")),
                cell(_metric(record, "recall")),
                cell(_metric(record, "precision")),
                cell(_metric(record, "f1")),
            ]
        rows.append(cells)
    return _table(header, rows)


def threshold_workload(view: LoadedRun) -> list[str]:
    header = ["Configuration", "Family"]
    for policy in POLICIES:
        header += [
            f"{policy} overflow",
            f"{policy} unused",
            f"{policy} mean load",
            f"{policy} days above",
            f"{policy} days below",
        ]
    rows = []
    for name in _configurations(view):
        cells = _row_label(name)
        for policy in POLICIES:
            record = _threshold(view, name, policy)
            cells += [
                cell(_metric(record, "total_overflow")),
                cell(_metric(record, "total_unused")),
                cell(_metric(record, "mean_load_ratio")),
                cell(_metric(record, "days_above_budget")),
                cell(_metric(record, "days_below_budget")),
            ]
        rows.append(cells)
    return _table(header, rows)


def rate_volume_components(record: Record) -> dict[str, float | int | None]:
    """The means of the rate component, the volume component and ln(n_d / K) over the days of
    ``record`` (an R2 evaluation) with an alert, and the number of days without one."""
    daily, k, capacity = record["daily"], record["k"], record["capacity"]
    mean_volume = record["counts"]["transactions"] / record["counts"]["days"]
    rates, volumes, totals, silent = [], [], [], 0
    for day in daily:
        if day["selected"] == 0:
            silent += 1
            continue
        rates.append(math.log(day["selected"] / day["transactions"] / capacity))
        volumes.append(math.log(day["transactions"] / mean_volume))
        totals.append(math.log(day["selected"] / k))

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "rate": mean(rates),
        "volume": mean(volumes),
        "total": mean(totals),
        "zero_alert_days": silent,
    }


def rate_volume(view: LoadedRun) -> list[str]:
    header = [
        "Configuration",
        "Family",
        "R2 alert rate",
        "Days without alert",
        "Mean rate component",
        "Mean volume component",
        "Mean ln(n_d / K)",
    ]
    rows = []
    for name in _configurations(view):
        record = _threshold(view, name, "R2")
        parts = None if record is None else rate_volume_components(record)
        realised = (
            None
            if record is None
            else record["counts"]["selected"] / record["counts"]["transactions"]
        )
        rows.append(
            [
                *_row_label(name),
                cell(realised),
                cell(None if parts is None else parts["zero_alert_days"]),
                cell(None if parts is None else parts["rate"]),
                cell(None if parts is None else parts["volume"]),
                cell(None if parts is None else parts["total"]),
            ]
        )
    return _table(header, rows)


def _diagnostic(record: Record | None, rows_used: str) -> Record | None:
    if record is None:
        return None
    found = [d for d in record["diagnostics"] if d["rows_used"] == rows_used]
    return found[0] if found else None


def calibration(view: LoadedRun) -> list[str]:
    header = [
        "Configuration",
        "Family",
        "Platt κ₀",
        "Platt κ₁",
        "Platt Brier",
        "Platt log loss",
        "Platt intercept",
        "Platt slope",
        "Isotonic knots",
        "Isotonic Brier",
        "Isotonic log loss",
        "Isotonic intercept",
        "Isotonic slope",
        "Latency Platt held-out Brier",
        "Latency Platt held-out slope",
    ]
    rows = []
    for name in _configurations(view):
        platt = _one(view, "calibration", configuration=name, method="platt", period="primary")
        isotonic = _one(
            view, "calibration", configuration=name, method="isotonic", period="primary"
        )
        latency = _one(view, "calibration", configuration=name, method="platt", period="latency")
        fit_platt, fit_iso = _diagnostic(platt, "fitting"), _diagnostic(isotonic, "fitting")
        held = _diagnostic(latency, "held_out")
        parameters = None if platt is None else platt["parameters"]

        def value(source: Record | None, key: str) -> object:
            return None if source is None else source[key]

        rows.append(
            [
                *_row_label(name),
                cell(None if parameters is None else parameters["intercept"]),
                cell(None if parameters is None else parameters["slope"]),
                cell(value(fit_platt, "brier")),
                cell(value(fit_platt, "log_loss")),
                cell(value(fit_platt, "intercept")),
                cell(value(fit_platt, "slope")),
                cell(value(isotonic, "knots")),
                cell(value(fit_iso, "brier")),
                cell(value(fit_iso, "log_loss")),
                cell(value(fit_iso, "intercept")),
                cell(value(fit_iso, "slope")),
                cell(value(held, "brier")),
                cell(value(held, "slope")),
            ]
        )
    return _table(header, rows)


def h3_selection(view: LoadedRun) -> list[str]:
    lines: list[str] = []
    for period, title in (
        ("primary", "Binding selection (all of V2)"),
        ("latency", "Label-latency check"),
    ):
        record = _one(view, "h3_selection", period=period)
        lines += ["", f"**{title}**", ""]
        if record is None:
            lines.append("No record.")
            continue
        days = record["selection_days"]
        steps = ", ".join(record["tie_break_steps"]) or "none"
        lines += [
            f"Days {days['first']}–{days['last']}, K = {record['k']}; selected "
            f"{code(record['selected'])}; tie-break steps: {steps}; rule version "
            f"{record['rule_version']}.",
            "",
        ]
        header = [
            "Configuration",
            "Family",
            "Frauds captured",
            "V2 frauds",
            "Recall",
            "Average Precision",
            "Selected",
        ]
        rows = [
            [
                *_row_label(c["configuration"]),
                cell(c["frauds_captured"]),
                cell(c["frauds"]),
                cell(c["recall"]),
                cell(c["average_precision"]),
                "yes" if c["configuration"] == record["selected"] else "",
            ]
            for c in record["candidates"]
        ]
        lines += _table(header, rows)
    return lines


def rolling_origin(view: LoadedRun) -> list[str]:
    records = sorted(view.of_kind("rolling_origin"), key=lambda r: r["origin"])
    header = [
        "Origin",
        "Training days",
        "Training rows",
        "Window days",
        "Transfer days",
        "R2 threshold",
        "K",
        "Window rate",
        "Transfer rate",
        "r / c − 1",
        "Days above K",
        "Days below K",
        "Overflow",
        "Unused",
        "Zero-alert days",
        "Rate component",
        "Volume component",
    ]
    rows = []
    for r in records:
        t, w, g = r["transfer"], r["window"], r["training"]
        rows.append(
            [
                cell(r["origin"]),
                f"{g['first_day']}–{g['last_day']}",
                cell(g["rows"]),
                f"{w['first_day']}–{w['last_day']}",
                f"{t['first_day']}–{t['last_day']}",
                cell(r["threshold"]["value"]),
                cell(r["threshold"]["k"]),
                cell(w["rate"]),
                cell(t["rate"]),
                cell(t["relative_deviation"]),
                cell(t["days_above_budget"]),
                cell(t["days_below_budget"]),
                cell(t["total_overflow"]),
                cell(t["total_unused"]),
                cell(t["zero_alert_days"]),
                cell(t["rate_component"]),
                cell(t["volume_component"]),
            ]
        )
    inside = sum(r["transfer"]["inside_band"] for r in records)
    lines = [f"Configuration {code(records[0]['configuration'])}.", ""] if records else []
    lines += _table(header, rows)
    lines += ["", f"Origins inside the band: {inside} of {len(records)}."]
    return lines


_BUILDERS = {
    "threshold_outcomes": threshold_outcomes,
    "threshold_workload": threshold_workload,
    "rate_volume": rate_volume,
    "calibration": calibration,
    "h3_selection": h3_selection,
    "rolling_origin": rolling_origin,
}

_KINDS = {
    "threshold_outcomes": ("evaluation",),
    "threshold_workload": ("evaluation",),
    "rate_volume": ("evaluation",),
    "calibration": ("calibration",),
    "h3_selection": ("h3_selection",),
    "rolling_origin": ("rolling_origin",),
}


def analysis_markdown(view: LoadedRun) -> list[str]:
    """The sections of the analysis summary that the records of ``view`` can fill: a section is
    written when at least one record of its kind with the analysis's policies or methods exists."""
    lines: list[str] = []
    present = {kind for _, record in view.records for kind in [record["record"]]}
    has_threshold = any(
        r["policy"] in POLICIES for r in view.of_kind("evaluation") if r["model_role"] == "selected"
    )
    for section in SECTIONS:
        if not set(_KINDS[section.name]) <= present:
            continue
        if section.name.startswith(("threshold", "rate")) and not has_threshold:
            continue
        lines += ["", f"### {section.title}", "", section.meaning, ""]
        built = _BUILDERS[section.name](view)
        lines += built
    return lines


def analysis_reference() -> list[str]:
    """The sections as the contract reference lists them."""
    lines = ["", "## Sections of the analysis summary"]
    for section in SECTIONS:
        lines += ["", f"### {section.title}", "", section.meaning]
    return lines


__all__ = [
    "SECTIONS",
    "AnalysisSection",
    "analysis_markdown",
    "analysis_reference",
    "rate_volume_components",
]
