"""The Markdown summary of a closed confirmatory run (Issue #209; DR-006, DR-007, DR-015 §9).

``render_summary`` is a pure function of the run's JSON records: the same records give the same
bytes, and every number in a table is the exact JSON literal of a record field (a compact code
span, as in the development summary). A table only selects and arranges record values; nothing
new is computed. The sections are, in order: the run and its attempts; the primary confirmatory
results (H1, H2, H3, the Holm decision); the sensitivities (S1, S2, the SESOI views, McNemar);
the secondary outcomes of every configuration and the reference; RQ2; the calibration
diagnostics; RQ3; the false-negative slices (the H3 configuration first); the Test scoring times;
and the fields of every record.

The summary reports values, intervals and the pre-declared labels (the Holm decisions, the H1
category, the interval positions, the regime of a cell, the break-even classification). It
assigns no "material miscalibration" label and makes no causal statement; the cautions of the
protocol are printed beside the tables they apply to.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from review_capacity.confirmatory.records import contract_of
from review_capacity.development.records import ALL_CONFIGURATIONS
from review_capacity.development.run_output import LoadedRun
from review_capacity.development.schema import ListOf, MapOf, Object
from review_capacity.development.summary import cell, code

type Record = Mapping[str, Any]

BOUND = 12  # a list longer than this is shown by its length in the record sections
RATIOS = (2, 5, 10, 20, 50, 100, 200, 500, 1000)
CAUTIONS = {
    "slices": (
        "Slice base rates differ strongly, so raw false-negative counts are never interpreted "
        "alone; a slice with fewer than the minimum number of frauds has no interval; the score "
        "band is defined on the missed frauds only and has no miss rate. The analysis is "
        "exploratory and non-causal."
    ),
    "rq3": (
        "The RQ3 intervals are conditional: the fitted models and the frozen calibrators stay "
        "fixed and whole Test days are resampled (day-level moving block, B replicates). They "
        "describe Test-period variation for the fitted policy and exclude model-development and "
        "calibrator-estimation uncertainty. Costs are scenario assumptions, never institutional "
        "costs; amounts are exposure proxies."
    ),
    "calibration": (
        "Intercept and slope are the diagnostics of logistic recalibration on the Test rows; "
        "Brier score and clipped log loss are reported as recorded. No threshold for a material "
        "miscalibration is applied here."
    ),
    "timing": (
        "The scoring seconds are the end-to-end boundary of the single scoring pass "
        "(DR-015 section 9). The tree-based and the neural timings are not a same-device "
        "comparison: each is taken on its device, and a difference reflects model, software, "
        "input representation, batching and hardware together."
    ),
}


def table(header: Sequence[str], body: Sequence[Sequence[str]]) -> list[str]:
    """A Markdown table; every cell is already formatted."""
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in body),
    ]


def span(interval: Record | None) -> str:
    """An interval as the pair of its exact endpoints."""
    return "—" if interval is None else cell([interval["lower"], interval["upper"]])


def _by(records: Sequence[Record], kind: str) -> list[Record]:
    return [r for r in records if r["record"] == kind]


def _configuration_order(name: str) -> int:
    return ALL_CONFIGURATIONS.index(name)


def _run_section(run: LoadedRun, records: Sequence[Record]) -> list[str]:
    start = _by(records, "test_run")[0]
    identity = start["identity"]
    rows = [
        ("Run identifier", identity["run_id"]),
        ("Freeze tag", identity["freeze"]["tag"]),
        ("Tag commit", identity["freeze"]["tag_commit"]),
        ("Evaluator commit", identity["evaluator"]["code_commit"]),
        ("Evaluator source SHA-256", identity["evaluator"]["source_sha256"]),
        ("Training run", identity["training"]["run_id"]),
        ("Training commit", identity["training"]["code_commit"]),
        ("Frozen-state identity", identity["frozen_state_sha256"]),
        ("Dataset manifest SHA-256", identity["dataset_manifest_sha256"]),
        ("Seeds", identity["seeds"]),
        ("Started", start["started"]),
        ("Test partition", start["test"]),
        ("H3 configuration", start["frozen_inputs"]["h3_configuration"]),
    ]
    phases = sorted(_by(records, "test_phase"), key=lambda r: (r["phase"], r["attempt"]))
    lines = [
        "## Run",
        "",
        *table(["Item", "Value"], [[name, cell(value)] for name, value in rows]),
        "",
        "### Attempts",
        "",
        *table(
            [
                "Phase",
                "Attempt",
                "Outcome",
                "Reached the consumption marker",
                "Started",
                "Finished",
            ],
            [
                [
                    cell(p["phase"]),
                    cell(p["attempt"]),
                    cell(p["outcome"]),
                    cell(p["labels_accessed"]),
                    cell(p["started"]),
                    cell(p["finished"]),
                ]
                for p in phases
            ],
        ),
    ]
    errors = [p for p in phases if p["error"] is not None]
    if errors:
        lines += ["", "Errors of the attempts that did not complete:", ""]
        lines += [
            f"- {cell(p['phase'])} attempt {p['attempt']}: {cell(p['error'])}" for p in errors
        ]
    if run.end is not None:
        lines += ["", f"Closed: {cell(run.end['outcome'])} at {cell(run.end['finished'])}."]
    return lines


def _primary(inference: Record) -> list[str]:
    h1, h2, h3, holm = inference["h1"], inference["h2"], inference["h3"], inference["holm"]
    adjusted, rejected = holm["adjusted"], holm["rejected"]
    decisions = inference["decisions"]
    return [
        "## Primary confirmatory results",
        "",
        "Holm over H1, H2 and H3; the adjusted p-value decides, the interval positions only "
        "describe.",
        "",
        *table(
            [
                "Hypothesis",
                "Estimate",
                "p",
                "Holm-adjusted p",
                "Rejected",
                "90 % interval",
                "Position",
            ],
            [
                [
                    "H1 family effect",
                    cell(h1["estimate"]),
                    cell(h1["p"]["value"]),
                    cell(adjusted[0]),
                    cell(rejected[0]),
                    span(h1["interval"]),
                    cell(h1["position"]),
                ],
                [
                    "H2 weighting equivalence",
                    "—",
                    cell(h2["p"]["value"]),
                    cell(adjusted[1]),
                    cell(rejected[1]),
                    "—",
                    "—",
                ],
                [
                    f"H3 alert-rate transfer ({h3['configuration']})",
                    cell(h3["rate"]),
                    cell(h3["primary"]["p"]["value"]),
                    cell(adjusted[2]),
                    cell(rejected[2]),
                    span(h3["primary"]["interval"]),
                    cell(h3["primary"]["position"]),
                ],
            ],
        ),
        "",
        "Decisions: " + ", ".join(f"{code(k)} = {cell(v)}" for k, v in decisions.items()) + ".",
        "",
        "### H2 by model",
        "",
        *table(
            [
                "Model",
                "Estimate",
                "p lower",
                "p upper",
                "90 % interval",
                "Position",
                "95 % interval",
            ],
            [
                [
                    cell(m["model"]),
                    cell(m["estimate"]),
                    cell(m["p_lower"]["value"]),
                    cell(m["p_upper"]["value"]),
                    span(m["interval"]),
                    cell(m["position"]),
                    span(m["difference_interval"]),
                ]
                for m in h2["models"]
            ],
        ),
        "",
        "### H3 variants and tolerance views",
        "",
        *table(
            ["Block length", "p lower", "p upper", "p", "90 % interval", "Position"],
            [
                [
                    cell(v["block_length"]),
                    cell(v["p_lower"]["value"]),
                    cell(v["p_upper"]["value"]),
                    cell(v["p"]["value"]),
                    span(v["interval"]),
                    cell(v["position"]),
                ]
                for v in [h3["primary"], *h3["sensitivity"]]
            ],
        ),
        "",
        *table(
            ["Tolerance", "Lower bound", "Upper bound", "Position"],
            [
                [cell(t["tolerance"]), cell(t["lower"]), cell(t["upper"]), cell(t["position"])]
                for t in h3["tolerances"]
            ],
        ),
        "",
        f"Infimum of the tolerances whose open band contains the interval (t_min; containment "
        f"holds for every larger tolerance): {cell(h3['smallest_tolerance'])}.",
        "",
        "H3 of the other configurations is descriptive and outside the Holm family:",
        "",
        *table(
            ["Configuration", "Alert rate", "90 % interval", "Position"],
            [
                [
                    cell(o["configuration"]),
                    cell(o["rate"]),
                    span(o["interval"]),
                    cell(o["position"]),
                ]
                for o in h3["others"]
            ],
        ),
        "",
        "### Descriptive estimands (95 % intervals)",
        "",
        *table(
            ["Estimand", "Estimate", "95 % interval"],
            [
                [cell(e["name"]), cell(e["estimate"]), span(e["interval"])]
                for group in ("within_family", "by_weighting", "cells")
                for e in h1[group]
            ]
            + [
                [
                    cell(h1["interaction"]["name"]),
                    cell(h1["interaction"]["estimate"]),
                    span(h1["interaction"]["interval"]),
                ]
            ],
        ),
        "",
        "### Effect units",
        "",
        *table(
            [
                "Estimand",
                "Percentage points",
                "Frauds per partition",
                "Frauds per day",
                "Precision@K equivalent (points)",
                "Normalised recall",
            ],
            [
                [
                    cell(u["estimand"]),
                    cell(u["percentage_points"]),
                    cell(u["frauds_per_partition"]),
                    cell(u["frauds_per_day"]),
                    cell(u["precision_at_k_points"]),
                    cell(u["normalised_recall"]),
                ]
                for u in inference["effect_units"]
            ],
        ),
        "",
        "Basis: " + cell(inference["effect_basis"]) + ".",
    ]


def _sensitivities(inference: Record) -> list[str]:
    s1, s2 = inference["s1"], inference["s2"]
    lines = [
        "## Sensitivities",
        "",
        "### S1 (Owen-corrected Wald)",
        "",
        *table(
            [
                "Quantity",
                "Estimate",
                "Pigeonhole variance",
                "Fraud-level variance",
                "Standard error",
            ],
            [
                [
                    cell(q["name"]),
                    cell(q["estimate"]),
                    cell(q["v_pigeonhole"]),
                    cell(q["v_naive"]),
                    cell(q["se"]),
                ]
                for q in s1["quantities"]
            ],
        ),
        "",
        f"p(H1) = {cell(s1['p_h1'])}; 90 % interval {span(s1['interval_h1'])}; position "
        f"{cell(s1['position_h1'])}; p(H2) = {cell(s1['p_h2'])}.",
    ]
    if s1["intervals_h2"] is not None:
        lines += [
            "",
            *table(
                ["Model", "S1 90 % interval of the weighting effect"],
                [
                    [cell(m["model"]), span(i)]
                    for m, i in zip(inference["h2"]["models"], s1["intervals_h2"], strict=True)
                ],
            ),
        ]
    if s1["reasons"]:
        lines += ["", "Not estimable: " + "; ".join(cell(r) for r in s1["reasons"]) + "."]
    comparison = s1["comparison"]
    if comparison is not None:
        lines += ["", "Comparison with the primary: " + cell(comparison) + "."]
    lines += [
        "",
        "### S2 (day-level moving block)",
        "",
        f"p(H1) = {cell(s2['p_h1']['value'])}; p(H2) = {cell(s2['p_h2']['value'])}; H1 90 % "
        f"interval {span(s2['h1_interval'])}; position {cell(s2['h1_position'])}.",
        "",
        "Comparison with the primary: " + cell(s2["comparison"]) + ".",
        "",
        "### SESOI views (not confirmatory)",
        "",
        *table(
            ["SESOI", "H1 position", "H1 category", "H2 positions"],
            [
                [
                    cell(v["sesoi"]),
                    cell(v["h1_position"]),
                    cell(v["h1_category"]),
                    cell(v["h2_positions"]),
                ]
                for v in inference["sesoi_views"]
            ],
        ),
        "",
        "### Exact McNemar tests (descriptive, outside the Holm family)",
        "",
        *table(
            ["First", "Second", "Only first", "Only second", "p"],
            [
                [
                    cell(m["first"]),
                    cell(m["second"]),
                    cell(m["only_first"]),
                    cell(m["only_second"]),
                    cell(m["p"]),
                ]
                for m in inference["mcnemar"]
            ],
        ),
    ]
    return lines


def _metric(entry: Record, name: str) -> str:
    value = entry["metrics"].get(name)
    return "—" if value is None else cell(value)


def _with_interval(entry: Record, name: str) -> list[str]:
    return [_metric(entry, name), span(entry["intervals"].get(name))]


def _secondary(evaluations: Sequence[Record]) -> list[str]:
    ordered = sorted(evaluations, key=lambda r: _configuration_order(r["configuration"]))
    capacities = [e["capacity"] for e in ordered[0]["r3"]]
    lines = ["## Secondary outcomes", "", "95 % percentile bootstrap intervals (DR-006).", ""]
    for index, capacity in enumerate(capacities):
        lines += [
            f"### R3 at capacity {cell(capacity)} (K = {cell(ordered[0]['r3'][index]['k'])})",
            "",
        ]
        header = [
            "Configuration",
            "Recall",
            "Recall interval",
            "Precision",
            "Precision interval",
            "Lift",
            "Lift interval",
            "Normalised recall",
            "Normalised interval",
            "Amount-weighted recall",
            "Amount interval",
            "False negatives",
            "FN interval",
        ]
        body = []
        for record in ordered:
            entry = record["r3"][index]
            body.append(
                [
                    cell(record["configuration"]),
                    *_with_interval(entry, "recall"),
                    *_with_interval(entry, "precision"),
                    *_with_interval(entry, "lift"),
                    *_with_interval(entry, "normalised_recall"),
                    *_with_interval(entry, "amount_weighted_recall"),
                    cell(entry["counts"]["false_negatives"]),
                    span(entry["intervals"]["false_negatives"]),
                ]
            )
        lines += [*table(header, body), ""]
    lines += [
        "The per-day tables, the cutoff ties and the remaining counts of every policy are in the "
        "records (section Records).",
        "",
        "### Ranking measures (day-level moving block)",
        "",
    ]
    lines += table(
        ["Configuration", "Average Precision", "Interval", "ROC-AUC", "Interval"],
        [
            [
                cell(r["configuration"]),
                _metric(r["ranking"], "average_precision"),
                span(r["ranking"]["intervals"]["average_precision"]),
                _metric(r["ranking"], "roc_auc"),
                span(r["ranking"]["intervals"]["roc_auc"]),
            ]
            for r in ordered
        ],
    )
    thresholds = [r for r in ordered if r["policies"]]
    for policy in ("R0", "R1", "R2"):
        lines += ["", f"### {policy} (uncapped alerts) against each budget", ""]
        header = [
            "Configuration",
            "Capacity",
            "Threshold",
            "Recall",
            "Recall interval",
            "Precision",
            "Precision interval",
            "F1",
            "F1 interval",
            "Alerts",
            "Total overflow",
            "Total unused",
            "Mean load ratio",
            "Zero-alert days",
        ]
        body = []
        for record in thresholds:
            for entry in (p for p in record["policies"] if p["policy"] == policy):
                body.append(
                    [
                        cell(record["configuration"]),
                        cell(entry["capacity"]),
                        cell(entry["threshold"]),
                        *_with_interval(entry, "recall"),
                        *_with_interval(entry, "precision"),
                        *_with_interval(entry, "f1"),
                        cell(entry["counts"]["selected"]),
                        _metric(entry, "total_overflow"),
                        _metric(entry, "total_unused"),
                        _metric(entry, "mean_load_ratio"),
                        _metric(entry, "zero_alert_days"),
                    ]
                )
        lines += table(header, body)
    return lines


def _rq2(workloads: Sequence[Record]) -> list[str]:
    ordered = sorted(workloads, key=lambda r: _configuration_order(r["configuration"]))
    lines = [
        "## RQ2: daily-budget fidelity of R2 (point values)",
        "",
        "The decomposition is against the V2 reference volume; the budget term is the rounding "
        "of K.",
        "",
    ]
    for index, entry in enumerate(ordered[0]["entries"]):
        lines += [f"### Capacity {cell(entry['capacity'])} (K = {cell(entry['k'])})", ""]
        header = [
            "Configuration",
            "Threshold",
            "Mean alerts",
            "Mean deviation",
            "Mean absolute deviation",
            "Days above",
            "Days below",
            "Total overflow",
            "Total unused",
            "Mean load ratio",
            "Zero-alert days",
            "Mean rate",
            "Mean volume",
            "Mean budget",
            "Mean total",
        ]
        body = []
        for record in ordered:
            item = record["entries"][index]
            f = item["fidelity"]
            body.append(
                [
                    cell(record["configuration"]),
                    cell(item["threshold"]),
                    cell(f["mean_alerts"]),
                    cell(f["mean_deviation"]),
                    cell(f["mean_absolute_deviation"]),
                    cell(f["days_above_budget"]),
                    cell(f["days_below_budget"]),
                    cell(f["total_overflow"]),
                    cell(f["total_unused"]),
                    cell(f["mean_load_ratio"]),
                    cell(f["zero_alert_days"]),
                    cell(f["mean_rate"]),
                    cell(f["mean_volume"]),
                    cell(f["mean_budget"]),
                    cell(f["mean_total"]),
                ]
            )
        lines += [*table(header, body), ""]
    return lines


def _calibration(calibrations: Sequence[Record]) -> list[str]:
    lines = ["## Calibration diagnostics on the Test rows", "", CAUTIONS["calibration"], ""]
    for method, role in (("platt", "primary"), ("isotonic", "sensitivity")):
        chosen = sorted(
            (r for r in calibrations if r["method"] == method),
            key=lambda r: _configuration_order(r["configuration"]),
        )
        lines += [f"### {method} ({role})", ""]
        body = []
        for r in chosen:
            d, i = r["diagnostic"], r["intervals"]
            body.append(
                [
                    cell(r["configuration"]),
                    cell(d["brier"]),
                    span(i["brier"]),
                    cell(d["log_loss"]),
                    span(i["log_loss"]),
                    cell(d["intercept"]),
                    span(i["intercept"]),
                    cell(d["slope"]),
                    span(i["slope"]),
                    cell(d["metric_clipped"]),
                    cell(d["out_of_range"]),
                ]
            )
        lines += [
            *table(
                [
                    "Configuration",
                    "Brier",
                    "Interval",
                    "Log loss",
                    "Interval",
                    "Intercept",
                    "Interval",
                    "Slope",
                    "Interval",
                    "Clipped",
                    "Out of range",
                ],
                body,
            ),
            "",
        ]
        for r in chosen:
            lines += [
                f"#### Reliability table: {r['configuration']} ({method})",
                "",
                *table(
                    [
                        "Bin",
                        "Min probability",
                        "Max probability",
                        "Mean probability",
                        "Transactions",
                        "Frauds",
                        "Observed rate",
                    ],
                    [
                        [
                            cell(b["bin"]),
                            cell(b["min_probability"]),
                            cell(b["max_probability"]),
                            cell(b["mean_probability"]),
                            cell(b["transactions"]),
                            cell(b["frauds"]),
                            cell(b["observed_rate"]),
                        ]
                        for b in r["diagnostic"]["reliability"]
                    ],
                ),
                "",
            ]
    return lines


def _rq3(analyses: Sequence[Record]) -> list[str]:
    ordered = sorted(
        analyses, key=lambda r: (r["calibrator"], _configuration_order(r["configuration"]))
    )
    lines = ["## RQ3: R4 against R3", "", CAUTIONS["rq3"], ""]
    for method in ("platt", "isotonic"):
        chosen = [r for r in ordered if r["calibrator"] == method]
        if not chosen:
            continue
        for index, capacity in enumerate(c["capacity"] for c in chosen[0]["capacities"]):
            k = chosen[0]["capacities"][index]["k"]
            lines += [f"### {method}, capacity {cell(capacity)} (K = {cell(k)})", ""]
            lines += ["Break-even and saturation ratios:", ""]
            body = []
            for r in chosen:
                e = r["capacities"][index]
                body.append(
                    [
                        cell(r["configuration"]),
                        cell(e["break_even"]["classification"]),
                        cell(e["break_even"]["ratio"]),
                        cell(e["saturation"]["ratio"]),
                        cell(e["break_even"]["single_interval"]),
                        cell([f["ratio"] for f in e["friction"]]),
                        cell(e["queue"]),
                        cell(len(e["curve"])),
                    ]
                )
            lines += [
                *table(
                    [
                        "Configuration",
                        "Classification",
                        "Break-even ratio",
                        "Saturation ratio",
                        "Single interval",
                        "C_FP conversions",
                        "R3 queue (reviews, frauds, captured)",
                        "Curve segments (all in the record)",
                    ],
                    body,
                ),
                "",
            ]
            lines += ["Regime map (ratio grid in columns):", ""]
            body = [
                [
                    cell(r["configuration"]),
                    *(cell(row["regime"]) for row in r["capacities"][index]["grid"]),
                ]
                for r in chosen
            ]
            lines += [*table(["Configuration", *(f"rho = {x}" for x in RATIOS)], body), ""]
            lines += [
                "Grid points (95 % day-block intervals; the frauds forgone also with the "
                "pigeonhole interval):",
                "",
            ]
            body = []
            for r in chosen:
                for row in r["capacities"][index]["grid"]:
                    i = row["intervals"]
                    body.append(
                        [
                            cell(r["configuration"]),
                            cell(row["ratio"]),
                            cell(row["unused"]),
                            span(i["unused"]),
                            cell(row["frauds_forgone"]),
                            span(i["frauds_forgone"]),
                            span(row["frauds_forgone_pigeonhole"]),
                            cell(row["cost_difference"]),
                            span(i["cost_difference"]),
                            cell(row["relative_difference"]),
                            span(i["relative_difference"]),
                            cell(row["regime"]),
                        ]
                    )
            lines += [
                *table(
                    [
                        "Configuration",
                        "rho",
                        "Unused",
                        "Interval",
                        "Frauds forgone",
                        "Interval",
                        "Pigeonhole interval",
                        "Delta J",
                        "Interval",
                        "Relative Delta J",
                        "Interval",
                        "Regime",
                    ],
                    body,
                ),
                "",
            ]
            lines += [
                "Costs, utilisation and the fixed-policy break-even:",
                "",
                *table(
                    [
                        "Configuration",
                        "rho",
                        "p*",
                        "R4 reviews",
                        "Utilisation",
                        "Interval",
                        "Days with unused",
                        "J R3",
                        "Interval",
                        "J R4",
                        "Interval",
                        "J no review",
                        "Fixed-policy status",
                        "Fixed-policy ratio",
                    ],
                    [
                        [
                            cell(r["configuration"]),
                            cell(row["ratio"]),
                            cell(row["threshold"]),
                            cell(row["reviews"]),
                            cell(row["utilisation"]),
                            span(row["intervals"]["utilisation"]),
                            cell(row["days_with_unused"]),
                            cell(row["cost_r3"]),
                            span(row["intervals"]["cost_r3"]),
                            cell(row["cost_r4"]),
                            span(row["intervals"]["cost_r4"]),
                            cell(row["cost_no_review"]),
                            cell(row["fixed_status"]),
                            cell(row["fixed_ratio"]),
                        ]
                        for r in chosen
                        for row in r["capacities"][index]["grid"]
                    ],
                ),
                "",
                "Declined-set diagnostic (fraud rate of the unused reviews next to their mean "
                "calibrated probability and p*) and the amount measures (exposure proxies):",
                "",
                *table(
                    [
                        "Configuration",
                        "rho",
                        "p*",
                        "Declined fraud rate",
                        "Interval",
                        "Declined mean probability",
                        "Interval",
                        "Exposed R3",
                        "Exposed R4",
                        "Interval",
                        "Forgone",
                        "Interval",
                        "Forgone per unused",
                        "Interval",
                        "Amount-weighted recall R3",
                        "Amount-weighted recall R4",
                        "Interval",
                    ],
                    [
                        [
                            cell(r["configuration"]),
                            cell(row["ratio"]),
                            cell(row["threshold"]),
                            cell(row["declined_fraud_rate"]),
                            span(row["intervals"]["declined_fraud_rate"]),
                            cell(row["declined_mean_probability"]),
                            span(row["intervals"]["declined_mean_probability"]),
                            cell(row["fraud_value_exposed_r3"]),
                            cell(row["fraud_value_exposed_r4"]),
                            span(row["intervals"]["fraud_value_exposed_r4"]),
                            cell(row["fraud_value_forgone"]),
                            span(row["intervals"]["fraud_value_forgone"]),
                            cell(row["forgone_per_unused"]),
                            span(row["intervals"]["forgone_per_unused"]),
                            cell(row["amount_weighted_recall_r3"]),
                            cell(row["amount_weighted_recall_r4"]),
                            span(row["intervals"]["amount_weighted_recall_r4"]),
                        ]
                        for r in chosen
                        for row in r["capacities"][index]["grid"]
                    ],
                ),
                "",
            ]
    return lines


def _false_negatives(tables: Sequence[Record], h3_configuration: str) -> list[str]:
    ordered = sorted(
        tables,
        key=lambda r: (
            r["configuration"] != h3_configuration,
            _configuration_order(r["configuration"]),
        ),
    )
    first = ordered[0]
    lines = [
        "## False-negative slices (R3 at the primary capacity)",
        "",
        CAUTIONS["slices"],
        "",
        "Slice boundaries (fixed without Test labels): " + cell(first["boundaries"]) + ".",
    ]
    for record in ordered:
        lines += [
            "",
            f"### {record['configuration']}"
            + (" (the H3 configuration)" if record["configuration"] == h3_configuration else ""),
            "",
            f"Frauds {cell(record['frauds'])}, false negatives {cell(record['false_negatives'])} "
            f"(capacity {cell(record['capacity'])}, K = {cell(record['k'])}).",
            "",
            *table(
                [
                    "Variable",
                    "Level",
                    "Frauds",
                    "Misses",
                    "Miss rate",
                    "Share of false negatives",
                    "Status",
                    "95 % interval",
                ],
                [
                    [
                        cell(s["variable"]),
                        cell(s["level"]),
                        cell(s["frauds"]),
                        cell(s["misses"]),
                        cell(s["miss_rate"]),
                        cell(s["share"]),
                        cell(s["status"]),
                        span(s["interval"]),
                    ]
                    for s in record["slices"]
                ],
            ),
        ]
    return lines


def _timing(scores: Sequence[Record]) -> list[str]:
    ordered = sorted(scores, key=lambda r: _configuration_order(r["configuration"]))
    return [
        "## Test scoring time",
        "",
        CAUTIONS["timing"],
        "",
        *table(
            ["Configuration", "Device", "Rows", "Seconds", "Rows per second", "Distinct scores"],
            [
                [
                    cell(r["configuration"]),
                    cell(r["device"]),
                    cell(r["rows"]),
                    cell(r["timing"]["seconds"]),
                    cell(r["timing"]["rows_per_second"]),
                    cell(r["distinct_scores"]),
                ]
                for r in ordered
            ],
        ),
    ]


def _long(kind: ListOf, items: list[Any]) -> bool:
    """A list that the record sections show by its length: long, or a list of objects."""
    return len(items) > BOUND or (isinstance(kind.item, Object) and bool(items))


SYMBOLS: tuple[tuple[str, str], ...] = (
    ("K", "fixed daily review budget (reviews per relative day) of a capacity scenario"),
    ("rho", "miss-to-review cost ratio of the scenario cost J = reviews + rho x missed frauds"),
    ("p*", "cost-derived probability threshold 1 / rho of R4"),
    (
        "J R3, J R4",
        "scenario cost of R3 (daily top-K review) and of R4 (the cost-aware capped policy)",
    ),
    ("Delta J", "J R4 - J R3 = rho f - u"),
    ("u, f", "unused reviews (R3 queue members R4 declines) and the frauds among them"),
    (
        "rho_sat, rho_BE",
        "saturation ratio (R4 fills the budget) and break-even ratio (DR-007 section 4)",
    ),
    ("C_FP", "extra cost of a false alert; its conversions are sensitivities of rho_BE"),
    (
        "R0 to R4",
        "decision policies: 0.5 threshold, F1-optimal threshold, capacity-rate threshold, "
        "daily top-K, calibrated cost-aware capped policy",
    ),
    (
        "H1, H2, H3",
        "the confirmatory hypotheses: family effect, weighting equivalence, alert-rate transfer",
    ),
    ("SESOI", "Smallest Effect Size Of Interest"),
    ("S1, S2", "sensitivity analyses: Owen-corrected Wald and day-level moving block"),
    ("B", "number of bootstrap replicates"),
    ("interval", "percentile bootstrap interval (95 % unless the level is stated)"),
    ("Brier, log loss", "mean squared error of the probability and epsilon-clipped log loss"),
    ("intercept, slope", "calibration intercept (logit offset) and slope (logistic recalibration)"),
)


def _symbols() -> list[str]:
    return [
        "## Symbols and notation",
        "",
        *table(["Symbol or term", "Meaning"], [[code(name), meaning] for name, meaning in SYMBOLS]),
    ]


def _entries(value: object) -> str:
    return (
        f"{len(value)} entries; all are in the record" if isinstance(value, list) else cell(value)
    )


def _flatten(obj: Object, value: Mapping[str, Any], prefix: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for f in obj.fields:
        path, item = f"{prefix}{f.name}", value[f.name]
        if isinstance(f.kind, Object) and item is not None:
            rows += _flatten(f.kind, item, f"{path}.")
        elif isinstance(f.kind, MapOf) and item:
            rows += [(f"{path}.{key}", _entries(entry)) for key, entry in item.items()]
        elif isinstance(f.kind, ListOf) and isinstance(item, list) and _long(f.kind, item):
            rows.append((path, _entries(item)))
        else:
            rows.append((path, cell(item)))
    return rows


def _record_section(name: str, record: Record) -> list[str]:
    """Every field of a record but its identity (shown once, in the run section), with a list
    of objects or a long list shown by its length; the record itself holds every entry."""
    body = Object(tuple(f for f in contract_of(record).body.fields if f.name != "identity"))
    rows = _flatten(body, record, "")
    return [
        "",
        f"### {code(name)}",
        "",
        *table(["Field", "Value"], [[code(p), v] for p, v in rows]),
    ]


def render_summary(run: LoadedRun) -> str:
    """The Markdown summary of the confirmatory run ``run``, from its records alone."""
    records: list[Record] = [r for _, r in run.records]
    inference = _by(records, "test_inference")
    h3 = _by(records, "test_run")[0]["frozen_inputs"]["h3_configuration"]
    lines = [
        f"# Confirmatory run {run.identity['run_id']}",
        "",
        "Generated from the JSON records of this run directory; not edited by hand. Every number "
        "is the exact literal of a record field.",
        "",
        *_run_section(run, records),
    ]
    if inference:
        lines += ["", *_primary(inference[0]), "", *_sensitivities(inference[0])]
    for builder, kind in (
        (_secondary, "test_evaluation"),
        (_rq2, "test_workload"),
        (_calibration, "test_calibration"),
        (_rq3, "test_rq3"),
    ):
        found = _by(records, kind)
        if found:
            lines += ["", *builder(found)]
    tables = _by(records, "test_false_negative")
    if tables:
        lines += ["", *_false_negatives(tables, h3)]
    scores = _by(records, "test_scores")
    if scores:
        lines += ["", *_timing(scores)]
    lines += ["", "## Records"]
    for name, record in [*run.records, *([("test_run_end.json", run.end)] if run.end else [])]:
        lines += _record_section(name, record)
    lines += ["", *_symbols()]
    return "\n".join(lines) + "\n"
