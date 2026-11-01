"""The reporting tables that the frozen records require beyond the plain record values (Issue #217).

``render_summary`` shows record values. DR-007 sections 4, 8 and 9 and DR-005 also require results
that are read across records, and this module builds them from the saved aggregates alone (no Test
row or label is read and no model is evaluated):

- the **calibration sensitivity** of RQ3 (DR-007 section 8): the break-even classification and the
  regime of every grid cell under the isotonic calibrator next to those under Platt (primary);
- the **non-monotone curves** (DR-007 section 4): the cheaper intervals and the zero crossings of
  every result whose cheaper set is not a single interval;
- the **per-day costs and the captured-fraud outcomes** of R3 and R4 at every grid ratio
  (DR-007 section 9), from the recorded queue, the unused reviews and the frauds forgone;
- the **converted intervals** of the effect units (DR-005): the linear conversions of an
  estimand's interval with the recorded basis (frauds, days, K, oracle ceiling).

A comparison or a linear conversion computes no new statistic: the cells are the record values, a
tally of a comparison, or a recorded value times a recorded constant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from review_capacity.development.records import ALL_CONFIGURATIONS
from review_capacity.development.summary import cell

type Record = Mapping[str, Any]

PRIMARY, SENSITIVITY = "platt", "isotonic"


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


def configuration_order(name: str) -> int:
    return ALL_CONFIGURATIONS.index(name)


def _by_configuration(analyses: Sequence[Record], method: str) -> dict[str, Record]:
    return {r["configuration"]: r for r in analyses if r["calibrator"] == method}


def calibration_sensitivity(analyses: Sequence[Record]) -> list[str]:
    """Where isotonic changes the break-even classification or the regime map of Platt."""
    primary = _by_configuration(analyses, PRIMARY)
    other = _by_configuration(analyses, SENSITIVITY)
    names = sorted((n for n in primary if n in other), key=configuration_order)
    if not names:
        return []
    body: list[list[str]] = []
    pairs = changed = cells = regimes = 0
    for name in names:
        for a, b in zip(primary[name]["capacities"], other[name]["capacities"], strict=True):
            moved = [
                (x["ratio"], x["regime"], y["regime"])
                for x, y in zip(a["grid"], b["grid"], strict=True)
                if x["regime"] != y["regime"]
            ]
            different = a["break_even"]["classification"] != b["break_even"]["classification"]
            pairs += 1
            changed += different
            cells += len(a["grid"])
            regimes += len(moved)
            body.append(
                [
                    cell(name),
                    cell(a["capacity"]),
                    cell(a["break_even"]["classification"]),
                    cell(a["break_even"]["ratio"]),
                    cell(b["break_even"]["classification"]),
                    cell(b["break_even"]["ratio"]),
                    cell(different),
                    cell(len(moved)),
                    ", ".join(f"{cell(r)}: {cell(p)} to {cell(i)}" for r, p, i in moved) or "—",
                ]
            )
    return [
        "### Calibration sensitivity (isotonic against Platt)",
        "",
        "Platt is primary; a change under isotonic in the break-even classification or in the "
        "regime of a grid cell is a calibration sensitivity of the RQ3 result (DR-007 section 8).",
        "",
        f"Classification changes: {changed} of {pairs} configuration-capacity pairs; regime "
        f"changes: {regimes} of {cells} grid cells.",
        "",
        *table(
            [
                "Configuration",
                "Capacity",
                "Platt classification",
                "Platt break-even ratio",
                "Isotonic classification",
                "Isotonic break-even ratio",
                "Classification changed",
                "Regime cells changed",
                "Changed cells (rho: Platt to isotonic)",
            ],
            body,
        ),
    ]


def non_monotone(analyses: Sequence[Record]) -> list[str]:
    """The cheaper intervals and the zero crossings of the results with a non-single interval."""
    ordered = sorted(
        analyses,
        key=lambda r: (r["calibrator"] != PRIMARY, configuration_order(r["configuration"])),
    )
    body: list[list[str]] = []
    total = 0
    for r in ordered:
        for e in r["capacities"]:
            total += 1
            if not e["break_even"]["single_interval"]:
                b = e["break_even"]
                body.append(
                    [
                        cell(r["calibrator"]),
                        cell(r["configuration"]),
                        cell(e["capacity"]),
                        cell(b["classification"]),
                        cell(b["ratio"]),
                        cell(b["intervals"]),
                        cell(b["zero_crossings"]),
                    ]
                )
    lines = [
        "### Non-monotone RQ3 curves",
        "",
        f"The set of ratios where R4 is cheaper than R3 is not a single interval in {len(body)} "
        f"of {total} results. The break-even ratio and the "
        "classification are the recorded values; R4 is cheaper on the listed intervals and on no "
        "other ratio, and the result is read from the full curve (DR-007 section 4).",
        "",
    ]
    if not body:
        return [*lines[:2], "Every result has a single cheaper interval."]
    return [
        *lines,
        *table(
            [
                "Calibrator",
                "Configuration",
                "Capacity",
                "Classification",
                "Break-even ratio",
                "Cheaper intervals",
                "Zero crossings",
            ],
            body,
        ),
    ]


def rq3_outcomes(
    chosen: Sequence[Record], index: int, ceilings: Mapping[tuple[str, int], float]
) -> list[str]:
    """Per-day costs and the captured-fraud outcomes of R3 and R4 at every grid ratio of one
    capacity (index ``index``); ``ceilings`` is the oracle ceiling of R3 by (configuration, index)
    """
    body: list[list[str]] = []
    for r in chosen:
        e = r["capacities"][index]
        q = e["queue"]
        days = q["reviews"] / e["k"]  # R3 reviews K a day on every Test day
        missed3 = q["frauds"] - q["captured"]
        ceiling = ceilings.get((r["configuration"], index))
        for row in e["grid"]:
            captured4 = q["captured"] - row["frauds_forgone"]
            body.append(
                [
                    cell(r["configuration"]),
                    cell(row["ratio"]),
                    cell(row["cost_r3"] / days),
                    cell(row["cost_r4"] / days),
                    cell(q["captured"]),
                    cell(missed3),
                    cell(q["captured"] / q["frauds"]),
                    cell(captured4),
                    cell(q["frauds"] - captured4),
                    cell(captured4 / q["frauds"]),
                    cell(captured4 / (ceiling * q["frauds"])) if ceiling else "—",
                ]
            )
    return [
        "Per-day costs and captured frauds, derived from the recorded costs, queue and frauds "
        "forgone (R4 captured = R3 captured - frauds forgone; normalised recall = captured / "
        "oracle-reviewable frauds):",
        "",
        *table(
            [
                "Configuration",
                "rho",
                "J R3 per day",
                "J R4 per day",
                "R3 captured",
                "R3 false negatives",
                "R3 recall",
                "R4 captured",
                "R4 false negatives",
                "R4 recall",
                "R4 normalised recall",
            ],
            body,
        ),
        "",
    ]


def ceilings_of(evaluations: Sequence[Record]) -> dict[tuple[str, int], float]:
    """The oracle ceiling of R3 by (configuration, capacity index) from the evaluation records."""
    return {
        (r["configuration"], index): entry["metrics"]["oracle_ceiling"]
        for r in evaluations
        for index, entry in enumerate(r["r3"])
    }


def converted_intervals(inference: Record) -> list[str]:
    """The effect units with the interval of the estimand converted by the recorded basis."""
    basis = inference["effect_basis"]
    h1, h2 = inference["h1"], inference["h2"]
    levels: dict[str, tuple[str, Record]] = {"family": ("90 %", h1["interval"])}
    for group in ("within_family", "by_weighting", "cells"):
        levels.update({e["name"]: ("95 %", e["interval"]) for e in h1[group]})
    levels[h1["interaction"]["name"]] = ("95 %", h1["interaction"]["interval"])
    levels.update(
        {f"weighting:{m['model']}": ("95 %", m["difference_interval"]) for m in h2["models"]}
    )
    slots = basis["k"] * basis["days"]

    def converted(x: float) -> list[str]:
        return [
            cell(100 * x),
            cell(x * basis["frauds"]),
            cell(x * basis["frauds"] / basis["days"]),
            cell(100 * x * basis["frauds"] / slots),
            cell(x / basis["oracle_ceiling"]),
        ]

    body = []
    for u in inference["effect_units"]:
        found = levels.get(u["estimand"])
        if found is None:
            continue
        level, interval = found
        body.append(
            [
                cell(u["estimand"]),
                level,
                *(
                    f"{lo} to {hi}"
                    for lo, hi in zip(
                        converted(interval["lower"]), converted(interval["upper"]), strict=True
                    )
                ),
            ]
        )
    return [
        "### Effect units with converted intervals",
        "",
        "The interval of each estimand multiplied by the same recorded factors as its point "
        "estimate (percentage points x 100; frauds per partition x frauds; per day / days; "
        "Precision@K points x frauds / (K x days) x 100; normalised recall / oracle ceiling).",
        "",
        *table(
            [
                "Estimand",
                "Level",
                "Percentage points",
                "Frauds per partition",
                "Frauds per day",
                "Precision@K equivalent (points)",
                "Normalised recall",
            ],
            body,
        ),
    ]


__all__ = [
    "calibration_sensitivity",
    "ceilings_of",
    "converted_intervals",
    "non_monotone",
    "rq3_outcomes",
]
