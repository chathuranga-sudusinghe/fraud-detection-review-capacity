"""The analysis tables of the summary: the rate and volume components and missing records."""

from __future__ import annotations

import math

import pytest

from review_capacity.development.analysis_summary import (
    SECTIONS,
    analysis_markdown,
    rate_volume_components,
    rolling_origin,
)
from review_capacity.development.run_output import LoadedRun
from support.development import example


def r2_record() -> dict[str, object]:
    """Two days of 1,000 and 2,000 transactions (mean 1,500), K = 30 at c = 0.02, and 40 and 10
    alerts; a third day without an alert."""
    days = [(1, 1000, 40), (2, 2000, 10), (3, 1500, 0)]
    return {
        "capacity": 0.02,
        "k": 30,
        "counts": {"transactions": 4500, "days": 3},
        "daily": [{"day": d, "transactions": n, "selected": a} for d, n, a in days],
    }


def test_the_components_are_the_means_over_the_days_with_an_alert() -> None:
    parts = rate_volume_components(r2_record())
    rate = (math.log(40 / 1000 / 0.02) + math.log(10 / 2000 / 0.02)) / 2
    volume = (math.log(1000 / 1500) + math.log(2000 / 1500)) / 2
    total = (math.log(40 / 30) + math.log(10 / 30)) / 2
    assert parts["rate"] == pytest.approx(rate)
    assert parts["volume"] == pytest.approx(volume)
    assert parts["total"] == pytest.approx(total)
    assert parts["zero_alert_days"] == 1
    # the decomposition is exact up to the rounding of K (K = 30 = 0.02 x 1,500)
    assert parts["total"] == pytest.approx((parts["rate"] or 0) + (parts["volume"] or 0))


def test_days_without_any_alert_leave_the_means_undefined() -> None:
    record = r2_record()
    record["daily"] = [{"day": 1, "transactions": 1000, "selected": 0}]
    record["counts"] = {"transactions": 1000, "days": 1}
    parts = rate_volume_components(record)
    assert (parts["rate"], parts["volume"], parts["total"]) == (None, None, None)
    assert parts["zero_alert_days"] == 1


def test_a_run_without_analysis_records_has_no_analysis_sections() -> None:
    run = LoadedRun(
        path=None,  # type: ignore[arg-type]
        records=[("run.json", example("run"))],
        end=None,
    )
    assert analysis_markdown(run) == []
    assert [section.name for section in SECTIONS] == [
        "threshold_outcomes",
        "threshold_workload",
        "rate_volume",
        "calibration",
        "h3_selection",
        "rolling_origin",
    ]


def test_the_rolling_table_lists_the_origins_in_order_with_threshold_and_silent_days() -> None:
    first = example("rolling_origin")
    second = example("rolling_origin")
    second["origin"] = 85
    second["threshold"] = second["threshold"] | {"value": 0.42}
    second["transfer"] = second["transfer"] | {"inside_band": False, "zero_alert_days": 2}
    run = LoadedRun(
        path=None,  # type: ignore[arg-type]
        records=[("b.json", second), ("a.json", first)],
        end=None,
    )
    text = "\n".join(rolling_origin(run))
    lines = [line for line in text.splitlines() if line.startswith(("| `71`", "| `85`"))]
    assert [line.split("|")[1].strip() for line in lines] == ["`71`", "`85`"]  # chronological
    header = text.splitlines()[2]
    assert "R2 threshold" in header
    assert "Zero-alert days" in header
    assert "`0.31`" in lines[0]
    assert "`0.42`" in lines[1]
    assert text.endswith("Origins inside the band: 1 of 2.")
