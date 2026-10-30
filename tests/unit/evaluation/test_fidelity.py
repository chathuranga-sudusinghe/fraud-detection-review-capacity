"""Daily-budget fidelity and the rate/volume decomposition (Issue #197)."""

from __future__ import annotations

import math

import pytest

from review_capacity.errors import DataValidationError
from review_capacity.evaluation.fidelity import decomposition_terms, fidelity

K = 56
CAPACITY = 0.02
REFERENCE = 2_800.0  # K = 56 = 0.02 x 2,800 exactly


def days(alerts: list[int], volumes: list[int]) -> list[dict[str, int]]:
    return [
        {"day": 155 + i, "selected": n, "transactions": v}
        for i, (n, v) in enumerate(zip(alerts, volumes, strict=True))
    ]


def test_the_statistics_follow_their_definitions() -> None:
    result = fidelity(days([60, 56, 40, 0], [3000, 2800, 2600, 2500]), CAPACITY, K, REFERENCE)
    assert result["days"] == 4
    assert result["mean_alerts"] == pytest.approx(39.0)
    assert result["mean_deviation"] == pytest.approx(-17.0)
    assert result["mean_absolute_deviation"] == pytest.approx((4 + 0 + 16 + 56) / 4)
    assert result["max_absolute_deviation"] == 56
    assert result["mean_relative_deviation"] == pytest.approx(-17 / 56)
    assert (result["days_above_budget"], result["days_below_budget"]) == (1, 2)
    assert (result["share_above_budget"], result["share_below_budget"]) == (0.25, 0.5)
    assert (result["total_overflow"], result["total_unused"]) == (4, 72)
    assert result["mean_load_ratio"] == pytest.approx((60 + 56 + 40) / 56 / 4)
    assert result["zero_alert_days"] == 1


def test_the_decomposition_is_exact_for_each_day_with_an_alert() -> None:
    result = fidelity(days([60, 56, 40, 0], [3000, 2800, 2600, 2500]), CAPACITY, K, REFERENCE)
    for term in result["terms"]:
        if term["alerts"] == 0:
            assert term["total"] is None
            continue
        assert term["rate"] + term["volume"] + term["budget"] == pytest.approx(term["total"])
        assert term["total"] == pytest.approx(math.log(term["alerts"] / K))
    assert result["mean_total"] == pytest.approx(
        result["mean_rate"] + result["mean_volume"] + result["mean_budget"]
    )
    assert result["mean_budget"] == pytest.approx(0.0, abs=1e-12)


def test_the_budget_term_is_the_rounding_of_k() -> None:
    # K = 139 for 5 % of 2,780 rows a day; a reference of 2,800 gives 140 and a budget term
    terms = decomposition_terms(139, 2_800, 0.05, 139, 2_800.0)
    assert terms["budget"] == pytest.approx(math.log(140 / 139))
    assert terms["rate"] + terms["volume"] + terms["budget"] == pytest.approx(terms["total"])


def test_a_day_without_alerts_is_left_out_of_the_logarithmic_means() -> None:
    result = fidelity(days([0, 0], [2500, 2600]), CAPACITY, K, REFERENCE)
    assert result["zero_alert_days"] == 2
    assert result["mean_total"] is None
    assert result["mean_rate"] is None
    assert result["total_unused"] == 2 * K


def test_inputs_are_checked() -> None:
    with pytest.raises(DataValidationError, match="at least one day"):
        fidelity([], CAPACITY, K, REFERENCE)
    with pytest.raises(DataValidationError, match="needs an alert"):
        decomposition_terms(0, 2800, CAPACITY, K, REFERENCE)
