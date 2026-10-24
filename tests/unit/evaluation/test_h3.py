"""The H3 configuration selection rule (DR-010)."""

from __future__ import annotations

import pytest

from review_capacity.development.records import CONFIGURATION_NAMES
from review_capacity.errors import DataValidationError
from review_capacity.evaluation.h3 import H3Candidate, select_h3


def candidates(captured: list[int], ap: list[float] | None = None) -> list[H3Candidate]:
    precision = ap or [0.1] * 8
    return [
        H3Candidate(name, hits, 100, p)
        for name, hits, p in zip(CONFIGURATION_NAMES, captured, precision, strict=True)
    ]


def test_the_configuration_that_captures_the_most_frauds_is_selected() -> None:
    choice = select_h3(candidates([50, 52, 51, 49, 53, 50, 52, 48]))
    assert choice.selected == "LGB-b"  # 53 frauds: the only maximum
    assert choice.tie_break_steps == ()


def test_equal_counts_are_decided_by_average_precision_then_the_fixed_order() -> None:
    by_precision = select_h3(
        candidates([50, 53, 51, 49, 53, 50, 52, 48], [0.1, 0.30, 0.1, 0.1, 0.31, 0.1, 0.1, 0.1])
    )
    assert (by_precision.selected, by_precision.tie_break_steps) == (
        "LGB-b",
        ("average_precision",),
    )
    by_order = select_h3(
        candidates([50, 53, 51, 49, 53, 50, 52, 48], [0.1, 0.30, 0.1, 0.1, 0.30, 0.1, 0.1, 0.1])
    )
    assert (by_order.selected, by_order.tie_break_steps) == (
        "Cat-u",  # earlier than LGB-b in the fixed order LGB-u, Cat-u, RES-u, FTT-u, LGB-b, ...
        ("average_precision", "fixed_order"),
    )


def test_a_count_always_beats_a_higher_average_precision() -> None:
    choice = select_h3(
        candidates([50, 51, 50, 50, 50, 50, 50, 50], [0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    )
    assert choice.selected == "Cat-u"
    assert choice.tie_break_steps == ()


def test_average_precision_is_compared_exactly() -> None:
    first = 0.1 + 0.2
    second = 0.3  # a different float: no tolerance
    assert first != second
    choice = select_h3(
        candidates([50, 50, 40, 40, 40, 40, 40, 40], [second, first, 0, 0, 0, 0, 0, 0])
    )
    assert (choice.selected, choice.tie_break_steps) == ("Cat-u", ("average_precision",))


def test_the_rule_needs_the_eight_configurations_and_consistent_counts() -> None:
    with pytest.raises(DataValidationError, match="eight configurations"):
        select_h3(candidates([1] * 8)[:7])
    with pytest.raises(DataValidationError, match="eight configurations"):
        select_h3(list(reversed(candidates([1] * 8))))
    mixed = candidates([1] * 8)
    mixed[3] = H3Candidate("FTT-u", 1, 99, 0.1)
    with pytest.raises(DataValidationError, match="same, non-zero"):
        select_h3(mixed)
    with pytest.raises(DataValidationError, match="between 0"):
        select_h3(candidates([101] + [1] * 7))
    with pytest.raises(DataValidationError, match="non-zero"):
        select_h3([H3Candidate(n, 0, 0, 0.0) for n in CONFIGURATION_NAMES])


def test_counts_and_average_precision_must_be_valid_numbers() -> None:
    nan = candidates([50] * 8)
    nan[0] = H3Candidate("LGB-u", 50, 100, float("nan"))
    with pytest.raises(DataValidationError, match="within"):
        select_h3(nan)
    outside = candidates([50] * 8)
    outside[2] = H3Candidate("RES-u", 50, 100, 1.5)
    with pytest.raises(DataValidationError, match="within"):
        select_h3(outside)
    fractional = candidates([50] * 8)
    fractional[1] = H3Candidate("Cat-u", 50.5, 100, 0.1)  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="integers"):
        select_h3(fractional)
    textual = candidates([50] * 8)
    textual[3] = H3Candidate("FTT-u", 50, 100, "0.1")  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="a number"):
        select_h3(textual)
