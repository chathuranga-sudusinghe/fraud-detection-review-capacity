"""The building blocks of the confirmatory inference: add-one p-values, intervals, positions against
open bounds, the H1 category, Holm, the smallest H3 tolerance and the effect units, checked by
hand."""

from __future__ import annotations

import math

import numpy as np
import pytest

from review_capacity.errors import DataValidationError
from review_capacity.evaluation.inference import (
    H1Category,
    Interval,
    Position,
    PValue,
    effect_units,
    h1_category,
    holm,
    interaction_splits_family_effect,
    largest,
    one_sided_p,
    percentile_interval,
    position_of,
    smallest_tolerance,
    two_sided_p_about_zero,
)

# --- p-values ------------------------------------------------------------------------------------


def test_one_sided_p_values_are_add_one_tail_proportions() -> None:
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    at_most = one_sided_p(values, 0.25, tail="at_most")  # two replicates are <= 0.25
    at_least = one_sided_p(values, 0.25, tail="at_least")  # three are >= 0.25
    assert at_most.value == pytest.approx(3 / 6)
    assert at_least.value == pytest.approx(4 / 6)
    assert at_most.mc_se == pytest.approx(math.sqrt(0.5 * 0.5 / 5))


def test_a_replicate_exactly_on_the_bound_counts_in_the_tail() -> None:
    values = np.array([0.1, 0.2, 0.3])
    assert one_sided_p(values, 0.2, tail="at_most").value == pytest.approx(3 / 4)
    assert one_sided_p(values, 0.2, tail="at_least").value == pytest.approx(3 / 4)


def test_undefined_replicates_count_as_not_rejecting_and_keep_b_fixed() -> None:
    values = np.array([0.5, 0.6, np.nan, np.nan])
    assert one_sided_p(values, 0.0, tail="at_most").value == pytest.approx(3 / 5)  # 1 + 2 NaN
    assert one_sided_p(values, 0.0, tail="at_least").value == pytest.approx(5 / 5)  # 1 + 2 + 2


def test_the_two_sided_p_value_doubles_the_smaller_tail() -> None:
    values = np.concatenate([np.full(10, -0.1), np.full(90, 0.1)])
    result = two_sided_p_about_zero(values)
    q = 11 / 101  # 1 + #{value <= 0}
    assert result.value == pytest.approx(2 * q)
    assert result.mc_se == pytest.approx(2 * math.sqrt(q * (1 - q) / 100))


def test_the_two_sided_p_value_of_an_all_positive_sample_is_the_smallest_possible() -> None:
    result = two_sided_p_about_zero(np.full(999, 0.2))
    assert result.value == pytest.approx(2 / 1000)


def test_the_two_sided_p_value_is_capped_at_one() -> None:
    values = np.array([-1.0, 1.0] * 50)
    assert two_sided_p_about_zero(values).value == 1.0


def test_the_largest_p_value_carries_its_own_standard_error() -> None:
    chosen = largest([PValue(0.01, 0.001), PValue(0.20, 0.004), PValue(0.05, 0.002)])
    assert chosen == PValue(0.20, 0.004)
    with pytest.raises(DataValidationError, match="at least one component"):
        largest([])


def test_infinite_replicates_and_bounds_are_rejected_but_nan_is_allowed() -> None:
    with pytest.raises(DataValidationError, match="must not contain infinities"):
        one_sided_p(np.array([0.1, np.inf]), 0.0, tail="at_least")
    with pytest.raises(DataValidationError, match="must not contain infinities"):
        two_sided_p_about_zero(np.array([-np.inf, 0.1]))
    with pytest.raises(DataValidationError, match="must not contain infinities"):
        percentile_interval(np.array([0.1, -np.inf, 0.3]), 0.9)
    with pytest.raises(DataValidationError, match="bound must be finite"):
        one_sided_p(np.array([0.1, 0.2]), np.nan, tail="at_most")
    with pytest.raises(DataValidationError, match="bound must be finite"):
        one_sided_p(np.array([0.1, 0.2]), np.inf, tail="at_least")
    assert one_sided_p(np.array([0.1, np.nan]), 0.05, tail="at_most").value == pytest.approx(2 / 3)


def test_a_single_replicate_gives_the_extreme_but_valid_results() -> None:
    assert one_sided_p(np.array([0.5]), 0.0, tail="at_most").value == pytest.approx(1 / 2)
    assert one_sided_p(np.array([0.5]), 0.0, tail="at_least").value == pytest.approx(2 / 2)
    assert two_sided_p_about_zero(np.array([0.5])).value == 1.0  # min(1, 2 x 1/2)
    interval = percentile_interval(np.array([0.5]), 0.9)
    assert (interval.lower, interval.upper, interval.replicates) == (0.5, 0.5, 1)


def test_p_values_need_a_vector_of_replicates() -> None:
    with pytest.raises(DataValidationError, match="non-empty vector"):
        one_sided_p(np.zeros((2, 2)), 0.0, tail="at_most")
    with pytest.raises(DataValidationError, match="non-empty vector"):
        two_sided_p_about_zero(np.array([]))


# --- intervals and positions ---------------------------------------------------------------------


def test_the_percentile_interval_is_linear_over_the_defined_replicates() -> None:
    interval = percentile_interval(np.arange(101, dtype=np.float64), 0.95)
    assert interval.lower == pytest.approx(2.5)
    assert interval.upper == pytest.approx(97.5)
    assert interval.level == 0.95
    assert (interval.undefined, interval.replicates) == (0, 101)


def test_undefined_replicates_are_left_out_of_the_interval_and_counted() -> None:
    values = np.concatenate([np.arange(101, dtype=np.float64), [np.nan, np.nan]])
    interval = percentile_interval(values, 0.90)
    assert interval.lower == pytest.approx(5.0)
    assert interval.upper == pytest.approx(95.0)
    assert (interval.undefined, interval.replicates) == (2, 103)


def test_an_interval_needs_a_level_and_a_defined_replicate() -> None:
    with pytest.raises(DataValidationError, match="level must lie in"):
        percentile_interval(np.arange(5.0), 1.0)
    with pytest.raises(DataValidationError, match="Every bootstrap replicate was undefined"):
        percentile_interval(np.full(5, np.nan), 0.9)


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (-0.01, 0.01, Position.INSIDE),
        (0.021, 0.04, Position.BEYOND),
        (-0.05, -0.021, Position.BEYOND),
        (-0.03, 0.01, Position.CROSSING),
        (-0.01, 0.03, Position.CROSSING),
        (-0.05, 0.05, Position.CROSSING),
        (0.0, 0.02, Position.CROSSING),  # an endpoint exactly on a bound: open bounds
        (-0.02, 0.0, Position.CROSSING),
        (0.02, 0.05, Position.CROSSING),
        (-0.05, -0.02, Position.CROSSING),
    ],
)
def test_the_position_uses_open_bounds(lower: float, upper: float, expected: Position) -> None:
    assert position_of(Interval(lower, upper, 0.9), -0.02, 0.02) is expected


# --- the H1 category and the interaction rule ----------------------------------------------------


@pytest.mark.parametrize(
    ("significant", "position", "expected"),
    [
        (True, Position.BEYOND, H1Category.MEANINGFUL),
        (True, Position.CROSSING, H1Category.UNCERTAIN_SIZE),
        (True, Position.INSIDE, H1Category.NEGLIGIBLE),
        (False, Position.INSIDE, H1Category.COMPATIBLE_NEGLIGIBLE),
        (False, Position.CROSSING, H1Category.INCONCLUSIVE),
        (False, Position.BEYOND, H1Category.INCONCLUSIVE),
    ],
)
def test_the_five_h1_categories(
    significant: bool, position: Position, expected: H1Category
) -> None:
    assert h1_category(significant, position) is expected


def test_the_interaction_splits_the_family_effect_only_when_it_is_clear_and_large() -> None:
    clear = Interval(0.01, 0.05, 0.95)
    assert interaction_splits_family_effect(clear, 0.03, 0.02)
    assert interaction_splits_family_effect(Interval(-0.06, -0.01, 0.95), -0.03, 0.02)
    assert not interaction_splits_family_effect(clear, 0.015, 0.02)  # excludes 0 but small
    assert not interaction_splits_family_effect(Interval(-0.01, 0.05, 0.95), 0.03, 0.02)
    assert not interaction_splits_family_effect(clear, 0.02, 0.02)  # exactly the SESOI


# --- Holm ----------------------------------------------------------------------------------------


def test_holm_reproduces_the_worked_example_of_the_protocol() -> None:
    result = holm((0.020, 0.021, 0.040), 0.05)
    assert result.adjusted == pytest.approx((0.060, 0.060, 0.060))
    assert result.rejected == (False, False, False)


def test_holm_keeps_the_input_order_and_rejects_while_the_adjusted_value_is_small() -> None:
    result = holm((0.04, 0.001, 0.02), 0.05)  # sorted: 0.001 x 3, 0.02 x 2, 0.04 x 1
    assert result.adjusted == pytest.approx((0.04, 0.003, 0.04))
    assert result.rejected == (True, True, True)


def test_holm_stops_at_the_first_failure() -> None:
    result = holm((0.2, 0.01, 0.03), 0.05)  # sorted: 0.03, 0.06, 0.2
    assert result.adjusted == pytest.approx((0.2, 0.03, 0.06))
    assert result.rejected == (False, True, False)


def test_holm_adjusted_values_never_exceed_one_and_never_decrease_with_rank() -> None:
    result = holm((0.5, 0.9, 0.8), 0.05)
    assert result.adjusted == (1.0, 1.0, 1.0)
    assert not any(result.rejected)


def test_holm_rejects_when_the_adjusted_value_equals_the_level_exactly() -> None:
    # two hypotheses: 2 x 0.025 = 0.05 is rejected ("at most"); a hair above is not
    assert holm((0.025, 0.9), 0.05).rejected[0] is True
    assert holm((0.0250001, 0.9), 0.05).rejected[0] is False


def test_holm_checks_its_inputs() -> None:
    with pytest.raises(DataValidationError, match="Holm needs"):
        holm((), 0.05)
    with pytest.raises(DataValidationError, match="Holm needs"):
        holm((0.1,), 1.0)
    with pytest.raises(DataValidationError, match=r"lie in \[0, 1\]"):
        holm((0.1, 1.2), 0.05)


# --- t_min and the effect units ------------------------------------------------------------------


def test_the_smallest_tolerance_is_the_larger_relative_distance_to_the_target() -> None:
    assert smallest_tolerance(Interval(0.0175, 0.0225, 0.9), 0.02) == pytest.approx(0.125)
    assert smallest_tolerance(Interval(0.018, 0.0235, 0.9), 0.02) == pytest.approx(0.175)
    with pytest.raises(DataValidationError, match="must be positive"):
        smallest_tolerance(Interval(0.0175, 0.0225, 0.9), 0.0)


def test_two_points_of_recall_in_the_units_of_the_protocol() -> None:
    units = effect_units(0.02, frauds=2660, days=28, k=56, oracle_ceiling=0.6)
    assert units.percentage_points == pytest.approx(2.0)
    assert units.frauds_per_partition == pytest.approx(53.2)  # about 53 frauds per 28 days
    assert units.frauds_per_day == pytest.approx(1.9)  # about 2 per day
    assert units.precision_at_k_points == pytest.approx(3.3929, abs=1e-4)  # about 3.4 pp
    assert units.normalised_recall == pytest.approx(0.02 / 0.6)


def test_the_effect_units_need_positive_counts() -> None:
    with pytest.raises(DataValidationError, match="positive counts"):
        effect_units(0.02, frauds=0, days=28, k=56, oracle_ceiling=0.6)
    with pytest.raises(DataValidationError, match="positive counts"):
        effect_units(0.02, frauds=10, days=28, k=56, oracle_ceiling=0.0)
