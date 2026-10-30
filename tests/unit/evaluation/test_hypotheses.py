"""H1, H2 and H3 from synthetic replicate matrices (no data is read): the scenarios are built so
that the outcome is known in advance, and the numbers are checked against the replicate vectors."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from review_capacity.errors import DataValidationError
from review_capacity.evaluation.contrasts import Estimands, estimands
from review_capacity.evaluation.declaration import EvaluationDeclaration, load_declaration
from review_capacity.evaluation.hypotheses import (
    H1Result,
    H2Result,
    H3Result,
    analyse_h1,
    analyse_h2,
    analyse_h3,
    decide,
    h3_interval_label,
)
from review_capacity.evaluation.inference import H1Category, Position, holm, smallest_tolerance
from review_capacity.evaluation.resampling import Replicates
from support.synthetic import REPO_PATHS

B = 2000
# Recall@2 % in the declared order LGB-u, Cat-u, RES-u, FTT-u, LGB-b, Cat-b, RES-b, FTT-b:
# neural minus tree is -0.06, weighting changes nothing, so no interaction.
RECALLS = np.array([0.40, 0.39, 0.34, 0.33, 0.40, 0.39, 0.34, 0.33])
# Balanced weighting costs the neural models 0.06 more than the unweighted ones: an interaction.
SPLIT_RECALLS = np.array([0.40, 0.39, 0.34, 0.33, 0.40, 0.39, 0.28, 0.27])
RATES = np.array([0.0195, 0.0205, 0.0135, 0.0260, 0.0200, 0.0190, 0.0210, 0.0200])
H3_CONFIGURATION = "LGB-b"  # the fifth configuration


@pytest.fixture(scope="module")
def declaration() -> EvaluationDeclaration:
    return load_declaration(REPO_PATHS)[0]


@pytest.fixture(scope="module")
def table(declaration: EvaluationDeclaration) -> Estimands:
    return estimands(declaration)


def around(base: np.ndarray, sigma: float, seed: int) -> Replicates:
    rng = np.random.default_rng(seed)
    return Replicates(base + rng.normal(0.0, sigma, (B, 8)))


@pytest.fixture(scope="module")
def h1(declaration: EvaluationDeclaration, table: Estimands) -> H1Result:
    return analyse_h1(declaration, table, RECALLS, around(RECALLS, 0.004, 1))


@pytest.fixture(scope="module")
def h2(declaration: EvaluationDeclaration, table: Estimands) -> H2Result:
    return analyse_h2(declaration, table, RECALLS, around(RECALLS, 0.004, 1))


def rate_sets(sigma: dict[int, float]) -> dict[int, Replicates]:
    return {length: around(RATES, value, 10 + length) for length, value in sigma.items()}


@pytest.fixture(scope="module")
def h3(declaration: EvaluationDeclaration) -> H3Result:
    return analyse_h3(
        declaration, H3_CONFIGURATION, RATES, rate_sets({3: 0.0008, 1: 0.0010, 7: 0.0006})
    )


# --- H1 ------------------------------------------------------------------------------------------


def test_h1_finds_the_family_effect_and_places_its_interval_beyond_the_sesoi(h1: H1Result) -> None:
    assert h1.estimate == pytest.approx(-0.06)
    assert h1.p.value == pytest.approx(2 / (B + 1))  # no replicate is on the other side of zero
    assert h1.interval.level == 0.90
    assert h1.interval.upper < -0.02  # entirely beyond -SESOI
    assert h1.position is Position.BEYOND
    assert h1.difference_interval.level == 0.95
    assert h1.difference_interval.lower < h1.interval.lower  # a 95 % interval is wider than 90 %


def test_h1_reports_the_descriptive_contrasts_with_95_percent_intervals(h1: H1Result) -> None:
    assert [s.name for s in h1.within_family] == ["within:Cat-LGB", "within:FTT-RES"]
    assert [s.estimate for s in h1.within_family] == pytest.approx([-0.01, -0.01])
    assert len(h1.cells) == 8
    assert h1.cells[0].name == "cell:LGB-b-LGB-u"
    assert h1.interaction.estimate == pytest.approx(0.0, abs=1e-12)
    assert h1.interaction.interval.level == 0.95
    assert not h1.interaction_splits


def test_h1_splits_the_family_effect_when_the_interaction_is_clear_and_large(
    declaration: EvaluationDeclaration, table: Estimands
) -> None:
    result = analyse_h1(declaration, table, SPLIT_RECALLS, around(SPLIT_RECALLS, 0.004, 2))
    assert result.interaction.estimate == pytest.approx(-0.06)
    assert result.interaction_splits


def test_h1_reports_the_family_effect_at_each_weighting_from_the_replicates(
    declaration: EvaluationDeclaration, table: Estimands
) -> None:
    replicates = around(SPLIT_RECALLS, 0.004, 2)
    result = analyse_h1(declaration, table, SPLIT_RECALLS, replicates)
    assert result.interaction_splits
    assert [s.name for s in result.by_weighting] == ["family:u", "family:b"]
    unweighted, balanced = result.by_weighting
    assert unweighted.estimate == pytest.approx(-0.06)  # (0.34 + 0.33) / 2 - (0.40 + 0.39) / 2
    assert balanced.estimate == pytest.approx(-0.12)  # (0.28 + 0.27) / 2 - (0.40 + 0.39) / 2
    ratios = replicates.ratios  # each contrast is taken within every replicate
    for summary, neural, tree in ((unweighted, [2, 3], [0, 1]), (balanced, [6, 7], [4, 5])):
        draws = ratios[:, neural].mean(axis=1) - ratios[:, tree].mean(axis=1)
        lower, upper = np.percentile(draws, [2.5, 97.5], method="linear")
        assert summary.interval.level == 0.95
        assert summary.interval.lower == pytest.approx(lower)
        assert summary.interval.upper == pytest.approx(upper)


def test_h1_counts_undefined_replicates_as_not_rejecting(
    declaration: EvaluationDeclaration, table: Estimands
) -> None:
    replicates = around(RECALLS, 0.004, 1)
    replicates.ratios[:10] = np.nan
    result = analyse_h1(declaration, table, RECALLS, replicates)
    assert result.p.value == pytest.approx(2 * (1 + 10) / (B + 1))
    assert result.interval.undefined == 10


# --- H2 ------------------------------------------------------------------------------------------


def test_h2_concludes_equivalence_for_all_four_models(h2: H2Result) -> None:
    assert [m.model for m in h2.models] == ["LGB", "Cat", "RES", "FTT"]
    assert all(m.position is Position.INSIDE for m in h2.models)
    assert h2.p.value <= 0.01  # every one-sided null is rejected
    assert max(m.p_lower.value for m in h2.models) <= h2.p.value
    assert max(m.p_upper.value for m in h2.models) <= h2.p.value
    assert all(m.interval.level == 0.90 and m.difference_interval.level == 0.95 for m in h2.models)


def test_h2_is_the_maximum_of_the_eight_one_sided_p_values(
    declaration: EvaluationDeclaration, table: Estimands
) -> None:
    shifted = RECALLS + np.array([0, 0, 0, 0, 0.06, 0, 0, 0])  # LGB-b is 6 pp above LGB-u
    result = analyse_h2(declaration, table, shifted, around(shifted, 0.004, 3))
    lgb = result.models[0]
    assert lgb.estimate == pytest.approx(0.06)
    assert lgb.p_upper.value == pytest.approx(1.0)  # delta >= +0.02 in every replicate
    assert result.p == lgb.p_upper  # the maximum, with its own standard error
    assert result.p.value == pytest.approx(1.0)


# --- H3 ------------------------------------------------------------------------------------------


def test_h3_tests_the_h3_configuration_against_the_open_band(h3: H3Result) -> None:
    assert h3.configuration == "LGB-b"
    assert h3.rate == pytest.approx(0.0200)
    primary = h3.primary
    assert primary.block_length == 3
    assert primary.p_lower.value == pytest.approx(1 / (B + 1))  # r* <= 0.016 never happens
    assert primary.p_upper.value == pytest.approx(1 / (B + 1))  # r* >= 0.024 never happens
    assert primary.p.value == pytest.approx(1 / (B + 1))
    assert primary.interval.level == 0.90
    assert primary.position is Position.INSIDE


def test_h3_reports_the_other_block_lengths_and_the_tolerance_views(h3: H3Result) -> None:
    assert [v.block_length for v in h3.sensitivity] == [1, 7]
    assert all(v.position is Position.INSIDE for v in h3.sensitivity)
    assert [t.tolerance for t in h3.tolerances] == [0.10, 0.25, 0.50]
    assert h3.tolerances[0].lower == pytest.approx(0.018)
    assert h3.tolerances[0].upper == pytest.approx(0.022)
    assert all(t.position is Position.INSIDE for t in h3.tolerances)
    assert h3.smallest_tolerance == pytest.approx(smallest_tolerance(h3.primary.interval, 0.02))
    assert 0.0 < h3.smallest_tolerance < 0.10


def test_h3_describes_the_other_seven_configurations(h3: H3Result) -> None:
    assert len(h3.others) == 7
    assert "LGB-b" not in [o.configuration for o in h3.others]
    by_name = {o.configuration: o for o in h3.others}
    assert by_name["RES-u"].rate == pytest.approx(0.0135)
    assert by_name["RES-u"].position is Position.BEYOND  # below the band
    assert by_name["FTT-u"].position is Position.BEYOND  # above the band
    assert by_name["LGB-u"].position is Position.INSIDE


def test_h3_labels_describe_the_interval_and_never_claim_preservation() -> None:
    assert h3_interval_label(Position.INSIDE) == "CI inside the band"
    assert h3_interval_label(Position.BEYOND) == "CI entirely outside the band"
    assert h3_interval_label(Position.CROSSING) == "CI overlapping a bound (inconclusive)"
    assert all("preserved" not in h3_interval_label(p) for p in Position)


def test_non_finite_observed_values_and_infinite_replicates_are_rejected(
    declaration: EvaluationDeclaration, table: Estimands
) -> None:
    replicates = around(RECALLS, 0.004, 1)
    nan_recalls = RECALLS.copy()
    nan_recalls[3] = np.nan
    with pytest.raises(DataValidationError, match="one finite value per configuration"):
        analyse_h1(declaration, table, nan_recalls, replicates)
    with pytest.raises(DataValidationError, match="one finite value per configuration"):
        analyse_h2(declaration, table, RECALLS[:7], replicates)
    infinite = around(RECALLS, 0.004, 1)
    infinite.ratios[0, 2] = np.inf
    with pytest.raises(DataValidationError, match="must not contain infinities"):
        analyse_h1(declaration, table, RECALLS, infinite)
    sets = rate_sets({3: 0.0008, 1: 0.001, 7: 0.0006})
    inf_rates = RATES.copy()
    inf_rates[0] = np.inf
    with pytest.raises(DataValidationError, match="one finite value per configuration"):
        analyse_h3(declaration, "LGB-b", inf_rates, sets)


def test_h3_rejects_a_missing_set_an_unknown_configuration_and_wrong_shapes(
    declaration: EvaluationDeclaration,
) -> None:
    sets = rate_sets({3: 0.0008, 1: 0.001, 7: 0.0006})
    with pytest.raises(DataValidationError, match="not one of the configurations"):
        analyse_h3(declaration, "XGB-u", RATES, sets)
    with pytest.raises(DataValidationError, match="replicate sets of the block lengths"):
        analyse_h3(declaration, "LGB-b", RATES, {3: sets[3]})
    with pytest.raises(DataValidationError, match="one finite value per configuration"):
        analyse_h3(declaration, "LGB-b", RATES[:5], sets)
    broken = {**sets, 7: Replicates(sets[7].ratios[:, :5])}
    with pytest.raises(DataValidationError, match="columns"):
        analyse_h3(declaration, "LGB-b", RATES, broken)


# --- the decision --------------------------------------------------------------------------------


def test_holm_decides_the_three_hypotheses(
    declaration: EvaluationDeclaration, h1: H1Result, h2: H2Result, h3: H3Result
) -> None:
    result = decide(declaration, h1, h2, h3)
    expected = holm((h1.p.value, h2.p.value, h3.primary.p.value), 0.05)  # the order H1, H2, H3
    assert result.holm == expected
    assert all(value <= 0.05 for value in result.holm.adjusted)
    assert result.h1_rejected
    assert result.h2_equivalent
    assert result.h3_preserved
    assert result.h1_category is H1Category.MEANINGFUL


def test_the_adjusted_p_value_decides_not_the_interval(
    declaration: EvaluationDeclaration, table: Estimands, h2: H2Result, h3: H3Result
) -> None:
    noisy = around(RECALLS, 0.08, 4)  # the family effect is lost in noise
    h1 = analyse_h1(declaration, table, RECALLS, noisy)
    result = decide(declaration, h1, h2, h3)
    assert h1.p.value > 0.05
    assert not result.h1_rejected
    assert result.h1_category is H1Category.INCONCLUSIVE  # not significant, interval wide
    assert result.holm.rejected[0] is False


def test_h3_is_not_preserved_when_the_adjusted_p_value_does_not_reject(
    declaration: EvaluationDeclaration, h1: H1Result, h2: H2Result
) -> None:
    wide = analyse_h3(
        declaration, H3_CONFIGURATION, RATES, rate_sets({3: 0.004, 1: 0.004, 7: 0.004})
    )
    result = decide(declaration, h1, h2, wide)
    assert wide.primary.position is Position.CROSSING
    assert not result.h3_preserved
    assert result.holm.adjusted[2] > 0.05


def test_an_interval_inside_the_band_can_accompany_a_non_rejection_after_holm(
    declaration: EvaluationDeclaration, table: Estimands, h2: H2Result
) -> None:
    # sigma 0.0022: the 90 % interval lies inside (0.016, 0.024), yet p_H3 is about 0.03, and
    # with H2 smaller and H1 larger the adjusted value is 2 x p_H3, above 0.05
    inside = analyse_h3(
        declaration, H3_CONFIGURATION, RATES, rate_sets({3: 0.0022, 1: 0.0022, 7: 0.0022})
    )
    h1 = analyse_h1(declaration, table, RECALLS, around(RECALLS, 0.08, 4))  # p_H1 is large
    assert inside.primary.position is Position.INSIDE
    assert 0.02 < inside.primary.p.value < 0.05
    result = decide(declaration, h1, h2, inside)
    assert result.holm.adjusted[2] > 0.05
    assert not result.h3_preserved
    assert h3_interval_label(inside.primary.position) == "CI inside the band"


def test_the_declared_tolerances_follow_the_declaration(
    declaration: EvaluationDeclaration, h3: H3Result
) -> None:
    changed = dataclasses.replace(
        declaration,
        inference=dataclasses.replace(declaration.inference, h3_tolerance_sensitivity=(0.05,)),
    )
    result = analyse_h3(
        changed, H3_CONFIGURATION, RATES, rate_sets({3: 0.0008, 1: 0.001, 7: 0.0006})
    )
    assert [t.tolerance for t in result.tolerances] == [0.05]
    assert result.tolerances[0].position is Position.CROSSING  # (0.019, 0.021) is narrower
    assert h3.primary.position is Position.INSIDE
