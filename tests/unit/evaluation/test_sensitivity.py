"""The sensitivity analyses S1 and S2 and the SESOI views, on synthetic replicate matrices (no data
is read). The Wald values are checked against normal-distribution formulas written with ``erfc``,
independently of the implementation."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from review_capacity.errors import DataValidationError
from review_capacity.evaluation.contrasts import Estimands, estimands
from review_capacity.evaluation.declaration import EvaluationDeclaration, load_declaration
from review_capacity.evaluation.hypotheses import (
    Confirmatory,
    analyse_h1,
    analyse_h2,
    analyse_h3,
    decide,
)
from review_capacity.evaluation.inference import H1Category, Position, holm
from review_capacity.evaluation.resampling import Replicates
from review_capacity.evaluation.sensitivity import (
    WaldQuantity,
    analyse_s1,
    analyse_s2,
    compare,
    sesoi_views,
    wald_interval,
)
from support.synthetic import REPO_PATHS

B = 2000
RECALLS = np.array([0.40, 0.39, 0.34, 0.33, 0.40, 0.39, 0.34, 0.33])  # neural - tree = -0.06
RATES = np.array([0.0195, 0.0205, 0.0135, 0.0260, 0.0200, 0.0190, 0.0210, 0.0200])
Z90 = 1.6448536269514722
SQRT2 = math.sqrt(2.0)


@pytest.fixture(scope="module")
def declaration() -> EvaluationDeclaration:
    return load_declaration(REPO_PATHS)[0]


@pytest.fixture(scope="module")
def table(declaration: EvaluationDeclaration) -> Estimands:
    return estimands(declaration)


def around(
    base: np.ndarray, sigma: float, seed: int, shape: tuple[int, int] = (B, 8)
) -> Replicates:
    rng = np.random.default_rng(seed)
    return Replicates(base + rng.normal(0.0, sigma, shape))


@pytest.fixture(scope="module")
def primary(declaration: EvaluationDeclaration, table: Estimands) -> Confirmatory:
    replicates = around(RECALLS, 0.004, 1)
    h3 = analyse_h3(
        declaration,
        "LGB-b",
        RATES,
        {length: around(RATES, 0.0008, 10 + length) for length in (3, 1, 7)},
    )
    return decide(
        declaration,
        analyse_h1(declaration, table, RECALLS, replicates),
        analyse_h2(declaration, table, RECALLS, replicates),
        h3,
    )


def sf(x: float) -> float:
    """The standard-normal upper tail, from the complementary error function."""
    return 0.5 * math.erfc(x / SQRT2)


def cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / SQRT2)


# --- the comparison rule -------------------------------------------------------------------------


def test_the_comparison_applies_holm_with_the_primary_h3_value(
    declaration: EvaluationDeclaration, primary: Confirmatory
) -> None:
    result = compare(declaration, primary, p_h1=0.001, p_h2=0.002, h1_position=Position.BEYOND)
    assert result.holm == holm((0.001, 0.002, primary.h3.primary.p.value), 0.05)
    assert (result.h1_rejected, result.h2_rejected) == (True, True)
    assert result.h1_category is H1Category.MEANINGFUL
    assert not result.differs


def test_a_changed_h1_decision_is_a_different_conclusion(
    declaration: EvaluationDeclaration, primary: Confirmatory
) -> None:
    result = compare(declaration, primary, p_h1=0.5, p_h2=0.002, h1_position=Position.BEYOND)
    assert not result.h1_rejected
    assert result.h1_category is H1Category.INCONCLUSIVE
    assert result.differs


def test_a_changed_h2_decision_is_a_different_conclusion(
    declaration: EvaluationDeclaration, primary: Confirmatory
) -> None:
    result = compare(declaration, primary, p_h1=0.001, p_h2=0.5, h1_position=Position.BEYOND)
    assert result.h1_rejected
    assert not result.h2_rejected
    assert result.differs


def test_a_changed_h1_category_alone_is_a_different_conclusion(
    declaration: EvaluationDeclaration, primary: Confirmatory
) -> None:
    result = compare(declaration, primary, p_h1=0.001, p_h2=0.002, h1_position=Position.CROSSING)
    assert (result.h1_rejected, result.h2_rejected) == (True, True)
    assert result.h1_category is H1Category.UNCERTAIN_SIZE
    assert result.differs


# --- S2 ------------------------------------------------------------------------------------------


def test_s2_with_the_same_evidence_does_not_change_the_conclusion(
    declaration: EvaluationDeclaration, table: Estimands, primary: Confirmatory
) -> None:
    outcome = analyse_s2(declaration, table, RECALLS, around(RECALLS, 0.004, 2), primary)
    assert outcome.h1.position is Position.BEYOND
    assert not outcome.comparison.differs
    assert outcome.comparison.p_h1 == outcome.h1.p.value
    assert outcome.comparison.p_h2 == outcome.h2.p.value


def test_s2_with_much_more_noise_changes_the_conclusion(
    declaration: EvaluationDeclaration, table: Estimands, primary: Confirmatory
) -> None:
    outcome = analyse_s2(declaration, table, RECALLS, around(RECALLS, 0.08, 3), primary)
    assert outcome.h1.p.value > 0.05
    assert not outcome.comparison.h1_rejected
    assert outcome.comparison.differs


# --- S1 ------------------------------------------------------------------------------------------


def test_a_wald_quantity_is_estimable_only_with_a_positive_corrected_variance() -> None:
    estimable = WaldQuantity("family", -0.06, v_pigeonhole=4e-4, v_naive=5e-5)
    assert estimable.corrected_variance == pytest.approx(3e-4)
    assert estimable.se == pytest.approx(math.sqrt(3e-4))
    assert WaldQuantity("family", -0.06, 1e-4, 5e-5).se is None  # exactly zero
    assert WaldQuantity("family", -0.06, 1e-4, 8e-5).se is None  # negative


def test_the_wald_interval_is_the_estimate_plus_and_minus_z_standard_errors() -> None:
    interval = wald_interval(0.10, 0.02, 0.95)
    assert interval.lower == pytest.approx(0.10 - 1.959963984540054 * 0.02)
    assert interval.upper == pytest.approx(0.10 + 1.959963984540054 * 0.02)
    assert interval.level == 0.95


def test_s1_follows_the_wald_formulas(
    declaration: EvaluationDeclaration, table: Estimands, primary: Confirmatory
) -> None:
    pigeonhole, naive = around(RECALLS, 0.004, 1), around(RECALLS, 0.0012, 5)
    outcome = analyse_s1(declaration, table, RECALLS, pigeonhole, naive, primary)
    assert [q.name for q in outcome.quantities] == [
        "family",
        "weighting:LGB",
        "weighting:Cat",
        "weighting:RES",
        "weighting:FTT",
    ]
    family = outcome.quantities[0]
    assert family.estimate == pytest.approx(-0.06)
    assert family.corrected_variance == pytest.approx(family.v_pigeonhole - 2 * family.v_naive)
    assert family.se is not None
    assert outcome.p_h1 == pytest.approx(math.erfc(abs(family.estimate) / family.se / SQRT2))
    assert outcome.interval_h1 is not None
    assert outcome.interval_h1.lower == pytest.approx(family.estimate - Z90 * family.se)
    assert outcome.interval_h1.upper == pytest.approx(family.estimate + Z90 * family.se)
    assert outcome.position_h1 is Position.BEYOND

    components = []
    for quantity in outcome.quantities[1:]:
        assert quantity.se is not None
        components.append(sf((quantity.estimate + 0.02) / quantity.se))  # H0a: delta <= -0.02
        components.append(cdf((quantity.estimate - 0.02) / quantity.se))  # H0b: delta >= +0.02
    assert outcome.p_h2 == pytest.approx(max(components))
    assert outcome.intervals_h2 is not None
    assert len(outcome.intervals_h2) == 4
    assert outcome.comparison is not None
    assert not outcome.comparison.differs


def test_s1_is_not_estimable_when_the_naive_variance_is_as_large_as_the_pigeonhole_one(
    declaration: EvaluationDeclaration, table: Estimands, primary: Confirmatory
) -> None:
    outcome = analyse_s1(
        declaration, table, RECALLS, around(RECALLS, 0.004, 1), around(RECALLS, 0.004, 6), primary
    )
    assert all(q.se is None for q in outcome.quantities)
    assert outcome.p_h1 is None
    assert outcome.interval_h1 is None
    assert outcome.p_h2 is None
    assert outcome.intervals_h2 is None
    assert outcome.comparison is None


def test_s1_comparison_needs_both_p_values(
    declaration: EvaluationDeclaration, table: Estimands, primary: Confirmatory
) -> None:
    pigeonhole = around(RECALLS, 0.004, 1)
    shock = np.random.default_rng(7).normal(0.0, 0.01, B)
    pigeonhole.ratios[:, [2, 3, 6, 7]] += shock[:, np.newaxis]  # moves delta_F, cancels in H2
    outcome = analyse_s1(
        declaration, table, RECALLS, pigeonhole, around(RECALLS, 0.004, 6), primary
    )
    assert outcome.quantities[0].se is not None  # the family effect is estimable
    assert outcome.p_h1 is not None
    assert outcome.p_h2 is None  # the weighting effects are not
    assert outcome.comparison is None  # Holm needs both


def test_s1_rejects_non_finite_recalls_and_infinite_replicates(
    declaration: EvaluationDeclaration, table: Estimands, primary: Confirmatory
) -> None:
    bad_recalls = RECALLS.copy()
    bad_recalls[0] = np.nan
    with pytest.raises(DataValidationError, match="one finite value per configuration"):
        analyse_s1(
            declaration,
            table,
            bad_recalls,
            around(RECALLS, 0.004, 1),
            around(RECALLS, 0.001, 5),
            primary,
        )
    infinite = around(RECALLS, 0.004, 1)
    infinite.ratios[3, 4] = np.inf
    with pytest.raises(DataValidationError, match="must not contain infinities"):
        analyse_s1(declaration, table, RECALLS, infinite, around(RECALLS, 0.001, 5), primary)


def test_s1_needs_two_defined_replicates(
    declaration: EvaluationDeclaration, table: Estimands, primary: Confirmatory
) -> None:
    one = around(RECALLS, 0.004, 1, shape=(1, 8))
    with pytest.raises(DataValidationError, match="at least two defined replicates"):
        analyse_s1(declaration, table, RECALLS, one, one, primary)


# --- the SESOI views -----------------------------------------------------------------------------


def test_the_sesoi_views_read_the_primary_intervals_against_each_alternative(
    declaration: EvaluationDeclaration, primary: Confirmatory
) -> None:
    changed = dataclasses.replace(
        declaration,
        inference=dataclasses.replace(declaration.inference, sesoi_sensitivity=(0.005, 0.06)),
    )
    narrow, wide = sesoi_views(changed, primary)
    assert (narrow.sesoi, wide.sesoi) == (0.005, 0.06)
    assert narrow.h1_position is Position.BEYOND  # [-0.065, -0.055] lies below -0.005
    assert narrow.h1_category is H1Category.MEANINGFUL
    assert wide.h1_position is Position.CROSSING  # -0.06 lies inside the interval
    assert wide.h1_category is H1Category.UNCERTAIN_SIZE
    assert narrow.h2_positions == (Position.CROSSING,) * 4
    assert wide.h2_positions == (Position.INSIDE,) * 4


def test_the_declared_sesoi_views_are_one_three_and_five_points(
    declaration: EvaluationDeclaration, primary: Confirmatory
) -> None:
    views = sesoi_views(declaration, primary)
    assert [v.sesoi for v in views] == [0.01, 0.03, 0.05]
    assert all(len(v.h2_positions) == 4 for v in views)
