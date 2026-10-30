"""The exact McNemar test of the cell contrasts, checked against hand-computed binomial tails."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from review_capacity.errors import DataValidationError
from review_capacity.evaluation.declaration import EvaluationDeclaration, load_declaration
from review_capacity.evaluation.mcnemar import cell_mcnemar, exact_p
from support.synthetic import REPO_PATHS


@pytest.fixture(scope="module")
def declaration() -> EvaluationDeclaration:
    return load_declaration(REPO_PATHS)[0]


def tail(n: int, k: int) -> float:
    """P(X <= k) for X ~ Binomial(n, 0.5), from exact integers."""
    return sum(comb(n, i) for i in range(k + 1)) / (1 << n)


def test_without_discordant_frauds_the_p_value_is_one() -> None:
    assert exact_p(0, 0) == 1.0


def test_equal_discordant_counts_give_a_p_value_of_one() -> None:
    assert exact_p(5, 5) == pytest.approx(1.0)
    assert exact_p(2, 3) == pytest.approx(1.0)  # twice a tail of exactly one half is capped


@pytest.mark.parametrize(("first", "second"), [(0, 5), (1, 9), (3, 12), (12, 3), (7, 40)])
def test_the_p_value_is_twice_the_smaller_binomial_tail(first: int, second: int) -> None:
    n, smaller = first + second, min(first, second)
    assert exact_p(first, second) == pytest.approx(min(1.0, 2 * tail(n, smaller)), rel=1e-9)


def test_the_p_value_does_not_depend_on_which_side_has_the_larger_count() -> None:
    assert exact_p(2, 11) == exact_p(11, 2)


def test_negative_counts_are_rejected() -> None:
    with pytest.raises(DataValidationError, match="must not be negative"):
        exact_p(-1, 3)


def test_each_declared_cell_contrast_is_tested_on_its_own_pair(
    declaration: EvaluationDeclaration,
) -> None:
    rng = np.random.default_rng(31)
    captures = rng.random((8, 400)) < np.linspace(0.30, 0.45, 8)[:, np.newaxis]
    results = cell_mcnemar(declaration, captures)
    assert [(r.first, r.second) for r in results] == list(declaration.contrasts)
    order = declaration.configurations.order
    for result in results:
        a, b = captures[order.index(result.first)], captures[order.index(result.second)]
        assert result.only_first == int(np.sum(a & ~b))
        assert result.only_second == int(np.sum(~a & b))
        assert result.p == pytest.approx(exact_p(result.only_first, result.only_second))


def test_identical_captures_give_no_discordant_fraud(declaration: EvaluationDeclaration) -> None:
    captures = np.tile(np.arange(50) % 3 == 0, (8, 1))
    for result in cell_mcnemar(declaration, captures):
        assert (result.only_first, result.only_second, result.p) == (0, 0, 1.0)


def test_the_capture_matrix_is_checked(declaration: EvaluationDeclaration) -> None:
    with pytest.raises(DataValidationError, match="boolean matrix with 8 rows"):
        cell_mcnemar(declaration, np.ones((8, 5)))  # floats, not booleans
    with pytest.raises(DataValidationError, match="boolean matrix with 8 rows"):
        cell_mcnemar(declaration, np.ones((7, 5), dtype=bool))
    with pytest.raises(DataValidationError, match="boolean matrix with 8 rows"):
        cell_mcnemar(declaration, np.ones(8, dtype=bool))
