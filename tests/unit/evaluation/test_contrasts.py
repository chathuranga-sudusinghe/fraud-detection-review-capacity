"""The estimand coefficient vectors of H1, H2 and the descriptive contrasts (DR-005), checked
against hand computations on a small recall vector."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from review_capacity.errors import DataValidationError, SpecificationError
from review_capacity.evaluation.contrasts import Estimands, estimands
from review_capacity.evaluation.declaration import EvaluationDeclaration, load_declaration
from support.synthetic import REPO_PATHS

# Recall@2 % in the declared order: LGB-u, Cat-u, RES-u, FTT-u, LGB-b, Cat-b, RES-b, FTT-b
RECALLS = np.array([0.40, 0.39, 0.35, 0.34, 0.41, 0.38, 0.33, 0.32])


@pytest.fixture(scope="module")
def declaration() -> EvaluationDeclaration:
    return load_declaration(REPO_PATHS)[0]


@pytest.fixture(scope="module")
def table(declaration: EvaluationDeclaration) -> Estimands:
    return estimands(declaration)


def test_the_names_follow_the_estimand_list(table: Estimands) -> None:
    assert table.names == (
        "family",
        "weighting:LGB",
        "weighting:Cat",
        "weighting:RES",
        "weighting:FTT",
        "within:Cat-LGB",
        "within:FTT-RES",
        "interaction",
        *(f"cell:{a}-{b}" for a, b in LOAD_CELLS),
    )
    assert table.coefficients.shape == (len(table.names), 8)


LOAD_CELLS = (
    ("LGB-b", "LGB-u"),
    ("Cat-b", "Cat-u"),
    ("RES-b", "RES-u"),
    ("FTT-b", "FTT-u"),
    ("Cat-u", "LGB-u"),
    ("Cat-b", "LGB-b"),
    ("FTT-u", "RES-u"),
    ("FTT-b", "RES-b"),
)


def test_every_estimand_is_a_difference(table: Estimands) -> None:
    assert np.allclose(table.coefficients.sum(axis=1), 0.0)
    assert np.allclose(table.apply(np.full(8, 0.37)), 0.0)  # equal recalls give no effect


def test_the_family_effect_is_neural_minus_tree(table: Estimands) -> None:
    row = table.row("family")
    assert row.tolist() == [-0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25, 0.25]
    assert table.apply(RECALLS)[0] == pytest.approx(0.335 - 0.395)  # neural mean - tree mean


def test_the_weighting_effects_are_balanced_minus_unweighted(table: Estimands) -> None:
    effects = table.apply(RECALLS)
    assert effects[1:5] == pytest.approx([0.01, -0.01, -0.02, -0.02])  # LGB, Cat, RES, FTT


def test_the_within_family_contrasts_average_over_weighting(table: Estimands) -> None:
    effects = table.apply(RECALLS)
    assert effects[5] == pytest.approx((0.39 + 0.38) / 2 - (0.40 + 0.41) / 2)  # Cat - LGB
    assert effects[6] == pytest.approx((0.34 + 0.32) / 2 - (0.35 + 0.33) / 2)  # FTT - RES


def test_the_interaction_compares_the_family_gap_across_weightings(table: Estimands) -> None:
    gap_balanced = (0.33 + 0.32) / 2 - (0.41 + 0.38) / 2
    gap_unweighted = (0.35 + 0.34) / 2 - (0.40 + 0.39) / 2
    assert table.apply(RECALLS)[7] == pytest.approx(gap_balanced - gap_unweighted)


def test_the_cell_contrasts_are_the_declared_differences(table: Estimands) -> None:
    order = ["LGB-u", "Cat-u", "RES-u", "FTT-u", "LGB-b", "Cat-b", "RES-b", "FTT-b"]
    recall = dict(zip(order, RECALLS, strict=True))
    effects = table.apply(RECALLS)
    for index, (first, second) in enumerate(LOAD_CELLS, start=8):
        assert effects[index] == pytest.approx(recall[first] - recall[second])


def test_apply_works_on_replicate_matrices_and_keeps_nan(table: Estimands) -> None:
    replicates = np.vstack([RECALLS, RECALLS + 0.01, np.full(8, np.nan)])
    effects = table.apply(replicates)
    assert effects.shape == (3, len(table.names))
    assert effects[0] == pytest.approx(table.apply(RECALLS))
    assert effects[1] == pytest.approx(table.apply(RECALLS), abs=1e-12)  # a shift cancels
    assert np.isnan(effects[2]).all()


def test_a_row_is_found_by_name_and_apply_checks_the_configurations(table: Estimands) -> None:
    assert table.row("interaction").shape == (8,)
    with pytest.raises(DataValidationError, match="Unknown estimand"):
        table.row("no-such-estimand")
    with pytest.raises(DataValidationError, match="8 configurations"):
        table.apply(np.ones(7))


def test_two_models_per_family_are_required(declaration: EvaluationDeclaration) -> None:
    configurations = dataclasses.replace(declaration.configurations, tree=("LGB", "Cat", "XGB"))
    broken = dataclasses.replace(declaration, configurations=configurations)
    with pytest.raises(SpecificationError, match="two models per family"):
        estimands(broken)
