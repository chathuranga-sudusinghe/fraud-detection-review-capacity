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

CELLS = (
    ("LGB-b", "LGB-u"),
    ("Cat-b", "Cat-u"),
    ("RES-b", "RES-u"),
    ("FTT-b", "FTT-u"),
    ("Cat-u", "LGB-u"),
    ("Cat-b", "LGB-b"),
    ("FTT-u", "RES-u"),
    ("FTT-b", "RES-b"),
)


@pytest.fixture(scope="module")
def declaration() -> EvaluationDeclaration:
    return load_declaration(REPO_PATHS)[0]


@pytest.fixture(scope="module")
def table(declaration: EvaluationDeclaration) -> Estimands:
    return estimands(declaration)


def value(table: Estimands, name: str) -> float:
    """The estimand ``name`` of the recall vector."""
    return float(table.apply(RECALLS)[table.names.index(name)])


def test_the_names_follow_the_estimand_list(table: Estimands) -> None:
    assert table.names == (
        "family",
        "family:u",
        "family:b",
        "weighting:LGB",
        "weighting:Cat",
        "weighting:RES",
        "weighting:FTT",
        "within:Cat-LGB",
        "within:FTT-RES",
        "interaction",
        *(f"cell:{a}-{b}" for a, b in CELLS),
    )
    assert table.coefficients.shape == (len(table.names), 8)


def test_every_estimand_is_a_difference(table: Estimands) -> None:
    assert np.allclose(table.coefficients.sum(axis=1), 0.0)
    assert np.allclose(table.apply(np.full(8, 0.37)), 0.0)  # equal recalls give no effect


def test_the_family_effect_is_neural_minus_tree(table: Estimands) -> None:
    row = table.row("family")
    assert row.tolist() == [-0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25, 0.25]
    assert value(table, "family") == pytest.approx(0.335 - 0.395)  # neural mean - tree mean


def test_the_family_effect_at_each_weighting_is_the_gap_of_the_family_means(
    table: Estimands,
) -> None:
    assert table.row("family:u").tolist() == [-0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
    assert table.row("family:b").tolist() == [0.0, 0.0, 0.0, 0.0, -0.5, -0.5, 0.5, 0.5]
    assert value(table, "family:u") == pytest.approx((0.35 + 0.34) / 2 - (0.40 + 0.39) / 2)
    assert value(table, "family:b") == pytest.approx((0.33 + 0.32) / 2 - (0.41 + 0.38) / 2)
    # delta_F is their mean, and the interaction is their difference
    assert value(table, "family") == pytest.approx(
        (value(table, "family:u") + value(table, "family:b")) / 2
    )
    assert value(table, "interaction") == pytest.approx(
        value(table, "family:b") - value(table, "family:u")
    )


def test_the_weighting_effects_are_balanced_minus_unweighted(table: Estimands) -> None:
    effects = [value(table, f"weighting:{model}") for model in ("LGB", "Cat", "RES", "FTT")]
    assert effects == pytest.approx([0.01, -0.01, -0.02, -0.02])


def test_the_within_family_contrasts_average_over_weighting(table: Estimands) -> None:
    assert value(table, "within:Cat-LGB") == pytest.approx((0.39 + 0.38) / 2 - (0.40 + 0.41) / 2)
    assert value(table, "within:FTT-RES") == pytest.approx((0.34 + 0.32) / 2 - (0.35 + 0.33) / 2)


def test_the_interaction_compares_the_family_gap_across_weightings(table: Estimands) -> None:
    gap_balanced = (0.33 + 0.32) / 2 - (0.41 + 0.38) / 2
    gap_unweighted = (0.35 + 0.34) / 2 - (0.40 + 0.39) / 2
    assert value(table, "interaction") == pytest.approx(gap_balanced - gap_unweighted)


def test_the_cell_contrasts_are_the_declared_differences(table: Estimands) -> None:
    order = ["LGB-u", "Cat-u", "RES-u", "FTT-u", "LGB-b", "Cat-b", "RES-b", "FTT-b"]
    recall = dict(zip(order, RECALLS, strict=True))
    for first, second in CELLS:
        assert value(table, f"cell:{first}-{second}") == pytest.approx(
            recall[first] - recall[second]
        )


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
