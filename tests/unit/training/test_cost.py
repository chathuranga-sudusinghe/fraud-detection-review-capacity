"""The predicted search cost (DR-015 §7), on synthetic candidate lists (no data read)."""

from __future__ import annotations

import pytest

from review_capacity.training.cost import (
    CONFIGURATIONS_PER_MODEL,
    FTT_DEFAULT,
    FTT_LARGEST,
    REFITS_PER_CONFIGURATION,
    ModelCost,
    ftt_epoch_minutes,
    predict_cost,
)
from review_capacity.training.declaration import TrainingDeclaration, load_declaration
from review_capacity.training.sampler import CandidateList, generate_candidates
from support.synthetic import REPO_PATHS


@pytest.fixture(scope="module")
def declaration() -> TrainingDeclaration:
    return load_declaration(REPO_PATHS)[0]


def constant_lists(
    declaration: TrainingDeclaration, layers: int, d_token: int
) -> dict[str, CandidateList]:
    """Lists in which every FT-Transformer candidate has the given size."""
    lists = generate_candidates(declaration)
    ftt = lists["ft_transformer"]
    fixed = tuple({**c, "n_layers": layers, "d_token": d_token} for c in ftt.candidates)
    resized = CandidateList(ftt.model, ftt.stream_index, ftt.spawn_key, fixed)
    return {**lists, "ft_transformer": resized}


def by_model(
    declaration: TrainingDeclaration, lists: dict[str, CandidateList]
) -> dict[str, ModelCost]:
    return {cost.model: cost for cost in predict_cost(declaration, lists)}


def hours(value: float | None) -> float:
    assert value is not None
    return value


def test_the_measured_settings_are_reproduced() -> None:
    assert ftt_epoch_minutes(FTT_DEFAULT.layers, FTT_DEFAULT.width) == pytest.approx(
        (FTT_DEFAULT.train, FTT_DEFAULT.total)
    )
    assert ftt_epoch_minutes(FTT_LARGEST.layers, FTT_LARGEST.width) == pytest.approx(
        (FTT_LARGEST.train, FTT_LARGEST.total)
    )


def test_the_average_candidate_costs_about_fourteen_minutes_per_epoch() -> None:
    """DR-015 §7: mean n_layers 3.5 and mean d_token 288 give about 14.02 minutes per epoch."""
    assert ftt_epoch_minutes(7, 288)[1] / 2 == pytest.approx(14.02, abs=0.01)


def test_a_wider_or_deeper_candidate_costs_more() -> None:
    assert ftt_epoch_minutes(3, 256)[1] > ftt_epoch_minutes(3, 192)[1]
    assert ftt_epoch_minutes(4, 192)[1] > ftt_epoch_minutes(3, 192)[1]
    assert ftt_epoch_minutes(1, 64)[1] > 0  # extrapolated below the default stays positive


def test_every_default_sized_candidate_gives_the_documented_search_time(
    declaration: TrainingDeclaration,
) -> None:
    costs = by_model(declaration, constant_lists(declaration, 3, 192))
    ftt = costs["ft_transformer"]
    minutes = 7.64  # the reference setting, both configurations, ten candidates
    assert (ftt.epochs_low, ftt.epochs_high) == (7, 15)
    assert hours(ftt.search_hours_low) == pytest.approx(
        10 * CONFIGURATIONS_PER_MODEL * 7 * minutes / 60
    )
    assert hours(ftt.search_hours_high) == pytest.approx(
        10 * CONFIGURATIONS_PER_MODEL * 15 * minutes / 60
    )


def test_the_refit_bound_follows_the_largest_candidate(declaration: TrainingDeclaration) -> None:
    """DR-015 §7: six refits of the largest setting at 15 epochs take about 60 hours."""
    costs = by_model(declaration, constant_lists(declaration, 6, 512))
    assert REFITS_PER_CONFIGURATION * CONFIGURATIONS_PER_MODEL == 6
    assert hours(costs["ft_transformer"].refit_hours) == pytest.approx(6 * 15 * 40.1 / 60)
    assert hours(costs["ft_transformer"].refit_hours) == pytest.approx(60.15)


def test_the_resnet_search_is_bounded_by_its_largest_measured_setting(
    declaration: TrainingDeclaration,
) -> None:
    resnet = by_model(declaration, generate_candidates(declaration))["resnet"]
    assert resnet.estimated
    assert hours(resnet.search_hours_high) == pytest.approx(2 * 10 * 15 * 0.48 / 60)
    assert 2.0 < hours(resnet.search_hours_high) < 2.5


def test_models_without_a_cost_model_are_recorded_as_not_estimated(
    declaration: TrainingDeclaration,
) -> None:
    costs = by_model(declaration, generate_candidates(declaration))
    for model in ("lightgbm", "catboost", "logistic_regression"):
        entry = costs[model]
        assert entry.estimated is False
        assert entry.candidate_epoch_minutes is None
        assert entry.search_hours_low is None
        assert entry.refit_hours is None


def test_the_actual_lists_are_priced_candidate_by_candidate(
    declaration: TrainingDeclaration,
) -> None:
    lists = generate_candidates(declaration)
    cost = by_model(declaration, lists)["ft_transformer"]
    expected = tuple(
        ftt_epoch_minutes(int(c["n_layers"]), int(c["d_token"]))[1]
        for c in lists["ft_transformer"].candidates
    )
    assert cost.candidate_epoch_minutes == expected
    assert expected[0] == pytest.approx(7.64)  # trial 1 is the reference setting
    assert hours(cost.search_hours_low) < hours(cost.search_hours_high)


def test_a_costlier_list_never_lowers_the_prediction(declaration: TrainingDeclaration) -> None:
    small = by_model(declaration, constant_lists(declaration, 1, 64))["ft_transformer"]
    large = by_model(declaration, constant_lists(declaration, 6, 512))["ft_transformer"]
    assert hours(large.search_hours_high) > hours(small.search_hours_high)
