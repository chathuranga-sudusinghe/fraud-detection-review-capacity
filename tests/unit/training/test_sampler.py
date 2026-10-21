"""The candidate sampler: the draw formulas of DR-015 §6, stream independence and list identity
(synthetic seeds; no data read)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from review_capacity.training.declaration import Parameter, TrainingDeclaration, load_declaration
from review_capacity.training.sampler import draw, generate_candidates, reference_candidate, streams
from support.synthetic import REPO_PATHS


@pytest.fixture(scope="module")
def declaration() -> TrainingDeclaration:
    return load_declaration(REPO_PATHS)[0]


def generator(seed: int = 7) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed))


def parameter(kind: str, low: float, high: float, factor: int | None = None) -> Parameter:
    return Parameter("x", kind, low, high, low, factor)


# --- the draw formulas ---------------------------------------------------------------------------


def test_uniform_is_one_continuous_draw() -> None:
    p = parameter("uniform", 0.5, 1.0)
    assert draw(generator(), p) == generator().uniform(0.5, 1.0)


def test_log_uniform_exponentiates_a_uniform_draw_on_the_log_scale() -> None:
    p = parameter("log_uniform", 0.01, 0.3)
    z = generator().uniform(np.log(0.01), np.log(0.3))
    assert draw(generator(), p) == float(np.exp(z))


def test_uniform_int_includes_both_bounds() -> None:
    p = parameter("uniform_int", 6, 10)
    rng = generator()
    values = {draw(rng, p) for _ in range(400)}
    assert values == {6, 7, 8, 9, 10}
    assert all(isinstance(v, int) for v in values)


def test_log_uniform_int_floors_the_exponential_and_stays_in_bounds() -> None:
    p = parameter("log_uniform_int", 20, 2000)
    z = generator().uniform(np.log(20), np.log(2001))
    assert draw(generator(), p) == min(math.floor(float(np.exp(z))), 2000)
    rng = generator(3)
    values = [draw(rng, p) for _ in range(2000)]
    assert min(values) >= 20
    assert max(values) <= 2000
    assert all(isinstance(v, int) for v in values)


def test_scaled_uniform_int_gives_multiples_of_its_factor() -> None:
    p = parameter("scaled_uniform_int", 8, 64, factor=8)
    rng = generator()
    values = [draw(rng, p) for _ in range(3000)]
    assert all(v % 8 == 0 and 64 <= v <= 512 for v in values)
    assert min(values) == 64  # both ends are reachable
    assert max(values) == 512


@pytest.mark.parametrize("kind", ["mixture_uniform", "mixture_log_uniform"])
def test_a_mixture_draws_its_branch_then_its_value_and_always_both(kind: str) -> None:
    p = parameter(kind, 0.1, 10.0)
    logarithmic = kind == "mixture_log_uniform"
    seen = set()
    for seed in range(200):
        rng, mirror = generator(seed), generator(seed)
        w = mirror.uniform(0.0, 1.0)
        z = mirror.uniform(np.log(0.1), np.log(10.0)) if logarithmic else mirror.uniform(0.1, 10.0)
        value = draw(rng, p)
        expected = 0.0 if w < 0.5 else (float(np.exp(z)) if logarithmic else float(z))
        assert value == expected
        assert rng.uniform() == mirror.uniform()  # the same stream position on either branch
        seen.add(value == 0.0)
    assert seen == {True, False}  # both branches occurred


def test_unknown_distributions_are_refused() -> None:
    with pytest.raises(ValueError, match="Unknown distribution"):
        draw(generator(), parameter("normal", 0.0, 1.0))


def test_the_reference_configuration_uses_no_draw(declaration: TrainingDeclaration) -> None:
    space = declaration.space("lightgbm")
    assert reference_candidate(space.parameters) == {
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "lambda_l2": 0.0,
    }


# --- the lists -----------------------------------------------------------------------------------


def test_every_model_gets_ten_candidates_starting_at_its_reference(
    declaration: TrainingDeclaration,
) -> None:
    lists = generate_candidates(declaration)
    assert tuple(lists) == declaration.streams
    for model, listed in lists.items():
        parameters = declaration.space(model).parameters
        assert len(listed.candidates) == declaration.trials == 10
        assert listed.candidates[0] == reference_candidate(parameters)
        assert all(list(c) == [p.name for p in parameters] for c in listed.candidates)


def test_random_candidates_lie_in_their_declared_spaces(declaration: TrainingDeclaration) -> None:
    for model, listed in generate_candidates(declaration).items():
        for candidate in listed.candidates[1:]:
            for p in declaration.space(model).parameters:
                value = candidate[p.name]
                if p.distribution.startswith("mixture"):
                    assert value == 0.0 or p.low <= value <= p.high, (model, p.name)
                elif p.distribution == "scaled_uniform_int":
                    assert p.low * (p.factor or 1) <= value <= p.high * (p.factor or 1)
                    assert value % (p.factor or 1) == 0
                else:
                    assert p.low <= value <= p.high, (model, p.name, value)
                if p.distribution in ("uniform_int", "log_uniform_int", "scaled_uniform_int"):
                    assert isinstance(value, int), (model, p.name)
                else:
                    assert isinstance(value, float), (model, p.name)


def test_the_lists_are_reproducible_and_depend_on_the_sampler_seed(
    declaration: TrainingDeclaration,
) -> None:
    first = generate_candidates(declaration)
    assert generate_candidates(declaration) == first
    other = generate_candidates(dataclasses.replace(declaration, sampler_seed=43))
    assert all(other[m].candidates[1:] != first[m].candidates[1:] for m in first)
    assert all(other[m].candidates[0] == first[m].candidates[0] for m in first)  # the reference


def test_random_candidates_are_distinct_within_a_list(declaration: TrainingDeclaration) -> None:
    for listed in generate_candidates(declaration).values():
        values = {tuple(c.values()) for c in listed.candidates}
        assert len(values) == len(listed.candidates)


def test_streams_are_children_of_one_parent_in_declared_order(
    declaration: TrainingDeclaration,
) -> None:
    children = streams(declaration)
    assert [child.spawn_key for child in children] == [(0,), (1,), (2,), (3,), (4,)]
    assert {child.entropy for child in children} == {42}
    assert [
        (listed.stream_index, listed.spawn_key)
        for listed in generate_candidates(declaration).values()
    ] == [(0, (0,)), (1, (1,)), (2, (2,)), (3, (3,)), (4, (4,))]


def test_the_fifth_stream_leaves_the_four_model_streams_unchanged() -> None:
    parent = np.random.SeedSequence(42)
    four = parent.spawn(4)
    after = parent.spawn(1)  # the counter continues: child (4,)
    five = np.random.SeedSequence(42).spawn(5)
    assert [c.spawn_key for c in five] == [c.spawn_key for c in [*four, *after]]
    for old, new in zip(four, five[:4], strict=True):
        assert np.random.default_rng(old).random(3).tolist() == (
            np.random.default_rng(new).random(3).tolist()
        )
    assert after[0].spawn_key == (4,)


def test_a_models_list_does_not_depend_on_the_other_models(
    declaration: TrainingDeclaration,
) -> None:
    """Changing one model's space (and so its number of draws) leaves every other list alone."""
    base = generate_candidates(declaration)
    lightgbm = declaration.space("lightgbm")
    shorter = dataclasses.replace(lightgbm, parameters=lightgbm.parameters[:2])
    changed = dataclasses.replace(declaration, models={**declaration.models, "lightgbm": shorter})
    other = generate_candidates(changed)
    assert other["lightgbm"] != base["lightgbm"]
    for model in ("catboost", "resnet", "ft_transformer", "logistic_regression"):
        assert other[model] == base[model]


def test_the_logistic_regression_list_has_its_declared_range(
    declaration: TrainingDeclaration,
) -> None:
    candidates = generate_candidates(declaration)["logistic_regression"].candidates
    assert candidates[0] == {"C": 1.0}
    assert all(1e-4 <= c["C"] <= 1e4 for c in candidates)
    assert len({c["C"] for c in candidates}) == 10
