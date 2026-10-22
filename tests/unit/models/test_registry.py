"""The eight configurations of DR-011 in their fixed DR-010 order."""

from __future__ import annotations

import pytest

from review_capacity.models.registry import (
    CONFIGURATIONS,
    MODELS,
    TREE_CONFIGURATIONS,
    Family,
    Weighting,
    configuration,
)


def test_eight_configurations_in_the_fixed_order() -> None:
    assert [c.name for c in CONFIGURATIONS] == [
        "LGB-u",
        "Cat-u",
        "RES-u",
        "FTT-u",
        "LGB-b",
        "Cat-b",
        "RES-b",
        "FTT-b",
    ]


def test_two_models_per_family() -> None:
    assert [m.name for m in MODELS if m.family is Family.TREE] == ["lightgbm", "catboost"]
    assert [m.name for m in MODELS if m.family is Family.NEURAL] == ["resnet", "ft_transformer"]


def test_the_tree_configurations_are_the_staged_first_path_in_dr011_order() -> None:
    assert TREE_CONFIGURATIONS == ("LGB-u", "Cat-u", "LGB-b", "Cat-b")


def test_configuration_lookup() -> None:
    found = configuration("Cat-b")
    assert found.model.name == "catboost"
    assert found.family is Family.TREE
    assert found.weighting is Weighting.BALANCED


def test_unknown_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown configuration 'XGB-u'"):
        configuration("XGB-u")
