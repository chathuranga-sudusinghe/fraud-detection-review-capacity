"""The eight model configurations of DR-011: four models, each unweighted and with balanced class
weighting, in the fixed order used for deterministic tie-breaks (DR-010).

A configuration is written ``<model>-<weighting>``, for example ``LGB-u`` or ``FTT-b``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Family(StrEnum):
    """Model family of DR-011."""

    TREE = "tree"
    NEURAL = "neural"


class Weighting(StrEnum):
    """Class weighting of the protocol (Class-Imbalance Treatment)."""

    UNWEIGHTED = "u"
    BALANCED = "b"


@dataclass(frozen=True)
class ModelSpec:
    """One model of DR-011: its package name, its short label and its family."""

    name: str
    label: str
    family: Family


@dataclass(frozen=True)
class Configuration:
    """One of the eight model × weighting configurations."""

    model: ModelSpec
    weighting: Weighting

    @property
    def name(self) -> str:
        return f"{self.model.label}-{self.weighting.value}"

    @property
    def family(self) -> Family:
        return self.model.family


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("lightgbm", "LGB", Family.TREE),
    ModelSpec("catboost", "Cat", Family.TREE),
    ModelSpec("resnet", "RES", Family.NEURAL),
    ModelSpec("ft_transformer", "FTT", Family.NEURAL),
)

# DR-011 order: every model unweighted, then every model with balanced weighting.
CONFIGURATIONS: tuple[Configuration, ...] = tuple(
    Configuration(model, weighting)
    for weighting in (Weighting.UNWEIGHTED, Weighting.BALANCED)
    for model in MODELS
)


# The tree-based configurations, in DR-011 order: the staged first path of the development run.
TREE_CONFIGURATIONS: tuple[str, ...] = tuple(
    c.name for c in CONFIGURATIONS if c.family is Family.TREE
)


def configuration(name: str) -> Configuration:
    """The configuration called ``name`` (e.g. ``"Cat-b"``); ``ValueError`` if there is none."""
    for candidate in CONFIGURATIONS:
        if candidate.name == name:
            return candidate
    known = ", ".join(c.name for c in CONFIGURATIONS)
    raise ValueError(f"Unknown configuration {name!r}; expected one of {known}")
