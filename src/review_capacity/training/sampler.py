"""The candidate sampler of DR-015 §1, §6, §7 and §8 (Issue #118).

For each model it generates one ordered list of candidates before any fit:

* trial 1 is the model's reference configuration and uses no random draw;
* trials 2 to ``trials`` draw the parameters one after another, in the declaration's order, each
  with one scalar call of NumPy's ``Generator`` (no ``size`` argument), from the model's own
  child stream;
* the sampler is ``numpy.random.default_rng`` (PCG64) over the child ``SeedSequence`` spawned from
  ``SeedSequence(sampler_seed)``: child ``i`` belongs to entry ``i`` of the declaration's streams
  (the four models in DR-011 order, then the logistic-regression reference as child ``(4,)``).
  ``SeedSequence.spawn`` continues the parent's child counter, so adding the fifth stream leaves
  the four earlier lists unchanged;
* the two configurations of a model (unweighted and balanced) share its list.

Values are kept at full float64 precision. The recorded candidate list is the reproducibility
record: bit-for-bit regeneration in another software environment is not assumed (DR-015 §6).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from review_capacity.logging_setup import log_step
from review_capacity.training.declaration import (
    MIXTURES,
    Parameter,
    TrainingDeclaration,
)

logger = logging.getLogger(__name__)

type Candidate = dict[str, int | float]

MIXTURE_WEIGHT = 0.5  # the probability of the zero branch of a mixture (DR-015 §6)


@dataclass(frozen=True)
class CandidateList:
    """The ordered candidates of one model: trial 1 is the reference configuration."""

    model: str
    stream_index: int
    spawn_key: tuple[int, ...]
    candidates: tuple[Candidate, ...]


def draw(rng: np.random.Generator, parameter: Parameter) -> int | float:
    """One draw of ``parameter`` from ``rng``, by the formulas of DR-015 §6 and §7.

    A mixture draws its branch ``w`` and then the value ``z``, always both, so that the stream
    position does not depend on the branch.
    """
    low, high = parameter.low, parameter.high
    kind = parameter.distribution
    if kind == "uniform":
        return float(rng.uniform(low, high))
    if kind == "log_uniform":
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
    if kind == "uniform_int":
        return int(rng.integers(int(low), int(high), endpoint=True))
    if kind == "scaled_uniform_int":
        return int(rng.integers(int(low), int(high), endpoint=True)) * int(parameter.factor or 1)
    if kind == "log_uniform_int":
        z = rng.uniform(np.log(low), np.log(high + 1))
        return min(int(np.floor(np.exp(z))), int(high))
    if kind in MIXTURES:
        w = rng.uniform(0.0, 1.0)
        z = (
            rng.uniform(np.log(low), np.log(high))
            if kind == "mixture_log_uniform"
            else rng.uniform(low, high)
        )
        if w < MIXTURE_WEIGHT:
            return 0.0
        return float(np.exp(z)) if kind == "mixture_log_uniform" else float(z)
    raise ValueError(f"Unknown distribution {kind!r}")


def reference_candidate(parameters: tuple[Parameter, ...]) -> Candidate:
    """Trial 1: the reference configuration, with no random draw."""
    return {p.name: p.reference for p in parameters}


def streams(declaration: TrainingDeclaration) -> list[np.random.SeedSequence]:
    """The child streams of the sampler, one per entry of ``declaration.streams``."""
    parent = np.random.SeedSequence(declaration.sampler_seed)
    return parent.spawn(len(declaration.streams))


def generate_candidates(declaration: TrainingDeclaration) -> dict[str, CandidateList]:
    """The candidate list of every model, by model name."""
    lists: dict[str, CandidateList] = {}
    with log_step(logger, "Generating the candidate lists"):
        for index, child in enumerate(streams(declaration)):
            model = declaration.streams[index]
            parameters = declaration.space(model).parameters
            rng = np.random.default_rng(child)
            candidates: list[Candidate] = [reference_candidate(parameters)]
            candidates.extend(
                {p.name: draw(rng, p) for p in parameters} for _ in range(declaration.trials - 1)
            )
            lists[model] = CandidateList(
                model, index, tuple(int(k) for k in child.spawn_key), tuple(candidates)
            )
            logger.info("%s: %d candidates from stream %d", model, len(candidates), index)
    return lists
