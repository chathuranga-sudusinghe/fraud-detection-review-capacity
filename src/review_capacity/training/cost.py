"""The predicted search cost of the actual candidate lists (DR-015 §7; Issue #118).

The neural search dominates the study's compute: one full Train epoch of FT-Transformer takes
7.6 to 44.5 minutes depending on the candidate. DR-015 §7 therefore asks that the cost of the
actual candidate lists be computed from the sampler before any fit and recorded; no list is
redrawn and no candidate is rejected for its cost. Whether a prediction is feasible is decided
explicitly before any run: this module only predicts.

**FT-Transformer.** The epoch time of a candidate is its layer count times a per-layer time that
is linear in ``d_token`` between the two settings measured in acceptance-probe run 5 (default:
3 layers, ``d_token`` 192; largest: 6 layers, ``d_token`` 512), extrapolated linearly below 192.
Times are scaled to the study's partitions (Train x 22.50, V1 evaluation x 16.00), as in DR-015
§7. The search covers the 10 candidates of both configurations (the shared list is fitted twice),
for between ``patience + 2`` epochs (the earliest stop, with the best V1 value at epoch 1) and
the maximum. The refits of DR-015 §4 (three per configuration, fixed length, Train time only)
are estimated for the costliest candidate of the list at the maximum length, because the
selected candidate is unknown before the search.

**ResNet-style MLP.** Only its largest setting was measured (16 layers, ``d`` 1,024,
``d_embedding`` 64: 0.48 minutes per epoch), so that time is used for every candidate. It is an
estimate, not a bound: the declared ``d_embedding`` range reaches 512, which was not measured.

**Tree-based models and the logistic-regression reference.** DR-015 §7 gives no cost model for
them (a fit takes seconds to a few minutes), so they are recorded as not estimated.
"""

from __future__ import annotations

from dataclasses import dataclass

from review_capacity.models.registry import Weighting
from review_capacity.training.declaration import TrainingDeclaration
from review_capacity.training.sampler import CandidateList

CONFIGURATIONS_PER_MODEL = len(Weighting)  # unweighted and balanced share one list
REFITS_PER_CONFIGURATION = 3  # two additional seeds and one same-seed repeat (DR-015 §4)
MINUTES_PER_HOUR = 60.0


@dataclass(frozen=True)
class Anchor:
    """A measured setting of acceptance-probe run 5, scaled to the study's partitions: minutes per
    full Train epoch (``train``) and in all with the V1 evaluation (``total``)."""

    layers: int
    width: int
    train: float
    total: float


FTT_DEFAULT = Anchor(layers=3, width=192, train=6.95, total=7.64)  # DR-015 §7 table
FTT_LARGEST = Anchor(layers=6, width=512, train=40.1, total=44.5)
RESNET_LARGEST_TOTAL = 0.48  # 16 layers, d 1,024, d_embedding 64
RESNET_LARGEST_TRAIN = 0.42


def _layer_minutes(width: int, default: float, largest: float) -> float:
    """Minutes per layer and epoch at token width ``width``: linear between the anchors."""
    low, high = FTT_DEFAULT.width, FTT_LARGEST.width
    return default + (width - low) / (high - low) * (largest - default)


def ftt_epoch_minutes(layers: int, d_token: int) -> tuple[float, float]:
    """(train, total) minutes per epoch of an FT-Transformer candidate."""
    train = _layer_minutes(
        d_token, FTT_DEFAULT.train / FTT_DEFAULT.layers, FTT_LARGEST.train / FTT_LARGEST.layers
    )
    total = _layer_minutes(
        d_token, FTT_DEFAULT.total / FTT_DEFAULT.layers, FTT_LARGEST.total / FTT_LARGEST.layers
    )
    return layers * train, layers * total


@dataclass(frozen=True)
class ModelCost:
    """The predicted cost of one model's search and refits; ``estimated`` is false for a model
    without a cost model in DR-015 §7 (its other fields are then ``None``)."""

    model: str
    estimated: bool
    basis: str
    epochs_low: int | None = None
    epochs_high: int | None = None
    candidate_epoch_minutes: tuple[float, ...] | None = None
    search_hours_low: float | None = None
    search_hours_high: float | None = None
    refit_hours: float | None = None


def _not_estimated(model: str) -> ModelCost:
    return ModelCost(model, False, "no cost model in DR-015 §7 (fits take seconds to minutes)")


def _neural_cost(
    model: str,
    basis: str,
    train_minutes: tuple[float, ...],
    total_minutes: tuple[float, ...],
    declaration: TrainingDeclaration,
) -> ModelCost:
    stopping = declaration.neural_stopping
    low, high = stopping.patience + 2, stopping.maximum
    configurations = CONFIGURATIONS_PER_MODEL
    per_epoch = sum(total_minutes) * configurations / MINUTES_PER_HOUR
    refits = REFITS_PER_CONFIGURATION * configurations
    return ModelCost(
        model,
        True,
        basis,
        low,
        high,
        total_minutes,
        per_epoch * low,
        per_epoch * high,
        refits * high * max(train_minutes) / MINUTES_PER_HOUR,
    )


def predict_cost(
    declaration: TrainingDeclaration, lists: dict[str, CandidateList]
) -> tuple[ModelCost, ...]:
    """The predicted cost of every model's search, in the declaration's model order."""
    costs: list[ModelCost] = []
    for model in declaration.streams:
        if model == "ft_transformer":
            pairs = [
                ftt_epoch_minutes(int(c["n_layers"]), int(c["d_token"]))
                for c in lists[model].candidates
            ]
            costs.append(
                _neural_cost(
                    model,
                    "estimate: per-layer time linear in d_token between the measured default "
                    "(192) and largest (512) settings of acceptance-probe run 5, extrapolated "
                    "linearly for widths below 192",
                    tuple(train for train, _ in pairs),
                    tuple(total for _, total in pairs),
                    declaration,
                )
            )
        elif model == "resnet":
            count = len(lists[model].candidates)
            costs.append(
                _neural_cost(
                    model,
                    "estimate: every candidate at the measured largest setting (16 layers, "
                    "d 1,024, d_embedding 64); the effect of the declared d_embedding range up "
                    "to 512 and of the smaller candidates is not modelled (DR-015 §7)",
                    (RESNET_LARGEST_TRAIN,) * count,
                    (RESNET_LARGEST_TOTAL,) * count,
                    declaration,
                )
            )
        else:
            costs.append(_not_estimated(model))
    return tuple(costs)
