"""What a development run needs to know about each thing it fits: the eight configurations of
DR-011 and the logistic-regression reference (Issue #151).

The family says how a model trains and scores: ``tree`` (rounds, CPU prediction, canonical frames),
``neural`` (epochs, GPU, the network inputs of the shared preprocessing artifact) or ``reference``
(solver convergence, CPU, the artifact's float64 matrix). The reference has no ``Family``: it is
outside the eight configurations (DR-011 §5).
"""

from __future__ import annotations

from review_capacity.config import ModelSettings
from review_capacity.models import catboost_model, lightgbm_model, logistic_regression_model
from review_capacity.models.neural_training import STATE_FILE
from review_capacity.models.registry import Family, Subject

TREE = "tree"
NEURAL = "neural"
REFERENCE = "reference"

# The file each model saves its state in (the file a ``fit`` record links to).
MODEL_FILES: dict[str, str] = {
    "lightgbm": lightgbm_model.MODEL_FILE,
    "catboost": catboost_model.MODEL_FILE,
    "resnet": STATE_FILE,
    "ft_transformer": STATE_FILE,
    "logistic_regression": logistic_regression_model.ARRAYS_FILE,
}


def kind_of(subject: Subject) -> str:
    """``tree``, ``neural`` or ``reference``."""
    if subject.family is None:
        return REFERENCE
    return TREE if subject.family is Family.TREE else NEURAL


def training_device(settings: ModelSettings, subject: Subject) -> str:
    """The device the subject trains on (DR-013 §1); the reference trains on the CPU."""
    if subject.family is None:
        return "cpu"
    return settings.training_device(subject.model.name)


def scoring_device(settings: ModelSettings, subject: Subject) -> str:
    """The device the subject's selected model scores on (DR-013 §5): the GPU for the neural
    models, the CPU for the tree-based models and the reference."""
    return training_device(settings, subject) if kind_of(subject) == NEURAL else "cpu"


def study_device(subject: Subject, measured_on: str) -> str:
    """The device a timing record names (DR-013 §1, §5): the neural models' study device is the
    GPU whatever device a unit test runs them on; every other timing names the device it ran on."""
    return "gpu" if kind_of(subject) == NEURAL else measured_on
