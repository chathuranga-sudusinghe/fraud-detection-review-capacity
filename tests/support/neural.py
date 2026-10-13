"""Synthetic neural-network inputs for unit tests (no project data is read).

A small input layout in the form of DR-012 (numeric inputs including a missingness indicator, and
categorical indices with ``"__MISSING__"`` at index 0) and data whose labels depend on numeric and
categorical inputs. ``preserved_torch_state`` restores the PyTorch settings, the default dtype and
the random-generator states that a fit or a test changes (``configure_torch``,
``seed_everything``), including the CUDA generators once CUDA is initialised.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from review_capacity.models.neural_training import InputLayout, NeuralData, NeuralInputs

LAYOUT = InputLayout(
    numeric=("TransactionAmt", "V1", "V1_missing", "tod_sin"),
    categorical=("ProductCD", "DeviceType"),
    cardinalities=(5, 3),
    missing_indices=(0, 0),
    category_dropout=0.01,
)

TRAINING: dict[str, Any] = {
    "lr": 1e-2,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "eval_batch_size": 96,
    "max_epochs": 12,
    "patience": 2,
    "metric": "average_precision",
}

RESNET: dict[str, Any] = {
    "n_layers": 2,
    "d": 16,
    "d_hidden_factor": 2.0,
    "hidden_dropout": 0.1,
    "residual_dropout": 0.05,
    "d_embedding": 4,
}

HYPERPARAMETERS: dict[str, Any] = {**TRAINING, **RESNET}


def neural_inputs(rows: int, seed: int) -> NeuralInputs:
    rng = np.random.default_rng(seed)
    numeric = rng.normal(size=(rows, len(LAYOUT.numeric))).astype(np.float32)
    numeric[:, 2] = (rng.random(rows) < 0.2).astype(np.float32)  # a missingness indicator
    categorical = np.column_stack(
        [rng.integers(0, size, rows) for size in LAYOUT.cardinalities]
    ).astype(np.int64)
    return NeuralInputs(numeric=numeric, categorical=categorical)


def neural_data(rows: int, seed: int, *, fraud_weight: float = 1.0) -> NeuralData:
    """Inputs and labels drawn from a logistic model of two numeric inputs and one category;
    each fraud row weighs ``fraud_weight``, each legitimate row 1."""
    inputs = neural_inputs(rows, seed)
    rng = np.random.default_rng(seed + 1)
    amount, v1 = inputs.numeric[:, 0], inputs.numeric[:, 1]
    product = inputs.categorical[:, 0]
    logit = 2.0 * amount - 1.5 * v1 + 1.5 * (product == 3) - 1.5
    labels = (rng.random(rows) < 1.0 / (1.0 + np.exp(-logit))).astype(np.int8)
    weights = np.where(labels == 1, fraud_weight, 1.0).astype(np.float64)
    return NeuralData(inputs=inputs, labels=labels, weights=weights)


@contextmanager
def preserved_torch_state() -> Iterator[None]:
    """Restore PyTorch's precision and determinism settings, its default dtype and the Python,
    NumPy, PyTorch CPU and (if initialised) CUDA generator states on exit."""
    matmul = torch.backends.cuda.matmul
    settings = (
        matmul.fp32_precision,
        (
            matmul.allow_bf16_reduced_precision_reduction,
            matmul.allow_bf16_reduced_precision_reduction_split_k,
        ),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
    )
    generators = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_ready = torch.cuda.is_initialized()  # type: ignore[no-untyped-call]
    cuda_generators = torch.cuda.get_rng_state_all() if cuda_ready else None
    default_dtype = torch.get_default_dtype()
    try:
        yield
    finally:
        torch.set_default_dtype(default_dtype)
        matmul.fp32_precision = settings[0]
        matmul.allow_bf16_reduced_precision_reduction = settings[1]
        torch.use_deterministic_algorithms(settings[2], warn_only=settings[3])
        torch.backends.cudnn.benchmark = settings[4]
        random.setstate(generators[0])
        np.random.set_state(generators[1])
        torch.set_rng_state(generators[2])
        if cuda_generators is not None:
            torch.cuda.set_rng_state_all(cuda_generators)
