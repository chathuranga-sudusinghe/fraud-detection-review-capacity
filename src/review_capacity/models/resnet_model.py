"""ResNet-style MLP of DR-011 (REF-087, Section 3.2 and Appendix F.1), following its reference
implementation (REF-103, ``bin/resnet.py`` at the pinned commit) with ReLU activations and batch
normalisation, the configuration of the paper's architecture:

* **Input:** each categorical input's index is embedded (one table for all categorical inputs,
  addressed through per-input offsets, initialised as in REF-103), and the embeddings are
  concatenated with the numeric inputs;
* **Network:** ``Linear`` to the layer size ``d``, then ``n_layers`` residual blocks
  ``x + Dropout(Linear(Dropout(ReLU(Linear(BatchNorm(x))))))`` with hidden size
  ``int(d * d_hidden_factor)``, then the prediction ``Linear(ReLU(BatchNorm(x)))`` to one logit;
* **Hyperparameters:** ``n_layers``, ``d``, ``d_hidden_factor``, ``hidden_dropout``,
  ``residual_dropout`` and, when there are categorical inputs, the embedding size
  ``d_embedding``: the parameters of the paper's search space (Table 14), named as in REF-103.
  Their values and ranges are fixed in the training rules. Batch normalisation keeps PyTorch's
  constants (``eps`` 1e-5, ``momentum`` 0.1), as REF-103, set explicitly.

Training, prediction and persistence are the shared neural ones (``models.neural_training``),
including REF-103's optimiser setting for this model: weight decay on every parameter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar

import torch
from torch import Tensor, nn
from torch.nn import functional

from review_capacity.errors import SpecificationError
from review_capacity.models.neural_training import NeuralModel, positive_integer, real_number

BATCHNORM_EPS = 1e-5
BATCHNORM_MOMENTUM = 0.1


class ResNetBlock(nn.Module):
    """``x + Dropout(Linear(Dropout(ReLU(Linear(BatchNorm(x))))))`` (REF-087, Equation 2)."""

    def __init__(
        self, d: int, d_hidden: int, *, hidden_dropout: float, residual_dropout: float
    ) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(d, eps=BATCHNORM_EPS, momentum=BATCHNORM_MOMENTUM)
        self.linear0 = nn.Linear(d, d_hidden)
        self.linear1 = nn.Linear(d_hidden, d)
        self.hidden_dropout = hidden_dropout
        self.residual_dropout = residual_dropout

    def forward(self, x: Tensor) -> Tensor:
        z: Tensor = self.linear0(self.norm(x))
        z = functional.relu(z)
        if self.hidden_dropout:
            z = functional.dropout(z, self.hidden_dropout, self.training)
        z = self.linear1(z)
        if self.residual_dropout:
            z = functional.dropout(z, self.residual_dropout, self.training)
        return x + z


class ResNet(nn.Module):
    """The ResNet-style MLP: one fraud logit per row from numeric inputs and category indices."""

    category_offsets: Tensor

    def __init__(
        self,
        *,
        n_numeric: int,
        cardinalities: tuple[int, ...],
        d_embedding: int | None,
        d: int,
        d_hidden_factor: float,
        n_layers: int,
        hidden_dropout: float,
        residual_dropout: float,
    ) -> None:
        super().__init__()
        d_in = n_numeric
        self.category_embeddings: nn.Embedding | None = None
        if cardinalities:
            if d_embedding is None:
                raise ValueError("Categorical inputs need an embedding size")
            d_in += len(cardinalities) * d_embedding
            offsets = torch.tensor((0, *cardinalities[:-1]), dtype=torch.int64).cumsum(0)
            self.register_buffer("category_offsets", offsets)
            self.category_embeddings = nn.Embedding(sum(cardinalities), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
        d_hidden = int(d * d_hidden_factor)
        self.first_layer = nn.Linear(d_in, d)
        self.blocks = nn.ModuleList(
            ResNetBlock(
                d, d_hidden, hidden_dropout=hidden_dropout, residual_dropout=residual_dropout
            )
            for _ in range(n_layers)
        )
        self.last_normalization = nn.BatchNorm1d(d, eps=BATCHNORM_EPS, momentum=BATCHNORM_MOMENTUM)
        self.head = nn.Linear(d, 1)

    def forward(self, numeric: Tensor, categorical: Tensor) -> Tensor:
        parts = [numeric]
        if self.category_embeddings is not None:
            embedded: Tensor = self.category_embeddings(categorical + self.category_offsets[None])
            parts.append(embedded.view(categorical.size(0), -1))
        x: Tensor = self.first_layer(torch.cat(parts, dim=-1))
        for block in self.blocks:
            x = block(x)
        x = functional.relu(self.last_normalization(x))
        logits: Tensor = self.head(x)
        return logits.squeeze(-1)


class ResNetModel(NeuralModel):
    """The ResNet-style MLP configurations RES-u and RES-b."""

    MODEL: ClassVar[str] = "resnet"
    ARCHITECTURE: ClassVar[tuple[str, ...]] = (
        "n_layers",
        "d",
        "d_hidden_factor",
        "hidden_dropout",
        "residual_dropout",
        "d_embedding",
    )
    MIN_BATCH_ROWS: ClassVar[int] = 2  # batch normalisation cannot normalise a single row
    ACCUMULATES: ClassVar[bool] = False  # its batch normalisation needs the whole effective batch

    def check_architecture(self, values: Mapping[str, Any]) -> dict[str, Any]:
        has_categories = bool(self.layout.categorical)
        required = [name for name in self.ARCHITECTURE if name != "d_embedding"]
        if has_categories:
            required.append("d_embedding")
        elif "d_embedding" in values:
            raise SpecificationError("d_embedding is set, but there are no categorical inputs")
        missing = [name for name in required if name not in values]
        if missing:
            raise SpecificationError(f"Hyperparameters must be set explicitly: {missing}")
        d = positive_integer(values["d"], "d")
        factor = real_number(values["d_hidden_factor"], "d_hidden_factor", open_low=True)
        if int(d * factor) < 1:
            raise SpecificationError("d * d_hidden_factor must leave a hidden size of at least 1")
        return {
            "n_layers": positive_integer(values["n_layers"], "n_layers"),
            "d": d,
            "d_hidden_factor": factor,
            "d_hidden": int(d * factor),
            "hidden_dropout": real_number(values["hidden_dropout"], "hidden_dropout", high=1.0),
            "residual_dropout": real_number(
                values["residual_dropout"], "residual_dropout", high=1.0
            ),
            "d_embedding": (
                positive_integer(values["d_embedding"], "d_embedding") if has_categories else None
            ),
        }

    def build_network(self, architecture: Mapping[str, Any]) -> nn.Module:
        return ResNet(
            n_numeric=len(self.layout.numeric),
            cardinalities=self.layout.cardinalities,
            d_embedding=architecture["d_embedding"],
            d=architecture["d"],
            d_hidden_factor=architecture["d_hidden_factor"],
            n_layers=architecture["n_layers"],
            hidden_dropout=architecture["hidden_dropout"],
            residual_dropout=architecture["residual_dropout"],
        )
