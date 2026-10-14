"""FT-Transformer of DR-011 (REF-087, Section 3.3 and Appendix E.1–E.2), following its reference
implementation (REF-103, ``bin/ft_transformer.py`` at the pinned commit) in the configuration of
the paper: ReGLU feed-forward modules, PreNorm layers, eight attention heads, Kaiming
initialisation and feature biases.

* **Feature Tokenizer:** numeric input j becomes ``b_j + x_j * W_j``; categorical input j becomes
  ``b_j`` plus its row of one embedding table addressed through per-input offsets. The [CLS] token
  is the first token: a weight row multiplied by the constant 1, without bias (as REF-103; with no
  positional encoding, the position of a token has no effect). Weights, biases and embeddings are
  initialised with ``kaiming_uniform_(a=√5)``, as in REF-103.
* **Transformer:** ``n_layers`` PreNorm blocks
  ``x + Dropout(MHSA(LayerNorm(x)))`` and ``x + Dropout(FFN(LayerNorm(x)))``, without the first
  attention normalisation of the first block (REF-087, Section 3.3). The feed-forward module is
  ``Linear(Dropout(ReGLU(Linear(x))))`` with hidden size ``int(d_token * d_ffn_factor)``. The last
  block computes attention for the [CLS] query only, which gives the [CLS] output unchanged
  (REF-103). Prediction: ``Linear(ReLU(LayerNorm(x[CLS])))``.
* **Attention (DR-013 §4):** ``scaled_dot_product_attention``, the attention of REF-103
  (``softmax(Q Kᵀ / √d_head)`` with dropout on the attention weights, then ``· V``). On the GPU the
  backend of ``config/models.yaml`` (memory-efficient attention) is required with ``sdpa_kernel``
  for training and prediction; on the CPU (unit tests and the protocol's CPU timing only),
  PyTorch's CPU selection applies. Attention dropout is 0 outside training. The memory-efficient
  kernel of PyTorch 2.11 requires a head size divisible by 8 in bf16 on this GPU and, unlike
  flash attention, does not pad. Heads are therefore zero-padded to a multiple of 8 and the output
  is cut back, with the scale of the true head size: zero columns add nothing to ``Q Kᵀ``, so the
  attention is exactly that of the unpadded heads, for every ``d_token`` that REF-103's search
  draws (multiples of the 8 heads).
* **Hyperparameters:** ``n_layers``, ``d_token`` (a multiple of the 8 heads, as REF-103 samples
  it), ``d_ffn_factor``, ``attention_dropout``, ``ffn_dropout`` and ``residual_dropout``: the
  parameters of the paper's search space (Table 13), named as in REF-103. Their values and ranges
  are fixed in the training rules. Fixed, as in the paper (not tuned): 8 heads, ReGLU, PreNorm,
  Kaiming initialisation, feature biases. LayerNorm keeps PyTorch's ``eps`` 1e-5, set explicitly.
* **Weight decay (REF-087, Appendix E.2–E.3):** none for the Feature Tokenizer, the normalisation
  layers and all biases; the other weights decay. REF-103 selects these parameters by name, which
  leaves the weight of the final LayerNorm decayed; the rule stated in the paper is followed here.

Training, prediction and persistence are the shared neural ones (``models.neural_training``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import Any, ClassVar

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.attention import SDPBackend, sdpa_kernel

from review_capacity.errors import SpecificationError
from review_capacity.models.neural_training import NeuralModel, positive_integer, real_number

N_HEADS = 8  # REF-087, Appendix E.1: not tuned
HEAD_ALIGNMENT = 8  # head-size multiple required by memory-efficient attention in bf16
LAYERNORM_EPS = 1e-5
SDPA_BACKENDS = {"efficient": SDPBackend.EFFICIENT_ATTENTION}


def attention_backend(device_type: str, gpu_backend: SDPBackend) -> AbstractContextManager[Any]:
    """The attention backend rule of DR-013 §4: on CUDA, ``sdpa_kernel`` restricted to the
    configured backend (any other backend is then ineligible, and an input the backend cannot run
    stops with an error instead of falling back); on the CPU, PyTorch's own selection."""
    if device_type == "cuda":
        restricted: AbstractContextManager[Any] = sdpa_kernel(gpu_backend)
        return restricted
    return nullcontext()


class Tokenizer(nn.Module):
    """The Feature Tokenizer with the [CLS] token first: one token per input."""

    category_offsets: Tensor

    def __init__(self, n_numeric: int, cardinalities: tuple[int, ...], d_token: int) -> None:
        super().__init__()
        self.category_embeddings: nn.Embedding | None = None
        if cardinalities:
            offsets = torch.tensor((0, *cardinalities[:-1]), dtype=torch.int64).cumsum(0)
            self.register_buffer("category_offsets", offsets)
            self.category_embeddings = nn.Embedding(sum(cardinalities), d_token)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
        self.weight = nn.Parameter(torch.empty(n_numeric + 1, d_token))  # row 0: [CLS]
        self.bias = nn.Parameter(torch.empty(n_numeric + len(cardinalities), d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, numeric: Tensor, categorical: Tensor) -> Tensor:
        cls = torch.ones(numeric.shape[0], 1, dtype=numeric.dtype, device=numeric.device)
        tokens = self.weight[None] * torch.cat([cls, numeric], dim=1)[:, :, None]
        if self.category_embeddings is not None:
            embedded: Tensor = self.category_embeddings(categorical + self.category_offsets[None])
            tokens = torch.cat([tokens, embedded], dim=1)
        bias = torch.cat([self.bias.new_zeros(1, self.bias.shape[1]), self.bias])
        return tokens + bias[None]


class MultiheadAttention(nn.Module):
    """Multi-head self-attention with query, key, value and output projections; the projection
    biases start at zero and the weights keep PyTorch's default (Kaiming) initialisation."""

    def __init__(
        self, d_token: int, *, dropout: float, gpu_backend: SDPBackend, n_heads: int = N_HEADS
    ) -> None:
        super().__init__()
        self.query = nn.Linear(d_token, d_token)
        self.key = nn.Linear(d_token, d_token)
        self.value = nn.Linear(d_token, d_token)
        self.output = nn.Linear(d_token, d_token)
        for projection in (self.query, self.key, self.value, self.output):
            nn.init.zeros_(projection.bias)
        self.n_heads = n_heads
        self.d_head = d_token // n_heads
        self.d_padded = -(-self.d_head // HEAD_ALIGNMENT) * HEAD_ALIGNMENT
        self.dropout = dropout
        self.gpu_backend = gpu_backend

    def _heads(self, x: Tensor) -> Tensor:
        """``(batch, tokens, d_token)`` as ``(batch, heads, tokens, d_padded)``."""
        batch, tokens, _ = x.shape
        heads = x.reshape(batch, tokens, self.n_heads, self.d_head).transpose(1, 2)
        if self.d_padded == self.d_head:
            return heads
        return functional.pad(heads, (0, self.d_padded - self.d_head))

    def _attend(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        dropout = self.dropout if self.training else 0.0
        scale = 1.0 / math.sqrt(self.d_head)  # of the true head size, not the padded one
        with attention_backend(query.device.type, self.gpu_backend):
            return functional.scaled_dot_product_attention(
                query, key, value, dropout_p=dropout, scale=scale
            )

    def forward(self, x_query: Tensor, x_key_value: Tensor) -> Tensor:
        attended = self._attend(
            self._heads(self.query(x_query)),
            self._heads(self.key(x_key_value)),
            self._heads(self.value(x_key_value)),
        )
        batch, _, tokens, _ = attended.shape
        merged = attended[..., : self.d_head].transpose(1, 2).reshape(batch, tokens, -1)
        output: Tensor = self.output(merged)
        return output


class TransformerBlock(nn.Module):
    """One PreNorm block: attention and ReGLU feed-forward residual branches."""

    def __init__(
        self,
        d_token: int,
        d_hidden: int,
        *,
        first: bool,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        gpu_backend: SDPBackend,
    ) -> None:
        super().__init__()
        self.attention = MultiheadAttention(
            d_token, dropout=attention_dropout, gpu_backend=gpu_backend
        )
        self.linear0 = nn.Linear(d_token, 2 * d_hidden)  # ReGLU: value and gate halves
        self.linear1 = nn.Linear(d_hidden, d_token)
        self.norm1 = nn.LayerNorm(d_token, eps=LAYERNORM_EPS)
        # PreNorm without the first attention normalisation of the first block (REF-087, §3.3)
        self.norm0 = None if first else nn.LayerNorm(d_token, eps=LAYERNORM_EPS)
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout

    def _residual(self, x: Tensor, branch: Tensor) -> Tensor:
        if self.residual_dropout:
            branch = functional.dropout(branch, self.residual_dropout, self.training)
        return x + branch

    def forward(self, x: Tensor, *, cls_only: bool) -> Tensor:
        branch: Tensor = x if self.norm0 is None else self.norm0(x)
        branch = self.attention(branch[:, :1] if cls_only else branch, branch)
        if cls_only:
            x = x[:, :1]
        x = self._residual(x, branch)
        branch = self.linear0(self.norm1(x))
        value, gate = branch.chunk(2, dim=-1)
        branch = value * functional.relu(gate)
        if self.ffn_dropout:
            branch = functional.dropout(branch, self.ffn_dropout, self.training)
        branch = self.linear1(branch)
        return self._residual(x, branch)


class FTTransformer(nn.Module):
    """The FT-Transformer: one fraud logit per row from numeric inputs and category indices."""

    def __init__(
        self,
        *,
        n_numeric: int,
        cardinalities: tuple[int, ...],
        d_token: int,
        n_layers: int,
        d_ffn_factor: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        gpu_backend: SDPBackend,
    ) -> None:
        super().__init__()
        self.tokenizer = Tokenizer(n_numeric, cardinalities, d_token)
        d_hidden = int(d_token * d_ffn_factor)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_token,
                d_hidden,
                first=index == 0,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                residual_dropout=residual_dropout,
                gpu_backend=gpu_backend,
            )
            for index in range(n_layers)
        )
        self.last_normalization = nn.LayerNorm(d_token, eps=LAYERNORM_EPS)
        self.head = nn.Linear(d_token, 1)

    def forward(self, numeric: Tensor, categorical: Tensor) -> Tensor:
        x = self.tokenizer(numeric, categorical)
        last = len(self.blocks) - 1
        for index, block in enumerate(self.blocks):
            x = block(x, cls_only=index == last)
        x = functional.relu(self.last_normalization(x[:, 0]))
        logits: Tensor = self.head(x)
        return logits.squeeze(-1)


class FTTransformerModel(NeuralModel):
    """The FT-Transformer configurations FTT-u and FTT-b."""

    MODEL: ClassVar[str] = "ft_transformer"
    ARCHITECTURE: ClassVar[tuple[str, ...]] = (
        "n_layers",
        "d_token",
        "d_ffn_factor",
        "attention_dropout",
        "ffn_dropout",
        "residual_dropout",
    )

    def _gpu_backend(self) -> SDPBackend:
        name = self.settings.neural.gpu_attention_backend
        if name not in SDPA_BACKENDS:
            raise SpecificationError(f"Unsupported GPU attention backend {name!r}")
        return SDPA_BACKENDS[name]

    def check_architecture(self, values: Mapping[str, Any]) -> dict[str, Any]:
        missing = [name for name in self.ARCHITECTURE if name not in values]
        if missing:
            raise SpecificationError(f"Hyperparameters must be set explicitly: {missing}")
        d_token = positive_integer(values["d_token"], "d_token")
        if d_token % N_HEADS:
            raise SpecificationError(f"d_token must be a multiple of the {N_HEADS} heads")
        factor = real_number(values["d_ffn_factor"], "d_ffn_factor", open_low=True)
        if int(d_token * factor) < 1:
            raise SpecificationError("d_token * d_ffn_factor must leave a hidden size of 1 or more")
        d_head = d_token // N_HEADS
        return {
            "n_layers": positive_integer(values["n_layers"], "n_layers"),
            "d_token": d_token,
            "d_ffn_factor": factor,
            "attention_dropout": real_number(
                values["attention_dropout"], "attention_dropout", high=1.0
            ),
            "ffn_dropout": real_number(values["ffn_dropout"], "ffn_dropout", high=1.0),
            "residual_dropout": real_number(
                values["residual_dropout"], "residual_dropout", high=1.0
            ),
            "n_heads": N_HEADS,
            "d_head": d_head,
            "d_head_padded": -(-d_head // HEAD_ALIGNMENT) * HEAD_ALIGNMENT,
            "d_hidden": int(d_token * factor),
            "n_tokens": 1 + len(self.layout.numeric) + len(self.layout.categorical),
            "fixed": {
                "activation": "reglu",
                "prenormalization": True,
                "initialization": "kaiming",
                "token_bias": True,
            },
        }

    def build_network(self, architecture: Mapping[str, Any]) -> nn.Module:
        return FTTransformer(
            n_numeric=len(self.layout.numeric),
            cardinalities=self.layout.cardinalities,
            d_token=architecture["d_token"],
            n_layers=architecture["n_layers"],
            d_ffn_factor=architecture["d_ffn_factor"],
            attention_dropout=architecture["attention_dropout"],
            ffn_dropout=architecture["ffn_dropout"],
            residual_dropout=architecture["residual_dropout"],
            gpu_backend=self._gpu_backend(),
        )

    def parameter_groups(self, network: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
        """Weight decay for every weight except those of the Feature Tokenizer and the
        normalisation layers; no weight decay for the tokenizer, normalisation layers and biases
        (REF-087, Appendix E.2–E.3)."""
        if not isinstance(network, FTTransformer):
            raise TypeError(f"Expected an FTTransformer, not {type(network).__name__}")
        exempt = {id(parameter) for parameter in network.tokenizer.parameters()}
        for module in network.modules():
            if isinstance(module, nn.LayerNorm):
                exempt.update(id(parameter) for parameter in module.parameters())
        decayed: list[nn.Parameter] = []
        undecayed: list[nn.Parameter] = []
        for name, parameter in network.named_parameters():
            if id(parameter) in exempt or name.endswith(".bias"):
                undecayed.append(parameter)
            else:
                decayed.append(parameter)
        return [
            {"params": decayed, "weight_decay": weight_decay},
            {"params": undecayed, "weight_decay": 0.0},
        ]

    def _precision_record(
        self, effective: Mapping[str, Any], device: torch.device
    ) -> dict[str, Any]:
        """The shared precision record and the attention backend of DR-013 §4."""
        backend = self.settings.neural.gpu_attention_backend
        return {
            **super()._precision_record(effective, device),
            "gpu_attention_backend": backend,
            "attention": (
                f"scaled_dot_product_attention, {backend} backend required (sdpa_kernel)"
                if device.type == "cuda"
                else "scaled_dot_product_attention, PyTorch CPU backend selection"
            ),
        }
