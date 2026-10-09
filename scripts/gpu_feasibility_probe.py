"""GPU memory feasibility probe for FT-Transformer and a ResNet-style MLP (GitHub Issue #50).

Engineering check only, not a research experiment:

- inputs are synthetic random tensors (GPU memory depends on shapes, not values);
- no IEEE-CIS data, labels or partitions are read;
- no research model is trained, tuned or selected; results decide nothing.

Architectures follow Gorishniy et al. (2021), "Revisiting Deep Learning Models for
Tabular Data", NeurIPS: main text Sections 3.2-3.3 and arXiv v5 Appendix E.1 / F.1
(PreNorm Transformer, ReGLU feed-forward, 8 heads, [CLS] token, first attention
normalisation removed; ResNet block with BatchNorm and two dropouts).

For each configuration the probe doubles the batch size until the GPU runs out of
memory and records the peak allocated memory and the time of one training step
(forward, backward, AdamW update).

The PyTorch allocator is capped below the free GPU memory. Without the cap, the
Windows/WSL NVIDIA driver may spill into system RAM ("sysmem fallback") instead of
raising out-of-memory, which makes oversized runs extremely slow rather than failing.

Usage (from the repository root):

    .venv/bin/python scripts/gpu_feasibility_probe.py --stage quick
    .venv/bin/python scripts/gpu_feasibility_probe.py --stage full
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

# Shapes of the current model input under DR-008: 382 numeric + 49 categorical columns.
N_NUMERIC = 382
N_CATEGORICAL = 49
# Assumed total number of categories over all categorical columns. Synthetic
# assumption (no data read); embedding tables are small next to attention memory.
TOTAL_CATEGORIES = 25_000
TRAIN_ROWS = (
    368_632  # Train partition size (DR-001), used only for epoch-time estimates
)
BATCHES = [64, 128, 256, 512, 1024, 2048, 4096]
SEED = 20261008
SAFETY_MARGIN_MIB = 512  # kept free below the memory available at start
SLOW_STEP_S = 30.0  # a training step slower than this is reported as "slow"


# ---------------------------------------------------------------- FT-Transformer
class FeatureTokenizer(nn.Module):
    """Numeric: T_j = b_j + x_j * W_j; categorical: T_j = b_j + Embedding(x_j)."""

    def __init__(self, n_num: int, n_cat: int, d: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_num, d))
        self.bias = nn.Parameter(torch.empty(n_num + n_cat, d))
        per_col = max(1, TOTAL_CATEGORIES // max(n_cat, 1))
        self.cat_embed = nn.Embedding(per_col * n_cat, d)
        self.register_buffer(
            "cat_offsets", torch.arange(n_cat) * per_col, persistent=False
        )
        for p in (self.weight, self.bias):
            nn.init.uniform_(p, -1 / math.sqrt(d), 1 / math.sqrt(d))

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        num = self.weight[None] * x_num[..., None]
        cat = self.cat_embed(x_cat + self.cat_offsets[None])
        return torch.cat([num, cat], dim=1) + self.bias[None]


class MultiHeadAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float, mode: str) -> None:
        super().__init__()
        self.n_heads, self.dropout, self.mode = n_heads, dropout, mode
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        q, k, v = (
            self.qkv(x)
            .view(b, t, 3, self.n_heads, d // self.n_heads)
            .permute(2, 0, 3, 1, 4)
        )
        p = self.dropout if self.training else 0.0
        if self.mode == "math":  # materialises the full (tokens x tokens) score matrix
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
            y = F.dropout(scores.softmax(dim=-1), p) @ v
        else:  # fused kernels only (flash / memory-efficient); no fallback to math
            with sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            ):
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class ReGLUFeedForward(nn.Module):
    def __init__(self, d: int, factor: float, dropout: float) -> None:
        super().__init__()
        hidden = int(d * factor)
        self.lin1 = nn.Linear(d, 2 * hidden)
        self.lin2 = nn.Linear(hidden, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        a, g = self.lin1(x).chunk(2, dim=-1)
        return self.lin2(self.drop(a * F.relu(g)))


class TransformerBlock(nn.Module):
    def __init__(self, d: int, first: bool, attn_mode: str) -> None:
        super().__init__()
        self.attn_norm = nn.Identity() if first else nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, n_heads=8, dropout=0.2, mode=attn_mode)
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = ReGLUFeedForward(d, factor=4 / 3, dropout=0.1)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_num: int,
        n_cat: int,
        d: int,
        n_layers: int,
        attn_mode: str,
        grad_ckpt: bool,
    ) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num, n_cat, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList(
            TransformerBlock(d, first=i == 0, attn_mode=attn_mode)
            for i in range(n_layers)
        )
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))
        self.grad_ckpt = grad_ckpt

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        x = self.tokenizer(x_num, x_cat)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
        for block in self.blocks:
            if self.grad_ckpt and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.head(x[:, 0]).squeeze(-1)


# ------------------------------------------------------------------ ResNet-style MLP
class ResNetBlock(nn.Module):
    def __init__(self, d: int, factor: float = 2.0) -> None:
        super().__init__()
        hidden = int(d * factor)
        self.net = nn.Sequential(
            nn.BatchNorm1d(d),
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, d),
            nn.Dropout(0.1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class ResNet(nn.Module):
    def __init__(
        self, n_num: int, n_cat: int, d: int, n_layers: int, emb: int = 16
    ) -> None:
        super().__init__()
        per_col = max(1, TOTAL_CATEGORIES // max(n_cat, 1))
        self.cat_embed = nn.Embedding(per_col * n_cat, emb)
        self.register_buffer(
            "cat_offsets", torch.arange(n_cat) * per_col, persistent=False
        )
        self.inp = nn.Linear(n_num + n_cat * emb, d)
        self.blocks = nn.Sequential(*(ResNetBlock(d) for _ in range(n_layers)))
        self.head = nn.Sequential(nn.BatchNorm1d(d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        cat = self.cat_embed(x_cat + self.cat_offsets[None]).flatten(1)
        return self.head(self.blocks(self.inp(torch.cat([x_num, cat], dim=1)))).squeeze(
            -1
        )


# ---------------------------------------------------------------------- measurement
@dataclass
class Config:
    model: str
    tokens: int  # number of model inputs (numeric + categorical); excludes [CLS]
    d: int
    layers: int
    precision: str  # fp32 | bf16
    attention: str  # math | sdpa | n/a
    grad_ckpt: bool


@dataclass
class Result(Config):
    max_batch: int
    peak_mib_at_max: float
    step_ms_at_max: float
    step_ms_at_512: float
    epoch_min_at_max: float
    params_m: float
    status: str


def split_tokens(tokens: int) -> tuple[int, int]:
    """Keep 49 categorical inputs; extra tokens (feature-engineering scenarios) are numeric."""
    return tokens - N_CATEGORICAL, N_CATEGORICAL


def build(cfg: Config) -> nn.Module:
    n_num, n_cat = split_tokens(cfg.tokens)
    if cfg.model == "ft_transformer":
        return FTTransformer(
            n_num, n_cat, cfg.d, cfg.layers, cfg.attention, cfg.grad_ckpt
        )
    return ResNet(n_num, n_cat, cfg.d, cfg.layers)


def free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def try_batch(cfg: Config, batch: int, steps: int = 3) -> tuple[str, float, float, int]:
    """Run `steps` training steps; return (status, peak MiB, mean step ms, params)."""
    free_gpu()
    torch.cuda.reset_peak_memory_stats()
    model = opt = x_num = x_cat = y = None
    try:
        model = build(cfg).cuda().train()
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        n_num, n_cat = split_tokens(cfg.tokens)
        per_col = max(1, TOTAL_CATEGORIES // n_cat)
        x_num = torch.randn(batch, n_num, device="cuda")
        x_cat = torch.randint(0, per_col, (batch, n_cat), device="cuda")
        y = (
            torch.rand(batch, device="cuda") < 0.035
        ).float()  # synthetic 3.5 % positives
        times = []
        for _ in range(steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=cfg.precision == "bf16"
            ):
                loss = F.binary_cross_entropy_with_logits(
                    model(x_num, x_cat).float(), y
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            if times[-1] > SLOW_STEP_S:
                return "slow", float("nan"), float("nan"), n_params
        peak = torch.cuda.max_memory_allocated() / 2**20
        step_ms = 1000 * sum(times[1:]) / max(len(times) - 1, 1)  # first step = warm-up
        return "ok", peak, step_ms, n_params
    except torch.OutOfMemoryError:
        return "oom", float("nan"), float("nan"), 0
    except RuntimeError as exc:  # e.g. no fused attention kernel for this dtype/device
        msg = str(exc).splitlines()[0][:80]
        return f"error: {msg}", float("nan"), float("nan"), 0
    finally:
        del model, opt, x_num, x_cat, y
        free_gpu()


def probe(cfg: Config) -> Result:
    best = None
    ms_512 = float("nan")
    status = "oom at smallest batch"
    for batch in BATCHES:
        st, peak, ms, n_params = try_batch(cfg, batch)
        if st != "ok":
            if st == "slow":
                status = f"step > {SLOW_STEP_S:.0f} s at batch {batch}"
            elif st != "oom":
                status = st
            break
        best = (batch, peak, ms, n_params)
        status = "ok"
        if batch == 512:
            ms_512 = ms
    if best is None:
        return Result(
            **asdict(cfg),
            max_batch=0,
            peak_mib_at_max=float("nan"),
            step_ms_at_max=float("nan"),
            step_ms_at_512=float("nan"),
            epoch_min_at_max=float("nan"),
            params_m=float("nan"),
            status=status,
        )
    batch, peak, ms, n_params = best
    epoch_min = math.ceil(TRAIN_ROWS / batch) * ms / 60_000
    return Result(
        **asdict(cfg),
        max_batch=batch,
        peak_mib_at_max=round(peak, 1),
        step_ms_at_max=round(ms, 1),
        step_ms_at_512=round(ms_512, 1),
        epoch_min_at_max=round(epoch_min, 2),
        params_m=round(n_params / 1e6, 3),
        status=status if batch < BATCHES[-1] else "ok (cap reached)",
    )


def configs(stage: str) -> list[Config]:
    if stage == "quick":
        return [
            Config("ft_transformer", 431, 192, 3, "fp32", "math", False),
            Config("ft_transformer", 431, 192, 3, "bf16", "sdpa", False),
            Config("ft_transformer", 431, 192, 3, "bf16", "sdpa", True),
            Config("resnet", 431, 256, 4, "fp32", "n/a", False),
        ]
    out: list[Config] = []
    dims, depths = [64, 128, 192, 256, 384, 512], [1, 2, 3, 4, 6]
    # Stage A: current input size, all precision / attention variants.
    for d in dims:
        for n_layers in depths:
            for prec, attn, ck in [
                ("fp32", "math", False),
                ("fp32", "sdpa", False),
                ("bf16", "sdpa", False),
                ("bf16", "sdpa", True),
            ]:
                out.append(Config("ft_transformer", 431, d, n_layers, prec, attn, ck))
    # Stage B: feature-engineering scenarios, memory-efficient variants only.
    for tokens in [600, 800]:
        for d in dims:
            for n_layers in depths:
                for ck in (False, True):
                    out.append(
                        Config(
                            "ft_transformer", tokens, d, n_layers, "bf16", "sdpa", ck
                        )
                    )
    # ResNet sanity check (Gorishniy search space upper ranges).
    for d in [256, 512, 1024]:
        for n_layers in [2, 4, 8, 16]:
            out.append(Config("resnet", 431, d, n_layers, "fp32", "n/a", False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["quick", "full"], default="quick")
    ap.add_argument("--out", type=Path, default=Path("outputs/gpu_probe"))
    ap.add_argument(
        "--budget-mib",
        type=int,
        default=None,
        help="allocator cap in MiB (default: free memory at start minus the safety margin)",
    )
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA GPU not available")

    torch.manual_seed(SEED)
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    budget = args.budget_mib or int(free / 2**20) - SAFETY_MARGIN_MIB
    torch.cuda.set_per_process_memory_fraction(budget * 2**20 / total)
    print(
        f"torch {torch.__version__} | CUDA {torch.version.cuda} | Python {platform.python_version()}"
    )
    print(
        f"GPU {props.name} | capability {props.major}.{props.minor} | "
        f"free {free / 2**20:.0f} / total {total / 2**20:.0f} MiB | "
        f"allocator cap {budget} MiB",
        flush=True,
    )

    cfgs = configs(args.stage)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"gpu_probe_{args.stage}.csv"
    fields = list(Result.__dataclass_fields__)
    started = time.perf_counter()
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, cfg in enumerate(cfgs, 1):
            res = probe(cfg)
            writer.writerow(asdict(res))
            fh.flush()
            print(
                f"[{i:3d}/{len(cfgs)}] {cfg.model:14s} tok={cfg.tokens} d={cfg.d:4d} "
                f"L={cfg.layers:2d} {cfg.precision} {cfg.attention:4s} "
                f"ckpt={'on ' if cfg.grad_ckpt else 'off'} -> max_batch={res.max_batch:5d} "
                f"peak={res.peak_mib_at_max:7.0f} MiB step={res.step_ms_at_max:7.1f} ms "
                f"epoch~{res.epoch_min_at_max:6.2f} min [{res.status}]",
                flush=True,
            )
    print(f"done in {(time.perf_counter() - started) / 60:.1f} min -> {path}")


if __name__ == "__main__":
    main()
