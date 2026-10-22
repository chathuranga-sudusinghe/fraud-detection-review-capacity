# GPU memory feasibility probe (Issue #50)

Engineering evidence for the graphics processing unit (GPU) memory limits quoted in [DR-011](../../docs/decisions/DR-011-model-set-tree-based-and-neural-families.md) (Decision Record 11) §4 and [DR-012](../../docs/decisions/DR-012-neural-network-input-representation.md). It is not a research experiment: the inputs are synthetic random tensors, no IEEE-CIS (IEEE Computational Intelligence Society) data, label or partition was read, and no research model was trained, tuned or selected.

## Files

| File | Content |
|---|---|
| [`gpu_probe_full.csv`](gpu_probe_full.csv) | One row per probed configuration (252 configurations), written by [`scripts/gpu_feasibility_probe.py`](../../scripts/gpu_feasibility_probe.py) with `--stage full` |
| [`gpu_probe_full.log`](gpu_probe_full.log) | The console log of the same run: the environment header, one line per configuration and the total run time |

Both files are unchanged copies of the run's outputs (`outputs/gpu_probe/`, which is not tracked by Git). SHA-256 (Secure Hash Algorithm, 256-bit) of `gpu_probe_full.csv`: `7040a81cf2fc0d72512192be383adcaf2bc5885c0b222929b58fb9579b1f9976`.

## Environment of the run

The log header records the environment:

- PyTorch 2.11.0, built for CUDA (NVIDIA's GPU computing platform) 12.8, on Python 3.12.3;
- NVIDIA GeForce RTX 5060 Laptop GPU, compute capability 12.0, 8,151 MiB (mebibytes) in total, 7,033 MiB free at the start;
- the PyTorch memory allocator capped at 6,521 MiB (the free memory minus a 512 MiB safety margin), so that an oversized configuration fails with an out-of-memory error instead of spilling into system memory.

The probe ran on 8 October 2026; the GPU, PyTorch, CUDA and memory figures are also recorded in the [Issue #50 results summary](https://github.com/chathuranga-sudusinghe/fraud-detection-review-capacity/issues/50#issuecomment-6071998608).

## Method

For each configuration, the probe tries batch sizes 64, 128, 256, 512, 1,024, 2,048 and 4,096, running up to three training steps at each size (forward pass, backward pass and an update by the AdamW optimiser, Adam with decoupled weight decay). It stops at the first out-of-memory error, runtime error or step longer than 30 seconds, or after completing the grid. None of the recorded configurations has a slow-step or other runtime-error status.

## Columns

| Column | Meaning |
|---|---|
| `model` | `ft_transformer` (FT-Transformer, Feature Tokenizer + Transformer) or `resnet` (ResNet-style MLP, a residual multilayer perceptron) |
| `tokens` | Number of model inputs (feature tokens), **excluding** the [CLS] classification token |
| `d` | Embedding size (FT-Transformer) or layer size (ResNet-style MLP) |
| `layers` | Number of Transformer layers or residual blocks |
| `precision` | `fp32` (32-bit floating point) or `bf16` (bfloat16 mixed precision) |
| `attention` | `math` (standard attention), `sdpa` (memory-efficient scaled dot-product attention) or `n/a` (not applicable: the ResNet-style MLP has no attention) |
| `grad_ckpt` | Whether gradient checkpointing was used |
| `max_batch` | Largest tested batch size that completed three training steps; 0 means that no tested batch size succeeded. For `ok` rows, the next doubled batch size failed with an out-of-memory error under this run's allocator cap |
| `peak_mib_at_max` | Peak allocated GPU memory at `max_batch`, in MiB |
| `step_ms_at_max`, `step_ms_at_512` | Mean elapsed training-step time at `max_batch` and at batch size 512, in milliseconds, over steps 2 and 3 (the first step is excluded as warm-up) |
| `epoch_min_at_max` | Synthetic training-time extrapolation: the number of batches needed for the 368,632 Train rows, rounded up, multiplied by the mean step time, in minutes. It excludes data loading, preprocessing and validation; no full epoch was measured |
| `params_m` | Number of model parameters, in millions |
| `status` | `ok`; `ok (cap reached)`: batch size 4,096 succeeded and larger sizes were not tested, so the maximum is unknown; `oom at smallest batch`: out of memory already at batch size 64 |

Measurements that are unavailable for a configuration are written as `nan` (not a number), not as zero. Of the 252 configurations, 229 are `ok`, 15 reached the cap and 8 are out of memory at the smallest batch size.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| AdamW | Adam optimiser with decoupled weight decay |
| bf16, fp32 | bfloat16 (16-bit) and 32-bit floating-point formats |
| [CLS] | Classification token of FT-Transformer |
| CUDA | NVIDIA's GPU computing platform |
| `d` | Embedding size (FT-Transformer) or layer size (ResNet-style MLP) |
| DR-NNN | Decision Record NNN in `docs/decisions/` |
| FT-Transformer | Feature Tokenizer + Transformer (a neural model for tabular data) |
| GPU | Graphics processing unit |
| IEEE-CIS | IEEE Computational Intelligence Society |
| MiB | Mebibyte (1,048,576 bytes) |
| MLP | Multilayer perceptron |
| `n/a` | Not applicable |
| `nan` | Not a number: the measurement is unavailable, not zero |
| oom | Out of memory |
| sdpa | Scaled dot-product attention (memory-efficient attention) |
| SHA-256 | Secure Hash Algorithm, 256-bit |
| Train | Model-fitting partition of DR-001 (relative days 1–105); its row count is used only for the time extrapolation |
