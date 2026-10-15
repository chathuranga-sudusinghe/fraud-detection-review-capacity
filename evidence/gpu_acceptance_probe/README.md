# GPU acceptance probe (Issue #77)

Records of the graphics processing unit (GPU) acceptance probe of [DR-013](../../docs/decisions/DR-013-training-device-policy.md) (Decision Record 13) §8, specified in [`docs/gpu-acceptance-probe.md`](../../docs/gpu-acceptance-probe.md). The probe is an engineering acceptance gate, not a research experiment: its inputs are synthetic, and no IEEE-CIS (IEEE Computational Intelligence Society) row, label or partition was read.

Each run has its own directory, written by `review-capacity probe-gpu`. The files are unchanged copies of the run's outputs.

## Runs of 15 October 2026: LightGBM CUDA failure

Two runs from commit `6973f49` (clean tree), the second given the first run's `probe.json`. Both stopped in the first LightGBM fit, then trained on the GPU through CUDA (NVIDIA's GPU computing platform): `LGB-u` (LightGBM, unweighted), 368,632 synthetic Train rows with early stopping on 100,000 synthetic Validation 1 (V1) rows.

| Directory | Started (Coordinated Universal Time, UTC) | Process ID |
|---|---|---|
| [`2026-10-15-run-1/`](2026-10-15-run-1/) | 2026-10-15 04:43:17 | 68502 |
| [`2026-10-15-run-2/`](2026-10-15-run-2/) | 2026-10-15 04:44:54 | 68731 |

**Native error, identical in both runs.** LightGBM wrote these lines directly to the terminal's standard error, not through the logger, and the process then aborted (exit code 134):

```text
[LightGBM] [Fatal] [CUDA] an illegal memory access was encountered .../src/treelearner/cuda/cuda_best_split_finder.cu 1811
[LightGBM] [Fatal] [CUDA] an illegal memory access was encountered .../src/io/cuda/cuda_tree.cpp 38
terminate called after throwing an instance of 'std::runtime_error'
  what():  [CUDA] an illegal memory access was encountered .../src/io/cuda/cuda_tree.cpp 38
```

**What the records show.** An abort inside LightGBM's native code ends the process without running the probe's failure handling. The records therefore stop at the last checkpoint written before the abort:

- `probe.json` and `probe.md`: stage `dataset identity`, overall result `in progress`; C4 (the CPU and GPU LightGBM Datasets, byte-identical) passed and C5 was recorded; every other check is `not run`;
- `probe.log`: the run log up to LightGBM's `Start training from score` line, without the native error lines above;
- the specification link in each `probe.md` is relative to the run's original output directory outside the repository, so it does not resolve here.

The native error lines are also recorded in [Issue #77](https://github.com/chathuranga-sudusinghe/fraud-detection-review-capacity/issues/77#issuecomment-6184658198).

**Cause and response.** The failure is a defect of LightGBM's CUDA learner on categorical features with more than 256 categories ([`references.md`](../../docs/references.md), REF-127; minimal reproduction on the project GPU in [Issue #101](https://github.com/chathuranga-sudusinghe/fraud-detection-review-capacity/issues/101#issuecomment-6185106140)). The DR-013 revision of Issue #101 moves LightGBM training to the central processing unit (CPU). The probe specification was revised for it, and the probe now writes a checkpoint before every fit, so that a native abort leaves the running fit in the record.

**SHA-256** (Secure Hash Algorithm, 256-bit) of the files:

| File | SHA-256 |
|---|---|
| `2026-10-15-run-1/probe.json` | `39dc23126c70a0f1e7332b09864fc5361828e0c9681ea6d7225dd3ffe9c8f068` |
| `2026-10-15-run-1/probe.log` | `aa5df75fe7531442c771899d7a8edbf08669d64719e849796cc592e67e2933a3` |
| `2026-10-15-run-1/probe.md` | `6be73232dade0297ca92465c87bb71e1bba1c4c9564b73ade17d751ef40415ca` |
| `2026-10-15-run-2/probe.json` | `6f21aacf4a0bff37f48b3cf85d91d2535726cd60b5b03183815b39b45df10b3b` |
| `2026-10-15-run-2/probe.log` | `6e0999e1d01920f0806daa0b92d2322f8525abd5af5663627b875a68b75c5f9a` |
| `2026-10-15-run-2/probe.md` | `f339cb36d70f28b2765ad687649f7ef9b37f71980e9d61b5f6377c34e0ed5e1b` |

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| C4, C5 | Probe checks: LightGBM Dataset check; categorical behaviour ([specification](../../docs/gpu-acceptance-probe.md) Section 3) |
| CPU, GPU | Central processing unit; graphics processing unit |
| CUDA | NVIDIA's GPU computing platform |
| DR-013 | Decision Record 13: training-device policy |
| IEEE-CIS | IEEE Computational Intelligence Society (the dataset's source) |
| `LGB-u` | LightGBM, unweighted configuration |
| REF-127 | Reference identifier in `docs/references.md` |
| SHA-256 | Secure Hash Algorithm, 256-bit |
| Train, V1 | Model-fitting partition; Validation 1 (early stopping); here, synthetic |
| UTC | Coordinated Universal Time |
