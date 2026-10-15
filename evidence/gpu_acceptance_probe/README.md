# GPU acceptance probe (Issue #77)

Records of the graphics processing unit (GPU) acceptance probe of [DR-013](../../docs/decisions/DR-013-training-device-policy.md) (Decision Record 13) §8, specified in [`docs/gpu-acceptance-probe.md`](../../docs/gpu-acceptance-probe.md). The probe is an engineering acceptance gate, not a research experiment: its inputs are synthetic, and no IEEE-CIS (IEEE Computational Intelligence Society) row, label or partition was read.

Each run has its own directory, written by `review-capacity probe-gpu`. The files are unchanged copies of the run's records (`probe.json`, `probe.md`, `probe.log`). Each run's comparison files (`scores/`, synthetic score arrays) stay in its local run directory under `outputs/gpu_acceptance_probe/`, which Git ignores. Each comparison file is identified in `probe.json` by the SHA-256 (Secure Hash Algorithm, 256-bit) of its canonical score bytes (little-endian 64-bit floats) and of its ordered synthetic row-identifier bytes (little-endian 64-bit integers); these are checked by loading the arrays, not by hashing the `.npz` file. C1–C13 identify the probe checks and T1–T4 its tolerances ([specification](../../docs/gpu-acceptance-probe.md) Sections 3 and 4).

**Result: the GPU acceptance gate of DR-013 §8 is passed** by acceptance runs 4 and 5 (below; [specification](../../docs/gpu-acceptance-probe.md) Section 7).

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

SHA-256 of the files:

| File | SHA-256 |
|---|---|
| `2026-10-15-run-1/probe.json` | `39dc23126c70a0f1e7332b09864fc5361828e0c9681ea6d7225dd3ffe9c8f068` |
| `2026-10-15-run-1/probe.log` | `aa5df75fe7531442c771899d7a8edbf08669d64719e849796cc592e67e2933a3` |
| `2026-10-15-run-1/probe.md` | `6be73232dade0297ca92465c87bb71e1bba1c4c9564b73ade17d751ef40415ca` |
| `2026-10-15-run-2/probe.json` | `6f21aacf4a0bff37f48b3cf85d91d2535726cd60b5b03183815b39b45df10b3b` |
| `2026-10-15-run-2/probe.log` | `6e0999e1d01920f0806daa0b92d2322f8525abd5af5663627b875a68b75c5f9a` |
| `2026-10-15-run-2/probe.md` | `f339cb36d70f28b2765ad687649f7ef9b37f71980e9d61b5f6377c34e0ed5e1b` |

## Run 3, 15 October 2026: GPU timeout-recovery failure

From commit `7bc3142` (clean tree), under the revised specification, with LightGBM on the CPU. Started 2026-10-15 07:28:43 UTC, process ID 125558; the laptop was on mains power and unattended.

- No pass/fail item failed before the interruption; the run was incomplete, with its C3 review items open. The largest unweighted Feature Tokenizer + Transformer configuration (`FTT-u`, largest) completed its first fit in about 363 s, and its C8 comparison recorded a maximum absolute logit difference of about 2.1e-6, within T1.
- During the C13 timing of that model, the process received `torch.AcceleratorError: CUDA error: an illegal memory access was encountered` and wrote its record: stage `failed`, overall result `failed`.
- At the same moment Windows stopped with bugcheck `0x116` (VIDEO_TDR_FAILURE: the display driver's timeout detection and recovery could not reset the GPU) and restarted. The Windows System log records event 1001, "The computer has rebooted from a bugcheck. The bugcheck was: 0x00000116 (…, …, 0xffffffffc000009a, 0x0000000000000004)", and event 41, an unclean shutdown; the system booted again at 13:19:03 (UTC+05:30). Details in [Issue #77](https://github.com/chathuranga-sudusinghe/fraud-detection-review-capacity/issues/77#issuecomment-6187485355).
- The failure occurred at the GPU device, driver and Windows level; its exact trigger is not recorded in the run log or the Windows event log. The same scoring path completed earlier in run 3 and in acceptance runs 4 and 5 under the same memory load, and the acceptance outcome rests on runs 4 and 5.
- The specification link in `probe.md` is relative to the run's original output directory outside the repository, so it does not resolve here.

| File | SHA-256 |
|---|---|
| `2026-10-15-run-3/probe.json` | `77841bf6d5e856e16771249f2270a6aa09a2735a55ff17c5bd59a92904e1a647` |
| `2026-10-15-run-3/probe.log` | `c369d9e5f6e78c9ae21a91a5b2c50d294fc25de4e5e9142d120b9f51f9958fab` |
| `2026-10-15-run-3/probe.md` | `7424422cf43a95d02eec41b2b9da2c9849d6f95272de64b063a878dd6c124c43` |

## Acceptance runs 4 and 5, 15 October 2026

From commit `7bc3142` (clean tree), under the revised specification. Both runs were attended, on mains power, with the display and sleep timeouts disabled. Run 5 ran in a fresh process and was given run 4's `probe.json`.

| Directory | Started (UTC) | Finished (UTC) | Process ID | Result |
|---|---|---|---|---|
| [`2026-10-15-run-4/`](2026-10-15-run-4/) | 2026-10-15 08:13:45 | 2026-10-15 09:06:25 | 9649 | `pending second run` (exit code 0) |
| [`2026-10-15-run-5/`](2026-10-15-run-5/) | 2026-10-15 09:24:35 | 2026-10-15 10:17:48 | 34666 | `review` (exit code 1) |

- **Checks:** C1, C2, C4–C13 pass in both runs. C3 confirms every fixed setting except CatBoost `sampling_unit`, which CatBoost 1.2.10 does not serialise; its review disposition (`Object`, CatBoost's documented default and the only scheme that applies to independent transactions) was accepted on 15 October 2026 ([Issue #77](https://github.com/chathuranga-sudusinghe/fraud-detection-review-capacity/issues/77#issuecomment-6190471082)).
- **C11 across the runs:** 74 comparisons. 62 are bit-for-bit identical: every LightGBM and neural score set, and all four LightGBM saved-model hashes. None failed. The 12 CatBoost comparisons are descriptive: the GPU fits of the unweighted CatBoost configuration (`Cat-u`) and the balanced one (`Cat-b`) show maximum differences of about 3.0e-5 and 3.6e-4, with no change in top-K membership (the K highest-scored rows, K as defined in specification C9), and the CPU refits are bit-for-bit identical.
- **Resources:** about 53 minutes per run; sampled whole-device GPU memory at most 6,788 of 8,151 MiB (mebibytes); no out-of-memory error and no monitoring problem.
- **`2026-10-15-run-4/gpu-monitor.csv`:** GPU telemetry read with `nvidia-smi` every 5 seconds during run 4, from 13:47:15 to 14:39:01 (UTC+05:30): 632 readings of temperature, power, clocks, memory, utilisation and active clock-event reasons (first line: column names). Maximum temperature 83 °C (degrees Celsius). The only active clock-event reason under load is `0x4`, software power capping at the 115 W (watt) limit reported by `nvidia-smi`.
- The specification link in run 4's `probe.md` is relative to its original output directory outside the repository; run 5's resolves here.

| File | SHA-256 |
|---|---|
| `2026-10-15-run-4/probe.json` | `27042e58e8713c29cbad56862d8fd686494dcf30e9ea0215db0f3595f6ec63e1` |
| `2026-10-15-run-4/probe.log` | `cc88bed5e951f483f325bdc6ad25e1e2ec9597efcbaa36183ac27e5284b71e9a` |
| `2026-10-15-run-4/probe.md` | `13597b0f1368e6961bbbb39c2a7c851302f79271566d70025bfdf681af5164b9` |
| `2026-10-15-run-4/gpu-monitor.csv` | `0d5ba6ab543aa73854cfd1ec3b7ae8cbdd2eaa77050ef4e6c1e40d19f391f81e` |
| `2026-10-15-run-5/probe.json` | `8449b3912741b25047e85a9007a959406ce0d3128737289ef6d3980e14925698` |
| `2026-10-15-run-5/probe.log` | `923bc7ae628780a99c5c4cb205a4d8ea94d10654101262dc6bd526628dabcc46` |
| `2026-10-15-run-5/probe.md` | `e9a0e83d8a7a61b3237994bf22a85bc59850c9cf220b20d77104fe75294dab0d` |

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| C1–C13, T1–T4 | Probe checks and tolerances ([specification](../../docs/gpu-acceptance-probe.md) Sections 3 and 4) |
| °C | Degree Celsius |
| CPU, GPU | Central processing unit; graphics processing unit |
| CUDA | NVIDIA's GPU computing platform |
| DR-013 | Decision Record 13: training-device policy |
| IEEE-CIS | IEEE Computational Intelligence Society (the dataset's source) |
| `LGB-u`, `Cat-u`, `Cat-b`, `FTT-u` | Configuration names: LightGBM, CatBoost or FT-Transformer (Feature Tokenizer + Transformer), unweighted (u) or with balanced class weighting (b) |
| K, top-K | Number of top-scored rows compared (specification C9); the K highest-scored rows |
| MiB | Mebibyte (2^20 bytes) |
| REF-127 | Reference identifier in `docs/references.md` |
| SHA-256 | Secure Hash Algorithm, 256-bit |
| Train, V1 | Model-fitting partition; Validation 1 (early stopping); here, synthetic |
| UTC | Coordinated Universal Time |
| W | Watt |
