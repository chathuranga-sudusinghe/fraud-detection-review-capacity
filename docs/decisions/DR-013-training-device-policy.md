# DR-013 — Training-device policy

| Field | Value |
|---|---|
| Status | Accepted (pre-freeze) by the project owner (Issue #66) in line with the University of Greenwich criteria ([criteria checklist](../greenwich-criteria-checklist.md)); acceptance gate: the graphics processing unit (GPU) probe, run by the project owner on synthetic data, confirms every setting below on the project hardware |
| Date | 2026-10-12 (Issue #66) |
| Evidence source | Official documentation and versioned source code of LightGBM 4.7.0, CatBoost 1.2.10, PyTorch 2.11 and cuBLAS (the CUDA Basic Linear Algebra Subprograms library) 12.8 [REF-116–REF-122]; Gorishniy et al. [REF-087]; the GPU memory probe of Issue #50 ([`evidence/gpu_probe/`](../../evidence/gpu_probe/README.md)). Two independent reviews of the proposal (Issue #66). No labels, model scores or Test-period data were used. |
| Related | [DR-006](DR-006-bootstrap-inference-strategy.md), [DR-008](DR-008-feature-declaration-and-model-input-policy.md), [DR-011](DR-011-model-set-tree-based-and-neural-families.md), [DR-012](DR-012-neural-network-input-representation.md), [protocol](../protocol.md) |

## Context

The eight configurations of [DR-011](DR-011-model-set-tree-based-and-neural-families.md) (LightGBM, CatBoost, ResNet-style multilayer perceptron (MLP) and Feature Tokenizer + Transformer (FT-Transformer), each unweighted and with balanced class weighting) are compared under one common procedure. The training device can change results, not only speed: GPU and central processing unit (CPU) implementations of the same library use different defaults, different arithmetic and, for CatBoost, different categorical statistics [REF-087, Appendix F.4; REF-118]. A device policy is therefore part of the comparison's fairness and is fixed before any model run.

## Decision

### 1. One device for all eight configurations

All eight configurations train on the local GPU (NVIDIA RTX 5060 Laptop, 8 gigabytes (GB), compute capability sm_120) through CUDA (NVIDIA's GPU computing platform). The device is chosen before any run and is not selected on performance.

**Rule.** Device-sensitive settings are controlled wherever equivalent controls exist on both devices; unavoidable implementation differences are documented and assessed descriptively. This controls avoidable differences in configuration without guaranteeing equal predictive performance across devices.

### 2. LightGBM

| Setting | Value | Reason |
|---|---|---|
| Device | `device_type="cuda"`, `gpu_device_id=0`, `num_gpu=1` | The CUDA implementation; the documented CUDA setting is double-precision histogram accumulation, as on CPU [REF-116] |
| Histogram bins | `max_bin=255` | The CPU-reference default; the GPU guide recommends fewer bins for speed [REF-116] |
| Unsupported-feature guards | `use_quantized_grad=False`; `linear_tree=False`; no forced splits | Quantised training with categorical features is rejected on GPU, and linear trees fall back to CPU under CUDA [REF-117]; forced-split support in the CUDA learner is not documented, so forced splits are not used |
| Categorical and missing values | As in [DR-008](DR-008-feature-declaration-and-model-input-policy.md) §5 | Categorical splits are implemented in the CUDA learner [REF-117] |
| Build | LightGBM built with CUDA (`cmake.define.USE_CUDA=ON`), run by the project owner; compiler, CUDA Toolkit, CMake options, target architecture and binary hash recorded | The standard wheel provides the Open Computing Language (OpenCL) `gpu` device only [REF-116]. Every run verifies at start-up that the installed binary is the recorded build |
| Determinism | Not available on GPU (`deterministic` is CPU-only) | Run-to-run variation is measured (§6) [REF-116] |

### 3. CatBoost

| Setting | Value | Reason |
|---|---|---|
| Device | `task_type="GPU"`, `devices="0"` | CatBoost's GPU implementation is CUDA; the Linux wheel includes it [REF-118] |
| Quantisation | `border_count=254` | The CPU default; the ordinary binary-classification GPU default is 128, allowing fewer candidate numerical borders [REF-118] |
| Bootstrap | `bootstrap_type="Bayesian"` on both devices | An owner-selected common setting, fixed before training and supported on both devices; the CPU default (MVS, Minimum Variance Sampling) differs from the GPU default (Bayesian) [REF-118]. Its associated settings are carried unchanged |
| Categorical statistics (CTRs) | Each device's own default set, declared explicitly for both `simple_ctr` and `combinations_ctr`: GPU Borders + FeatureFreq; CPU Borders + Counter | These are the native default sets of each implementation [REF-119]; restricting both devices to their shared types would remove a default capability from both. The difference is an implementation difference, documented here and assessed in §7 |
| Boosting | `boosting_type="Plain"` | The default on both devices above 50,000 training rows (Train has 368,632) [REF-118]. Ordered target statistics, part of CatBoost's categorical handling, remain in use |
| Parameter record | Resolved parameters (`get_all_params()`), including the full CTR descriptors (priors, binarisation), exported for every fitted model | Detects conditional defaults that a hand-written list could miss |
| Determinism | GPU training is non-deterministic: "the order of floating point summations is non-deterministic in this implementation" [REF-118] | Run-to-run variation is measured (§6) |

**Settings fixed in the training rules, explicitly and identically on both devices:** `learning_rate` (never the automatic, device-dependent value), `model_size_reg`, `fold_permutation_block`, the metric period and evaluation metric, `feature_border_type`, `score_function` and `grow_policy`, and the shared sampling controls that apply to Bayesian bootstrapping. Backend-only controls (for example the CPU-only `sampling_frequency`) and documented differences in their effect are recorded separately: CatBoost regularises feature combinations more aggressively on GPU, because combination cost is computed differently on the two devices [REF-118]. CatBoost's Train-only counter calculation (`counter_calc_method`, default `SkipTest`) is asserted, not assumed.

### 4. ResNet-style MLP and FT-Transformer

Both neural models use the same settings:

| Setting | Value | Reason |
|---|---|---|
| Device | The same GPU through CUDA (PyTorch 2.11, CUDA 12.8) | §1 |
| Training precision | bf16 (bfloat16) autocast; float32 parameters and optimiser state | The project's engineering choice for the 8 GB GPU (Issue #50 probe); not a setting of the primary source [REF-087] |
| Loss | Weighted binary cross-entropy from logits, computed in float32 | Recommended for autocast [REF-120] |
| Matrix-multiplication precision | `torch.backends.cuda.matmul.fp32_precision = "ieee"`; bf16 reduced-precision reduction set explicitly and recorded; old and new precision controls not mixed | Explicit rather than inherited defaults [REF-120] |
| Determinism | `torch.use_deterministic_algorithms(True, warn_only=False)`; `torch.backends.cudnn.benchmark = False`; every random generator seeded (PyTorch, NumPy, Python, the DataLoader generator and worker seeds, preprocessing noise, category dropout) | Strict mode selects the deterministic memory-efficient attention backward; warning-only mode keeps the non-deterministic path [REF-121] |
| Determinism exceptions | An operation with no usable deterministic implementation within 8 GB is an exception scoped to that operation and documented (operation, backend, warning, measured variation); any unrecognised determinism failure stops the run | Keeps every available deterministic path |
| Prediction | float32 with autocast disabled, in evaluation mode, for every score: Validation 1 (V1) early-stopping evaluation and the V1, Validation 2 (V2) and Test scores | Full score resolution avoids artificial ties at the daily top-K cutoff, where K is the fixed daily review budget; casting bf16 scores to float32 would not restore resolution |
| `CUBLAS_WORKSPACE_CONFIG` | Recorded as found (including unset), not set as a blanket rule | PyTorch 2.11 provides a separate cuBLAS workspace per stream, one of the documented reproducibility mechanisms [REF-121; REF-122] |

**FT-Transformer only:** the memory-efficient scaled dot-product attention backend, named in the configuration and recorded in each run; attention dropout set to 0 in evaluation, because the attention function always applies the dropout it is given [REF-120]. The model has 758 tokens (757 neural inputs of [DR-012](DR-012-neural-network-input-representation.md) and the [CLS] classification token).

### 5. Prediction backend for the tree-based models

Tree-based predictions used for study evaluation, in both the main study and the comparison of §7, are computed through each library's CPU prediction backend. This holds the prediction backend fixed across training devices. Prediction repeatability is checked for the same fitted model, inputs and recorded environment.

### 6. Reproducibility

Three sources of variation are reported separately, not as independent variance components:

1. **same-seed repeats:** numerical and backend non-determinism, measured in fresh processes with full model reinitialisation;
2. **different seeds:** stochastic training variation, from the seed list of the training rules;
3. **CPU vs GPU:** implementation differences plus residual training variation (§7).

Thread settings, software and build versions, resolved parameters, attention backends and all warnings are recorded with every run.

### 7. Supportive CPU–GPU comparison (tree-based models)

- **Design.** After GPU selection, each of the four tree configurations is refitted on CPU using the same seed, the GPU-selected shared hyperparameters and the GPU-fitted tree count, subject to the documented backend-specific settings in §3. No early stopping or best-model truncation is applied to the refit. LightGBM's CPU refit uses `deterministic=True` with one fixed histogram mode and the same build. CatBoost's CPU refit uses its CPU default categorical statistics (§3).
- **What it estimates.** The difference between these two specified training procedures, conditional on GPU selection. Documented backend-specific differences (CatBoost's categorical statistics and combination regularisation) remain; the comparison does not isolate hardware alone and does not estimate independently optimised CPU performance.
- **Evaluation.** Validation 2 (V2) only, under R3 (daily top-K), which needs no thresholds. Recall@2% (the share of frauds reviewed at 2 % review capacity) differences (CPU − GPU) are reported with the paired bootstrap of [DR-006](DR-006-bootstrap-inference-strategy.md), applied to fixed fraud-capture indicators with identical resampling weights for each pair and without reranking. Other metrics are reported as point differences only. Results are shown alongside the same-seed and different-seed variation (§6). A confidence interval that contains zero is not read as equivalence.
- **Status.** Descriptive only; outside the confirmatory hypotheses H1 (model family), H2 (class weighting) and H3 (alert-rate transfer fidelity) and their Holm family. Its result cannot change the device policy, model selection or any other decision. An implementation defect it reveals is corrected transparently.

### 8. GPU probe (acceptance gate)

A probe on synthetic data, prepared as a package command and run by the project owner, confirms on the project hardware, with acceptance criteria defined before it runs:

- **backend identity:** the intended GPU learner or kernel ran, with no silent CPU fallback and with ignored-option and unsupported-option warnings captured;
- **resolved settings:** the effective CatBoost and LightGBM parameters on both devices, and the LightGBM build identity;
- **categorical behaviour:** low and high cardinality, missing labels, unseen categories and categorical combinations;
- **all eight configurations,** with non-uniform sample weights;
- **neural computation:** embeddings, normalisation, category dropout, forward and backward passes, optimiser state and any checkpointing path, under strict determinism at 758 FT-Transformer tokens;
- **float32 evaluation path:** autocast disabled, with its attention backend and memory use;
- **numerical behaviour:** finite values, agreement with a float32 reference within the predefined tolerances, prediction repeatability, score ties and changes in top-K membership;
- **persistence:** prediction after saving and reloading a model;
- **resources and timing:** whole-device and host memory, and synchronised timing that separates start-up, preprocessing, fitting and prediction.

The probe fails visibly on an unsupported required capability; it never falls back to CPU, removes categorical inputs or reduces a model. It establishes engineering behaviour, not the size of variation on the research data. If it fails for a model, the project owner decides the response on that evidence.

## Evidence

- **[L]** GPU and CPU versions of CatBoost can give different results [REF-087, Appendix F.4]; CatBoost documents device-dependent defaults for `border_count`, bootstrap type and boosting type, non-deterministic GPU training and stronger GPU regularisation of feature combinations [REF-118], and builds different default categorical statistics on each device [REF-119].
- **[L]** LightGBM's CUDA implementation accumulates in double precision only; `deterministic` applies to CPU only; the standard wheel's GPU device is OpenCL, and CUDA needs a source build [REF-116]. The CUDA learner implements categorical splits but rejects quantised training with categorical features, and linear trees fall back to CPU [REF-117].
- **[L]** PyTorch documents that reproducibility is not guaranteed across releases, platforms or CPU and GPU [REF-120]; its attention backward selects a deterministic memory-efficient path only under strict deterministic mode [REF-121]. cuBLAS documents bit-wise reproducibility for a fixed toolkit version on GPUs with the same architecture and number of streaming multiprocessors, under its stated stream and routine conditions; separate per-stream workspaces are one documented mechanism for concurrent streams [REF-122].

## Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Tree-based models on CPU, neural models on GPU | Two devices in one comparison; the device would differ by family and confound the H1 family contrast with implementation differences |
| A CPU-or-GPU choice per model | A per-model device choice is a free design parameter that could be tuned on performance |
| Reduced tree quantisation resolution: LightGBM's recommended `max_bin=63` and CatBoost's GPU-default `border_count=128` | Rejected to retain the CPU-reference quantisation settings; predictive performance differences are not assumed in advance |
| LightGBM's OpenCL `gpu` device | Single precision by default; the CUDA implementation is documented as faster and uses double precision |
| A restricted common CatBoost CTR set on both devices (Borders, Buckets) | Removes each implementation's default counter statistic; a reduced configuration rather than the reference one |
| Warning-only determinism (`warn_only=True`) | Keeps the non-deterministic attention backward although a deterministic one is available under strict mode |
| Standard (math) attention for FT-Transformer | Deterministic but runs out of memory on 8 GB at the larger settings of the search space (Issue #50), which would restrict the model |
| bf16 prediction | bf16 scores lose resolution and can create artificial ties at the top-K cutoff |
| Independently tuned CPU models for the comparison | Would compare two tuning outcomes rather than two training procedures at the same selected settings |

## Rationale

Training all eight configurations on one GPU standardises the training hardware and keeps the family comparison free of a device confound. Matching equivalent device-sensitive controls limits avoidable configuration differences, while library-specific computation remains part of each model's implementation. Where determinism is unavailable, variation is measured and reported. The supportive comparison estimates how the specified CPU and GPU training procedures differ after GPU-based selection.

## Consequences

- The protocol's training procedure and [DR-011](DR-011-model-set-tree-based-and-neural-families.md) §3–§4 state GPU training for all eight configurations.
- LightGBM needs a CUDA build in the project environment, which the project owner installs and records; the lock file is updated accordingly.
- The training rules fix hyperparameter values and shared device-sensitive controls explicitly and identically across each CPU–GPU pair; backend-specific categorical statistics and controls remain as specified in §3.
- The four tree configurations are refitted once each on CPU for the supportive comparison; there is no additional search.
- The GPU probe is a package command with logging, replacing manual probing; the Issue #50 probe script remains unchanged as the record of its evidence.

## Threats and limitations

- **Hardware novelty.** Support for compute capability sm_120 in the LightGBM CUDA build and the CatBoost wheel is confirmed only by the probe.
- **Non-deterministic GPU training.** CatBoost on GPU, and any documented PyTorch exception, cannot be made bit-wise repeatable; their variation is measured and reported.
- **Implementation differences remain.** CatBoost's categorical statistics and combination regularisation differ between devices by design of the library; the supportive comparison reports their joint effect, not a hardware effect alone.
- **bf16 training.** A project engineering adaptation; float32 safeguards limit, but do not remove, numerical differences from float32 training.
- **Synthetic probe.** It confirms behaviour on the exercised paths; variation on the research data is measured in the development runs.

## References

REF-087, REF-116, REF-117, REF-118, REF-119, REF-120, REF-121, REF-122 — see [`../references.md`](../references.md).

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [L] | Evidence tag: literature or official documentation |
| bf16 | bfloat16: 16-bit floating-point format with the exponent range of float32 |
| [CLS] | Classification token whose final representation gives the FT-Transformer prediction |
| CPU, GPU | Central processing unit; graphics processing unit |
| CTR | CatBoost's categorical statistics computed from target and frequency information |
| CUDA | NVIDIA's GPU computing platform |
| cuBLAS | CUDA Basic Linear Algebra Subprograms library |
| DR, REF-NNN | Decision Record; reference identifier in `docs/references.md` |
| float32 | 32-bit floating-point format |
| FT-Transformer | Feature Tokenizer + Transformer model for tabular data |
| GB | Gigabyte |
| H1, H2, H3 | Confirmatory hypotheses: model family; class weighting; alert-rate transfer fidelity |
| K | Fixed daily review budget |
| MLP | Multilayer perceptron |
| MVS | Minimum Variance Sampling, CatBoost's CPU default bootstrap type |
| OpenCL | Open Computing Language: a vendor-neutral GPU programming interface |
| R3 | Decision policy: daily top-K review |
| Recall@2% | Share of frauds reviewed under daily top-K review at 2 % capacity |
| sm_120 | Compute capability of the project GPU (NVIDIA Blackwell generation) |
| Test | Sealed final evaluation partition (relative days 155–182) |
| Train | Model-fitting partition (relative days 1–105) |
| top-K | The K highest-scored transactions of a relative day |
| V1, V2 | Validation 1 (relative days 106–126); Validation 2 (relative days 127–154) |
