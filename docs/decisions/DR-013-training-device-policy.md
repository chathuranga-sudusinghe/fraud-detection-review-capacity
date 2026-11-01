# DR-013 — Training-device policy

| Field | Value |
|---|---|
| Status | Frozen (accepted before the freeze), Issue #66, in line with the University of Greenwich criteria ([criteria checklist](../greenwich-criteria-checklist.md)); revised for the LightGBM training device (Issue #101); acceptance gate passed: the graphics processing unit (GPU) acceptance probe on synthetic data ([`gpu-acceptance-probe.md`](../gpu-acceptance-probe.md), Issue #77) confirmed every setting below on the project hardware |
| Date | 2026-10-12 (Issue #66); revised 2026-10-15 (Issue #101); 2026-10-22 (Issue #121: the additional neural CPU timing removed) |
| Evidence source | Official documentation and versioned source code of LightGBM 4.7.0, CatBoost 1.2.10, PyTorch 2.11 and cuBLAS (the Basic Linear Algebra Subprograms library of CUDA, NVIDIA's GPU computing platform) 12.8 [REF-116–REF-122]; LightGBM's issue reports, proposed fix and source code at its release tags for the CUDA defect of §2 [REF-127]; the scikit-build-core 1.1.1 source [REF-123]; Gorishniy et al. [REF-087]; the GPU memory probe of Issue #50 ([`evidence/gpu_probe/`](../../evidence/gpu_probe/README.md)); the project's LightGBM build; the GPU acceptance probe runs of Issue #77 and the minimal reproduction of Issue #101. Independent source-level reviews of the proposal and of the final configuration (Issue #66). No labels, model scores or Test-period data were used. |
| Related | [DR-006](DR-006-bootstrap-inference-strategy.md), [DR-008](DR-008-feature-declaration-and-model-input-policy.md), [DR-011](DR-011-model-set-tree-based-and-neural-families.md), [DR-012](DR-012-neural-network-input-representation.md), [protocol](../protocol.md) |

## Context

The eight configurations of [DR-011](DR-011-model-set-tree-based-and-neural-families.md) (LightGBM, CatBoost, ResNet-style multilayer perceptron (MLP) and Feature Tokenizer + Transformer (FT-Transformer), each unweighted and with balanced class weighting) are compared under one common procedure. The training device can change results, not only speed: GPU and central processing unit (CPU) implementations of the same library use different defaults, different arithmetic and, for CatBoost, different categorical statistics [REF-087, Appendix F.4; REF-118]. Library documentation and source code can also disagree about a device's behaviour [REF-116; REF-117]. A device policy is therefore part of the comparison's fairness and is fixed before any model run.

## Decision

### 1. Training devices

Six configurations train on the local GPU (NVIDIA RTX 5060 Laptop, 8 gigabytes (GB), compute capability sm_120) through CUDA (NVIDIA's GPU computing platform): CatBoost, the ResNet-style MLP and FT-Transformer, each unweighted and with balanced class weighting. The two LightGBM configurations train on the local central processing unit (CPU; Intel Core Ultra 9 275HX, 24 cores), because LightGBM's CUDA learner fails on the study's categorical features (§2). The devices are fixed before any run and are not selected on performance.

**Rule.** Device-sensitive settings are controlled wherever equivalent controls exist on both devices, and are set explicitly rather than inherited from defaults; unavoidable implementation differences are documented and assessed descriptively. This controls avoidable differences in configuration without guaranteeing equal predictive performance across devices.

### 2. LightGBM

| Setting | Value | Reason |
|---|---|---|
| Device | `device_type="cpu"`, `num_threads=24` (the 24 physical cores) | LightGBM's CUDA learner fails when a categorical feature has more than 256 categories. Training aborts with an illegal memory access or stops with an invalid-argument error, and the CPU device trains the same data [REF-127]. The pinned 4.7.0 learner aborted in the first LightGBM fit of the GPU acceptance probe (Issue #77), and the minimal reproduction of [REF-127] gives the same 256/257-category boundary on the project GPU (Issue #101). The code that the proposed upstream fix identifies (the categorical bitset sizing, and the global-memory split path selected when a feature has more than 256 histogram bins) is present at each inspected release tag, 4.0.0, 4.5.0, 4.6.0 and 4.7.0; the fix is unmerged and in no release [REF-127]. Eight of the 49 categorical features have more than 256 documented Train levels: `card1`, `DeviceInfo`, `card2`, `id_19`, `id_21`, `id_20`, `addr1` and `id_25` ([D] [feature catalog](../feature-catalog.md)). LightGBM's OpenCL `gpu` device rejects bins above 256, as reported in [REF-127]. For the best speed, LightGBM's documentation sets the thread count to the number of physical cores [REF-116] |
| Determinism | `deterministic=True` | Deterministic training is available on the CPU device only, and gives stable results for the same data, parameters, LightGBM version, compiled binary and system; the documentation asks for a fixed histogram mode with it [REF-116]. Same-seed training is therefore repeatable for the recorded build, which the acceptance probe confirms |
| Histogram bins | `max_bin=255`, `min_data_in_bin=3`, `bin_construct_sample_cnt=200000` | LightGBM's defaults, set explicitly. `max_bin` does not cap categorical features, which can have more bins [REF-117] |
| Dataset construction | `enable_bundle=False`, `is_enable_sparse=False`, `force_row_wise=False`, `force_col_wise=True`, `feature_pre_filter=False` | Exclusive Feature Bundling (EFB) allows a small number of sampled conflicts, so it is not lossless [REF-117]; without bundling, each feature forms its own group. Sparse optimisation [REF-116] is disabled with it, so that the stored layout does not depend on a data-dependent storage choice [I]. The histogram mode is fixed, because with neither mode forced LightGBM tries both and keeps the faster one, and deterministic training asks for a forced mode [REF-116]. Column-wise construction is the mode LightGBM's documentation recommends for many columns, many total bins (114,495 in the probe's synthetic Train Dataset, which follows the documented levels) and more than 20 threads, and it uses less memory than row-wise construction [REF-116]. `feature_pre_filter=False` keeps a reused Dataset valid when the search changes `min_data_in_leaf` |
| Missing values | `use_missing=True`, `zero_as_missing=False` | Native handling as in [DR-008](DR-008-feature-declaration-and-model-input-policy.md) §5, set explicitly |
| Seeds | `seed` only; the derived `data_random_seed`, `bagging_seed`, `feature_fraction_seed`, `extra_seed`, `drop_seed` and `objective_seed` are recorded | The master seed deterministically derives the subordinate seeds; their effective values are recorded, and explicitly supplied subordinate seeds would override this derivation [REF-117]. Streams that the selected algorithm does not use stay inactive. Seed values: §6 |
| Model-structure guards | `use_quantized_grad=False`; `linear_tree=False`; no forced splits and no forced bins (empty `forcedsplits_filename` and `forcedbins_filename`) | Library defaults, set explicitly so that none is inherited: quantised gradients, linear trees, forced splits and forced bins are not part of the model of [DR-011](DR-011-model-set-tree-based-and-neural-families.md) |
| Categorical values | As in [DR-008](DR-008-feature-declaration-and-model-input-policy.md) §5, with the 49 categorical columns named explicitly | LightGBM's native categorical splits [REF-117] |
| Histogram cache | `histogram_pool_size=-1`, recorded | The cache size for historical histograms; a negative value means no limit [REF-116] |
| Build | The recorded build is kept, and its CPU learner is used: LightGBM 4.7.0 built from source with `USE_CUDA=ON` for `CMAKE_CUDA_ARCHITECTURES=120`, with the CUDA Toolkit 12.8, GCC (the GNU Compiler Collection) 13.3.0 and NCCL (the NVIDIA Collective Communications Library) 2.26.2; build tools pinned (scikit-build-core 1.1.1, CMake 4.4.4, Ninja 1.13.2), with the CMake and Ninja versions also required through the build settings. The build record includes the paths and versions of the tools actually used, the generator and the relevant overrides (`CMAKE_EXECUTABLE`, `CMAKE_MAKE_PROGRAM`, `CMAKE_GENERATOR`, `CMAKE_ARGS`), including when unset. A build is accepted only if its actual generator is Ninja and its actual tool versions match the specified versions; conflicting overrides are rejected; the CUDA runtime and NCCL are linked statically; the library's SHA-256 (Secure Hash Algorithm, 256-bit) hash is recorded | The standard wheel provides the Open Computing Language (OpenCL) `gpu` device only [REF-116]. scikit-build-core otherwise uses any suitable CMake or Ninja found on the system [REF-123]. The build is not bit-for-bit reproducible (see Threats and limitations); every rebuild is recorded anew with its hash. Every run verifies at start-up that the installed binary is the recorded build. Keeping the build leaves the project environment unchanged; its CUDA learner is not used |

The settings are passed before the training Dataset is constructed, because construction settings cannot be changed afterwards [REF-117].

### 3. CatBoost

| Setting | Value | Reason |
|---|---|---|
| Device | `task_type="GPU"`, `devices="0"` | CatBoost's GPU implementation is CUDA; the Linux wheel includes it [REF-118]. CPU and GPU parameter sets are built separately from the shared settings, because an option unsupported on a device can be ignored, warned about or rejected, depending on the option [REF-119] |
| Quantisation | `border_count=254`, `feature_border_type="GreedyLogSum"`, `nan_mode="Min"` | `border_count` is the CPU default; the GPU default is 128, allowing fewer candidate numerical borders [REF-118; REF-119]. The other two are the defaults on both devices, set explicitly |
| Bootstrap | `bootstrap_type="Bayesian"` on both devices; `subsample` not set | A common setting, fixed before training and supported on both devices; the CPU default (MVS, Minimum Variance Sampling) differs from the GPU default (Bayesian) [REF-118]. `subsample` is rejected with Bayesian bootstrapping [REF-119] |
| Boosting | `boosting_type="Plain"` | The default on both devices above 50,000 training rows (Train has 368,632) [REF-118] |
| Object order | `has_time=True` on both devices, with the training Pool (CatBoost's data container) in time order | Ordered target-dependent statistics follow the supplied Train order ([DR-008](DR-008-feature-declaration-and-model-input-policy.md) §5) |
| Categorical statistics (CTRs) | Each device's own default set, declared explicitly as lists for both `simple_ctr` and `combinations_ctr`: GPU `["Borders", "FeatureFreq"]`; CPU `["Borders", "Counter"]` | These are the native default sets of each implementation [REF-119]; restricting both devices to their shared types would remove a default capability from both. The difference is an implementation difference, documented here and assessed in §7 |
| CTR controls | `max_ctr_complexity=4`, `one_hot_max_size=2`, `counter_calc_method="SkipTest"` | Defaults on both devices, set explicitly; `SkipTest` keeps counter statistics on Train only [REF-119] |
| Tree structure | `grow_policy="SymmetricTree"`, `score_function="Cosine"`, `random_score_type="NormalWithModelSizeDecrease"`, `leaf_estimation_method="Newton"`, `leaf_estimation_backtracking="AnyImprovement"`, `sampling_unit="Object"` | Valid on both devices; for binary log-loss the resolved leaf estimation is Newton on both devices, and GPU training requires this `random_score_type` [REF-119] |
| Feature bundling | Not used on GPU; CatBoost's CPU default kept | GPU quantisation disables CatBoost's exclusive feature bundling; on CPU, bundling at the default `sparse_features_conflict_fraction=0` admits no conflicting rows, so it changes storage, not information [REF-119] |
| Parameter record | Resolved parameters (`get_all_params()`), including the full CTR descriptors (priors, binarisation), plus the requested parameters, the model metadata (`params`, `output_options`) and the thread settings, saved for every fitted model | Detects conditional defaults that a hand-written list could miss; `get_all_params()` omits developer options and `thread_count` [REF-119] |
| Determinism | GPU training is non-deterministic: "the order of floating point summations is non-deterministic in this implementation" [REF-118] | Run-to-run variation is measured (§6) |

**Settings fixed in the training rules, explicitly and identically on both devices:** `learning_rate` (never the automatic, device-dependent value), `model_size_reg`, `fold_permutation_block` and `permutation_count` (with `has_time=True` no permutation is made: the CPU uses one learning fold whose block is the whole Train set, and the GPU uses the identity permutation, so both settings are inactive; they are still set explicitly and recorded, `fold_permutation_block` as a positive value because an explicit 0 resolves differently on the two devices [REF-119]), `random_strength`, `l2_leaf_reg`, `leaf_estimation_iterations`, `bagging_temperature`, the evaluation metric and `metric_period` (set explicitly, because GPU training changes the default period when a requested metric has no GPU implementation [REF-119]). Backend-only controls are recorded separately and not presented as shared settings: CPU-only `sampling_frequency`, `store_all_simple_ctr` and `ctr_leaf_count_limit`; GPU-only `gpu_cat_features_storage` and `ctr_history_unit` [REF-119]. CatBoost regularises feature combinations more aggressively on GPU, because combination cost is computed differently on the two devices; equal `model_size_reg` values therefore keep this difference [REF-118]. CatBoost's Train-only counter calculation is asserted, not assumed.

### 4. ResNet-style MLP and FT-Transformer

Both neural models use the same settings:

| Setting | Value | Reason |
|---|---|---|
| Device | The same GPU through CUDA (PyTorch 2.11, CUDA 12.8) | §1 |
| Training precision | bf16 (bfloat16) autocast; float32 (32-bit floating point) parameters and optimiser state | The project's engineering choice for the 8 GB GPU (Issue #50 probe); not a setting of the primary source [REF-087] |
| Loss | Weighted binary cross-entropy from logits, computed in float32, with float32 logits, labels and weights and an explicit reduction | Recommended for autocast [REF-120] |
| Matrix-multiplication precision | `torch.backends.cuda.matmul.fp32_precision = "ieee"`; `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` | Explicit rather than inherited defaults. The bf16 reduction control is separate from the float32 precision interface, so the two are not mixed [REF-120; REF-121]. Reduced-precision reduction in bf16 matrix products is on by default; switching it off removes one avoidable reduced-precision accumulation [REF-120] |
| Determinism | `torch.use_deterministic_algorithms(True, warn_only=False)`; `torch.backends.cudnn.benchmark = False`; every random generator seeded (PyTorch, NumPy, Python, the DataLoader generator and worker seeds, preprocessing noise, category dropout) | Strict mode selects the deterministic memory-efficient attention backward; warning-only mode keeps the non-deterministic path [REF-121] |
| Determinism exceptions | An operation with no usable deterministic implementation within 8 GB is an exception scoped to that operation and documented (operation, backend, warning, measured variation); any unrecognised determinism failure stops the run | Keeps every available deterministic path |
| Prediction | float32 with autocast disabled, in evaluation mode, for every score: Validation 1 (V1) early-stopping evaluation and the V1, Validation 2 (V2) and Test scores | Full score resolution avoids artificial ties at the daily top-K cutoff, where K is the fixed daily review budget; casting bf16 scores to float32 would not restore resolution |
| `CUBLAS_WORKSPACE_CONFIG` | Recorded as found (including unset), not set as a blanket rule; related workspace settings of cuBLASLt (cuBLAS's lightweight matrix-multiplication interface) recorded | PyTorch 2.11 provides a separate cuBLAS workspace per stream, one of the documented reproducibility mechanisms [REF-121; REF-122] |

**FT-Transformer only:**

- **Attention backend.** Memory-efficient scaled dot-product attention, required explicitly with `sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)` for GPU training and GPU study prediction, because enabling a backend leaves other backends eligible [REF-121]. The memory-efficient backend has no CPU implementation: on CPU, PyTorch considers only its CPU flash-attention and math backends [REF-121]. No study run executes a neural model on the CPU; the CPU path exists only in the synthetic unit tests and the synthetic acceptance probe, which use PyTorch's CPU backend selection. Attention dropout is 0 in evaluation, because the attention function always applies the dropout it is given [REF-120].
- **float32 attention arithmetic.** The memory-efficient kernel computes its float32 matrix products with a fast float32 multiply-add mode (`OpMultiplyAddFastF32`), not with the cuBLAS path that `fp32_precision` controls [REF-121]. The float32 prediction path is therefore described as float32 tensors through that kernel, and the GPU probe compares it with a float32 reference computed through the math attention backend (§8).
- **Tokens.** 758: the 757 neural inputs of [DR-012](DR-012-neural-network-input-representation.md) and the [CLS] classification token.

### 5. Prediction backend for the tree-based models

Tree-based scores used for study evaluation, in both the main study and the comparison of §7, are computed through each library's CPU prediction backend. This holds the prediction backend fixed across training devices:

- **LightGBM:** `Booster.predict` runs on the host predictor, which has no CUDA path [REF-117];
- **CatBoost:** `predict_proba(..., task_type="CPU")`, using probabilities rather than class labels [REF-119].

Prediction thread counts and the tree range (the fitted tree count) are set explicitly. Prediction repeatability is checked for the same fitted model, inputs and recorded environment. This rule governs the exported study scores; the evaluation that each library performs inside GPU training for early stopping is part of the GPU training procedure.

### 6. Reproducibility

All eight configurations use the **main seed 42**, fixed before any model run; every main result is computed with it. A common pre-specified list of additional seeds, identical for all configurations and fixed in the training rules, measures stochastic variation only: the additional seeds never replace the main seed and are not used for model selection.

Three sources of variation are reported separately, not as independent variance components:

1. **same-seed repeats:** numerical and backend non-determinism, measured in fresh processes with full model reinitialisation (for LightGBM's deterministic CPU training, a check that repeats are identical);
2. **different seeds:** stochastic training variation across the main and additional seeds, reported descriptively for every configuration and compared with the smallest effect size of interest (SESOI) of 2 percentage points of Recall@2% (the share of frauds reviewed under daily top-K review at 2 % capacity);
3. **CPU vs GPU (CatBoost):** implementation differences plus residual training variation (§7).

Thread settings, CPU affinity, the processors and memory limit visible to the Windows Subsystem for Linux (WSL), OpenMP (Open Multi-Processing, the CPU multithreading interface) settings, software and build versions, resolved parameters, attention backends and all warnings are recorded with every run.

### 7. Supportive CPU–GPU comparison (CatBoost)

- **Design.** After GPU selection, each of the two CatBoost configurations is refitted on CPU using the same seed, the GPU-selected shared hyperparameters and the GPU-fitted tree count, subject to the documented backend-specific settings in §3: its CPU default categorical statistics and `thread_count=24`. No early stopping or best-model truncation is applied to the refit. LightGBM trains on the CPU only (§2), so it has no CPU–GPU pair.
- **What it estimates.** The difference between these two specified training procedures, conditional on GPU selection. Documented implementation differences remain: CatBoost's categorical statistics and combination regularisation, and the order of floating-point summation. The comparison does not isolate hardware alone and does not estimate independently optimised CPU performance.
- **Evaluation.** Validation 2 (V2) only, under R3 (daily top-K), which needs no thresholds. Recall@2% (the share of frauds reviewed at 2 % review capacity) differences (CPU − GPU) are reported with the paired bootstrap of [DR-006](DR-006-bootstrap-inference-strategy.md), applied to fixed fraud-capture indicators with identical resampling weights for each pair and without reranking. Other metrics are reported as point differences only. Results are shown alongside the same-seed and different-seed variation (§6). A confidence interval that contains zero is not read as equivalence.
- **Status.** Descriptive only; outside the confirmatory hypotheses H1 (model family), H2 (class weighting) and H3 (alert-rate transfer fidelity) and their Holm family. Its result cannot change the device policy, model selection or any other decision. An implementation defect it reveals is corrected transparently.

### 8. GPU probe (acceptance gate)

A probe on synthetic data, prepared as a package command ([`gpu-acceptance-probe.md`](../gpu-acceptance-probe.md)), confirms on the project hardware, with acceptance criteria defined before it runs:

- **backend identity:** each configuration ran on its device of §1 with the intended learner or kernel, with no silent fallback to another device and with ignored-option and unsupported-option warnings captured;
- **resolved settings:** the requested and the effective LightGBM parameters (CPU) and CatBoost parameters (both devices), and the LightGBM build identity;
- **LightGBM determinism:** same-seed CPU training repeated in one process and across fresh processes gives identical models and scores;
- **categorical behaviour:** low and high cardinality, missing labels, unseen categories and categorical combinations;
- **all eight configurations,** with non-uniform sample weights;
- **neural computation:** embeddings, normalisation, category dropout, forward and backward passes, optimiser state and its data types and any checkpointing path, under strict determinism at 758 FT-Transformer tokens, with the required attention backend;
- **float32 evaluation path:** autocast disabled, with its attention backend and memory use, compared with a math-backend float32 reference on small synthetic batches within a tolerance defined in advance;
- **numerical behaviour:** finite values, agreement with a float32 reference within the predefined tolerances, prediction repeatability, score ties and changes in top-K membership;
- **prediction backend:** CPU prediction for the tree models, with the retained tree count;
- **persistence:** prediction after saving and reloading a model;
- **resources and timing:** whole-device and host memory, and synchronised timing that separates start-up, preprocessing, fitting and prediction, using the measurement procedure of the protocol's computational-cost outcome.

The probe fails visibly on an unsupported required capability; it never falls back to another device, removes categorical inputs or reduces a model. It establishes engineering behaviour, not the size of variation on the research data. A failure for a model is resolved on its evidence through an issue and a revision of this record before any development run, as for LightGBM's CUDA learner (§2; Issues #77 and #101).

## Evidence

- **[L]** GPU and CPU versions of CatBoost can give different results [REF-087, Appendix F.4]; CatBoost documents device-dependent defaults for `border_count`, bootstrap type and boosting type, non-deterministic GPU training and stronger GPU regularisation of feature combinations [REF-118]. Its source builds different default categorical statistics on each device, disables exclusive feature bundling on GPU, resolves `fold_permutation_block` and `metric_period` differently by device, makes no permutation when `has_time` is set, and omits developer options and `thread_count` from `get_all_params()` [REF-119].
- **[L]** LightGBM's parameter documentation states that the CUDA implementation supports double precision only; `deterministic` applies to CPU only, with a forced histogram mode; with neither histogram mode forced, both are tried and the faster kept; column-wise construction is recommended for many columns, many bins and more than 20 threads; the standard wheel's GPU device is OpenCL, and CUDA needs a source build [REF-116]. The 4.7.0 source selects single- or double-precision partial histograms on CUDA from `gpu_use_dp` (default false), permits sampled conflicts in feature bundling, groups features differently under GPU devices, disables sparse bundling and forces row-wise histograms under CUDA, and derives the subordinate seeds from `seed`. The CUDA learner implements categorical splits but rejects quantised training with categorical features, and linear trees fall back to CPU; prediction runs on the host predictor [REF-117].
- **[L]** LightGBM's CUDA learner fails when a categorical feature has more than 256 categories: an exact 256/257 boundary, reproduced on two GPU architectures, with an invalid-argument error or an aborting illegal memory access at `cuda_best_split_finder.cu:1811`; failures are also reported on 4.5.0 at category counts one greater than a positive multiple of 32, including 33, 65 and 97. The proposed fix attributes the boundary to an undersized categorical bitset and reports further defects across the CUDA path for features with more than 256 bins; it is unmerged and in no release. The bitset sizing and the global-memory split path are present at each inspected release tag: 4.0.0, 4.5.0, 4.6.0 and 4.7.0 [REF-127].
- The GPU acceptance probe aborted in the first LightGBM fit on the project GPU, with the illegal memory access of [REF-127] (Issue #77); the minimal reproduction of [REF-127] trained at 256 categories and failed at 257 with the invalid-argument error (Issue #101).
- **[L]** PyTorch documents that reproducibility is not guaranteed across releases, platforms or CPU and GPU [REF-120]; its attention backward selects a deterministic memory-efficient path only under strict deterministic mode, the memory-efficient kernel computes float32 products with a fast float32 multiply-add mode, and CPU attention selection considers only the CPU flash-attention and math backends [REF-121]. cuBLAS documents bit-wise reproducibility for a fixed toolkit version on GPUs with the same architecture and number of streaming multiprocessors, under its stated stream and routine conditions; separate per-stream workspaces are one documented mechanism for concurrent streams [REF-122].

## Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Tree-based models on CPU, neural models on GPU | Two devices in one comparison; the device would differ by family and confound the H1 family contrast with implementation differences. It would also move CatBoost, whose GPU learner has no such defect, off the GPU |
| A CPU-or-GPU choice per model | A per-model device choice is a free design parameter that could be tuned on performance. LightGBM's CPU device (§2) is not such a choice: it is fixed by a documented library defect before any run |
| LightGBM's CUDA learner (4.7.0) | Fails on the study's categorical features above 256 categories, by aborting or with an error [REF-127]; further defects are reported in the same path [REF-127] |
| An earlier LightGBM release | The defective code is present at each inspected release tag (4.0.0, 4.5.0, 4.6.0) [REF-127], and no release contains the proposed fix; releases before 4.0 use an earlier CUDA implementation, not tested here, and would replace the pinned environment |
| A LightGBM build with the proposed upstream fix | Unmerged, unreleased library code under research results [REF-127] |
| Capping or regrouping categories, or treating category codes as numbers, for LightGBM only | Changes the model inputs of one library: manual feature engineering against [DR-008](DR-008-feature-declaration-and-model-input-policy.md) and the rule of §8 |
| LightGBM row-wise histograms (`force_row_wise=True`) | Recommended for few total bins and at most 16 threads, and doubles the Dataset's memory cost [REF-116]; the study's inputs have many columns and bins, trained with 24 threads |
| Reduced tree quantisation resolution: LightGBM's recommended `max_bin=63` and CatBoost's GPU-default `border_count=128` | Rejected to retain the CPU-reference quantisation settings; predictive performance differences are not assumed in advance |
| LightGBM's OpenCL `gpu` device | Single precision by default; the CUDA implementation is documented as faster; it rejects bins above 256, as reported in [REF-127] |
| LightGBM's CUDA default histogram precision (`gpu_use_dp=False`) | Single-precision partial accumulation on GPU against double precision on CPU: an avoidable arithmetic difference with greater susceptibility to rounding error; its effect on observed variation is workload-dependent |
| LightGBM's default dataset construction (feature bundling, sparse optimisation, automatic histogram mode) | Approximate bundling that differs by device, and a CPU histogram mode chosen from a timing test; the construction differences would mix with the training-device comparison |
| LightGBM with only one of `enable_bundle=False` or `is_enable_sparse=False` | Either leaves part of the device-dependent grouping or the CPU histogram-mode choice in place |
| Explicit LightGBM subordinate seeds equal to `seed` | Replaces the library's derived seed streams with a project-specific scheme without a benefit; the derived values are recorded instead |
| A restricted common CatBoost CTR set on both devices (Borders, Buckets) | Removes each implementation's default counter statistic; a reduced configuration rather than the reference one |
| Switching off CatBoost's CPU feature bundling | It admits no conflicting rows at the default setting, so switching it off is unnecessary to prevent information loss; storage, runtime and possible numerical effects remain implementation details |
| Warning-only determinism (`warn_only=True`) | Keeps the non-deterministic attention backward although a deterministic one is available under strict mode |
| Standard (math) attention for FT-Transformer training | Deterministic but runs out of memory on 8 GB at the larger settings of the search space (Issue #50), which would restrict the model |
| Math attention for FT-Transformer prediction only | Would use a different attention implementation in prediction than in training; the memory-efficient kernel's float32 path is instead compared with a math-backend reference in the probe |
| bf16 prediction | bf16 scores lose resolution and can create artificial ties at the top-K cutoff |
| Independently tuned CPU models for the comparison | Would compare two tuning outcomes rather than two training procedures at the same selected settings |

## Rationale

Training six configurations on one GPU standardises the training hardware for CatBoost and both neural models. LightGBM trains on the CPU because the pinned 4.7.0 CUDA learner cannot train the study's categorical features, no release contains the proposed fix, and the alternatives that keep LightGBM on a GPU change its inputs or rest on unreleased code. Its CPU learner supports deterministic training [REF-116], so LightGBM's same-seed results are repeatable. The device is a fixed, documented property of each model's implementation, not a tuned choice: the inputs, native categorical handling, hyperparameter rules, selection procedure and evaluation are identical for all eight configurations. Matching equivalent device-sensitive controls, and setting them explicitly where documentation and source disagree, limits avoidable configuration differences, while library-specific computation remains part of each model's implementation. Where determinism is unavailable, variation is measured and reported. The supportive comparison estimates how CatBoost's specified CPU and GPU training procedures differ after GPU-based selection.

## Consequences

- The protocol's training procedure and [DR-011](DR-011-model-set-tree-based-and-neural-families.md) §3 state the training devices of §1.
- The recorded LightGBM build stays in the project environment: the CUDA Toolkit 12.8 and NCCL 2.26.2 remain installed on the system, and the build settings and build tools stay pinned in `pyproject.toml` ([environment setup](../environment-setup.md)). Its CUDA learner is not used.
- Without feature bundling, LightGBM training uses more memory and time; both are measured in the probe and in the computational-cost outcome of the protocol.
- LightGBM's hyperparameter search runs on the CPU; the search budget of the training rules is set from CPU fit times.
- The training rules fix hyperparameter values and shared device-sensitive controls explicitly and identically across each CatBoost CPU–GPU pair; backend-specific categorical statistics and controls remain as specified in §3.
- The two CatBoost configurations are refitted once each on CPU for the supportive comparison; there is no additional search.
- The GPU probe is a package command with logging, replacing manual probing; the Issue #50 probe script remains unchanged as the record of its evidence.

## Threats and limitations

- **Hardware novelty.** Support for compute capability sm_120 in the CatBoost wheel and in PyTorch is confirmed only by the probe.
- **Two devices within the tree family.** LightGBM trains on the CPU and CatBoost on the GPU, so the tree family's results for H1 combine two training devices. The device is fixed by a documented defect, recorded with every result and reported with the H1 results; its effect on LightGBM is not estimated, because LightGBM has no GPU training to compare with.
- **Non-deterministic GPU training.** Bit-wise repeatability is not guaranteed for CatBoost GPU training or for any documented PyTorch exception; observed variation is measured and reported.
- **Implementation differences remain.** CatBoost's categorical statistics and combination regularisation differ between devices by design of the library; the supportive comparison reports their joint effect, not a hardware effect alone.
- **bf16 training.** A project engineering adaptation; float32 safeguards limit, but do not remove, numerical differences from float32 training.
- **float32 attention arithmetic.** FT-Transformer's float32 prediction runs through a fast float32 multiply-add mode in the attention kernel; its agreement with a math-backend reference is measured in the probe, not assumed.
- **Synthetic probe.** It confirms behaviour on the exercised paths; variation on the research data is measured in the development runs.
- **Build identity.** Two builds of LightGBM from the same source, toolchain and settings produced different library hashes. Temporary file and directory names generated during each build, which can enter the binary, are a known cause of such differences; the hash therefore identifies one installed binary, not the source. Research reproducibility does not depend on identical bytes: it rests on the pinned source version, toolchain and build settings with the acceptance rule of §2, the fixed seeds and settings of §6, and the recorded hash, which ties every run to the binary it used. A rebuild is recorded with its new hash before any run uses it.

## References

REF-087, REF-116, REF-117, REF-118, REF-119, REF-120, REF-121, REF-122, REF-123, REF-127 — see [`../references.md`](../references.md).

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [D], [I], [L] | Evidence tags: dataset evidence; inference; literature or official documentation, issue reports and source code |
| bf16 | bfloat16: 16-bit floating-point format with the exponent range of float32 |
| [CLS] | Classification token whose final representation gives the FT-Transformer prediction |
| CPU, GPU | Central processing unit; graphics processing unit |
| CTR | CatBoost's categorical statistics computed from target and frequency information |
| CUDA | NVIDIA's GPU computing platform |
| cuBLAS, cuBLASLt | CUDA Basic Linear Algebra Subprograms library and its lightweight matrix-multiplication interface |
| DR, REF-NNN | Decision Record; reference identifier in `docs/references.md` |
| EFB | Exclusive Feature Bundling: grouping of features that rarely take non-default values in the same row into one stored column |
| float32 | 32-bit floating-point format |
| FT-Transformer | Feature Tokenizer + Transformer model for tabular data |
| GB | Gigabyte |
| GCC | GNU Compiler Collection: the C and C++ compiler used for the LightGBM build |
| H1, H2, H3 | Confirmatory hypotheses: model family; class weighting; alert-rate transfer fidelity |
| K | Fixed daily review budget |
| MLP | Multilayer perceptron |
| MVS | Minimum Variance Sampling, CatBoost's CPU default bootstrap type |
| NCCL | NVIDIA Collective Communications Library, required by LightGBM's CUDA build |
| OpenCL | Open Computing Language: a vendor-neutral GPU programming interface |
| OpenMP | Open Multi-Processing: the CPU multithreading interface used by LightGBM |
| Pool | CatBoost's training-data container |
| R3 | Decision policy: daily top-K review |
| Recall@2% | Share of frauds reviewed under daily top-K review at 2 % capacity |
| SESOI | Smallest Effect Size Of Interest |
| SHA-256 | Secure Hash Algorithm, 256-bit: the checksum recorded for the LightGBM library |
| sm_120 | Compute capability of the project GPU (NVIDIA Blackwell generation) |
| Test | Sealed final evaluation partition (relative days 155–182) |
| Train | Model-fitting partition (relative days 1–105) |
| top-K | The K highest-scored transactions of a relative day |
| V1, V2 | Validation 1 (relative days 106–126); Validation 2 (relative days 127–154) |
| WSL | Windows Subsystem for Linux |
