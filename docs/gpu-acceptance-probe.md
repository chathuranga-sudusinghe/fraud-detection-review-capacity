# GPU acceptance probe (DR-013 §8)

| Field | Value |
|---|---|
| Status | **Specification, approved on 14 October 2026 (Issue #77)**, before implementation and before the probe runs; **revised and approved on 15 October 2026** (Issue #77) for the training devices of the DR-013 revision of Issue #101. **Acceptance gate passed on 15 October 2026** (Section 7). **Section 8** (CatBoost tree-objective acceptance, Issue #123): approved on 21 October 2026, before implementation and before it runs; **acceptance passed on 21 October 2026** (Section 8, Result). **Section 9** (neural batching acceptance, Issue #151): specified before implementation and before it runs; result pending |
| Governing record | Decision Record (DR) [DR-013](decisions/DR-013-training-device-policy.md) §8: what the probe must confirm. This document owns how: the synthetic inputs, the checks, the acceptance criteria and the tolerances |
| Command | `review-capacity probe-gpu` (package command); Section 8: `review-capacity probe-catboost-objective`; Section 9: `review-capacity probe-neural-batching` |
| Evidence | [`evidence/gpu_acceptance_probe/`](../evidence/gpu_acceptance_probe/README.md) |

Statements are tagged by source: **[L]** library documentation or source code, cited with its reference identifier (REF-NNN, [`references.md`](references.md)); **[D]** documented dataset facts; **[I]** an engineering choice of this probe.

## Purpose and boundaries

The graphics processing unit (GPU) acceptance probe confirms, on the project hardware (NVIDIA RTX 5060 Laptop, 8 gigabytes (GB), compute capability sm_120), that every device setting of DR-013 works as specified for the eight configurations of [DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md), before any development run.

- **Synthetic data only.** No IEEE-CIS (IEEE Computational Intelligence Society) row, label or partition is read; the synthetic inputs take only the shape of the study's inputs (names, roles, counts and documented category counts).
- **Engineering behaviour only:** not model performance, and not the size of variation on the research data (DR-013 §8).
- **No fallback.** Each configuration runs on its training device of DR-013 §1: LightGBM on the central processing unit (CPU), the other models on the GPU. The probe fails visibly on an unsupported required capability: it never falls back to another device, removes categorical inputs, reduces a model or reduces a batch size (DR-013 §8). A failure is resolved on its evidence before any development run, and the response is recorded in Issue #77.
- **Probe values.** The hyperparameters, batch sizes and measurement counts below are probe values [I], chosen to exercise the code paths within a bounded run time. They are not research values: the training rules (Issue #91) fix those, including the protocol's timing procedure.

## 1. Synthetic inputs

All synthetic data are drawn from one generator with its own recorded seed (the data seed), separate from the model seeds, so that refitting a model never changes its inputs.

**Category counts [D].** Each of the 49 categorical inputs has as many synthetic labels as it has levels in the study's Train (model-fitting) partition in the [feature catalog](feature-catalog.md) (column "Levels in Train", read by `review_capacity.data.catalog`): for example 11,870 for `card1`, 1,470 for `DeviceInfo` and 5 for `ProductCD`, 16,797 levels in total. The embedding tables therefore have the study's dimensions.

**Tree-based models** receive the canonical model inputs of [DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md): the 433 inputs of [`config/features.yaml`](../config/features.yaml), with their declared names and roles (49 categorical and 384 numeric, including the time-of-day inputs `tod_sin` and `tod_cos`, the sine and cosine of the time of day).

| Partition | Rows | Purpose |
|---|---|---|
| Train | 368,632 (the size of the study's Train partition) | Fitting, resources and timing at the study's scale |
| Validation 1 (V1) | 100,000 | Early stopping |
| Score set | 100,000 | Prediction checks, including categories unseen in Train |

- **Categorical inputs:** in every input, 5 % of values are the missing label `"__MISSING__"`. In the score set, 1 % of values per input are labels absent from Train (unseen categories).
- **Numeric inputs:** a mixture of normal, log-normal and zero-inflated columns, with 10 % missing values (NaN, not-a-number) in half of them; `tod_sin` and `tod_cos` from uniformly drawn times of day.
- **Labels:** drawn from a logistic model of a few numeric and categorical inputs, including one interaction of two categorical inputs, with a fraud rate of about 3.5 %.
- **Row order:** increasing `TransactionDT` (synthetic transaction time), ties by increasing `TransactionID` (synthetic transaction identifier) (CatBoost's `has_time`, DR-008 §5); each row has a synthetic row identifier for ranking ties.
- **Weights:** the protocol's Train-only class-weight rule through `models.base.class_weights`. The unweighted configurations receive unit weights; the balanced ones weight each fraud row by the ratio of legitimate to fraud Train rows. The synthetic row counts and the resulting weights are recorded.

**Neural models** receive inputs in the form of [DR-012](decisions/DR-012-neural-network-input-representation.md):
- 708 numeric inputs: 384 transformed values (standard normal) and 324 binary missingness indicators;
- the 49 categorical inputs as indices. Each vocabulary holds the input's documented Train levels plus the reserved `"__MISSING__"` entry: 16,846 embedding rows in total.

This gives 757 inputs and, for the Feature Tokenizer + Transformer (FT-Transformer), 758 tokens including the classification token [CLS].

Neural fits use a row subset (16,384 Train rows, 4,096 V1 rows; the score set's first 4,096 rows for scoring) [I]. The Issue #50 probe estimated about half an hour for one full Train epoch of the largest FT-Transformer setting ([`gpu_probe_full.csv`](../evidence/gpu_probe/gpu_probe_full.csv): 800 tokens, embedding size 512, 6 layers, batch size 64). Its batch limits do not establish feasibility here, because it measured other vocabulary sizes, ResNet hidden factors and attention-backend settings. The batch sizes below are therefore proposed acceptance workloads, confirmed or refuted by this probe.

## 2. Configurations and probe values [I]

Architecture variables: `n_layers` (layer count), `d` (ResNet width), `d_token` (FT-Transformer token width), `d_embedding` (ResNet category-embedding width), `d_hidden_factor` and `d_ffn_factor` (hidden-width ratios). The other parameters are named as in their libraries:

| Parameter names | Meaning |
|---|---|
| `num_iterations`, `iterations` | Maximum tree counts (LightGBM, CatBoost) |
| `early_stopping_round`, `od_wait` | Early-stopping waits: rounds without improvement before stopping |
| `metric`, `eval_metric` | Evaluation metrics |
| `learning_rate`, `lr` | Learning rates (trees; neural optimiser) |
| `num_leaves`, `depth` | Maximum LightGBM leaves per tree; CatBoost tree depth |
| `metric_period` | CatBoost metric-reporting interval |
| `model_size_reg` | CatBoost model-size regularisation |
| `fold_permutation_block`, `permutation_count` | CatBoost permutation block size; number of permutations |
| `random_strength` | CatBoost split-score randomisation strength |
| `l2_leaf_reg`, `leaf_estimation_iterations` | CatBoost leaf-value regularisation; leaf-estimation steps |
| `bagging_temperature` | CatBoost Bayesian-bootstrap temperature |
| `hidden_dropout`, `attention_dropout`, `ffn_dropout`, `residual_dropout` | Dropout probabilities of hidden activations, attention weights, feed-forward activations and residual branches |
| `eval_batch_size` | Evaluation batch size |
| `weight_decay` | Decoupled optimiser weight decay |
| `max_epochs`, `patience` | Training epoch limit; consecutive epochs without V1 improvement allowed before stopping |

| Model | Probe values |
|---|---|
| LightGBM | `num_iterations` 300, `early_stopping_round` 20, `metric` `binary_logloss`, `learning_rate` 0.1, `num_leaves` 63 |
| CatBoost | `iterations` 300, `od_wait` 20, `eval_metric` `Logloss`, `metric_period` 1, `learning_rate` 0.1, `depth` 6, `model_size_reg` 0.5, `fold_permutation_block` 1, `permutation_count` 4, `random_strength` 1, `l2_leaf_reg` 3, `leaf_estimation_iterations` 10, `bagging_temperature` 1 |
| ResNet-style multilayer perceptron (MLP) | Small (`n_layers` 2, `d` 256, `d_hidden_factor` 2) and the largest of the large-dataset search space (B) of [DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md) (`n_layers` 16, `d` 1024, `d_hidden_factor` 4); both with `d_embedding` 64, `hidden_dropout` 0.1, `residual_dropout` 0.1, training batch size 1024 and `eval_batch_size` 4096 |
| FT-Transformer | The paper's default (`n_layers` 3, `d_token` 192, training batch size 256); the largest of space (B) (`n_layers` 6, `d_token` 512, training batch size 64); and a padded-head setting (`n_layers` 2, `d_token` 200, so heads of 25 units padded to 32, training batch size 128). All with `d_ffn_factor` 4/3, `attention_dropout` 0.1, `ffn_dropout` 0.1, `residual_dropout` 0.1 and `eval_batch_size` 16 |
| Neural loop | The Adam optimiser with decoupled weight decay (AdamW) with `lr` 1e-4 and `weight_decay` 1e-5; `max_epochs` 3, `patience` 1; V1 early stopping on the shared V1 objective of synthetic V1 rows with identifiers and times (daily budget 20 on 21 days) |

`metric_period` is 1 because, with early stopping and an evaluation set, CatBoost computes the evaluation metric on every iteration and warns that a larger `metric_period` is ignored for it ([L] REF-119: `catboost/libs/train_lib/train_model.cpp`, v1.2.10); revised on 14 October 2026 (Issue #77) after the probe's own check C2 reported that warning in a CPU test run.

- All eight configurations (four models, unweighted and balanced) run on their training devices of DR-013 §1 with V1 early stopping and the main seed 42: both LightGBM configurations on the CPU with deterministic training (DR-013 §2), CatBoost and both neural models on the GPU. Each LightGBM configuration is fitted twice with the same seed (C11). The two CatBoost configurations are also refitted on the CPU at the GPU-kept tree count (DR-013 §7).
- Neural variants and repetitions run one after another; each finished network and optimiser is released before the next fit.

## 3. Checks and acceptance criteria

Each check is pass/fail (P/F) or descriptive (D: recorded, no pass or fail). The probe passes when every P/F check passes.

| ID | DR-013 §8 item | Method | Criterion |
|---|---|---|---|
| C1 | Backend identity | Tree models: a successful fit with the build verified at start-up (LightGBM) and the native resolved training device (LightGBM `device_type`, CatBoost `task_type`); device-identification messages of the libraries are kept. PyTorch: the CUDA (NVIDIA's GPU computing platform) device of inputs, outputs and parameters, and one forward and backward pass of each FT-Transformer variant through the required attention backend (`sdpa_kernel`, which raises an error if that backend cannot run, [L] REF-121). GPU activity and timing are recorded as corroboration only | P/F: LightGBM `cpu` with the recorded build; CatBoost `GPU`; all neural tensors on CUDA; no attention-backend error |
| C2 | Ignored-option and unsupported-option warnings | Every warning is kept with its library, phase and text | P/F: any warning of an ignored setting, an unsupported capability or a fallback fails. Accepted automatically, from LightGBM's own messages only: "No further splits with positive gain", provided C6 passes; and the two notices "Categorical features with more bins than the configured maximum bin number found." and "For categorical features, max_bin and max_bin_by_feature may be ignored with a large number of categories.", provided the probe's own Dataset has a categorical feature with more bins than `max_bin` (LightGBM's configured numeric-feature bin limit; C4). Only the exact notice text is accepted; a message with any other content follows the rules of this check. LightGBM prints both notices for exactly that condition, and `max_bin` does not cap categorical features by design (DR-013 §2; [L] REF-117: `dataset_loader.cpp`). Any other warning blocks acceptance until its classification is reviewed and recorded |
| C3 | Resolved settings | The probe compares every fixed setting of `config/models.yaml` with each library's native resolved record, for every device the library is declared for (LightGBM: CPU; CatBoost: GPU and CPU), after mapping aliases and representations. Settings missing from that record are checked through their documented native metadata or by observation. The determinism and histogram-mode flags are included | P/F: every fixed setting is confirmed; requested values alone are not evidence; missing or contradictory evidence fails |
| C4 | LightGBM Dataset reproducibility (DR-013 §8: LightGBM determinism) | Two LightGBM Datasets are built independently from the same rows, weights, category mappings, seed and CPU construction settings, and saved with `save_binary` to fresh files. The file holds the header, the metadata and every feature group, whose bin mappers hold the bin boundaries or categorical mappings and are followed by the binned values; alignment padding is zero-filled ([L] REF-117: `Dataset::SaveBinaryFile`, `SerializeHeader`, `FeatureGroup::SerializeToBinary`, `BinaryWriter::AlignedWrite`). The largest bin count of a categorical feature is recorded for C2 | P/F: the two files are byte-identical (T4) |
| C5 | Categorical behaviour | The realised cardinalities and the missing and unseen counts are recorded. LightGBM's conversion of unseen categories to missing is verified; the neural inputs use the reserved `"__MISSING__"` index. CatBoost's saved model is inspected for categorical-combination statistics over two or more inputs | P/F: every fit and prediction succeeds with scores in [0, 1]; a combination over two or more inputs is present in CatBoost's model, otherwise the combination path counts as not exercised and the check fails. No performance threshold |
| C6 | All eight configurations, non-uniform weights | Fits on each configuration's training device with V1 early stopping; the weights reaching training and V1 are verified | P/F: each configuration keeps at least one tree or epoch; its V1 and score-set scores pass `checked_scores`; balanced fits receive the non-uniform weights and unweighted fits unit weights |
| C7 | Neural computation under strict determinism | Every neural fit runs with `torch.use_deterministic_algorithms(True, warn_only=False)`, through embeddings, normalisation, category dropout (0.01), and forward and backward passes, including FT-Transformer at 758 tokens and the padded-head setting. The category-dropout replacements are counted by replaying the recorded seeds through the same batching and dropout functions. Gradients, parameters and optimiser-state tensors are inspected | P/F: no determinism error; category-dropout replacements occur; gradients and parameters finite; training logits observed in bfloat16 (bf16, a 16-bit floating-point format) and optimiser state in float32 (32-bit floating point) |
| C8 | float32 evaluation path | For each fitted neural model, the scores of 4,096 rows on the required float32 path are compared with a float32 reference, with identical parameters and buffers (including batch-normalisation running statistics), in evaluation mode, with autocast and gradients disabled, in identical batch partitions. ResNet reference: the CPU in float32. FT-Transformer reference: a second network built with `SDPBackend.MATH` as its attention backend and loaded with the same state, on the GPU, keeping the head padding and the true-head scale. An outer backend context cannot replace the network's own; the required path is confirmed to use memory-efficient attention | P/F: the maximum absolute logit difference is within T1. D: logit magnitudes, maximum and mean absolute differences, differences relative to the logit scale, score differences, and the peak GPU memory of both paths |
| C9 | Numerical behaviour | All scores of every configuration. Rankings are by descending score, ties broken by ascending synthetic row identifier. K (number of top-scored rows compared) is 2 % of the scored rows, rounded up to the next integer: 82 for the neural score set and 2,000 for the tree-model score set [I]. This is a probe diagnostic, not the study's daily review budget | P/F: all scores finite. D: ties among the top-K scores, the score margin at the cutoff, and changes in top-K membership across the two runs (C11) and between each float32 path and its reference (C8). Numerical closeness does not guarantee identical top-K membership |
| C10 | Prediction repeatability and backend | Every model scores the same rows twice in the same process; tree models through their CPU prediction backend with the kept tree count (DR-013 §5) | P/F: identical scores, bit for bit (T2) |
| C11 | Same-seed repeats | Each LightGBM and neural configuration is fitted twice with the same seed in one process; the whole probe is run twice, in fresh processes (Section 6). For every LightGBM fit, the SHA-256 of its complete saved model text is recorded, keyed by configuration and repetition | P/F: LightGBM's saved model text and scores are identical within a process and across the two fresh-process runs, with a model-text hash for every LightGBM fit (a missing hash fails); neural scores are identical within and across processes (T3). D (CatBoost): GPU training is non-deterministic ([L] REF-118), so the maximum score difference and the top-K membership changes across runs are recorded |
| C12 | Persistence | Every fitted model is saved, reloaded and scores the same rows again | P/F: identical scores, bit for bit (T2) |
| C13 | Resources and timing | GPU memory: the device capacity, the baseline use, and the sampled whole-device maximum from `nvidia-smi` every 0.5 seconds (recorded as a sampled maximum, with interval and units). PyTorch peak allocated and reserved memory per phase; peak host memory and its scope. Timing: GPU work synchronised at every timing boundary; start-up, input preparation, fitting and prediction separately; model-only and end-to-end scoring separately; neural scoring also timed on the CPU, with those scores discarded. Three warm-up calls, five timed repetitions, and repeated single-row latency on 128 fixed synthetic rows (median, 95th and 99th percentiles) [I]. Neural per-step times extrapolated to a full Train epoch, labelled as extrapolations that exclude validation and other unmeasured work | P/F: no out-of-memory error at the fixed workload. D: every measurement |

**Gradient checkpointing** is not implemented. If the training rules adopt it, its implementation and a revised probe specification are reviewed, and the acceptance checks are rerun with checkpointing enabled before a development run uses it.

## 4. Tolerances

| ID | Quantity | Tolerance | Basis |
|---|---|---|---|
| T1 | float32 logits against the float32 reference (C8) | Maximum absolute difference at most 1e-4, with no relative allowance | A pre-specified engineering acceptance budget [I], not a numerical error bound guaranteed by PyTorch. Error accumulated through 16 residual layers or 6 Transformer layers is not bounded by the sources: the memory-efficient kernel computes float32 products in a fast float32 multiply-add mode ([L] REF-121), and the GPU and CPU sum in different orders ([L] REF-120). The budget implies a score difference of at most 2.5e-5 (the sigmoid's slope is at most 0.25). A failure requires numerical diagnosis and prevents acceptance; it does not by itself establish an implementation defect. The tolerance is never widened after the results are seen |
| T2 | Repeated prediction and prediction after reload (C10, C12) | Bit-for-bit identical | Required for the fixed-model prediction path with identical input bytes, row order, backend, batch partitions, threads, precision settings and software and hardware environment [I] |
| T3 | LightGBM and neural same-seed repeats (C11) | Bit-for-bit identical: LightGBM's saved model text and scores; neural scores | Required for compatible same-seed fits within one process and across the two fresh-process runs, with every data and training generator controlled and identical execution order and settings: LightGBM's deterministic CPU training with a fixed thread count and histogram mode for the same build and system (DR-013 §2; [L] REF-116), and the neural models' strict deterministic mode (DR-013 §4). A mismatch fails this gate and requires diagnosis of input, configuration, random-state, implementation or backend differences; it is not automatically classified as a code defect. No blanket `CUBLAS_WORKSPACE_CONFIG` setting is introduced (DR-013 §4) |
| T4 | Two LightGBM CPU Datasets (C4) | Byte-identical | Identical rows, weights, category mappings, seed and construction settings, with bundling and sparse optimisation disabled (DR-013 §2). Identical files establish identical bin boundaries and binned values; different files can also reflect differences in metadata or construction settings ([L] REF-117) |

## 5. Records

The probe writes, step by step:

- `probe.log`: the run log, with every library message and warning;
- `probe.json`: the record of every check. It includes:
  - the environment of DR-013 §6;
  - the run identity: code commit, the SHA-256 (Secure Hash Algorithm, 256-bit) of this specification and of `config/models.yaml`, the data seed and the model seeds;
  - for every set of scores, a record keyed by configuration, architecture variant, device, repetition and scoring path, holding the path, SHA-256, dtype, shape and row-order hash of its comparison file;
- comparison files: each run keeps its synthetic score arrays, with their ordered synthetic row identifiers, as local comparison files in its run directory. Their hashes are computed from a documented canonical byte representation and recorded in `probe.json`. Being synthetic, they are not research data;
- `probe.md`: a summary generated by code from `probe.json`, never written by hand (as for the development runs, Issue #96).

## 6. Run procedure

1. This specification and its tolerances are approved before the implementation begins. The implementation is then reviewed and committed before either acceptance run. A change of settings, tolerances or exercised paths requires an approved revision of this specification and new acceptance evidence.
2. `review-capacity probe-gpu` runs from the clean committed tree.
3. A first run alone records the result "pending second run", never a pass. Each run writes to a new directory of its own, with its log file `probe.log`: the directory given, or by default `outputs/gpu_acceptance_probe/<UTC start time>/` (the start time in Coordinated Universal Time, UTC, written `YYYY-MM-DDTHHMMSSZ`), inside the repository and ignored by Git, so that a run never changes the clean tree of the next. A second run, in a fresh process, receives the first run's `probe.json` and comparison files. Before any numerical comparison, it verifies that the code, this specification, the settings, the synthetic inputs, labels, weights, layouts and execution conditions match. An incompatible pair is reported as not comparable and cannot satisfy C11. A compatible pair gives bit-for-bit equality, the maximum score difference and the top-K membership changes.
4. The records of both runs (`probe.json`, `probe.md`, `probe.log`) are added to `evidence/gpu_acceptance_probe/` and reviewed; the comparison files stay in the run directories, identified by their SHA-256 values in `probe.json`. The result is recorded in this document and in Issue #77.

## 7. Results

**Runs of 15 October 2026** (commit `6973f49`, clean tree; records in [`evidence/gpu_acceptance_probe/`](../evidence/gpu_acceptance_probe/README.md)). Both runs stopped in the first LightGBM fit, then on the GPU (`LGB-u`: LightGBM, unweighted; 368,632 Train rows), with LightGBM's `[CUDA] an illegal memory access was encountered` at `cuda_best_split_finder.cu:1811` and a process abort (exit code 134). Only C4 (then the CPU and GPU Datasets: byte-identical) and C5 were recorded. The cause is a defect of LightGBM's CUDA learner on categorical features with more than 256 categories ([L] REF-127). The response is the DR-013 revision of Issue #101: LightGBM trains on the CPU. This specification was revised for it on 15 October 2026, and the acceptance runs are repeated under the revision.

**Run 3, 15 October 2026** (commit `7bc3142`, clean tree, revised specification). No pass/fail item failed before the interruption; the run was incomplete, with its C3 review items open. The largest unweighted FT-Transformer configuration (`FTT-u`, largest) completed its first fit, and its C8 comparison recorded a maximum absolute logit difference of about 2.1e-6, within T1. During C13 timing, the process received `CUDA error: an illegal memory access was encountered` and wrote its record (stage `failed`), at the same moment as a Windows bugcheck `0x116` (VIDEO_TDR_FAILURE: the display driver's timeout detection and recovery could not reset the GPU) and restart ([Issue #77](https://github.com/chathuranga-sudusinghe/fraud-detection-review-capacity/issues/77#issuecomment-6187485355)). The failure occurred at the GPU device, driver and Windows level; its exact trigger is not recorded in the run log or the Windows event log. The same scoring path completed earlier in run 3 and in acceptance runs 4 and 5 under the same memory load, and the acceptance outcome rests on runs 4 and 5.

**Acceptance runs 4 and 5, 15 October 2026** (commit `7bc3142`, clean tree, revised specification). Run 4 ended as `pending second run`. Run 5, in a fresh process and given run 4's `probe.json`, ended as `review`, with the pair comparable on code, this specification, the settings, the synthetic inputs and the environment:

- C1, C2, C4–C13 pass in both runs. C2 accepted LightGBM's two categorical `max_bin` notices under the exact-text rule; the largest categorical feature has more bins than `max_bin`.
- C11 across the runs: 74 comparisons. **62 are bit-for-bit identical**: every LightGBM and neural score set and all four LightGBM saved-model hashes (T3). None failed. The 12 CatBoost comparisons are descriptive: the GPU fits of the unweighted CatBoost configuration (`Cat-u`) and the balanced one (`Cat-b`) show maximum differences of about 3.0e-5 and 3.6e-4, with no change in top-K membership, and the CPU refits are bit-for-bit identical.
- C13: about 53 minutes per run; sampled whole-device GPU memory at most 6,788 of 8,151 MiB (mebibytes); no out-of-memory error and no monitoring problem.
- C3 confirms every fixed setting from the libraries' resolved records, except CatBoost `sampling_unit`, which CatBoost 1.2.10 does not serialise into the fitted model or `get_all_params()` ([L] REF-119: `bootstrap_options.h`). **Review disposition, accepted on 15 October 2026 (Issue #77):** `sampling_unit = "Object"` is confirmed. It is CatBoost's documented default and the only scheme that applies to the study's data, where each transaction is an independent object with no group structure; and the observed behaviour agrees (bit-for-bit identical CPU refits across fresh processes, GPU differences within the documented non-determinism of DR-013).

**Outcome: the GPU acceptance gate of DR-013 §8 is passed.** Every pass/fail check passes on the project hardware, and the only review item is settled.

## 8. CatBoost tree-objective acceptance (Issue #123)

[DR-015](decisions/DR-015-common-training-rules.md) §3 trains CatBoost on the shared V1 objective as a user-defined evaluation metric on the GPU learner, for up to 2,000 boosting rounds (§2) and tree depths 6 to 10 (§6). Sections 1 to 7 measured depth 6 at 300 iterations with a built-in metric only. This section confirms, on the project hardware and before any development run, the largest depth of the search space at the maximum length with the user-defined metric. Everything else in Sections 1 to 7 is unchanged: this section adds a separate command and a separate record, so the evidence of the accepted probe is neither reopened nor reused.

- **Synthetic data only**, from the generator of Section 1 with its data seed: the tree-model Train partition (368,632 rows) and V1 partition (100,000 rows) with their synthetic `TransactionDT` and `TransactionID`. The relative day of a row is `TransactionDT // 86400` (DR-001). No IEEE-CIS row is read.
- **Engineering behaviour only**, as in the purpose above. Model performance is not assessed.
- **No fallback.** The fit runs on the GPU. A failure (an unsupported user-defined metric on the GPU learner, an out-of-memory error, an error in the metric) is recorded and resolved on its evidence before any development run; the probe never moves to the CPU, reduces the depth, the length or the data, or replaces the metric.

**Fit.** One CatBoost fit, `Cat-u` (unweighted; balanced class weighting changes the Train loss and so the trees, while the V1 objective is unweighted in both, so the balanced configuration is not separately probed [I]), seed 42, V1 early stopping through `CatBoostModel.fit(..., objective=...)` with the V1 objective of `models.v1_objective`. Probe values [I], with the reference candidate of DR-015 §6 except the depth:

| Parameter | Value |
|---|---|
| `depth` | 10 (the largest of the space) |
| `iterations` | 2000 (the maximum of §2) |
| `od_wait` | 2000, so that the overfitting detector cannot stop the fit before the maximum (the study's value is 50, §2) |
| `eval_metric`, `boost_from_average`, `metric_period` | `v1_objective`, `False`, 1 |
| `learning_rate` | 0.03234 (the reference value of DR-015 §6) |
| `l2_leaf_reg`, `bagging_temperature`, `leaf_estimation_iterations` | 3, 1, 10 (the reference values) |
| `model_size_reg`, `fold_permutation_block`, `permutation_count`, `random_strength` | 0.5, 1, 4, 1 (DR-015 §6) |

The daily budget of the V1 objective is `K = round(0.02 × V1 rows / V1 days)` (the rule of DR-015 §3), computed from the synthetic V1 partition and recorded.

| ID | Method | Criterion |
|---|---|---|
| G1 | Device and completion. The fit runs to its end; the resolved `task_type` is read from CatBoost's native record | P/F: `GPU`; no error from the GPU learner or the metric |
| G2 | Length. The history of the V1 objective, one entry per round | P/F: 2,000 rounds trained and 2,000 V1 evaluations recorded; stopping reason `maximum`; CatBoost's own evaluation history has 2,000 entries; the kept tree count equals the earliest best checkpoint of the history. The fit trains all 2,000 rounds; with the best-model cut it keeps only the rounds up to the earliest best |
| G3 | Objective correctness. The model is scored on V1 through the CPU prediction path of DR-013 §5 at the kept tree count, and the shared daily queue computes the objective from those scores | P/F: the numerator equals the numerator recorded for the kept checkpoint (exact integers). D: the number of evaluations of other sets that the metric did not score. The metric ranks the logistic transform of CatBoost's raw approximations computed with NumPy; on other CPUs these probabilities differ from CatBoost's native `predict_proba` by a few units in the last place (seen in continuous integration), so tie groups can differ only for scores that close. Equal numerators show that the objective agrees; they do not show identical queue membership, and the GPU's in-training approximations are not the CPU-recomputed scores of DR-013 §5. A mismatch requires numerical diagnosis and prevents acceptance; the criterion is never widened after the results are seen |
| G4 | Resolved settings. `get_all_params()` and the model's metadata | P/F: `depth` 10, `boost_from_average` `False`, every fixed setting of `config/models.yaml` confirmed as in C3 |
| G5 | Persistence. The model is saved, reloaded and scores the V1 rows again | P/F: identical scores, bit for bit (T2); the saved record holds the objective's identity and `eval_metric` `v1_objective` |
| G6 | Warnings. Every library message and warning is kept as in C2 | P/F: the rule of C2 (an ignored setting, an unsupported capability or a fallback fails). The CatBoost notice that numba is not installed ("Failed to import numba for optimizing custom metrics and objectives") is accepted by its exact text: it concerns the speed of the metric only |
| G7 | Resources and timing as in C13: GPU memory (capacity, baseline, sampled whole-device maximum every 0.5 seconds), peak host memory, the fit's total time and the increase of the elapsed time over the 2,000 checkpoints | P/F: no out-of-memory error. D: every measurement |

**Records.** `acceptance.json` (the environment of DR-013 §6, the run identity of Section 5, the parameters, the history of the V1 objective, G1 to G7), `acceptance.md` (generated from it by code, presenting every recorded field and the V1 objective after every round) and `acceptance.log`, written in a new directory (one that does not exist or is empty) `outputs/catboost_objective_acceptance/<UTC start time>/` inside the repository (ignored by Git) as in Section 6. GPU training is not deterministic (DR-013 §2), so one run is the acceptance run; no second run is compared. The records are added to `evidence/` in a reviewed pull request, and the result is recorded in this section and in Issue #123.

**Acceptance run.** A run is the acceptance run only on the GPU, at depth 10 and 2,000 rounds, on the Train and V1 sizes of Section 1, from a clean committed tree whose code is the code that runs (the running package lies inside the checkout whose Git state is recorded); `acceptance.json` records `acceptance_evidence` and, when false, the reasons (`evidence_blockers`). Any other run is exploratory and its summary says so.

**Procedure.** This section is approved before the implementation begins; the implementation is reviewed and committed before the run; `review-capacity probe-catboost-objective` runs from the clean committed tree. A change of values or paths requires an approved revision and a new run.

**Result, acceptance run of 21 October 2026 (UTC)** (commit `03bf5be`, clean tree; records in [`evidence/catboost_objective_acceptance/2026-10-21T222733Z/`](../evidence/catboost_objective_acceptance/2026-10-21T222733Z/)). Overall result **pass**; the run is the acceptance run (`acceptance_evidence` true, no blockers). The daily budget of the synthetic V1 partition (100,000 rows, 20 days, 3,462 frauds) is K = 100.

- G1: the resolved task type is `GPU`; the GPU learner accepted the user-defined metric and the fit completed.
- G2: 2,000 rounds trained and 2,000 V1 evaluations recorded, stopping reason `maximum`, CatBoost's own evaluation history of 2,000 entries; the earliest best checkpoint is round 107 and 107 trees are kept.
- G3: the numerator recomputed from the CPU-scored V1 rows at the kept model (432) equals the recorded numerator of the kept checkpoint (432). CatBoost also evaluated the metric on the learn set at every round (2,000 evaluations of other sets), which the metric left unscored.
- G4: `depth` 10 and `boost_from_average` `False` in the resolved parameters; all 21 fixed settings confirmed; `sampling_unit` as in Section 7.
- G5: reloaded scores bit for bit identical; the objective's identity and `eval_metric` `v1_objective` are in the saved record.
- G6: one warning, CatBoost's numba notice (accepted by its text, it concerns the speed of the metric only).
- G7: fit time 485.5 s (about 0.24 s per round after the first); sampled whole-device GPU memory at most 6,788 of 8,151 MiB (975 samples, no monitoring problem); peak host memory 6.56 GB; no out-of-memory error.

**Outcome: the CatBoost tree-objective acceptance of Issue #123 is passed.** The values are of synthetic data and carry no information about model performance.

## 9. Neural batching acceptance (Issue #151)

[DR-015](decisions/DR-015-common-training-rules.md) §7 splits each effective batch of 1,024 Train rows into micro-batches whose gradients are accumulated, with one AdamW update per effective batch, and fixes the micro-batch of every candidate by a preflight; its consequences require the synthetic verification of this accumulation before any neural development run. Sections 1 to 7 trained each FT-Transformer at one fixed batch size and never accumulated gradients. This section adds a separate command and a separate record, so the evidence of the accepted probe is neither reopened nor reused. The micro-batch preflight itself (`review-capacity preflight-neural`) is a procedure of the development run, not an acceptance check; this section confirms the machinery it relies on, and the corners of the search spaces it must handle.

- **Synthetic data only**, from the generator of Section 1 with its data seed and the neural layout of Section 1 (757 inputs, 16,846 embedding rows). No IEEE-CIS row is read.
- **Engineering behaviour only**, as in the purpose above.
- **No fallback.** Every fit runs on the GPU under the strict determinism and the precision of DR-013 §4. A failure is recorded and resolved on its evidence before any neural development run; the acceptance never moves to the CPU, reduces a model or a batch, or drops a case.

**Probe values [I].** The gradient comparisons use an FT-Transformer with dropout disabled (`attention_dropout`, `ffn_dropout` and the fixed `residual_dropout` equal to 0; the Train category dropout is not applied), the reference values of DR-015 §7 otherwise, and one effective batch of the first `rows` synthetic rows. The unsplit reference must itself fit in 8 GB, so the default and padded widths use 256 rows and the light architecture (one layer, token width 72) uses the 1,016 rows of the final batch of the study's Train partition (368,632 rows = 359 × 1,024 + 1,016). The weights are those of the protocol's class-weight rule (`models.base.class_weights`, balanced) or unit weights.

| Case | Layers; token width (head width, padded) | Rows | Micro-batch sizes | Weights |
|---|---|---|---|---|
| default width, equal micro-batches | 3; 192 (24) | 256 | 64, 64, 64, 64 | unit |
| default width, unequal micro-batches | 3; 192 (24) | 256 | 96, 96, 64 | balanced |
| padded heads (25 to 32), unequal micro-batches | 2; 200 (25, padded to 32) | 256 | 96, 96, 64 | balanced |
| final batch of 1,016 rows, padded heads (9 to 16) | 1; 72 (9, padded to 16) | 1,016 | 256, 256, 256, 248 | balanced |
| final batch of 1,016 rows, unequal micro-batches | 1; 72 (9, padded to 16) | 1,016 | 384, 384, 248 | balanced |

| ID | Method | Criterion |
|---|---|---|
| B1 | Gradient accumulation. For each case, the network is built once; the gradient of the unsplit effective batch is computed twice and the gradient of the same batch accumulated over its micro-batches once, through `neural_training.accumulate_gradients`, the function the training loop itself uses. The comparison is over every parameter gradient in float64: the relative L2 norm of the whole difference, the largest relative difference of a single tensor, the cosine, and the losses. Because both gradients come from the same loss function, a defect of that function (ignored weights, a wrong denominator) would not show; a second comparison therefore sets the accumulated gradient, computed with the forward pass in float32, against an independent oracle: one forward pass over all rows of the effective batch, the weighted binary cross-entropy from `torch.nn.functional` summed and divided by the rows of the batch (DR-015 §7), written without `accumulate_gradients` or `weighted_bce` | P/F: the unsplit gradient repeats bit for bit (the comparison is meaningful); every gradient is finite and over the same parameters; the relative L2 norm of the difference is at most T5; the relative L2 norm of the float32 accumulated gradient against the oracle is at most T6. D: the other measures, including the largest relative difference of a single tensor against the oracle |
| B2 | Repeatability under a plan. The ResNet-style MLP (micro-batch 1,024, its only plan) and the FT-Transformer (micro-batch 256), each at the reference candidate of DR-015 §6 and §7 with balanced weights, are fitted twice with the same seed in one process on 3,064 Train rows (two effective batches and a final batch of 1,016 rows) for two epochs | P/F: the scores of all rows and the per-epoch Train losses are identical, bit for bit (T3) |
| B3 | Corners of the search spaces, through the preflight of DR-015 §7 (`training.preflight.run_preflight`: full-size synthetic inputs, training phase of two effective batches, float32 evaluation inside the loop, declared free-memory margin): the largest ResNet-style MLP of space (B) (16 layers, width 1,024, hidden factor 4, category embedding 512), tried at 1,024; and FT-Transformers with 6 layers at token widths 72, 200, 456 and 504 (heads of 9, 25, 57 and 63 units, padded to 16, 32, 64 and 64) and at the largest width, 512. These are representative widths, not proven worst cases: each real candidate must still pass the candidate-specific preflight | P/F: each corner passes training and evaluation at some listed micro-batch with at least 1,024 MiB free at the sampled peak. A corner that passes at none is a failure: it shows that the search space contains a candidate the study could not run (DR-015 §7) |
| B4 | Warnings, as in C2 | P/F: the rule of C2 (an ignored setting, an unsupported capability or a fallback fails; any other warning needs a recorded review). Accepted by exact form, as many times as B3 recorded attempts that ran out of GPU memory: the line `Fit <configuration> on gpu (<n> Train rows): failed after <s> s` that the step logger writes for such an attempt (a micro-batch too large, which the preflight tries and abandons) |
| B5 | Resources, as in C13: whole-device memory sampled every 0.5 seconds, host peak memory | P/F: valid monitoring; no out-of-memory error outside the cases that record one. D: every measurement |

| ID | Quantity | Tolerance | Basis |
|---|---|---|---|
| T6 | Relative L2 norm of the difference between the float32 accumulated gradient and the independent float32 oracle (B1) | At most 1e-4 | A pre-specified engineering budget [I], specified before any acceptance run. With the forward pass in float32 the two gradients differ by the order of float32 sums only (the memory-efficient attention kernel computes float32 products in a fast multiply-add mode, [L] REF-121), far below a defect such as unit weights or a denominator one row too large; a tail denominator of 1,024 for 1,016 rows (an error of 0.78 %) is 78 times the budget. The tolerance is never widened after the results are seen |
| T5 | Relative L2 norm of the difference between the accumulated and the unsplit gradient (B1) | At most 1e-2 | A pre-specified engineering acceptance budget [I]. In exact arithmetic the two gradients are equal (DR-015 §7: every micro-batch loss divides by the rows of the whole effective batch). In the training precision they differ by rounding: bfloat16 forward products and a different order of float32 sums with each micro-batch plan. A wrong normalisation (for example a mean per micro-batch) differs by a factor of the number of micro-batches, and a lost micro-batch by its share of the rows, both far above the budget. The tolerance is never widened after the results are seen; a failure requires numerical diagnosis |

**Records.** `batching.json` (the environment of DR-013 §6, the run identity of Section 5 with the training declaration's SHA-256, the workload, B1 to B5 with every recorded measure), `batching.md` (generated from it by code, presenting every recorded field) and `batching.log`, written in a new directory (one that does not exist or is empty) `outputs/neural_batching_acceptance/<UTC start time>/` inside the repository (ignored by Git) as in Section 6. One run is the acceptance run. The records are added to `evidence/` in a reviewed pull request, and the result is recorded in this section and in Issue #151.

**Acceptance run.** A run is the acceptance run only on the GPU, with the default workload and the effective batch of 1,024, from a clean committed tree whose code is the code that runs; `batching.json` records `acceptance_evidence` and, when false, the reasons (`evidence_blockers`). Any other run is exploratory and its summary says so.

**Procedure.** This section is approved before the implementation begins; the implementation is reviewed and committed before the run; `review-capacity probe-neural-batching` runs from the clean committed tree. A change of values or paths requires an approved revision and a new run.

**Result.** Pending: no acceptance run has been made.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [D], [I], [L] | Source tags: documented dataset fact; engineering choice of this probe; library documentation or source code |
| AdamW | Adam optimiser with decoupled weight decay |
| bf16, float32 | bfloat16 (16-bit) and 32-bit floating-point formats |
| C1–C13 | Probe checks (Section 3) |
| [CLS] | Classification token of FT-Transformer |
| CPU, GPU | Central processing unit; graphics processing unit |
| CUDA | NVIDIA's GPU computing platform |
| `d`, `d_token`, `d_embedding` | ResNet width; FT-Transformer token width; ResNet category-embedding width |
| `d_hidden_factor`, `d_ffn_factor` | Hidden-width ratios of the ResNet blocks and the FT-Transformer feed-forward modules |
| D, P/F | Descriptive check; pass-or-fail check |
| DR, DR-NNN | Decision Record; Decision Record NNN in `docs/decisions/` |
| FT-Transformer | Feature Tokenizer + Transformer |
| G1–G7 | Checks of the CatBoost tree-objective acceptance (Section 8) |
| GB | Gigabyte |
| IEEE-CIS | IEEE Computational Intelligence Society (the dataset's source) |
| MiB | Mebibyte (2^20 bytes) |
| `max_bin` | LightGBM's maximum number of bins of a numeric feature (DR-013 §2); it does not cap categorical features |
| `LGB-u`, `Cat-u`, `Cat-b`, `FTT-u` | Configuration names: LightGBM, CatBoost or FT-Transformer, unweighted (u) or with balanced class weighting (b) |
| K | Number of top-scored rows compared in C9 and C11: 2 % of the scored rows, rounded up (82 neural rows; 2,000 tree-model rows); in Section 8, the daily review budget of the V1 objective |
| B1–B5 | Checks of the neural batching acceptance (Section 9) |
| Effective batch, micro-batch | The 1,024 Train rows of one optimiser update (DR-015 §7); the part of it processed at once, whose gradients are added |
| `boost_from_average`, `v1_objective` | CatBoost: start from the average label (set to `False`); the name of the shared V1 objective (daily Recall@K, unweighted) used as `eval_metric` |
| MLP | Multilayer perceptron |
| `n_layers` | Number of layers (residual blocks or Transformer layers) |
| Relative L2 norm | The Euclidean norm of a difference divided by the Euclidean norm of the reference |
| Probe parameter names | `num_iterations`, `iterations`, `early_stopping_round`, `od_wait`, `metric`, `eval_metric`, `learning_rate`, `lr`, `num_leaves`, `depth`, `metric_period`, `model_size_reg`, `fold_permutation_block`, `permutation_count`, `random_strength`, `l2_leaf_reg`, `leaf_estimation_iterations`, `bagging_temperature`, the four dropouts, `eval_batch_size`, `weight_decay`, `max_epochs`, `patience`: defined in the parameter table of Section 2 |
| NaN | Not-a-number: a missing numeric value |
| REF-NNN | Reference identifier in [`references.md`](references.md) |
| SHA-256 | Secure Hash Algorithm, 256-bit |
| sm_120 | Compute capability of the project GPU |
| space (B) | The large-dataset hyperparameter search space of REF-087 used by DR-011 |
| T1–T4 | Tolerances (Section 4) |
| T5, T6 | Tolerances of the gradient comparisons of B1 (Section 9): accumulated against unsplit gradient; float32 accumulated gradient against the independent oracle |
| `tod_sin`, `tod_cos` | Sine and cosine of the time of day (DR-008 §4) |
| Train | The study's model-fitting partition (relative days 1–105); here, the synthetic fitting partition |
| `TransactionDT`, `TransactionID` | Transaction time and transaction identifier; here, synthetic |
| UTC | Coordinated Universal Time |
| V1 | Validation 1 (relative days 106–126); here, the synthetic early-stopping partition |
