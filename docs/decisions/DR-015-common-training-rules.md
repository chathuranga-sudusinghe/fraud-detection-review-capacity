# DR-015 — Common training rules: search, early stopping, Validation 1 (V1) objective, seeds and tie rules

| Field | Value |
|---|---|
| Status | Accepted (pre-freeze) through Issues #105, #106, #107, #112, #115 and #138, in line with the University of Greenwich criteria ([criteria checklist](../greenwich-criteria-checklist.md)); supervisor feedback welcome, not a precondition |
| Date | 2026-10-23 (Issue #151: numerical settings of the logistic-regression reference, §8); 2026-10-22 (Issue #138: computational-cost session conditions, §9; Issue #121: computational-cost procedure, §9); 2026-10-16 (Issue #105: common rules; Issue #106: tree-based search spaces and reference configurations; Issue #107: neural search spaces, reference configurations and training values); 2026-10-17 (Issue #112: neural preprocessing seed; Issue #115: logistic-regression reference search, §8) |
| Evidence source | Gorishniy et al. [REF-087] and McElfresh et al. [REF-089] (tuning protocols, read in full); Bergstra and Bengio [REF-128]; the tree-based tuning spaces of Prokhorenkova et al. [REF-012, arXiv v5 Appendix D.2], Gorishniy et al. [REF-087, Table 17; REF-129, Tables 23–24] and the TabZilla code [REF-130]; LightGBM and CatBoost documentation and source code [REF-116; REF-118; REF-119]; the neural spaces, defaults and batch sizes of REF-087 (Tables 7 and 12–14), the authors' tuning code and configurations [REF-103], the TabZilla ResNet [REF-130] and the `rtdl` package [REF-131]; PyTorch's AdamW documentation [REF-120]; the training-time and memory measurements of the graphics processing unit (GPU) acceptance probe ([`gpu-acceptance-probe.md`](../gpu-acceptance-probe.md), Section 7; Issue #77) and of the GPU memory probe of Issue #50 ([`evidence/gpu_probe/`](../../evidence/gpu_probe/README.md)); the partition sizes of [DR-001](DR-001-temporal-split-strategy.md); independent reviews in Issues #105, #106 and #107. The only labels used are the Train labels of the logistic-regression convergence diagnostic (Evidence, Issue #151); no V1 or V2 labels, model scores or Test-period data were used. |
| Related | [DR-001](DR-001-temporal-split-strategy.md), [DR-002](DR-002-review-capacity-definition.md), [DR-005](DR-005-sesoi-and-h1-h2-inference.md), [DR-010](DR-010-h3-confirmatory-configuration.md), [DR-011](DR-011-model-set-tree-based-and-neural-families.md), [DR-013](DR-013-training-device-policy.md), [protocol](../protocol.md) |

## Context

The eight configurations of [DR-011](DR-011-model-set-tree-based-and-neural-families.md) (LightGBM, CatBoost, the ResNet-style multilayer perceptron (MLP) and the Feature Tokenizer + Transformer (FT-Transformer), each unweighted (u) and with balanced class weighting (b)) are compared under one common procedure. They are fitted on Train (the model-fitting partition, relative days 1–105), tuned and early-stopped on Validation 1 (V1, days 106–126), and evaluated on Validation 2 (V2, days 127–154); the Test partition (days 155–182) is sealed ([DR-001](DR-001-temporal-split-strategy.md)). DR-011 §3 (Level 1) requires the same hyperparameter search method, the same budget per configuration, the same V1 objective, the same early-stopping rule and the same seeds for all eight. This record fixes those values before any development run.

The study's primary outcome is Recall@2%: the share of frauds captured under R3, the daily top-K policy, which reviews the K highest-scored transactions of each relative day, where K is the fixed daily review budget (K = 56 at 2 % capacity, from V2 volume; [DR-002](DR-002-review-capacity-definition.md)). H3, the alert-rate transfer fidelity hypothesis, is confirmatory for one configuration selected on V2 ([DR-010](DR-010-h3-confirmatory-configuration.md)).

The models differ greatly in training cost on the project hardware. Extrapolated from the GPU acceptance probe, one full pass over the Train partition takes about 0.1–0.4 minutes for the ResNet-style MLP and about 5, 7 and 40 minutes for the padded, default and largest FT-Transformer settings (training only; §7 adds the V1 evaluation of each epoch); a LightGBM fit takes seconds and a CatBoost GPU fit about 15–20 seconds at the probe values. Published tabular benchmarks tune each model with up to 30 to 100 trials [REF-087; REF-089].

The search spaces and reference configurations are fixed in §6 for the tree-based models (Issue #106) and in §7 for the neural models, with their training values (Issue #107).

## Decision

### 1. Hyperparameter search

- **Method:** random search [REF-128], identical for all eight configurations.
- **Budget: 10 trials per configuration.**
  - Trial 1 is the model's **reference configuration**: a fully specified setting with sources, including the settings fixed by [DR-013](DR-013-training-device-policy.md).
  - Trials 2–10 are drawn at random from the model's approved search space.
- **Candidate lists:** for each model, one ordered list of 10 candidates is generated before any fit and recorded before any V1 result is seen. The unweighted and balanced configurations of a model use the **same ordered list**, so the weighting comparison (H2, the class-weighting hypothesis) is paired by candidate.
- **Sampler:** NumPy's default generator (`numpy.random.default_rng`, the PCG64 bit generator) with sampler seed 42. One independent child stream per model is spawned from that seed (`SeedSequence.spawn`, in the fixed model order of DR-011), so each model's list does not depend on the others. The sampler streams are separate from the training seeds.
- **Search boundary:** no additional, repeated or replacement trial enters selection after V1 results are seen, and no configuration receives a budget extension. The separate reproducibility checks of §4 do not alter selection.
- **Execution note:** trials run one after another.
- **Selection:** the trial with the highest V1 objective (§3); ties as in §5.

Fairness is defined as the same procedure and the same number of trials for every configuration, not as equal compute time.

### 2. Early stopping within each trial

The V1 objective (§3) is evaluated after every boosting round (tree-based models) or every epoch (neural models). Training follows the family-specific stopping conditions below, or stops at the maximum; the earliest best checkpoint is the trial's result.

| Family | Patience | Maximum |
|---|---|---|
| Tree-based (LightGBM, CatBoost) | 50 boosting rounds: training stops after 50 consecutive rounds without improvement (LightGBM `early_stopping_round` = 50; CatBoost `od_type` = `"Iter"`, `od_wait` = 50) | 2,000 boosting rounds |
| Neural (ResNet-style MLP, FT-Transformer) | 5 epochs: training stops at the sixth consecutive epoch without improvement (the neural loop's `patience` = 5, which stops when the count of epochs without improvement exceeds the patience) | 15 epochs |

The rule is the same for all eight configurations. Patience is set in each family's natural training unit, because one boosting round adds a single tree, a far smaller step than one epoch, which passes over all Train rows with hundreds of parameter updates (360 at the effective batch of 1,024 rows; §7). The tree-based values follow REF-087 (CatBoost: 2,000 iterations, 50 early-stopping rounds); the neural values are an engineering adaptation to the measured training cost. Each trial records its stopping round or epoch, its stopping reason (early stop or maximum) and its runtime.

### 3. V1 objective

**V1 Recall@2% under daily top-K review**, the V1 analogue of the study's primary outcome (Recall@2% under R3), identical for the unweighted and balanced configurations:

$$
\begin{aligned}
K_{\mathrm{V1}} &= \operatorname{round}\left(0.02 \times \frac{N_{\mathrm{V1}}}{D_{\mathrm{V1}}}\right) \\
&= \operatorname{round}\left(0.02 \times \frac{65{,}544}{21}\right) \\
&= \operatorname{round}(62.42) \\
&= 62
\end{aligned}
$$

where:

- `K_V1` = the daily review count used for tuning on V1;
- `N_V1` = number of V1 transactions (65,544);
- `D_V1` = number of V1 relative days (21; days 106–126).

$$
\mathrm{Recall@2\%}_{\mathrm{V1}} = \frac{\sum_{d} F_{d}^{\mathrm{top}}}{\sum_{d} F_{d}}
$$

where:

- `d` = a V1 relative day;
- `F_d^top` = number of fraud transactions among the 62 highest-ranked transactions of day d (ranking as in §5);
- `F_d` = number of fraud transactions of day d.

- **Unweighted:** each V1 transaction counts once. The balanced class weights apply in the Train loss only.
- **Use:** the same value drives early stopping (§2) and trial selection (§1).
- **Partition roles:** `K_V1` is computed from V1 volume only and is used for tuning only. It neither replaces nor determines the study's daily review budget K = 56, which is fixed from V2 volume ([DR-002](DR-002-review-capacity-definition.md)).
- **Implementation:** one shared scorer computes the objective for all eight configurations: a custom evaluation function for LightGBM, a user-defined evaluation metric for CatBoost on its GPU learner, and one computation per epoch on the full V1 scores for the neural models. The LightGBM and CatBoost paths require adapter changes and are verified on synthetic data before any development run (Consequences).

### 4. Seeds

- **Search:** every trial trains with the main seed 42 ([DR-013](DR-013-training-device-policy.md) §6).
- **Additional seeds: 43 and 44**, the same for all eight configurations. Each configuration's selected setting is refitted on Train at its selected length (the retained round or epoch count), without early stopping: 2 × 8 = 16 refits. The search is not repeated.
- **Same-seed repeats** ([DR-013](DR-013-training-device-policy.md) §6, item 1): each configuration's selected setting is refitted once more with seed 42, in a fresh process, at its selected length: 8 refits. Its V2 scores are compared with those of the original selected model: identical scores are expected for LightGBM's deterministic training and the neural models' strict deterministic mode; the differences of CatBoost's non-deterministic GPU training are measured.
- **Evaluation:** V2 only, under R3 at the four capacities with the fixed daily budgets; the primary measure is Recall@2% with K = 56. Neither kind of refit is fitted on any partition other than Train.
- **Use:** descriptive only. The range over the three seeds 42, 43 and 44 is compared with the Smallest Effect Size Of Interest (SESOI; 2 percentage points of Recall@2%, [DR-005](DR-005-sesoi-and-h1-h2-inference.md)) as the different-seed variation of DR-013 §6. It never changes the selection, the H3 configuration or any other decision.
- **Neural preprocessing seed: 0.** The quantile-fitting noise of [DR-012](DR-012-neural-network-input-representation.md) is drawn from a dedicated NumPy generator (`numpy.random.default_rng(0)`, PCG64), separate from the candidate sampler (§1) and from model training. The seed does not change for trials, weighting configurations, additional-seed refits or same-seed repeats, so every neural fit uses the same fitted preprocessing ([DR-012](DR-012-neural-network-input-representation.md) sets the draw order). The integer follows the run seed 0 of REF-103's tuning base configurations [L]; holding it fixed and independent of the training seeds is this study's adaptation, because REF-103 passes each run's seed into its preprocessing [REF-103] [I].

### 5. Tie rules

**Equal V1 values**

- Improvement is strict: a checkpoint improves only if its objective is greater than the best so far; an equal value counts as no improvement.
- Equal values across checkpoints of a trial: the earliest checkpoint is kept.
- Equal best values across trials: the lowest trial number is selected.
- Values are compared exactly, with no rounding or tolerance, and no other metric is used as a secondary criterion.

**Equal scores at the daily review cutoff.** Within each relative day, transactions are ordered by:

1. model score, descending, compared exactly;
2. for exactly equal scores, the SHA-256 (Secure Hash Algorithm, 256-bit) hexadecimal digest of the transaction's `TransactionID`, written as its decimal string in UTF-8 (Unicode Transformation Format, 8-bit), ascending;
3. `TransactionID` ascending, only for an identical digest.

The rule applies to the V1 objective (daily top-62), R3 (daily top-K) and R4 (which reviews from R3's queue), for every configuration, seed and capacity. The number of days with tied scores crossing the cutoff, and the size of those groups, are reported. This ordering is used only for review queues: it is not used to split score ties within whole-ranking metrics, and neither `TransactionID` (the transaction row identifier) nor its digest is a model input.

### 6. Search spaces and reference configurations: tree-based models

**Sources.** CatBoost uses the large-dataset space (B) of REF-087 (Table 17), the source and dataset-size class of the neural search spaces ([DR-011](DR-011-model-set-tree-based-and-neural-families.md) §4), with the learning-rate adaptation below. REF-087 does not tune LightGBM; its space follows the later benchmark of the same research group [REF-129, Table 23], which extends the tuning of REF-087. The search distributions (Uniform, LogUniform: continuous uniform and log-uniform; UniformInt, LogUniformInt: their integer forms) are defined under "Sampling distributions" below. Two adaptations apply to both libraries:

- **Learning rate: LogUniform[0.01, 0.3]** for both models, the learning-rate range of the Optuna parameter definitions in REF-089's code [REF-130], which REF-089 used for its additional 100-iteration tuning (Appendix D.7). REF-089's main random search uses different ranges in a separate code path (CatBoost [0.03, 0.3), LightGBM [0.03, 30)) [REF-130]. The range is an engineering adaptation to nine random draws and the 2,000-round maximum (§2) [I], not a reproduction of either search. The source ranges (REF-087 for CatBoost: LogUniform[1e-5, 1]; REF-129 for LightGBM: LogUniform[0.001, 1]) were searched with 100–200 Tree-structured Parzen Estimator trials, whose adaptive phase concentrates later trials in productive regions. Under them, 60 % of draws from LogUniform[1e-5, 1] and one third of draws from LogUniform[0.001, 1] fall below 0.01; both libraries' documentation states that a smaller learning rate needs more iterations [REF-116; REF-118], so such draws would spend a large share of nine trials where the 2,000-round maximum is most likely to bind [I].
- **Leaf-size control.** LightGBM's `min_sum_hessian_in_leaf`, searched in REF-129, is replaced by `min_data_in_leaf`. An absolute Hessian threshold scales with the sample weights, so under balanced weighting the same value constrains leaves on a different scale. LightGBM estimates a candidate leaf's row count from its share of the parent node's Hessian sum, multiplied by the parent's row count [REF-116; REF-117]: the overall weight scale cancels, but class-dependent weights still shift the estimate. The shared candidate list (§1) therefore pairs the same nominal settings, and regularisation can still interact with class weighting; `min_data_in_leaf` is the scale-normalised control of the two [I].

**Sampling distributions.** Each random candidate (trials 2–10, in trial order) draws its parameters independently, in the order of the tables below, from its model's child stream (§1); the reference configuration (trial 1) uses no random draws:

$$
\begin{aligned}
\mathrm{Uniform}[x_{\min}, x_{\max}]: &\quad x = z, && z \sim U[x_{\min}, x_{\max}) \\
\mathrm{LogUniform}[x_{\min}, x_{\max}]: &\quad x = \exp(z), && z \sim U[\ln x_{\min}, \ln x_{\max}) \\
\mathrm{UniformInt}[x_{\min}, x_{\max}]: &\quad x \sim \text{uniform on } \lbrace x_{\min}, x_{\min} + 1, \dots, x_{\max} \rbrace \\
\mathrm{LogUniformInt}[x_{\min}, x_{\max}]: &\quad x = \min\left(\lfloor \exp(z) \rfloor, x_{\max}\right), && z \sim U[\ln x_{\min}, \ln(x_{\max} + 1))
\end{aligned}
$$

where:

- `x` = the sampled parameter value;
- `x_min`, `x_max` = the lower and upper bounds of the range;
- `z` = a continuous uniform draw;
- `U[·, ·)` = the continuous uniform distribution on a half-open interval (NumPy's `Generator.uniform`);
- `⌊·⌋` = rounding down to an integer; the minimum with `x_max` guards against floating-point rounding at the upper end.

UniformInt uses `Generator.integers(x_min, x_max, endpoint=True)`. A **mixture** `0.5 · {0} + 0.5 · LogUniform[x_min, x_max]` draws w ~ U[0, 1) and then z as for LogUniform, and takes x = 0 if w < 0.5 and x = exp(z) otherwise; both draws are always made, so the stream position does not depend on the branch. Every draw is one scalar call (no `size` argument), with NumPy's `log` and `exp`. Candidate values are recorded at full float64 precision, and the recorded candidate list is the reproducibility record: bit-for-bit regeneration in another software environment is not assumed.

**Reference configurations.** Trial 1 of each tree-based model uses the library's default values for the pinned version, written out explicitly, as the default setting that REF-089 includes among its trials [REF-089]. Each model keeps its own default learning rate; no common value is imposed across models, because fairness is the common procedure, not identical values ([DR-011](DR-011-model-set-tree-based-and-neural-families.md) §3). Every reference value lies within its parameter's declared bounds; a reference value may sit at an upper bound that the half-open continuous sampling interval excludes (for example a fraction of 1.0).

**CatBoost's reference learning rate.** For binary log-loss, CatBoost chooses its learning rate automatically when `learning_rate`, `leaf_estimation_method`, `leaf_estimation_iterations` and `l2_leaf_reg` are all unset, from the device, the loss, `use_best_model`, `boost_from_average`, the number of training rows and the iteration count; it uses no labels [REF-119]. The study sets these parameters explicitly ([DR-013](DR-013-training-device-policy.md) §3), so the rule is evaluated once for the training device and recorded as one explicit value, used identically on both devices:

$$
\begin{aligned}
\eta_{\mathrm{ref}} &= \operatorname{round}_6\left(\min\left(\exp\left(c_N \ln N_{\mathrm{Train}} + c_0\right) \cdot \left(\frac{I_{\max}}{1000}\right)^{c_I},\ 0.5\right)\right) \\
&= \operatorname{round}_6\left(\exp\left(-0.085 \ln 368{,}632 - 2.055\right) \cdot 2^{-0.414}\right) \\
&= \operatorname{round}_6\left(0.043091 \times 0.750513\right) \\
&= 0.032340
\end{aligned}
$$

where:

- `η_ref` = CatBoost's reference learning rate;
- `round_6` = rounding to six decimal places;
- `c_N`, `c_0`, `c_I` = CatBoost's coefficients for binary log-loss on the GPU with `use_best_model` on and `boost_from_average` off: −0.085, −2.055 and −0.414 [REF-119];
- `N_Train` = number of Train rows (368,632);
- `I_max` = maximum boosting rounds (2,000; §2).

The CPU coefficients of the same rule give 0.100904; this device dependence is why [DR-013](DR-013-training-device-policy.md) §3 excludes the automatic value. LightGBM has no automatic learning rate; its default is 0.1 [REF-116].

**LightGBM** (central processing unit (CPU); LightGBM 4.7.0):

| Parameter | Search distribution | Reference (trial 1) | Source |
|---|---|---|---|
| `learning_rate` | LogUniform[0.01, 0.3] | 0.1 | [REF-130] (Optuna range; adaptation above); default [REF-116] |
| `num_leaves` | UniformInt[4, 768] | 31 | [REF-129, Table 23]; default [REF-116] |
| `min_data_in_leaf` | LogUniformInt[20, 2000] | 20 | [I]: lower bound the default; upper bound from the documentation's guidance that "hundreds or thousands is enough for a large dataset" [REF-116] |
| `feature_fraction` | Uniform[0.5, 1.0] | 1.0 | [REF-129, Table 23; REF-012, Appendix D.2]; default [REF-116] |
| `bagging_fraction` | Uniform[0.5, 1.0] | 1.0 | [REF-129, Table 23; REF-012, Appendix D.2]; default [REF-116] |
| `lambda_l2` | 0.5 · {0} + 0.5 · LogUniform[0.1, 10] | 0 | Range [REF-129, Table 23]; mixture weight 0.5 [REF-012, Appendix D.2]; default [REF-116] |

`lambda_l2` and `lambda_l1` are LightGBM's L2 (squared-value) and L1 (absolute-value) penalties on leaf values.

Fixed LightGBM settings, set explicitly: `boosting="gbdt"`, `data_sample_strategy="bagging"`, `bagging_freq=1` (bagging is effective only when `bagging_freq` > 0 and `bagging_fraction` < 1.0 [REF-116], so the reference configuration trains without bagging), `max_depth=-1`, `min_sum_hessian_in_leaf=1e-3`, `lambda_l1=0`, `min_gain_to_split=0`, `max_delta_step=0`, `path_smooth=0`, `extra_trees=false`, `feature_fraction_bynode=1.0`, `boost_from_average=true`; categorical controls `max_cat_to_onehot=4`, `max_cat_threshold=32`, `cat_l2=10`, `cat_smooth=10`, `min_data_per_group=100`; early stopping `num_iterations=2000`, `early_stopping_round=50`, `early_stopping_min_delta=0` (§2), with `metric="None"` and the shared V1 objective as a custom evaluation function (§3). All are LightGBM 4.7.0 defaults [REF-116] except `bagging_freq` [I], the early-stopping values (§2) and the metric (§3). The device, dataset-construction and seed settings are those of [DR-013](DR-013-training-device-policy.md) §2.

**CatBoost** (graphics processing unit (GPU); CatBoost 1.2.10):

| Parameter | Search distribution | Reference (trial 1) | Source |
|---|---|---|---|
| `depth` | UniformInt[6, 10] | 6 | [REF-087, Table 17, space (B)]; default [REF-119] |
| `learning_rate` | LogUniform[0.01, 0.3] | 0.032340 | [REF-130] (Optuna range; adaptation above); CatBoost's automatic rule for this setting, recorded as an explicit value (above) [REF-119] |
| `l2_leaf_reg` | LogUniform[1, 10] | 3 | [REF-087, Table 17; REF-012, Appendix D.2]; Logloss default [REF-119] |
| `bagging_temperature` | Uniform[0, 1] | 1 | [REF-087, Table 17; REF-012, Appendix D.2]; default [REF-118] |
| `leaf_estimation_iterations` | UniformInt[1, 10] | 10 | [REF-087, Table 17]; default under Logloss with Newton leaf estimation [REF-119] |

Fixed CatBoost settings of [DR-013](DR-013-training-device-policy.md) §3, set explicitly and identically on both devices:

- `random_strength=1`, `model_size_reg=0.5` and `permutation_count=4`: library defaults [REF-119];
- `fold_permutation_block=1`: an explicit positive setting, as DR-013 §3 requires (the source default 0 is resolved differently by device) [I]; both permutation controls are inactive under `has_time`;
- `boost_from_average=false`: the default for binary log-loss, set explicitly because CatBoost resolves it by loss function and it enters the automatic learning-rate rule [REF-119];
- `eval_metric` = the shared V1 objective as a user-defined metric (§3); `metric_period=1`, because the objective is evaluated after every boosting round (§2);
- `iterations=2000` and `od_wait=50` (§2).

`random_strength` is not searched, so CatBoost's space is REF-087's space (B) with the learning-rate adaptation above. The remaining CatBoost settings are those of [DR-013](DR-013-training-device-policy.md) §3.

### 7. Search spaces, reference configurations and training values: neural models

**Sources.** Both neural models use the large-dataset space (B) of REF-087 ([DR-011](DR-011-model-set-tree-based-and-neural-families.md) §4): Table 14 for the ResNet-style MLP and Table 13 for FT-Transformer, with the distributions as the authors' code samples them [REF-103: `bin/tune.py` and the Covertype tuning configurations]. One adaptation applies: space (B) has no category-embedding entry for the ResNet-style MLP, because its datasets have no categorical features, so the embedding width is searched over the range REF-087 uses for Adult, its one dataset with categorical features [REF-087, Table 14; REF-103, Adult configuration] [I]. This keeps the category-vector size of the ResNet-style MLP searched over the same range as FT-Transformer's token width.

**Sampling.** The distributions, draw order and recording rules of §6 apply: each random candidate (trials 2–10) draws its parameters in the order of the tables below, and trial 1 uses no random draws. Two forms are added:

- **Token width:** `d_token` (the length of each FT-Transformer token vector) is uniform over the 57 multiples of 8 from 64 to 512 (8 · UniformInt[8, 64]), because the width must divide among the 8 attention heads [REF-103: `bin/tune.py`, step `n_heads`].
- **Optional parameters:** `0.5 · {0} + 0.5 · Uniform[x_min, x_max]` follows the mixture rule of §6 with a continuous uniform draw in place of the log-uniform one. The probability 0.5 is the authors' choice between "off" and "on" (`suggest_categorical([False, True])`) [REF-103].

The parameters are: `n_layers`, the number of residual blocks or Transformer layers; `d`, the ResNet block width (a parameter name, distinct from the relative day d of §3); `d_hidden_factor`, the ratio of a block's hidden width to `d`; `d_embedding` and `d_token`, the lengths of the ResNet's category-embedding vectors and of FT-Transformer's token vectors; `hidden_dropout`, `residual_dropout`, `attention_dropout` and `ffn_dropout`, the dropout probabilities inside a ResNet block, on a residual branch, on the attention weights and inside the feed-forward module; `d_ffn_factor`, the ratio of the feed-forward hidden size to `d_token`; `lr`, the learning rate; and `weight_decay`, AdamW's decoupled weight-decay coefficient.

**ResNet-style MLP** (GPU):

| Parameter | Search distribution | Reference (trial 1) | Source |
|---|---|---|---|
| `n_layers` | UniformInt[1, 16] | 2 | [REF-087, Table 14 (B); REF-103]; reference [REF-130] |
| `d` | UniformInt[64, 1024] | 128 | [REF-087, Table 14 (B); REF-103]; reference [REF-130] |
| `d_hidden_factor` | Uniform[1, 4] | 2 (hidden width 256) | [REF-087, Table 14 (B); REF-103]; reference [REF-130] |
| `hidden_dropout` | Uniform[0, 0.5] | 0.25 | [REF-087, Table 14 (B); REF-103]; reference [REF-130] |
| `residual_dropout` | 0.5 · {0} + 0.5 · Uniform[0, 0.5] | 0.1 | [REF-087, Table 14 (B); REF-103]; reference [REF-130] |
| `d_embedding` | UniformInt[64, 512] | 192 | Range [REF-087, Table 14 (Adult); REF-103] [I]; reference [I] (below) |
| `lr` | LogUniform[1e-5, 1e-2] | 1e-3 | [REF-087, Table 14 (B); REF-103]; reference [REF-130] |
| `weight_decay` | 0.5 · {0} + 0.5 · LogUniform[1e-6, 1e-3] | 0 | [REF-087, Table 14 (B); REF-103]; reference [I] (below) |

The hidden width of each residual block is int(`d` × `d_hidden_factor`); the parameter `d` is the block width, distinct from the relative day d of §3. Every categorical input uses the same `d_embedding` within a trial. Weight decay applies to every parameter, as in REF-103.

**FT-Transformer** (GPU):

| Parameter | Search distribution | Reference (trial 1) | Source |
|---|---|---|---|
| `n_layers` | UniformInt[1, 6] | 3 | [REF-087, Table 13 (B); REF-103]; reference [REF-087, Table 12] |
| `d_token` | 8 · UniformInt[8, 64] | 192 | [REF-087, Table 13 (B); REF-103]; reference [REF-087, Table 12] |
| `attention_dropout` | Uniform[0, 0.5] | 0.2 | [REF-087, Table 13 (B); REF-103]; reference [REF-087, Table 12] |
| `ffn_dropout` | Uniform[0, 0.5] | 0.1 | [REF-087, Table 13 (B); REF-103]; reference [REF-087, Table 12] |
| `lr` | LogUniform[3e-5, 3e-4] | 1e-4 | [REF-087, Table 13 (B); REF-103]; reference [REF-087, Table 12; REF-131] |
| `weight_decay` | LogUniform[1e-6, 1e-3] | 1e-5 | [REF-087, Table 13 (B); REF-103]; reference [REF-087, Table 12; REF-131] |

Fixed for every FT-Transformer candidate, as space (B) and Table 12 specify [REF-087]: `residual_dropout=0`, `d_ffn_factor=4/3`, 8 attention heads, the ReGLU (rectified gated linear unit) activation, pre-normalisation and Kaiming initialisation; weight decay is 0 for the feature tokenizer, every LayerNorm and every bias.

**Reference configurations.** Trial 1 of each neural model is based on a published default, as the tree models use their libraries' defaults (§6):

- **FT-Transformer:** the default configuration of REF-087 (Table 12), which the paper also evaluates untuned; the same learning rate and weight decay are the defaults of the authors' `rtdl` package [REF-131].
- **ResNet-style MLP:** REF-087 gives no default configuration. Trial 1 uses the default ResNet of the TabZilla benchmark [REF-130: `rtdl_ResNet`], which REF-089 evaluates untuned and reports among the algorithms that "perform fairly well both with and without hyperparameter tuning" (Appendix D.2.1) [REF-089]. TabZilla builds it with the `rtdl` package's ResNet, whose blocks apply `dropout_first` after the activation and `dropout_second` after the second linear layer, as REF-087's ResNet block applies its hidden and residual dropout [REF-131]; its `dropout_first` 0.25 and `dropout_second` 0.1 are therefore `hidden_dropout` and `residual_dropout`. Two TabZilla values lie outside space (B) and are replaced [I]:
  - its category-embedding width of 8 by **192**, the token width of the default FT-Transformer, so both neural reference configurations represent each category with a vector of 192 numbers;
  - its weight decay, which TabZilla inherits from PyTorch's AdamW default of 0.01 [REF-130; REF-120], by **0**, the zero branch of space (B).

**Optimiser.** AdamW (Adam with decoupled weight decay) with `betas` (0.9, 0.999), the decay coefficients of its first- and second-moment estimates; `eps` 1e-8, its numerical-stability constant; AMSGrad, the variant that keeps the maximum of past second moments, disabled; and the single-tensor implementation (`foreach=False`, `fused=False`), set explicitly, with no learning-rate schedule [REF-087, Section 4; REF-103].

**Effective batch.** Both neural models update on an **effective batch of 1,024 Train rows**, the batch size REF-087 uses for every dataset with more than 300,000 training rows (Table 7; Covertype: 371,847) [REF-087; REF-103]. Train has 368,632 rows, so each epoch has 360 updates: 359 full batches and a final batch of 1,016 rows, which is kept. Each epoch's row order is drawn from the training seed, as in the neural loop.

**Micro-batches and gradient accumulation.** A micro-batch is the part of an effective batch that the GPU processes at once. Before the search, a synthetic, data-free preflight on the project GPU fixes the micro-batch of each of a model's 10 candidates [I]:

1. The preflight uses the production implementation with full-size synthetic inputs and vocabularies, strict determinism, the precision and attention backend of [DR-013](DR-013-training-device-policy.md) §4, and initialised optimiser state.
2. It tries micro-batches of 1,024, 512, 256, 128 and 64 rows in that order and records the largest that completes forward, backward and optimiser steps with at least 1,024 MiB (mebibytes) of device memory free at the sampled peak; training and float32 evaluation are checked separately.
3. The ResNet-style MLP must run each candidate at 1,024 directly, because its BatchNorm statistics are computed over the whole batch; FT-Transformer, whose LayerNorm acts on each row, accumulates gradients when its micro-batch is smaller.
4. The probe evidence supports expecting every FT-Transformer candidate to pass at 64 or more [I]: the largest setting of space (B) trained at 64 in the acceptance probe, with about 1,363 MiB of the 8,151 MiB device free at the sampled peak ([`gpu-acceptance-probe.md`](../gpu-acceptance-probe.md), Section 7), and smaller settings need less memory. The probe did not exercise every width or the accumulation path, so this is not established for every candidate: each candidate must pass the preflight, in training and in float32 evaluation, at a listed size. A candidate that passes at no listed size (or, for the ResNet-style MLP, not at 1,024) stops the search for an issue; no candidate is dropped or replaced.
5. The preflight uses its own generator and leaves the training random state unchanged. The recorded plan is used for the unweighted and balanced configurations and for every refit (§4), and is not changed after V1 results are seen.

With accumulation, every micro-batch adds its gradients in float32, and AdamW updates once per effective batch. The update follows the effective-batch loss below; it equals a whole-batch update in exact arithmetic with the same dropout masks. In ordinary execution the dropout masks and floating-point reduction order depend on the micro-batch plan, so repeatability is required for the recorded plan, not equality with an unsplit batch.

**Loss.** Training minimises the weighted binary cross-entropy of the effective batch, divided by its actual row count:

$$
L_{\mathrm{batch}} = \frac{1}{m} \sum_{i \in \mathcal{B}} w_i \, \ell_i
$$

where:

- `L_batch` = the training loss of one effective batch;
- `𝓑` = the Train rows of the effective batch;
- `i` = the index of a Train row in the effective batch;
- `m` = its actual row count (1,024, or 1,016 for the final batch);
- `w_i` = the explicit sample weight of row i (1 for every row in the unweighted configurations; the balanced weights of the protocol otherwise);
- `ℓ_i` = the binary cross-entropy of row i, computed from its logit in float32 ([DR-013](DR-013-training-device-policy.md) §4).

Every micro-batch divides its weighted sum by the same m. The row count is the denominator, not the batch's weight sum: it gives an unbiased mini-batch estimate of the mean weighted Train loss (for the ResNet-style MLP, up to BatchNorm's dependence of each row's loss on its batch) and needs no information from other micro-batches, whereas a weight-sum denominator changes with the number of frauds in each batch [I]. Training-loss normalisation does not change the V1 objective (§3), which is unweighted.

**Other values.**

- **Epochs:** at most 15, patience 5 (§2); with the best V1 value at epoch 1, the earliest stop is at epoch 7.
- **Evaluation batch:** 4,096 rows for the ResNet-style MLP and 16 for FT-Transformer, the workloads exercised in the acceptance probe, fixed per model for every candidate [I]. It is fixed and recorded because it affects memory and speed and can also change floating-point scores; repeatability is checked with identical batch partitions.

**Compute estimate** [D] [I]. The per-epoch times are the three-epoch means of the explicitly identified fits in acceptance-probe run 5 (`FTT-u default fit 1`, `FTT-b largest fit 1`, `RES-u largest fit 1`; 16,384 Train rows and 4,096 V1 rows per epoch), scaled to the study's partitions: Train time × 368,632 / 16,384 (22.50) and V1 evaluation time × 65,544 / 4,096 (16.00).

| Setting | Train (min per epoch) | V1 evaluation (min per epoch) | Total (min per epoch) |
|---|---|---|---|
| FT-Transformer default (3 layers, `d_token` 192) | 6.95 | 0.69 | 7.64 |
| FT-Transformer largest (6 layers, `d_token` 512) | 40.1 | 4.39 | 44.5 |
| ResNet-style MLP largest (16 layers, `d` 1,024, `d_embedding` 64) | 0.42 | 0.065 | 0.48 |

Interpolating the per-layer time linearly in `d_token` between the two FT-Transformer settings gives about 14.02 minutes per epoch for the average candidate of space (B) (mean `d_token` 288, mean `n_layers` 3.5). The search time of the two FT-Transformer configurations is then:

$$
\begin{aligned}
T_{\mathrm{FTT}}(E) &= 2 \cdot E \cdot \left(t_{\mathrm{ref}} + 9 \, \bar{t}\right) \\
&= 2 \cdot E \cdot (7.64 + 9 \times 14.02)\ \text{min} \\
&\approx 2 \cdot E \cdot 133.8\ \text{min}
\end{aligned}
$$

where:

- `T_FTT(E)` = the search time of FT-Transformer's two configurations;
- `E` = the number of epochs per trial (7 to 15);
- `t_ref` = the epoch time of the reference configuration (7.64 minutes);
- `t̄` = the epoch time of an average candidate (about 14.02 minutes);
- 9 = the random candidates per configuration.

This gives about **31.2 hours at 7 epochs** and **66.9 hours at 15 epochs**. The refits of §4 (three per configuration, fixed length, no per-epoch V1 evaluation) add up to about 60 hours if the largest setting is selected and retained for 15 epochs (6 × 15 × 40.1 minutes). The ResNet-style MLP takes at most about 2.5 hours for its search at the measured embedding width. These are estimates, not bounds: they do not measure accumulation, padding or the cheaper final FT-Transformer block. The predicted cost of the actual candidate lists is computed from the sampler before any fit and recorded; no list is redrawn and no candidate is rejected for its cost. If the predicted cost is not feasible, the search is paused for an explicit decision before any run.

### 8. Search and fitting of the logistic-regression reference

The logistic-regression reference of [DR-011](DR-011-model-set-tree-based-and-neural-families.md) §5 shares the search budget, the V1 objective (§3) and the trial tie rule (§5) of the eight configurations, with convergence-based fitting in place of checkpoint-based early stopping:

- **Candidates.** Ten: trial 1 uses C (inverse regularisation strength) = 1, the library default; trials 2–10 draw C from LogUniform[10⁻⁴, 10⁴] with the sampling formula of §6. The range is that of scikit-learn's default `LogisticRegressionCV` grid (`np.logspace(-4, 4, Cs)`, where `Cs`, when an integer, is the number of grid values) [REF-135]; the nine log-uniform draws are this study's adaptation [I], not a claim that this range or budget is sufficient for the dataset.
- **Sampler stream.** The candidates come from a fifth child stream of the sampler of §1, spawned from the same parent after the four model streams (child (4,)). NumPy's `SeedSequence.spawn` continues the parent's child counter, so the four existing streams, and with them the candidate lists of §6 and §7, are unchanged [REF-136].
- **Fitting.** Each candidate is fitted on Train to convergence with scikit-learn 1.9.1 `LogisticRegression`: `solver="lbfgs"`, L2 penalty through `l1_ratio=0`, `fit_intercept=True`, `class_weight=None` and `warm_start=False` [REF-135] (`penalty` is deprecated in this version); the study fixes `tol=1e-5` and `max_iter=5000`, a tighter tolerance and a larger iteration allowance than the library defaults (1e-4 and 100) [I; Issues #115 and #151], which do not guarantee convergence. In scikit-learn 1.9.1 the solver passes `tol` to SciPy's L-BFGS-B as its gradient stopping threshold, with the function-change threshold fixed separately at about 1.4e-14 [REF-135], so a converged fit is not claimed to give probabilities accurate to `tol`; it is a fit that met the declared stopping criteria. Individual fits are used rather than `LogisticRegressionCV`'s cross-validation. The iteration count, settings and warnings are recorded. Non-convergence, non-finite coefficients or invalid probabilities (non-finite or outside [0, 1]) stop the analysis for an issue, with no solver substitution, candidate replacement or iteration extension, and no fallback to other settings: a change of the numerical settings applies to all ten candidates and is made only through an issue.
- **Candidate check.** Before the first development run, the ten recorded candidates are fitted on Train alone with these settings; the iteration count, the time, the warnings and the finiteness of coefficients and probabilities are recorded. V1 plays no part in it, and no setting is chosen by a validation result.
- **Selection.** The highest V1 objective; exact ties to the lowest trial number. The fit is deterministic, so the seed refits and same-seed repeats of §4 do not apply.

### 9. Computational-cost procedure

The protocol's computational-cost outcome (Outcomes) is measured on the development partitions under the following procedure, fixed before any development run (Issue #121). The timings are descriptive and never enter model, threshold or policy selection. The values are the `timing` block of `config/training.yaml`.

- **Devices.** The tree-based models are trained on their study devices (LightGBM on the CPU, CatBoost on the GPU) and score on the CPU ([DR-013](DR-013-training-device-policy.md) §5); the ResNet-style MLP and FT-Transformer train and score on the GPU only. No neural model is timed on the CPU. The timings of the tree-based and the neural models are therefore **not a same-device comparison**: each is taken on its study device, and a difference reflects model, software, input representation, batching and hardware together.
- **Fit.** The timed fit is the **selected main-seed trial's own elapsed time during the search**, including early stopping and the discarded patience rounds or epochs; no extra fit is made for timing. It runs from the construction of the fit-specific data containers to a fitted model that is ready to predict with its device work complete. Process, library and CUDA start-up, the shared preprocessing and model persistence are recorded beside it, not inside it. The completed length and the retained length of the fit are both recorded.
- **Search.** The search time of a configuration is the elapsed time of its sequential ten-trial loop (trial set-up, fit, V1 objective, recording and selection). Scoring benchmarks and the descriptive refits are outside it. An interrupted search is recorded as interrupted, with its completed and declared trial counts, and never as a completed ten-trial measurement; a search that was not interrupted has completed every declared trial.
- **Scoring and throughput.** The selected main-seed model scores the complete V1 and V2 partitions and returns probabilities (not logits). Two boundaries are reported separately: **model-only** (prepared native inputs to scores on the prediction device) and **end-to-end** (conversion from the in-memory features, application of the already fitted preprocessing artifact, native containers, host–device transfers, prediction and scores returned on the host). Disk loading, model loading, preprocessing fitting, calibration, ranking and metrics are outside both. The neural evaluation batches are those of §7 (4,096 rows for the ResNet-style MLP, 16 for FT-Transformer, including the final short batch); the tree-based models score each partition in one call. Warm-up is untimed: **one full prediction per boundary** for the tree-based models and **three batch calls** (the normal batch shape and the final short batch) for the neural models. Each partition and boundary is timed over **3 timed full passes**; the median, every pass and their range are stored, slow passes are kept, and throughput is the rows of the partition divided by the median seconds.
- **Latency.** Single-transaction latency is the time of one batch-of-one scoring call, at both boundaries, on **10,000 distinct V1 rows**: those with the smallest SHA-256 digests of the decimal `TransactionID` written in UTF-8 (identical digests by ascending identifier), in that fixed order, with the SHA-256 identity of the ordered rows stored. Neither labels nor scores enter the selection. Each block begins with **100 untimed warm-up calls**, and the rows are passed **3 rounds** per boundary; for each round the median, 95th (p95) and 99th (p99) percentile are computed with `numpy.percentile` using the `linear` method. p99 is a descriptive empirical estimate with limited tail support, and the measure is serial scoring latency (one call at a time): queueing and request arrival are not modelled. The tree-based models use their production prediction thread counts (24, `config/models.yaml`); there is no thread sweep.
- **Threads and affinity.** The effective CPU affinity is recorded with every CPU timing and is not changed; CPU identifiers inside the WSL2 (Windows Subsystem for Linux 2) guest are not assumed to correspond to physical performance cores.
- **GPU synchronisation.** Neural timings synchronise the explicit CUDA device before the clock starts and after the work completes, around each full pass and around each latency call. CPU timings use no synchronisation.
- **Session conditions.** The timings are taken under the machine's working conditions at the time and are valid descriptive measurements: no condition is required, and other local workloads may run during a timing. The conditions are recorded as context with the run: a **60-second settling interval** before each standalone scoring block, the one-minute load average, the visible and allowed CPUs, the kernel, the WSL2 memory and swap limits, and GPU temperature and clock data where the system exposes them. A timing is read together with its recorded context and reflects the machine's conditions as well as the model.
- **GPU memory.** The absolute sampled whole-device maximum, the idle baseline and the increase above the baseline are stored separately and called a **sampled maximum**, not a true peak. The neural models also record the allocator's maximum allocated and reserved bytes after a reset at the start of the phase. The libraries of the tree-based models are invisible to those counters, which are the allocator counters of PyTorch on CUDA (NVIDIA's GPU computing platform).
- **Host memory.** The peak resident memory of each phase is the process high-water mark (`VmHWM`) read after resetting it by writing `5` to `/proc/self/clear_refs` (which resets it to the current resident size, not to zero; the check resets first, so that an earlier, higher mark cannot hide it). Whether the reset works on the project machine is checked with a synthetic allocation in the first run; where it does not, a sampler thread reads the resident size **every 100 ms** (the `timing` block) and the record names the method and the interval; a failed sampler or one that does not stop invalidates the phase's measurement instead of returning a peak. The scope (parent process, resident inputs, models, and whether warm-up lies inside the interval) is stated by the caller with each value.
- **Test period.** Test scoring time is obtained only by instrumenting the single authorised Test scoring pass after the protocol freeze; no Test pass is repeated for timing.
- **Records.** One versioned `timing` record per measurement; a scoring record covers the whole partition and its throughput is the rows divided by the median seconds; an FT-Transformer timing names the efficient attention backend of [DR-013](DR-013-training-device-policy.md) §4; the free `settings` map holds the procedure version, the durations of every pass, the number of distinct rows and of timed calls, the percentile method, the memory method, baseline and maxima, and the session conditions.

## Evidence

- **[D]** Logistic-regression reference, trial 1 (`C = 1`) on processed Train (368,632 rows; the shared preprocessing artifact of DR-012; a matrix of 17,554 columns and 159,617,656 stored values), scikit-learn 1.9.1, `solver="lbfgs"` (Issue #151, 2026-10-23): with `tol=1e-6` and `max_iter=1000` the solver stopped without converging after 1,000 iterations (237 s); with `tol=1e-5` and `max_iter=5000` it converged in 1,377 iterations (344 s); with `tol=1e-4` and `max_iter=5000` in 550 iterations (130 s). The diagnostic establishes feasibility for `C = 1` only; it compared no probabilities and no other candidate.
- **[L]** In scikit-learn 1.9.1 the lbfgs solver calls SciPy's `L-BFGS-B` with `gtol` equal to `tol`, `ftol` fixed at `64 × float64 epsilon`, `maxiter` equal to `max_iter` and `maxls` 50, on the objective averaged over the rows [REF-135]. The installed source was read for this record.
- **[L]** REF-087 tunes with a Tree-structured Parzen Estimator in Optuna: 100 iterations for most models and 50 for FT-Transformer on large datasets, neural patience 16, and CatBoost with 2,000 iterations and 50 early-stopping rounds. Its Appendix D.2 reports good FT-Transformer results after a few randomly sampled configurations [REF-087].
- **[L]** REF-089 allows each algorithm at most 30 hyperparameter settings per dataset split (one default and up to 29 random), within ten hours per algorithm and two hours per fit: budget ceilings, not equal completed trial counts [REF-089].
- **[L]** REF-128 gives empirical and theoretical support for random search over grid search in the settings it examines [REF-128].
- **[L]** REF-087 tunes CatBoost on GPU with 2,000 iterations and 50 early-stopping rounds over depth (UniformInt[3, 10] for smaller datasets, UniformInt[6, 10] for the large-dataset group (B)), learning rate LogUniform[1e-5, 1], bagging temperature Uniform[0, 1], L2 leaf regularisation LogUniform[1, 10] and leaf-estimation iterations UniformInt[1, 10] (Table 17, Appendix F.4) [REF-087].
- **[L]** REF-129 tunes LightGBM over `feature_fraction` and `bagging_fraction` Uniform[0.5, 1.0], `lambda_l2` {0, LogUniform[0.1, 10.0]}, `learning_rate` LogUniform[0.001, 1.0], `num_leaves` UniformInt[4, 768] and `min_sum_hessian_in_leaf` LogUniform[0.0001, 100.0], and CatBoost with depth UniformInt[3, 14], each with 200 Tree-structured Parzen Estimator iterations, 4,000 trees and 200 early-stopping rounds (Tables 23–24, Appendix D.18–D.19) [REF-129].
- **[L]** REF-012 tunes LightGBM and CatBoost with 50 Tree-structured Parzen Estimator steps; its LightGBM space includes `feature_fraction` and `bagging_fraction` Uniform[0.5, 1] and L1 and L2 penalties as the mixture 0.5 · degenerate at 0 + 0.5 · log-uniform, and its CatBoost space includes `l2_leaf_reg` log-uniform [1, 10], `bagging_temperature` Uniform[0, 1] and `gradient_iterations` as a "Discrete uniform distribution over a set {1, 10}" (arXiv v5, Appendix D.2) [REF-012].
- **[L]** The Optuna parameter definitions in REF-089's code sample the learning rate of both LightGBM and CatBoost from LogUniform[0.01, 0.3]; its separate random-search code draws CatBoost learning rates over [0.03, 0.3) and LightGBM learning rates over [0.03, 30) [REF-130]. LightGBM documents `min_data_in_leaf` as "an approximation based on the Hessian" [REF-116]; its split search estimates each side's row count as the side's Hessian sum times the parent's ratio of rows to Hessian sum [REF-117]. CatBoost's automatic learning rate for log-loss is a fixed function of device, `use_best_model`, `boost_from_average`, training-row count and iteration count, applied only when the learning-rate, leaf-estimation and L2 parameters are unset [REF-119]. LightGBM's tuning guide states that a reduced iteration count calls for a larger learning rate, and that `min_data_in_leaf` of "hundreds or thousands is enough for a large dataset" [REF-116]; CatBoost's guide states that a smaller learning rate requires more iterations and recommends depths of 6 to 10 [REF-118]. In CatBoost 1.2.10, binary log-loss defaults to ten Newton leaf-estimation iterations and L2 regularisation 3, the base learning-rate default is 0.03, and the default depth is 6 [REF-119].
- **[L]** REF-087's space (B) for the ResNet-style MLP (Table 14) is `n_layers` UniformInt[1, 16], layer size UniformInt[64, 1024], hidden factor Uniform[1, 4], hidden dropout Uniform[0, 0.5], residual dropout {0, Uniform[0, 0.5]}, learning rate LogUniform[1e-5, 1e-2] and weight decay {0, LogUniform[1e-6, 1e-3]}, with category embedding size UniformInt[64, 512] for Adult only; for FT-Transformer (Table 13) it is `n_layers` UniformInt[1, 6], embedding size UniformInt[64, 512], residual dropout Const(0.0), attention and FFN (feed-forward network) dropout Uniform[0, 0.5], FFN factor Const(4/3), learning rate LogUniform[3e-5, 3e-4] and weight decay LogUniform[1e-6, 1e-3]. Its default FT-Transformer (Table 12) has 3 layers, embedding size 192, 8 heads, attention, FFN and residual dropout 0.2, 0.1 and 0.0, AdamW with learning rate 1e-4 and weight decay 1e-5 (0 for the feature tokenizer, LayerNorm and biases). It uses AdamW without learning-rate schedules and a predefined batch size per dataset, 1,024 for every dataset above 300,000 training rows (Section 4; Table 7) [REF-087].
- **[L]** REF-103's sampler chooses each optional parameter's branch with `suggest_categorical([False, True])` and samples `d_token` in steps of `n_heads`; its Covertype configurations use batch size 1,024 for both models, and its Adult ResNet configuration searches `d_embedding` UniformInt[64, 512] [REF-103].
- **[L]** TabZilla's default ResNet has 2 blocks, main width 128, hidden width 256, `dropout_first` 0.25, `dropout_second` 0.1, category embedding width 8 and learning rate 0.001, trained with PyTorch's AdamW without an explicit weight decay [REF-130]; AdamW's documented default weight decay is 0.01 [REF-120]. REF-089 ranks every algorithm's default setting separately and reports that "CatBoost, XGBoost, LightGBM, and ResNet, perform fairly well both with and without hyperparameter tuning" (arXiv v4, Appendix D.2.1) [REF-089]. The `rtdl` ResNet block applies `dropout_first` after the activation and `dropout_second` after the second linear layer, and the package's default FT-Transformer optimiser uses learning rate 1e-4 and weight decay 1e-5 [REF-131].
- **[D]** GPU memory: in acceptance-probe run 5, the largest FT-Transformer setting of space (B) trained at batch 64 and the default at 256, with sampled whole-device memory at most 6,788 of 8,151 MiB; the largest ResNet-style MLP setting trained at 1,024 ([`gpu-acceptance-probe.md`](../gpu-acceptance-probe.md), Section 7). The Issue #50 probe found, at 800 tokens with bf16 (bfloat16) and memory-efficient attention, largest batches from 64 (6 layers, `d_token` 512) to 2,048 (1 layer, `d_token` 64) ([`evidence/gpu_probe/`](../../evidence/gpu_probe/README.md)).
- **[D]** Training cost per full Train epoch from the GPU acceptance probe (Context; §7); daily transaction volume per partition from DR-001 (V1: 65,544 transactions over 21 days).
- **[I]** Assuming independent random draws and a high-performing region with probability mass p under the sampling distribution, the probability of at least one draw in that region is:

  $$
  P_{\mathrm{hit}} = 1 - (1 - p)^{n}
  $$

  where:

  - `P_hit` = probability of at least one draw in the region;
  - `p` = probability mass of the region under the sampling distribution;
  - `n` = number of independent random draws.

  The ten-trial design has nine random draws and one fixed reference configuration. With p = 0.05, the random component reaches the region with probability about 37.0 %; the reference configuration's contribution is outside this calculation, which illustrates search breadth and does not measure optimisation quality.

## Alternatives considered

| Alternative | Why not chosen |
|---|---|
| `tol=1e-6` with `max_iter=1000` for the logistic-regression reference | Did not converge for `C = 1` on Train within the iteration limit (Evidence) |
| `tol=1e-4` (the library default) with `max_iter=5000` for the reference | Converges faster (550 iterations for `C = 1`), but a looser gradient threshold leaves rare-category and nearly collinear directions less well determined; the tighter `1e-5` converged for `C = 1` at a cost of about six minutes [I] |
| Column scaling, dropping a category, or a solver other than lbfgs to make the reference converge | Changes the penalised representation or the declared fit; DR-011 §5 requires every category and no further scaling [I] |
| An automatic fallback to looser settings when a candidate does not converge | Would mix tolerances across candidates; the stop rule requires an issue and one setting for all candidates [I] |
| Tree-structured Parzen Estimator search | With 10 trials, little information is available for its adaptive phase; random search keeps one sampling procedure for every configuration and allows the paired candidate lists |
| A predefined grid of configurations | Leaves the choice of combinations to manual selection, with sparse joint coverage of multidimensional spaces [REF-128] |
| Equal compute time per configuration | Fast configurations would receive many more trials than slow ones; equal trial counts are the fairness criterion |
| 30 trials with time limits per configuration and per trial | The time limits bind for FT-Transformer, which would complete fewer trials than the other models, breaking the equal-trial criterion |
| 6 trials per configuration | One reference configuration and five random draws: less random exploration (about 22.6 % under the illustration of the Evidence section) |
| The same patience count in rounds and in epochs (for example 6) | A boosting round is far smaller than an epoch; a short round count would stop the tree-based models early |
| A 12-epoch neural maximum | Fifteen epochs allows more training for slow-learning neural settings; actual training length remains subject to early stopping |
| Average precision (non-interpolated area under the precision–recall curve) as the V1 objective | A whole-ranking measure: it also rewards the order of transactions that are never reviewed under limited capacity, and a higher value need not mean higher Recall@2% |
| The area under the Receiver Operating Characteristic curve (ROC-AUC) or log-loss as the V1 objective | ROC-AUC summarises the ordering of frauds against legitimate transactions across the whole ranking, without targeting the daily review cutoff; log-loss also evaluates the probability scale, which balanced training changes by design |
| Recall at the V2 budget K = 56 on V1 | Would carry a V2-derived quantity into tuning; `K_V1` is computed from V1 volume |
| Weighted V1 objective | Changes the evaluated population; applied to the balanced configurations only, it would also change the selection criterion between the weightings compared by H2 |
| Re-ranking the best trials with the additional seeds | Additional seeds measure variation only and never influence selection ([DR-013](DR-013-training-device-policy.md) §6) |
| Score ties broken by `TransactionID` alone | `TransactionID` increases with time, so earlier transactions would always be favoured |
| Reviewing every tied transaction, or a fresh random order per run | The first breaks the daily budget; the second is not reproducible |
| The source learning-rate ranges (CatBoost LogUniform[1e-5, 1] [REF-087]; LightGBM LogUniform[0.001, 1] [REF-129]) | Designed for 100–200 adaptive trials. Under nine random draws, these ranges place a large share of draws below 0.01, where training needs more iterations; adequate convergence within 2,000 rounds is not established for such values [I] |
| LightGBM `min_sum_hessian_in_leaf` searched, as in REF-129 | An absolute Hessian threshold scales with the sample weights, so one value from the shared candidate list would act on a different scale in the unweighted and balanced configurations |
| Tuned or benchmark-specific defaults as reference configurations (for example TabZilla's LightGBM default with 512 leaves and learning rate 0.08 [REF-130]) | A project-chosen starting point; the library defaults are documented, reproducible and lie within the declared bounds |
| One common reference learning rate for all models | Not meaningful across families ([DR-011](DR-011-model-set-tree-based-and-neural-families.md) §3); between the two tree libraries the same value has different effects (leaf-wise trees with 31 leaves against symmetric depth-6 trees with ten Newton steps), and any shared value would match one library's default and not the other's |
| CatBoost reference learning rate 0.03 (its base default) | 0.03 applies only because the leaf-estimation and L2 parameters are set explicitly; 0.032340 is CatBoost's own choice for this configuration, data size and iteration count |
| CatBoost's automatic learning rate left active | Device-dependent (0.032340 on the GPU, 0.100904 on the CPU), excluded by [DR-013](DR-013-training-device-policy.md) §3 |
| CatBoost depth UniformInt[3, 14] [REF-129] | Leaves the space (B) of REF-087 used for the neural models, and adds deep symmetric trees whose GPU memory on this hardware is unmeasured |
| CatBoost `random_strength` searched, as in REF-012 | Not tuned in REF-087's space (B); one fewer dimension for nine random draws |
| LightGBM categorical controls (`cat_smooth`, `max_cat_threshold`, `min_data_per_group`) searched | Not tuned in any of the cited spaces; fixed at the documented defaults with native categorical handling ([DR-008](DR-008-feature-declaration-and-model-input-policy.md) §5) |
| LightGBM `lambda_l1` searched, as in REF-012 and REF-130 | Not tuned in REF-129; fewer dimensions for nine random draws, with L2 regularisation searched |
| One fixed small batch for every FT-Transformer candidate (64, the largest setting's limit) | Departs from REF-087's batch for datasets of this size; changes the number of updates, the optimisation noise and the cumulative weight decay; slower for small candidates and not measured |
| A different batch size per candidate, set by its memory limit | Batch size would vary with the architecture and confound the comparison of candidates; accumulation keeps the effective batch fixed |
| Gradient checkpointing instead of accumulation | Needs a revised probe and adds recomputation; accumulation reaches the same effective batch within the measured memory limits |
| ResNet trial 1 from the commented ResNet example in the `rtdl` notebook | A usage example with no reported evaluation; TabZilla's default ResNet, built on the same package, is evaluated untuned in REF-089 |
| TabZilla's ResNet default unchanged | Its category-embedding width (8) and inherited weight decay (0.01) lie outside space (B) |
| ResNet `d_embedding` fixed for every candidate | REF-087 tunes the embedding size whenever categorical features exist; fixing it would also leave the ResNet's category-vector size untuned while FT-Transformer's is searched |
| ResNet `d_embedding` searched over [192, 512] | Not a published range; removes short vectors, which can overfit less on rare categories |
| Weight-sum denominator for the training loss | A self-normalised ratio estimator whose denominator varies with the number of frauds in each batch; dividing each micro-batch by its own weight sum would not implement effective-batch normalisation. Reduction A gives the additive row-count objective |
| `LogisticRegressionCV` with its default grid and internal cross-validation for the logistic-regression reference | Individual Train fits with one default candidate and nine log-uniform draws keep the shared search budget and V1 selection procedure [I] |
| Neural preprocessing seed 42 | Valid with a separate generator, but reuses the value of the main training seed and the sampler seed, inviting confusion between streams |
| A preprocessing seed derived from a project root through `SeedSequence` | Adds machinery without a benefit for a single fixed draw |
| FT-Transformer evaluation batch 8,192, as in REF-103 | Not exercised on the 8 GB GPU in float32 at 758 tokens; 16 is the probe-verified workload |
| A separate fixed-length refit to time one fit | Answers a different question and omits the early-stopping overhead that the protocol asks for; the selected trial's own elapsed time needs no extra fit |
| 1,000 latency rows, one round | About ten observations lie in the upper 1 % tail; 10,000 distinct rows with three rounds give about a hundred, at a cost of minutes per configuration |
| Neural models timed on the CPU as well as on the GPU | The study never runs a neural model on the CPU; the added path and its backend differences would measure a configuration that is not part of the study |
| Pinning the process to a fixed CPU set | Under WSL2 the guest's CPU numbering does not reveal which are performance cores of the hybrid processor, so a pin could only guess; the affinity is recorded instead |
| Requiring a controlled timing session (alternating-current power, a fixed power profile, no competing workload) with an operator confirmation | Computational cost describes the recorded runs and does not enter model, threshold or policy selection. The FT-Transformer search may span several days while the machine remains in use; recording the system context accommodates these working conditions without requiring an isolated timing session |
| Enforcing session conditions in software | Changing the operating system's power settings or aborting on a load figure is outside the repository's control; the conditions are recorded as context instead |
| Process-lifetime peak resident memory (`ru_maxrss`) per phase | It is cumulative across phases in one process; the per-phase high-water reset or a sampler measures the phase |
| One fresh process per measured phase | Changes the process architecture and adds start-up to every phase; used only if the simpler methods prove inadequate |

## Rationale

Each configuration receives ten candidate positions: one predefined reference configuration and nine settings drawn by the fixed sampler, so every configuration has the same search budget and no randomly sampled setting is chosen by hand. Early stopping on V1 limits overfitting within each trial, and the maxima bound training length. The V1 objective measures what the study's question concerns: the frauds captured within limited daily review capacity. The fixed sampler makes candidate generation reproducible; for fixed model scores, the comparison and tie rules (§5) make selection and queue ordering deterministic. Residual training variation is measured as DR-013 §6 specifies (§4). The tree-based search spaces come from published tuning protocols, CatBoost's from the same source and dataset-size class as the neural spaces; the two adaptations fit the spaces to nine random draws within the round maximum and keep the same nominal candidate list for both weightings with the scale-normalised leaf-size control, while the library's regularisation can still interact with class weighting (§6). Each tree-based reference configuration is its library's default for this setting, so every tree search contains the setting its developers chose. The neural spaces are REF-087's large-dataset spaces as its authors sample them. FT-Transformer's reference is the published default evaluated in REF-087; the ResNet-style MLP's reference adapts the TabZilla default evaluated in REF-089, replacing its category-embedding width and weight decay as §7 specifies [I], so the adapted configuration itself has not been evaluated in that source. One effective batch for every candidate keeps the optimisation rule fixed across the search, while the micro-batch plan only adapts the work to the GPU's memory (§7).

## Consequences

- The protocol's Training Procedure, the R3 policy and design item 3 state these rules.
- The development pipeline (Issue #96) saves every trial (settings, V1 objective per checkpoint, stopping round or epoch, stopping reason, runtime), the selected settings, the seed refits and the V2 metrics as JSON (JavaScript Object Notation) records, with a Markdown summary generated from them.
- The search spaces and reference configurations of §6 and §7 complete the search rules of the training rules before any development run; the neural representation is that of [DR-012](DR-012-neural-network-input-representation.md), with the preprocessing seed of §4.
- The neural loop implements the following and is verified on synthetic data: effective batches split into recorded micro-batches with one AdamW update per effective batch and the loss of §7; the pre-search micro-batch preflight; the shared daily Recall@2% objective of §3; and the recording of the batch plan. The verification compares accumulated and unsplit gradients with dropout disabled, covers unequal micro-batches, non-uniform weights and the final batch of 1,016 rows, and checks same-seed repeatability for a recorded plan, the largest ResNet `d_embedding` (512) and representative FT-Transformer widths with padded heads. Because these change exercised paths, the synthetic acceptance evidence is extended under [`gpu-acceptance-probe.md`](../gpu-acceptance-probe.md) (Section 9).
- The development pipeline (Issue #96) reads the search spaces, sampling order and reference configurations, and the neural preprocessing seed of §4, from one machine-readable declaration that matches §4, §6, §7 and §8.
- Both tree adapters accept the shared V1 objective and are verified on synthetic data: LightGBM accepts `metric="None"` with the shared V1 objective as a custom evaluation function, and CatBoost accepts the shared objective as a user-defined evaluation metric. The verification covers the daily grouping, unweighted evaluation, the cutoff tie rule, strict improvement, patience, earliest-best checkpoint retention, and parameter recording and reloading with the custom metric.
- The computational-cost procedure of §9 is implemented in `training/timing.py` and the versioned `timing` record contract; the development timings are taken inside an approved run of the search runner, and the first approved run checks the host-memory reset on the project machine with a synthetic allocation.
- The synthetic CatBoost verification also fits the largest depth of the space (10) on the GPU at the maximum of 2,000 iterations, because the acceptance probe measured depth 6 only.

## Threats and limitations

- **Reference convergence.** The reference meets a gradient stopping threshold (`tol`), not a bound on probability error; coefficients of rare categories can remain imprecise. It is a descriptive reference outside H1–H3, and its probabilities are not read as exact. A candidate that does not converge within `max_iter` stops the analysis for an issue; the ten candidates are checked on Train before the first development run (§8).
- **Limited search budget.** Ten trials is below the budgets of up to 30–100 trials in published benchmarks [REF-087; REF-089]; it is a defined budget boundary set by the measured compute. Conclusions concern these pipelines under this budget, not each family's globally best achievable performance.
- **Training length.** Fifteen epochs may stop slow-learning neural settings before convergence. Trials that reach the maximum are recorded, and a selected model that reached it is reported as such.
- **V1 objective.** Recall@2% on V1 does not reflect ranking changes below the daily cutoff, so early stopping may end some trials earlier than a whole-ranking measure would.
- **Seed variation.** Three seeds show a range, not a precise distribution of stochastic variation. CatBoost's GPU training is non-deterministic, so a fixed seed does not reproduce its fits exactly; the same-seed repeats measure that difference.
- **Search coverage.** Equal trial counts give an equal search budget, not equal coverage of search spaces of different size, nor equal convergence.
- **Neural updates.** At the effective batch of 1,024 and at most 15 epochs, a neural trial makes at most 5,400 updates, fewer than REF-087's training until 17 epochs without improvement; trials that reach the maximum are recorded (§2).
- **Accumulation.** FT-Transformer's accumulated updates follow the effective-batch loss but do not reproduce an unsplit batch's dropout masks or floating-point order; repeatability holds for the recorded micro-batch plan.
- **ResNet reference.** The ResNet-style MLP's reference configuration is a benchmark default with two values replaced [I], not a configuration from REF-087.
- **Compute estimate.** The neural times of §7 are extrapolations from synthetic probe fits, not measured development runtimes; the predicted cost of the actual candidate lists is recorded before any run.
- **Adapted spaces.** The tree-based spaces adapt published spaces to a smaller random budget (§6); the narrowed learning-rate range excludes settings that larger adaptive searches can reach. Trials that reach the 2,000-round maximum are recorded (§2).

- **Timing conditions.** The project machine is a laptop with power and thermal limits, a hybrid processor and a virtualised Linux guest; timings are single-machine, descriptive, taken under the machine's working conditions at the time (other local workloads may run, including during searches that may span several days) and carry their recorded context. They support no claim about model-family efficiency, energy use, monetary cost or service capacity.
- **Memory.** GPU memory is a sampled maximum of the whole device and can miss short spikes; the increase above the baseline is not memory attributable to the model alone. Host memory covers the process, not the whole Windows host.
- **Latency.** Serial scoring latency (one call at a time) excludes queueing, so it cannot establish loaded-service tail latency.

## References

REF-012, REF-087, REF-089, REF-103, REF-116, REF-118, REF-119, REF-120, REF-128, REF-129, REF-130, REF-131, REF-135, REF-136 — see [`../references.md`](../references.md).

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [D], [I], [L] | Evidence tags: dataset or project evidence; inference; literature |
| CatBoost, LightGBM | Gradient-boosted decision-tree libraries (the tree-based models of DR-011) |
| d | A V1 relative day |
| `D_V1` | Number of V1 relative days (21) |
| DR, REF-NNN | Decision Record; reference identifier in `docs/references.md` |
| `F_d`, `F_d^top` | Fraud transactions of day d; those among its 62 highest-ranked transactions |
| FT-Transformer | Feature Tokenizer + Transformer |
| GPU | Graphics processing unit |
| H2, H3 | Confirmatory hypotheses: class-weighting effect; alert-rate transfer fidelity |
| JSON | JavaScript Object Notation: the machine-readable record format of the pipeline |
| K, `K_V1` | Daily review budget (K = 56, from V2 volume); daily review count for tuning on V1 (62) |
| MLP | Multilayer perceptron |
| `P_hit`, p, n | Probability of at least one random draw in a region; the region's probability mass; number of independent random draws |
| PCG64 | The bit generator behind NumPy's default random generator |
| CUDA | NVIDIA's GPU computing platform |
| p95, p99 | 95th and 99th percentile of the single-transaction latencies |
| `VmHWM`, `/proc/self/clear_refs` | The Linux process's peak resident memory ("high-water mark") and the file that resets it |
| WSL2 | Windows Subsystem for Linux 2: the virtualised Linux environment of the project machine |
| `early_stopping_round`; `od_type`, `od_wait`; `patience` | LightGBM's early-stopping round count; CatBoost's overfitting-detector type and wait; the neural loop's patience |
| Recall@2%_V1 | V1 Recall@2% under daily top-62 review (the V1 objective) |
| Σ_d | Sum over the V1 relative days |
| `TransactionID` | Transaction row identifier of the dataset |
| Optuna | Hyperparameter-optimisation library used by REF-087 |
| `N_V1` | Number of V1 transactions (65,544) |
| R3, R4 | Decision policies: daily top-K; calibrated cost-aware capped policy |
| Recall@2% | Share of frauds captured under daily top-K review at 2 % capacity |
| ROC, ROC-AUC | Receiver Operating Characteristic curve; the area under it |
| SESOI | Smallest Effect Size Of Interest |
| SHA-256 | Secure Hash Algorithm, 256-bit |
| Test | Sealed final evaluation partition (relative days 155–182) |
| Train | Model-fitting partition (relative days 1–105) |
| u, b | Unweighted; balanced class weighting |
| UTF-8 | Unicode Transformation Format, 8-bit: the text encoding of `TransactionID` before hashing |
| V1, V2 | Validation 1 (relative days 106–126); Validation 2 (relative days 127–154) |
| CPU | Central processing unit |
| Uniform, LogUniform, UniformInt, LogUniformInt | Search distributions: continuous uniform; log-uniform; uniform on the integers; log-uniform rounded down to integers (§6) |
| 0.5 · {0} + 0.5 · LogUniform | Mixture: exactly 0 with probability 0.5, otherwise log-uniform |
| x, `x_min`, `x_max` | Sampled parameter value; lower and upper bounds of its range |
| z, w | Continuous uniform draws: the value draw; the mixture-branch draw |
| U[·, ·) | Continuous uniform distribution on a half-open interval |
| ⌊·⌋, ln, exp | Rounding down to an integer; natural logarithm; exponential function |
| C | Inverse regularisation strength of the logistic-regression reference |
| `lbfgs`, `l1_ratio`, `tol`, `max_iter`, `fit_intercept`, `class_weight`, `warm_start`, `penalty`, `Cs` | scikit-learn logistic-regression settings: quasi-Newton solver; elastic-net mixing (0 = L2); convergence tolerance; iteration limit; whether an intercept is fitted; class weighting (None = none); whether a fit starts from the previous solution; the deprecated penalty selector; the number of `LogisticRegressionCV` grid values |
| float64 | 64-bit floating-point format |
| space (B) | The large-dataset search space of REF-087 |
| TabR, TabZilla | The benchmark of REF-129; the benchmark code of REF-089 (REF-130) |
| `learning_rate`, `num_leaves`, `min_data_in_leaf`, `min_sum_hessian_in_leaf`, `feature_fraction`, `bagging_fraction`, `bagging_freq`, `lambda_l1`, `lambda_l2` | LightGBM: shrinkage per tree; maximum leaves per tree; minimum rows per leaf; minimum Hessian sum per leaf; share of features per tree; share of rows per bagging draw; bagging interval in rounds; L1 and L2 leaf-value penalties |
| `depth`, `l2_leaf_reg`, `bagging_temperature`, `leaf_estimation_iterations`, `random_strength` | CatBoost: symmetric-tree depth; L2 leaf-value penalty; Bayesian-bootstrap weight intensity; Newton steps per leaf-value estimate; split-score noise scale |
| `eval_metric`, `metric_period`, `iterations`; `metric`, `num_iterations` | CatBoost's evaluation metric, its evaluation interval and maximum trees; LightGBM's metric and maximum trees |
| L1, L2 | Absolute-value and squared-value penalties; applied to tree leaf values and, for L2, logistic-regression coefficients |
| `η_ref` | CatBoost's reference learning rate (0.032340) |
| `round_6` | Rounding to six decimal places |
| `c_N`, `c_0`, `c_I` | Coefficients of CatBoost's automatic learning-rate rule (binary log-loss, GPU, `use_best_model` on, `boost_from_average` off) |
| `N_Train` | Number of Train rows (368,632) |
| `I_max` | Maximum boosting rounds (2,000) |
| `Generator.uniform`, `Generator.integers` | NumPy's continuous and integer uniform draws from a seeded generator |
| `boosting`, `data_sample_strategy`, `max_depth`, `min_gain_to_split`, `max_delta_step`, `path_smooth`, `extra_trees`, `feature_fraction_bynode`, `boost_from_average`, `early_stopping_min_delta` | LightGBM: boosting algorithm; row-sampling strategy; depth limit (−1 = none); minimum split gain; leaf-output step limit; leaf-value smoothing towards the parent; randomised split thresholds; share of features per node; start from the average label; minimum improvement counted by early stopping |
| `max_cat_to_onehot`, `max_cat_threshold`, `cat_l2`, `cat_smooth`, `min_data_per_group` | LightGBM categorical controls: largest category count split one-vs-rest; maximum categories on one side of a split; L2 penalty in categorical splits; smoothing of category statistics; minimum rows per category group |
| `model_size_reg`, `permutation_count`, `fold_permutation_block`, `boost_from_average`, `use_best_model`, `leaf_estimation_method`, `has_time` | CatBoost: model-size regularisation coefficient; number of permutations; permutation block size; start from the average label; keep the best-iteration model; leaf-value estimation method; use the supplied row order as time |
| `n_layers`, `d`, `d_hidden_factor`, `hidden_dropout`, `residual_dropout`, `d_embedding` | ResNet-style MLP: number of residual blocks; block width (the parameter `d`, distinct from the relative day d); ratio of a block's hidden width to `d`; dropout inside a block; dropout on a block's residual branch; length of each category's embedding vector |
| `n_layers`, `d_token`, `attention_dropout`, `ffn_dropout`, `residual_dropout`, `d_ffn_factor` | FT-Transformer: number of Transformer layers; length of each input's token vector; dropout of the attention weights; dropout inside the feed-forward module; dropout on the residual branches; ratio of the feed-forward hidden size to `d_token` |
| `lr`, `weight_decay` | Neural learning rate; AdamW's decoupled weight-decay coefficient |
| `dropout_first`, `dropout_second` | The `rtdl` ResNet block's dropout after the activation and after the second linear layer (REF-131) |
| AdamW | Adam optimiser with decoupled weight decay |
| `betas`, `eps`, AMSGrad | AdamW's decay coefficients of the first- and second-moment estimates; its numerical-stability constant; the variant that keeps the maximum of past second moments (disabled) |
| i | Index of a Train row in an effective batch |
| BatchNorm, LayerNorm | Batch normalisation (statistics over the rows of a batch); layer normalisation (over the values of one row) |
| ReGLU, FFN | Rectified gated linear unit; feed-forward network (FT-Transformer's feed-forward module) |
| bf16, float32 | bfloat16: 16-bit floating-point format with the exponent range of float32; 32-bit floating-point format |
| MiB | Mebibyte (2^20 bytes) |
| Effective batch, micro-batch | The Train rows of one optimiser update (1,024); the part of an effective batch processed by the GPU at once |
| `suggest_categorical` | Optuna's categorical choice, used by REF-103 to switch an optional parameter off or on |
| `L_batch`, `𝓑`, m, `w_i`, `ℓ_i` | Training loss of an effective batch; its Train rows; its row count; sample weight of row i; binary cross-entropy of row i |
| `T_FTT(E)`, E, `t_ref`, `t̄` | Search time of FT-Transformer's two configurations; epochs per trial; epoch time of the reference configuration; epoch time of an average candidate |
