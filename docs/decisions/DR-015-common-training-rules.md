# DR-015 — Common training rules: search, early stopping, Validation 1 (V1) objective, seeds and tie rules

| Field | Value |
|---|---|
| Status | Accepted (pre-freeze) through Issues #105 and #106, in line with the University of Greenwich criteria ([criteria checklist](../greenwich-criteria-checklist.md)); supervisor feedback welcome, not a precondition |
| Date | 2026-10-16 (Issue #105: common rules; Issue #106: tree-based search spaces and reference configurations) |
| Evidence source | Gorishniy et al. [REF-087] and McElfresh et al. [REF-089] (tuning protocols, read in full); Bergstra and Bengio [REF-128]; the tree-based tuning spaces of Prokhorenkova et al. [REF-012, arXiv v5 Appendix D.2], Gorishniy et al. [REF-087, Table 17; REF-129, Tables 23–24] and the TabZilla code [REF-130]; LightGBM and CatBoost documentation and source code [REF-116; REF-118; REF-119]; the training-time measurements of the graphics processing unit (GPU) acceptance probe ([`gpu-acceptance-probe.md`](../gpu-acceptance-probe.md), Section 7; Issue #77); the partition sizes of [DR-001](DR-001-temporal-split-strategy.md); two independent proposals reviewed in Issue #105. No labels, model scores or Test-period data were used. |
| Related | [DR-001](DR-001-temporal-split-strategy.md), [DR-002](DR-002-review-capacity-definition.md), [DR-005](DR-005-sesoi-and-h1-h2-inference.md), [DR-010](DR-010-h3-confirmatory-configuration.md), [DR-011](DR-011-model-set-tree-based-and-neural-families.md), [DR-013](DR-013-training-device-policy.md), [protocol](../protocol.md) |

## Context

The eight configurations of [DR-011](DR-011-model-set-tree-based-and-neural-families.md) (LightGBM, CatBoost, the ResNet-style multilayer perceptron (MLP) and the Feature Tokenizer + Transformer (FT-Transformer), each unweighted (u) and with balanced class weighting (b)) are compared under one common procedure. They are fitted on Train (the model-fitting partition, relative days 1–105), tuned and early-stopped on Validation 1 (V1, days 106–126), and evaluated on Validation 2 (V2, days 127–154); the Test partition (days 155–182) is sealed ([DR-001](DR-001-temporal-split-strategy.md)). DR-011 §3 (Level 1) requires the same hyperparameter search method, the same budget per configuration, the same V1 objective, the same early-stopping rule and the same seeds for all eight. This record fixes those values before any development run.

The study's primary outcome is Recall@2%: the share of frauds captured under R3, the daily top-K policy, which reviews the K highest-scored transactions of each relative day, where K is the fixed daily review budget (K = 56 at 2 % capacity, from V2 volume; [DR-002](DR-002-review-capacity-definition.md)). H3, the alert-rate transfer fidelity hypothesis, is confirmatory for one configuration selected on V2 ([DR-010](DR-010-h3-confirmatory-configuration.md)).

The models differ greatly in training cost on the project hardware. Extrapolated from the GPU acceptance probe, one full pass over the Train partition takes about 0.1–0.4 minutes for the ResNet-style MLP and about 5, 7 and 40 minutes for the padded, default and largest FT-Transformer settings; a LightGBM fit takes seconds and a CatBoost GPU fit about 15–20 seconds at the probe values. Published tabular benchmarks tune each model with up to 30 to 100 trials [REF-087; REF-089].

The search spaces and reference configurations of the tree-based models are fixed in §6 (Issue #106); those of the neural models are fixed in Issue #107 and recorded with this record.

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

The rule is the same for all eight configurations. Patience is set in each family's natural training unit, because one boosting round adds a single tree, a far smaller step than one epoch, which passes over all Train rows with thousands of parameter updates. The tree-based values follow REF-087 (CatBoost: 2,000 iterations, 50 early-stopping rounds); the neural values are an engineering adaptation to the measured training cost. Each trial records its stopping round or epoch, its stopping reason (early stop or maximum) and its runtime.

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
- **Evaluation:** V2 only, Recall@2% under R3 with K = 56; neither kind of refit is fitted on any partition other than Train.
- **Use:** descriptive only. The range over the three seeds 42, 43 and 44 is compared with the Smallest Effect Size Of Interest (SESOI; 2 percentage points of Recall@2%, [DR-005](DR-005-sesoi-and-h1-h2-inference.md)) as the different-seed variation of DR-013 §6. It never changes the selection, the H3 configuration or any other decision.

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

**Reference configurations.** Trial 1 of each model uses the library's default values for the pinned version, written out explicitly, as the default setting that REF-089 includes among its trials [REF-089]. Each model keeps its own default learning rate; no common value is imposed across models, because fairness is the common procedure, not identical values ([DR-011](DR-011-model-set-tree-based-and-neural-families.md) §3). Every reference value lies within its parameter's declared bounds; a reference value may sit at an upper bound that the half-open continuous sampling interval excludes (for example a fraction of 1.0).

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

## Evidence

- **[L]** REF-087 tunes with a Tree-structured Parzen Estimator in Optuna: 100 iterations for most models and 50 for FT-Transformer on large datasets, neural patience 16, and CatBoost with 2,000 iterations and 50 early-stopping rounds. Its Appendix D.2 reports good FT-Transformer results after a few randomly sampled configurations [REF-087].
- **[L]** REF-089 allows each algorithm at most 30 hyperparameter settings per dataset split (one default and up to 29 random), within ten hours per algorithm and two hours per fit: budget ceilings, not equal completed trial counts [REF-089].
- **[L]** REF-128 gives empirical and theoretical support for random search over grid search in the settings it examines [REF-128].
- **[L]** REF-087 tunes CatBoost on GPU with 2,000 iterations and 50 early-stopping rounds over depth (UniformInt[3, 10] for smaller datasets, UniformInt[6, 10] for the large-dataset group (B)), learning rate LogUniform[1e-5, 1], bagging temperature Uniform[0, 1], L2 leaf regularisation LogUniform[1, 10] and leaf-estimation iterations UniformInt[1, 10] (Table 17, Appendix F.4) [REF-087].
- **[L]** REF-129 tunes LightGBM over `feature_fraction` and `bagging_fraction` Uniform[0.5, 1.0], `lambda_l2` {0, LogUniform[0.1, 10.0]}, `learning_rate` LogUniform[0.001, 1.0], `num_leaves` UniformInt[4, 768] and `min_sum_hessian_in_leaf` LogUniform[0.0001, 100.0], and CatBoost with depth UniformInt[3, 14], each with 200 Tree-structured Parzen Estimator iterations, 4,000 trees and 200 early-stopping rounds (Tables 23–24, Appendix D.18–D.19) [REF-129].
- **[L]** REF-012 tunes LightGBM and CatBoost with 50 Tree-structured Parzen Estimator steps; its LightGBM space includes `feature_fraction` and `bagging_fraction` Uniform[0.5, 1] and L1 and L2 penalties as the mixture 0.5 · degenerate at 0 + 0.5 · log-uniform, and its CatBoost space includes `l2_leaf_reg` log-uniform [1, 10], `bagging_temperature` Uniform[0, 1] and `gradient_iterations` as a "Discrete uniform distribution over a set {1, 10}" (arXiv v5, Appendix D.2) [REF-012].
- **[L]** The Optuna parameter definitions in REF-089's code sample the learning rate of both LightGBM and CatBoost from LogUniform[0.01, 0.3]; its separate random-search code draws CatBoost learning rates over [0.03, 0.3) and LightGBM learning rates over [0.03, 30) [REF-130]. LightGBM documents `min_data_in_leaf` as "an approximation based on the Hessian" [REF-116]; its split search estimates each side's row count as the side's Hessian sum times the parent's ratio of rows to Hessian sum [REF-117]. CatBoost's automatic learning rate for log-loss is a fixed function of device, `use_best_model`, `boost_from_average`, training-row count and iteration count, applied only when the learning-rate, leaf-estimation and L2 parameters are unset [REF-119]. LightGBM's tuning guide states that a reduced iteration count calls for a larger learning rate, and that `min_data_in_leaf` of "hundreds or thousands is enough for a large dataset" [REF-116]; CatBoost's guide states that a smaller learning rate requires more iterations and recommends depths of 6 to 10 [REF-118]. In CatBoost 1.2.10, binary log-loss defaults to ten Newton leaf-estimation iterations and L2 regularisation 3, the base learning-rate default is 0.03, and the default depth is 6 [REF-119].
- **[D]** Training cost per full Train epoch from the GPU acceptance probe (Context); daily transaction volume per partition from DR-001 (V1: 65,544 transactions over 21 days).
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
| Tree-structured Parzen Estimator search | With 10 trials, little information is available for its adaptive phase; random search keeps one sampling procedure for every configuration and allows the paired candidate lists |
| A predefined grid of configurations | Leaves the choice of combinations to the researcher, with sparse joint coverage of multidimensional spaces [REF-128] |
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

## Rationale

Each configuration receives ten candidate positions: one predefined reference configuration and nine settings drawn by the fixed sampler, so every configuration has the same search budget and no randomly sampled setting is chosen by hand. Early stopping on V1 limits overfitting within each trial, and the maxima bound training length. The V1 objective measures what the study's question concerns: the frauds captured within limited daily review capacity. The fixed sampler makes candidate generation reproducible; for fixed model scores, the comparison and tie rules (§5) make selection and queue ordering deterministic. Residual training variation is measured as DR-013 §6 specifies (§4). The tree-based search spaces come from published tuning protocols, CatBoost's from the same source and dataset-size class as the neural spaces; the two adaptations fit the spaces to nine random draws within the round maximum and keep the same nominal candidate list for both weightings with the scale-normalised leaf-size control, while the library's regularisation can still interact with class weighting (§6). Each reference configuration is its library's default for this setting, so every search contains the setting its developers chose.

## Consequences

- The protocol's Training Procedure, the R3 policy and design item 3 state these rules.
- The development pipeline (Issue #96) saves every trial (settings, V1 objective per checkpoint, stopping round or epoch, stopping reason, runtime), the selected settings, the seed refits and the V2 metrics as JSON (JavaScript Object Notation) records, with a Markdown summary generated from them.
- The tree-based search spaces and reference configurations (§6) and the neural ones of Issue #107 complete the training rules before any development run.
- The development pipeline (Issue #96) reads the search spaces, sampling order and reference configurations from one machine-readable declaration that matches §6.
- Before any development run, both tree adapters change and are verified on synthetic data: LightGBM accepts `metric="None"` with the shared V1 objective as a custom evaluation function (its adapter currently requires a built-in metric and passes no evaluation function), and CatBoost accepts the shared objective as a user-defined evaluation metric (its adapter currently requires a metric name). The verification covers the daily grouping, unweighted evaluation, the cutoff tie rule, strict improvement, patience, earliest-best checkpoint retention, and parameter recording and reloading with the custom metric.
- The synthetic CatBoost verification also fits the largest depth of the space (10) on the GPU at the maximum of 2,000 iterations, because the acceptance probe measured depth 6 only.

## Threats and limitations

- **Limited search budget.** Ten trials is below the budgets of up to 30–100 trials in published benchmarks [REF-087; REF-089]; it is a defined budget boundary set by the measured compute. Conclusions concern these pipelines under this budget, not each family's globally best achievable performance.
- **Training length.** Fifteen epochs may stop slow-learning neural settings before convergence. Trials that reach the maximum are recorded, and a selected model that reached it is reported as such.
- **V1 objective.** Recall@2% on V1 does not reflect ranking changes below the daily cutoff, so early stopping may end some trials earlier than a whole-ranking measure would.
- **Seed variation.** Three seeds show a range, not a precise distribution of stochastic variation. CatBoost's GPU training is non-deterministic, so a fixed seed does not reproduce its fits exactly; the same-seed repeats measure that difference.
- **Search coverage.** Equal trial counts give an equal search budget, not equal coverage of search spaces of different size, nor equal convergence.
- **Adapted spaces.** The tree-based spaces adapt published spaces to a smaller random budget (§6); the narrowed learning-rate range excludes settings that larger adaptive searches can reach. Trials that reach the 2,000-round maximum are recorded (§2).

## References

REF-012, REF-087, REF-089, REF-116, REF-118, REF-119, REF-128, REF-129, REF-130 — see [`../references.md`](../references.md).

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
| `early_stopping_round`; `od_type`, `od_wait`; `patience` | LightGBM's early-stopping round count; CatBoost's overfitting-detector type and wait; the neural loop's patience |
| Recall@2%_V1 | V1 Recall@2% under daily top-62 review (the V1 objective) |
| Σ_d | Sum over the V1 relative days |
| `TransactionID` | Transaction row identifier of the dataset |
| Optuna | Hyperparameter-optimisation library used by REF-087 |
| `N_V1` | Number of V1 transactions (65,544) |
| R3, R4 | Decision policies: daily top-K; calibrated cost-aware capped policy |
| Recall@2% | Share of frauds captured under daily top-K review at 2 % capacity |
| ROC-AUC | Area under the Receiver Operating Characteristic curve |
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
| float64 | 64-bit floating-point format |
| space (B) | The large-dataset search space of REF-087 |
| TabR, TabZilla | The benchmark of REF-129; the benchmark code of REF-089 (REF-130) |
| `learning_rate`, `num_leaves`, `min_data_in_leaf`, `min_sum_hessian_in_leaf`, `feature_fraction`, `bagging_fraction`, `bagging_freq`, `lambda_l1`, `lambda_l2` | LightGBM: shrinkage per tree; maximum leaves per tree; minimum rows per leaf; minimum Hessian sum per leaf; share of features per tree; share of rows per bagging draw; bagging interval in rounds; L1 and L2 leaf-value penalties |
| `depth`, `l2_leaf_reg`, `bagging_temperature`, `leaf_estimation_iterations`, `random_strength` | CatBoost: symmetric-tree depth; L2 leaf-value penalty; Bayesian-bootstrap weight intensity; Newton steps per leaf-value estimate; split-score noise scale |
| `eval_metric`, `metric_period`, `iterations`; `metric`, `num_iterations` | CatBoost's evaluation metric, its evaluation interval and maximum trees; LightGBM's metric and maximum trees |
| L1, L2 | Absolute-value and squared-value penalties on leaf values |
| `η_ref` | CatBoost's reference learning rate (0.032340) |
| `round_6` | Rounding to six decimal places |
| `c_N`, `c_0`, `c_I` | Coefficients of CatBoost's automatic learning-rate rule (binary log-loss, GPU, `use_best_model` on, `boost_from_average` off) |
| `N_Train` | Number of Train rows (368,632) |
| `I_max` | Maximum boosting rounds (2,000) |
| `Generator.uniform`, `Generator.integers` | NumPy's continuous and integer uniform draws from a seeded generator |
| `boosting`, `data_sample_strategy`, `max_depth`, `min_gain_to_split`, `max_delta_step`, `path_smooth`, `extra_trees`, `feature_fraction_bynode`, `boost_from_average`, `early_stopping_min_delta` | LightGBM: boosting algorithm; row-sampling strategy; depth limit (−1 = none); minimum split gain; leaf-output step limit; leaf-value smoothing towards the parent; randomised split thresholds; share of features per node; start from the average label; minimum improvement counted by early stopping |
| `max_cat_to_onehot`, `max_cat_threshold`, `cat_l2`, `cat_smooth`, `min_data_per_group` | LightGBM categorical controls: largest category count split one-vs-rest; maximum categories on one side of a split; L2 penalty in categorical splits; smoothing of category statistics; minimum rows per category group |
| `model_size_reg`, `permutation_count`, `fold_permutation_block`, `boost_from_average`, `use_best_model`, `leaf_estimation_method`, `has_time` | CatBoost: model-size regularisation coefficient; number of permutations; permutation block size; start from the average label; keep the best-iteration model; leaf-value estimation method; use the supplied row order as time |
