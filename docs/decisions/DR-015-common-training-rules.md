# DR-015 — Common training rules: search, early stopping, Validation 1 (V1) objective, seeds and tie rules

| Field | Value |
|---|---|
| Status | Accepted (pre-freeze) through Issue #105, in line with the University of Greenwich criteria ([criteria checklist](../greenwich-criteria-checklist.md)); supervisor feedback welcome, not a precondition |
| Date | 2026-10-16 (Issue #105) |
| Evidence source | Gorishniy et al. [REF-087] and McElfresh et al. [REF-089] (tuning protocols, read in full); Bergstra and Bengio [REF-128]; the training-time measurements of the graphics processing unit (GPU) acceptance probe ([`gpu-acceptance-probe.md`](../gpu-acceptance-probe.md), Section 7; Issue #77); the partition sizes of [DR-001](DR-001-temporal-split-strategy.md); two independent proposals reviewed in Issue #105. No labels, model scores or Test-period data were used. |
| Related | [DR-001](DR-001-temporal-split-strategy.md), [DR-002](DR-002-review-capacity-definition.md), [DR-005](DR-005-sesoi-and-h1-h2-inference.md), [DR-010](DR-010-h3-confirmatory-configuration.md), [DR-011](DR-011-model-set-tree-based-and-neural-families.md), [DR-013](DR-013-training-device-policy.md), [protocol](../protocol.md) |

## Context

The eight configurations of [DR-011](DR-011-model-set-tree-based-and-neural-families.md) (LightGBM, CatBoost, the ResNet-style multilayer perceptron (MLP) and the Feature Tokenizer + Transformer (FT-Transformer), each unweighted (u) and with balanced class weighting (b)) are compared under one common procedure. They are fitted on Train (the model-fitting partition, relative days 1–105), tuned and early-stopped on Validation 1 (V1, days 106–126), and evaluated on Validation 2 (V2, days 127–154); the Test partition (days 155–182) is sealed ([DR-001](DR-001-temporal-split-strategy.md)). DR-011 §3 (Level 1) requires the same hyperparameter search method, the same budget per configuration, the same V1 objective, the same early-stopping rule and the same seeds for all eight. This record fixes those values before any development run.

The study's primary outcome is Recall@2%: the share of frauds captured under R3, the daily top-K policy, which reviews the K highest-scored transactions of each relative day, where K is the fixed daily review budget (K = 56 at 2 % capacity, from V2 volume; [DR-002](DR-002-review-capacity-definition.md)). H3, the alert-rate transfer fidelity hypothesis, is confirmatory for one configuration selected on V2 ([DR-010](DR-010-h3-confirmatory-configuration.md)).

The models differ greatly in training cost on the project hardware. Extrapolated from the GPU acceptance probe, one full pass over the Train partition takes about 0.1–0.4 minutes for the ResNet-style MLP and about 5, 7 and 40 minutes for the padded, default and largest FT-Transformer settings; a LightGBM fit takes seconds and a CatBoost GPU fit about 15–20 seconds at the probe values. Published tabular benchmarks tune each model with up to 30 to 100 trials [REF-087; REF-089].

The search spaces and the reference configuration of each model are fixed in Issues #106 (tree-based models) and #107 (neural models), and are recorded with this record.

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
- **Implementation:** one shared scorer computes the objective for all eight configurations: a custom evaluation function for LightGBM, a user-defined evaluation metric for CatBoost on its GPU learner, and one computation per epoch on the full V1 scores for the neural models. The CatBoost path is verified on synthetic data before any development run.

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

## Evidence

- **[L]** REF-087 tunes with a Tree-structured Parzen Estimator in Optuna: 100 iterations for most models and 50 for FT-Transformer on large datasets, neural patience 16, and CatBoost with 2,000 iterations and 50 early-stopping rounds. Its Appendix D.2 reports good FT-Transformer results after a few randomly sampled configurations [REF-087].
- **[L]** REF-089 allows each algorithm at most 30 hyperparameter settings per dataset split (one default and up to 29 random), within ten hours per algorithm and two hours per fit: budget ceilings, not equal completed trial counts [REF-089].
- **[L]** REF-128 gives empirical and theoretical support for random search over grid search in the settings it examines [REF-128].
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

## Rationale

Each configuration receives ten candidate positions: one predefined reference configuration and nine settings drawn by the fixed sampler, so every configuration has the same search budget and no randomly sampled setting is chosen by hand. Early stopping on V1 limits overfitting within each trial, and the maxima bound training length. The V1 objective measures what the study's question concerns: the frauds captured within limited daily review capacity. The fixed sampler makes candidate generation reproducible; for fixed model scores, the comparison and tie rules (§5) make selection and queue ordering deterministic. Residual training variation is measured as DR-013 §6 specifies (§4).

## Consequences

- The protocol's Training Procedure, the R3 policy and design item 3 state these rules.
- The development pipeline (Issue #96) saves every trial (settings, V1 objective per checkpoint, stopping round or epoch, stopping reason, runtime), the selected settings, the seed refits and the V2 metrics as JSON (JavaScript Object Notation) records, with a Markdown summary generated from them.
- The CatBoost adapter accepts the shared V1 objective as a user-defined evaluation metric on its GPU learner; the change and its synthetic verification precede any development run.
- The search spaces and reference configurations of Issues #106 and #107 complete the training rules before any development run.

## Threats and limitations

- **Limited search budget.** Ten trials is below the budgets of up to 30–100 trials in published benchmarks [REF-087; REF-089]; it is a defined budget boundary set by the measured compute. Conclusions concern these pipelines under this budget, not each family's globally best achievable performance.
- **Training length.** Fifteen epochs may stop slow-learning neural settings before convergence. Trials that reach the maximum are recorded, and a selected model that reached it is reported as such.
- **V1 objective.** Recall@2% on V1 does not reflect ranking changes below the daily cutoff, so early stopping may end some trials earlier than a whole-ranking measure would.
- **Seed variation.** Three seeds show a range, not a precise distribution of stochastic variation. CatBoost's GPU training is non-deterministic, so a fixed seed does not reproduce its fits exactly; the same-seed repeats measure that difference.
- **Search coverage.** Equal trial counts give an equal search budget, not equal coverage of search spaces of different size, nor equal convergence.

## References

REF-087, REF-089, REF-128 — see [`../references.md`](../references.md).

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
