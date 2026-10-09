# COMP1885 Research Protocol

## Status

**Working protocol — pre-freeze. Not yet confirmatory.**

This file defines the confirmatory research design. It builds on the dataset feasibility audit ([`notebooks/01_dataset_feasibility_audit.ipynb`](../notebooks/01_dataset_feasibility_audit.ipynb), conclusion: *"Suitable with methodological changes"*), the data-understanding notebooks 02–03 and the structured literature review ([`literature-review.md`](literature-review.md)). Methodology decisions are recorded in [`docs/decisions/`](decisions/README.md) (DR-001–DR-011).

The remaining items are listed under [Design items before protocol freeze](#design-items-before-protocol-freeze). The protocol becomes confirmatory only when every item is resolved and the file is tagged in Git.

**Test-set status:** the final Test period (relative days 155–182) is **sealed**. Its labels have not been used for any analysis.

Citations use Reference IDs from [`docs/references.md`](references.md).

## Research Title

**Fraud Detection Under Limited Review Capacity: A Temporally Validated Study of Tree-Based and Neural Models and Decision Policies**

## Main Research Question

> **Under constrained daily review capacity, how do model family, class weighting, and score-to-review decision policies affect fraud capture, review workload, and cost-sensitive outcomes on temporally unseen transaction data?**

## Supporting Research Questions

The supporting research questions are numbered RQ1–RQ3.

### RQ1 — Model family and weighting

> **How do tree-based and neural model families, with and without class weighting, differ in fraud capture at fixed review capacity on a later temporal test period?**

The model families are defined in [DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md): tree-based (LightGBM, CatBoost) and neural (ResNet-style multilayer perceptron, MLP; FT-Transformer).

### RQ2 — Policy-transfer fidelity

> **When a score threshold selected on validation data is transferred to a later temporal period, how well does it preserve the intended alert rate, and how does the resulting daily review workload compare with the fixed review budget?**

Defined in [DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md), which separates alert-rate drift (model and data) from daily-volume variation, which exists regardless of the model.

### RQ3 — Cost-aware capacity use

> **Under different assumed miss-to-review cost ratios, when does a calibrated cost-aware review policy outperform always filling the available review capacity?**

RQ3 is answered by estimation and sensitivity analysis across the capacity grid, not by a separate significance test. Its scope and outputs are specified under [Cost Model](#cost-model).

## Confirmatory Hypotheses

The confirmatory hypotheses are H1–H3. A subscript ₀ marks the null hypothesis and ₁ the alternative.

### H1 — Model-family effect

Defined per [DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md). Recall@2% (the share of Test frauds reviewed at 2 % review capacity) is evaluated under policy R3 (daily top-K: review the K highest-scoring transactions each relative day, with fixed daily review budget K = 56). All eight model × weighting configurations ([DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md)) are scored on the same Test frauds. The paired model-family effect is:

$$
\delta_F = \frac{1}{4} \sum_{m \in N} \sum_{w \in \lbrace u, b \rbrace} R^{m,w} \;-\; \frac{1}{4} \sum_{m \in T} \sum_{w \in \lbrace u, b \rbrace} R^{m,w}
$$

where:

- `δ_F` = model-family effect: neural minus tree-based Recall@2%, averaged over the two models of each family and over both weighting conditions;
- `T` = tree-based family {LGB (LightGBM), Cat (CatBoost)}; `N` = neural family {RES (ResNet-style MLP), FTT (FT-Transformer)};
- `R^m,w` = pooled Test Recall@2% of model m with weighting w ∈ {u (unweighted), b (balanced)}.

δ_F is positive when the neural family reviews more frauds than the tree-based family.

- **H1₀:** Averaged over the two models of each family and over the unweighted and balanced-weighting conditions, tree-based and neural models do not differ in Recall@2% on the Test period (δ_F = 0).
- **H1₁:** Averaged over the two models of each family and over the unweighted and balanced-weighting conditions, tree-based and neural models differ in Recall@2% on the Test period (δ_F ≠ 0).

**Smallest effect size of interest (SESOI) = 2 percentage points of Recall@2%.** At the projected ≈ 2,660 Test frauds this is ≈ 53 frauds per 28 days, or ≈ 2 per day, at identical review workload (≈ 3.4 percentage points, pp, of Precision@K, the share of reviewed transactions that are fraud).

H1 is tested as a two-sided **difference** test, with a pre-specified practical interpretation against ±SESOI ([Statistical Plan](#statistical-plan)).

### H2 — Class-weighting effect

Defined per [DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md). The paired weighting effect of model m ∈ {LGB, Cat, RES, FTT} is:

$$
\delta_{W,m} = R^{m,b} - R^{m,u}
$$

where `δ_W,m` = balanced-minus-unweighted Recall@2% of model m. It is positive when balanced weighting helps that model.

- **H2₀:** For at least one of the four models (LightGBM, CatBoost, ResNet-style MLP or FT-Transformer), balanced class weighting changes Recall@2% by 2 percentage points or more relative to the corresponding unweighted model (|δ_W,m| ≥ 0.02 for at least one m).
- **H2₁:** For all four models, balanced class weighting changes Recall@2% by less than 2 percentage points relative to the corresponding unweighted model (|δ_W,m| < 0.02 for every m).

H2 is an **equivalence** hypothesis, because the literature expects class weighting mainly to distort calibration rather than ranking [REF-019, REF-020]. The substantive question is therefore whether weighting matters *materially*. A non-significant difference must never be read as "no effect" [REF-030, REF-063].

H2 is tested by an intersection-union equivalence test over the four models ([Statistical Plan](#statistical-plan)).

### H3 — Alert-rate transfer fidelity

Defined per [DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md). τ₂ is the score threshold selected on Validation 2 (V2, relative days 127–154) so that 2 % of V2 transactions are flagged (policy R2, the capacity-rate threshold). The realised Test alert rate at τ₂, pooled over Test days, is:

$$
r = \frac{\sum_d n_d}{\sum_d N_d}
$$

where:

- `d` = a Test relative day;
- `n_d` = number of transactions on day d with score ≥ τ₂;
- `N_d` = total number of transactions on day d;
- `Σ_d` = sum over the 28 Test days.

r is the share of all Test transactions that the transferred threshold flags. **In confirmatory H3, r, n_d and τ₂ refer only to the rule-selected H3 configuration** (defined under *Configuration* below; [DR-010](decisions/DR-010-h3-confirmatory-configuration.md)), i.e. r = r^f*,w*. r is not pooled or averaged across the eight configurations. Δ in the hypotheses below is the equivalence tolerance on r.

- **H3₀:** The realised alert rate under the validation-selected 2% threshold lies outside the pre-specified equivalence band around the 2% target (|r − 0.02| ≥ Δ, where Δ = 0.004, i.e. ±20% of the target).
- **H3₁:** The realised alert rate under the validation-selected 2% threshold lies within the pre-specified equivalence band around the 2% target (|r − 0.02| < Δ, where Δ = 0.004, i.e. ±20% of the target).

**Equivalence tolerance.**
- Δ is a **relative tolerance of 20 % of the 2 % target**:

  $$
  \begin{aligned}
  \Delta &= 0.20 \times 0.02 \\
  &= 0.004 \\[6pt]
  \text{equivalence band} &= (0.02 - \Delta,\ 0.02 + \Delta) \\
  &= (1.6\%,\ 2.4\%)
  \end{aligned}
  $$

  At K = 56 this is ±11 reviews per day.
- It was fixed before any Test access (Issue #7).
- No fraud- or alerting-specific literature standard exists, so the magnitude is a pre-specified researcher judgement. It is anchored in V2 volume-driven workload variation (upper quartile 16.7 %, 90th percentile 26.5 %) and in a precision-feasibility check.
- The relative form follows the common practice of defining equivalence margins relative to a reference value [REF-065].
- Full justification and the alternatives considered (absolute, budget-derived; 10 %, 25 %, 50 %) are in [DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md).

H3 is tested by TOST (Two One-Sided Tests) on the H3 configuration defined below ([Statistical Plan](#statistical-plan)).

**Configuration** ([DR-010](decisions/DR-010-h3-confirmatory-configuration.md)). H3 is confirmatory for **one** of the eight model × weighting configurations, the *H3 configuration*: the configuration with the highest pooled V2 Recall@2% under R3 (K = 56):

$$
(f^*, w^*) = \underset{(f,\,w)}{\operatorname{arg\,max}} \; R^{f,w}_{\mathrm{V2}}
$$

where:

- `(f*, w*)` = the H3 configuration;
- `R^f,w_V2` = pooled V2 Recall@2% under R3 of model f with weighting w (defined as for R^m,w under H1, but on V2);
- `arg max` = the configuration at which the quantity is largest.

- **Tie-break** (only for exactly equal V2 fraud counts captured): higher V2 PR-AUC (Precision–Recall Area Under the Curve), then the fixed order LGB-u, Cat-u, RES-u, FTT-u, LGB-b, Cat-b, RES-b, FTT-b.
- The rule uses Train, V1 and V2 only. The eight V2 recall values and the selected configuration are committed to the repository **before the Test period is unsealed**, and are final.
- The H3 configuration is a pre-specified, validation-selected deployment proxy: it is chosen on held-out development data by a criterion that does not directly optimise H3 alert-rate transfer. It does not model any particular organisation's model-selection process, it is not evidence for H1 or H2, and it is not described as the "best" model.
- A confirmatory H3 conclusion applies to the H3 configuration only.
- For the other seven configurations, r and its 90 % confidence interval (CI) are reported **descriptively**: they are outside the Holm family and no equivalence conclusion is drawn from them. Together they give a descriptive tree-vs-neural comparison of threshold transfer.

Fidelity of the threshold's daily alert demand to the fixed daily review budget is a separate descriptive outcome, not a hypothesis ([Outcomes](#outcomes)).

## Analytic Approach

The study combines four analytic components. Each research question and hypothesis is answered by the components marked below; the methods are specified in the sections linked.

| Component | What it produces | Answers | Specified in |
|---|---|---|---|
| **Predictive** | A fraud score for every transaction from each of the eight model × weighting configurations | Input to every policy; RQ1 | [Models](#models), [Training Procedure](#training-procedure), [Feature Policy](#feature-policy), [Class-Imbalance Treatment](#class-imbalance-treatment) |
| **Prescriptive** | Score-to-review decisions on each relative day: threshold-generated alert demand under R0–R2 and capped review selections under R3 and R4, all assessed against the fixed daily review budget K | RQ1 (through R3), RQ2 (through R2), RQ3 (R4 against R3) | [Review Capacity](#review-capacity), [Decision Policies](#decision-policies), [Calibration](#calibration) |
| **Inferential (confirmatory)** | Decisions on H1–H3 with family-wise error control | H1, H2, H3 | [Statistical Plan](#statistical-plan) |
| **Estimation and sensitivity (descriptive)** | Effect sizes with intervals; daily-budget fidelity; cost-difference curves, break-even and saturation ratios; false-negative slices | RQ2 (daily-budget fidelity), RQ3, false-negative analysis | [Outcomes](#outcomes), [Cost Model](#cost-model), [False-Negative Analysis](#false-negative-analysis) |

**Units of analysis.**
- **Transaction:** the unit that is scored and reviewed.
- **Relative day:** the unit of the review queue, of the fixed daily budget K and of the day-level bootstrap used for H3.
- **Fraud:** the unit of Recall@2% (H1, H2), resampled jointly by relative day and pseudo-entity ([Statistical Plan](#statistical-plan)).

**Order of use of the data.** Models are fitted on Train and tuned and early-stopped on Validation 1. The quantities that depend on data are fixed on Validation 2: the R1 and R2 thresholds, the R4 calibrator, the daily budget K and the H3 configuration. The R0 threshold (0.5) and the R4 cost-threshold rule are pre-specified and are not selected on data. The Test period is scored once, after the protocol freeze ([Experimental Split Structure](#experimental-split-structure)).

## Dataset

**IEEE-CIS Fraud Detection** [REF-047].

Official source: https://www.kaggle.com/competitions/ieee-fraud-detection/data

Primary labelled source files:

- `train_transaction.csv` (590,540 rows)
- `train_identity.csv` (144,233 rows; 24.4 % of transactions have an identity record)

- **Join:** left join transaction ← identity on `TransactionID`; one-to-one, row-preserving. Verified on keys for all 590,540 rows; only `TransactionID` is shared, and `isFraud` exists only in the transaction table ([notebooks 02–03](../notebooks/README.md)).
- **Temporal ordering variable:** `TransactionDT`, a relative time delta in seconds from an undisclosed reference point [REF-047]. Each transaction is assigned a relative day, covering days 1–182:

  $$
  \text{relative day} = \left\lfloor \frac{\texttt{TransactionDT}}{86{,}400} \right\rfloor
  $$

  where `86,400` = seconds per day and `⌊ · ⌋` = rounding down to a whole number (`//` in Python). **No calendar dates are inferred.**

The competition test data are not used because their labels are unavailable.

**Class-imbalance terminology.**
- Development-data prevalence is 3.49 %, an imbalance ratio (legitimate : fraud) of ≈ 28 : 1 [D]. By partition: Train 3.39 %, V1 4.22 %, V2 3.39 %; Test is not computed ([notebook 03](../notebooks/03_merged_dataset_inspection.ipynb), bar charts; notebook 01 §5).
- The dataset is described as **imbalanced / low-prevalence**. It is **not** described as "severely" or "highly" imbalanced: that wording is conventionally reserved for majority:minority ratios of about 100 : 1 and above [REF-002, REF-001].
- Imbalance ratio alone does not determine difficulty [REF-003, REF-043]. Here the difficulty comes from low prevalence combined with temporal drift and a review budget smaller than the daily fraud count.

**Labelling caveat.** According to the organiser, a reported chargeback marks the card's transaction as fraud, and later transactions linked by account, email or billing address are also labelled fraud, within a 120-day window [REF-048, partially verified]. Labels are therefore not independent across transactions, and fraud that was never reported is labelled legitimate.

## Data Requirements

What each research question and hypothesis needs from the data, where it comes from, and how the data's limits are handled. All requirements are met by the two labelled source files; quantities the dataset does not contain are introduced as stated assumptions, never as invented data.

| Requirement | Used for | Source | Partitions | Limit of the dataset and its handling |
|---|---|---|---|---|
| Fraud label `isFraud` | Model fitting (Train); tuning and early stopping (V1); R1 threshold, R4 calibrator and H3-configuration selection (V2); Recall@2% (H1, H2), RQ3 costs and false-negative analysis (Test) | `train_transaction.csv` | Train, V1, V2; Test only at the confirmatory evaluation | No label-report dates: label-latency sensitivity on V2 days 127–147 ([Experimental Split Structure](#experimental-split-structure)). Labels propagate across linked transactions: dependence-aware inference ([Statistical Plan](#statistical-plan)) and a stated threat to validity ([Leakage Controls](#leakage-controls)) |
| Transaction time `TransactionDT` | Relative day; partitions; daily review queue; daily budget; day-level bootstrap | `train_transaction.csv` | All | Reference point undisclosed: relative days only, no calendar or seasonal claims |
| Join and row key `TransactionID` | Joining the two files; aligning scores, labels and partitions row by row | Both files | All | Monotone in time, so never a model input ([Feature Policy](#feature-policy)) |
| Model inputs (431 declared columns) | Fraud scores | Both files, left-joined on `TransactionID` | All; every fitted transformation uses Train only | Most meanings masked by the provider: the organiser's declared roles are adopted ([Feature Policy](#feature-policy)) |
| Daily transaction volume (label-free) | K (from V2 volume); realised alert rate r (H3); daily-budget fidelity | Derived from `TransactionDT` | V2 for K; Test for r | — |
| Transaction amount `TransactionAmt` | Amount-weighted recall and fraud value exposed (exposure proxies) | `train_transaction.csv` | V2, Test | Not a realised loss; currency per the organiser (partially verified) ([Cost Model](#cost-model)) |
| Slice variables (identity presence, `ProductCD`, `card4`, `card6`, `DeviceType`, amount band, relative week, score band) | False-negative analysis | Both files; model scores | Test | — |
| Review capacity | K and every capacity-based outcome | **Not in the dataset** | — | Simulated scenarios of 1 %, 2 %, 5 % and 10 % ([Review Capacity](#review-capacity)) |
| Review and missed-fraud costs | RQ3 | **Not in the dataset** | — | Assumed cost ratios with sensitivity and break-even analysis ([Cost Model](#cost-model)) |
| Account or customer identifier | Dependence-aware inference (H1, H2); recurring vs new entity slice | **Not in the dataset** | — | Heuristic fine pseudo-entity key built from `card1`, `addr1`, `P_emaildomain` and relative day − `D1` ([DR-006](decisions/DR-006-bootstrap-inference-strategy.md)). The inference key uses Test-period features only; the recurring vs new slice also uses Train features and Train labels. Used only for audit, inference and slicing, never as a model input ([Feature Policy](#feature-policy)) |
| Review outcomes and analyst behaviour | — | **Not in the dataset** | — | Perfect verification is assumed; human review is an exogenous capacity constraint, and no human participants are involved |

## Experimental Split Structure

Chronological, day-aligned split, fixed from transaction volume only, before any target analysis ([DR-001](decisions/DR-001-temporal-split-strategy.md)):

| Partition | Relative days | Rows | Role |
|---|---|---|---|
| Train | 1–105 | 368,632 | Model fitting. **No refit** on later data. |
| Validation 1 (V1) | 106–126 | 65,544 | Hyper-parameter tuning and early stopping only |
| Validation 2 (V2) | 127–154 | 77,822 | Probability calibration (for R4, the calibrated cost-aware policy); threshold selection (R1, the F1-optimal threshold; R2); daily budget K; selection of the H3 configuration ([DR-010](decisions/DR-010-h3-confirmatory-configuration.md)) |
| Gap | — | — | **None in the primary design** |
| Final Test | 155–182 | 78,542 | **Sealed.** One confirmatory evaluation |

Partition hashes and `TransactionDT` bounds are recorded in [DR-001](decisions/DR-001-temporal-split-strategy.md).

**Supplementary analyses** (exploratory; development data only; no Test access):
- **Label-latency sensitivity:** repeat label-dependent choices (R1 threshold, R4 calibrator) using only V2 days 127–147. Also report which configuration the H3 selection rule would choose on those days (descriptive; the H3 configuration is not changed; [DR-010](decisions/DR-010-h3-confirmatory-configuration.md)).
- **Rolling-origin supplement for RQ2:** several V2-like → next-28-day transfers inside days 1–154, on the primary configuration only [REF-024, REF-025, REF-046]. The primary configuration is the H3 configuration's model and weighting ([DR-010](decisions/DR-010-h3-confirmatory-configuration.md)).

## Models

Defined in [DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md): two model families with two models each, each trained unweighted and with balanced class weighting, giving **eight configurations**.

| Family | Models | Configurations |
|---|---|---|
| Tree-based (T) | LightGBM (LGB) [REF-011]; CatBoost (Cat) [REF-012] | LGB-u, LGB-b, Cat-u, Cat-b |
| Neural (N) | ResNet-style MLP (RES); FT-Transformer (FTT) [REF-087] | RES-u, RES-b, FTT-u, FTT-b |

The tree-vs-neural family question is open in the literature for tabular data: no family is universally superior, and the gap is often small relative to tuning effects [REF-013, REF-014, REF-087, REF-088, REF-089, REF-090] ([literature review](literature-review.md), Section 4.8).

Optional reference (descriptive only, outside H1–H3; inclusion decided at the design review):

- L2-regularised (ridge-penalised) logistic regression. Any preprocessing pipeline is fitted on Train only.

Analytic baselines (always reported):
- random review (precision = prevalence);
- oracle ceiling at each capacity.

The model set is fixed by DR-011; no further model families are added.

## Training Procedure

**Common procedure (comparison fairness).** All eight configurations follow one procedure ([DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md) §3). The study does not alter any model's internal computation; every choice it controls is made identical at the widest level the models permit:

1. **Identical for all eight configurations:**
   - the partitions: Train for fitting, V1 for tuning and early stopping, V2 for calibration, thresholds, K and the H3 configuration;
   - the same 431 declared model inputs; any input added later through an approved GitHub Issue enters all eight configurations;
   - the same explicit Train-only class weights;
   - the same hyperparameter search method and budget per configuration;
   - the same V1 objective, early-stopping rule and seeds;
   - the same decision policies, capacity settings and evaluation.
2. **Identical within each family:**
   - tree-based: native categorical and missing-value handling ([DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md) §5), weighted log-loss, CPU training;
   - neural: the representation of [DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md) §9, weighted binary cross-entropy, bf16 mixed precision on the GPU.
3. **Model-specific:** each model's architecture and internal computation (self-attention, residual blocks, ordered boosting, leaf-wise growth) and its own search space (neural models: the large-dataset spaces (B) of REF-087, used by the authors for datasets of this size), tuned under the Level 1 method and budget.

The values of these elements (search method, budget, objective, patience, seeds, aggregation of neural-network seeds, tie rule for equal scores) are fixed in the training rules before the development runs ([Design items](#design-items-before-protocol-freeze)).

**Training environment.** Tree-based models train on CPU, with the CatBoost device fixed. Neural models train on the local GPU (8 GB) with bf16 mixed precision (bfloat16 computation, float32 weights), with memory-efficient attention for FT-Transformer. The engineering probe of Issue #50 (synthetic tensors) covered the width and depth grid of the FT-Transformer space (B), with its feed-forward size factor of 4/3. Under these settings, at 431 input tokens, FT-Transformer fits at batch sizes 512 (default setting) to 128 (largest setting); at 800 input tokens the largest setting fits at batch size 64, and at 128 with gradient checkpointing. float32 with standard attention runs out of memory for the larger settings ([DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md) §4).

## Feature Policy

- **Raw or minimally processed features are the primary design.** There is no feature-engineering study.
- **`TransactionID` and raw `TransactionDT` are not model features.** `TransactionID` is monotone in time, so it is a disguised timestamp [D]. Raw time cannot be extrapolated by tree models.
- **No future-looking aggregate features.** Any frequency, count or entity-level aggregate would have to be computed from Train only or with past-only windows. None is planned for the primary design.
- **No manual target encoding.** Categorical handling is native to each model family: CatBoost ordered target statistics [REF-012]; LightGBM categorical splits [REF-011]; learned category embeddings in the neural networks ([DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md) §9).
- **No engineered entity identifiers as features.** Heuristic keys such as card + address + implied account-start day are used **only** for audit and slicing.
- **Feature declaration** ([DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md); machine-readable in [`config/features.yaml`](../config/features.yaml); column-by-column reference in the [feature catalog](feature-catalog.md)). Of the 434 joined columns:
  - 1 is the target (`isFraud`) and 2 are excluded (`TransactionID`, `TransactionDT`);
  - **49 are categorical**, exactly the organiser's published list [REF-047]: `ProductCD`, `card1`–`card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `M1`–`M9`, `DeviceType`, `DeviceInfo`, `id_12`–`id_38`. This includes the numeric-coded fields `card1`–`card3`, `card5`, `addr1`, `addr2` and 12 `id_*` codes;
  - **382 are numeric**: every other column (`TransactionAmt`, `dist1`, `dist2`, `C1`–`C14`, `D1`–`D15`, the `V1`–`V339` feature columns, `id_01`–`id_11`).
- **Model input parity.** All four models receive the same 431 columns, built from one canonical frame:
  - categorical values are written as text labels, with missing values labelled `"__MISSING__"`;
  - **tree-based models:** numeric values stay `float64` with native missing-value handling; LightGBM receives pandas categoricals whose categories are fixed from Train only, and CatBoost receives the text labels directly [REF-076, REF-077]; no imputation, scaling, rare-level grouping or high-cardinality exclusion is applied;
  - **neural models (proposed, DR-008 §9; identical for both, attached to input roles):** per-input category vocabularies fitted on Train with `"__MISSING__"` reserved in every vocabulary, values unseen in Train mapped to it, and category dropout (probability 0.01, training only) so that its embedding is trained; numeric values quantile-transformed to a normal distribution fitted on Train, with Gaussian noise added to every numeric input during fitting only, scaled as in the reference implementation [REF-087, REF-103]; missing values set to 0 after the transformation, and one missingness indicator per numeric input with missing Train values (≈ 720 inputs in total).

## Class-Imbalance Treatment

Primary comparison:
- no class weighting;
- balanced class weighting.

**Implementation.**
- Weighting is **explicit** and computed **from Train only**. Each fraud row receives the sample weight below; legitimate rows receive 1:

  $$
  w_{\mathrm{pos}} = \frac{n_{\mathrm{neg}}}{n_{\mathrm{pos}}} \approx 28.5 \quad (\text{in Train})
  $$

  where `w_pos` = weight of each fraud (positive-class) row, `n_neg` = number of legitimate Train transactions and `n_pos` = number of fraud Train transactions.
- The same weights are applied to all four models (as sample weights in each model's loss), rather than any library's own "balanced" switch.

**Excluded.** SMOTE (Synthetic Minority Over-sampling Technique), random over-sampling and random under-sampling are outside the design:
- rebalancing distorts calibration without improving discrimination [REF-019, REF-020, REF-021];
- synthetic points have no valid timestamp in a temporal design;
- SMOTE is ill-suited to high-cardinality nominal data [REF-022].

## Review Capacity

Defined in [DR-002](decisions/DR-002-review-capacity-definition.md).

- **Capacity grid:** 1 %, **2 % (primary)**, 5 %, 10 %. These are **simulated research scenarios**, not real institutional capacities.
- **Fixed daily budget:**

  $$
  K = \operatorname{round}\left( c \times \text{mean daily transaction volume in V2} \right)
  $$

  where `K` = fixed daily review budget and `c` = review-capacity share. This gives K = 28 / **56** / 139 / 278 reviews per relative day for c = 1 / 2 / 5 / 10 %.
- **Capacity ÷ prevalence** is reported with every result. At 2 % it is 0.59 (binding regime); at 5 % and 10 % it is above 1 (slack regime).

Prior art for capacity-constrained fraud evaluation exists [REF-006, REF-004, REF-005, REF-046, REF-084], including capacity-aware threshold selection and allocation [REF-081, REF-082, REF-083] and temporal evaluation of capacity-constrained top-K review policies on IEEE-CIS [REF-078]. No novelty is claimed for it.
**Overflow rule** for the threshold policies (R0, the default 0.5 threshold; R1; R2) — **report-only, no truncation** ([DR-009](decisions/DR-009-overflow-handling-for-threshold-policies.md)). They are evaluated as uncapped threshold policies: every threshold-generated alert counts as review demand and is never truncated to K. Workload is measured against the fixed budget:

$$
\begin{aligned}
O_{d,j,c} &= \max\left(A_{d,j} - K_c,\ 0\right) \\[4pt]
U_{d,j,c} &= \max\left(K_c - A_{d,j},\ 0\right) \\[4pt]
L_{d,j,c} &= \frac{A_{d,j}}{K_c}
\end{aligned}
$$

where:

- `A_d,j` = alerts generated by threshold policy j ∈ {R0, R1, R2} on relative day d (threshold-generated review demand);
- `K_c` = fixed daily review budget for capacity scenario c;
- `O_d,j,c` = overflow (alerts in excess of the budget);
- `U_d,j,c` = unused capacity (budget left unfilled by the alerts);
- `L_d,j,c` = load ratio (demand relative to the budget).

R2_c is compared with its own K_c; R0 and R1 have one threshold each and are compared descriptively with every K_c.

- **Truncation is rejected.** The top K of a threshold's alerts are the top K of the whole day, so truncation would reproduce R3 on every overflow day and remove the workload demand that RQ2 studies.
- **Frauds in overflow.** Labels are retrospective research ground truth, not review outcomes. Frauds among alerts in excess of K are **not** counted as missed or as false negatives. Frauds below a threshold are false negatives of that threshold classifier. Any count of "flagged frauds in excess demand" is descriptive and conditional on score-priority ordering (DR-009).
- **Scope boundary.** The research measures the gap between alert demand and review capacity. It does not prescribe how an organisation closes that gap (carry-over, extra staff, escalation, dropping or deferral are not assumed).

## Decision Policies

### R0 — Default threshold

Threshold 0.5 on the raw model score (the model's uncalibrated predicted fraud probability).

Purpose: naive reference only. Its alert volume mainly reflects the raw score scale (which changes with class weighting). **Descriptive only; excluded from confirmatory tests.** Evaluated uncapped under the report-only overflow rule (DR-009).

### R1 — F1-optimal threshold

The threshold that maximises F1 (the harmonic mean of precision and recall), selected using V2 only [REF-009]. Capacity-agnostic; evaluated uncapped under the report-only overflow rule, with its alert demand, overflow and unused capacity reported against K (DR-009). Secondary comparator.

### R2 — Capacity-rate threshold

The score threshold τ_c is chosen on V2 so that a fraction c of V2 transactions is flagged (τ₂ is τ_c at c = 2 %). It uses V2 scores and needs no labels.

Purpose: object of RQ2/H3. Is a static validation-selected operating point transferable in time? Evaluated uncapped: its daily alert demand is compared with K_c, never truncated to it (DR-009).

### R3 — Daily top-K

Review the K highest-scoring transactions within each relative day.

Purpose: ranking-quality benchmark at full capacity, used for H1/H2. Note that it is an idealised end-of-day batch policy, not a streaming policy [REF-033].

R3 is the explicit **hard-capacity** policy: it reviews min(K, N_d) transactions per day, where N_d is the day's transaction volume. It therefore has no overflow by construction. It also has no unused capacity, because the feasibility audit's label-free volume inspection recorded at least 2,048 transactions on every relative day (≥ K for every capacity scenario; notebook §3). R0–R2, by contrast, describe threshold-generated **demand**, which may lie above or below K (DR-009).

### R4 — Calibrated cost-aware capped policy

Review transaction i if its calibrated probability p̂_i (the Platt-scaled fraud probability) is at least the analytic cost threshold p*, capped at K per day [REF-008, REF-073]. Defined in [DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md), where the derivation is shown step by step.

**Primary threshold** (review cost and missed-fraud cost only):

$$
\begin{aligned}
p^* &= \frac{C_R}{C_{\mathrm{miss}}} \\
&= \frac{1}{\rho}
\end{aligned}
$$

where:

- `p*` = cost-derived probability threshold; transaction i is reviewed if its calibrated probability p̂_i ≥ p*;
- `C_R` = cost of reviewing one transaction, charged on every review;
- `C_miss` = cost of missing a fraud (an unreviewed fraud);
- `ρ` = C_miss / C_R, the miss-to-review cost ratio.

A transaction is reviewed when its calibrated fraud probability is at least one divided by the miss-to-review cost ratio.

**Sensitivity: with a false-alert cost.** If an extra false-alert (customer-friction) cost C_FP per reviewed legitimate transaction is included:

$$
\begin{aligned}
p^* &= \frac{C_R + C_{\mathrm{FP}}}{C_{\mathrm{miss}} + C_{\mathrm{FP}}} \\
&= \frac{1}{\rho'} \\[6pt]
\rho' &= \frac{C_{\mathrm{miss}} + C_{\mathrm{FP}}}{C_R + C_{\mathrm{FP}}}
\end{aligned}
$$

where:

- `C_FP` = additional false-alert/customer-friction cost for reviewing a legitimate transaction;
- `ρ′` = effective cost ratio, which equals ρ when C_FP = 0.

The threshold keeps the same form, with the effective ratio ρ′ in place of ρ. The R4-vs-R3 comparison depends on costs only through ρ′, so C_FP is handled as a relabelling of the ratio axis (sensitivity only).

- Both equations follow from the expected-cost rule [REF-008] under the assumptions listed in DR-007: perfect verification, per-transaction costs, C_R on every review, and calibrated probabilities.
- **Queue:** R4 reviews the transactions of R3's daily queue (ranked by raw score, same tie rule) whose p̂_i ≥ p*. R4 is therefore always a subset of R3, and can only leave capacity unused.

The threshold is **derived, not tuned**.

Purpose: RQ3. When does leaving capacity unused become the cheaper choice under stated cost assumptions?

## Calibration

Calibration is a supporting component, not a separate research question.

- **Fitted on V2, not V1.** V1 scores are optimistic because V1 is used for early stopping, and V1 prevalence (4.22 %) differs from V2 (3.39 %): a −0.23 log-odds shift [D]. V2 is also closest in time to Test.
- **Relevant to R4 only.** R0–R3 rank and threshold on raw model scores, so calibration does not affect them. Platt scaling with a positive fitted slope is strictly monotone and preserves the ranking; isotonic regression is monotone but can merge scores into ties, which is why it serves only as a secondary check for R4 [I; REF-015, REF-017].
- **Primary method:** Platt scaling [REF-016]. **Secondary check:** isotonic regression [REF-017].
- **Metrics:** Brier score, log loss, calibration intercept and slope, reliability diagram [REF-018]. Reported on V2 (in-sample for the calibrator) and on Test.

## Cost Model

**Scope of RQ3.**
- RQ3 is evaluated **across the whole capacity grid (1 %, 2 %, 5 %, 10 %)**, not at 2 % alone.
- Its **main output is the break-even miss-to-review cost ratio ρ_BE at each capacity and for each model configuration**. ρ_BE is the largest assumed ratio of missed-fraud cost to review cost at which the cost-aware policy is cheaper than filling capacity. It is reported with the saturation ratio ρ_sat (the ratio above which the cost-aware policy fills the budget every day) and exact cost-difference curves [REF-039, REF-040, REF-072]. Definitions, grid and outputs are fixed in [DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md).
- At 2 % the budget is below the daily fraud count on every validation day. The cost-aware policy is expected to coincide with daily top-K review (reviewing the K highest-scoring transactions each day, K = fixed daily review budget) above a low saturation ratio, so RQ3 at 2 % is informative only at low ratios. The informative contrasts are expected at 5 % and 10 % ([DR-002](decisions/DR-002-review-capacity-definition.md), [DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md)).
- **Positioning.** Zhang et al. [REF-078] show, on IEEE-CIS, that a calibrated cost-threshold policy and full-capacity review are largely compressed toward the same capacity ceiling when capacity is tight and diverge when capacity is slack. RQ3 replicates and extends this pattern under this study's fixed daily capacity design, summarised as break-even and saturation ratios. It is not the study's main research gap.

The analysis separates:
- review effort (C_R);
- additional false-alert / customer-friction cost (C_FP, optional);
- missed-fraud cost (C_miss).

**Data-supported quantities [D]:**
- count of frauds captured and missed;
- amount-weighted recall (`TransactionAmt`, in United States dollars, USD, per the organiser [REF-048, partially verified]);
- fraud value exposed (Σ amount of missed fraud) as a proxy for loss exposure.

Amount-weighted results will have wide uncertainty: the top 20 % of development frauds by amount carry 62 % of fraud value [D].

**Assumed quantities** ([DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md)):
- all cost values are **simulated scenario assumptions** [REF-039];
- **primary cost model**, in units of one review cost, for policy π:

  $$
  \begin{aligned}
  J_\pi(\rho) &= N_\pi + \rho \cdot \mathrm{FN}_\pi \\
  \rho &= \frac{C_{\mathrm{miss}}}{C_R}
  \end{aligned}
  $$

  where:

  - `J_π(ρ)` = scenario cost of policy π (R3 or R4), in units of one review cost;
  - `N_π` = number of reviews; `FN_π` = number of frauds not reviewed;
  - `ρ` = C_miss / C_R, the miss-to-review cost ratio.

  The scenario cost counts one unit per review and ρ units per missed fraud;
- C_FP enters **only as a sensitivity analysis**: break-even ratios are converted for φ = C_FP / C_R ∈ {1, 5};
- **ratio grid:** ρ ∈ {2, 5, 10, 20, 50, 100, 200, 500, 1000} (log 1–2–5 spacing; same grid at every capacity). Curves are evaluated exactly over ρ ∈ (1, 1000] without interpolation. ρ ≤ 1 is excluded because reviewing can then never reduce expected cost [REF-008];
- **saturation ratio** ρ_sat: the smallest ratio at which R4 uses the full budget on every day (label-free; also computed on V2 before Test);
- **break-even ratio** ρ_BE, the largest ratio at which R4's scenario cost is strictly below R3's:

  $$
  \begin{aligned}
  \Delta J(\rho) &= J_{\mathrm{R4}}(\rho) - J_{\mathrm{R3}}(\rho) \\
  \rho_{\mathrm{BE}} &= \sup \lbrace \rho \in (1, 1000] : \Delta J(\rho) < 0 \rbrace
  \end{aligned}
  $$

  where:

  - `ΔJ(ρ)` = scenario-cost difference between R4 and R3 at ratio ρ;
  - `sup` = supremum: the largest value (least upper bound) of the set;
  - `(1, 1000]` = continuous analysis range of ρ.

  ΔJ is negative when the cost-aware policy is cheaper than filling capacity. ρ_BE is reported per capacity and configuration, with ties, censoring and non-monotone cases as defined in DR-007;
- amount-dependent miss costs are **not** used to change decisions. Amount enters only as an evaluation-only sensitivity analysis (amount-weighted recall, fraud value exposed, fraud value forgone), labelled as exposure proxies, never as loss or saving [REF-073].

**Uncertainty (descriptive).** 95 % percentile intervals at grid ratios come from a day-level moving-block bootstrap (block length ℓ = 3 days, B (number of bootstrap replicates) = 10,000, R3 and R4 paired). A DR-006 pigeonhole check is applied to frauds forgone. ρ_sat and ρ_BE are point estimates.

**Calibration dependence.** Under calibrated probabilities, R4 cannot be worse than R3 in expectation. An observed loss for R4 is consistent with calibration-transfer failure at the margin [REF-074] **or** with finite-sample variation among the few marginal transactions; the grid-point intervals, isotonic calibration and a declined-set calibration diagnostic are used to distinguish the two ([DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md)).

No assumed value will be presented as an actual institutional banking cost.

## Outcomes

### Primary outcome

Recall@2% under R3 (daily top-K, K = 56), pooled over the Test days. Reported with:
- the **oracle recall ceiling** (≈ 0.6 at 2 % in V2);
- **pooled normalised recall** ([DR-002](decisions/DR-002-review-capacity-definition.md)):

  $$
  \text{pooled normalised recall} = \frac{\sum_d \mathrm{TP}_d}{\sum_d \min(K, F_d)}
  $$

  where `TP_d` (true positives) = frauds reviewed on day d and `F_d` = frauds on day d. It compares the frauds reviewed with the most any policy could review, and equals Recall@K divided by the oracle ceiling. Days without frauds contribute 0 to both sums and are excluded from per-day summaries.

Precision@K is a rescaling of Recall@K (share of all frauds reviewed under budget K) under fixed K. It is reported descriptively and **not** treated as independent confirmatory evidence.

### Secondary outcomes

- Recall@1 %, 5 %, 10 %;
- false negatives at capacity;
- frauds per 1,000 reviews;
- lift over random review;
- utilisation of the capped policies R3 and R4; alert demand, overflow, unused capacity and load ratio of the threshold policies R0–R2 (see RQ2 and [DR-009](decisions/DR-009-overflow-handling-for-threshold-policies.md));
- Average Precision / PR-AUC (Precision–Recall Area Under the Curve) [REF-010] and ROC-AUC (Receiver Operating Characteristic Area Under the Curve);
- amount-weighted recall;
- scenario cost;
- calibration metrics for R4.

### Daily-budget fidelity (RQ2)

**Daily-budget fidelity** is a separate operational outcome, reported descriptively and **not** hypothesis-tested. Measures:
- daily alerts generated by the threshold (review demand, A_d; not a count of reviews performed);
- deviation and absolute deviation from K;
- percentage of days above and below K;
- total overflow and total unused capacity, as defined in [DR-009](decisions/DR-009-overflow-handling-for-threshold-policies.md);
- rate/volume decomposition.

See [DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md).

## Statistical Plan

**Primary confirmatory comparison:** Recall@2% (H1, H2), plus the realised alert rate r (H3). The H1/H2 inference plan is defined in [DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md).

**Uncertainty must be cluster- or day-block-aware.** Fraud labels cluster by entity: V2 frauds average ≈ 2 per heuristic pseudo-entity, with an upper-bound design effect (variance inflation due to clustering) ≈ 4.3 [D]. 45 % of V2 frauds belong to entities that span several days, and daily fraud counts are overdispersed (≈ 5 times binomial variance in V2) [D]. Transaction-level resampling would therefore understate uncertainty.

- **Primary (H1, H2):** a **two-way (pigeonhole) joint paired bootstrap** over relative days × fine pseudo-entities, as specified in [DR-006](decisions/DR-006-bootstrap-inference-strategy.md) [REF-068, REF-070]:
  - day multiplicities and entity multiplicities are drawn independently;
  - each Test fraud is weighted by the product of its day's and its entity's multiplicities;
  - the same weights are applied to the fixed daily top-K capture indicators of all eight configurations;
  - B = 10,000, percentile CIs [REF-044].

  Both day-level and cross-day entity dependence are observed in V2 [D], and each one-way scheme misses one of them. The method is mildly conservative by design [REF-068]. Resampling that assumes independent transactions or frauds is not acceptable.
- **Pre-declared sensitivity analyses (not confirmatory):**
  - S1: Owen-corrected two-way variance with Wald intervals:

    $$
    \mathrm{SE}^2 = V_{\mathrm{pigeonhole}} - 2 \cdot V_{\mathrm{naive}}
    $$

    where `SE` = corrected standard error, `V_pigeonhole` = variance across pigeonhole replicates and `V_naive` = variance from a fraud-level independent bootstrap;
  - S2: a key-free day-level moving-block bootstrap (ℓ = 3 days) that preserves each day's top-K exactly [REF-037, REF-064].

  A different H1 or H2 conclusion under S1 or S2 is reported as sensitivity to dependence assumptions. Confirmatory decisions use the primary method only.
- **H1: two-sided difference test.** It is not a winner-selection test. p_H1 is the two-sided bootstrap p-value for δ_F = 0, in the add-one form bounded to (0, 1] ([DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md)):

  $$
  p_{\mathrm{H1}} = \min\left\lbrace 1,\ 2 \cdot \min\left( \frac{1 + \#\lbrace \delta^*_F \le 0 \rbrace}{B + 1},\ \frac{1 + \#\lbrace \delta^*_F \ge 0 \rbrace}{B + 1} \right) \right\rbrace
  $$

  where `δ*_F` = δ_F in one bootstrap replicate and `#{ · }` = the number of replicates satisfying the condition.

  The 90 % CI for δ_F is additionally interpreted against ±0.02 (±SESOI). This interpretation is pre-specified but not a separate confirmatory test, and has five possible outcomes:
  - a practically meaningful difference;
  - a difference of uncertain practical size;
  - a detectable but negligible difference;
  - data compatible only with negligible differences;
  - inconclusive.

  The within-family model contrasts (CatBoost vs LightGBM; FT-Transformer vs ResNet-style MLP), the family × weighting interaction and the cell contrasts are reported descriptively.
- **H2: intersection-union TOST** with SESOI = 0.02 [REF-061, REF-062]. p_H2 is the maximum of the eight one-sided p-values (two per model), each in the same add-one form. H2₁ is concluded only if the Holm-adjusted p-value of H2 is at most 0.05.
  - Equivalence is required for each model, rather than for an averaged effect, so that opposite effects cannot cancel.
  - The 90 % CIs of the four δ_W,m and their position relative to (−0.02, +0.02) describe uncertainty only and do not determine rejection.
  - The two-sided difference CIs are also reported descriptively.
- **H3: TOST** with Δ = 0.004 (±20 % relative), on the H3 configuration only ([DR-010](decisions/DR-010-h3-confirmatory-configuration.md)) [REF-061, REF-062, REF-030]. p_H3 is the larger of the two one-sided bootstrap p-values, and H3₁ is concluded only if the Holm-adjusted p-value of H3 is at most 0.05 [REF-029] ([DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md)).
  - Uncertainty comes from a **moving-block bootstrap over Test days**, because the outcome does not depend on labels: block length 3 days, B = 10,000 bootstrap replicates, percentile interval [REF-037, REF-064]. Sensitivity block lengths are 1 and 7 days.
  - The 90 % CI for r is reported in every case, with its position relative to (0.016, 0.024). It describes uncertainty only and does not determine rejection: the decision uses the Holm-adjusted p-value, so a 90 % CI inside the band can accompany a non-rejection after multiplicity adjustment. A CI overlapping a bound is **inconclusive**.
  - Inference is conditional on the H3 configuration's trained model and its τ₂.
  - Non-confirmatory sensitivity: the CI is also reported against ±10 %, ±25 % and ±50 % bounds, together with the smallest tolerance at which equivalence would be concluded.
- **Multiplicity:** **Holm** (sequentially rejective multiple-testing procedure) at family-wise α = 0.05 over the confirmatory family {H1, H2, H3} [REF-029]: three members, with H3 tested once, on the H3 configuration. Decisions use **Holm-adjusted p-values**:

  $$
  \tilde{p}_{(k)} = \max_{i \le k} \ \min\left\lbrace 1,\ (4 - i) \cdot p_{(i)} \right\rbrace, \qquad k = 1, 2, 3
  $$

  where:

  - `p_(1) ≤ p_(2) ≤ p_(3)` = the three p-values p_H1, p_H2 and p_H3 in increasing order;
  - `i`, `k` = rank positions in that order (1, 2 or 3);
  - `p̃_(k)` = Holm-adjusted p-value of the hypothesis with the k-th smallest p-value.

  A hypothesis is rejected if and only if its adjusted p-value is at most 0.05. This applies Holm's step-down stopping rule: the ordered p-values are compared with 0.05 / 3 ≈ 0.0167, 0.05 / 2 = 0.025 and 0.05 in turn, and once one comparison fails, no later hypothesis is rejected. For example, p-values {0.020, 0.021, 0.040} give adjusted values {0.060, 0.060, 0.060}, so no hypothesis is rejected. The unadjusted and the Holm-adjusted p-value are both reported for H1, H2 and H3; only the adjusted value decides. H3 results for the other seven configurations are descriptive and outside the family (DR-010).
- **Secondary:** exact McNemar tests on the cell-level paired contrasts [REF-035, REF-028]. They assume independent frauds, and under intracluster correlation their size can greatly exceed the nominal level [REF-071]. They are descriptive only, **not** in the Holm family.
- **Effect sizes** with CIs, in pp, frauds per 28 days and per day, Precision@K equivalent and normalised recall. Sensitivity: H1/H2 interpretations are also shown against SESOIs of 1, 3 and 5 pp (not confirmatory).
- **Secondary outcomes** use estimation with confidence intervals, not additional hypothesis tests. Proportions use Wilson intervals [REF-034] where a bootstrap is not required.

**Power (projection, not Test data).** About 2,660 Test frauds are projected from V2 prevalence. The minimum detectable paired difference in Recall@2% is roughly:
- 1.2–2.8 percentage points if frauds are treated as independent;
- 2–5.6 percentage points under design effects of 2–4.

At the SESOI of 2 pp, under central assumptions (≈ 10 % of frauds discordant between configurations, design effect ≈ 2):
- power is ≈ 0.47–0.64 for the H1 difference test;
- power is ≈ 0.14–0.49 for a single-contrast equivalence test.

H2's intersection-union test over four models has lower power than a single contrast, because every model must show equivalence.

It is higher under lower discordance or clustering, and lower otherwise (see [DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md)). The dependence diagnostics in [DR-006](decisions/DR-006-bootstrap-inference-strategy.md) suggest that two-way variance may exceed the design effects of 1–4 assumed there, so real power may be lower. H1 and H2 are therefore powered only for moderate effects. **Inconclusive results are a pre-declared, legitimate outcome** and will be reported neutrally.

Statistical significance will not be treated as equivalent to practical importance.

## False-Negative Analysis

Exploratory, descriptive and non-causal. False negatives are defined under **R3 at the primary capacity** (frauds outside the daily top-K, K = 56). Frauds among threshold alerts in excess of K under R0–R2 are not false negatives in this analysis ([DR-009](decisions/DR-009-overflow-handling-for-threshold-policies.md)). Pre-declared slices:

- identity information present / absent;
- ProductCD;
- card4 and card6;
- DeviceType (including missing);
- transaction-amount bands;
- relative week within Test;
- score band;
- **recurring vs new pseudo-entity** (whether the transaction's heuristic entity already had a Train-period fraud label).

The analysis will report:
- fraud count per slice;
- miss rate (FN, false negatives, ÷ frauds in slice);
- uncertainty interval;
- share of total false negatives.

**Interpretability rule:** slices with fewer than about 100 frauds are pooled or reported descriptively only.

Slice base rates differ strongly (e.g. identity present 9.4 % vs absent 2.1 % in V2 [D]), so raw false-negative counts are never interpreted alone. No causal claim will be made from subgroup associations.

## Leakage Controls

Before the protocol is frozen [REF-026, REF-027]:

- all partitions are mutually exclusive and chronological (asserted and hashed; [DR-001](decisions/DR-001-temporal-split-strategy.md));
- no random or shuffled splitting or cross-validation across time;
- preprocessing, encoders, category vocabularies, quantile transformations, class weights and any baseline pipelines are fitted on Train only;
- tuning and early stopping use V1; calibration, threshold selection and K use V2;
- final Test labels do not influence features, models, policies, capacities or cost scenarios;
- no future-looking aggregates; no manual target encoding; `TransactionID` and raw `TransactionDT` are excluded;
- all Test access is logged.

**Known uncontrollable risks, stated as threats to validity:**
- entity-level label dependence combined with the absence of label-report dates, which gives optimistic results;
- organiser-engineered C / D / V feature groups (named by their column prefixes) whose look-back windows are undocumented.

## Ethics

The study uses secondary, anonymised data and does not recruit human participants.

The University of Greenwich Research Ethics Policy applies to master's students' research (§1.2) [REF-050]. It lists research involving "harmful, criminal, particularly sensitive … subject matters" as requiring ethical consideration (§1.3.4) and as **Higher Risk** (§2.2.1 iv) [REF-050]. Separate guidance states that fully anonymised secondary data should not normally need approval, and that taught students are advised on the route by their programme or module leader [REF-051].

**Ethics and governance status (Issue #31).** The supervisor has given written approval of the project topic. A separate, explicit determination of the ethics route (including whether §1.3.4 applies) was requested but was not provided. The project therefore proceeds under the available University of Greenwich research-governance guidance [REF-050, REF-051] and the project data-governance controls below. **No formal ethics approval, ethics exemption or ethics identifier is claimed.** The competition data-use terms have been verified [REF-049]: the Rules permit non-commercial use for academic research and education (Section 7A) and prohibit making the Competition Data available to any party not participating in the competition (Section 7B). As a stricter project policy, raw data and row-level derived data stay local, out of Git, and are not shared with anyone, including supervisors or examiners; aggregate results may be published ([`dataset.md`](dataset.md#data-use-terms)).

Artificial intelligence (AI) assistance used in this project must be declared in the AI Reflection (AIR) required by University guidance [REF-053].

## Design items before protocol freeze

**Fixed design elements** (recorded in Decision Records; supervisor review pending):

| Element | Specification | Record |
|---|---|---|
| Temporal split | Train 1–105, V1 106–126, V2 127–154, Test 155–182 (sealed); no primary gap | [DR-001](decisions/DR-001-temporal-split-strategy.md) |
| Review capacity | 1 / 2 (primary) / 5 / 10 %; fixed daily budget K = 28 / 56 / 139 / 278 from V2 volume | [DR-002](decisions/DR-002-review-capacity-definition.md) |
| RQ2 and H3 | Alert-rate transfer equivalence, Δ = 0.004 (±20 % relative); TOST with a day-level moving-block bootstrap, decided by its Holm-adjusted p-value | [DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md) |
| SESOI, H1 and H2 | SESOI = 2 pp of Recall@2%; H1 family-effect difference test with practical interpretation; H2 intersection-union TOST over four models; Holm over {H1, H2, H3} | [DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md) |
| H1/H2 uncertainty | Two-way pigeonhole bootstrap (relative day × fine pseudo-entity); sensitivity S1 (Owen-corrected variance) and S2 (day blocks, ℓ = 3) | [DR-006](decisions/DR-006-bootstrap-inference-strategy.md) |
| RQ3 cost analysis | Review + missed-fraud cost model (C_FP as a sensitivity relabelling); ρ ∈ {2, …, 1000} with exact curves; saturation and break-even ratios per capacity and configuration | [DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md) |
| Model inputs | 49 categorical, 382 numeric, 2 excluded; canonical labels with `"__MISSING__"`; tree-based representation fixed | [DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md) §§1–8 |
| Overflow rule | Report-only: R0–R2 evaluated uncapped; demand, overflow, unused capacity and load ratio against K | [DR-009](decisions/DR-009-overflow-handling-for-threshold-policies.md) |
| H3 configuration | Highest pooled V2 Recall@2% under R3 among eight configurations; committed before Test access | [DR-010](decisions/DR-010-h3-confirmatory-configuration.md) |
| Model set | LightGBM, CatBoost, ResNet-style MLP, FT-Transformer × unweighted/balanced; common-procedure fairness | [DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md) |
| Data-use terms | Non-commercial academic use (Rules Section 7A); no redistribution (Section 7B); raw and row-level data stay local | [`dataset.md`](dataset.md#data-use-terms) |
| Ethics and governance | Documented status: topic approval received; no formal ethics approval, exemption or identifier claimed | [Ethics](#ethics) |

**Items to resolve before the freeze:**

| # | Item | Notes |
|---|---|---|
| 1 | Supervisor review of the design (DR-001–DR-012) | Design review with the supervisor (Issue #56) |
| 2 | Neural-network input representation | DR-008 §9, proposed: vocabularies with reserved `"__MISSING__"` and category dropout, quantile transformation with the reference-implementation noise and 1,000 quantiles [REF-103], missingness indicators; noise seed and constant or all-missing inputs fixed in the training rules |
| 3 | Training rules | Search method and budget, V1 objective, early-stopping rule, seeds, aggregation of neural-network seeds, tie rule for equal scores, calibration details |
| 4 | Feature-engineering policy | Current design: declared raw inputs only (DR-008); confirmed or changed at the design review |
| 5 | Logistic-regression reference baseline | Include as a descriptive reference, or not |
| 6 | University of Greenwich AI-use tier and AI Reflection (AIR) | Tier set by the assessment brief; confirmed with the supervisor [REF-053] |
| 7 | Research-methodology framing | IBM Foundational Methodology adaptation [REF-085], recorded after the design review |
| 8 | Dataset manifest | SHA-256 (Secure Hash Algorithm, 256-bit) values and partition hashes reproduced by the split-generation code |

## Protocol Freeze

The protocol becomes confirmatory only after:

1. every design item above is resolved;
2. the literature evidence base justifies the design ([`literature-review.md`](literature-review.md), [`literature-evidence-matrix.md`](literature-evidence-matrix.md));
3. the project's ethics and governance status is documented against the available University of Greenwich guidance (see [Ethics](#ethics) and [Design items](#design-items-before-protocol-freeze));
4. the raw dataset statistics and split hashes are reproduced by the split-generation code;
5. hypotheses, SESOIs, Δ and statistical tests are fixed;
6. cost scenarios are declared;
7. the training rules are fixed and the implementation passes its automated checks.

After that point the protocol is tagged in Git before the final Test evaluation.

## Decision Records

- [DR-001 — Temporal split strategy](decisions/DR-001-temporal-split-strategy.md)
- [DR-002 — Review-capacity definition](decisions/DR-002-review-capacity-definition.md)
- [DR-003 — RQ2 / H3: policy-transfer fidelity](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md)
- [DR-005 — SESOI and H1/H2 inference plan](decisions/DR-005-sesoi-and-h1-h2-inference.md)
- [DR-006 — Bootstrap inference strategy for H1 and H2](decisions/DR-006-bootstrap-inference-strategy.md)
- [DR-007 — RQ3 cost-ratio grid and break-even rule](decisions/DR-007-rq3-cost-ratio-and-break-even.md)
- [DR-008 — Feature declaration and model-input policy](decisions/DR-008-feature-declaration-and-model-input-policy.md)
- [DR-009 — Overflow handling for threshold-based policies R0, R1 and R2](decisions/DR-009-overflow-handling-for-threshold-policies.md)
- [DR-010 — Confirmatory configuration for H3](decisions/DR-010-h3-confirmatory-configuration.md)
- [DR-011 — Model set: tree-based and neural model families](decisions/DR-011-model-set-tree-based-and-neural-families.md)

## References

Full entries and verification status are in [`docs/references.md`](references.md).

REF-001, REF-002, REF-003, REF-004, REF-005, REF-006, REF-008, REF-009, REF-010, REF-011, REF-012, REF-013, REF-014, REF-015, REF-016, REF-017, REF-018, REF-019, REF-020, REF-021, REF-022, REF-024, REF-025, REF-026, REF-027, REF-028, REF-029, REF-030, REF-033, REF-034, REF-035, REF-037, REF-039, REF-040, REF-043, REF-044, REF-046, REF-047, REF-048\*, REF-049, REF-050, REF-051, REF-053, REF-061, REF-062, REF-063, REF-064, REF-065, REF-068, REF-070, REF-071, REF-072, REF-073, REF-074, REF-076, REF-077, REF-078, REF-081\*, REF-082, REF-083, REF-084, REF-085, REF-087, REF-088, REF-089, REF-090, REF-103.

\* Partially verified.

## Symbols and notation

Every symbol used in this protocol, in alphabetical order. Each is also defined where it first appears.

| Symbol / abbreviation | Meaning |
|---|---|
| [D], [I] | Evidence tags: dataset observation; inference |
| A_d, A_d,j | Alerts generated by a threshold policy (j ∈ {R0, R1, R2}) on day d: threshold-generated review demand, not reviews performed |
| O_d,j,c | Overflow: max(A_d,j − K_c, 0) |
| U_d,j,c | Unused capacity: max(K_c − A_d,j, 0) |
| L_d,j,c | Load ratio: A_d,j / K_c |
| j | A threshold policy: R0, R1 or R2 |
| K_c | Fixed daily review budget for capacity scenario c (K when the scenario is clear) |
| α | Significance level (family-wise 0.05) |
| p_H1, p_H2, p_H3 | p-values of H1, H2 and H3 (H2: maximum of eight one-sided p-values; H3: larger of two) |
| p_(k), p̃_(k) | k-th smallest of p_H1, p_H2, p_H3; its Holm-adjusted p-value |
| i, k | Rank positions (1–3) of the ordered p-values in Holm's procedure |
| # { · } | Number of bootstrap replicates satisfying a condition |
| bf16 | bfloat16: 16-bit floating-point format used for mixed-precision neural-network training |
| arg max | The argument (here, configuration) at which a quantity is largest |
| AI, AIR | Artificial intelligence; AI Reflection (University of Greenwich AI-use declaration) |
| IBM | International Business Machines Corporation (author of the methodology framing, REF-085) |
| B | Number of bootstrap replicates (10,000) |
| c | Review-capacity share (1 %, 2 %, 5 % or 10 %) |
| C_FP | Extra false-alert/customer-friction cost of reviewing a legitimate transaction (sensitivity only) |
| C_miss | Cost of missing (not reviewing) a fraud |
| C_R | Cost of reviewing one transaction, charged on every review |
| Cat, LGB, RES, FTT | CatBoost, LightGBM, ResNet-style MLP, FT-Transformer |
| CPU, GPU | Central and graphics processing units |
| CI | Confidence interval |
| d | A relative day |
| Δ | H3 equivalence tolerance on r (0.004 = 20 % of the 2 % target) |
| δ_F, δ*_F | Model-family effect: neural mean minus tree-based mean of Recall@2% (H1 estimand); its value in one bootstrap replicate |
| δ_W,m | Weighting effect of model m (H2 estimands) |
| ΔJ(ρ) | Scenario-cost difference J_R4 − J_R3 (negative = R4 cheaper) |
| `__MISSING__` | Explicit label given to a missing categorical value (all models) |
| DR | Decision Record |
| F1 | Harmonic mean of precision and recall |
| `float64` | 64-bit floating-point numeric type |
| F_d | Number of frauds on day d |
| f, w | Model in the H3 selection rule (LGB, Cat, RES or FTT); weighting (u unweighted, b balanced) |
| FT-Transformer | Feature Tokenizer + Transformer model for tabular data |
| (f*, w*) | H3 configuration: highest pooled V2 Recall@2% under R3 (DR-010) |
| LGB-u, Cat-u, RES-u, FTT-u, LGB-b, Cat-b, RES-b, FTT-b | The eight model × weighting configurations (model-weighting), in tie-break order |
| m | A model: LGB, Cat, RES or FTT |
| MLP | Multilayer perceptron |
| N, T | Neural family {RES, FTT}; tree-based family {LGB, Cat} |
| FN, FN_π | False negatives: frauds not reviewed (by policy π) |
| H1, H2, H3 | Confirmatory hypotheses: model family; class weighting; alert-rate transfer fidelity. Subscript ₀ = null, ₁ = alternative |
| J_π(ρ) | Scenario cost of policy π, in units of one review cost |
| K | Fixed daily review budget (28 / 56 / 139 / 278 reviews per relative day) |
| ℓ | Block length of a moving-block bootstrap, in days |
| n_d | Number of transactions on day d with score ≥ τ₂ (for confirmatory H3, the H3 configuration's score and τ₂) |
| N_d | Total number of transactions on day d (volume; not to be confused with N_π) |
| n_neg, n_pos | Numbers of legitimate and fraud transactions in Train |
| N_π | Number of reviews made by policy π |
| p* | Cost-derived probability threshold used by R4 |
| p̂_i | Calibrated (Platt-scaled) fraud probability of transaction i |
| φ | False-alert-to-review cost ratio, C_FP / C_R |
| π | A decision policy (R3 or R4) in J_π and N_π |
| pp | Percentage points |
| PR-AUC | Precision–Recall Area Under the Curve |
| Precision@K | Share of reviewed transactions that are fraud |
| r | Realised Test alert rate at τ₂, pooled over Test days; in confirmatory H3, r = r^f*,w* (H3 configuration only) |
| r^f*,w* | Realised Test alert rate of the H3 configuration at its own τ₂ (DR-010) |
| R0–R4 | Decision policies: default 0.5 threshold; F1-optimal threshold; capacity-rate threshold; daily top-K; calibrated cost-aware capped policy |
| R^m,w | Pooled Test Recall@2% of model m with weighting w |
| R^f,w_V2 | Pooled V2 Recall@2% under R3 of model f with weighting w (H3 configuration selection) |
| Recall@K, Recall@2% | Share of all frauds reviewed under budget K (at 2 % capacity) |
| REF-NNN | Reference identifier in `docs/references.md` |
| ρ | Miss-to-review cost ratio, C_miss / C_R |
| ρ′ | Effective cost ratio, (C_miss + C_FP) / (C_R + C_FP) |
| ρ_BE | Break-even ratio: largest ρ at which R4 is strictly cheaper than R3 |
| ρ_sat | Saturation ratio: smallest ρ at which R4 uses the full budget every day |
| ROC-AUC | Receiver Operating Characteristic Area Under the Curve |
| RQ1–RQ3 | Research questions 1–3 |
| S1, S2 | Sensitivity analyses: Owen-corrected variance; day-level moving-block bootstrap |
| SE | Standard error |
| SESOI | Smallest Effect Size Of Interest (2 percentage points of Recall@2%) |
| SHA-256 | Secure Hash Algorithm, 256-bit |
| SMOTE | Synthetic Minority Over-sampling Technique |
| sup | Supremum: the largest value (least upper bound) of a set |
| Σ_d | Sum over days |
| τ₂, τ_c | V2-selected score threshold flagging 2 % (a fraction c) of V2 transactions |
| TOST | Two One-Sided Tests |
| TP_d | True positives: frauds reviewed on day d |
| pooled normalised recall | Σ_d TP_d / Σ_d min(K, F_d): Recall@K divided by the oracle ceiling |
| `TransactionDT` | Time offset in seconds from an undisclosed reference point |
| USD | United States dollars |
| V_naive, V_pigeonhole | Variance from a fraud-level independent bootstrap; variance across pigeonhole replicates |
| V1, V2 | Validation 1 (relative days 106–126); Validation 2 (relative days 127–154) |
| `V1`–`V339` | Organiser-engineered numeric **feature columns** (not the V1/V2 validation partitions) |
| `ProductCD`, `card1`–`card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `M1`–`M9`, `DeviceType`, `DeviceInfo`, `id_01`–`id_38` | Raw dataset columns; `id_12`–`id_38` and the others listed here except `id_01`–`id_11` are categorical (DR-008) |
| `TransactionAmt`, `dist1`, `dist2`, `C1`–`C14`, `D1`–`D15` | Raw numeric dataset columns (amount, distances, counts, time deltas) |
| `TransactionID` | Row identifier and join key (excluded from model input) |
| `isFraud` | Target: 1 = fraud, 0 = legitimate |
| w_pos | Sample weight of each fraud row under balanced weighting, n_neg / n_pos |
| ⌊ · ⌋ | Round down to a whole number |
