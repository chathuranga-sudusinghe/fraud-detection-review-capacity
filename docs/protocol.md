# COMP1885 Research Protocol

## Status

**Working protocol — pre-freeze. Not yet confirmatory.**

This file defines the confirmatory research design. It builds on the dataset feasibility audit ([`notebooks/01_dataset_feasibility_audit.ipynb`](../notebooks/01_dataset_feasibility_audit.ipynb), conclusion: *"Suitable with methodological changes"*), the data-understanding notebooks 02–03 and the structured literature review ([`literature-review.md`](literature-review.md)). Methodology decisions are recorded in [`docs/decisions/`](decisions/README.md) (DR-001–DR-015).

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

The model families are defined in [DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md): tree-based (LightGBM, CatBoost) and neural (ResNet-style multilayer perceptron, MLP; Feature Tokenizer + Transformer, FT-Transformer).

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
- No fraud- or alerting-specific literature standard exists, so the magnitude is a pre-specified judgement. It is anchored in V2 volume-driven workload variation (upper quartile 16.7 %, 90th percentile 26.5 %) and in a precision-feasibility check.
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

**IEEE Computational Intelligence Society (IEEE-CIS) Fraud Detection** [REF-047].

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
| Transaction time `TransactionDT` | Relative day; partitions; daily review queue; daily budget; day-level bootstrap; the two time-of-day inputs | `train_transaction.csv` | All | Reference point undisclosed: relative days and a phase within the relative day only, no calendar, clock-time or seasonal claims. Complete on all 182 relative days [D]; a missing or invalid value stops data validation ([Feature Policy](#feature-policy)) |
| Join and row key `TransactionID` | Joining the two files; aligning scores, labels and partitions row by row | Both files | All | Monotone in time, so never a model input ([Feature Policy](#feature-policy)) |
| Model inputs (433: 431 declared columns and two time-of-day inputs) | Fraud scores | Both files, left-joined on `TransactionID`; time-of-day inputs derived from `TransactionDT` | All; every fitted transformation uses Train only | Most meanings masked by the provider: the organiser's declared roles are adopted ([Feature Policy](#feature-policy)) |
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
| Validation 1 (V1) | 106–126 | 65,544 | Hyperparameter tuning and early stopping only |
| Validation 2 (V2) | 127–154 | 77,822 | Probability calibration (for R4, the calibrated cost-aware policy); threshold selection (R1, the F1-optimal threshold; R2); daily budget K; selection of the H3 configuration ([DR-010](decisions/DR-010-h3-confirmatory-configuration.md)) |
| Gap | — | — | **None in the primary design** |
| Final Test | 155–182 | 78,542 | **Sealed.** One confirmatory evaluation |

Partition hashes and `TransactionDT` bounds are recorded in [DR-001](decisions/DR-001-temporal-split-strategy.md).

**Supplementary analyses** (exploratory; development data only; no Test access):
- **Label-latency sensitivity:** repeat label-dependent choices (R1 threshold, R4 calibrator) using only V2 days 127–147. For R1, the same selection rule is applied to days 127–147 and the resulting frozen threshold is evaluated on days 148–154; the primary full-V2 threshold is unchanged. For R4, the selected main-seed models are unchanged, and both calibrators (Platt and isotonic) are refitted on days 127–147 with their smoothing counts recomputed and evaluated on days 148–154; these calibrators do not replace the primary full-V2 calibrators. Also report which configuration the H3 selection rule would choose on those days (descriptive; the H3 configuration is not changed; [DR-010](decisions/DR-010-h3-confirmatory-configuration.md)).
- **Rolling-origin supplement for RQ2:** an exploratory, descriptive look at how variable the transfer of an R2 threshold is across several origins inside days 1–154, on the primary configuration only [REF-024, REF-025, REF-046]. The primary configuration is the H3 configuration's model and weighting ([DR-010](decisions/DR-010-h3-confirmatory-configuration.md)); the supplement runs after the H3 selection and never changes it, any threshold or the confirmatory plan. It is a **retrospective sensitivity analysis conditional on the final development-selected specification**: that configuration and its hyperparameters were selected on V1 and V2, which lie after the early origins, so it is not a prospective deployment simulation [I]. The design is fixed in [DR-001](decisions/DR-001-temporal-split-strategy.md) and `config/dataset.yaml` (`rolling_origin`):
  - **Origins.** The origin day d_o (the first day of the threshold window) is 71, 85 or 99, 14 days apart. At d_o, the primary configuration is refitted on relative days 1 to d_o − 22 (days 1–49, 1–63 and 1–77), leaving a gap of 21 days, as between the end of Train and V2 in the main design; the threshold window is days d_o to d_o + 27 and the transfer window the next 28 days, d_o + 28 to d_o + 55 (the last origin's transfer window is days 127–154).
  - **Refit.** The hyperparameters and the fixed length (the retained rounds or epochs) of the selected trial, the main seed, no search and no early stopping; the training procedure, devices and numerical settings are those of [DR-015](decisions/DR-015-common-training-rules.md) and [DR-013](decisions/DR-013-training-device-policy.md). Everything learned from data is fitted on the origin's training days only: the balanced class weights and, for a neural model, the shared preprocessing of [DR-012](decisions/DR-012-neural-network-input-representation.md) with its declared seed; the main-run artifact is not reused. The batch plan of the selected trial is reused, and a plan that would leave a last micro-batch below the architecture's minimum batch size on an origin's training rows (two rows for the ResNet-style MLP) stops the supplement before any fit. A fixed length chosen on the 105 training days is applied to shorter training sets, and the origins differ in calendar position, training history and refit; the results combine these.
  - **Threshold and budget.** R2 at 2 % capacity on the scores of the threshold window (no labels): the threshold is the score ranked ⌈0.02 N_window⌉ among the N_window transactions of that window, and a transaction is flagged when its score is at least the threshold. The origin's fixed daily budget is K = round(0.02 × the mean daily volume of the threshold window), rounded half up ([DR-002](decisions/DR-002-review-capacity-definition.md) §3), and is never recomputed from the transfer window.
  - **Reported.** For each origin: the training days and rows, the threshold, K, the realised rate of the threshold window, the pooled transfer rate r (alerts over transactions of the transfer window), r / 0.02 − 1, the days above and below K, the total overflow and unused capacity, and the rate and volume components of [DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md) (with the window's mean daily volume). Across origins: the origins in chronological order and the number of origins whose r lies strictly inside the H3 band (0.016, 0.024), written "x of 3". No interval or test is given and no claim of equivalence or reliability is made: the origins share training days and their transfer windows overlap.

## Models

Defined in [DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md): two model families with two models each, each trained unweighted and with balanced class weighting, giving **eight configurations**.

| Family | Models | Configurations |
|---|---|---|
| Tree-based (T) | LightGBM (LGB) [REF-011]; CatBoost (Cat) [REF-012] | LGB-u, LGB-b, Cat-u, Cat-b |
| Neural (N) | ResNet-style MLP (RES); FT-Transformer (FTT) [REF-087] | RES-u, RES-b, FTT-u, FTT-b |

The tree-vs-neural family question is open in the literature for tabular data: no family is universally superior, and the gap is often small relative to tuning effects [REF-013, REF-014, REF-087, REF-088, REF-089, REF-090] ([literature review](literature-review.md), Section 2).

**Logistic-regression reference (descriptive only)** ([DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md) §5; Issue #115). One unweighted L2-regularised (ridge-penalised) logistic regression is reported as a reference. It is outside the eight configurations, H1–H3, the Holm adjustment, the H3-configuration selection and RQ2/RQ3, and supports no conclusion about class weighting for logistic regression.

- **Inputs.** The shared Train-fitted preprocessing artifact of [DR-012](decisions/DR-012-neural-network-input-representation.md): quantile-transformed numeric inputs, numeric missing values set to 0, the 324 missingness indicators and the Train vocabularies. Categorical inputs are one-hot encoded over every vocabulary category, including `"__MISSING__"`, with no category dropped (dropping one breaks the symmetry of the representation under a penalty [REF-135]); values unseen in Train map to `"__MISSING__"`. The matrix is a 64-bit floating-point (float64) compressed sparse row (CSR) matrix with no additional scaling and no category dropout; its column order, shape and the artifact's identity are recorded. No preprocessing is fitted on V1, V2 or Test.
- **Fitting.** scikit-learn 1.9.1 `LogisticRegression`: `solver="lbfgs"` (the quasi-Newton solver), `l1_ratio=0` (L2 penalty), `fit_intercept=True` (fit an intercept), `class_weight=None` (no class weighting) and `warm_start=False` (start each fit independently) [REF-135]. The study fixes `tol=1e-5` (convergence tolerance, the gradient stopping threshold of the solver) and `max_iter=5000` (iteration limit), a tighter tolerance and a larger iteration allowance than the library defaults (1e-4 and 100) [I; Issues #115 and #151]; these settings do not guarantee convergence, and the ten candidates are checked on Train before the first development run ([DR-015](decisions/DR-015-common-training-rules.md) §8). Each candidate is fitted to convergence on Train, without V1 early stopping. Non-convergence, non-finite coefficients or invalid probabilities (non-finite or outside [0, 1]) stop the analysis for an issue, with no solver substitution, candidate replacement or iteration extension.
- **Search.** Ten candidates: trial 1 uses C = 1 (the library default; C is the inverse regularisation strength); trials 2–10 draw C from LogUniform[10⁻⁴, 10⁴] (continuous log-uniform sampling) with the sampling formula of [DR-015](decisions/DR-015-common-training-rules.md) §6 (search and fitting: DR-015 §8). The range is that of scikit-learn's default `LogisticRegressionCV` grid [REF-135]; the log-uniform draw is this study's adaptation [I]. Selection uses the V1 objective of DR-015 §3 (unweighted daily top-62 recall), with exact ties to the lowest trial number. The candidates come from the fifth child stream of the sampler (DR-015 §1), spawned from the same parent after the four model streams, so the existing candidate lists are unchanged [REF-136].
- **Reported.** R3 recall at 1 %, 2 %, 5 % and 10 % capacity (K = 28, 56, 139 and 278), with the oracle-normalised recall, the random-review context and the computational cost; on V2 descriptively and on Test only in the authorised confirmatory evaluation. It is not calibrated and is not evaluated under R0–R2 or R4. Being deterministic, it has no additional-seed refits.

Analytic baselines (always reported):
- random review (precision = prevalence);
- oracle ceiling at each capacity.

The model set is fixed by DR-011; no further model families are added.

## Training Procedure

**Common procedure (comparison fairness).** All eight configurations follow one procedure ([DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md) §3). The study does not alter any model's internal computation; every choice it controls is made identical at the widest level the models permit:

1. **Identical for all eight configurations:**
   - the partitions: Train for fitting, V1 for tuning and early stopping, V2 for calibration, thresholds, K and the H3 configuration;
   - the same 433 model inputs (the 431 declared columns and the two time-of-day inputs); any input added later through an approved GitHub Issue enters all eight configurations;
   - the same explicit Train-only class weights;
   - the same hyperparameter search method and budget per configuration;
   - the same V1 objective, early-stopping rule and seeds;
   - the same decision policies, capacity settings and evaluation.
2. **Identical within each family:**
   - tree-based: native categorical and missing-value handling ([DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md) §5), weighted log-loss, the device settings of [DR-013](decisions/DR-013-training-device-policy.md);
   - neural: the representation of [DR-012](decisions/DR-012-neural-network-input-representation.md), weighted binary cross-entropy, bfloat16 (bf16) mixed-precision training and 32-bit floating-point (float32) prediction on the graphics processing unit (GPU; [DR-013](decisions/DR-013-training-device-policy.md)).
3. **Model-specific:** each model's architecture and internal computation (self-attention, residual blocks, ordered target statistics, leaf-wise growth), its training device ([DR-013](decisions/DR-013-training-device-policy.md) §1: LightGBM on the central processing unit (CPU), the other models on the GPU) and its own search space (neural models: the large-dataset spaces (B) of REF-087, used by the authors for datasets of this size), tuned under the Level 1 method and budget.

**Training rules** ([DR-015](decisions/DR-015-common-training-rules.md); identical for all eight configurations):

- **Search:** random search with **10 trials per configuration**. Trial 1 is the model's reference configuration; trials 2–10 are drawn from its search space. Each model's candidate list (10 ordered candidates) is generated before any fit and recorded before any V1 result is seen. Sampling uses `numpy.random.default_rng` with the PCG64 bit generator and sampler seed 42. One independent child stream per model is created using `SeedSequence.spawn`, assigned in DR-011 model order (LightGBM, CatBoost, ResNet-style MLP, FT-Transformer). The sampler streams are separate from the training seeds. The unweighted and balanced configurations of a model use the same list. No additional, repeated or replacement trial enters selection after V1 results are seen, and no configuration receives a budget extension. The separate reproducibility checks (Seeds, below) do not alter selection. The search spaces and reference configurations of both families are listed below.
- **Tree-based search spaces and reference configurations** ([DR-015](decisions/DR-015-common-training-rules.md) §6). Uniform and LogUniform denote continuous uniform and log-uniform sampling between the bounds; UniformInt denotes uniform sampling over the integers between the inclusive bounds; LogUniformInt denotes a log-uniform draw rounded down to an integer; the mixture 0.5 · {0} + 0.5 · LogUniform gives exactly 0 with probability 0.5. For trials 2–10, parameters are drawn independently, in the order listed, from the model's child stream; the exact sampling formulas and the sources of every range are in DR-015 §6. Trial 1 uses no random draws: it is each library's default for this setting, written out explicitly, with each model's own default learning rate (no common value across models). CatBoost's reference learning rate is its automatic log-loss rule evaluated for the GPU, 368,632 Train rows and 2,000 rounds, recorded as one explicit value for both devices (DR-015 §6):

  | Model | Parameter | Search distribution | Reference (trial 1) |
  |---|---|---|---|
  | LightGBM (CPU) | `learning_rate` | LogUniform[0.01, 0.3] | 0.1 |
  | | `num_leaves` | UniformInt[4, 768] | 31 |
  | | `min_data_in_leaf` | LogUniformInt[20, 2000] | 20 |
  | | `feature_fraction` | Uniform[0.5, 1.0] | 1.0 |
  | | `bagging_fraction` | Uniform[0.5, 1.0] | 1.0 |
  | | `lambda_l2` | 0.5 · {0} + 0.5 · LogUniform[0.1, 10] | 0 |
  | CatBoost (GPU) | `depth` | UniformInt[6, 10] | 6 |
  | | `learning_rate` | LogUniform[0.01, 0.3] | 0.032340 |
  | | `l2_leaf_reg` | LogUniform[1, 10] | 3 |
  | | `bagging_temperature` | Uniform[0, 1] | 1 |
  | | `leaf_estimation_iterations` | UniformInt[1, 10] | 10 |

  - LightGBM fixed: `boosting="gbdt"`, `data_sample_strategy="bagging"`, `bagging_freq=1`, `max_depth=-1`, `min_sum_hessian_in_leaf=1e-3`, `lambda_l1=0`, `min_gain_to_split=0`, `max_delta_step=0`, `path_smooth=0`, `extra_trees=false`, `feature_fraction_bynode=1.0`, `boost_from_average=true`, `max_cat_to_onehot=4`, `max_cat_threshold=32`, `cat_l2=10`, `cat_smooth=10`, `min_data_per_group=100`, `num_iterations=2000`, `early_stopping_round=50`, `early_stopping_min_delta=0`, `metric="None"` with the V1 objective as a custom evaluation function.
  - CatBoost fixed, identically on both devices ([DR-013](decisions/DR-013-training-device-policy.md) §3): `random_strength=1`, `model_size_reg=0.5`, `permutation_count=4`, `fold_permutation_block=1`, `boost_from_average=false`, `eval_metric` = the V1 objective as a user-defined metric, `metric_period=1`, `iterations=2000`, `od_wait=50`.
  - `lambda_l1` and `lambda_l2` are L1 (absolute-value) and L2 (squared-value) penalties on leaf values.
  - Before any development run, the synthetic CatBoost verification also fits depth 10 on the GPU at 2,000 iterations.
- **Neural search spaces, reference configurations and training values** ([DR-015](decisions/DR-015-common-training-rules.md) §7). Both models use REF-087's large-dataset space (B) as the authors' code samples it [REF-103]; an optional parameter (0.5 · {0} + 0.5 · a distribution) is exactly 0 with probability 0.5. The parameters are: `n_layers`, the number of residual blocks or Transformer layers; `d`, the ResNet block width (a parameter name, distinct from the relative day d); `d_hidden_factor`, the ratio of a block's hidden width to `d`; `d_embedding` and `d_token`, the lengths of the ResNet's category-embedding vectors and of FT-Transformer's token vectors (`d_token` is a multiple of the 8 attention heads); the dropout probabilities inside a ResNet block (`hidden_dropout`), on a residual branch (`residual_dropout`), on the attention weights (`attention_dropout`) and inside the feed-forward module (`ffn_dropout`); `d_ffn_factor`, the ratio of the feed-forward hidden size to `d_token`; `lr`, the learning rate; and `weight_decay`, AdamW's decoupled weight-decay coefficient. Trial 1 is, for FT-Transformer, the published default evaluated in REF-087 (Table 12); for the ResNet-style MLP, which REF-087 gives no default, an adaptation of the TabZilla default ResNet evaluated in REF-089, with its two values outside space (B) replaced (category-embedding width 192, the default FT-Transformer's token width, and weight decay 0) [I]:

  | Model | Parameter | Search distribution | Reference (trial 1) |
  |---|---|---|---|
  | ResNet-style MLP (GPU) | `n_layers` | UniformInt[1, 16] | 2 |
  | | `d` (block width) | UniformInt[64, 1024] | 128 |
  | | `d_hidden_factor` | Uniform[1, 4] | 2 |
  | | `hidden_dropout` | Uniform[0, 0.5] | 0.25 |
  | | `residual_dropout` | 0.5 · {0} + 0.5 · Uniform[0, 0.5] | 0.1 |
  | | `d_embedding` | UniformInt[64, 512] | 192 |
  | | `lr` | LogUniform[1e-5, 1e-2] | 1e-3 |
  | | `weight_decay` | 0.5 · {0} + 0.5 · LogUniform[1e-6, 1e-3] | 0 |
  | FT-Transformer (GPU) | `n_layers` | UniformInt[1, 6] | 3 |
  | | `d_token` | 8 · UniformInt[8, 64] | 192 |
  | | `attention_dropout` | Uniform[0, 0.5] | 0.2 |
  | | `ffn_dropout` | Uniform[0, 0.5] | 0.1 |
  | | `lr` | LogUniform[3e-5, 3e-4] | 1e-4 |
  | | `weight_decay` | LogUniform[1e-6, 1e-3] | 1e-5 |

  - FT-Transformer fixed: `residual_dropout=0`, `d_ffn_factor=4/3`, 8 heads, ReGLU (rectified gated linear unit), pre-normalisation, Kaiming initialisation, and no weight decay on the feature tokenizer, LayerNorm and biases [REF-087]. The ResNet-style MLP's `d_embedding` range is an adaptation of REF-087's Adult range, because space (B) has no categorical features.
  - Optimiser: AdamW (Adam with decoupled weight decay) with `betas` (0.9, 0.999), the decay coefficients of its moment estimates, `eps` 1e-8, its numerical-stability constant, AMSGrad (the variant keeping the maximum of past second moments) disabled, the single-tensor implementation and no learning-rate schedule.
  - **Effective batch: 1,024 Train rows** for both models (REF-087 Table 7, datasets above 300,000 training rows): 360 updates per epoch, the final batch of 1,016 rows kept.
  - **Micro-batch plan:** before the search, a data-free synthetic preflight on the project GPU tries 1,024, 512, 256, 128 and 64 rows for each of the 10 candidates and records the largest that leaves at least 1,024 MiB (mebibytes) free. The ResNet-style MLP runs at 1,024 directly, because its BatchNorm (batch normalisation) uses statistics of the whole batch; FT-Transformer, whose LayerNorm (layer normalisation) acts on each row, accumulates gradients over its micro-batches, with one AdamW update per effective batch. Both training and float32 evaluation must pass the preflight; a candidate that cannot meet its required workload and safety margin stops execution for an issue, and is not dropped or replaced. The plan is the same for both weightings and every refit, and is not changed after V1 results.
  - **Loss:** the weighted binary cross-entropy summed over the effective batch and divided by its row count (1,024, or 1,016 for the final batch).
  - Evaluation batch: 4,096 rows (ResNet-style MLP) and 16 (FT-Transformer), fixed and recorded, because it can also change floating-point scores.
  - **Compute estimate [D] [I]:** synthetic-probe extrapolations give about 7.6 minutes per epoch for the default FT-Transformer, 44.5 for the largest and 14 for an average candidate. Under the interpolation of DR-015 §7, the FT-Transformer search is estimated at 31–67 hours for 7–15 epochs per trial; six refits at the largest setting and 15 retained epochs would add about 60 Train-only hours. These are estimates, not bounds. The estimate for the actual candidate lists is recorded before any fit; no list is redrawn and no candidate is rejected for its cost, and an infeasible estimate pauses execution for an explicit decision before any run.
  - The neural loop implements gradient accumulation, the preflight and the shared daily Recall@2% objective; they are verified on synthetic data and covered by the synthetic acceptance evidence ([`gpu-acceptance-probe.md`](gpu-acceptance-probe.md)).
- **Early stopping within each trial:** the V1 objective is evaluated after every boosting round or epoch; training follows the family-specific stopping conditions below, or stops at the maximum; the earliest best checkpoint is retained.
  - Tree-based: patience 50 boosting rounds (stopping after 50 consecutive rounds without improvement), maximum 2,000 rounds.
  - Neural: patience 5 epochs (stopping at the sixth consecutive epoch without improvement), maximum 15 epochs.
- **V1 objective:** V1 Recall@2% under daily top-K review, unweighted, for both weightings. Each V1 relative day's 62 highest-ranked transactions count as reviewed:

  $$
  \begin{aligned}
  K_{\mathrm{V1}} &= \operatorname{round}\left(0.02 \times \frac{N_{\mathrm{V1}}}{D_{\mathrm{V1}}}\right) \\
  &= \operatorname{round}\left(0.02 \times \frac{65{,}544}{21}\right) \\
  &= 62
  \end{aligned}
  $$

  where `K_V1` = daily review count for tuning on V1, `N_V1` = number of V1 transactions and `D_V1` = number of V1 relative days. The objective is the frauds captured in the daily top-62 queues, summed over the 21 days, divided by all V1 frauds. It drives both early stopping and trial selection. `K_V1` is used for tuning only; the daily review budget remains K = 56 from V2 volume ([DR-002](decisions/DR-002-review-capacity-definition.md)).
- **Ties:** improvement is strict, and equality counts as no improvement. Equal values keep the earliest checkpoint, and equal best trials select the lowest trial number. Comparisons use exact values without rounding or tolerance; no other metric is a secondary selection criterion. Equal scores in review queues follow the R3 tie rule ([Decision Policies](#r3--daily-top-k)).
- **Implementation and records:** one shared scorer computes the V1 objective for all eight configurations. The LightGBM adapter accepts `metric="None"` with the V1 objective as a custom evaluation function, and the CatBoost adapter accepts it as a user-defined evaluation metric on its GPU learner. Both paths are verified on synthetic data: daily grouping, unweighted evaluation, the cutoff tie rule, strict improvement, patience, earliest-best checkpoint retention, and parameter recording and reloading. Each trial records its retained checkpoint, stopping round or epoch, stopping reason and runtime; a selected trial that reached its maximum is reported as such.

**Seeds.** All eight configurations use the main seed 42 for every search trial and every main result. The additional seeds **43 and 44** repeat each configuration's selected hyperparameters on Train at its retained round or epoch count, without early stopping: 16 refits, without repeating the search. One further fresh-process refit per configuration with seed 42 at the same length gives the same-seed repeats (8 refits): identical scores are expected for LightGBM and the neural models, and CatBoost's GPU differences are measured. All refits are evaluated on V2 only, under R3 at the four capacities with the fixed daily budgets (the primary measure is Recall@2% with the daily budget K). The range over seeds 42, 43 and 44 is compared descriptively with the SESOI of 2 percentage points. The refits never replace the main-seed models and never influence model selection, the H3 configuration or any other decision ([DR-013](decisions/DR-013-training-device-policy.md) §6; [DR-015](decisions/DR-015-common-training-rules.md) §4). The neural preprocessing seed is **0**, in a dedicated generator separate from the candidate sampler and model training; it never changes, so every neural fit and refit of the main study uses the same fitted preprocessing, whereas the rolling-origin supplement fits it again on each origin's training days ([DR-015](decisions/DR-015-common-training-rules.md) §4; [DR-012](decisions/DR-012-neural-network-input-representation.md)).

**Training environment.** CatBoost and both neural models (six configurations) train on the local GPU (8 gigabytes) through CUDA (NVIDIA's GPU computing platform). The two LightGBM configurations train on the local CPU (24 cores) with deterministic training, following the documented failure of the pinned CUDA learner on categorical features with more than 256 categories ([DR-013](decisions/DR-013-training-device-policy.md) §2). All configurations use the model-specific precision and device settings, scoped determinism rules and prediction backends specified in [DR-013](decisions/DR-013-training-device-policy.md); a synthetic acceptance probe confirmed the required capabilities before the development runs ([`gpu-acceptance-probe.md`](gpu-acceptance-probe.md)). The engineering probe of Issue #50 (synthetic tensors) covered the width and depth grid of the FT-Transformer space (B), with its feed-forward size factor of 4/3. Under these settings, at 431 input tokens, FT-Transformer fits at batch sizes 512 (default setting) to 128 (largest setting); at 800 input tokens the largest setting fits at batch size 64, and at 128 with gradient checkpointing. float32 with standard attention runs out of memory for the larger settings ([DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md) §4).

## Feature Policy

- **Declared raw inputs plus one fixed time-of-day pair are the primary design.** There is no feature-engineering study: the design is set before any model run and is not selected or tuned on V1 or V2 performance.
- **`TransactionID` and raw `TransactionDT` are not model features.** `TransactionID` is monotone in time, so it is a disguised timestamp [D]. Raw time cannot be extrapolated by tree models.
- **Time-of-day inputs** ([DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md) §4). Two numeric inputs place each transaction at its continuous position within its relative day, from its own `TransactionDT` value t (in seconds):

  $$
  \begin{aligned}
  \theta &= 360^\circ \times \frac{t \bmod 86400}{86400} \\
  \mathrm{tod\_sin} &= \sin\theta \\
  \mathrm{tod\_cos} &= \cos\theta
  \end{aligned}
  $$

  where:

  - `t` = `TransactionDT` of the transaction, in seconds from the undisclosed reference point;
  - `θ` = time-of-day angle: the fraction of the relative day elapsed, expressed as an angle on a full circle (0° ≤ θ < 360°);
  - `tod_sin`, `tod_cos` = the two time-of-day inputs.

  The inputs are row-local and label-free, and every model receives them as ordinary numeric inputs. θ is a position within the relative day, not the local clock time. Data preparation stops with an error if any `TransactionDT` value is missing, non-finite or not a whole number of seconds, or if its relative day lies outside 1–182; no row is dropped or imputed. The organiser column `D9`, which partly encodes the hour of the relative day and is missing on 87.3 % of development rows [D], is kept unchanged.
- **No future-looking aggregate features.** Any frequency, count or entity-level aggregate would have to be computed from Train only or with past-only windows. None is planned for the primary design.
- **No manual target encoding.** Categorical handling is native to each model family: CatBoost ordered target statistics, computed in the time order of the Train rows [REF-012] ([DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md) §5); LightGBM categorical splits [REF-011]; learned category embeddings in the neural networks ([DR-012](decisions/DR-012-neural-network-input-representation.md)).
- **No engineered entity identifiers as features.** Heuristic keys such as card + address + implied account-start day are used **only** for audit and slicing.
- **Transactions and supplied values are preserved** ([DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md) §9). Data preparation applies only the declared parsing and input-representation conversions: no transaction is discarded by content-based filtering, de-duplication or outlier-row removal, and no numeric model-input value is altered as corrective cleaning (no clipping, winsorising, numeric imputation or scaling in the canonical frame). Repeated input vectors are counted descriptively and kept ([data-preparation record](data-preparation.md)).
- **Feature declaration** ([DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md); machine-readable in [`config/features.yaml`](../config/features.yaml); column-by-column reference in the [feature catalog](feature-catalog.md)). Of the 434 joined columns:
  - 1 is the target (`isFraud`) and 2 are excluded (`TransactionID`, `TransactionDT`);
  - **49 are categorical**, exactly the organiser's published list [REF-047]: `ProductCD`, `card1`–`card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `M1`–`M9`, `DeviceType`, `DeviceInfo`, `id_12`–`id_38`. This includes the numeric-coded fields `card1`–`card3`, `card5`, `addr1`, `addr2` and 12 `id_*` codes;
  - **382 are numeric**: every other column (`TransactionAmt`, `dist1`, `dist2`, `C1`–`C14`, `D1`–`D15`, the `V1`–`V339` feature columns, `id_01`–`id_11`);
  - the two time-of-day inputs are added as numeric inputs, giving 384 numeric and 433 model inputs in total.
- **Model input parity.** All four models receive the same 433 inputs, built from one canonical frame:
  - categorical values are written as text labels, with missing values labelled `"__MISSING__"`;
  - **tree-based models:** numeric values stay `float64` with native missing-value handling; LightGBM receives pandas categoricals whose categories are fixed from Train only, and CatBoost receives the text labels directly [REF-076, REF-077]; no imputation, scaling, rare-level grouping or high-cardinality exclusion is applied;
  - **neural models ([DR-012](decisions/DR-012-neural-network-input-representation.md); identical for both, attached to input roles):** per-input category vocabularies fitted on Train with `"__MISSING__"` reserved in every vocabulary, values unseen in Train mapped to it, and category dropout (probability 0.01, training only) so that its embedding is trained; numeric values, including the time-of-day inputs, quantile-transformed to a normal distribution fitted on their non-missing Train values, with Gaussian noise added during fitting only, scaled by the reference implementation's formula [REF-087, REF-103]; missing values set to 0 after the transformation, and one missingness indicator per numeric input with missing Train values (757 inputs in total; FT-Transformer: 757 feature tokens plus the [CLS] classification token). The preprocessing is fitted once on Train, with its noise drawn from the preprocessing seed 0 in a fixed input and row order, and one fitted artifact is reused unchanged by all four neural configurations, every candidate and every refit of the main study (the rolling-origin supplement fits it again on each origin's training days).

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

**Selection rule** ([DR-009](decisions/DR-009-overflow-handling-for-threshold-policies.md#r1-threshold-selection), "R1 threshold selection"; Issue #115). For each selected main-seed configuration, R1 flags every transaction whose raw score is at least the threshold τ_R1. The candidates for τ_R1 are every distinct V2 score, including the smallest (so an all-flag optimum is allowed); transactions with equal scores are never separated. For a candidate a, F1 is computed from the pooled, unweighted V2 counts:

$$
F_1(a) = \frac{2\,\mathrm{TP}_a}{P + A_a}
$$

where:

- `TP_a` = true positives (frauds with score ≥ a);
- `A_a` = alerts (transactions with score ≥ a);
- `P` = frauds in the selection set; P + A_a = 2TP_a + FP_a + FN_a, the usual denominator, with `FP_a` and `FN_a` the false positives and false negatives at a.

For two candidate thresholds a and b, TP_b and A_b are defined as TP_a and A_a are. Candidates are compared exactly through the integer cross-products, computed with arithmetic that cannot overflow:

$$
\mathrm{TP}_a\,(P + A_b) \quad \text{and} \quad \mathrm{TP}_b\,(P + A_a)
$$

The larger product identifies the larger F1, and equal products identify an exact tie; no rounding or tolerance is used. Among equal maxima the **highest** candidate threshold (the fewest alerts) is selected [I]. The selection set must be non-empty, contain at least one fraud and have finite scores; otherwise selection stops with a recorded error. The selected threshold is frozen for later evaluation; its full-V2 F1 is a selection-set result.

### R2 — Capacity-rate threshold

The score threshold τ_c is chosen on V2 so that a fraction c of V2 transactions is flagged (τ₂ is τ_c at c = 2 %). It uses V2 scores and needs no labels.

Purpose: object of RQ2/H3. Is a static validation-selected operating point transferable in time? Evaluated uncapped: its daily alert demand is compared with K_c, never truncated to it (DR-009).

### R3 — Daily top-K

Review the K highest-scoring transactions within each relative day.

Purpose: ranking-quality benchmark at full capacity, used for H1/H2. Note that it is an idealised end-of-day batch policy, not a streaming policy [REF-033].

**Tie rule.** Within each relative day, transactions are ranked by raw score, descending, compared exactly without rounding or grouping near-equal scores. Exactly equal scores are ordered by the SHA-256 (Secure Hash Algorithm, 256-bit) hexadecimal digest of the decimal `TransactionID` string encoded in UTF-8 (Unicode Transformation Format, 8-bit), ascending; identical digests are resolved by `TransactionID` ascending. The rule applies to the V1 objective, R3 and R4 for every configuration, seed and capacity. It is fixed, reproducible and label-free, without the time bias of ordering by `TransactionID` alone. It is not used to split ties within whole-ranking metrics or as a model input. The number of days with ties crossing the cutoff, and the sizes of those tied groups, are reported ([DR-015](decisions/DR-015-common-training-rules.md) §5).

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
- **Relevant to R4 only.** R0–R3 rank and threshold on raw model scores, so calibration does not affect them. The positive-slope Platt sigmoid is strictly increasing in its input, but the input clipping can merge extreme raw scores; isotonic regression is monotone but can merge scores into ties, which is why it serves only as a secondary check for R4 [I; REF-015, REF-017]. R4 therefore keeps R3's raw-score order and tie rule.
- **Primary method:** Platt scaling [REF-016]. **Secondary check:** isotonic regression [REF-017].
- **Scope.** One Platt and one isotonic calibrator per configuration (eight of each), fitted on the V2 scores of the configuration's selected main-seed model and applied unchanged to Test. The seed refits and the CatBoost CPU refit are evaluated under R3 only and are not calibrated. R4's queue stays R3's raw-score order.
- **Platt input and map.** Platt scaling is applied to the float64 log-odds of the raw fraud score s:

  $$
  \begin{aligned}
  s_c &= \min\left(\max\left(s,\ \epsilon\right),\ 1 - \epsilon\right), \qquad \epsilon = 10^{-15} \\
  z &= \ln s_c - \ln(1 - s_c) \\
  \hat p &= \frac{1}{1 + \exp\left(-(\kappa_0 + \kappa_1 z)\right)}, \qquad \kappa_1 > 0
  \end{aligned}
  $$

  where:

  - `s`, `s_c` = the raw fraud score s_i of a transaction and its clipped value;
  - `ε` = the clipping constant (10⁻¹⁵);
  - `z` = the log-odds of the clipped score (the second logarithm computed as `log1p(-s_c)`);
  - `p̂` = the calibrated fraud probability;
  - `κ_0`, `κ_1` = the calibrator's intercept and slope (distinct from the significance level α).

  On log-odds, the map's family contains the identity on the unclipped interior, which logistic calibration on the score itself does not [REF-132]; its ability to represent an ideal class-weighting odds correction (an intercept shift on the log-odds scale) motivates the choice [I], and exact correction of the fitted weighted models is not assumed. The number of clipped scores is recorded. Clipping can merge extreme scores, so the composite map is non-decreasing.
- **Platt fit.** Unweighted binary cross-entropy is minimised on all V2 rows pooled over days 127–154, against Platt's smoothed targets t₊ = (N₊ + 1)/(N₊ + 2) for frauds and t₋ = 1/(N₋ + 2) for legitimate rows, where N₊ and N₋ are the fraud and legitimate counts of the fitting rows [REF-016; REF-133]; no coefficient penalty is added. This is a smoothed-target fit, not ordinary maximum likelihood against the original labels; the diagnostics use the original labels. Both classes and at least two distinct clipped log-odds values are required. Non-identifiability, non-convergence, non-finite parameters or a non-positive fitted κ_1 stops the analysis for an issue, with no automatic substitute and no forcing of the slope to a positive boundary. The solver, its initialisation, tolerance and iteration limit are fixed, tested and recorded in the implementation (Issue #96) before any calibration run.
- **Isotonic check.** Fitted on the raw float64 scores and the original labels, with unit weight per V2 transaction, increasing, bounded to [0, 1], linear interpolation between the fitted knots and endpoint clipping outside the fitting range; identical scores share their mean label and keep their combined weight [REF-017; REF-134].
- **Prediction outside the V2 range.** Platt applies its frozen map to every score, subject only to the input clipping; isotonic returns its endpoint values. Out-of-range counts are recorded. For log loss and the diagnostic logit(p̂), a metric-only copy of p̂ is clipped to [ε, 1 − ε] with ε = 10⁻¹⁵, the clipping counts are reported and log loss is reported as ε-clipped log loss; R4 and the Brier score use the original calibrated probabilities.
- **Metrics:** Brier score, log loss, calibration intercept and slope, reliability diagram [REF-018]. Reported on V2 (in-sample for the calibrator) and on Test.
  - **Calibration intercept:** unweighted logistic regression of the original labels with logit(p̂) as an offset and only an intercept estimated (target 0). **Calibration slope:** unweighted logistic regression of the labels on logit(p̂) with a free intercept (target 1) [REF-018]. These diagnostic fits are separate from the calibrator fit; their slopes are not constrained, and a non-estimable fit is reported as such.
  - **Reliability diagram:** descriptive, with 20 equal-count bins formed from p̂ without labels (identical probabilities kept together, so fewer bins are possible), reporting each bin's probability range, mean prediction, observed fraud rate, transaction count and fraud count. The diagram is descriptive only: Wilson or other binomial confidence-interval bars are not displayed, because frauds cluster across days ([DR-006](decisions/DR-006-bootstrap-inference-strategy.md)). The exact bin-edge algorithm, including the assignment of tied values at an edge, is fixed and tested in the implementation (Issue #96). The decision-region check is the declined-set diagnostic of [DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md) §8.
- **Records:** the model identity, the fitting rows and their counts, the clipping settings and counts, the parameters or isotonic knots, the solver settings and convergence, the software versions and the artifact hashes; predictions are verified after reloading.

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

**Uncertainty (descriptive).** 95 % percentile intervals at grid ratios come from a day-level moving-block bootstrap over Test days (block length ℓ = 3 days, B (number of bootstrap replicates) = 10,000, R3 and R4 paired). A DR-006 pigeonhole check is applied to frauds forgone. ρ_sat and ρ_BE are point estimates. The intervals are conditional on the fitted models and the frozen V2 calibrators: calibrators, R4 eligibility and within-day queues stay fixed while Test-day blocks are resampled, so the intervals exclude uncertainty from model development, model selection and calibrator estimation, and do not quantify future calibration drift ([DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md) §7).

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

- Recall@1%, Recall@5%, Recall@10%;
- false negatives at capacity;
- frauds per 1,000 reviews;
- lift over random review;
- utilisation of the capped policies R3 and R4; alert demand, overflow, unused capacity and load ratio of the threshold policies R0–R2 (see RQ2 and [DR-009](decisions/DR-009-overflow-handling-for-threshold-policies.md));
- Average Precision / PR-AUC (Precision–Recall Area Under the Curve) [REF-010] and ROC-AUC (Receiver Operating Characteristic Area Under the Curve);
- precision, recall and F1 of the threshold policies R0–R2 at their thresholds, uncapped (descriptive);
- amount-weighted recall;
- scenario cost;
- calibration metrics for R4;
- **computational cost** (descriptive, every configuration): training time (one fit including early stopping, and the total hyperparameter search), scoring time for V1, V2 and Test, batch throughput (transactions scored per second), single-transaction latency (median, 95th and 99th percentiles over a fixed sample of rows scored one at a time, after warm-up runs) and peak GPU and host memory. Every timing is taken on the project machine under its working conditions at the time (other local workloads may run, and the conditions are recorded as context), with GPU work synchronised before each clock reading, and records its device. Tree-based models score on CPU and neural models on the GPU ([DR-013](decisions/DR-013-training-device-policy.md) §5); no neural model is timed on the CPU, so the timings are not a same-device comparison. The complete timing procedure is fixed before the development runs ([Computational-cost procedure](#computational-cost-procedure-development); [DR-015](decisions/DR-015-common-training-rules.md) §9). Test scoring time is measured only during the single authorised Test scoring pass after the protocol freeze, without consulting Test labels.

### Computational-cost procedure (development)

The computational-cost outcome above is measured on the development partitions under the following procedure, fixed before any development run ([DR-015](decisions/DR-015-common-training-rules.md) §9; Issue #121). The timings are descriptive and never enter model, threshold or policy selection. The values are the `timing` block of `config/training.yaml`.

- **Devices.** The tree-based models are trained on their study devices (LightGBM on the CPU, CatBoost on the GPU) and score on the CPU ([DR-013](decisions/DR-013-training-device-policy.md) §5); the ResNet-style MLP and FT-Transformer train and score on the GPU only. No neural model is timed on the CPU. The timings of the tree-based and the neural models are therefore **not a same-device comparison**: each is taken on its study device, and a difference reflects model, software, input representation, batching and hardware together.
- **Fit.** The timed fit is the **selected main-seed trial's own elapsed time during the search**, including early stopping and the discarded patience rounds or epochs; no extra fit is made for timing. It runs from the construction of the fit-specific data containers to a fitted model that is ready to predict with its device work complete. Process, library and CUDA start-up, the shared preprocessing and model persistence are recorded beside it, not inside it. The completed length and the retained length of the fit are both recorded.
- **Search.** The search time of a configuration is the elapsed time of its sequential ten-trial loop (trial set-up, fit, V1 objective, recording and selection). Scoring benchmarks and the descriptive refits are outside it. An interrupted search is recorded as interrupted, with its completed and declared trial counts, and never as a completed ten-trial measurement; a search that was not interrupted has completed every declared trial.
- **Scoring and throughput.** The selected main-seed model scores the complete V1 and V2 partitions and returns probabilities (not logits). Two boundaries are reported separately: **model-only** (prepared native inputs to scores on the prediction device) and **end-to-end** (conversion from the in-memory features, application of the already fitted preprocessing artifact, native containers, host–device transfers, prediction and scores returned on the host). Disk loading, model loading, preprocessing fitting, calibration, ranking and metrics are outside both. The neural evaluation batches are those of [DR-015](decisions/DR-015-common-training-rules.md) §7 (4,096 rows for the ResNet-style MLP, 16 for FT-Transformer, including the final short batch); the tree-based models score each partition in one call. Warm-up is untimed: **one full prediction per boundary** for the tree-based models and **three batch calls** (the normal batch shape and the final short batch) for the neural models. Each partition and boundary is timed over **3 timed full passes**; the median, every pass and their range are stored, slow passes are kept, and throughput is the rows of the partition divided by the median seconds.
- **Latency.** Single-transaction latency is the time of one batch-of-one scoring call, at both boundaries, on **10,000 distinct V1 rows**: those with the smallest SHA-256 digests of the decimal `TransactionID` written in UTF-8 (identical digests by ascending identifier), in that fixed order, with the SHA-256 identity of the ordered rows stored. Neither labels nor scores enter the selection. Each block begins with **100 untimed warm-up calls**, and the rows are passed **3 rounds** per boundary; for each round the median, 95th (p95) and 99th (p99) percentile are computed with `numpy.percentile` using the `linear` method. p99 is a descriptive empirical estimate with limited tail support, and the measure is serial scoring latency (one call at a time): queueing and request arrival are not modelled. The tree-based models use their production prediction thread counts (24, `config/models.yaml`); there is no thread sweep.
- **Threads and affinity.** The effective CPU affinity is recorded with every CPU timing and is not changed; CPU identifiers inside the WSL2 (Windows Subsystem for Linux 2) guest are not assumed to correspond to physical performance cores.
- **GPU synchronisation.** Neural timings synchronise the explicit CUDA device before the clock starts and after the work completes, around each full pass and around each latency call. CPU timings use no synchronisation.
- **Session conditions.** The timings are taken under the machine's working conditions at the time and are valid descriptive measurements: no condition is required, and other local workloads may run during a timing. The conditions are recorded as context with the run: a **60-second settling interval** before each standalone scoring block, the one-minute load average, the visible and allowed CPUs, the kernel, the WSL2 memory and swap limits, and GPU temperature and clock data where the system exposes them. A timing is read together with its recorded context and reflects the machine's conditions as well as the model.
- **GPU memory.** The absolute sampled whole-device maximum, the idle baseline and the increase above the baseline are stored separately and called a **sampled maximum**, not a true peak. The neural models also record the allocator's maximum allocated and reserved bytes after a reset at the start of the phase. The libraries of the tree-based models are invisible to those counters.
- **Host memory.** The peak resident memory of each phase is the process high-water mark (`VmHWM`) read after resetting it by writing `5` to `/proc/self/clear_refs` (which resets it to the current resident size, not to zero; the check resets first, so that an earlier, higher mark cannot hide it). Whether the reset works on the project machine is checked with a synthetic allocation in the first run; where it does not, a sampler thread reads the resident size **every 100 ms** (the `timing` block) and the record names the method and the interval; a failed sampler or one that does not stop invalidates the phase's measurement instead of returning a peak. The scope (parent process, resident inputs, models, and whether warm-up lies inside the interval) is stated by the caller with each value.
- **Test period.** Test scoring time is obtained only by instrumenting the single authorised Test scoring pass after the protocol freeze; no Test pass is repeated for timing.
- **Records.** One versioned `timing` record per measurement; a scoring record covers the whole partition and its throughput is the rows divided by the median seconds; an FT-Transformer timing names the efficient attention backend ([DR-013](decisions/DR-013-training-device-policy.md) §4); the free `settings` map holds the procedure version, the durations of every pass, the number of distinct rows and of timed calls, the percentile method, the memory method, baseline and maxima, and the session conditions.

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

**Supportive analysis: training device (CatBoost).** Each of the two CatBoost configurations is refitted once on CPU with its GPU-selected hyperparameters and GPU-fitted tree count; LightGBM trains on the CPU only, so it has no CPU–GPU pair. The CPU − GPU difference in Recall@2% under R3 on V2 is reported with the paired bootstrap of [DR-006](decisions/DR-006-bootstrap-inference-strategy.md); other metrics are reported as point differences, alongside same-seed and different-seed variation. The analysis estimates the difference between the two specified training procedures, not a hardware effect alone; it is descriptive, outside H1–H3 and the Holm family, and cannot change any decision ([DR-013](decisions/DR-013-training-device-policy.md) §7).

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
- after the one-time Parquet build, development work reads only the verified processed files; the raw files are read again only by the confirmatory Test evaluation ([data access rule](dataset.md#data-access-after-the-parquet-build));
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

**Ethics and governance status (Issues #31 and #63).** The supervisor has given written approval of the project topic. **In written guidance of 3 October 2026, the supervisor confirmed that no ethics application is required, because the project uses no personal data or personally identifiable information (PII), and that there are no restrictions on the use of the data, the reporting of results, the publication of code or the storage of derived outputs** (Issue #31); the supervisor's comment on the dataset itself is pending. This is the anonymised secondary-data route of the University guidance [REF-051]. The criminal-subject-matter provision [REF-050 §1.3.4] is addressed through the documented ethical considerations below ([criteria checklist](greenwich-criteria-checklist.md#4-ethics)). **No formal ethics approval, ethics exemption or ethics identifier is claimed; none is required under that guidance.** The competition data-use terms have been verified [REF-049]: the Rules permit non-commercial use for academic research and education (Section 7A) and prohibit making the Competition Data available to any party not participating in the competition (Section 7B). As a stricter project policy, raw data and row-level derived data stay local, out of Git, and are not shared with anyone, including supervisors or examiners; aggregate results may be published ([`dataset.md`](dataset.md#data-use-terms)).

**Ethical considerations.** Following the University of Greenwich guidance [REF-050, REF-051], the ethics assessment considers:
- the status and anonymisation of the secondary data;
- permitted use under the competition rules ([`dataset.md`](dataset.md#data-use-terms));
- privacy and re-identification risk;
- false-positive and false-negative harms;
- bias and fairness;
- human oversight;
- responsible interpretation of model outputs.

Artificial intelligence (AI) assistance used in this project will be declared at submission in an AI Reflection (AIR), which replaces the Declaration of AI use from 2026/27 [REF-109, REF-111]; the AI-use tier is Tier 2 ("Supported by AI") as the working basis, to be confirmed against the COMP1885 assessment brief ([criteria checklist](greenwich-criteria-checklist.md#5-use-of-ai)).

## Design items before protocol freeze

**Fixed design elements** (recorded in Decision Records and accepted in line with the University of Greenwich criteria; supervisor feedback welcome, not a precondition):

| Element | Specification | Record |
|---|---|---|
| Temporal split | Train 1–105, V1 106–126, V2 127–154, Test 155–182 (sealed); no primary gap | [DR-001](decisions/DR-001-temporal-split-strategy.md) |
| Review capacity | 1 / 2 (primary) / 5 / 10 %; fixed daily budget K = 28 / 56 / 139 / 278 from V2 volume | [DR-002](decisions/DR-002-review-capacity-definition.md) |
| RQ2 and H3 | Alert-rate transfer equivalence, Δ = 0.004 (±20 % relative); TOST with a day-level moving-block bootstrap, decided by its Holm-adjusted p-value | [DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md) |
| SESOI, H1 and H2 | SESOI = 2 pp of Recall@2%; H1 family-effect difference test with practical interpretation; H2 intersection-union TOST over four models; Holm over {H1, H2, H3} | [DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md) |
| H1/H2 uncertainty | Two-way pigeonhole bootstrap (relative day × fine pseudo-entity); sensitivity S1 (Owen-corrected variance) and S2 (day blocks, ℓ = 3) | [DR-006](decisions/DR-006-bootstrap-inference-strategy.md) |
| RQ3 cost analysis | Review + missed-fraud cost model (C_FP as a sensitivity relabelling); ρ ∈ {2, …, 1000} with exact curves; saturation and break-even ratios per capacity and configuration | [DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md) |
| Model inputs | 49 categorical, 382 numeric, 2 excluded, plus 2 time-of-day inputs (`tod_sin`, `tod_cos`): 433 inputs; canonical labels with `"__MISSING__"`; tree-based representation fixed | [DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md) |
| Overflow rule | Report-only: R0–R2 evaluated uncapped; demand, overflow, unused capacity and load ratio against K | [DR-009](decisions/DR-009-overflow-handling-for-threshold-policies.md) |
| H3 configuration | Highest pooled V2 Recall@2% under R3 among eight configurations; committed before Test access | [DR-010](decisions/DR-010-h3-confirmatory-configuration.md) |
| Model set | LightGBM, CatBoost, ResNet-style MLP, FT-Transformer × unweighted/balanced; common-procedure fairness | [DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md) |
| Common training rules | Random search, 10 trials per configuration (trial 1 = reference configuration; one candidate list per model for both weightings); early stopping: trees 50 / 2,000 rounds, neural 5 / 15 epochs; V1 objective V1 Recall@2% (daily top-62, unweighted); seeds 42 (search and main results), 43 and 44 (descriptive refits); exact tie rules, R3 cutoff ties by SHA-256 of `TransactionID`; tree-based search spaces (CatBoost: REF-087 space (B); LightGBM: REF-129; learning rate LogUniform[0.01, 0.3] for both) with library-default reference configurations; neural search spaces of REF-087 space (B) with the published FT-Transformer default and an adapted TabZilla ResNet default as reference configurations, effective batch 1,024 with a pre-search micro-batch plan, row-count loss normalisation | [DR-015](decisions/DR-015-common-training-rules.md) |
| Data-use terms | Non-commercial academic use (Rules Section 7A); no redistribution (Section 7B); raw and row-level data stay local | [`dataset.md`](dataset.md#data-use-terms) |
| Ethics and governance | Topic approval received; no ethics application required (supervisor's written guidance, 3 October 2026; Issue #31); no formal ethics approval, exemption or identifier claimed | [Ethics](#ethics) |

**Items to resolve before the freeze:**

| # | Item | Notes |
|---|---|---|
| 1 | University of Greenwich criteria | Resolved (Issue #63): requirements mapped in the [criteria checklist](greenwich-criteria-checklist.md); supervisor feedback on the design (DR-001–DR-014) welcome, not a precondition (Issue #56) |
| 2 | Neural-network input representation | Resolved: representation accepted on 2026-10-17 (Issue #107); preprocessing seed and the draw and shared-artifact rules fixed in Issue #112 ([DR-012](decisions/DR-012-neural-network-input-representation.md); [DR-015](decisions/DR-015-common-training-rules.md) §4). The shared artifact is implemented (Issue #149) and the `config/features.yaml` status is `accepted` |
| 3 | Training rules | Common rules resolved (Issue #105): search method and budget, early stopping, V1 objective, seeds and tie rules ([DR-015](decisions/DR-015-common-training-rules.md)); tree-based search spaces and reference configurations resolved (Issue #106; DR-015 §6). neural search spaces, reference configurations and training values resolved (Issue #107; DR-015 §7); R4 calibration details resolved (Issue #91; [Calibration](#calibration); [DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md) §8) |
| 4 | Feature-engineering policy | Resolved (Issue #64): declared inputs plus the two time-of-day inputs; no other engineered feature ([DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md) §4) |
| 5 | Logistic-regression reference baseline | Resolved (Issue #115): one unweighted, tuned L2-regularised logistic regression on the DR-012 representation, reported descriptively under R3 ([Models](#models); [DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md) §5) |
| 6 | University of Greenwich AI-use tier and AI Reflection (AIR) | Resolved as a working basis (Issue #63): Tier 2, AIR at submission [REF-109, REF-111]; tier to be confirmed against the COMP1885 assessment brief |
| 7 | Research-methodology framing | Resolved (Issue #85): the ten stages of the Foundational Methodology for Data Science published by International Business Machines Corporation (IBM) [REF-085], adapted to the confirmatory design ([DR-014](decisions/DR-014-research-methodology-framing.md)) |
| 8 | Dataset manifest | Resolved (Issue #62): SHA-256 (Secure Hash Algorithm, 256-bit) values and partition hashes reproduced by the data-preparation code and recorded in `data/datapackage.json` ([dataset manifest](dataset.md#dataset-manifest)) |
| 9 | Rolling-origin supplement design | Resolved (Issue #159): three origins 14 days apart, refits on the days before a 21-day gap with origin-specific preprocessing and class weights, the R2 threshold set on the next 28 days and transferred to the 28 days after, K from the threshold window; a retrospective sensitivity analysis ([Experimental Split Structure](#experimental-split-structure); [DR-001](decisions/DR-001-temporal-split-strategy.md)). The execution budget is decided after the H3 configuration is known |

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
- [DR-012 — Neural-network input representation](decisions/DR-012-neural-network-input-representation.md)
- [DR-013 — Training-device policy](decisions/DR-013-training-device-policy.md)
- [DR-014 — Research-methodology framing](decisions/DR-014-research-methodology-framing.md)
- [DR-015 — Common training rules](decisions/DR-015-common-training-rules.md)

## References

Full entries and verification status are in [`docs/references.md`](references.md).

REF-001, REF-002, REF-003, REF-004, REF-005, REF-006, REF-008, REF-009, REF-010, REF-011, REF-012, REF-013, REF-014, REF-015, REF-016, REF-017, REF-018, REF-019, REF-020, REF-021, REF-022, REF-024, REF-025, REF-026, REF-027, REF-028, REF-029, REF-030, REF-033, REF-034, REF-035, REF-037, REF-039, REF-040, REF-043, REF-044, REF-046, REF-047, REF-048\*, REF-049, REF-050, REF-051, REF-061, REF-062, REF-063, REF-064, REF-065, REF-068, REF-070, REF-071, REF-072, REF-073, REF-074, REF-076, REF-077, REF-078, REF-081\*, REF-082, REF-083, REF-084, REF-085, REF-087, REF-088, REF-089, REF-090, REF-103, REF-109, REF-111.

\* Partially verified.

## Symbols and notation

Every symbol used in this protocol, in alphabetical order. Each is also defined where it first appears.

| Symbol / abbreviation | Meaning |
|---|---|
| [D], [I] | Evidence tags: dataset observation; inference |
| A_d, A_d,j | Alerts generated by a threshold policy (j ∈ {R0, R1, R2}) on day d: threshold-generated review demand, not reviews performed |
| d_o | Origin day of the rolling-origin supplement: the first day of its threshold window (71, 85 or 99) |
| O_d,j,c | Overflow: max(A_d,j − K_c, 0) |
| p95, p99 | 95th and 99th percentile of the single-transaction latencies |
| U_d,j,c | Unused capacity: max(K_c − A_d,j, 0) |
| L_d,j,c | Load ratio: A_d,j / K_c |
| j | A threshold policy: R0, R1 or R2 |
| K_c | Fixed daily review budget for capacity scenario c (K when the scenario is clear) |
| `VmHWM`, `/proc/self/clear_refs` | The Linux process's peak resident memory ("high-water mark") and the file that resets it |
| WSL2 | Windows Subsystem for Linux 2: the virtualised Linux environment of the project machine |
| α | Significance level (family-wise 0.05) |
| p_H1, p_H2, p_H3 | p-values of H1, H2 and H3 (H2: maximum of eight one-sided p-values; H3: larger of two) |
| p_(k), p̃_(k) | k-th smallest of p_H1, p_H2, p_H3; its Holm-adjusted p-value |
| i, k | Rank positions (1–3) of the ordered p-values in Holm's procedure |
| # { · } | Number of bootstrap replicates satisfying a condition |
| bf16 | bfloat16: 16-bit floating-point format used for mixed-precision neural-network training |
| float32 | 32-bit floating-point format used for prediction |
| arg max | The argument (here, configuration) at which a quantity is largest |
| AI, AIR | Artificial intelligence; AI Reflection (University of Greenwich statement of AI use, replacing the Declaration of AI use from 2026/27) |
| COMP1885 | University of Greenwich Individual Project module for which this research is conducted |
| Tier 2 | University of Greenwich AI-use tier "Supported by AI": AI may support planning, clarification and feedback; analysis and final work remain the student's |
| B | Number of bootstrap replicates (10,000) |
| c | Review-capacity share (1 %, 2 %, 5 % or 10 %) |
| C_FP | Extra false-alert/customer-friction cost of reviewing a legitimate transaction (sensitivity only) |
| C_miss | Cost of missing (not reviewing) a fraud |
| C_R | Cost of reviewing one transaction, charged on every review |
| Cat, LGB, RES, FTT | CatBoost, LightGBM, ResNet-style MLP, FT-Transformer |
| CPU, GPU | Central and graphics processing units |
| CUDA | NVIDIA's GPU computing platform |
| [CLS] | Classification token that FT-Transformer adds to its feature tokens |
| CI | Confidence interval |
| d | A relative day |
| `D9` | Organiser time-delta column; equals the hour of the relative day ÷ 24 on most rows where present |
| Δ | H3 equivalence tolerance on r (0.004 = 20 % of the 2 % target) |
| δ_F, δ*_F | Model-family effect: neural mean minus tree-based mean of Recall@2% (H1 estimand); its value in one bootstrap replicate |
| δ_W,m | Weighting effect of model m (H2 estimands) |
| ΔJ(ρ) | Scenario-cost difference J_R4 − J_R3 (negative = R4 cheaper) |
| `__MISSING__` | Explicit label given to a missing categorical value (all models) |
| DR | Decision Record |
| IEEE-CIS | IEEE Computational Intelligence Society |
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
| IBM | International Business Machines Corporation, publisher of the methodology framework of REF-085 |
| J_π(ρ) | Scenario cost of policy π, in units of one review cost |
| K | Fixed daily review budget (28 / 56 / 139 / 278 reviews per relative day) |
| `K_V1`, `N_V1`, `D_V1` | Daily review count for tuning on V1 (62); number of V1 transactions (65,544); number of V1 relative days (21) |
| ℓ | Block length of a moving-block bootstrap, in days |
| n_d | Number of transactions on day d with score ≥ τ₂ (for confirmatory H3, the H3 configuration's score and τ₂) |
| N_window | Number of transactions of a threshold window (rolling-origin supplement) |
| N_d | Total number of transactions on day d (volume; not to be confused with N_π) |
| n_neg, n_pos | Numbers of legitimate and fraud transactions in Train |
| N_π | Number of reviews made by policy π |
| p* | Cost-derived probability threshold used by R4 |
| p̂_i | Calibrated (Platt-scaled) fraud probability of transaction i |
| s, s_c, ε, z | Raw fraud score of a transaction; its value clipped to [ε, 1 − ε]; the clipping constant 10⁻¹⁵; the log-odds of the clipped score (Calibration) |
| κ_0, κ_1 | Intercept and slope of the Platt calibrator on the log-odds scale |
| t₊, t₋, N₊, N₋ | Platt's smoothed targets for frauds and legitimate rows; the fraud and legitimate counts of the calibrator's fitting rows |
| logit | Log-odds: logit(p) = ln(p / (1 − p)) |
| φ | False-alert-to-review cost ratio, C_FP / C_R |
| π | A decision policy (R3 or R4) in J_π and N_π |
| pp | Percentage points |
| PII | Personally identifiable information |
| Parquet | Column-oriented file format of the local processed development files ([`dataset.md`](dataset.md#data-access-after-the-parquet-build)) |
| PCG64 | NumPy bit generator fixed for hyperparameter sampling |
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
| UTF-8 | Unicode Transformation Format, 8-bit |
| sin, cos | Sine and cosine functions |
| SMOTE | Synthetic Minority Over-sampling Technique |
| sup | Supremum: the largest value (least upper bound) of a set |
| Σ_d | Sum over days |
| t | `TransactionDT` value of a transaction, in seconds |
| `t mod 86400` | Seconds elapsed since the start of the transaction's relative day |
| θ | Time-of-day angle: fraction of the relative day elapsed × 360° |
| `tod_sin`, `tod_cos` | Time-of-day inputs: sine and cosine of θ |
| τ₂, τ_c | V2-selected score threshold flagging 2 % (a fraction c) of V2 transactions |
| τ_R1 | R1's V2-selected threshold |
| a, b | Candidate thresholds in the R1 comparison |
| F_1(a) | Pooled, unweighted F1 at candidate threshold a |
| TP_a, TP_b | True-positive counts at thresholds a and b |
| A_a, A_b | Alert counts at thresholds a and b |
| P | Number of frauds in the R1 selection set |
| FP_a, FN_a | False-positive and false-negative counts at threshold a |
| C | Inverse regularisation strength of the logistic-regression reference (distinct from the costs C_R, C_miss, C_FP) |
| CSR | Compressed sparse row: the sparse-matrix format of the logistic-regression inputs |
| `l1_ratio`, `lbfgs`, `tol`, `max_iter`, `fit_intercept`, `class_weight`, `warm_start` | scikit-learn logistic-regression settings: elastic-net mixing (0 = L2 penalty); the quasi-Newton solver; the convergence tolerance; the iteration limit; whether an intercept is fitted; class weighting (None = none); whether a fit starts from the previous solution |
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
| Uniform, LogUniform, UniformInt, LogUniformInt | Search distributions: continuous uniform; log-uniform; uniform on the integers; log-uniform rounded down to integers (DR-015 §6) |
| 0.5 · {0} + 0.5 · LogUniform | Mixture: exactly 0 with probability 0.5, otherwise log-uniform |
| space (B) | The large-dataset search space of REF-087 |
| `learning_rate`, `num_leaves`, `min_data_in_leaf`, `min_sum_hessian_in_leaf`, `feature_fraction`, `bagging_fraction`, `bagging_freq`, `lambda_l1`, `lambda_l2` | LightGBM: shrinkage per tree; maximum leaves per tree; minimum rows per leaf; minimum Hessian sum per leaf; share of features per tree; share of rows per bagging draw; bagging interval in rounds; L1 and L2 leaf-value penalties |
| `depth`, `l2_leaf_reg`, `bagging_temperature`, `leaf_estimation_iterations`, `random_strength` | CatBoost: symmetric-tree depth; L2 leaf-value penalty; Bayesian-bootstrap weight intensity; Newton steps per leaf-value estimate; split-score noise scale |
| `eval_metric`, `metric_period`, `iterations`, `od_wait`; `metric`, `num_iterations`, `early_stopping_round` | CatBoost's evaluation metric, evaluation interval, maximum trees and early-stopping wait; LightGBM's metric, maximum trees and early-stopping rounds |
| L1, L2 | Absolute-value and squared-value penalties; applied to tree leaf values and, for L2, logistic-regression coefficients |
| `boosting`, `data_sample_strategy`, `max_depth`, `min_gain_to_split`, `max_delta_step`, `path_smooth`, `extra_trees`, `feature_fraction_bynode`, `boost_from_average`, `early_stopping_min_delta` | LightGBM: boosting algorithm; row-sampling strategy; depth limit (−1 = none); minimum split gain; leaf-output step limit; leaf-value smoothing towards the parent; randomised split thresholds; share of features per node; start from the average label; minimum improvement counted by early stopping |
| `max_cat_to_onehot`, `max_cat_threshold`, `cat_l2`, `cat_smooth`, `min_data_per_group` | LightGBM categorical controls: largest category count split one-vs-rest; maximum categories on one side of a split; L2 penalty in categorical splits; smoothing of category statistics; minimum rows per category group |
| `model_size_reg`, `permutation_count`, `fold_permutation_block`, `boost_from_average` | CatBoost: model-size regularisation coefficient; number of permutations; permutation block size; start from the average label |
| `n_layers`, `d`, `d_hidden_factor`, `hidden_dropout`, `residual_dropout`, `d_embedding` | ResNet-style MLP: number of residual blocks; block width (the parameter `d`, distinct from the relative day d); ratio of a block's hidden width to `d`; dropout inside a block; dropout on a block's residual branch; length of each category's embedding vector |
| `n_layers`, `d_token`, `attention_dropout`, `ffn_dropout`, `d_ffn_factor` | FT-Transformer: number of Transformer layers; length of each input's token vector; dropout of the attention weights; dropout inside the feed-forward module; ratio of the feed-forward hidden size to `d_token` |
| `lr`, `weight_decay` | Neural learning rate; AdamW's decoupled weight-decay coefficient |
| AdamW, AMSGrad | Adam optimiser with decoupled weight decay; its variant with a maximum of past second moments (not used) |
| `betas`, `eps` | AdamW's decay coefficients of the first- and second-moment estimates; its numerical-stability constant |
| BatchNorm, LayerNorm | Batch normalisation (statistics over the rows of a batch); layer normalisation (over the values of one row) |
| ReGLU | Rectified gated linear unit (FT-Transformer's feed-forward activation) |
| MiB | Mebibyte (2^20 bytes) |
| Effective batch, micro-batch | The Train rows of one optimiser update (1,024); the part of an effective batch processed by the GPU at once |
| TabZilla | The benchmark of REF-089, whose default ResNet gives the ResNet-style MLP's reference configuration |
