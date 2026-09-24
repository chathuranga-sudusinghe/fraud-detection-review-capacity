# COMP1885 Research Protocol

## Status

**Working protocol — pre-freeze. Not yet confirmatory.**

This file defines the intended confirmatory research design. It has been revised to incorporate the accepted findings of the dataset feasibility audit ([`notebooks/01_dataset_feasibility_audit.ipynb`](../notebooks/01_dataset_feasibility_audit.ipynb), conclusion: *"Suitable with methodological changes"*). The resulting methodology decisions are recorded in [`docs/decisions/`](decisions/README.md).

Several parameters are still **open** (see [Open items before protocol freeze](#open-items-before-protocol-freeze)). The protocol becomes confirmatory only when every open item is resolved and the file is tagged in Git.

**Test-set status:** the final Test period (relative days 155–182) is **sealed**. Its labels have not been used for any analysis.

Citations use Reference IDs from [`docs/references.md`](references.md).

## Research Title

**Fraud Detection Under Limited Review Capacity: A Temporally Validated Study of Gradient-Boosting Models and Decision Policies**

## Main Research Question

> **Under constrained daily review capacity, how do model family, class weighting, and score-to-review decision policies affect fraud capture, review workload, and cost-sensitive outcomes on temporally unseen transaction data?**

## Supporting Research Questions

The supporting research questions are numbered RQ1–RQ3.

### RQ1 — Model and weighting

> **How do LightGBM and CatBoost, with and without class weighting, differ in fraud capture at fixed review capacity on a later temporal test period?**

### RQ2 — Policy-transfer fidelity

> **When a score threshold selected on validation data is transferred to a later temporal period, how well does it preserve the intended alert rate, and how does the resulting daily review workload compare with the fixed review budget?**

Revised per [DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md). The previous wording mixed alert-rate drift (model/data) with daily-volume variation, which exists regardless of the model.

### RQ3 — Cost-aware capacity use

> **Under different assumed miss-to-review cost ratios, when does a calibrated cost-aware review policy outperform always filling the available review capacity?**

**Scope (revised):**
- RQ3 is evaluated **across the whole capacity grid (1 %, 2 %, 5 %, 10 %)**, not at 2 % alone.
- Its **main output is the break-even miss-to-review cost ratio ρ_BE at each capacity and for each model configuration**. ρ_BE is the largest assumed ratio of missed-fraud cost to review cost at which the cost-aware policy is cheaper than filling capacity. It is reported with the saturation ratio ρ_sat (the ratio above which the cost-aware policy fills the budget every day) and exact cost-difference curves [REF-039, REF-040, REF-072]. Definitions, grid and outputs are fixed in [DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md).
- At 2 % the budget is below the daily fraud count on every validation day. The cost-aware policy is expected to coincide with daily top-K review (reviewing the K highest-scoring transactions each day, K = fixed daily review budget) above a low saturation ratio, so RQ3 at 2 % is informative only at low ratios. The informative contrasts are expected at 5 % and 10 % ([DR-002](decisions/DR-002-review-capacity-definition.md), [DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md)).
- RQ3 is analysed by estimation and sensitivity analysis, not by a separate significance test.

## Confirmatory Hypotheses

The confirmatory hypotheses are H1–H3. A subscript ₀ marks the null hypothesis and ₁ the alternative.

### H1 — Model-family effect

Defined per [DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md). Recall@2% (the share of Test frauds reviewed at 2 % review capacity) is evaluated under policy R3 (daily top-K: review the K highest-scoring transactions each relative day, with fixed daily review budget K = 56). All four model × weighting configurations are scored on the same Test frauds. The paired model-family main effect is:

$$
\delta_M = \frac{1}{2}\left[ \left(R^{\mathrm{Cat},u} - R^{\mathrm{LGB},u}\right) + \left(R^{\mathrm{Cat},b} - R^{\mathrm{LGB},b}\right) \right]
$$

where:

- `δ_M` = model-family main effect (CatBoost minus LightGBM, averaged over weighting);
- `R^f,w` = pooled Test Recall@2% of model family f ∈ {LGB (LightGBM), Cat (CatBoost)} with weighting w ∈ {u (unweighted), b (balanced)}.

δ_M is positive when CatBoost reviews more frauds than LightGBM.

- **H1₀:** Averaged over the unweighted and balanced-weighting conditions, CatBoost and LightGBM do not differ in Recall@2% on the Test period (δ_M = 0).
- **H1₁:** Averaged over the unweighted and balanced-weighting conditions, CatBoost and LightGBM differ in Recall@2% on the Test period (δ_M ≠ 0).

**Smallest effect size of interest (SESOI) = 2 percentage points of Recall@2%.** At the projected ≈ 2,660 Test frauds this is ≈ 53 frauds per 28 days, or ≈ 2 per day, at identical review workload (≈ 3.4 percentage points, pp, of Precision@K, the share of reviewed transactions that are fraud).

H1 remains a two-sided **difference** test; it is not a winner-selection test. The 90 % confidence interval (CI) for δ_M is additionally interpreted against ±0.02. This interpretation is pre-specified but not a separate confirmatory test, and has five possible outcomes:
- a practically meaningful difference;
- a difference of uncertain practical size;
- a detectable but negligible difference;
- data compatible only with negligible differences;
- inconclusive.

The model × weighting interaction and the two cell contrasts are reported descriptively.

### H2 — Class-weighting effect

Defined per [DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md). The paired weighting effect within model family f ∈ {LightGBM, CatBoost} is:

$$
\delta_{W,f} = R^{f,b} - R^{f,u}
$$

where `δ_W,f` = balanced-minus-unweighted Recall@2% in family f. It is positive when balanced weighting helps that family.

- **H2₀:** For at least one model family (LightGBM or CatBoost), balanced class weighting changes Recall@2% by 2 percentage points or more relative to the corresponding unweighted model (|δ_W,f| ≥ 0.02 for at least one f).
- **H2₁:** For both LightGBM and CatBoost, balanced class weighting changes Recall@2% by less than 2 percentage points relative to the corresponding unweighted model (|δ_W,f| < 0.02 for both f).

H2 is an **equivalence** hypothesis, because the literature expects class weighting mainly to distort calibration rather than ranking [REF-019, REF-020]. The substantive question is therefore whether weighting matters *materially*. A non-significant difference must never be read as "no effect" [REF-030, REF-063].

**Test.**
- An intersection-union TOST (Two One-Sided Tests) [REF-061, REF-062]. H2₁ is concluded only if the (1 − 2α_H2) CIs for **both** δ_W,LGB and δ_W,Cat lie inside (−0.02, +0.02), where α_H2 is the significance level assigned to H2 by Holm's multiple-testing procedure.
- Equivalence is required in each family, rather than for the averaged effect, so that opposite effects cannot cancel.
- The two-sided difference CIs are also reported descriptively.

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

r is the share of all Test transactions that the transferred threshold flags. Δ in the hypotheses below is the equivalence tolerance on r.

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

**Test.**
- TOST at significance level α = 0.05 [REF-061, REF-062, REF-030]: H3₁ is concluded only if the 90 % CI for r lies entirely inside (0.016, 0.024).
- A CI overlapping a bound is **inconclusive**.
- The CI comes from a **moving-block bootstrap over Test days**: block length 3 days, B = 10,000 bootstrap replicates, percentile interval [REF-037, REF-064]. Sensitivity block lengths are 1 and 7 days.
- Inference is conditional on the deployed model and τ₂.
- Non-confirmatory sensitivity: the CI is also reported against ±10 %, ±25 % and ±50 % bounds, together with the smallest tolerance at which equivalence would be concluded.

The old point-null H3 ("the threshold preserves the intended workload") is withdrawn.

**Daily-budget fidelity** is a separate operational outcome, reported descriptively and **not** hypothesis-tested. Measures:
- realised daily review count;
- deviation and absolute deviation from K;
- percentage of days above and below K;
- total overflow and total unused capacity;
- rate/volume decomposition.

See [DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md).

## Dataset

**IEEE-CIS Fraud Detection** [REF-047].

Official source: https://www.kaggle.com/competitions/ieee-fraud-detection/data

Primary labelled source files:

- `train_transaction.csv` (590,540 rows)
- `train_identity.csv` (144,233 rows; 24.4 % of transactions have an identity record)

- **Join:** left join transaction ← identity on `TransactionID`; one-to-one, row-preserving.
- **Temporal ordering variable:** `TransactionDT`, a relative time delta in seconds from an undisclosed reference point [REF-047]. Each transaction is assigned a relative day, covering days 1–182:

  $$
  \text{relative day} = \left\lfloor \frac{\texttt{TransactionDT}}{86{,}400} \right\rfloor
  $$

  where `86,400` = seconds per day and `⌊ · ⌋` = rounding down to a whole number (`//` in Python). **No calendar dates are inferred.**

The competition test data are not used because their labels are unavailable.

**Class-imbalance terminology.**
- Development-data prevalence is 3.49 %, an imbalance ratio (legitimate : fraud) of ≈ 28 : 1 [D].
- The dataset is described as **imbalanced / low-prevalence**. It is **not** described as "severely" or "highly" imbalanced: that wording is conventionally reserved for majority:minority ratios of about 100 : 1 and above [REF-002, REF-001].
- Imbalance ratio alone does not determine difficulty [REF-003, REF-043]. Here the difficulty comes from low prevalence combined with temporal drift and a review budget smaller than the daily fraud count.

**Labelling caveat.** According to the organiser, a reported chargeback marks the card's transaction as fraud, and later transactions linked by account, email or billing address are also labelled fraud, within a 120-day window [REF-048, partially verified]. Labels are therefore not independent across transactions, and fraud that was never reported is labelled legitimate.

## Experimental Split Structure

Chronological, day-aligned split, frozen from transaction volume only, before any target analysis ([DR-001](decisions/DR-001-temporal-split-strategy.md)):

| Partition | Relative days | Rows | Role |
|---|---|---|---|
| Train | 1–105 | 368,632 | Model fitting. **No refit** on later data. |
| Validation 1 (V1) | 106–126 | 65,544 | Hyper-parameter tuning and early stopping only |
| Validation 2 (V2) | 127–154 | 77,822 | Probability calibration (for R4, the calibrated cost-aware policy); threshold selection (R1, the F1-optimal threshold; R2); daily budget K |
| Gap | — | — | **None in the primary design** |
| Final Test | 155–182 | 78,542 | **Sealed.** One confirmatory evaluation |

Partition hashes and `TransactionDT` bounds are recorded in [DR-001](decisions/DR-001-temporal-split-strategy.md).

**Supplementary analyses** (exploratory; development data only; no Test access):
- **Label-latency sensitivity:** repeat label-dependent choices (R1 threshold, R4 calibrator) using only V2 days 127–147.
- **Rolling-origin supplement for RQ2:** several V2-like → next-28-day transfers inside days 1–154, on the primary configuration only [REF-024, REF-025, REF-046].

## Models

Primary:

- LightGBM [REF-011]
- CatBoost [REF-012]

Optional reference (descriptive only, outside H1–H3):

- L2-regularised (ridge-penalised) logistic regression. Any preprocessing pipeline is fitted on Train only.

Analytic baselines (always reported):
- random review (precision = prevalence);
- oracle ceiling at each capacity.

**No additional model families will be added.** Tree ensembles are the reference family for tabular data of this kind [REF-013, REF-014].

**Parity controls.** Both libraries receive:
- the same features;
- the same declared categorical columns ([DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md));
- the same tuning budget and search procedure on V1;
- the same early-stopping rule.

This ensures the comparison reflects model family, not preprocessing.

## Feature Policy

- **Raw or minimally processed features are the primary design.** There is no feature-engineering study.
- **`TransactionID` and raw `TransactionDT` are not model features.** `TransactionID` is monotone in time, so it is a disguised timestamp [D]. Raw time cannot be extrapolated by tree models.
- **No future-looking aggregate features.** Any frequency, count or entity-level aggregate would have to be computed from Train only or with past-only windows. None is planned for the primary design.
- **No manual target encoding.** Categorical handling is native to each library: CatBoost ordered target statistics [REF-012]; LightGBM categorical splits [REF-011].
- **No engineered entity identifiers as features.** Heuristic keys such as card + address + implied account-start day are used **only** for audit and slicing.
- **Feature declaration** ([DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md); machine-readable in [`config/features.yaml`](../config/features.yaml); column-by-column reference in the [feature catalog](feature-catalog.md)). Of the 434 joined columns:
  - 1 is the target (`isFraud`) and 2 are excluded (`TransactionID`, `TransactionDT`);
  - **49 are categorical**, exactly the organiser's published list [REF-047]: `ProductCD`, `card1`–`card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `M1`–`M9`, `DeviceType`, `DeviceInfo`, `id_12`–`id_38`. This includes the numeric-coded fields `card1`–`card3`, `card5`, `addr1`, `addr2` and 12 `id_*` codes;
  - **382 are numeric**: every other column (`TransactionAmt`, `dist1`, `dist2`, `C1`–`C14`, `D1`–`D15`, the `V1`–`V339` feature columns, `id_01`–`id_11`).
- **Model input parity.** Both libraries receive the same 431 columns, built from one canonical frame:
  - categorical values are written as text labels, with missing values labelled `"__MISSING__"`;
  - numeric values stay `float64` with native missing-value handling;
  - LightGBM receives pandas categoricals whose categories are fixed from Train only, and CatBoost receives the text labels directly [REF-076, REF-077];
  - no imputation, scaling, rare-level grouping or high-cardinality exclusion is applied.

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
- The same weights are applied to both libraries, rather than each library's own "balanced" switch.

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

Prior art for capacity-constrained fraud evaluation exists [REF-006, REF-004, REF-005, REF-046]. No novelty is claimed for it.

**Primary metric:** Recall@2% under R3 (daily top-K, K = 56), pooled over the Test days. Reported with:
- the **oracle recall ceiling** (≈ 0.6 at 2 % in V2);
- **normalised recall**, computed per day and pooled:

  $$
  \text{normalised recall}_d = \frac{\mathrm{TP}_d}{\min(K, F_d)}
  $$

  where `TP_d` (true positives) = frauds reviewed on day d and `F_d` = frauds on day d. It compares the frauds reviewed with the most any policy could review.

Precision@K is a rescaling of Recall@K (share of all frauds reviewed under budget K) under fixed K. It is reported descriptively and **not** treated as independent confirmatory evidence.

**Secondary outcomes:**
- Recall@1 %, 5 %, 10 %;
- false negatives at capacity;
- frauds per 1,000 reviews;
- lift over random review;
- utilisation, overflow and underuse (see RQ2);
- Average Precision / PR-AUC (Precision–Recall Area Under the Curve) [REF-010] and ROC-AUC (Receiver Operating Characteristic Area Under the Curve);
- amount-weighted recall;
- scenario cost;
- calibration metrics for R4.

**Overflow rule** for uncapped policies (R0, the default 0.5 threshold; R1; R2): to be declared at freeze (open item). Options are to report overflow only, or to truncate to the top-K of the flagged transactions.

## Decision Policies

### R0 — Default threshold

Probability threshold = 0.5.

Purpose: naive reference only. Its alert volume mainly reflects the score scale (which changes with class weighting). **Descriptive only; excluded from confirmatory tests.**

### R1 — F1-optimal threshold

The threshold that maximises F1 (the harmonic mean of precision and recall), selected using V2 only [REF-009]. Capacity-agnostic, so the overflow rule applies. Secondary comparator.

### R2 — Capacity-rate threshold

The score threshold τ_c is chosen on V2 so that a fraction c of V2 transactions is flagged (τ₂ is τ_c at c = 2 %). It uses V2 scores and needs no labels.

Purpose: object of RQ2/H3. Is a static validation-selected operating point transferable in time?

### R3 — Daily top-K

Review the K highest-scoring transactions within each relative day.

Purpose: ranking-quality benchmark at full capacity, used for H1/H2. Note that it is an idealised end-of-day batch policy, not a streaming policy [REF-033].

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
- **Relevant to R4 (and the descriptive R0) only.** Platt and isotonic calibration are monotone. They do not change the ranking, so R2 and R3 are unaffected, and R1 selects the same flagged set on either scale [I; REF-015].
- **Primary method:** Platt scaling [REF-016]. **Secondary check:** isotonic regression [REF-017].
- **Metrics:** Brier score, log loss, calibration intercept and slope, reliability diagram [REF-018]. Reported on V2 (in-sample for the calibrator) and on Test.

## Cost Model

The analysis separates:
- review effort (C_R);
- additional false-alert / customer-friction cost (C_FP, optional);
- missed-fraud cost (C_miss).

**Data-supported quantities [D]:**
- count of frauds captured and missed;
- amount-weighted recall (`TransactionAmt`, in United States dollars, USD, per the organiser [REF-048]);
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

**Uncertainty (descriptive).** 95 % percentile intervals at grid ratios come from a day-level moving-block bootstrap (block length ℓ = 3 days, B = 10,000, R3 and R4 paired). A DR-006 pigeonhole check is applied to frauds forgone. ρ_sat and ρ_BE are point estimates.

**Calibration dependence.** Under calibrated probabilities, R4 cannot be worse than R3 in expectation. An observed loss for R4 is read as a calibration-transfer finding [REF-074], and is checked with isotonic calibration and a declined-set calibration diagnostic.

No assumed value will be presented as an actual institutional banking cost.

## Statistical Plan

**Primary confirmatory comparison:** Recall@2% (H1, H2), plus H3 as specified above. The H1/H2 inference plan is defined in [DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md).

**Uncertainty must be cluster- or day-block-aware.** Fraud labels cluster by entity: V2 frauds average ≈ 2 per heuristic pseudo-entity, with an upper-bound design effect (variance inflation due to clustering) ≈ 4.3 [D]. 45 % of V2 frauds belong to entities that span several days, and daily fraud counts are overdispersed (≈ 5 times binomial variance in V2) [D]. Transaction-level resampling would therefore understate uncertainty.

- **Primary (H1, H2):** a **two-way (pigeonhole) joint paired bootstrap** over relative days × fine pseudo-entities, as specified in [DR-006](decisions/DR-006-bootstrap-inference-strategy.md) [REF-068, REF-070]:
  - day multiplicities and entity multiplicities are drawn independently;
  - each Test fraud is weighted by the product of its day's and its entity's multiplicities;
  - the same weights are applied to the fixed daily top-K capture indicators of all four configurations;
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
- H3 uses the day-level moving-block bootstrap specified above, because its outcome does not depend on labels.
- **H1:** two-sided bootstrap p-value for δ_M = 0, plus the pre-specified practical interpretation of the 90 % CI against ±SESOI (DR-005).
- **H2:** intersection-union TOST with SESOI = 0.02; p_H2 = the maximum of the four one-sided p-values [REF-061, REF-062].
- **H3:** TOST with Δ = 0.004 (±20 % relative) and a day-level moving-block bootstrap [REF-061, REF-062, REF-064].
- **Multiplicity:** **Holm** (sequentially rejective multiple-testing procedure) at family-wise α = 0.05 over the confirmatory family {H1, H2, H3} [REF-029]. If H3 is tested on more than one configuration (open item 7), each H3 test is an additional family member.
- **Secondary:** exact McNemar tests on the four cell-level paired contrasts [REF-035, REF-028]. They assume independent frauds, and under intracluster correlation their size can greatly exceed the nominal level [REF-071]. They are descriptive only, **not** in the Holm family.
- **Effect sizes** with CIs, in pp, frauds per 28 days and per day, Precision@K equivalent and normalised recall. Sensitivity: H1/H2 interpretations are also shown against SESOIs of 1, 3 and 5 pp (not confirmatory).
- **Secondary outcomes** use estimation with confidence intervals, not additional hypothesis tests. Proportions use Wilson intervals [REF-034] where a bootstrap is not required.

**Power (projection, not Test data).** About 2,660 Test frauds are projected from V2 prevalence. The minimum detectable paired difference in Recall@2% is roughly:
- 1.2–2.8 percentage points if frauds are treated as independent;
- 2–5.6 percentage points under design effects of 2–4.

At the SESOI of 2 pp, under central assumptions (≈ 10 % of frauds discordant between configurations, design effect ≈ 2):
- power is ≈ 0.47–0.64 for the H1 difference test;
- power is ≈ 0.14–0.49 for a single-contrast equivalence test.

It is higher under lower discordance or clustering, and lower otherwise (see [DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md)). The dependence diagnostics in [DR-006](decisions/DR-006-bootstrap-inference-strategy.md) suggest that two-way variance may exceed the design effects of 1–4 assumed there, so real power may be lower. H1 and H2 are therefore powered only for moderate effects. **Inconclusive results are a pre-declared, legitimate outcome** and will be reported neutrally.

Statistical significance will not be treated as equivalent to practical importance.

## False-Negative Analysis

Exploratory, descriptive and non-causal. Pre-declared slices:

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
- preprocessing, encoders, class weights and any baseline pipelines are fitted on Train only;
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

Whether this project falls under §1.3.4 must be **confirmed in writing** before confirmatory experiments (open item). The data-use terms of the competition must also be verified [REF-049].

Artificial intelligence (AI) assistance used in this project must be declared in the AI Reflection (AIR) required by University guidance [REF-053].

## Open items before protocol freeze

| # | Item | Notes |
|---|---|---|
| 1 | ~~Justify and freeze the H3 tolerance Δ~~ | **Resolved (Issue #7), pending supervisor review:** Δ = 0.004 (±20 % relative); TOST with a day-level moving-block bootstrap ([DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md)) |
| 2 | ~~Justify and freeze the SESOI for H1 and H2~~ | **Resolved (Issue #8), pending supervisor review:** SESOI = 2 pp of Recall@2%; H1 difference test with practical interpretation; H2 intersection-union TOST ([DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md)) |
| 3 | ~~Decide the primary clustered-bootstrap unit for H1/H2~~ | **Resolved (Issue #14), pending supervisor review:** two-way pigeonhole bootstrap (relative day × fine pseudo-entity), with S1 (Owen-corrected variance) and S2 (day blocks, ℓ = 3) as sensitivity analyses ([DR-006](decisions/DR-006-bootstrap-inference-strategy.md)). (H3: day blocks, ℓ = 3; DR-003.) |
| 4 | ~~Declare the cost-ratio grid for RQ3~~ | **Resolved (Issue #13), pending supervisor review:** primary cost model with review and missed-fraud costs only (C_FP as a sensitivity relabelling); ρ ∈ {2, 5, 10, 20, 50, 100, 200, 500, 1000} with exact curves over (1, 1000]; saturation and break-even ratios per capacity and configuration ([DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md)) |
| 5 | ~~Finalise the feature-category declaration~~ | **Resolved (Issue #23), pending supervisor review:** 49 categorical (organiser list), 382 numeric, 2 excluded (`TransactionID`, `TransactionDT`); canonical text labels with `"__MISSING__"` for categorical missing values; Train-fixed categories for LightGBM ([DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md)) |
| 6 | **Declare the overflow rule** for R0/R1/R2 | Report-only vs truncate-to-top-K |
| 7 | **Declare the H3 configuration(s)** | Which model × weighting configuration(s) H3 is tested on. The multiplicity family is defined in DR-005 as Holm over {H1, H2, H3}, with each additional H3 configuration adding a family member. |
| 8 | **Resolve the University of Greenwich ethics route in writing** | Research Ethics Policy §1.3.4 (criminal subject matter) [REF-050, REF-051] |
| 9 | **Verify the Kaggle / IEEE-CIS data-use terms** | REF-049 unverified |
| 10 | **Verify all currently unverified or partially verified references** | REF-016, REF-037, REF-041, REF-042, REF-045, REF-046, REF-047, REF-048, REF-049, REF-055, REF-056–REF-060 |
| 11 | **Obtain the COMP1885 module handbook** | Marking criteria, required structure, AI-use tier [REF-055] |
| 12 | **Create the dataset manifest** | SHA-256 (Secure Hash Algorithm, 256-bit) values and partition hashes from the feasibility notebook |

## Protocol Freeze

The protocol becomes confirmatory only after:

1. every open item above is resolved;
2. the literature evidence review is complete enough to justify the design;
3. the ethics route is confirmed;
4. the raw dataset statistics and split hashes are reproduced by the split-generation code;
5. hypotheses, SESOIs, Δ and statistical tests are fixed;
6. cost scenarios are declared.

After that point the protocol is tagged in Git before the final Test evaluation.

## Decision Records

- [DR-001 — Temporal split strategy](decisions/DR-001-temporal-split-strategy.md)
- [DR-002 — Review-capacity definition](decisions/DR-002-review-capacity-definition.md)
- [DR-003 — RQ2 / H3: policy-transfer fidelity](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md)
- [DR-005 — SESOI and H1/H2 inference plan](decisions/DR-005-sesoi-and-h1-h2-inference.md)
- [DR-006 — Bootstrap inference strategy for H1 and H2](decisions/DR-006-bootstrap-inference-strategy.md)
- [DR-007 — RQ3 cost-ratio grid and break-even rule](decisions/DR-007-rq3-cost-ratio-and-break-even.md)
- [DR-008 — Feature declaration and model-input policy](decisions/DR-008-feature-declaration-and-model-input-policy.md)

## References

Full entries and verification status are in [`docs/references.md`](references.md).

REF-001, REF-002, REF-003, REF-004, REF-005, REF-006, REF-007, REF-008, REF-009, REF-010, REF-011, REF-012, REF-013, REF-014, REF-015, REF-016\*, REF-017, REF-018, REF-019, REF-020, REF-021, REF-022, REF-024, REF-025, REF-026, REF-027, REF-028, REF-029, REF-030, REF-033, REF-034, REF-035, REF-036, REF-037\*, REF-039, REF-040, REF-043, REF-044, REF-046\*, REF-047\*, REF-048\*, REF-049†, REF-050, REF-051, REF-053, REF-061, REF-062, REF-063, REF-064, REF-065, REF-066, REF-067, REF-068, REF-069, REF-070, REF-071, REF-072, REF-073, REF-074, REF-075, REF-076, REF-077.

\* Partially verified. † Unverified; open item.

## Symbols and notation

Every symbol used in this protocol, in alphabetical order. Each is also defined where it first appears.

| Symbol / abbreviation | Meaning |
|---|---|
| [D], [I] | Evidence tags: dataset observation; inference |
| α | Significance level (family-wise 0.05) |
| α_H2 | Significance level assigned to H2 by Holm's procedure |
| AI, AIR | Artificial intelligence; AI Reflection (University of Greenwich AI-use declaration) |
| B | Number of bootstrap replicates (10,000) |
| c | Review-capacity share (1 %, 2 %, 5 % or 10 %) |
| C_FP | Extra false-alert/customer-friction cost of reviewing a legitimate transaction (sensitivity only) |
| C_miss | Cost of missing (not reviewing) a fraud |
| C_R | Cost of reviewing one transaction, charged on every review |
| Cat, LGB | CatBoost, LightGBM |
| CI | Confidence interval |
| d | A relative day |
| Δ | H3 equivalence tolerance on r (0.004 = 20 % of the 2 % target) |
| δ_M | Model-family main effect (H1 estimand) |
| δ_W,f | Weighting effect in model family f (H2 estimands) |
| ΔJ(ρ) | Scenario-cost difference J_R4 − J_R3 (negative = R4 cheaper) |
| `__MISSING__` | Explicit label given to a missing categorical value (both libraries) |
| DR | Decision Record |
| F1 | Harmonic mean of precision and recall |
| `float64` | 64-bit floating-point numeric type |
| F_d | Number of frauds on day d |
| f, w | Model family (LGB or Cat); weighting (u unweighted, b balanced) |
| FN, FN_π | False negatives: frauds not reviewed (by policy π) |
| H1, H2, H3 | Confirmatory hypotheses: model family; class weighting; alert-rate transfer fidelity. Subscript ₀ = null, ₁ = alternative |
| J_π(ρ) | Scenario cost of policy π, in units of one review cost |
| K | Fixed daily review budget (28 / 56 / 139 / 278 reviews per relative day) |
| ℓ | Block length of a moving-block bootstrap, in days |
| n_d | Number of transactions on day d with score ≥ τ₂ |
| N_d | Total number of transactions on day d (volume; not to be confused with N_π) |
| n_neg, n_pos | Numbers of legitimate and fraud transactions in Train |
| N_π | Number of reviews made by policy π |
| p* | Cost-derived probability threshold used by R4 |
| p̂_i | Calibrated (Platt-scaled) fraud probability of transaction i |
| p_H2 | p-value of H2 |
| φ | False-alert-to-review cost ratio, C_FP / C_R |
| π | A decision policy (R3 or R4) in J_π and N_π |
| pp | Percentage points |
| PR-AUC | Precision–Recall Area Under the Curve |
| Precision@K | Share of reviewed transactions that are fraud |
| r | Realised Test alert rate at τ₂, pooled over Test days |
| R0–R4 | Decision policies: default 0.5 threshold; F1-optimal threshold; capacity-rate threshold; daily top-K; calibrated cost-aware capped policy |
| R^f,w | Pooled Test Recall@2% of family f with weighting w |
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
