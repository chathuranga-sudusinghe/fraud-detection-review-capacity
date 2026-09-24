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

### RQ1 — Model and weighting

> **How do LightGBM and CatBoost, with and without class weighting, differ in fraud capture at fixed review capacity on a later temporal test period?**

### RQ2 — Policy-transfer fidelity

> **When a score threshold selected on validation data is transferred to a later temporal period, how well does it preserve the intended alert rate, and how does the resulting daily review workload compare with the fixed review budget?**

Revised per [DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md). The previous wording mixed alert-rate drift (model/data) with daily-volume variation, which exists regardless of the model.

### RQ3 — Cost-aware capacity use

> **Under different assumed miss-to-review cost ratios, when does a calibrated cost-aware review policy outperform always filling the available review capacity?**

**Scope (revised):**
- RQ3 is evaluated **across the whole capacity grid (1 %, 2 %, 5 %, 10 %)**, not at 2 % alone.
- Its **main output is the break-even miss-to-review cost ratio at each capacity**, supported by cost curves [REF-039, REF-040].
- At 2 % the budget is below the daily fraud count on every validation day, so the cost-aware policy is expected to coincide with daily top-K for most plausible ratios. The informative contrasts are expected at 5 % and 10 % and at low ratios ([DR-002](decisions/DR-002-review-capacity-definition.md)).
- RQ3 is analysed by estimation and sensitivity analysis, not by a separate significance test.

## Confirmatory Hypotheses

### H1 — Model-family effect

Defined per [DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md). Recall@2% is evaluated under policy R3 (daily top-K, K = 56), and all four model × weighting configurations are scored on the same Test frauds. δ_M = ½[(R^Cat,u − R^LGB,u) + (R^Cat,b − R^LGB,b)] is the paired model-family main effect.

- **H1₀:** Averaged over the unweighted and balanced-weighting conditions, CatBoost and LightGBM do not differ in Recall@2% on the Test period (δ_M = 0).
- **H1₁:** Averaged over the unweighted and balanced-weighting conditions, CatBoost and LightGBM differ in Recall@2% on the Test period (δ_M ≠ 0).

**Smallest effect size of interest (SESOI) = 2 percentage points of Recall@2%.** At the projected ≈ 2,660 Test frauds this is ≈ 53 frauds per 28 days, or ≈ 2 per day, at identical review workload (≈ 3.4 pp Precision@K).

H1 remains a two-sided **difference** test; it is not a winner-selection test. The 90 % CI for δ_M is additionally interpreted against ±0.02. This interpretation is pre-specified but not a separate confirmatory test, and has five possible outcomes:
- a practically meaningful difference;
- a difference of uncertain practical size;
- a detectable but negligible difference;
- data compatible only with negligible differences;
- inconclusive.

The model × weighting interaction and the two cell contrasts are reported descriptively.

### H2 — Class-weighting effect

Defined per [DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md). δ_W,f = R^f,b − R^f,u is the paired weighting effect within model family f ∈ {LightGBM, CatBoost}.

- **H2₀:** For at least one model family (LightGBM or CatBoost), balanced class weighting changes Recall@2% by 2 percentage points or more relative to the corresponding unweighted model (|δ_W,f| ≥ 0.02 for at least one f).
- **H2₁:** For both LightGBM and CatBoost, balanced class weighting changes Recall@2% by less than 2 percentage points relative to the corresponding unweighted model (|δ_W,f| < 0.02 for both f).

H2 is an **equivalence** hypothesis, because the literature expects class weighting mainly to distort calibration rather than ranking [REF-019, REF-020]. The substantive question is therefore whether weighting matters *materially*. A non-significant difference must never be read as "no effect" [REF-030, REF-063].

**Test.**
- An intersection-union TOST [REF-061, REF-062]. H2₁ is concluded only if the (1 − 2α_H2) CIs for **both** δ_W,LGB and δ_W,Cat lie inside (−0.02, +0.02), where α_H2 is the Holm-assigned level.
- Equivalence is required in each family, rather than for the averaged effect, so that opposite effects cannot cancel.
- The two-sided difference CIs are also reported descriptively.

### H3 — Alert-rate transfer fidelity

Defined per [DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md). τ₂ is the V2-selected 2 % rate threshold (policy R2). r is the realised Test alert rate at τ₂, pooled over Test days: r = Σ_d n_d / Σ_d N_d.

- **H3₀:** The realised alert rate under the validation-selected 2% threshold lies outside the pre-specified equivalence band around the 2% target (|r − 0.02| ≥ Δ, where Δ = 0.004, i.e. ±20% of the target).
- **H3₁:** The realised alert rate under the validation-selected 2% threshold lies within the pre-specified equivalence band around the 2% target (|r − 0.02| < Δ, where Δ = 0.004, i.e. ±20% of the target).

**Equivalence tolerance.**
- Δ = 0.004 is a **relative tolerance of 20 % of the 2 % target**, giving the band (1.6 %, 2.4 %), i.e. ±11 reviews per day at K = 56.
- It was fixed before any Test access (Issue #7).
- No fraud- or alerting-specific literature standard exists, so the magnitude is a pre-specified researcher judgement. It is anchored in V2 volume-driven workload variation (upper quartile 16.7 %, 90th percentile 26.5 %) and in a precision-feasibility check.
- The relative form follows the common practice of defining equivalence margins relative to a reference value [REF-065].
- Full justification and the alternatives considered (absolute, budget-derived; 10 %, 25 %, 50 %) are in [DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md).

**Test.**
- TOST at α = 0.05 [REF-061, REF-062, REF-030]: H3₁ is concluded only if the 90 % CI for r lies entirely inside (0.016, 0.024).
- A CI overlapping a bound is **inconclusive**.
- The CI comes from a **moving-block bootstrap over Test days**: block length 3 days, B = 10,000, percentile interval [REF-037, REF-064]. Sensitivity block lengths are 1 and 7 days.
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
- **Temporal ordering variable:** `TransactionDT`, a relative time delta from an undisclosed reference point [REF-047]. Relative day = `TransactionDT // 86,400`, covering days 1–182. **No calendar dates are inferred.**

The competition test data are not used because their labels are unavailable.

**Class-imbalance terminology.**
- Development-data prevalence is 3.49 %, an imbalance ratio of ≈ 28 : 1 [D].
- The dataset is described as **imbalanced / low-prevalence**. It is **not** described as "severely" or "highly" imbalanced: that wording is conventionally reserved for majority:minority ratios of about 100 : 1 and above [REF-002, REF-001].
- Imbalance ratio alone does not determine difficulty [REF-003, REF-043]. Here the difficulty comes from low prevalence combined with temporal drift and a review budget smaller than the daily fraud count.

**Labelling caveat.** According to the organiser, a reported chargeback marks the card's transaction as fraud, and later transactions linked by account, email or billing address are also labelled fraud, within a 120-day window [REF-048, partially verified]. Labels are therefore not independent across transactions, and fraud that was never reported is labelled legitimate.

## Experimental Split Structure

Chronological, day-aligned split, frozen from transaction volume only, before any target analysis ([DR-001](decisions/DR-001-temporal-split-strategy.md)):

| Partition | Relative days | Rows | Role |
|---|---|---|---|
| Train | 1–105 | 368,632 | Model fitting. **No refit** on later data. |
| Validation 1 (V1) | 106–126 | 65,544 | Hyper-parameter tuning and early stopping only |
| Validation 2 (V2) | 127–154 | 77,822 | Probability calibration (for R4); threshold selection (R1, R2); daily budget K |
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

- L2 logistic regression. Any preprocessing pipeline is fitted on Train only.

Analytic baselines (always reported):
- random review (precision = prevalence);
- oracle ceiling at each capacity.

**No additional model families will be added.** Tree ensembles are the reference family for tabular data of this kind [REF-013, REF-014].

**Parity controls.** Both libraries receive:
- the same features;
- the same declared categorical columns;
- the same tuning budget and search procedure on V1;
- the same early-stopping rule.

This ensures the comparison reflects model family, not preprocessing.

## Feature Policy

- **Raw or minimally processed features are the primary design.** There is no feature-engineering study.
- **`TransactionID` and raw `TransactionDT` are not model features.** `TransactionID` is monotone in time, so it is a disguised timestamp [D]. Raw time cannot be extrapolated by tree models.
- **No future-looking aggregate features.** Any frequency, count or entity-level aggregate would have to be computed from Train only or with past-only windows. None is planned for the primary design.
- **No manual target encoding.** Categorical handling is native to each library: CatBoost ordered target statistics [REF-012]; LightGBM categorical splits [REF-011].
- **No engineered entity identifiers as features.** Heuristic keys such as card + address + implied account-start day are used **only** for audit and slicing.
- The exact list of columns declared categorical is still open. Several nominal variables (`card1`–`card3`, `card5`, `addr1`, `addr2`, many `id_*`) are stored as numbers, so the declaration must be explicit.

## Class-Imbalance Treatment

Primary comparison:
- no class weighting;
- balanced class weighting.

**Implementation.**
- Weighting is **explicit** and computed **from Train only**: per-row sample weight w_pos = n_neg / n_pos (≈ 28.5 in Train) for fraud rows and 1 for legitimate rows.
- The same weights are applied to both libraries, rather than each library's own "balanced" switch.

**Excluded.** SMOTE, random over-sampling and random under-sampling are outside the design:
- rebalancing distorts calibration without improving discrimination [REF-019, REF-020, REF-021];
- synthetic points have no valid timestamp in a temporal design;
- SMOTE is ill-suited to high-cardinality nominal data [REF-022].

## Review Capacity

Defined in [DR-002](decisions/DR-002-review-capacity-definition.md).

- **Capacity grid:** 1 %, **2 % (primary)**, 5 %, 10 %. These are **simulated research scenarios**, not real institutional capacities.
- **Fixed daily budget:** K = round(c × mean daily transaction volume in V2), i.e. K = 28 / **56** / 139 / 278 reviews per relative day.
- **Capacity ÷ prevalence** is reported with every result. At 2 % it is 0.59 (binding regime); at 5 % and 10 % it is above 1 (slack regime).

Prior art for capacity-constrained fraud evaluation exists [REF-006, REF-004, REF-005, REF-046]. No novelty is claimed for it.

**Primary metric:** Recall@2% under R3 (daily top-K, K = 56), pooled over the Test days. Reported with:
- the **oracle recall ceiling** (≈ 0.6 at 2 % in V2);
- **normalised recall** = TP / min(K, daily frauds).

Precision@K is a rescaling of Recall@K under fixed K. It is reported descriptively and **not** treated as independent confirmatory evidence.

**Secondary outcomes:**
- Recall@1 %, 5 %, 10 %;
- false negatives at capacity;
- frauds per 1,000 reviews;
- lift over random review;
- utilisation, overflow and underuse (see RQ2);
- Average Precision / PR-AUC [REF-010] and ROC-AUC;
- amount-weighted recall;
- scenario cost;
- calibration metrics for R4.

**Overflow rule** for uncapped policies (R0, R1, R2): to be declared at freeze (open item). Options are to report overflow only, or to truncate to the top-K of the flagged transactions.

## Decision Policies

### R0 — Default threshold

Probability threshold = 0.5.

Purpose: naive reference only. Its alert volume mainly reflects the score scale (which changes with class weighting). **Descriptive only; excluded from confirmatory tests.**

### R1 — F1-optimal threshold

Selected using V2 only [REF-009]. Capacity-agnostic, so the overflow rule applies. Secondary comparator.

### R2 — Capacity-rate threshold

τ_c is chosen on V2 so that a fraction c of V2 transactions is flagged. It uses V2 scores and needs no labels.

Purpose: object of RQ2/H3. Is a static validation-selected operating point transferable in time?

### R3 — Daily top-K

Review the K highest-scoring transactions within each relative day.

Purpose: ranking-quality benchmark at full capacity, used for H1/H2. Note that it is an idealised end-of-day batch policy, not a streaming policy [REF-033].

### R4 — Calibrated cost-aware capped policy

Review transaction i if its calibrated probability p_i exceeds the analytic cost threshold, capped at K per day (highest p_i first) [REF-008]:
- p* = C_R / C_miss if only review cost and missed-fraud cost are modelled;
- p* = (C_R + C_FP) / (C_miss + C_FP) if an additional false-alert friction cost C_FP is included.

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
- amount-weighted recall (`TransactionAmt`, in USD per the organiser [REF-048]);
- fraud value exposed (Σ amount of missed fraud) as a proxy for loss exposure.

Amount-weighted results will have wide uncertainty: the top 20 % of development frauds by amount carry 62 % of fraud value [D].

**Assumed quantities:**
- all cost values are **scenario assumptions** [REF-039];
- miss-to-review ratios C_miss / C_R are evaluated over a declared grid (open item);
- results are summarised as **break-even ratios per capacity** and cost curves [REF-040];
- amount-dependent miss costs may be used as a sensitivity scenario [REF-007].

No assumed value will be presented as an actual institutional banking cost.

## Statistical Plan

**Primary confirmatory comparison:** Recall@2% (H1, H2), plus H3 as specified above. The H1/H2 inference plan is defined in [DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md).

**Uncertainty must be cluster- or day-block-aware.** Fraud labels cluster by entity: V2 frauds average ≈ 2 per heuristic pseudo-entity, with an upper-bound design effect ≈ 4.3 [D]. 45 % of V2 frauds belong to entities that span several days, and daily fraud counts are overdispersed (≈ 5 times binomial variance in V2) [D]. Transaction-level resampling would therefore understate uncertainty.

- **Primary (H1, H2):** a **two-way (pigeonhole) joint paired bootstrap** over relative days × fine pseudo-entities, as specified in [DR-006](decisions/DR-006-bootstrap-inference-strategy.md) [REF-068, REF-070]:
  - day multiplicities and entity multiplicities are drawn independently;
  - each Test fraud is weighted by the product of its day's and its entity's multiplicities;
  - the same weights are applied to the fixed daily top-K capture indicators of all four configurations;
  - B = 10,000, percentile CIs [REF-044].

  Both day-level and cross-day entity dependence are observed in V2 [D], and each one-way scheme misses one of them. The method is mildly conservative by design [REF-068]. Resampling that assumes independent transactions or frauds is not acceptable.
- **Pre-declared sensitivity analyses (not confirmatory):**
  - S1: Owen-corrected two-way variance (V_pigeonhole − 2·V_naive) with Wald intervals;
  - S2: a key-free day-level moving-block bootstrap (ℓ = 3 days) that preserves each day's top-K exactly [REF-037, REF-064].

  A different H1 or H2 conclusion under S1 or S2 is reported as sensitivity to dependence assumptions. Confirmatory decisions use the primary method only.
- H3 uses the day-level moving-block bootstrap specified above, because its outcome does not depend on labels.
- **H1:** two-sided bootstrap p-value for δ_M = 0, plus the pre-specified practical interpretation of the 90 % CI against ±SESOI (DR-005).
- **H2:** intersection-union TOST with SESOI = 0.02; p_H2 = the maximum of the four one-sided p-values [REF-061, REF-062].
- **H3:** TOST with Δ = 0.004 (±20 % relative) and a day-level moving-block bootstrap [REF-061, REF-062, REF-064].
- **Multiplicity:** **Holm** at family-wise α = 0.05 over the confirmatory family {H1, H2, H3} [REF-029]. If H3 is tested on more than one configuration (open item 7), each H3 test is an additional family member.
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
- miss rate (FN ÷ frauds in slice);
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
- organiser-engineered C / D / V features whose look-back windows are undocumented.

## Ethics

The study uses secondary, anonymised data and does not recruit human participants.

The University of Greenwich Research Ethics Policy applies to master's students' research (§1.2) [REF-050]. It lists research involving "harmful, criminal, particularly sensitive … subject matters" as requiring ethical consideration (§1.3.4) and as **Higher Risk** (§2.2.1 iv) [REF-050]. Separate guidance states that fully anonymised secondary data should not normally need approval, and that taught students are advised on the route by their programme or module leader [REF-051].

Whether this project falls under §1.3.4 must be **confirmed in writing** before confirmatory experiments (open item). The data-use terms of the competition must also be verified [REF-049].

AI assistance used in this project must be declared in the AI Reflection (AIR) required by University guidance [REF-053].

## Open items before protocol freeze

| # | Item | Notes |
|---|---|---|
| 1 | ~~Justify and freeze the H3 tolerance Δ~~ | **Resolved (Issue #7), pending supervisor review:** Δ = 0.004 (±20 % relative); TOST with a day-level moving-block bootstrap ([DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md)) |
| 2 | ~~Justify and freeze the SESOI for H1 and H2~~ | **Resolved (Issue #8), pending supervisor review:** SESOI = 2 pp of Recall@2%; H1 difference test with practical interpretation; H2 intersection-union TOST ([DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md)) |
| 3 | ~~Decide the primary clustered-bootstrap unit for H1/H2~~ | **Resolved (Issue #14), pending supervisor review:** two-way pigeonhole bootstrap (relative day × fine pseudo-entity), with S1 (Owen-corrected variance) and S2 (day blocks, ℓ = 3) as sensitivity analyses ([DR-006](decisions/DR-006-bootstrap-inference-strategy.md)). (H3: day blocks, ℓ = 3; DR-003.) |
| 4 | **Declare the cost-ratio grid for RQ3** | e.g. C_miss / C_R ∈ {1, 2, 5, 10, 20, 50, 100} as a candidate; whether C_FP is included; precise break-even definition |
| 5 | **Finalise the feature-category declaration** | Exact columns declared categorical for both libraries; confirm the excluded columns |
| 6 | **Declare the overflow rule** for R0/R1/R2 | Report-only vs truncate-to-top-K |
| 7 | **Declare the H3 configuration(s)** | Which model × weighting configuration(s) H3 is tested on. The multiplicity family is defined in DR-005 as Holm over {H1, H2, H3}, with each additional H3 configuration adding a family member. |
| 8 | **Resolve the University of Greenwich ethics route in writing** | Research Ethics Policy §1.3.4 (criminal subject matter) [REF-050, REF-051] |
| 9 | **Verify the Kaggle / IEEE-CIS data-use terms** | REF-049 unverified |
| 10 | **Verify all currently unverified or partially verified references** | REF-016, REF-037, REF-041, REF-042, REF-045, REF-046, REF-047, REF-048, REF-049, REF-055, REF-056–REF-060 |
| 11 | **Obtain the COMP1885 module handbook** | Marking criteria, required structure, AI-use tier [REF-055] |
| 12 | **Create the dataset manifest** | SHA-256 values and partition hashes from the feasibility notebook |

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

## References

Full entries and verification status are in [`docs/references.md`](references.md).

REF-001, REF-002, REF-003, REF-004, REF-005, REF-006, REF-007, REF-008, REF-009, REF-010, REF-011, REF-012, REF-013, REF-014, REF-015, REF-016\*, REF-017, REF-018, REF-019, REF-020, REF-021, REF-022, REF-024, REF-025, REF-026, REF-027, REF-028, REF-029, REF-030, REF-033, REF-034, REF-035, REF-036, REF-037\*, REF-039, REF-040, REF-043, REF-044, REF-046\*, REF-047\*, REF-048\*, REF-049†, REF-050, REF-051, REF-053, REF-061, REF-062, REF-063, REF-064, REF-065, REF-066, REF-067, REF-068, REF-069, REF-070, REF-071.

\* Partially verified. † Unverified; open item.
