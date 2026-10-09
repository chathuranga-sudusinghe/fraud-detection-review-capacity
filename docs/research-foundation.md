# COMP1885 Research Foundation

## Project

**Programme:** MSc Data Science, University of Greenwich  
**Module:** COMP1885 Individual Project  
**Repository:** `fraud-detection-review-capacity`  
**Research type:** Quantitative, non-participant, secondary-data experimental study

## Working Research Title

**Fraud Detection Under Limited Review Capacity: A Temporally Validated Study of Tree-Based and Neural Models and Decision Policies**

## Research Problem

Transaction fraud detection is a low-prevalence, imbalanced binary-classification problem in which missed fraud can be costly, while excessive false alerts can create unnecessary manual-review workload and customer friction.

In operational settings, review resources are limited. Therefore, a useful fraud-detection system must do more than produce a strong global discrimination score: it must prioritise transactions effectively under a fixed review budget, and the decision policy chosen during validation should remain operationally stable when applied to later data.

Temporal change can alter score distributions, fraud prevalence, feature distributions, and realised alert volume. This creates a research problem at the boundary between predictive ranking and operational decision policy.

In the IEEE-CIS (IEEE Computational Intelligence Society) Fraud Detection development data, fraud prevalence is about 3.5 %, an imbalance ratio (legitimate : fraud) of about 28 : 1. This is described as **imbalanced / low-prevalence**, not "severely" imbalanced: that label is conventionally reserved for ratios of about 100 : 1 and above [REF-002, REF-001]. The practical difficulty comes from low prevalence combined with temporal drift and a review budget that, at 2 % capacity, is smaller than the daily number of frauds ([DR-002](decisions/DR-002-review-capacity-definition.md)).

## Main Research Question

> **Under constrained daily review capacity, how do model family, class weighting, and score-to-review decision policies affect fraud capture, review workload, and cost-sensitive outcomes on temporally unseen transaction data?**

## Supporting Research Questions

The supporting research questions are numbered RQ1–RQ3.

### RQ1

> **How do tree-based and neural model families, with and without class weighting, differ in fraud capture at fixed review capacity on a later temporal test period?**

Tree-based: LightGBM and CatBoost; neural: ResNet-style multilayer perceptron (MLP) and FT-Transformer ([DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md)).

### RQ2

> **When a score threshold selected on validation data is transferred to a later temporal period, how well does it preserve the intended alert rate, and how does the resulting daily review workload compare with the fixed review budget?**

[DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md) separates alert-rate drift, which depends on the model and data, from daily-volume variation, which exists regardless of the model.

### RQ3

> **Under different assumed miss-to-review cost ratios, when does a calibrated cost-aware review policy outperform always filling the available review capacity?**

RQ3 is evaluated **across the capacity grid** (1 %, 2 %, 5 %, 10 %). Its main output is the **break-even miss-to-review cost ratio at each capacity**: the largest assumed miss-to-review cost ratio ρ (cost of a missed fraud C_miss ÷ cost of one review C_R) at which the cost-aware policy is strictly cheaper than filling capacity. It is reported with the saturation ratio, above which the two policies are identical ([DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md)). At 2 % the budget is below the daily fraud count, so the cost-aware policy is expected to coincide with filling capacity above a low ratio. The informative contrasts are expected at 5 % and 10 % ([DR-002](decisions/DR-002-review-capacity-definition.md)).

## Confirmatory Hypotheses

The confirmatory hypotheses are H1–H3. A subscript ₀ marks the null hypothesis and ₁ the alternative. Recall@2% is the share of Test frauds reviewed when 2 % of transactions can be reviewed each day.

### H1 — Model-family effect

H1 concerns the model-family effect across the eight configurations of [DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md):

$$
\delta_F = \frac{1}{4} \sum_{m \in N} \sum_{w \in \lbrace u, b \rbrace} R^{m,w} \;-\; \frac{1}{4} \sum_{m \in T} \sum_{w \in \lbrace u, b \rbrace} R^{m,w}
$$

where:

- `δ_F` = neural-minus-tree difference in Recall@2%, averaged over the two models of each family and the two weighting conditions;
- `T` = tree-based family {LGB (LightGBM), Cat (CatBoost)}; `N` = neural family {RES (ResNet-style MLP), FTT (FT-Transformer)};
- `R^m,w` = pooled Test Recall@2% of model m with weighting w ∈ {u (unweighted), b (balanced)}.

**H1₀:** Averaged over the two models of each family and over the unweighted and balanced-weighting conditions, tree-based and neural models do not differ in Recall@2% on the Test period (δ_F = 0).

**H1₁:** Averaged over the two models of each family and over the unweighted and balanced-weighting conditions, tree-based and neural models differ in Recall@2% on the Test period (δ_F ≠ 0).

H1 is a two-sided difference test. It is not a winner-selection test. Its 90 % confidence interval (CI) is also interpreted against the smallest effect size of interest (SESOI) of ±2 percentage points, to distinguish practically meaningful, negligible and inconclusive differences ([DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md)).

### H2 — Class-weighting effect

H2 concerns the weighting effect of each model m:

$$
\delta_{W,m} = R^{m,b} - R^{m,u} \qquad \text{for } m \in \lbrace \mathrm{LGB}, \mathrm{Cat}, \mathrm{RES}, \mathrm{FTT} \rbrace
$$

where `δ_W,m` = balanced-minus-unweighted Recall@2% of model m.

**H2₀:** For at least one of the four models (LightGBM, CatBoost, ResNet-style MLP or FT-Transformer), balanced class weighting changes Recall@2% by 2 percentage points or more relative to the corresponding unweighted model (|δ_W,m| ≥ 0.02 for at least one m).

**H2₁:** For all four models, balanced class weighting changes Recall@2% by less than 2 percentage points relative to the corresponding unweighted model (|δ_W,m| < 0.02 for every m).

H2 is an equivalence hypothesis, tested by an intersection-union TOST (Two One-Sided Tests) over all four models [REF-061, REF-062]. This prevents a non-significant difference being read as "no effect" [REF-030].

**SESOI = 2 percentage points of Recall@2%** (≈ 53 frauds per 28 days, or ≈ 2 per day, at identical review workload). It was fixed before any Test access. No fraud-domain standard exists, so it is a pre-specified researcher judgement based on operational meaning and on the design's resolution ([DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md)).

### H3 — Alert-rate transfer fidelity

H3 concerns the realised alert rate r of the score threshold selected on Validation 2 (V2, relative days 127–154) to flag 2 % of transactions, applied unchanged to the Test period:

$$
r = \frac{\sum_d n_d}{\sum_d N_d}
$$

where:

- `d` = a Test relative day;
- `n_d` = number of transactions on day d flagged by the threshold;
- `N_d` = total number of transactions on day d;
- `Σ_d` = sum over the 28 Test days.

Δ (the equivalence tolerance) bounds the deviation of r from 0.02: r preserves the target when it deviates by less than Δ.

**H3₀:** The realised alert rate under the validation-selected 2% threshold lies outside the pre-specified equivalence band around the 2% target (|r − 0.02| ≥ Δ, where Δ = 0.004, i.e. ±20% of the target).

**H3₁:** The realised alert rate under the validation-selected 2% threshold lies within the pre-specified equivalence band around the 2% target (|r − 0.02| < Δ, where Δ = 0.004, i.e. ±20% of the target).

Δ is a relative tolerance of 20 % of the target:

$$
\begin{aligned}
\Delta &= 0.20 \times 0.02 \\
&= 0.004 \\[6pt]
\text{equivalence band} &= (0.02 - \Delta,\ 0.02 + \Delta) \\
&= (1.6\%,\ 2.4\%)
\end{aligned}
$$

It was fixed before any Test access. No literature standard exists for this tolerance, so the magnitude is a pre-specified researcher judgement, anchored in development-period workload variation ([DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md)). H3 is tested with TOST, using a day-level moving-block bootstrap, and decided by its Holm-adjusted p-value [REF-061, REF-062, REF-064]. It is confirmatory for one of the eight model × weighting configurations only: the one with the highest Recall@2% on V2 under the daily top-K policy, selected by a frozen rule before any Test access. H3 results for the other seven configurations are descriptive and include a tree-vs-neural comparison of threshold transfer ([DR-010](decisions/DR-010-h3-confirmatory-configuration.md)). Fidelity to the fixed daily budget (daily threshold-generated alert demand vs K, the fixed daily review budget; overflow and unused capacity) is a separate descriptive operational outcome, not a hypothesis. Threshold alerts are never truncated to K, and alerts in excess of K are not counted as missed frauds ([DR-009](decisions/DR-009-overflow-handling-for-threshold-policies.md)).

RQ3 will primarily be evaluated through cost-ratio sensitivity analysis and break-even analysis across the capacity grid rather than a separate omnibus hypothesis test.

## Research Aim

To quantify how predictive ranking quality and score-to-review decision policies jointly determine fraud capture, workload fidelity, and cost-sensitive outcomes under fixed review capacity and temporal shift.

## Research Objectives

1. Establish and freeze a reproducible, leakage-controlled temporal experimental protocol using the IEEE-CIS Fraud Detection dataset.
2. Compare tree-based (LightGBM, CatBoost) and neural (ResNet-style MLP, FT-Transformer) model families under one common procedure: identical data, model inputs, class weights, tuning method and budget, and evaluation conditions, with family-specific input representations ([DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md) §3).
3. Evaluate unweighted and balanced class-weighted variants of each model.
4. Measure fraud capture under fixed review-capacity scenarios using capacity-aware metrics.
5. Evaluate how validation-selected score thresholds transfer to later data, separating alert-rate fidelity from fixed daily-budget fidelity.
6. Evaluate calibrated cost-aware review policies across the capacity grid, using scenario-based cost ratios and break-even analysis.
7. Quantify uncertainty using pre-specified, cluster- or day-block-aware confidence intervals, paired comparisons, and effect sizes.
8. Characterise false-negative behaviour across pre-declared transaction subgroups without causal interpretation.
9. Document ethical considerations, assumptions, limitations, and threats to validity.

## Core Experimental Factors

### Model family

Fixed in [DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md):

- **Tree-based:** LightGBM; CatBoost.
- **Neural:** ResNet-style MLP; FT-Transformer [REF-087].

An L2-regularised logistic-regression model is considered as a descriptive reference baseline, outside the hypotheses; its inclusion is decided at the design review.

### Feature set

All four models receive the same 431 raw input columns ([DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md); [`config/features.yaml`](../config/features.yaml)):

- 49 categorical columns, exactly the organiser's published list;
- 382 numeric columns;
- `TransactionID` (row identifier) and raw `TransactionDT` (time offset in seconds) excluded;
- the target `isFraud` never used as an input.

No feature engineering, target encoding or entity identifiers are added. Missing categorical values share one explicit label for all models. The tree-based models handle missing numeric values natively; the neural models receive Train-fitted quantile-transformed values with missingness indicators (proposed, DR-008 §9).

### Class weighting

- unweighted
- balanced weighting: explicit sample weights computed from Train only. Each fraud row receives the weight

  $$
  w_{\mathrm{pos}} = \frac{n_{\mathrm{neg}}}{n_{\mathrm{pos}}}
  $$

  where `w_pos` = weight of each fraud (positive-class) row, `n_neg` = number of legitimate Train transactions and `n_pos` = number of fraud Train transactions. Legitimate rows keep weight 1.

Resampling methods (SMOTE, the Synthetic Minority Over-sampling Technique; over- and under-sampling) are excluded [REF-019, REF-020, REF-022].

### Decision policies

Decision policies, as defined in the [protocol](protocol.md):

- R0: default probability threshold 0.5;
- R1: validation-selected F1-optimal threshold;
- R2: validation-selected capacity-rate threshold;
- R3: fixed daily top-K review policy;
- R4: calibrated cost-aware capped review policy.

### Review-capacity scenarios

Capacity scenarios, fixed in [DR-002](decisions/DR-002-review-capacity-definition.md):

- 1%
- 2% — primary scenario
- 5%
- 10%

These are simulated research scenarios. They must not be represented as real institutional review capacities. Each capacity share c is turned into a fixed daily review budget ([DR-002](decisions/DR-002-review-capacity-definition.md)):

$$
K = \operatorname{round}\left( c \times \text{mean daily transaction volume in V2} \right)
$$

where `K` = fixed daily review budget and `c` = review-capacity share. This gives 28 / 56 / 139 / 278 reviews per relative day.

## Primary and Secondary Outcomes

### Primary outcome

- Recall@2% review capacity (daily top-K, K = 56), reported with its oracle ceiling (≈ 0.6, the recall of a perfect ranker) and normalised recall (frauds reviewed ÷ the most that could be reviewed each day)

Precision@Capacity (share of reviewed transactions that are fraud) is a rescaling of Recall@Capacity (share of frauds reviewed) under fixed K. It is reported descriptively, not as independent confirmatory evidence.

### Secondary outcomes

- Recall@1%, 5%, and 10%
- Precision@Capacity
- false negatives at capacity
- frauds detected per 1,000 reviews
- lift over random review
- review utilisation (capped policies R3 and R4)
- threshold-generated alert demand (R0–R2)
- overflow and unused capacity relative to K, and percentage of days above and below K (R0–R2; [DR-009](decisions/DR-009-overflow-handling-for-threshold-policies.md))
- Average Precision / PR-AUC (Precision–Recall Area Under the Curve)
- ROC-AUC (Receiver Operating Characteristic Area Under the Curve)
- scenario cost
- calibration metrics where relevant

## Operational Interpretation

The research represents model-assisted manual-review prioritisation:

```text
Transaction
    ↓
Fraud-risk score
    ↓
Score-to-review decision policy
    ↓
Capacity-constrained review queue
    ↓
Manual verification
    ↓
Final operational action
```

The study does not evaluate analyst behaviour. Human review is represented as an external capacity constraint under a stated verification assumption.

## Experimental Principle

The research will use separate temporal stages:

```text
TRAIN
  -> model fitting

VALIDATION 1   (relative days 106-126)
  -> hyperparameter selection
  -> early stopping

VALIDATION 2   (relative days 127-154)
  -> probability calibration (for the cost-aware policy R4)
  -> threshold selection
  -> daily review budget K
  -> selection of the H3 configuration

NO PRIMARY GAP

FINAL TEST     (relative days 155-182, sealed)
  -> one confirmatory evaluation
```

Train covers relative days 1–105. The day boundaries were chosen from transaction volume alone, before any target analysis, and are recorded with partition hashes in [DR-001](decisions/DR-001-temporal-split-strategy.md). Calibration is fitted on V2 rather than V1 because V1 is used for early stopping and its fraud prevalence differs from V2's. Calibration affects only the cost-aware policy: the other policies rank and threshold on raw model scores.

The final test set must not be used to select:

- model family;
- features;
- hyperparameters;
- class weighting;
- calibrator;
- decision threshold;
- review-capacity policy;
- cost scenario.

## Statistical Evaluation

The final statistical plan will prioritise:

- joint paired bootstrap confidence intervals over all eight model × weighting configurations. The primary method is a two-way (pigeonhole) bootstrap over relative days × heuristic pseudo-entities, because both day-level and cross-day entity dependence are present. Sensitivity analyses use an Owen-corrected variance and a key-free day-block bootstrap ([DR-006](decisions/DR-006-bootstrap-inference-strategy.md)) [REF-036, REF-037, REF-068, REF-070];
- exact McNemar testing as a secondary, descriptive check for cell-level paired fraud-capture differences. It is not confirmatory, because clustering can inflate its size [REF-035, REF-071];
- effect sizes, with practical interpretation against the SESOI of 2 percentage points [REF-066];
- Holm (sequentially rejective multiple-testing) correction over the confirmatory family {H1, H2, H3}, three members, with H3 tested once on the rule-selected configuration; decisions use Holm-adjusted p-values, so Holm's step-down stopping rule applies [REF-029]; [DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md), [DR-010](decisions/DR-010-h3-confirmatory-configuration.md);
- equivalence testing (TOST) for H2 (intersection-union over four models, SESOI = 2 percentage points, pp) and H3 (Δ = 0.004, ±20 % relative) [REF-030, REF-061, REF-062].

Statistical significance will not be treated as equivalent to practical importance.

## Cost-Sensitive Evaluation

The research will use scenario-based cost analysis rather than unsupported claims about real banking costs.

The primary formulation ([DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md)) counts cost in units of one review:

$$
\begin{aligned}
\text{Scenario cost} &= \text{number of reviews} + \rho \times \text{number of frauds not reviewed} \\
\rho &= \frac{C_{\mathrm{miss}}}{C_R}
\end{aligned}
$$

where:

- `ρ` = miss-to-review cost ratio;
- `C_miss` = assumed cost of missing a fraud;
- `C_R` = assumed cost of reviewing one transaction.

The scenario cost counts one unit per review and ρ units per missed fraud.

The cost-aware policy R4 reviews a transaction from the daily top-K queue only if its calibrated fraud probability reaches the threshold [REF-008, REF-073]:

$$
\begin{aligned}
p^* &= \frac{C_R}{C_{\mathrm{miss}}} \\
&= \frac{1}{\rho}
\end{aligned}
$$

where:

- `p*` = cost-derived probability threshold used by R4.

A transaction is reviewed when at least one review in ρ is expected to find a fraud.

An additional false-alert / customer-friction cost C_FP is a sensitivity analysis only. For the R4-versus-R3 comparison it is equivalent to replacing ρ by the effective ratio:

$$
\rho' = \frac{C_{\mathrm{miss}} + C_{\mathrm{FP}}}{C_R + C_{\mathrm{FP}}}
$$

where:

- `ρ′` = effective cost ratio;
- `C_FP` = additional false-alert/customer-friction cost for reviewing a legitimate transaction.

With C_FP included, the threshold becomes p* = 1 / ρ′, so the same analysis applies with a relabelled ratio axis.

The declared grid is ρ ∈ {2, 5, 10, 20, 50, 100, 200, 500, 1000}, with exact curves over (1, 1000]. All cost ratios are illustrative research scenarios, not real banking costs, and are analysed using sensitivity and break-even analysis [REF-039, REF-040, REF-072]. `TransactionAmt` is used only for exposure-proxy sensitivity outcomes (amount-weighted recall, fraud value exposed), never as a realised loss.

## False-Negative Analysis

False-negative analysis is exploratory, descriptive and non-causal. Its slices are pre-declared in the [protocol](protocol.md) before final test evaluation:

- identity information present / absent;
- product category (`ProductCD`);
- card characteristics (`card4` and `card6`);
- device information (`DeviceType`, including missing);
- transaction-amount bands;
- temporal sub-period (relative week within Test);
- score band;
- recurring vs new pseudo-entity (whether the transaction's heuristic entity already had a Train-period fraud label).

Slices with fewer than about 100 frauds are pooled or reported descriptively only, not interpreted independently.

## Research Contribution

The study proposes no new fraud-detection algorithm, decision method, cost framework or temporal protocol. Its contribution is an **integrated, pre-specified confirmatory evaluation** of the full path from model family to operational review decision, on one real chronological benchmark (IEEE-CIS) and under one fixed daily review budget:

```text
model family (tree-based vs neural)
    ×
class weighting
    ×
score-to-review policy
    ×
fixed daily review capacity
    ×
temporal transfer
    ↓
fraud capture
workload fidelity
cost-sensitive outcome
```

The structured literature review ([`literature-review.md`](literature-review.md), Sections 4.6–4.9 and 8; [`literature-evidence-matrix.md`](literature-evidence-matrix.md), Section 3) finds each element in prior work, but no study that combines a fixed daily review budget, a pre-specified test of threshold transfer, a tree-vs-neural comparison under one common procedure and dependence-aware inference on real chronological data.

**No novelty is claimed for any single element**, each of which has established prior work:
- fraud-detection algorithms, and the tree-vs-neural comparison on general tabular benchmarks [REF-087, REF-089, REF-090];
- temporal evaluation, including chronological evaluation on this dataset [REF-023, REF-041, REF-058];
- capacity-constrained evaluation and top-K review, including daily alert budgets [REF-006, REF-004, REF-005, REF-046, REF-084, REF-086], and capacity-motivated ranking metrics on IEEE-CIS [REF-099];
- validation-selected thresholds applied unchanged to later data [REF-058, REF-079, REF-080];
- temporal transfer of review policies under capacity, including a separate policy-selection period [REF-078];
- cost-sensitive thresholding and calibrated cost-threshold policies [REF-008, REF-039, REF-078];
- binding-versus-slack capacity effects and cost-ratio sweeps [REF-078, REF-080];
- capacity-aware threshold selection or case allocation [REF-081, REF-082, REF-083].

**The most distinctive component.** The closest study, Zhang et al. [REF-078], transfers a historical review *rate* (which preserves the review fraction by construction) and evaluates workload rules against a retrospective oracle; it notes that attributing workload displacement to its sources would require further controlled comparison. In the studies reviewed, the alert rate of a *static* validation-selected score threshold is not formally tested against its intended target, and deviations from a fixed daily review budget are not separated into alert-rate drift and volume variation.

The contribution therefore has these components, in order of distinctiveness:

1. **RQ2 / H3 (confirmatory).** A formal equivalence test (TOST) of whether the realised Test alert rate of the V2-selected 2 % threshold lies within ±20 % of its 2 % target, with a descriptive tree-vs-neural comparison of threshold transfer.
2. **Daily-budget fidelity (descriptive, separate from H3).** Threshold-generated demand compared with the fixed daily budget K (overflow, unused capacity, load ratio), with the workload deviation separated into an alert-rate component and a volume-driven component ([DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md), [DR-009](decisions/DR-009-overflow-handling-for-threshold-policies.md)).
3. **RQ1 / H1, H2 (confirmatory).** A pre-specified, dependence-aware comparison of tree-based and neural model families, and of class weighting, at fixed daily capacity on chronological data, under one common training procedure ([DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md), [DR-006](decisions/DR-006-bootstrap-inference-strategy.md), [DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md)). Fraud-specific evidence on this family question is limited and mostly non-temporal [REF-098, REF-099, REF-100, REF-101]; the study contributes a capacity-based, temporally validated answer for these pipelines.
4. **RQ3 as a replication-plus-extension.** Zhang et al. [REF-078] show that a calibrated cost threshold and full-capacity review are largely compressed toward the same capacity ceiling when capacity is tight and diverge when capacity is slack. RQ3 re-examines this pattern under this study's fixed daily cap and summarises it as break-even and saturation cost ratios (ρ_BE, ρ_sat; [DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md)).

The conclusions are bounded by one benchmark, one validation-to-test transfer, the four models studied and simulated capacity and cost assumptions.

## Next Research Tasks

1. Design review with the supervisor ([Issue #54](https://github.com/chathuranga-sudusinghe/fraud-detection-review-capacity/issues/54)); resolve the design items in [`protocol.md`](protocol.md#design-items-before-protocol-freeze), including the neural-network representation and the training rules.
2. Create the dataset manifest and split-generation code that reproduces the DR-001 partition hashes, without reading Test-period labels.
3. Define the data validation contract; implement and test the reusable data, model and policy pipelines.
4. Run the approved development experiments: fit on Train, tune and early-stop on Validation 1, calibrate and select thresholds, review budgets and the H3 configuration on Validation 2.
5. Freeze the selected models, calibrators, thresholds, budgets and H3 configuration, and tag the protocol.
6. Run the confirmatory Test evaluation once; complete the statistical analysis and reporting.

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

Full entries and verification status are in [`references.md`](references.md).

REF-001, REF-002, REF-004, REF-005, REF-006, REF-008, REF-019, REF-020, REF-022, REF-023, REF-029, REF-030, REF-035, REF-036, REF-037, REF-039, REF-040, REF-041, REF-046, REF-058, REF-061, REF-062, REF-064, REF-066, REF-068, REF-070, REF-071, REF-072, REF-073, REF-078, REF-079, REF-080, REF-081\*, REF-082, REF-083, REF-084, REF-086, REF-087, REF-089, REF-090, REF-098, REF-099, REF-100, REF-101.

\* Partially verified.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| c | Review-capacity share (1 %, 2 %, 5 % or 10 %) |
| `card4`, `card6`, `DeviceType`, `ProductCD` | Dataset columns used as false-negative slices (card attributes, device type, product code) |
| C_FP | Extra false-alert/customer-friction cost of reviewing a legitimate transaction (sensitivity only) |
| C_miss | Assumed cost of missing a fraud |
| C_R | Assumed cost of reviewing one transaction |
| Cat, LGB, RES, FTT | CatBoost, LightGBM, ResNet-style MLP, FT-Transformer |
| `isFraud` | Target: 1 = fraud, 0 = legitimate |
| `TransactionDT`, `TransactionID` | Raw time offset and row identifier (excluded from model input) |
| CI | Confidence interval |
| d | A Test relative day |
| Δ | H3 equivalence tolerance on r (0.004) |
| δ_F | Model-family effect: neural mean minus tree-based mean of Recall@2% (H1) |
| δ_W,m | Weighting effect of model m (H2) |
| DR | Decision Record |
| m, w | Model (LGB, Cat, RES or FTT); weighting (u unweighted, b balanced) |
| MLP | Multilayer perceptron |
| N, T | Neural family {RES, FTT}; tree-based family {LGB, Cat} |
| H1, H2, H3 | Confirmatory hypotheses; subscript ₀ = null, ₁ = alternative |
| IEEE-CIS | IEEE Computational Intelligence Society (dataset organiser) |
| K | Fixed daily review budget (reviews per relative day) |
| n_d | Number of Test transactions on day d flagged by the V2-selected 2 % threshold |
| N_d | Total number of transactions on day d |
| n_neg, n_pos | Numbers of legitimate and fraud transactions in Train |
| p* | Cost-derived probability threshold used by R4 |
| pp | Percentage points |
| PR-AUC | Precision–Recall Area Under the Curve |
| Precision@Capacity | Share of reviewed transactions that are fraud at a given capacity |
| r | Realised Test alert rate of the V2-selected 2 % threshold |
| R0–R4 | Decision policies: default 0.5 threshold; F1-optimal threshold; capacity-rate threshold; daily top-K; calibrated cost-aware capped policy |
| R^m,w | Pooled Test Recall@2% of model m with weighting w |
| Recall@2%, Recall@Capacity | Share of frauds reviewed at 2 % (a given) review capacity |
| REF-NNN | Reference identifier in `references.md` |
| ρ | Miss-to-review cost ratio, C_miss / C_R |
| ρ′ | Effective cost ratio, (C_miss + C_FP) / (C_R + C_FP) |
| ρ_BE, ρ_sat | Break-even ratio (largest ρ at which R4 is strictly cheaper than R3) and saturation ratio (smallest ρ at which R4 uses the full budget every day) |
| ROC-AUC | Receiver Operating Characteristic Area Under the Curve |
| RQ1–RQ3 | Research questions 1–3 |
| SESOI | Smallest Effect Size Of Interest (2 percentage points of Recall@2%) |
| SMOTE | Synthetic Minority Over-sampling Technique |
| Σ_d | Sum over Test days |
| TOST | Two One-Sided Tests |
| V1, V2 | Validation 1 (relative days 106–126); Validation 2 (relative days 127–154) |
| w_pos | Sample weight of each fraud row under balanced weighting |
