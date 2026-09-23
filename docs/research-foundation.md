# COMP1885 Research Foundation

## Project

**Programme:** MSc Data Science, University of Greenwich  
**Module:** COMP1885 Individual Project  
**Repository:** `fraud-detection-review-capacity`  
**Research type:** Quantitative, non-participant, secondary-data experimental study

## Working Research Title

**Fraud Detection Under Limited Review Capacity: A Temporally Validated Study of Gradient-Boosting Models and Decision Policies**

## Research Problem

Transaction fraud detection is a low-prevalence, imbalanced binary-classification problem in which missed fraud can be costly, while excessive false alerts can create unnecessary manual-review workload and customer friction.

In operational settings, review resources are limited. Therefore, a useful fraud-detection system must do more than produce a strong global discrimination score: it must prioritise transactions effectively under a fixed review budget, and the decision policy chosen during validation should remain operationally stable when applied to later data.

Temporal change can alter score distributions, fraud prevalence, feature distributions, and realised alert volume. This creates a research problem at the boundary between predictive ranking and operational decision policy.

In the IEEE-CIS development data, fraud prevalence is about 3.5 %, an imbalance ratio of about 28 : 1. This is described as **imbalanced / low-prevalence**, not "severely" imbalanced: that label is conventionally reserved for ratios of about 100 : 1 and above [REF-002, REF-001]. The practical difficulty comes from low prevalence combined with temporal drift and a review budget that, at 2 % capacity, is smaller than the daily number of frauds ([DR-002](decisions/DR-002-review-capacity-definition.md)).

## Main Research Question

> **Under constrained daily review capacity, how do model family, class weighting, and score-to-review decision policies affect fraud capture, review workload, and cost-sensitive outcomes on temporally unseen transaction data?**

## Supporting Research Questions

### RQ1

> **How do LightGBM and CatBoost, with and without class weighting, differ in fraud capture at fixed review capacity on a later temporal test period?**

### RQ2

> **When a score threshold selected on validation data is transferred to a later temporal period, how well does it preserve the intended alert rate, and how does the resulting daily review workload compare with the fixed review budget?**

Revised after the feasibility audit ([DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md)). The earlier wording mixed alert-rate drift, which depends on the model and data, with daily-volume variation, which exists regardless of the model.

### RQ3

> **Under different assumed miss-to-review cost ratios, when does a calibrated cost-aware review policy outperform always filling the available review capacity?**

RQ3 is evaluated **across the capacity grid** (1 %, 2 %, 5 %, 10 %). Its main output is the **break-even miss-to-review cost ratio at each capacity**. At 2 % the budget is below the daily fraud count, so the cost-aware policy is expected to coincide with filling capacity for most plausible ratios. The informative contrasts are expected at 5 % and 10 % ([DR-002](decisions/DR-002-review-capacity-definition.md)).

## Confirmatory Hypotheses

### H1 — Model-family effect

**H1₀:** There is no difference in Recall@2% review capacity between CatBoost and LightGBM on the temporal test period.

**H1₁:** There is a difference in Recall@2% review capacity between CatBoost and LightGBM on the temporal test period.

### H2 — Class-weighting effect

**H2₀:** Class weighting does not change Recall@2% review capacity relative to the corresponding unweighted model.

**H2₁:** Class weighting changes Recall@2% review capacity relative to the corresponding unweighted model.

Smallest effects of interest for H1 and H2 must be declared before protocol freeze. An equivalence test for H2 may be added only once its SESOI is justified [REF-030].

### H3 — Alert-rate transfer fidelity

**H3₀:** The realised alert rate under the validation-selected 2% threshold lies outside the pre-specified equivalence band around the 2% target (|r − 0.02| ≥ Δ, where Δ = 0.004, i.e. ±20% of the target).

**H3₁:** The realised alert rate under the validation-selected 2% threshold lies within the pre-specified equivalence band around the 2% target (|r − 0.02| < Δ, where Δ = 0.004, i.e. ±20% of the target).

Here r is the realised alert rate pooled over the Test period. Δ = 0.004 is a relative tolerance of 20 % of the target, giving the band (1.6 %, 2.4 %). It was fixed before any Test access. No literature standard exists for this tolerance, so the magnitude is a pre-specified researcher judgement, anchored in development-period workload variation ([DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md)). H3 is tested with two one-sided tests (TOST), using a day-level moving-block bootstrap confidence interval [REF-061, REF-062, REF-064]. Fidelity to the fixed daily budget (daily review count vs K, overflow and underuse days) is a separate descriptive operational outcome, not a hypothesis.

RQ3 will primarily be evaluated through cost-ratio sensitivity analysis and break-even analysis across the capacity grid rather than a separate omnibus hypothesis test.

## Research Aim

To quantify how predictive ranking quality and score-to-review decision policies jointly determine fraud capture, workload fidelity, and cost-sensitive outcomes under fixed review capacity and temporal shift.

## Research Objectives

1. Establish and freeze a reproducible, leakage-controlled temporal experimental protocol using the IEEE-CIS Fraud Detection dataset.
2. Compare LightGBM and CatBoost under identical data, features, tuning budgets, and evaluation conditions.
3. Evaluate unweighted and balanced class-weighted variants of each primary model family.
4. Measure fraud capture under fixed review-capacity scenarios using capacity-aware metrics.
5. Evaluate how validation-selected score thresholds transfer to later data, separating alert-rate fidelity from fixed daily-budget fidelity.
6. Evaluate calibrated cost-aware review policies across the capacity grid, using scenario-based cost ratios and break-even analysis.
7. Quantify uncertainty using pre-specified, cluster- or day-block-aware confidence intervals, paired comparisons, and effect sizes.
8. Characterise false-negative behaviour across pre-declared transaction subgroups without causal interpretation.
9. Document ethical considerations, assumptions, limitations, and threats to validity.

## Core Experimental Factors

### Model family

- LightGBM
- CatBoost

A simple L2-regularised Logistic Regression model may be included only as a reference baseline if justified by the final methodology.

### Class weighting

- unweighted
- balanced weighting: explicit sample weights, w_pos = n_neg / n_pos computed from Train only

Resampling methods (SMOTE, over- and under-sampling) are excluded [REF-019, REF-020, REF-022].

### Decision policies

Candidate policies:

- R0: default probability threshold 0.5;
- R1: validation-selected F1-optimal threshold;
- R2: validation-selected capacity-quantile threshold;
- R3: fixed daily top-K review policy;
- R4: calibrated cost-aware capped review policy.

### Review-capacity scenarios

Candidate capacity scenarios:

- 1%
- 2% — planned primary scenario
- 5%
- 10%

These are simulated research scenarios. They must not be represented as real institutional review capacities. Each rate c is turned into a fixed daily budget K = round(c × mean daily volume in V2), giving 28 / 56 / 139 / 278 reviews per relative day ([DR-002](decisions/DR-002-review-capacity-definition.md)).

## Primary and Secondary Outcomes

### Primary outcome

- Recall@2% review capacity (daily top-K, K = 56), reported with its oracle ceiling (≈ 0.6) and normalised recall

Precision@Capacity is a rescaling of Recall@Capacity under fixed K. It is reported descriptively, not as independent confirmatory evidence.

### Secondary outcomes

- Recall@1%, 5%, and 10%
- Precision@Capacity
- false negatives at capacity
- frauds detected per 1,000 reviews
- lift over random review
- review utilisation
- realised workload
- percentage of periods exceeding capacity
- Average Precision / PR-AUC
- ROC-AUC
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

NO PRIMARY GAP

FINAL TEST     (relative days 155-182, sealed)
  -> one confirmatory evaluation
```

Train covers relative days 1–105. The day boundaries were chosen from transaction volume alone, before any target analysis, and are recorded with partition hashes in [DR-001](decisions/DR-001-temporal-split-strategy.md). Calibration is fitted on V2 rather than V1 because V1 is used for early stopping and its fraud prevalence differs from V2's. Calibration affects only the cost-aware policy, since monotone calibrators do not change rankings.

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

- paired bootstrap confidence intervals that respect fraud clustering and temporal dependence, using heuristic entity-cluster or day-block resampling [REF-036, REF-037];
- exact McNemar testing as a secondary check for paired fraud-capture differences [REF-035];
- effect sizes;
- Holm correction for pre-specified confirmatory comparisons [REF-029];
- equivalence testing (TOST) for H3 (Δ = 0.004, ±20 % relative), and for class weighting only if a justified smallest effect size of interest is defined [REF-030, REF-061].

Statistical significance will not be treated as equivalent to practical importance.

## Cost-Sensitive Evaluation

The research will use scenario-based cost analysis rather than unsupported claims about real banking costs.

A candidate formulation is:

```text
Total Cost =
    review cost
  + additional false-alert/customer-friction cost
  + missed-fraud cost
```

All cost ratios will be declared as illustrative research scenarios and analysed using sensitivity and break-even analysis.

## False-Negative Analysis

False-negative analysis will be exploratory and pre-declared before final test evaluation.

Candidate slices include:

- identity information present / absent;
- product category;
- card characteristics;
- device information;
- transaction-amount bands;
- temporal sub-period;
- score band.

Subgroups with insufficient fraud counts will not be interpreted independently.

## Research Contribution

The expected contribution is **a controlled, pre-specified, uncertainty-quantified evaluation of how score-to-review policies transfer across time under fixed review capacity on a public benchmark.** It evaluates:

```text
ranking model
    ×
class weighting
    ×
score-to-review policy
    ×
fixed review capacity
    ×
temporal shift
    ↓
fraud capture
workload fidelity
cost-sensitive outcome
```

The study does not claim a novel fraud-detection algorithm. It also claims **no novelty** for any of the following, each of which has established prior work:
- capacity-constrained evaluation itself [REF-006, REF-004, REF-005, REF-046];
- temporal evaluation itself, including on this dataset [REF-023, REF-041 (partially verified)];
- cost-sensitive thresholding itself [REF-008, REF-039].

The value lies in the controlled combination:
- a split frozen before label analysis;
- the separation of alert-rate fidelity from budget fidelity;
- break-even analysis across capacity regimes;
- cluster-aware uncertainty.

## Next Research Tasks

1. Resolve and document the University ethics route.
2. Complete the literature evidence matrix and verify the outstanding references in [`references.md`](references.md).
3. ~~Verify the raw dataset profile and candidate temporal split.~~ Done: see the feasibility audit notebook and [DR-001](decisions/DR-001-temporal-split-strategy.md).
4. Resolve the open items listed in [`protocol.md`](protocol.md#open-items-before-protocol-freeze), then freeze `docs/protocol.md`.
5. Create the dataset manifest.
6. Implement leakage-safe data and split pipelines.
7. Run validation-only model development.
8. Lock the final decision policies.
9. Run the final test evaluation once.

## Decision Records

- [DR-001 — Temporal split strategy](decisions/DR-001-temporal-split-strategy.md)
- [DR-002 — Review-capacity definition](decisions/DR-002-review-capacity-definition.md)
- [DR-003 — RQ2 / H3: policy-transfer fidelity](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md)

## References

Full entries and verification status are in [`references.md`](references.md).

REF-001, REF-002, REF-004, REF-005, REF-006, REF-008, REF-019, REF-020, REF-022, REF-023, REF-029, REF-030, REF-035, REF-036, REF-037\*, REF-039, REF-041\*, REF-046\*, REF-061, REF-062, REF-064.

\* Partially verified.
