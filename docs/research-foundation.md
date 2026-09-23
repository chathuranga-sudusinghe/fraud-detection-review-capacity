# COMP1885 Research Foundation

## Project

**Programme:** MSc Data Science, University of Greenwich  
**Module:** COMP1885 Individual Project  
**Repository:** `cost-sensitive-fraud-detection-research`  
**Research type:** Quantitative, non-participant, secondary-data experimental study

## Working Research Title

**Fraud Detection Under Limited Review Capacity: A Temporally Validated Study of Gradient-Boosting Models and Decision Policies**

## Research Problem

Transaction fraud detection is a low-prevalence binary-classification problem in which missed fraud can be costly, while excessive false alerts can create unnecessary manual-review workload and customer friction.

In operational settings, review resources are limited. Therefore, a useful fraud-detection system must do more than produce a strong global discrimination score: it must prioritise transactions effectively under a fixed review budget, and the decision policy chosen during validation should remain operationally stable when applied to later data.

Temporal change can alter score distributions, fraud prevalence, feature distributions, and realised alert volume. This creates a research problem at the boundary between predictive ranking and operational decision policy.

## Main Research Question

> **Under constrained daily review capacity, how do model family, class weighting, and score-to-review decision policies affect fraud capture, review workload, and cost-sensitive outcomes on temporally unseen transaction data?**

## Supporting Research Questions

### RQ1

> **How do LightGBM and CatBoost, with and without class weighting, differ in fraud capture at fixed review capacity on a later temporal test period?**

### RQ2

> **When score thresholds selected on validation data are applied to a later period, how accurately do they preserve the intended review workload, and what effect does workload drift have on missed fraud?**

### RQ3

> **Under different assumed miss-to-review cost ratios, when does a calibrated cost-aware review policy outperform always filling the available review capacity?**

## Confirmatory Hypotheses

### H1 — Model-family effect

**H1₀:** There is no difference in Recall@2% review capacity between CatBoost and LightGBM on the temporal test period.

**H1₁:** There is a difference in Recall@2% review capacity between CatBoost and LightGBM on the temporal test period.

### H2 — Class-weighting effect

**H2₀:** Class weighting does not change Recall@2% review capacity relative to the corresponding unweighted model.

**H2₁:** Class weighting changes Recall@2% review capacity relative to the corresponding unweighted model.

### H3 — Policy-transfer effect

**H3₀:** A validation-selected 2% review threshold preserves the intended review workload on the later temporal test period.

**H3₁:** A validation-selected 2% review threshold does not preserve the intended review workload on the later temporal test period.

RQ3 will primarily be evaluated through cost-ratio sensitivity analysis and break-even analysis rather than a separate omnibus hypothesis test.

## Research Aim

To quantify how predictive ranking quality and score-to-review decision policies jointly determine fraud capture, workload fidelity, and cost-sensitive outcomes under fixed review capacity and temporal shift.

## Research Objectives

1. Establish and freeze a reproducible, leakage-controlled temporal experimental protocol using the IEEE-CIS Fraud Detection dataset.
2. Compare LightGBM and CatBoost under identical data, features, tuning budgets, and evaluation conditions.
3. Evaluate unweighted and balanced class-weighted variants of each primary model family.
4. Measure fraud capture under fixed review-capacity scenarios using capacity-aware metrics.
5. Evaluate how validation-selected score thresholds transfer to later data in terms of realised workload and missed fraud.
6. Evaluate calibrated cost-aware review policies using scenario-based cost ratios and sensitivity analysis.
7. Quantify uncertainty using pre-specified confidence intervals, paired comparisons, and effect sizes.
8. Characterise false-negative behaviour across pre-declared transaction subgroups without causal interpretation.
9. Document ethical considerations, assumptions, limitations, and threats to validity.

## Core Experimental Factors

### Model family

- LightGBM
- CatBoost

A simple L2-regularised Logistic Regression model may be included only as a reference baseline if justified by the final methodology.

### Class weighting

- unweighted
- balanced weighting

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

These are experimental scenarios and must not be represented as real institutional review capacities unless supported by evidence.

## Primary and Secondary Outcomes

### Primary outcome

- Recall@2% review capacity

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

VALIDATION 1
  -> hyperparameter selection
  -> early stopping
  -> calibration fitting where required

VALIDATION 2
  -> threshold selection
  -> review-capacity policy selection
  -> cost-rule selection

OPTIONAL GAP
  -> label-maturity / temporal separation if justified

FINAL TEST
  -> one confirmatory evaluation
```

The exact day boundaries will be validated from the raw dataset and frozen in the research protocol before confirmatory experiments.

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

- paired bootstrap confidence intervals;
- exact McNemar testing where appropriate for paired fraud-capture differences;
- effect sizes;
- Holm correction for pre-specified confirmatory comparisons;
- equivalence testing for class weighting only if a justified smallest effect size of interest is defined.

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

The expected contribution is a reproducible empirical evaluation of:

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

The study does not claim a novel fraud-detection algorithm.

## Next Research Tasks

1. Resolve and document the University ethics route.
2. Complete the literature evidence matrix.
3. Verify the raw dataset profile and candidate temporal split.
4. Freeze `docs/protocol.md`.
5. Create the dataset manifest.
6. Implement leakage-safe data and split pipelines.
7. Run validation-only model development.
8. Lock the final decision policies.
9. Run the final test evaluation once.
