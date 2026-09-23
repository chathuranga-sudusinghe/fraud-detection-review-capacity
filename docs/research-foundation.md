# COMP1885 Research Foundation

## Project

**Programme:** MSc Data Science, University of Greenwich  
**Module:** COMP1885 Individual Project  
**Repository:** `cost-sensitive-fraud-detection-research`  
**Research type:** Quantitative, non-participant, secondary-data experimental study

## Working Research Title

**Cost-Sensitive Transaction Fraud Detection Under Severe Class Imbalance and Constrained Human-Review Capacity: A Comparative Study of Model and Threshold Strategies**

## Research Problem

Transaction fraud detection is a severely imbalanced classification problem in which fraudulent transactions are rare and false negatives can have serious consequences. However, lowering a decision threshold to capture more fraud can increase false positives and the number of cases requiring manual investigation.

In realistic fraud-review workflows, analysts have limited capacity. Therefore, predictive performance alone is not sufficient: the operational policy that converts model scores into review decisions also matters.

This project investigates how model choice and decision-threshold strategy interact under severe class imbalance and constrained human-review capacity.

The intended real-world interpretation is a Human-in-the-Loop (HITL) decision-support workflow in which a Machine Learning (ML) model produces a fraud-risk score and prioritises transactions for review, while the final operational decision remains with a human reviewer. The research does not recruit human participants or evaluate real analysts.

## Main Research Question

> **How do model choice and decision-threshold strategy affect fraud-detection effectiveness under severe class imbalance and constrained human-review capacity?**

## Supporting Research Questions

### RQ1

> **How do LightGBM and CatBoost compare in fraud-detection performance under a consistent, time-aware experimental setting?**

### RQ2

> **How do alternative decision-threshold strategies affect fraud recall, precision, false-negative risk, and review workload?**

### RQ3

> **How does constrained human-review capacity influence the effectiveness and operational trade-offs of different fraud-detection policies?**

## Hypotheses

### H0 — Null Hypothesis

> **There is no statistically significant difference in fraud-detection effectiveness between the evaluated model and decision-threshold strategies under constrained human-review capacity.**

### H1 — Alternative Hypothesis

> **There is a statistically significant difference in fraud-detection effectiveness between at least some of the evaluated model and decision-threshold strategies under constrained human-review capacity.**

## Research Aim

To investigate how model choice and cost-sensitive decision-threshold strategies influence fraud-detection effectiveness under severe class imbalance and constrained human-review capacity.

## Research Objectives

1. Establish a reproducible, time-aware experimental dataset and evaluation protocol using the IEEE-CIS Fraud Detection dataset.
2. Compare LightGBM and CatBoost under a consistent preprocessing, feature-engineering, and data-partitioning protocol.
3. Evaluate selected class-imbalance and cost-sensitive learning strategies.
4. Compare alternative decision-threshold strategies under explicit review-capacity constraints.
5. Quantify trade-offs using fraud recall, precision, false negatives, false positives, PR-AUC, ROC-AUC, review workload, and cost-sensitive outcomes.
6. Evaluate statistical uncertainty and determine whether observed differences are statistically meaningful.
7. Analyse the characteristics of fraud cases missed under constrained operating policies.
8. Discuss ethical, operational, and validity limitations of model-assisted fraud-review decision support.

## Core Research Scope

The academic contribution is centred on:

- severe class imbalance;
- LightGBM and CatBoost comparison;
- class-weight / cost-sensitive treatment;
- decision-threshold strategies;
- constrained human-review capacity;
- false-negative analysis;
- operational trade-offs;
- statistical evaluation;
- reproducibility.

Candidate review-capacity scenarios will be defined explicitly during methodology design, for example:

- 1% of transactions;
- 2% of transactions;
- 5% of transactions;
- 10% of transactions.

These values are research scenarios and must not be presented as real bank analyst capacities unless supported by evidence.

## Human-in-the-Loop Interpretation

The intended production concept is:

```text
Transaction
    ↓
ML fraud-risk model
    ↓
Fraud probability / risk score
    ↓
Decision-threshold or capacity policy
    ↓
Selected suspicious transactions
    ↓
Human review queue
    ↓
Human verification
    ↓
Final operational action
```

The model is therefore treated as **decision support**, not as an unquestionable autonomous final decision-maker.

The research itself remains non-participant research: no real fraud analysts, interviews, questionnaires, usability studies, or human-labelling exercises are required for the core experimental design.

## Experimental Principle

The final MSc results must follow a strict separation:

```text
TRAIN
  -> fit models

VALIDATION
  -> select hyperparameters
  -> select class-weight strategy
  -> select threshold / review-capacity policy

TEST
  -> final unbiased evaluation only
```

The test set must not be used to choose the final model, threshold, class weight, or operating policy.

## Academic Contribution

The primary research contribution is the controlled empirical study of:

```text
model choice
    ×
imbalance treatment
    ×
decision-threshold strategy
    ×
review-capacity constraint
    ↓
fraud-detection effectiveness and operational trade-offs
```

## Key Methodological Gaps to Address

Before final experiments are run, the research repository must add or confirm:

- a frozen dataset version;
- reproducible time-aware split generation;
- a fixed feature-engineering contract;
- validation-only model and policy selection;
- untouched final test evaluation;
- explicit review-capacity definitions;
- an explicit cost model or cost-ratio sensitivity analysis;
- confidence intervals and paired statistical comparison;
- final false-negative subgroup analysis;
- complete experiment manifests and result documentation.

## Planned Academic Documentation

The research repository should eventually include:

```text
docs/
├── research-foundation.md
├── literature-review.md
├── methodology.md
├── ethics.md
├── threats-to-validity.md
└── results/
```

## Status

**Research foundation established.**

The next stage is to develop the literature review and methodology before implementation begins.
