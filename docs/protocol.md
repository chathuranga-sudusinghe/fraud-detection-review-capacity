# COMP1885 Research Protocol

## Status

**Working protocol — not yet frozen.**

This file defines the intended confirmatory research design. Exact dataset split boundaries, statistical effect-size thresholds, and final cost scenarios must be validated before the protocol is tagged as final.

## Research Title

**Fraud Detection Under Limited Review Capacity: A Temporally Validated Study of Gradient-Boosting Models and Decision Policies**

## Main Research Question

> **Under constrained daily review capacity, how do model family, class weighting, and score-to-review decision policies affect fraud capture, review workload, and cost-sensitive outcomes on temporally unseen transaction data?**

## Supporting Research Questions

### RQ1 — Model and weighting

> **How do LightGBM and CatBoost, with and without class weighting, differ in fraud capture at fixed review capacity on a later temporal test period?**

### RQ2 — Policy transfer

> **When score thresholds selected on validation data are applied to a later period, how accurately do they preserve the intended review workload, and what effect does workload drift have on missed fraud?**

### RQ3 — Cost-aware capacity use

> **Under different assumed miss-to-review cost ratios, when does a calibrated cost-aware review policy outperform always filling the available review capacity?**

## Confirmatory Hypotheses

### H1 — Model-family effect

- **H1₀:** ΔRecall@2% (CatBoost − LightGBM) = 0.
- **H1₁:** ΔRecall@2% (CatBoost − LightGBM) ≠ 0.

### H2 — Class-weighting effect

- **H2₀:** ΔRecall@2% (weighted − unweighted) = 0.
- **H2₁:** ΔRecall@2% (weighted − unweighted) ≠ 0.

An equivalence analysis may additionally be used if a smallest effect size of interest is justified before final testing.

### H3 — Policy-transfer effect

- **H3₀:** A validation-selected 2% review threshold preserves the intended review workload on the later temporal test period.
- **H3₁:** A validation-selected 2% review threshold does not preserve the intended review workload on the later temporal test period.

RQ3 will be analysed through estimation, sensitivity analysis, and break-even cost ratios.

## Dataset

**IEEE-CIS Fraud Detection**

Official source:

https://www.kaggle.com/competitions/ieee-fraud-detection/data

Primary labelled source files:

- `train_transaction.csv`
- `train_identity.csv`

Join key:

- `TransactionID`

Temporal ordering variable:

- `TransactionDT`

The competition test data are not used as the dissertation's primary labelled test set because the target labels are unavailable.

## Experimental Split Structure

The final split will be chronological and day-aligned where practical.

Planned structure:

```text
Train
  -> model fitting

Validation 1
  -> hyperparameter tuning
  -> early stopping
  -> calibration fitting

Validation 2
  -> threshold selection
  -> review-capacity policy selection
  -> cost-rule selection

Optional temporal gap
  -> included only if justified

Final Test
  -> one confirmatory evaluation
```

Exact day boundaries must be calculated from the raw data and justified before protocol freeze.

## Models

Primary:

- LightGBM
- CatBoost

Optional reference:

- L2 Logistic Regression

No additional model family will be added without a clear research justification.

## Class-Imbalance Treatment

Primary comparison:

- no class weighting;
- balanced class weighting.

Sampling-based approaches such as SMOTE, random over-sampling, and random under-sampling are outside the core design unless the literature review identifies a compelling reason to include them.

## Review Capacity

Candidate capacity scenarios:

- 1%
- 2% — primary
- 5%
- 10%

The final implementation will convert these scenarios into fixed review limits derived from validation-period transaction volume.

Primary metric:

- Recall@2% review capacity

Secondary capacity metrics:

- Precision@Capacity
- false negatives at capacity
- frauds per 1,000 reviews
- lift
- utilisation
- workload overflow / capacity violation
- normalised recall relative to an oracle ceiling

## Decision Policies

### R0 — Default threshold

Probability threshold = 0.5.

Purpose: naive reference policy.

### R1 — F1-optimal threshold

Selected using Validation 2 only.

Purpose: common capacity-agnostic threshold reference.

### R2 — Capacity-quantile threshold

Selected on Validation 2 to target the intended review capacity.

Purpose: test whether a static validation-selected threshold transfers operationally to later data.

### R3 — Daily top-K

Review the highest-scoring K transactions within each evaluation period.

Purpose: ranking-quality benchmark under full capacity utilisation.

### R4 — Calibrated cost-aware capped policy

Use calibrated probabilities and an explicit cost rule, subject to the review-capacity limit.

Purpose: determine when leaving capacity unused is economically preferable under stated cost assumptions.

## Calibration

Calibration is a supporting component, not a separate research question.

Candidate method:

- Platt scaling on Validation 1.

Secondary comparison, if justified:

- isotonic regression.

Candidate metrics:

- Brier score;
- log loss;
- calibration intercept;
- calibration slope;
- reliability diagram.

## Cost Model

The analysis will separate:

- review effort;
- additional false-alert/customer-friction cost;
- missed-fraud cost.

All cost values are scenario assumptions.

The final protocol will define:

- normalised review cost;
- miss-to-review cost ratios;
- optional false-alert friction ratios;
- break-even analyses.

No assumed value will be presented as an actual institutional banking cost.

## Statistical Plan

Primary confirmatory comparison:

- Recall@2%.

Candidate methods:

- paired bootstrap 95% confidence intervals;
- exact McNemar test for paired fraud-capture differences where applicable;
- effect-size reporting;
- Holm correction across pre-specified confirmatory tests.

Class-weighting equivalence testing may be added only after a defensible smallest effect size of interest has been declared.

Secondary outcomes will primarily use estimation with confidence intervals rather than additional hypothesis tests.

## False-Negative Analysis

Pre-declared exploratory slices may include:

- identity information present / absent;
- product category;
- card characteristics;
- device-information status;
- transaction-amount bands;
- temporal sub-periods;
- score bands.

The analysis will report:

- fraud count per slice;
- miss rate;
- uncertainty interval;
- share of total false negatives.

No causal claim will be made from subgroup associations.

## Leakage Controls

Before the protocol is frozen:

- all dataset partitions must be mutually exclusive;
- preprocessing and encoders must be fitted on permitted training data only;
- threshold and policy selection must use validation data only;
- final test labels must not influence feature, model, or policy design;
- no future-looking aggregate may be used;
- all test access will be logged.

## Ethics

The study uses secondary data and does not recruit human participants.

Before confirmatory experiments, the University of Greenwich ethics route must be confirmed and documented because the topic concerns fraud and potentially sensitive/criminal subject matter.

## Protocol Freeze

The protocol becomes confirmatory only after:

1. literature evidence review is complete enough to justify the design;
2. ethics route is confirmed;
3. raw dataset statistics are independently reproduced;
4. split boundaries are justified;
5. primary metric and capacity are fixed;
6. hypotheses and statistical tests are fixed;
7. cost scenarios are declared.

After that point, the protocol should be tagged in Git before final-test evaluation.
