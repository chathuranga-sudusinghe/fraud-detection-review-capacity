# Fraud Detection Under Limited Review Capacity

MSc Data Science research project on temporally validated fraud-alert prioritisation under constrained review capacity.

**University:** University of Greenwich  
**Module:** COMP1885 Individual Project  
**Research type:** Quantitative, non-participant, secondary-data experimental study

## Working Research Title

**Fraud Detection Under Limited Review Capacity: A Temporally Validated Study of Gradient-Boosting Models and Decision Policies**

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

RQ3 is primarily estimation and sensitivity analysis rather than a separate significance-testing exercise.

## Dataset

This study uses the **IEEE-CIS Fraud Detection** dataset.

Official source:

https://www.kaggle.com/competitions/ieee-fraud-detection/data

The labelled transaction and identity training data will be used to construct a reproducible time-aware experimental protocol.

See [Dataset documentation](docs/dataset.md).

## Research Focus

The research focuses on:

- temporal generalisation;
- LightGBM and CatBoost;
- class weighting;
- fixed review capacity;
- score-to-review decision policies;
- Recall@Capacity and Precision@Capacity;
- workload fidelity;
- cost-sensitive decision analysis;
- probability calibration where required;
- statistical uncertainty;
- false-negative analysis.

## Operational Interpretation

The research models a capacity-constrained alert-triage workflow:

```text
Transaction
    ↓
ML fraud-risk score
    ↓
Decision policy
    ↓
Prioritised review queue
    ↓
Manual verification
    ↓
Final operational action
```

Human review is represented as an exogenous capacity constraint. The research does not recruit or evaluate human participants.

## Research Documentation

- [Research foundation](docs/research-foundation.md)
- [Research protocol](docs/protocol.md)
- [Literature review foundation](docs/literature-review.md)
- [Dataset documentation](docs/dataset.md)

Planned next:

- ethics;
- literature evidence matrix;
- threats to validity;
- dataset manifest;
- split-generation code;
- model and policy experiments;
- statistical analysis;
- final results.

## Status

**Research direction revised and protocol established.**

The next stage is to resolve the ethics route, complete the evidence matrix, and validate the dataset/split protocol before any confirmatory experiments are run.
