# Cost-Sensitive Fraud Detection Research

MSc Data Science research project on cost-sensitive transaction fraud detection under severe class imbalance and constrained human-review capacity.

**University:** University of Greenwich  
**Module:** COMP1885 Individual Project  
**Research type:** Quantitative, non-participant, secondary-data experimental study

## Working Research Title

**Cost-Sensitive Transaction Fraud Detection Under Severe Class Imbalance and Constrained Human-Review Capacity: A Comparative Study of Model and Threshold Strategies**

## Main Research Question

> **How do model choice and decision-threshold strategy affect fraud-detection effectiveness under severe class imbalance and constrained human-review capacity?**

### Supporting Research Questions

1. **How do LightGBM and CatBoost compare in fraud-detection performance under a consistent, time-aware experimental setting?**
2. **How do alternative decision-threshold strategies affect fraud recall, precision, false-negative risk, and review workload?**
3. **How does constrained human-review capacity influence the effectiveness and operational trade-offs of different fraud-detection policies?**

## Hypotheses

### H0 — Null Hypothesis

> **There is no statistically significant difference in fraud-detection effectiveness between the evaluated model and decision-threshold strategies under constrained human-review capacity.**

### H1 — Alternative Hypothesis

> **There is a statistically significant difference in fraud-detection effectiveness between at least some of the evaluated model and decision-threshold strategies under constrained human-review capacity.**

## Dataset

This study uses the **IEEE-CIS Fraud Detection** dataset.

Official source:

https://www.kaggle.com/competitions/ieee-fraud-detection/data

The labelled transaction and identity training data will be used to create a reproducible time-aware Train → Validation → Final Test protocol for the MSc experiments.

See [Dataset documentation](docs/dataset.md).

## Research Scope

The academic contribution focuses on:

- severe class imbalance;
- LightGBM and CatBoost comparison;
- class-weight / cost-sensitive treatment;
- decision-threshold strategies;
- constrained human-review capacity;
- false-negative analysis;
- statistical uncertainty;
- reproducible temporal evaluation.

FastAPI, Docker, Kafka, Airflow, monitoring, CI/CD, Terraform, and AWS are supporting engineering topics from the earlier project and are not the primary COMP1885 research contribution.

## Human-in-the-Loop Interpretation

The intended real-world interpretation is a Human-in-the-Loop (HITL) fraud-review workflow:

```text
Transaction
    ↓
ML fraud-risk score
    ↓
Threshold / capacity policy
    ↓
Prioritised review queue
    ↓
Human verification
    ↓
Final operational action
```

The research itself does not recruit or evaluate real human analysts. Human review is represented as an operational capacity constraint.

## Research Documentation

- [Research foundation](docs/research-foundation.md)
- [Literature review foundation](docs/literature-review.md)
- [Dataset documentation](docs/dataset.md)

Planned next:

- literature evidence matrix;
- methodology;
- ethics;
- threats to validity;
- provenance;
- reproducible experiment and result documentation.

## Relationship to the Earlier Project

This project builds on selected implementation experience from:

`chathuranga-sudusinghe/fraud-detection-ml-system`

The earlier repository is a production-oriented ML engineering project. This repository is a separate MSc research project. Final research claims will be based on a new frozen experimental protocol and newly generated evidence.

## Status

**Research foundation and dataset source established.**

The next stage is to build the literature evidence matrix and then freeze the detailed methodology before final experiments are run.
