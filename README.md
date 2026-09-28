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

The supporting research questions are numbered RQ1–RQ3.

### RQ1

> **How do LightGBM and CatBoost, with and without class weighting, differ in fraud capture at fixed review capacity on a later temporal test period?**

### RQ2

> **When a score threshold selected on validation data is transferred to a later temporal period, how well does it preserve the intended alert rate, and how does the resulting daily review workload compare with the fixed review budget?**

Revised per [DR-003](docs/decisions/DR-003-rq2-h3-policy-transfer-fidelity.md).

### RQ3

> **Under different assumed miss-to-review cost ratios, when does a calibrated cost-aware review policy outperform always filling the available review capacity?**

## Confirmatory Hypotheses

The confirmatory hypotheses are H1–H3, worded exactly as in the [research protocol](docs/protocol.md). A subscript ₀ marks the null hypothesis and ₁ the alternative. Recall@2% is the share of Test-period frauds that are reviewed when 2 % of transactions can be reviewed each day (daily top-K review: the K highest-scoring transactions are reviewed, where K is the fixed daily review budget).

Symbols used below (full definitions and equations are in the protocol):

- `δ_M` = model-family main effect: the CatBoost-minus-LightGBM difference in Recall@2%, averaged over the unweighted and balanced-weighting conditions;
- `δ_W,f` = weighting effect in model family f: balanced-minus-unweighted Recall@2%;
- `r` = realised Test alert rate of the validation-selected 2 % threshold. For confirmatory H3 this is the threshold of one rule-selected H3 configuration, chosen before Test access ([DR-010](docs/decisions/DR-010-h3-confirmatory-configuration.md));
- `Δ` = H3 equivalence tolerance on r (0.004, i.e. ±20 % of the 2 % target).

### H1 — Model-family effect

**H1₀:** Averaged over the unweighted and balanced-weighting conditions, CatBoost and LightGBM do not differ in Recall@2% on the Test period (δ_M = 0).

**H1₁:** Averaged over the unweighted and balanced-weighting conditions, CatBoost and LightGBM differ in Recall@2% on the Test period (δ_M ≠ 0).

### H2 — Class-weighting effect

**H2₀:** For at least one model family (LightGBM or CatBoost), balanced class weighting changes Recall@2% by 2 percentage points or more relative to the corresponding unweighted model (|δ_W,f| ≥ 0.02 for at least one f).

**H2₁:** For both LightGBM and CatBoost, balanced class weighting changes Recall@2% by less than 2 percentage points relative to the corresponding unweighted model (|δ_W,f| < 0.02 for both f).

### H3 — Alert-rate transfer fidelity

**H3₀:** The realised alert rate under the validation-selected 2% threshold lies outside the pre-specified equivalence band around the 2% target (|r − 0.02| ≥ Δ, where Δ = 0.004, i.e. ±20% of the target).

**H3₁:** The realised alert rate under the validation-selected 2% threshold lies within the pre-specified equivalence band around the 2% target (|r − 0.02| < Δ, where Δ = 0.004, i.e. ±20% of the target).

H1 is a two-sided difference test. H2 and H3 are equivalence hypotheses tested with Two One-Sided Tests (TOST). The Smallest Effect Size Of Interest (SESOI) for H1 and H2 is 2 percentage points of Recall@2% ([DR-005](docs/decisions/DR-005-sesoi-and-h1-h2-inference.md)); Δ is justified in [DR-003](docs/decisions/DR-003-rq2-h3-policy-transfer-fidelity.md). H3 is evaluated confirmatorily on one model × weighting configuration, selected on Validation 2 by a frozen rule before Test access ([DR-010](docs/decisions/DR-010-h3-confirmatory-configuration.md)); the other three are reported descriptively. Uncertainty for H1/H2 uses the two-way bootstrap of [DR-006](docs/decisions/DR-006-bootstrap-inference-strategy.md).

RQ3 is primarily estimation and sensitivity analysis rather than a separate significance-testing exercise. Its cost-ratio grid and break-even rule are defined in [DR-007](docs/decisions/DR-007-rq3-cost-ratio-and-break-even.md).

## Dataset

This study uses the **IEEE-CIS Fraud Detection** dataset (IEEE-CIS = IEEE Computational Intelligence Society, the competition co-organiser).

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
- Recall@Capacity and Precision@Capacity (share of frauds reviewed, and share of reviewed transactions that are fraud, at a fixed review capacity);
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
ML (machine learning) fraud-risk score
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
- [Feature catalog](docs/feature-catalog.md)
- [Methodology Decision Records](docs/decisions/README.md)
- [Reference register](docs/references.md)

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

The ethics and governance status is documented in the [protocol](docs/protocol.md#ethics): topic approval was received, and no separate explicit ethics-route determination was provided despite the request; no formal ethics approval or exemption is claimed. The next stage is to complete the evidence matrix and validate the dataset/split protocol before any confirmatory experiments are run.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| Δ | H3 equivalence tolerance on r (0.004 = ±20 % of the 2 % target) |
| δ_M | Model-family main effect in Recall@2% (CatBoost minus LightGBM, averaged over weighting) |
| δ_W,f | Weighting effect in model family f (balanced minus unweighted Recall@2%) |
| f | A model family (LightGBM or CatBoost) |
| H1, H2, H3 | Confirmatory hypotheses; subscript ₀ = null hypothesis, ₁ = alternative |
| K | Fixed daily review budget (reviews per day) |
| IEEE-CIS | IEEE Computational Intelligence Society |
| ML | Machine learning |
| Precision@Capacity | Share of reviewed transactions that are fraud at a fixed review capacity |
| r | Realised Test alert rate of the validation-selected 2 % threshold (confirmatory H3: the rule-selected H3 configuration only) |
| Recall@2%, Recall@Capacity | Share of frauds reviewed at 2 % (a fixed) review capacity |
| RQ1–RQ3 | Research questions 1–3 |
| SESOI | Smallest Effect Size Of Interest |
| TOST | Two One-Sided Tests |
