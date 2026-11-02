# COMP1885 Research Foundation

## Project

**University:** University of Greenwich\
**Programme:** MSc Data Science and Its Applications\
**Module:** COMP1885 Individual Project\
**Repository:** `fraud-detection-review-capacity`\
**Research type:** Quantitative, non-participant, secondary-data experimental study

This document states why the study is needed and what it contributes: the research problem, aim, objectives and contribution. The title, the research questions (RQ1–RQ3), the confirmatory hypotheses (H1–H3) and the methods are specified in the [research protocol](protocol.md); the reasons for each design choice are in the [Decision Records](decisions/README.md).

## Research Problem

Transaction fraud detection is a low-prevalence, imbalanced binary-classification problem in which missed fraud can be costly, while excessive false alerts can create unnecessary manual-review workload and customer friction.

In operational settings, review resources are limited. Therefore, a useful fraud-detection system must do more than produce a strong global discrimination score: it must prioritise transactions effectively under a fixed review budget, and the decision policy chosen during validation should remain operationally stable when applied to later data.

Temporal change can alter score distributions, fraud prevalence, feature distributions, and realised alert volume. This creates a research problem at the boundary between predictive ranking and operational decision policy.

In the IEEE-CIS (IEEE Computational Intelligence Society) Fraud Detection development data, fraud prevalence is about 3.5 %, an imbalance ratio (legitimate : fraud) of about 28 : 1 **[D]** ([dataset documentation](dataset.md)). This is described as **imbalanced / low-prevalence**, not "severely" imbalanced: that label is conventionally reserved for ratios of about 100 : 1 and above [REF-002, REF-001]. The practical difficulty comes from low prevalence combined with temporal drift and a review budget that, at 2 % capacity, is smaller than the daily number of frauds ([DR-002](decisions/DR-002-review-capacity-definition.md)).

### Operational setting

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

## Research Aim

To quantify how predictive ranking quality and score-to-review decision policies jointly determine fraud capture, workload fidelity, and cost-sensitive outcomes under fixed review capacity and temporal shift.

## Research Objectives

1. Establish and freeze a reproducible, leakage-controlled temporal experimental protocol using the IEEE-CIS Fraud Detection dataset.
2. Compare tree-based (LightGBM, CatBoost) and neural (ResNet-style multilayer perceptron, MLP; Feature Tokenizer + Transformer, FT-Transformer) model families under one common procedure: identical data, model inputs, class weights, tuning method and budget, and evaluation conditions, with family-specific input representations ([DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md) §3).
3. Evaluate unweighted and balanced class-weighted variants of each model.
4. Measure fraud capture under fixed review-capacity scenarios using capacity-aware metrics.
5. Evaluate how validation-selected score thresholds transfer to later data, separating alert-rate fidelity from fixed daily-budget fidelity.
6. Evaluate calibrated cost-aware review policies across the capacity grid, using scenario-based cost ratios and break-even analysis.
7. Quantify uncertainty using pre-specified, cluster- or day-block-aware confidence intervals, paired comparisons, and effect sizes.
8. Characterise false-negative behaviour across pre-declared transaction subgroups without causal interpretation.
9. Document ethical considerations, assumptions, limitations, and threats to validity.

## Research Contribution

The study proposes no new fraud-detection algorithm, decision method, cost framework or temporal protocol. Its contribution is an **integrated, pre-specified confirmatory evaluation** of the full path from model family to operational review decision, on one real chronological benchmark (IEEE-CIS) and under fixed daily review budgets across the declared capacity scenarios ([protocol: Review Capacity](protocol.md#review-capacity)):

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

The structured literature review ([`literature-review.md`](literature-review.md), Sections 2–6 and 8; [`literature-evidence-matrix.md`](literature-evidence-matrix.md), Section 3) finds each element in prior work, but no study that combines a fixed daily review budget, a pre-specified test of threshold transfer, a tree-vs-neural comparison under one common procedure and dependence-aware inference on real chronological data.

**No novelty is claimed for any single element**, each of which has established prior work:
- fraud-detection algorithms, and the tree-vs-neural comparison on general tabular benchmarks [REF-087, REF-089, REF-090];
- temporal evaluation, including chronological evaluation on this dataset [REF-023, REF-041, REF-058];
- capacity-constrained evaluation and top-K review (K: the fixed number of reviews per relative day), including daily alert budgets [REF-006, REF-004, REF-005, REF-046, REF-084, REF-086], and capacity-motivated ranking metrics on IEEE-CIS [REF-099];
- validation-selected thresholds applied unchanged to later data [REF-058, REF-079, REF-080];
- temporal transfer of review policies under capacity, including a separate policy-selection period [REF-078];
- cost-sensitive thresholding and calibrated cost-threshold policies [REF-008, REF-039, REF-078];
- binding-versus-slack capacity effects and cost-ratio sweeps [REF-078, REF-080];
- capacity-aware threshold selection or case allocation [REF-081, REF-082, REF-083].

**The most distinctive component.** The closest study, Zhang et al. [REF-078], transfers a historical review *rate* (which preserves the review fraction by construction) and evaluates workload rules against a retrospective oracle; it notes that attributing workload displacement to its sources would require further controlled comparison. In the studies reviewed, the alert rate of a *static* validation-selected score threshold is not formally tested against its intended target, and deviations from a fixed daily review budget are not separated into alert-rate drift and volume variation.

The contribution therefore has these components, in order of distinctiveness:

1. **RQ2 / H3 (confirmatory).** A pre-specified equivalence test of whether the alert rate of a validation-selected score threshold stays within a pre-specified margin of its target on later data ([protocol: H3](protocol.md#h3--alert-rate-transfer-fidelity)), with a descriptive tree-vs-neural comparison of threshold transfer.
2. **Daily-budget fidelity (descriptive, separate from H3).** Threshold-generated demand compared with the fixed daily review budget K (overflow, unused capacity, load ratio), with the workload deviation separated into an alert-rate component and a volume-driven component ([DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md), [DR-009](decisions/DR-009-overflow-handling-for-threshold-policies.md)).
3. **RQ1 / H1, H2 (confirmatory).** A pre-specified, dependence-aware comparison of tree-based and neural model families, and of class weighting, at fixed daily capacity on chronological data, under one common training procedure ([DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md), [DR-006](decisions/DR-006-bootstrap-inference-strategy.md), [DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md)). Fraud-specific evidence on this family question is limited and mostly non-temporal [REF-098, REF-099, REF-100, REF-101]; the study contributes a capacity-based, temporally validated answer for these pipelines.
4. **RQ3 as a replication-plus-extension.** Zhang et al. [REF-078] show that a calibrated cost threshold and full-capacity review are largely compressed toward the same capacity ceiling when capacity is tight and diverge when capacity is slack. RQ3 re-examines this pattern under this study's fixed daily cap and summarises it as break-even and saturation cost ratios (ρ_BE, ρ_sat; [DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md)).

The conclusions are bounded by one benchmark, one validation-to-test transfer, the four models studied and simulated capacity and cost assumptions.

## References

Full entries and verification status are in [`references.md`](references.md).

REF-001, REF-002, REF-004, REF-005, REF-006, REF-008, REF-023, REF-039, REF-041, REF-046, REF-058, REF-078, REF-079, REF-080, REF-081\*, REF-082, REF-083, REF-084, REF-086, REF-087, REF-089, REF-090, REF-098, REF-099, REF-100, REF-101.

\* Partially verified.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| DR-NNN | Decision Record NNN in `decisions/` |
| FT-Transformer | Feature Tokenizer + Transformer (a neural model for tabular data) |
| H1, H2, H3 | Confirmatory hypotheses: model-family effect; class-weighting effect; alert-rate transfer fidelity (stated in the protocol) |
| IEEE-CIS | IEEE Computational Intelligence Society (organiser of the Fraud Detection dataset) |
| K | Fixed daily review budget (reviews per relative day) |
| MLP | Multilayer perceptron |
| REF-NNN | Reference identifier in `references.md` |
| Relative day | `TransactionDT` (the dataset's time offset in seconds from an undisclosed reference point) divided by 86,400 seconds and rounded down |
| ρ_BE, ρ_sat | Break-even and saturation miss-to-review cost ratios (RQ3) |
| RQ1–RQ3 | Research questions 1–3 (stated in the protocol) |
