# Literature Review Foundation

## Project Context

**Programme:** MSc Data Science, University of Greenwich  
**Module:** COMP1885 Individual Project  
**Research repository:** `fraud-detection-review-capacity`

**Working title:**  
**Fraud Detection Under Limited Review Capacity: A Temporally Validated Study of Gradient-Boosting Models and Decision Policies**

This document establishes the literature-review structure for the project. It is a research working document and will be expanded into the final dissertation literature review after the evidence base has been systematically collected and critically synthesised.

---

## 1. Literature Review Purpose

The literature review will establish:

1. why transaction fraud detection is a difficult imbalanced-classification problem;
2. why accuracy and ROC-AUC (Receiver Operating Characteristic Area Under the Curve) alone are insufficient for operational fraud detection;
3. how LightGBM, CatBoost, and related gradient-boosting methods have been used for fraud detection;
4. how class weighting and cost-sensitive learning address asymmetric fraud-detection errors;
5. why decision-threshold selection is a separate problem from model training;
6. how limited review capacity changes the meaning of an effective fraud-detection policy;
7. how false negatives and false positives create different operational risks;
8. how model-assisted human review can be framed as Human-in-the-Loop (HITL) decision support;
9. why temporal validation and statistical uncertainty matter in fraud-detection research;
10. what research gap remains for the present COMP1885 study.

---

## 2. Main Research Question

> **Under constrained daily review capacity, how do model family, class weighting, and score-to-review decision policies affect fraud capture, review workload, and cost-sensitive outcomes on temporally unseen transaction data?**

### Supporting Questions

The supporting research questions are numbered RQ1–RQ3.

**RQ1**

> How do LightGBM and CatBoost, with and without class weighting, differ in fraud capture at fixed review capacity on a later temporal test period?

**RQ2**

> When score thresholds selected on validation data are applied to a later period, how accurately do they preserve the intended review workload, and what effect does workload drift have on missed fraud?

**RQ3**

> Under different assumed miss-to-review cost ratios, when does a calibrated cost-aware review policy outperform always filling the available review capacity?

The literature review must provide the theoretical and empirical basis for these questions rather than merely summarising unrelated fraud-detection papers.

---

## 3. Planned Literature Review Themes

### 3.1 Transaction Fraud Detection as an Imbalanced Classification Problem

Key issues to review:

- rarity of fraudulent transactions;
- misleading behaviour of raw accuracy under class imbalance;
- importance of minority-class detection;
- precision–recall trade-offs;
- asymmetric consequences of false negatives and false positives;
- non-stationary and evolving fraud patterns.

Expected argument:

> Fraud detection is not a conventional balanced binary-classification problem. Evaluation and decision policies must account for minority-class rarity and unequal error consequences.

---

### 3.2 Evaluation Metrics for Class Imbalance

Metrics to review critically:

- ROC-AUC;
- PR-AUC (Precision–Recall Area Under the Curve) / Average Precision;
- precision;
- recall / sensitivity;
- F1-score (harmonic mean of precision and recall);
- false-negative rate;
- false-positive rate;
- confusion-matrix counts;
- recall at review capacity;
- fraud captured per fixed review budget.

Expected argument:

> Threshold-independent discrimination metrics are useful but do not fully represent the operational consequences of a deployed fraud policy. Threshold-dependent and capacity-aware measures are required.

---

### 3.3 Gradient-Boosting Models for Fraud Detection

The primary model families in this study are:

- LightGBM;
- CatBoost.

Literature should examine:

- suitability for high-dimensional tabular data;
- handling of non-linear feature interactions;
- categorical-feature treatment;
- class weighting;
- ranking/discrimination behaviour;
- probability outputs;
- computational efficiency;
- limitations under temporal distribution shift.

XGBoost and other models may appear in the literature as comparison context, but this MSc project will not become a broad model leaderboard.

---

### 3.4 Class-Imbalance Handling and Cost-Sensitive Learning

Approaches to review:

- class weighting;
- `scale_pos_weight` (a library parameter that up-weights the positive, fraud, class);
- weighted loss functions;
- under-sampling;
- over-sampling;
- SMOTE (Synthetic Minority Over-sampling Technique) and synthetic minority generation;
- threshold adjustment;
- cost-sensitive decision rules.

The review must distinguish:

```text
training-time imbalance treatment
            from
post-training decision-threshold selection
```

Expected argument:

> Improving minority-class sensitivity during model fitting and choosing an operational threshold are related but separate decisions.

---

### 3.5 Decision-Threshold Optimisation

Threshold-selection literature should cover:

- default threshold (for example 0.5);
- F1-optimised thresholds;
- recall-constrained thresholds;
- precision-constrained thresholds;
- alert-rate-constrained thresholds;
- cost-minimising thresholds;
- validation-only threshold selection;
- risks of selecting thresholds using test data.

Expected argument:

> The model determines a score or ranking, while the threshold determines the operational decision policy. A strong model can still produce an unsuitable operating policy if its threshold is poorly chosen.

---

### 3.6 Review-Capacity-Constrained Fraud Detection

This is a central theme for the present study.

Literature should be searched for:

- alert budgets;
- investigation budgets;
- top-k fraud review;
- limited analyst capacity;
- queue prioritisation;
- constrained decision making;
- recall at fixed alert rate;
- precision at fixed review capacity;
- operational risk controls.

The project will model human-review capacity computationally rather than recruit human participants.

Candidate experimental capacities may include:

- 1%;
- 2%;
- 5%;
- 10%

of transactions, subject to final methodology justification.

Expected argument:

> In operational settings, a model may generate more alerts than investigators can review. Therefore, fraud-detection effectiveness should also be evaluated under explicit capacity constraints.

---

### 3.7 Cost-Sensitive Fraud-Detection Policies

The study intends to distinguish capacity constraints from explicit cost sensitivity.

A generic cost model may take the form:

```text
Total Cost  =  C_miss × FN
            +  C_FP   × FP
            +  C_R    × N_rev
```

where:

- `C_miss` = assumed relative cost of a missed fraud (written C_FN in some sources);
- `FN` = number of false negatives (frauds not reviewed);
- `C_FP` = assumed relative cost of unnecessary intervention (a false positive), charged in addition to the review cost C_R;
- `FP` = number of false positives (legitimate transactions reviewed);
- `C_R` = assumed review/investigation cost per reviewed transaction;
- `N_rev` = number of reviewed transactions.

Every review costs C_R; a reviewed legitimate transaction costs C_FP more, and each missed fraud costs C_miss.

Because the IEEE-CIS (IEEE Computational Intelligence Society) Fraud Detection dataset does not provide verified institutional investigation costs, the study should use **scenario-based cost ratios and sensitivity analysis**, not claim real bank cost values.

Expected argument:

> Cost-sensitive conclusions are conditional on the chosen cost assumptions and should therefore be evaluated across multiple plausible scenarios.

---

### 3.8 False-Negative Analysis

False negatives are especially important because they represent fraudulent transactions missed by the operating policy.

Literature and project analysis should consider:

- transaction characteristics associated with missed fraud;
- missing identity/device information;
- categorical subgroups;
- temporal differences;
- score distributions;
- changes in false-negative patterns as review capacity decreases.

Expected argument:

> Aggregate recall alone cannot explain which fraud cases remain undetected. Error analysis is required to understand the limitations of each policy.

---

### 3.9 Probability Calibration

The project should review the relationship between:

- discrimination;
- calibration;
- decision thresholds;
- cost-based decisions.

Candidate measures:

- Brier score;
- calibration curve;
- Expected Calibration Error (ECE), if justified.

Possible calibration methods:

- Platt scaling;
- isotonic regression.

Calibration will only be added to the final methodology if the literature review demonstrates that it materially strengthens the research question.

---

### 3.10 Temporal Validation and Distribution Shift

Fraud patterns change over time.

The review should therefore investigate:

- temporal train/validation/test splits;
- data leakage risk;
- concept drift;
- future-slice evaluation;
- differences between random and chronological validation;
- stability of model rankings over time.

Expected argument:

> A fraud model evaluated only with a random split may overstate real-world generalisation. A time-aware protocol is therefore preferable for the final MSc experiments.

---

### 3.11 Statistical Uncertainty and Model Comparison

The MSc study should strengthen point-estimate evaluation by reviewing:

- bootstrap confidence intervals;
- paired bootstrap comparisons;
- McNemar's test for paired classification disagreement, where appropriate;
- uncertainty in PR-AUC, recall, precision, and capacity-aware metrics;
- multiple-comparison considerations.

Expected argument:

> Small differences between model-policy combinations should not automatically be interpreted as meaningful without uncertainty analysis.

---

### 3.12 Human-in-the-Loop Fraud Decision Support

The real-world system concept is:

```text
ML (machine learning) model
   ↓
risk score
   ↓
threshold / capacity policy
   ↓
prioritised review queue
   ↓
human verification
   ↓
final action
```

Literature should cover:

- decision support rather than blind automation;
- automation bias;
- human oversight;
- selective escalation;
- operational accountability;
- false-positive harm;
- customer friction;
- explainability where relevant.

Important research boundary:

> This project does not conduct a participant-based HITL experiment. Human review is represented as an operational capacity constraint and system-design interpretation.

---

## 4. Initial Peer-Reviewed Evidence Base

The following sources form an initial evidence set and must be checked in full before being relied on in the final dissertation.

### 4.1 Systematic Review of Credit-Card Fraud Detection

**Credit card fraud detection in the era of disruptive technologies: A systematic review**  
Journal of King Saud University - Computer and Information Sciences, 2023.

Initial relevance:

- reviews fraud-detection research from 2015–2021;
- identifies class imbalance as a central challenge;
- discusses under-sampling, over-sampling, cost-sensitive learning, and machine-learning approaches;
- provides background evidence for the problem formulation.

---

### 4.2 Threshold Optimisation Under Imbalance

**Threshold optimization and random undersampling for imbalanced credit card data**  
Journal of Big Data, 2023.

Initial relevance:

- explicitly studies threshold optimisation;
- compares several threshold-selection approaches;
- evaluates threshold-dependent and threshold-independent metrics;
- supports the separation of classifier training from operating-threshold selection.

---

### 4.3 Recent Cost-Sensitive and Calibrated Fraud Detection

**TEMPLAR fraud: a verifier-grounded, calibrated and cost-sensitive framework for transaction fraud detection**  
Discover Data, 2026.

Initial relevance:

- frames fraud detection around class imbalance, temporal behaviour, and asymmetric error costs;
- includes calibration;
- applies cost-sensitive thresholding;
- reports temporal future-slice evaluation;
- reinforces the value of validation-selected operating policies.

This source is recent and particularly relevant, but its methods and assumptions must be critically compared with the narrower MSc research scope.

---

### 4.4 Recent Explainable Fraud-Detection Research

**Explainable ensemble machine learning for financial transaction fraud detection: Insights from XGBoost and deep neural networks**  
2026.

Initial relevance:

- considers extreme class imbalance;
- evaluates class weighting and threshold optimisation;
- explicitly selects thresholds using validation data before final evaluation;
- reinforces the use of precision, recall, F1, ROC-AUC, and PR-AUC rather than accuracy alone.

---

### 4.5 Recent Risk-Control-Oriented Research

**DISCO: Decoupling representation learning and risk control for reliable credit card fraud detection**  
Decision Support Systems, 2026.

Initial relevance:

- distinguishes predictive representation from downstream risk control;
- explicitly focuses on operational trade-offs;
- provides contemporary evidence that fraud detection should not be evaluated purely as a classifier-selection problem.

This study uses a different modelling approach from the present project and therefore serves primarily as conceptual and methodological context.

---

## 5. Search Strategy

The literature search should use combinations of terms such as:

```text
"transaction fraud detection" AND "class imbalance"
"credit card fraud detection" AND "cost sensitive"
"fraud detection" AND "decision threshold"
"fraud detection" AND "threshold optimization"
"fraud detection" AND "alert rate"
"fraud detection" AND "review capacity"
"fraud detection" AND "investigation budget"
"fraud detection" AND "human in the loop"
"fraud detection" AND "false negatives"
"fraud detection" AND "temporal validation"
"fraud detection" AND "probability calibration"
"fraud detection" AND "LightGBM"
"fraud detection" AND "CatBoost"
```

Preferred evidence sources:

- peer-reviewed journal articles;
- peer-reviewed conference papers;
- systematic reviews;
- authoritative methodological papers;
- highly relevant recent work.

Potential databases/search platforms:

- IEEE Xplore;
- ScienceDirect;
- SpringerLink;
- ACM Digital Library;
- Scopus / Web of Science if available through the University;
- Google Scholar for discovery only, followed by verification at the publisher/source.

---

## 6. Inclusion and Exclusion Criteria

### Include

- research directly related to transaction, payment, or credit-card fraud detection;
- studies addressing class imbalance;
- cost-sensitive learning or threshold optimisation;
- review-capacity, alert-budget, or risk-based decision policies;
- LightGBM/CatBoost or comparable tabular classifiers where methodologically useful;
- temporal evaluation;
- calibration where relevant;
- HITL decision-support literature relevant to high-stakes classification.

### Exclude or De-prioritise

- papers focused on unrelated fraud domains where the decision problem is materially different;
- papers reporting only accuracy on highly imbalanced data;
- purely descriptive articles without reproducible methodology;
- vendor marketing material;
- unsourced blog posts;
- papers included only because they use a fashionable model but do not inform the research questions.

---

## 7. Critical Review Approach

The final literature review must not become a list of paper summaries.

For each important study, evaluate:

- dataset and fraud prevalence;
- model family;
- train/validation/test design;
- temporal or random split;
- imbalance treatment;
- threshold-selection rule;
- whether the test set influenced model/policy selection;
- evaluation metrics;
- handling of false negatives;
- operational constraints;
- statistical uncertainty;
- calibration;
- explainability;
- limitations;
- relevance to the present research question.

A literature evidence matrix should be created later so that studies can be compared consistently.

---

## 8. Preliminary Research Gap

A preliminary gap supported by the initial literature review is:

> Fraud-detection studies frequently compare predictive models using global discrimination or threshold-based classification metrics, while operational fraud detection also requires converting model scores into review decisions under class imbalance, asymmetric error costs, and limited investigation resources. Although class weighting, threshold optimisation, calibration, and cost-sensitive approaches are individually well represented in the literature, there remains value in a controlled empirical study that jointly examines model family and decision-threshold strategy under explicit human-review capacity constraints, while preserving temporal evaluation, untouched test data, statistical uncertainty, and false-negative analysis.

The MSc project will therefore focus on:

```text
model choice
        ×
decision-threshold strategy
        ×
class-imbalance treatment
        ×
human-review capacity
        ↓
fraud capture
false-negative risk
precision
review workload
cost-sensitive outcome
statistical uncertainty
```

The final research-gap statement must be revised after the complete literature search. It must not claim novelty beyond what the reviewed evidence supports.

---

## 9. Next Literature-Review Tasks

1. Build a literature evidence matrix.
2. Collect approximately 20–30 high-relevance academic sources, prioritising quality over volume.
3. Identify foundational and recent papers for each core theme.
4. Verify full bibliographic details and DOI (Digital Object Identifier) for every retained source.
5. Separate background sources from directly comparable empirical studies.
6. Refine the research gap after the evidence matrix is complete.
7. Only then convert this working document into dissertation-style critical synthesis.

## Status

**Literature-review foundation established.**

This file defines the review scope and initial evidence base. It is not yet the final dissertation literature review.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| ACM, IEEE | Association for Computing Machinery; Institute of Electrical and Electronics Engineers (publishers / databases) |
| C_FP | Assumed relative cost of unnecessary intervention (a false positive), in addition to C_R |
| C_miss | Assumed relative cost of a missed fraud (C_FN in some sources) |
| C_R | Assumed review/investigation cost per reviewed transaction |
| DOI | Digital Object Identifier |
| ECE | Expected Calibration Error |
| F1 | Harmonic mean of precision and recall |
| FN, FP | False negatives (frauds not reviewed); false positives (legitimate transactions reviewed) |
| HITL | Human-in-the-Loop |
| IEEE-CIS | IEEE Computational Intelligence Society (dataset organiser) |
| ML | Machine learning |
| N_rev | Number of reviewed transactions |
| PR-AUC | Precision–Recall Area Under the Curve |
| ROC-AUC | Receiver Operating Characteristic Area Under the Curve |
| RQ1–RQ3 | Research questions 1–3 |
| SMOTE | Synthetic Minority Over-sampling Technique |
| `scale_pos_weight` | Library parameter that up-weights the positive (fraud) class |
