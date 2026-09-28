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

> When a score threshold selected on validation data is transferred to a later temporal period, how well does it preserve the intended alert rate, and how does the resulting daily review workload compare with the fixed review budget?

RQ2 was revised by [DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md) to separate alert-rate drift (model and data) from daily-volume variation.

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

The experimental capacities, fixed in [DR-002](decisions/DR-002-review-capacity-definition.md), are:

- 1%;
- 2% (primary);
- 5%;
- 10%

of transactions. They are simulated research scenarios, not real institutional capacities.

Expected argument:

> In operational settings, a model may generate more alerts than investigators can review. Therefore, fraud-detection effectiveness should also be evaluated under explicit capacity constraints.

---

### 3.7 Cost-Sensitive Fraud-Detection Policies

The study intends to distinguish capacity constraints from explicit cost sensitivity.

A generic cost model may take the form:

$$
\begin{aligned}
\text{Total Cost} &= C_{\mathrm{miss}} \times \mathrm{FN} \\
&\quad + C_{\mathrm{FP}} \times \mathrm{FP} \\
&\quad + C_R \times N_{\mathrm{rev}}
\end{aligned}
$$

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

**Verification status (Issue #33).** Each source below was confirmed to exist, and its bibliographic details were completed from Crossref, the publisher's Digital Object Identifier (DOI) registration. Full entries are in [`references.md`](references.md). The full texts could not be read from the verification environment, so each "Checked relevance" list contains only points stated in the abstract. For REF-060, Issue #33 review: the official ScienceDirect publisher abstract was checked on 26 September 2026. Automated access from the verification environment was refused (HTTP 403). Points in earlier drafts of this section that could not be confirmed have been removed, and inferences are marked **[I]**.

### 4.1 Systematic Review of Credit-Card Fraud Detection

**Cherif, A., Badhib, A., Ammar, H., Alshehri, S., Kalkatawi, M. & Imine, A. (2023) 'Credit card fraud detection in the era of disruptive technologies: a systematic review'**\
*Journal of King Saud University – Computer and Information Sciences*, 35(1), pp. 145–174. doi:10.1016/j.jksuci.2022.11.008 [REF-056].

Checked relevance (abstract):

- reviews 40 credit-card fraud-detection studies published from 2015 to 2021;
- categorises them by topic, including the class-imbalance problem and feature engineering, and by machine-learning technology (traditional and deep learning);
- provides background for the problem formulation **[I]**.

---

### 4.2 Threshold Optimisation Under Imbalance

**Leevy, J.L., Johnson, J.M., Hancock, J. & Khoshgoftaar, T.M. (2023) 'Threshold optimization and random undersampling for imbalanced credit card data'**\
*Journal of Big Data*, 10, 58. doi:10.1186/s40537-023-00738-z [REF-057].

Checked relevance (abstract):

- studies output thresholding as a way to address class imbalance without changing the dataset or the learner;
- evaluates four threshold-optimisation techniques with eight threshold-dependent and two threshold-agnostic metrics;
- reports the best optimal-threshold results without random undersampling;
- uses the public Credit Card Fraud Detection Dataset, not IEEE-CIS;
- is consistent with treating threshold selection as a step separate from classifier training **[I]**.

---

### 4.3 Recent Cost-Sensitive and Calibrated Fraud Detection

**Hossain, M.S., Akter, S., Chy, K.S., Islam, M.A., Shahiduzzaman, M., Kabtia, M., Sourav, M.S.A. & Raihan, D.I. (2026) 'TEMPLAR fraud a verifier grounded calibrated and cost sensitive framework for transaction fraud detection'**\
*Discover Data*, 4, 20. doi:10.1007/s44248-026-00117-y [REF-058].

Checked relevance (abstract):

- frames transaction fraud detection around class imbalance, temporal drift and asymmetric error costs;
- includes probability calibration (one-dimensional optimal-transport calibration);
- applies validation-selected cost-sensitive thresholding;
- evaluates on IEEE-CIS and BAF Base (a bank-account fraud dataset) with chronological train/validation/test splits and temporal future-slice evaluation;
- reinforces the value of validation-selected operating policies **[I]**.

This source is recent and particularly relevant, but its methods and assumptions must be critically compared with the narrower MSc research scope. It is further evidence that temporal evaluation on IEEE-CIS is not new.

---

### 4.4 Recent Explainable Fraud-Detection Research

**Anh Luong, N.D. & Xie, S. (2026) 'Explainable ensemble machine learning for financial transaction fraud detection: insights from XGBoost and deep neural networks'**\
*The Journal of Finance and Data Science*, 12, 100195. doi:10.1016/j.jfds.2026.100195 [REF-059].

Checked relevance (abstract):

- addresses extreme class imbalance;
- evaluates focal loss, class weighting and threshold optimisation;
- combines an Extreme Gradient Boosting (XGBoost) model with a deep neural network, with SHAP-based (SHapley Additive exPlanations) interpretability;
- describes its data only as "a real-world transaction dataset".

Whether thresholds are selected on validation data, and which metrics are reported, has **not** been checked.

---

### 4.5 Recent Risk-Control-Oriented Research

**Zhu, R., Zhang, X., Wang, T., Liao, J., Chung, S.-H. & Zhang, X. (2026) 'DISCO: decoupling representation learning and risk control for reliable credit card fraud detection'**\
*Decision Support Systems*, 208, 114717. doi:10.1016/j.dss.2026.114717 [REF-060].

Checked relevance (publisher abstract; full text not read):

- separates representation learning from risk-controlled decision-making;
- uses deep metric learning for representation learning;
- uses conformal risk control to provide a statistical guarantee on the false-negative rate;
- discusses operational trade-offs / operational efficiency.

It is related-work context only. No result, dataset or detail beyond the abstract is attributed to it.

### 4.6 Closest Comparators and the Remaining Gap (Issue #35)

Issue #35 compared the study with the closest work found, to answer the supervisor's request that novelty be shown against existing studies. Full entries and verification notes are in [`references.md`](references.md); the table below states only what the checked text supports. Symbols used: K (the fixed daily review budget), R2 (the static capacity-rate threshold chosen on Validation 2, V2), R3 (daily top-K review), R4 (the calibrated cost-aware capped policy), H3 (the alert-rate transfer hypothesis), ρ_BE (break-even cost ratio) and ρ_sat (saturation cost ratio).

| Source | What it already establishes | Overlap with this study | What it does not establish | Checked |
|---|---|---|---|---|
| **Zhang et al. (2026)** [REF-078], closest comparator | On IEEE-CIS, separate model-fitting, validation, policy-selection and future periods; capacity-constrained top-K review; transfer of a historical review *rate*; a Platt-calibrated cost threshold compared with full-capacity review; tight capacity (3 %) compresses the policies, slack capacity (10 %) separates them; cost-coefficient sweeps; regret against a retrospective oracle | Four-period temporal design; top-K and full-capacity review (R3); a calibrated cost threshold of the same form as R4's; binding-versus-slack comparison (RQ3) | No target-rate transfer of a static score threshold; no alert-rate fidelity or equivalence test; capacity scales with period size rather than a fixed daily K; sources of workload displacement explicitly left unseparated; descriptive uncertainty (training seeds) | Full text (accepted author version) |
| **Elbadraoui et al. (2026)** [REF-080] | A validation-selected cost-optimal threshold applied unchanged to a chronologically later test period; a cost-ratio sweep | Threshold transfer from validation to a later period; cost-ratio sensitivity (RQ3) | Any review-capacity constraint; full-capacity comparison; target alert rate or alert-rate fidelity; equivalence testing; break-even ratios | Full text |
| **Mazumder (2026)** [REF-079], bounded comparator | An F2-maximising threshold selected on validation under an alert-rate ceiling and applied unchanged to test | Capacity-constrained threshold selection before transfer | The later alert rate is not reported or tested; no equivalence test or decomposition; synthetic anti-money-laundering (AML) data; unreconciled split description and several internal inconsistencies, so it is cited only as an example | Full text |
| **Shen and Kurshan (2020)** [REF-081] | Static alert thresholds do not account for alert-processing capacity (dropped alerts); adaptive threshold selection under capacity | Motivation for measuring threshold-generated demand against capacity | Content known from a secondary abstract only | Abstract (secondary source) |
| **Vanderschueren et al. (2024)** [REF-082] | Limited, stochastic investigation capacity built into the learning objective (learning to rank) | Capacity-aware review decisions | A method proposal, not an evaluation of static thresholds over time | Abstract (arXiv) |
| **Alves et al. (2024)** [REF-083] | Cost-sensitive allocation of fraud cases between a model and several analysts under individual workload limits | Human capacity and cost in fraud review | Uses synthetic analysts and models reviewer error; this study assumes one pooled budget and perfect verification | Abstract (OpenReview) |
| **Capacity-constrained evaluation** [REF-006, REF-004, REF-046, REF-084] | Evaluation of the highest-ranked alerts at an investigator-limited depth; daily precision top-k; daily investigation budgets | Daily top-K review (R3) and capacity-based metrics | Transfer fidelity of a threshold's alert rate | Register entries |
| **Chronological IEEE-CIS evaluation** [REF-041, REF-058] | Chronological evaluation on this dataset; validation-selected cost-sensitive thresholding (REF-058) | Temporal design on IEEE-CIS | Review capacity; alert-rate fidelity | Publisher abstracts |

**Consequences for the research gap.** In the studies reviewed, capacity-constrained review, chronological evaluation, validation-selected and cost-sensitive thresholds, calibrated cost-threshold policies, binding-versus-slack capacity effects and cost-ratio sweeps are all established, so none is claimed as novel. The remaining gap is narrower (Section 8): whether a *static* validation-selected score threshold preserves its intended alert rate on later data, and how its demand compares with a *fixed daily* review budget. RQ3 is positioned as a replication-plus-extension of Zhang et al. (2026).

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

## 8. Research Gap

This section replaces the preliminary gap statement drafted before the closest-work comparison (Issue #35, Section 4.6).

**What is not claimed as novel.** In the studies reviewed, the following are established: fraud-detection algorithms; chronological and temporal evaluation, including on IEEE-CIS [REF-041, REF-058]; capacity-constrained fraud evaluation and top-K review [REF-006, REF-004, REF-046, REF-084]; validation-selected thresholds applied unchanged to later data [REF-058, REF-079, REF-080]; temporal transfer of review policies under capacity [REF-078]; cost-sensitive thresholding and calibrated cost-threshold policies [REF-008, REF-078]; binding-versus-slack capacity effects [REF-078]; cost-ratio sweeps [REF-078, REF-080]; and capacity-aware threshold selection or allocation [REF-081, REF-082, REF-083]. The study proposes no new algorithm, decision method, cost framework or temporal protocol.

**The remaining gap.** Zhang et al. (2026) transfer a historical review *rate*, which preserves the review fraction by construction, and compare workload rules against a retrospective oracle. Mazumder (2026) constrains the alert rate only when the threshold is selected. Elbadraoui et al. (2026) transfer a cost-optimal threshold without any capacity constraint. In the studies reviewed, the alert rate produced by a *static* validation-selected score threshold is not formally tested against its intended target on later data, and workload deviations from a fixed daily review budget are not separated into alert-rate drift and volume variation; Zhang et al. note that attributing workload displacement would require further controlled comparison. The remaining gap is therefore narrower than the one drafted earlier.

**Contribution.** The contribution is a pre-specified confirmatory empirical evaluation of whether a static, validation-selected score threshold preserves its intended alert rate on later IEEE-CIS data. It has two distinct parts:

1. **H3 (confirmatory):** a formal equivalence test of the realised alert rate of the Validation-2-selected 2 % threshold against its 2 % target, within a pre-specified margin of ±20 %.
2. **Daily-budget fidelity (descriptive, not a hypothesis):** threshold-generated demand compared with a fixed daily review budget K (overflow, unused capacity, load), with the deviation separated into an alert-rate component and a volume component.

This study extends prior work in two further, secondary ways. RQ1 (H1, H2) provides confirmatory, dependence-aware comparisons of LightGBM and CatBoost with and without class weighting at fixed daily capacity; these are empirical comparisons, not novelty claims. RQ3 replicates and extends Zhang et al.'s binding-versus-slack finding under a fixed daily cap, summarised as break-even (ρ_BE) and saturation (ρ_sat) cost ratios; this is an alternative operationalisation, not the main gap.

**Limits.** The conclusions are limited to one benchmark, one validation-to-test transfer and simulated capacity and cost assumptions. The comparison rests on the studies reviewed; a University-database search (for example Scopus or Web of Science) should confirm it before the dissertation is written.

---

## 9. Next Literature-Review Tasks

1. Build a literature evidence matrix.
2. Collect approximately 20–30 high-relevance academic sources, prioritising quality over volume.
3. Identify foundational and recent papers for each core theme.
4. Verify full bibliographic details and DOI (Digital Object Identifier) for every retained source.
5. Separate background sources from directly comparable empirical studies.
6. ~~Refine the research gap after the evidence matrix is complete.~~ Refined by Issue #35 (Sections 4.6 and 8); re-check it after the University-database search.
7. Only then convert this working document into dissertation-style critical synthesis.

## Status

**Literature-review foundation established.**

This file defines the review scope and initial evidence base. It is not yet the final dissertation literature review.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [I] | Evidence tag: inference or recommendation (not a statement from the source) |
| ACM, IEEE | Association for Computing Machinery; Institute of Electrical and Electronics Engineers (publishers / databases) |
| AML | Anti-money laundering |
| BAF Base | Bank Account Fraud dataset suite, base variant |
| C_FP | Assumed relative cost of unnecessary intervention (a false positive), in addition to C_R |
| C_miss | Assumed relative cost of a missed fraud (C_FN in some sources) |
| C_R | Assumed review/investigation cost per reviewed transaction |
| DOI | Digital Object Identifier |
| ECE | Expected Calibration Error |
| F1 | Harmonic mean of precision and recall |
| F2 | Weighted harmonic mean of precision and recall that weights recall twice as heavily |
| FN, FP | False negatives (frauds not reviewed); false positives (legitimate transactions reviewed) |
| H1, H2, H3 | Confirmatory hypotheses: model family; class weighting; alert-rate transfer fidelity |
| HITL | Human-in-the-Loop |
| IEEE-CIS | IEEE Computational Intelligence Society (dataset organiser) |
| K | Fixed daily review budget (reviews per relative day) |
| ML | Machine learning |
| N_rev | Number of reviewed transactions |
| PR-AUC | Precision–Recall Area Under the Curve |
| R2, R3, R4 | Decision policies: capacity-rate threshold chosen on V2; daily top-K review; calibrated cost-aware capped policy |
| ROC-AUC | Receiver Operating Characteristic Area Under the Curve |
| REF-NNN | Reference identifier in `references.md` |
| RQ1–RQ3 | Research questions 1–3 |
| ρ_BE, ρ_sat | Break-even and saturation miss-to-review cost ratios (RQ3) |
| SHAP | SHapley Additive exPlanations (a feature-attribution method) |
| SMOTE | Synthetic Minority Over-sampling Technique |
| V2 | Validation 2 (relative days 127–154) |
| `scale_pos_weight` | Library parameter that up-weights the positive (fraud) class |
| XGBoost | Extreme Gradient Boosting library |
