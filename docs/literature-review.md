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

## 4. Evidence Base

The evidence was selected by the structured search in [`literature-search-method.md`](literature-search-method.md) and is compared field by field in [`literature-evidence-matrix.md`](literature-evidence-matrix.md). Full citations, the Digital Object Identifier (DOI) of each source and its verification status are in [`references.md`](references.md), which also records whether the full text or only the abstract was read. Each statement below is limited to the text that was read, and inferences are marked **[I]**. Sources read only as abstracts must be read in full before they support a central claim in the dissertation.

Sections 4.1–4.5 cover individual recent studies (abstracts read, except where stated); Section 4.6 compares the closest studies; Sections 4.7–4.9 synthesise the evidence by theme.

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
| **Capacity-constrained evaluation** [REF-006, REF-004, REF-046, REF-084, REF-005, REF-086] | Evaluation of the highest-ranked alerts at an investigator-limited depth; daily precision top-k with a budget of k = 100 alerts or cards per day on real card streams (REF-005, REF-086) | Daily top-K review (R3) and capacity-based metrics | Transfer fidelity of a threshold's alert rate | Register entries; REF-005 and REF-086 full text |
| **Chronological IEEE-CIS evaluation** [REF-041, REF-058] | Chronological evaluation on this dataset; validation-selected cost-sensitive thresholding (REF-058) | Temporal design on IEEE-CIS | Review capacity; alert-rate fidelity | Publisher abstracts |
| **Ali et al. (2025)** [REF-098] | Claims that sequence Transformers and hybrids outperform tree ensembles on IEEE-CIS under chronological splits, with calibration and monetary utility | Tree-vs-neural comparison on IEEE-CIS; chronological evaluation | Any review-capacity constraint; reported split, seeds or uncertainty. Its IEEE-CIS PR-AUC values (0.83–0.91) are far above other reports, so it is cited as weak evidence only | Full text |
| **Namit et al. (2026)** [REF-099] | Transformer embeddings combined with gradient-boosted trees on IEEE-CIS; Precision@K motivated by limited investigation capacity | Capacity-motivated ranking metric on IEEE-CIS | A temporal holdout (it uses a stratified 80/20 split); a daily budget; uncertainty; test-period results | Full text |

**Consequences for the research gap.** In the studies reviewed, capacity-constrained review, chronological evaluation, validation-selected and cost-sensitive thresholds, calibrated cost-threshold policies, binding-versus-slack capacity effects and cost-ratio sweeps are all established, so none is claimed as novel. The remaining gap is narrower (Section 8): whether a *static* validation-selected score threshold preserves its intended alert rate on later data, and how its demand compares with a *fixed daily* review budget. RQ3 is positioned as a replication-plus-extension of Zhang et al. (2026).

### 4.7 Keeping the Natural Class Imbalance

- **Resampling inflates and distorts results.** A systematic review restricted to studies that keep the original class imbalance reports that much of the literature changes the class distribution through resampling, which may inflate reported performance [REF-094, abstract]. Applying SMOTE (Synthetic Minority Over-sampling Technique) before the train–test split produces deceptively strong results (99.9 % recall) on a public card-fraud dataset, and inadequate temporal validation is listed among the main methodological problems of the field [REF-095].
- **Resampling costs false alerts.** On a large labelled card dataset, imbalance-handling methods raised sensitivity but produced more false alarms than detected frauds [REF-091]. Under a fixed review budget, false alerts consume review capacity directly **[I]**.
- **Rebalancing preserves ranking but harms probabilities.** Undersampling changes the training prior: it leaves the ranking of transactions unchanged but distorts the predicted probabilities, which must be corrected before a probability threshold is used [REF-021]. Imbalance corrections worsened calibration without improving discrimination for logistic regression [REF-019], and in simulations across machine-learning algorithms, models without correction had equal or better calibration [REF-020].
- **Class weighting is not reliably beneficial.** In LightGBM, class balancing alone may not improve results [REF-093]; in a chronological study, a class-weighted random forest had a higher test cost than the unweighted one at cost-optimal thresholds [REF-080].

**Implication [I].** Ranking-based policies (R3) are unaffected by rebalancing, whereas probability-based policies (R4) need calibration on separate data. The evidence supports keeping the natural imbalance, excluding resampling, and testing class weighting rather than assuming it helps (H2).

### 4.8 Tree-Based and Neural Models for Tabular Data

- **General tabular benchmarks.** Gradient-boosted decision trees (GBDTs) remain strong on tabular data [REF-013, REF-014, REF-090]. In the largest comparison found (19 algorithms, 176 datasets), the gap between neural networks and GBDTs is often negligible, light tuning of CatBoost or a ResNet-style network often matters more than the choice of family, and GBDTs handle skewed and heavy-tailed features better [REF-089]. Well-regularised multilayer perceptrons (MLPs) can outperform XGBoost [REF-088].
- **Strong neural baselines.** A ResNet-style MLP is a strong deep baseline and FT-Transformer (Feature Tokenizer + Transformer) is the best deep model on most of eleven benchmarks, yet neither family is universally superior to GBDTs [REF-087]. FT-Transformer's attention cost grows quadratically with the number of input features, which makes hardware a practical constraint [REF-087].
- **Fraud-specific evidence is thin.** On IEEE-CIS, Transformer embeddings combined with gradient-boosted trees gave a small gain over tabular LightGBM (PR-AUC 0.31 vs 0.29) under a stratified, non-temporal split [REF-099]; a self-supervised tabular model outperformed XGBoost by 2.14 % AUROC [REF-101, abstract]; CatBoost and a deep network were applied to different user groups rather than compared head-to-head [REF-100, abstract]; and a study claiming Transformer superiority under chronological splits does not report enough detail to be verified [REF-098]. The closest comparator used logistic regression, LightGBM and TabNet [REF-078].

**Implication [I].** No study found compares the two families under one tuning procedure, a chronological holdout and a fixed daily review budget. The general benchmarks suggest that any difference may be small relative to tuning effects, which makes a pre-specified smallest effect size of interest important when interpreting family differences. The choice of model families is set in the protocol.

### 4.9 Review Capacity and Human Review

- In real card-fraud operations, investigators can check only a limited number of alerts each day: k = 100 alerts [REF-086] or k = 100 cards [REF-005] per day. Alert precision among those k is the operational measure, and top-k performance and AUC (Area Under the ROC Curve) can rank models differently [REF-005].
- Daily top-k metrics vary strongly from day to day because the number of frauds per day varies [REF-005], which supports day-level dependence in the inference plan **[I]**.
- Alert investigation is "strongly constrained by available human resources", and prioritisation of alerts is needed [REF-096]. The same review quotes an industry study that one-fourth of fraud costs go to manual review [REF-096, secondary figure].
- Reducing false-positive alerts is valued because it lowers analyst workload [REF-102, abstract].
- Mainstream evaluation still relies on recall, AUC, precision and F1, without capacity-based metrics [REF-092]; reviews call for cost-aware evaluation [REF-094, abstract].

**Implication [I].** A fixed daily review budget, daily top-K review and capacity-based metrics model the human review step directly; this study does so computationally, without human participants.

---

## 5. Search Strategy

The search is documented in [`literature-search-method.md`](literature-search-method.md). In summary:

- **Structured search:** IEEE Xplore (Institute of Electrical and Electronics Engineers digital library), 8 October 2026, five theme queries (class imbalance; review capacity and human-in-the-loop review; drift and thresholds; calibration and cost; IEEE-CIS studies), publication years 2015–2026. It returned 1,540 records, of which 1,328 were unique; all were screened on title, 64 on abstract, and 9 in full text.
- **Targeted searches and citation follow-up:** tree-based vs neural models for tabular data, systematic reviews under the original class imbalance, evaluation leakage and the cost of manual review, plus follow-up from the closest comparators.
- **Selection:** 17 sources were added (REF-086–REF-102) to the sources already in the register.
- **Main limitation:** only one database was searched in a structured way; a cross-publisher search (Scopus or Web of Science) would strengthen the gap statement.

---

## 6. Inclusion and Exclusion Criteria

The full criteria and exclusion codes are in [`literature-search-method.md`](literature-search-method.md), Section 4. In brief, a source is included when it is peer-reviewed (or an authoritative methodological source), concerns transaction, payment or card fraud or a method this study uses directly, and informs a research question, hypothesis or decision policy. Sources are excluded for a materially different fraud domain, a method outside the study's scope, weak evaluation (accuracy alone, resampling before the split, or unreportable design), or lack of peer review.

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

These fields are recorded for the selected studies in [`literature-evidence-matrix.md`](literature-evidence-matrix.md).

---

## 8. Research Gap

The gap is stated against the closest studies (Section 4.6), the thematic evidence (Sections 4.7–4.9) and the design-element comparison in [`literature-evidence-matrix.md`](literature-evidence-matrix.md), Section 3.

**What is not claimed as novel.** In the studies reviewed, the following are established: fraud-detection algorithms; chronological and temporal evaluation, including on IEEE-CIS [REF-041, REF-058]; capacity-constrained fraud evaluation and top-K review, including daily alert budgets on real card streams [REF-006, REF-004, REF-046, REF-084, REF-005, REF-086]; capacity-motivated ranking metrics on IEEE-CIS [REF-099]; validation-selected thresholds applied unchanged to later data [REF-058, REF-079, REF-080]; temporal transfer of review policies under capacity [REF-078]; cost-sensitive thresholding and calibrated cost-threshold policies [REF-008, REF-078]; binding-versus-slack capacity effects [REF-078]; cost-ratio sweeps [REF-078, REF-080]; and capacity-aware threshold selection or allocation [REF-081, REF-082, REF-083]. The study proposes no new algorithm, decision method, cost framework or temporal protocol.

**The remaining gap.** Zhang et al. (2026) transfer a historical review *rate*, which preserves the review fraction by construction, and compare workload rules against a retrospective oracle. Mazumder (2026) constrains the alert rate only when the threshold is selected. Elbadraoui et al. (2026) transfer a cost-optimal threshold without any capacity constraint. In the studies reviewed, the alert rate produced by a *static* validation-selected score threshold is not formally tested against its intended target on later data, and workload deviations from a fixed daily review budget are not separated into alert-rate drift and volume variation; Zhang et al. note that attributing workload displacement would require further controlled comparison. The structured search (1,328 unique IEEE Xplore records; Section 5) found no study that does either. The studies on IEEE-CIS found by the search either use a non-temporal split [REF-099] or do not report enough detail to be verified [REF-098], and none uses a fixed daily review budget.

**Contribution.** The contribution is a pre-specified confirmatory empirical evaluation of whether a static, validation-selected score threshold preserves its intended alert rate on later IEEE-CIS data. It has two distinct parts:

1. **H3 (confirmatory):** a formal equivalence test of the realised alert rate of the Validation-2-selected 2 % threshold against its 2 % target, within a pre-specified margin of ±20 %.
2. **Daily-budget fidelity (descriptive, not a hypothesis):** threshold-generated demand compared with a fixed daily review budget K (overflow, unused capacity, load), with the deviation separated into an alert-rate component and a volume component.

This study extends prior work in two further, secondary ways. RQ1 (H1, H2) provides confirmatory, dependence-aware comparisons of LightGBM and CatBoost with and without class weighting at fixed daily capacity; these are empirical comparisons, not novelty claims. RQ3 replicates and extends Zhang et al.'s binding-versus-slack finding under a fixed daily cap, summarised as break-even (ρ_BE) and saturation (ρ_sat) cost ratios; this is an alternative operationalisation, not the main gap.

**Limits.** The conclusions are limited to one benchmark, one validation-to-test transfer and simulated capacity and cost assumptions. The comparison rests on the studies reviewed: one database was searched in a structured way, so a cross-publisher search (for example Scopus or Web of Science) would strengthen it before the dissertation is written.

---

## Status

**Evidence base selected.** The search, selection and evidence matrix are documented; the research gap has been re-checked against the selected evidence. This file is not yet the final dissertation literature review.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [I] | Evidence tag: inference or recommendation (not a statement from the source) |
| ACM, IEEE | Association for Computing Machinery; Institute of Electrical and Electronics Engineers (publishers / databases) |
| AML | Anti-money laundering |
| AUC, AUROC | Area under the receiver operating characteristic (ROC) curve |
| FT-Transformer | Feature Tokenizer + Transformer model for tabular data |
| GBDT | Gradient-boosted decision trees (e.g. LightGBM, CatBoost, XGBoost) |
| k | Number of alerts (or cards) investigators can check per day in the cited studies |
| MLP, ResNet | Multilayer perceptron; residual-network-style MLP |
| Precision@K | Precision among the K highest-ranked transactions |
| TabNet | Attention-based deep model for tabular data |
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
