# Literature Review

This review sets out the evidence behind the study's research questions and its research gap. The research questions (RQ1–RQ3) and hypotheses (H1–H3) are stated in the [research protocol](protocol.md); the study's contribution is stated in the [research foundation](research-foundation.md#research-contribution).

**Evidence base.** The sources were selected by the structured search in [`literature-search-method.md`](literature-search-method.md) and are compared field by field in [`literature-evidence-matrix.md`](literature-evidence-matrix.md). Full citations, the Digital Object Identifier (DOI) of each source and its verification status are in [`references.md`](references.md), which also records whether the full text or only the abstract was read. Each statement below is limited to the text that was read, and inferences are marked **[I]**. Sources read only as abstracts must be read in full before they support a central claim in the dissertation.

## 1. Overview of credit-card fraud-detection research

**Cherif et al. (2023)** [REF-056](references.md#ref-056).

Checked relevance (abstract):

- reviews 40 credit-card fraud-detection studies published from 2015 to 2021;
- categorises them by topic, including the class-imbalance problem and feature engineering, and by machine-learning technology (traditional and deep learning);
- provides background for the problem formulation **[I]**.

## 2. Tree-based and neural models for tabular data

- **General tabular benchmarks.** Gradient-boosted decision trees (GBDTs) remain strong on tabular data [REF-013, REF-014, REF-090]. In the largest comparison found (19 algorithms, 176 datasets), the gap between neural networks and GBDTs is often negligible, light tuning of CatBoost or a ResNet-style (residual-network-style) network often matters more than the choice of family, and GBDTs handle skewed and heavy-tailed features better [REF-089]. Well-regularised multilayer perceptrons (MLPs) can outperform XGBoost (Extreme Gradient Boosting) [REF-088].
- **Strong neural baselines.** A ResNet-style MLP is a strong deep baseline and FT-Transformer (Feature Tokenizer + Transformer) is the best deep model on most of eleven benchmarks, yet neither family is universally superior to GBDTs [REF-087]. FT-Transformer's attention cost grows quadratically with the number of input features, which makes hardware a practical constraint [REF-087].
- **Fraud-specific evidence is thin.** On IEEE-CIS (the IEEE Computational Intelligence Society Fraud Detection dataset), Transformer embeddings combined with gradient-boosted trees gave a small gain over tabular LightGBM (Precision–Recall Area Under the Curve, PR-AUC, 0.31 vs 0.29) under a stratified, non-temporal split [REF-099]; a self-supervised tabular model outperformed XGBoost by 2.14 % AUROC (area under the receiver operating characteristic curve) [REF-101, abstract]; CatBoost and a deep network were applied to different user groups rather than compared head-to-head [REF-100, abstract]; and a study claiming Transformer superiority under chronological splits does not report enough detail to be verified [REF-098]. The closest comparator used logistic regression, LightGBM and TabNet (an attention-based deep model for tabular data) [REF-078].

**Implication [I].** No study found compares the two families under one tuning procedure, a chronological holdout and a fixed daily review budget. The general benchmarks suggest that any difference may be small relative to tuning effects, which makes a pre-specified smallest effect size of interest important when interpreting family differences. The model set follows this evidence ([DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md)).

**Combined tree and neural pipelines.**

**Anh Luong & Xie (2026)** [REF-059](references.md#ref-059).

Checked relevance (abstract):

- addresses extreme class imbalance;
- evaluates focal loss, class weighting and threshold optimisation;
- combines an XGBoost model with a deep neural network, with SHAP-based (SHapley Additive exPlanations) interpretability;
- describes its data only as "a real-world transaction dataset".

Whether thresholds are selected on validation data, and which metrics are reported, has **not** been checked.

## 3. Class imbalance: resampling and class weighting

- **Resampling can distort evaluation.** A systematic review restricted to studies that keep the original class imbalance reports that much of the literature changes the class distribution through resampling, which may inflate reported performance [REF-094, abstract]. When SMOTE (Synthetic Minority Over-sampling Technique) is applied before the train–test split, so that the test data are also resampled, it produces deceptively strong results (99.9 % recall) on a public card-fraud dataset, and inadequate temporal validation is listed among the main methodological problems of the field [REF-095].
- **Resampling costs false alerts.** On a large labelled card dataset, imbalance-handling methods raised sensitivity but produced more false alarms than detected frauds [REF-091]. Under a fixed review budget, false alerts consume review capacity directly **[I]**.
- **Rebalancing distorts probabilities.** Undersampling shifts the class prior seen in training and distorts the predicted probabilities, which must be corrected before a probability threshold is used; the correction is monotone, so it restores the probabilities without changing the fitted model's ranking [REF-021]. Retraining on rebalanced data, or with class weights, can still change the learned ranking **[I]**. Imbalance corrections worsened calibration without improving discrimination for logistic regression [REF-019], and in simulations across machine-learning algorithms, models without correction had equal or better calibration [REF-020].
- **Class weighting is not reliably beneficial.** In LightGBM, class balancing alone may not improve results [REF-093]; in a chronological study, a class-weighted random forest had a higher test cost than the unweighted one at cost-optimal thresholds [REF-080].

**Implication [I].** Probability-based policies (such as R4, the calibrated cost-aware capped policy) need calibration on separate data after any rebalancing. Whether class weighting changes the learned ranking, and therefore what ranking-based policies review (such as R3, daily review of the K highest-scoring transactions, with K the fixed daily review budget), is an empirical question. The evidence supports keeping the natural imbalance, excluding resampling, and testing class weighting rather than assuming it helps (hypothesis H2).

## 4. Thresholds, calibration and cost-sensitive decisions

**Threshold selection as a separate step.**

**Leevy et al. (2023)** [REF-057](references.md#ref-057).

Checked relevance (abstract):

- studies output thresholding as a way to address class imbalance without changing the dataset or the learner;
- evaluates four threshold-optimisation techniques with eight threshold-dependent and two threshold-agnostic metrics;
- reports the best optimal-threshold results without random undersampling;
- uses the public Credit Card Fraud Detection Dataset, not IEEE-CIS;
- is consistent with treating threshold selection as a step separate from classifier training **[I]**.

**Calibrated, cost-sensitive operating policies.**

**Hossain et al. (2026)** [REF-058](references.md#ref-058).

Checked relevance (abstract):

- frames transaction fraud detection around class imbalance, temporal drift and asymmetric error costs;
- includes probability calibration (one-dimensional optimal-transport calibration);
- applies validation-selected cost-sensitive thresholding;
- evaluates on IEEE-CIS and BAF Base (a bank-account fraud dataset) with chronological train/validation/test splits and temporal future-slice evaluation;
- reinforces the value of validation-selected operating policies **[I]**.

This source is recent and particularly relevant; its methods and assumptions are compared critically with this study's scope. It is further evidence that temporal evaluation on IEEE-CIS is not new.

Absolute cost-derived probability thresholds require calibrated probability estimates [REF-008, REF-073].

**Risk-controlled decisions.**

**Zhu et al. (2026)** [REF-060](references.md#ref-060).

Checked relevance (publisher abstract; full text not read):

- separates representation learning from risk-controlled decision-making;
- uses deep metric learning for representation learning;
- uses conformal risk control to provide a statistical guarantee on the false-negative rate;
- discusses operational trade-offs / operational efficiency.

It is related-work context only. No result, dataset or detail beyond the abstract is attributed to it.

**Synthesis [I].** The checked studies fix the operating point by three different objectives: a threshold-dependent metric optimised after training [REF-057]; an expected-cost rule [REF-058, REF-078, REF-080], applied to calibrated probabilities in two of them [REF-058, REF-078]; and a statistical guarantee on the false-negative rate [REF-060]. Two select a score threshold on validation data and apply it to a chronologically later period [REF-058, REF-080], while Zhang et al. transfer a historical review rate rather than a score threshold [REF-078] (Section 6). None of the checked texts reports whether a fixed score threshold keeps its intended alert rate in the later period; for the sources read only as abstracts [REF-057, REF-058, REF-060], this cannot be established from what was read. The theme therefore supplies the two operating rules this study evaluates over time: a static validation-selected score threshold, whose alert-rate transfer is the subject of RQ2, and a cost-derived threshold on calibrated probabilities, compared with filling the review budget in RQ3.

## 5. Review capacity and human review

- In real card-fraud operations, investigators can check only a limited number of alerts each day: k = 100 alerts [REF-086] or k = 100 cards [REF-005] per day. Alert precision among those k is the operational measure, and top-k performance and AUC (area under the receiver operating characteristic, ROC, curve) can rank models differently [REF-005].
- Daily top-k metrics vary strongly from day to day because the number of frauds per day varies [REF-005], which supports day-level dependence in the inference plan **[I]**.
- Alert investigation is "strongly constrained by available human resources", and prioritisation of alerts is needed [REF-096]. The same review quotes an industry study that one-fourth of fraud costs go to manual review [REF-096, secondary figure].
- Reducing false-positive alerts is valued because it lowers analyst workload [REF-102, abstract].
- Mainstream evaluation still relies on recall, AUC, precision and F1 (the harmonic mean of precision and recall), without capacity-based metrics [REF-092]; reviews call for cost-aware evaluation [REF-094, abstract].

**Implication [I].** A fixed daily review budget, daily top-K review and capacity-based metrics model the human review step directly; this study does so computationally, without human participants.

## 6. Closest comparators

Full entries and verification notes are in [`references.md`](references.md); the table below states only what the checked text supports. Symbols used: K (the fixed daily review budget), R2 (the static capacity-rate threshold chosen on Validation 2, V2), R3 (daily top-K review), R4 (the calibrated cost-aware capped policy), H3 (the alert-rate transfer hypothesis), ρ_BE (break-even cost ratio) and ρ_sat (saturation cost ratio).

| Source | What it already establishes | Overlap with this study | What it does not establish | Checked |
|---|---|---|---|---|
| **Zhang et al. (2026)** [REF-078], closest comparator | On IEEE-CIS, separate model-fitting, validation, policy-selection and future periods; capacity-constrained top-K review; transfer of a historical review *rate*; a Platt-calibrated cost threshold compared with full-capacity review; tight capacity (3 %) compresses the policies, slack capacity (10 %) separates them; cost-coefficient sweeps; regret against a retrospective oracle | Four-period temporal design; top-K and full-capacity review (R3); a calibrated cost threshold of the same form as R4's; binding-versus-slack comparison (RQ3) | No target-rate transfer of a static score threshold; no alert-rate fidelity or equivalence test; capacity scales with period size rather than a fixed daily K; sources of workload displacement explicitly left unseparated; descriptive uncertainty (training seeds) | Full text (accepted author version) |
| **Elbadraoui et al. (2026)** [REF-080] | A validation-selected cost-optimal threshold applied unchanged to a chronologically later test period; a cost-ratio sweep | Threshold transfer from validation to a later period; cost-ratio sensitivity (RQ3) | Any review-capacity constraint; full-capacity comparison; target alert rate or alert-rate fidelity; equivalence testing; break-even ratios | Full text |
| **Mazumder (2026)** [REF-079], bounded comparator | An F2-maximising (recall weighted twice as heavily as precision) threshold selected on validation under an alert-rate ceiling and applied unchanged to test | Capacity-constrained threshold selection before transfer | The later alert rate is not reported or tested; no equivalence test or decomposition; synthetic anti-money-laundering (AML) data; unreconciled split description and several internal inconsistencies, so it is cited only as an example | Full text |
| **Shen and Kurshan (2020)** [REF-081] | Static alert thresholds do not account for alert-processing capacity (dropped alerts); adaptive threshold selection under capacity | Motivation for measuring threshold-generated demand against capacity | Content known from a secondary abstract only | Abstract (secondary source) |
| **Vanderschueren et al. (2024)** [REF-082] | Limited, stochastic investigation capacity built into the learning objective (learning to rank) | Capacity-aware review decisions | A method proposal, not an evaluation of static thresholds over time | Abstract (arXiv) |
| **Alves et al. (2024)** [REF-083] | Cost-sensitive allocation of fraud cases between a model and several analysts under individual workload limits | Human capacity and cost in fraud review | Uses synthetic analysts and models reviewer error; this study assumes one pooled budget and perfect verification | Abstract (OpenReview) |
| **Capacity-constrained evaluation** [REF-006, REF-004, REF-046, REF-084, REF-005, REF-086] | Evaluation of the highest-ranked alerts at an investigator-limited depth; daily precision top-k with a budget of k = 100 alerts or cards per day on real card streams (REF-005, REF-086) | Daily top-K review (R3) and capacity-based metrics | Transfer fidelity of a threshold's alert rate | Register entries; REF-005 and REF-086 full text |
| **Chronological IEEE-CIS evaluation** [REF-041, REF-058] | Chronological evaluation on this dataset; validation-selected cost-sensitive thresholding (REF-058) | Temporal design on IEEE-CIS | Review capacity; alert-rate fidelity | Publisher abstracts |
| **Ali et al. (2025)** [REF-098] | Claims that sequence Transformers and hybrids outperform tree ensembles on IEEE-CIS under chronological splits, with calibration and monetary utility | Tree-vs-neural comparison on IEEE-CIS; chronological evaluation | Any review-capacity constraint; reported split, seeds or uncertainty. Its IEEE-CIS PR-AUC values (0.83–0.91) are far above other reports, so it is cited as weak evidence only | Full text |
| **Namit et al. (2026)** [REF-099] | Transformer embeddings combined with gradient-boosted trees on IEEE-CIS; Precision@K motivated by limited investigation capacity | Capacity-motivated ranking metric on IEEE-CIS | A temporal holdout (it uses a stratified 80/20 split); a daily budget; uncertainty; test-period results | Full text |

**Consequences for the research gap [I].** In the studies reviewed, capacity-constrained review, chronological evaluation, validation-selected and cost-sensitive thresholds, calibrated cost-threshold policies, binding-versus-slack capacity effects and cost-ratio sweeps are all established, so none is claimed as novel. The remaining gap is narrower (Section 8): whether a *static* validation-selected score threshold preserves its intended alert rate on later data, and how its demand compares with a *fixed daily* review budget. RQ3 is positioned as a replication-plus-extension of Zhang et al. (2026).

## 7. Methodological literature behind the design

The literature behind each methodological decision is set out, with its evidence tags, in the Decision Record that takes the decision:

- temporal validation and leakage control: [DR-001](decisions/DR-001-temporal-split-strategy.md);
- capacity-constrained evaluation and top-k metrics: [DR-002](decisions/DR-002-review-capacity-definition.md);
- equivalence testing and relative equivalence margins: [DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md);
- the smallest effect size of interest and multiplicity control: [DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md);
- bootstrap inference for clustered and dependent data: [DR-006](decisions/DR-006-bootstrap-inference-strategy.md);
- cost-derived thresholds, cost curves and break-even analysis: [DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md);
- native categorical handling in the tree-based libraries: [DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md);
- the tree-based and neural model families: [DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md).

## 8. Research gap

The gap is stated against the closest studies (Section 6), the thematic evidence (Sections 1–5) and the design-element comparison in [`literature-evidence-matrix.md`](literature-evidence-matrix.md), Section 3.

**What is not claimed as novel.** In the studies reviewed, the following are established: fraud-detection algorithms; chronological and temporal evaluation, including on IEEE-CIS [REF-041, REF-058]; capacity-constrained fraud evaluation and top-K review, including daily alert budgets on real card streams [REF-006, REF-004, REF-046, REF-084, REF-005, REF-086]; capacity-motivated ranking metrics on IEEE-CIS [REF-099]; validation-selected thresholds applied unchanged to later data [REF-058, REF-079, REF-080]; temporal transfer of review policies under capacity [REF-078]; cost-sensitive thresholding and calibrated cost-threshold policies [REF-008, REF-078]; binding-versus-slack capacity effects [REF-078]; cost-ratio sweeps [REF-078, REF-080]; and capacity-aware threshold selection or allocation [REF-081, REF-082, REF-083]. The study proposes no new algorithm, decision method, cost framework or temporal protocol.

**The remaining gap [I].** Zhang et al. (2026) transfer a historical review *rate*, which preserves the review fraction by construction, and compare workload rules against a retrospective oracle. Mazumder (2026) constrains the alert rate only when the threshold is selected. Elbadraoui et al. (2026) transfer a cost-optimal threshold without any capacity constraint. In the studies reviewed, the alert rate produced by a *static* validation-selected score threshold is not formally tested against its intended target on later data, and workload deviations from a fixed daily review budget are not separated into alert-rate drift and volume variation; Zhang et al. note that attributing workload displacement would require further controlled comparison. The structured search (1,328 unique IEEE Xplore records; [`literature-search-method.md`](literature-search-method.md)) found no study that does either. The studies on IEEE-CIS found by the search either use a non-temporal split [REF-099] or do not report enough detail to be verified [REF-098], and none uses a fixed daily review budget.

**Contribution.** The study's contribution, and the order of distinctiveness of its components, are stated in the [research foundation](research-foundation.md#research-contribution).

**Limits.** The conclusions are bounded by one benchmark, one validation-to-test transfer, the four models studied and simulated capacity and cost assumptions. The comparison rests on the studies reviewed: one database was searched in a structured way, so a cross-publisher search (for example Scopus or Web of Science) would strengthen it before the dissertation is written.

## References

Full entries and verification status are in [`references.md`](references.md).

REF-004, REF-005, REF-006, REF-008, REF-013, REF-014, REF-019, REF-020, REF-021, REF-041, REF-046, REF-056, REF-057, REF-058, REF-059, REF-060, REF-073, REF-078, REF-079, REF-080, REF-081\*, REF-082, REF-083, REF-084, REF-086, REF-087, REF-088, REF-089, REF-090, REF-091, REF-092, REF-093, REF-094, REF-095, REF-096, REF-098, REF-099, REF-100, REF-101, REF-102.

\* Partially verified.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [I] | Evidence tag: inference or recommendation (not a statement from the source) |
| AML | Anti-money laundering |
| AUC, AUROC | Area under the receiver operating characteristic (ROC) curve |
| BAF Base | Bank Account Fraud dataset suite, base variant |
| DOI | Digital Object Identifier |
| DR-NNN | Decision Record NNN in `decisions/` |
| F1, F2 | Harmonic mean of precision and recall; weighted version that weights recall twice as heavily |
| FT-Transformer | Feature Tokenizer + Transformer model for tabular data |
| GBDT | Gradient-boosted decision trees (e.g. LightGBM, CatBoost, XGBoost) |
| H1, H2, H3 | Confirmatory hypotheses: model family (tree-based vs neural); class weighting; alert-rate transfer fidelity |
| IEEE-CIS | IEEE Computational Intelligence Society (dataset organiser) |
| K | Fixed daily review budget (reviews per relative day) |
| k | Number of alerts (or cards) investigators can check per day in the cited studies |
| MLP, ResNet | Multilayer perceptron; residual-network-style MLP |
| PR-AUC | Precision–Recall Area Under the Curve |
| Precision@K | Precision among the K highest-ranked transactions |
| R2, R3, R4 | Decision policies: capacity-rate threshold chosen on V2; daily top-K review; calibrated cost-aware capped policy |
| REF-NNN | Reference identifier in `references.md` |
| Relative day | `TransactionDT` (the dataset's time offset in seconds from an undisclosed reference point) divided by 86,400 seconds and rounded down |
| ρ_BE, ρ_sat | Break-even and saturation miss-to-review cost ratios (RQ3) |
| ROC | Receiver operating characteristic |
| RQ1–RQ3 | Research questions 1–3 |
| SHAP | SHapley Additive exPlanations (a feature-attribution method) |
| SMOTE | Synthetic Minority Over-sampling Technique |
| TabNet | Attention-based deep model for tabular data |
| V2 | Validation 2 (relative days 127–154) |
| XGBoost | Extreme Gradient Boosting library |
