# Literature Evidence Matrix

This matrix compares the selected studies on the same fields, so that their designs and findings can be judged consistently. The search and selection procedure is in [`literature-search-method.md`](literature-search-method.md); full citations and verification status are in [`references.md`](references.md); the synthesis is in [`literature-review.md`](literature-review.md).

Each cell states only what the text that was read supports. The **Read** column gives that text: *Full* (full text), *Abstract* (abstract only) or *Register* (the note already recorded in `references.md`). "n.r." means not reported in the text that was read. Inferences are tagged **[I]**.

---

## 1. Empirical fraud-detection studies

Data type: *real* = real transactions; *synthetic* = simulated data. ULB = the Université Libre de Bruxelles `creditcard.csv` dataset (European cardholders, two days of September 2013). IEEE-CIS = the IEEE Computational Intelligence Society Fraud Detection dataset used by this study.

| ID | Study | Data | Split | Models | Imbalance handling | Capacity / top-K | Calibration / cost | Uncertainty | Quality notes | Key finding | Relevance | Read |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REF-005 | Dal Pozzolo et al. (2018) | Real; European e-commerce card transactions, > 75 million over three years | Daily stream; classifiers updated on preceding days | Random-forest ensembles | Each tree trained on a balanced bootstrap sample (majority class undersampled) | **Yes:** investigators check k = 100 cards per day; precision top-k, card precision top-k and a normalised version for days with fewer frauds than k | n.r. | Friedman test and paired tests on daily ranks | Private data | AUC and top-k precision rank models differently; daily top-k metrics vary strongly because daily fraud counts vary; recent feedback is informative and the stream is non-stationary | Daily top-K review (R3); capacity-based primary metric; normalisation; day-level dependence in inference | Full |
| REF-086 | Dal Pozzolo et al. (2015) | Real; two European streams (2013: about 160,000 transactions and 304 frauds per day; 2014: about 173,000 and 380) | Daily updates; sliding window and ensemble | Random forests | Balanced resampling in the delayed-label model; none in the feedback model | **Yes:** budget of k = 100 alerts per day; alert precision P_k | n.r. | n.r. | Private data | Training separate classifiers on investigator feedback and on delayed labels gives more precise alerts and faster adaptation to concept drift | Capacity framing; drift motivates the transfer question (H3) | Full |
| REF-078 | Zhang et al. (2026) | Real; IEEE-CIS (two temporal settings) and ULB | Temporal: model fitting, validation, policy selection and future evaluation separated | Logistic regression, LightGBM, TabNet | n.r. in register | **Yes:** top-K with K set per evaluation period (round(q·N)), not per day | Platt-calibrated cost threshold; cost-coefficient sweeps | Descriptive (five training seeds) | Closest comparator | Tight capacity (3 %) compresses policy differences; slack capacity (10 %) separates them | Closest prior art for R3, R4 and RQ3 | Register (full, accepted version) |
| REF-080 | Elbadraoui et al. (2026) | Real; ULB | Chronological 60/20/20 | Deep neural network; random forests | Class weighting compared | No | Validation-selected cost-optimal threshold applied unchanged to test; cost-ratio sweep | One seed; bootstrap interval for test PR-AUC only | — | Class-weighted random forest had higher test cost than unweighted at cost-optimal thresholds | Threshold transfer; RQ3 cost sweep | Register (full) |
| REF-079 | Mazumder (2026) | **Synthetic** anti-money-laundering data | Described both as stratified and as chronological | Several | n.r. in register | Alert-rate ceiling at threshold selection only | F2-maximising threshold | Transaction-level bootstrap | Internal inconsistencies | Threshold chosen under an alert ceiling and transferred unchanged | Bounded comparator only | Register (full) |
| REF-058 | Hossain et al. (2026) | Real; IEEE-CIS and BAF Base | Chronological train/validation/test; future-slice evaluation | Framework (n.r. in abstract) | n.r. | No | Optimal-transport calibration; validation-selected cost-sensitive thresholds | n.r. | — | Temporal evaluation on IEEE-CIS with calibration and cost thresholds | Temporal evaluation on IEEE-CIS is not new | Abstract |
| REF-041 | Kang & Kim (2026) | Real; IEEE-CIS | Chronological out-of-time split and a pseudo-entity-disjoint holdout | Hybrid explainable boosting machines; CatBoost | n.r. | No | n.r. | n.r. | — | CatBoost was the "predictive ceiling" | Temporal evaluation on IEEE-CIS is not new | Abstract |
| REF-098 | Ali et al. (2025) | Real (ULB, IEEE-CIS) and synthetic (PaySim, BankSim) | "Strictly chronological"; dates and proportions not given | Rules, logistic regression, tree ensembles, sequence Transformer, hybrid | Class-weighted or cost-sensitive objectives | No (recall at 1 % false-positive rate) | Brier score; expected monetary utility | None reported | Per-account sequences although IEEE-CIS has no account identifier; IEEE-CIS PR-AUC of 0.83 (trees), 0.89 (Transformer) and 0.91 (hybrid), far above other reports (e.g. 0.29–0.31 in REF-099); a placeholder citation | Claims Transformer and hybrid models beat tree ensembles on all four datasets | Tree-vs-neural claim on IEEE-CIS, but **not verifiable**; cited as weak evidence only | Full |
| REF-099 | Namit et al. (2026) | Real; IEEE-CIS training file (590,540 transactions) | **Stratified 80/20 split** grouped by `card1`; not a temporal holdout | Self-supervised Transformer embeddings + LightGBM / XGBoost vs tabular LightGBM | Class-weighted losses; no resampling | Precision@K over the pooled validation set, not per day | No | None | Validation results only | Embeddings + XGBoost reach PR-AUC 0.31 vs 0.29 for tabular LightGBM | Neural representations add little over trees on IEEE-CIS; pooled, not daily, capacity metric | Full |
| REF-100 | Nguyen et al. (2022) | Real; IEEE-CIS | n.r. in abstract | CatBoost for previously seen users, deep neural network for new users | Imbalance handling (method n.r.) | No | No | n.r. | AUCs come from different user groups, not a head-to-head comparison | AUC 0.97 (CatBoost) and 0.84 (deep network) | Context for trees vs neural on IEEE-CIS | Abstract |
| REF-101 | Almaraz-Rivera et al. (2025) | Real; IEEE-CIS | n.r. in abstract | Self-supervised tabular learning (SubTab) vs XGBoost | n.r. | No | No | n.r. | — | +2.14 % AUROC over XGBoost; 67.44 % of frauds detected | Context for trees vs neural on IEEE-CIS | Abstract |
| REF-102 | San Miguel Carrasco & Sicilia-Urban (2020) | Alerts from an operational fraud-detection system (source n.r. in abstract) | n.r. | Deep neural networks classifying alerts | n.r. | Alert workload reduction | No | n.r. | — | 91.79 % of fraud cases kept with 35.16 % fewer alerts | Analyst workload as the operational constraint | Abstract |
| REF-091 | Makki et al. (2019) | 10 million labelled card records from the Revolution Analytics dataset repository (whether real or simulated is not stated) | Random 70/30; 2 % subsample | Eight classifiers (e.g. C5.0, SVM, ANN, LR) | Resampling, cost-sensitive learning, one-class methods | No | Cost-sensitive learning | None | Random split | Imbalance methods raise sensitivity but produce more false alarms than detected frauds | Resampling cost in false alerts; supports exclusion of resampling | Full |
| REF-093 | Zhao et al. (2024) | ULB, a Sparkov-simulated dataset and a 2020 Kaggle dataset | Stratified 2-fold cross-validation, repeated 5 times | LightGBM variants vs XGBoost, CatBoost, random forest | Class balancing, oversampling, cost-harmonisation loss | No | Cost savings | Repeated folds | Random folds | Combined methods raise F2; class balancing alone may not improve results | Class weighting alone is not reliably beneficial (H2 context) | Full |
| REF-095 | Hayat & Magnier (2025) | Real; ULB | Deliberately flawed: SMOTE before the split | Minimal neural network | SMOTE (as the flaw studied) | No | No | n.r. | Purpose is to expose leakage | Leakage yields 99.9 % recall; four issues: preprocessing leakage, vague reporting, inadequate temporal validation, recall-only metrics | Justifies excluding resampling and using temporal splits | Full (arXiv version) |
| REF-021 | Dal Pozzolo et al. (2015) | Synthetic data and real imbalanced datasets incl. ULB | 10-fold cross-validation | Several | Undersampling | No | Posterior correction for the changed prior; adjusted threshold | Cross-validation | — | Undersampling preserves ranking but distorts probabilities; correcting the posterior restores calibration | Ranking-based policies (R3) unaffected by rebalancing; R4 needs calibration | Full |

---

## 2. Benchmarks, reviews and methodological sources

| ID | Study | Type | Scope | Key finding | Relevance | Read |
|---|---|---|---|---|---|---|
| REF-087 | Gorishniy et al. (2021) | Benchmark (NeurIPS) | 11 public datasets, none fraud, one with categorical features; one fixed train/validation/test split each; Optuna tuning; 15 seeds | ResNet-style MLP is a strong baseline; FT-Transformer is the best deep model on most tasks; no universally superior solution between deep models and gradient-boosted trees. FT-Transformer cost grows quadratically with the number of features | Primary source for the ResNet-style and FT-Transformer models; tuning protocol; hardware feasibility | Full (NeurIPS and arXiv v5 with appendix) |
| REF-089 | McElfresh et al. (2023) | Benchmark (NeurIPS) | 19 algorithms, 176 datasets, 10 folds | The neural-vs-tree gap is often negligible; light tuning of CatBoost or ResNet often matters more than the family; trees handle skewed and heavy-tailed features better and tend to win on larger datasets | Expectation for the tree-vs-neural comparison; tuning fairness | Full (arXiv) |
| REF-088 | Kadra et al. (2021) | Benchmark (NeurIPS) | 40 datasets | Plain MLPs with a tuned "cocktail" of 13 regularisers outperform specialised deep models and XGBoost | Counter-view: well-regularised neural networks can compete | Full |
| REF-090 | Borisov et al. (2024) | Survey and benchmark (IEEE TNNLS) | Five datasets; stratified 5-fold cross-validation | Gradient-boosted trees outperform deep models except on the very large HIGGS dataset | Background for tree-vs-neural expectations | Full |
| REF-013 | Grinsztajn et al. (2022) | Benchmark (NeurIPS) | 45 medium-sized datasets | Tree-based models remain state of the art on medium-sized tabular data | As above | Abstract |
| REF-014 | Shwartz-Ziv & Armon (2022) | Benchmark | Several datasets | XGBoost outperforms the deep models tested and needs less tuning | As above | Abstract |
| REF-094 | Baisholan et al. (2025) | Systematic review (PRISMA 2020) | Only studies preserving the original class imbalance in training and evaluation | Reliance on a few benchmark datasets; preference for tree ensembles; limited use of AUC-PR; need for cost-aware evaluation | Supports keeping the natural imbalance; supports capacity- and cost-aware evaluation | Abstract |
| REF-092 | Kalid et al. (2024) | Systematic review (PRISMA) | 87 studies, 2016–2023 | European and Taiwan datasets dominate; recommends deep learning, ensembles and sampling; metrics: recall, AUC, precision, F1 | Shows capacity-based metrics are absent from mainstream evaluation; counter-view on sampling | Full |
| REF-097 | Dastidar et al. (2024) | Survey (IEEE Access) | Online card-fraud detection research | Taxonomy: skewed data, evolving behaviour, transaction context | Background for problem framing | Abstract |
| REF-096 | Nesvijevskaia et al. (2021) | Review (Data & Policy) | Fraud management in banking | Investigation is "strongly constrained by available human resources"; cites an industry study (LexisNexis) that one-fourth of fraud costs go to manual review | HITL and capacity motivation (the cost share is a secondary industry figure) | Full |
| REF-019 | van den Goorbergh et al. (2022) | Simulation study | Logistic regression; medical case study | Imbalance corrections worsen calibration without improving discrimination | Supports class weighting over resampling and calibration before R4 | Register |
| REF-020 | Carriero et al. (2025) | Simulation study | Machine-learning algorithms incl. XGBoost | Models without imbalance correction had equal or better calibration in all scenarios; miscalibration from correction was not always fixable by recalibration | As above | Abstract |

---

## 3. Coverage of this study's design elements by the closest studies

"Yes" = the element is present; "Partly" = a related but weaker form; "No" = absent. Proposed elements that are not yet part of the frozen design are marked †.

| Design element | REF-078 | REF-080 | REF-079 | REF-058 | REF-098 | REF-099 | REF-005 / REF-086 |
|---|---|---|---|---|---|---|---|
| Real IEEE-CIS data with a chronological holdout | Yes | No (ULB) | No (synthetic) | Yes | Claimed, not verifiable | No (stratified) | No (private data; temporal) |
| Fixed **daily** review budget K | No (per period) | No | No | No | No | No (pooled) | Yes (k per day) |
| Daily top-K review policy (R3) | Partly (per period) | No | No | No | No | Partly (pooled) | Yes |
| Calibrated cost-aware policy capped at capacity (R4) | Yes | No | No | Partly (no cap) | No | No | No |
| Static threshold transferred to later data | Partly (rate, not score) | Yes | Yes | Yes | n.r. | No | No |
| Formal test of the transferred threshold's alert rate (H3) | No | No | No | No | No | No | No |
| Class weighting compared, with an equivalence test (H2) | No | Partly (descriptive) | No | No | No | No | No |
| Tree vs neural families under one tuning procedure † | Partly (LR, LightGBM, TabNet) | No | No | n.r. | Claimed, procedure not reported | Partly | No |
| Cost-ratio sweep with break-even ratios (RQ3) | Partly (sweep) | Partly (sweep) | No | No | No | No | No |
| Dependence-aware inference (day-level bootstrap, multiplicity control) | No (seeds) | No | No | n.r. | No | No | Partly (rank tests over days) |

**Reading [I].** Every element except the formal alert-rate transfer test appears somewhere, but no study reviewed combines a fixed daily budget, a pre-specified transfer test and dependence-aware inference on real chronological data. This is consistent with the gap stated in [`literature-review.md`](literature-review.md), Section 8.

---

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [I] | Evidence tag: inference (not a statement from the source) |
| † | Proposed design element, not yet adopted in the protocol |
| ANN | Artificial neural network |
| AUC, AUROC | Area under the receiver operating characteristic curve |
| AUC-PR, PR-AUC | Area under the precision–recall curve |
| BAF Base | Bank Account Fraud dataset suite, base variant |
| F1, F2 | Harmonic mean of precision and recall; variant weighting recall twice as heavily |
| FT-Transformer | Feature Tokenizer + Transformer model for tabular data (REF-087) |
| H2, H3 | Hypotheses: class weighting; alert-rate transfer fidelity |
| IEEE-CIS | IEEE Computational Intelligence Society Fraud Detection dataset |
| IEEE TNNLS | *IEEE Transactions on Neural Networks and Learning Systems* |
| K, k | Number of transactions (or cards) that can be reviewed per day |
| LR | Logistic regression |
| MLP | Multilayer perceptron |
| n.r. | Not reported in the text that was read |
| N, q | Period size and capacity rate in REF-078's per-period budget round(q·N) |
| NeurIPS | Conference on Neural Information Processing Systems |
| P_k | Alert precision: share of frauds among the k daily alerts |
| PRISMA | Preferred Reporting Items for Systematic Reviews and Meta-Analyses |
| R3, R4 | Decision policies: daily top-K review; calibrated cost-aware capped policy |
| REF-NNN | Reference identifier in `references.md` |
| RQ3 | Research question 3 (cost-aware capacity use) |
| SMOTE | Synthetic Minority Over-sampling Technique |
| SVM | Support vector machine |
| ULB | Université Libre de Bruxelles `creditcard.csv` dataset |
