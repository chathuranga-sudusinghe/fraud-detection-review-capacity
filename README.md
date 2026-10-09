# Fraud Detection Under Limited Review Capacity

MSc Data Science research project on temporally validated fraud-alert prioritisation under constrained review capacity.

**University:** University of Greenwich  
**Module:** COMP1885 Individual Project  
**Research type:** Quantitative, non-participant, secondary-data experimental study

## Working Research Title

**Fraud Detection Under Limited Review Capacity: A Temporally Validated Study of Tree-Based and Neural Models and Decision Policies**

## Start here

**The study in one paragraph.** Fraud teams can review only a limited number of transactions per day, so a fraud model is useful only through the rule that turns its scores into a daily review queue. Using the IEEE-CIS Fraud Detection data (IEEE-CIS = IEEE Computational Intelligence Society), this study trains two tree-based models (LightGBM, CatBoost) and two neural models (ResNet-style multilayer perceptron, FT-Transformer), each with and without class weighting, on the earliest period, selects thresholds and review rules on later validation periods, and evaluates them **once** on a sealed final period. The contribution is an integrated, pre-specified evaluation of this full path. Its most distinctive component is a test of whether a score threshold chosen on validation data still produces its intended alert rate on later data (research question RQ2, hypothesis H3). The study also compares the tree-based and neural families and class weighting at fixed capacity (RQ1, hypotheses H1 and H2) and asks when a calibrated cost-aware rule is cheaper than always filling the review budget (RQ3).

**Reading order.**

1. This page: research questions, research path and repository map.
2. [Research protocol](docs/protocol.md): the full design, open items and freeze conditions.
3. [Decision Records](docs/decisions/README.md) (DR-001 onwards): why each design choice was made.
4. [Dataset](docs/dataset.md) and [notebooks](notebooks/README.md): the data, its use terms and the data-understanding evidence.

## Dataset source

**Dataset:** IEEE-CIS Fraud Detection, labelled training data (`train_transaction.csv`, `train_identity.csv`).

**Official source:** [IEEE-CIS Fraud Detection — Kaggle competition data](https://www.kaggle.com/competitions/ieee-fraud-detection/data) [REF-047].

**Competition host:** IEEE Computational Intelligence Society (IEEE-CIS), on Kaggle.

**Data provider:** Vesta Corporation; the data are real-world e-commerce transactions.

**Data-use terms:** IEEE-CIS Fraud Detection Competition Rules, Section 7 [REF-049]: non-commercial use, including academic research and education; no redistribution to anyone who has not accepted the Rules. Raw data stay on the local machine and are never committed ([data-use terms](docs/dataset.md#data-use-terms)).

**Use in this study:** the data are used exactly as published. No rows are generated, simulated or altered ([data provenance](docs/dataset.md#data-provenance)).

**Why this dataset.** It is a real-world, public transaction-fraud benchmark with transaction timing, which allows a chronological split, and it is used by the prior fraud-detection studies compared in the [literature evidence matrix](docs/literature-evidence-matrix.md). Its defined boundaries are documented in the [dataset documentation](docs/dataset.md): most feature meanings are masked by the provider, the time reference point is undisclosed, and the labelling rule is the organiser's.

## Main Research Question

> **Under constrained daily review capacity, how do model family, class weighting, and score-to-review decision policies affect fraud capture, review workload, and cost-sensitive outcomes on temporally unseen transaction data?**

## Supporting Research Questions

The supporting research questions are numbered RQ1–RQ3.

### RQ1

> **How do tree-based and neural model families, with and without class weighting, differ in fraud capture at fixed review capacity on a later temporal test period?**

### RQ2

> **When a score threshold selected on validation data is transferred to a later temporal period, how well does it preserve the intended alert rate, and how does the resulting daily review workload compare with the fixed review budget?**

Defined in [DR-003](docs/decisions/DR-003-rq2-h3-policy-transfer-fidelity.md).

### RQ3

> **Under different assumed miss-to-review cost ratios, when does a calibrated cost-aware review policy outperform always filling the available review capacity?**

## Confirmatory Hypotheses

The confirmatory hypotheses are H1–H3, worded exactly as in the [research protocol](docs/protocol.md). A subscript ₀ marks the null hypothesis and ₁ the alternative. Recall@2% is the share of Test-period frauds that are reviewed under daily top-K review: on each relative day the K highest-scoring transactions are reviewed, with the fixed daily review budget K = 56. "2 %" names the capacity scenario from which K is derived (2 % of mean daily transaction volume in Validation 2; [DR-002](docs/decisions/DR-002-review-capacity-definition.md)); the share actually reviewed on a given day varies with that day's volume.

Symbols used below (full definitions and equations are in the protocol):

- `δ_F` = model-family effect: the neural-minus-tree difference in Recall@2%, averaged over the two models of each family (tree-based: LightGBM, CatBoost; neural: ResNet-style MLP, FT-Transformer) and over the unweighted and balanced-weighting conditions;
- `δ_W,m` = weighting effect of model m: balanced-minus-unweighted Recall@2%;
- `r` = realised Test alert rate of the validation-selected 2 % threshold. For confirmatory H3 this is the threshold of one rule-selected H3 configuration, chosen before Test access ([DR-010](docs/decisions/DR-010-h3-confirmatory-configuration.md));
- `Δ` = H3 equivalence tolerance on r (0.004, i.e. ±20 % of the 2 % target).

### H1 — Model-family effect

**H1₀:** Averaged over the two models of each family and over the unweighted and balanced-weighting conditions, tree-based and neural models do not differ in Recall@2% on the Test period (δ_F = 0).

**H1₁:** Averaged over the two models of each family and over the unweighted and balanced-weighting conditions, tree-based and neural models differ in Recall@2% on the Test period (δ_F ≠ 0).

### H2 — Class-weighting effect

**H2₀:** For at least one of the four models (LightGBM, CatBoost, ResNet-style MLP or FT-Transformer), balanced class weighting changes Recall@2% by 2 percentage points or more relative to the corresponding unweighted model (|δ_W,m| ≥ 0.02 for at least one m).

**H2₁:** For all four models, balanced class weighting changes Recall@2% by less than 2 percentage points relative to the corresponding unweighted model (|δ_W,m| < 0.02 for every m).

### H3 — Alert-rate transfer fidelity

**H3₀:** The realised alert rate under the validation-selected 2% threshold lies outside the pre-specified equivalence band around the 2% target (|r − 0.02| ≥ Δ, where Δ = 0.004, i.e. ±20% of the target).

**H3₁:** The realised alert rate under the validation-selected 2% threshold lies within the pre-specified equivalence band around the 2% target (|r − 0.02| < Δ, where Δ = 0.004, i.e. ±20% of the target).

H1 is a two-sided difference test. H2 and H3 are equivalence hypotheses tested with Two One-Sided Tests (TOST). The Smallest Effect Size Of Interest (SESOI) for H1 and H2 is 2 percentage points of Recall@2% ([DR-005](docs/decisions/DR-005-sesoi-and-h1-h2-inference.md)); Δ is justified in [DR-003](docs/decisions/DR-003-rq2-h3-policy-transfer-fidelity.md). H3 is evaluated confirmatorily on one of the eight model × weighting configurations, selected on Validation 2 by a frozen rule before Test access ([DR-010](docs/decisions/DR-010-h3-confirmatory-configuration.md)); the other seven are reported descriptively. The model set is defined in [DR-011](docs/decisions/DR-011-model-set-tree-based-and-neural-families.md). Holm's procedure controls the family-wise error rate over H1–H3. Uncertainty for H1/H2 uses the two-way bootstrap of [DR-006](docs/decisions/DR-006-bootstrap-inference-strategy.md).

RQ3 is primarily estimation and sensitivity analysis rather than a separate significance-testing exercise. Its cost-ratio grid and break-even rule are defined in [DR-007](docs/decisions/DR-007-rq3-cost-ratio-and-break-even.md).

## Model selection

Models are selected by a fixed rule on development data only. The Test period is never used for selection ([DR-010](docs/decisions/DR-010-h3-confirmatory-configuration.md), [DR-011](docs/decisions/DR-011-model-set-tree-based-and-neural-families.md)).

| Stage | Data | Rule |
|---|---|---|
| Hyperparameters | V1 (relative days 106–126) | Each of the eight configurations is tuned with the same search method and budget, the same V1 objective and the same early-stopping rule. |
| Validation-selected configuration | V2 (relative days 127–154) | The configuration with the **highest pooled V2 Recall@2% under R3** (daily top-K review, K = 56) is selected. Ties follow the fixed order LGB-u, Cat-u, RES-u, FTT-u, LGB-b, Cat-b, RES-b, FTT-b. The eight V2 values and the selected configuration are committed to the repository before the Test period is unsealed. |
| Confirmation | Test (relative days 155–182) | No selection. All eight configurations are evaluated once. Conclusions about model family and class weighting come only from H1 and H2. |

Pooled V2 Recall@2% is the share of all V2 frauds that fall inside the daily top 56 scores of their relative day. The selected configuration is the deployment proxy whose threshold transfer H3 tests. It is reported as the **validation-selected configuration**, not as the best model, because differences between configurations on V2 can be smaller than their sampling error.

## Research path

The study is organised along the ten stages of the Foundational Methodology for Data Science published by IBM (International Business Machines Corporation) [REF-085], adapted to a confirmatory research design. This framing is proposed and pending supervisor review; the adaptations below will be recorded in a Decision Record after that review.

- Stages 1–8 carry the confirmatory research; stage 9 (Deployment) is realised as a local demonstration application and stage 10 (Feedback) as supervisor review and reporting.
- Iteration between stages is allowed only before the protocol freeze, only on the development partitions, and only within each partition's fixed role: Train for model fitting; Validation 1 (V1) for hyperparameter tuning and early stopping; Validation 2 (V2) for calibration, thresholds, the daily review budget and the H3 configuration. V2 results never feed back into hyperparameters or features.
- The Test evaluation runs once and nothing is refined from it.

| # | Stage (IBM) | In this study | Status | Evidence |
|---|---|---|---|---|
| 1 | Business understanding | Research problem, gap, RQ1–RQ3, H1–H3; structured literature review | Done; design review with the supervisor pending | [Protocol](docs/protocol.md), [research foundation](docs/research-foundation.md), [literature review](docs/literature-review.md) |
| 2 | Analytic approach | Predictive (fraud scores), prescriptive (score-to-review policies R0–R4 under capacity: default 0.5 threshold, F1-optimal threshold, capacity-rate threshold, daily top-K, calibrated cost-aware capped policy), inferential (pre-specified tests) | Documented in the protocol; explicit statement pending | [Protocol](docs/protocol.md) |
| 3 | Data requirements | Data each RQ/H needs, and what the dataset cannot provide (real costs, real capacity, label report dates) | Pending (traceability matrix) | [Dataset](docs/dataset.md) |
| 4 | Data collection | Secondary data: source, use terms, checksums, manifest | Use terms verified; checksums recorded; manifest pending (protocol design item 8) | [Dataset](docs/dataset.md), notebooks 01–02 |
| 5 | Data understanding | Schema, join, time structure, class imbalance, drift, entity dependence | Done | [Notebooks 01–03](notebooks/README.md), [feature catalog](docs/feature-catalog.md) |
| 6 | Data preparation | Chronological split; feature policy; validation contract; data pipeline | Split and feature policy done; validation contract and pipeline pending | [DR-001](docs/decisions/DR-001-temporal-split-strategy.md), [DR-008](docs/decisions/DR-008-feature-declaration-and-model-input-policy.md) |
| 7 | Modeling | Tree-based (LightGBM, CatBoost) and neural (ResNet-style MLP, FT-Transformer) × unweighted/balanced; common training procedure; development runs | Model set defined; GPU feasibility verified (Issue #50); training rules pending | [Protocol](docs/protocol.md), [DR-011](docs/decisions/DR-011-model-set-tree-based-and-neural-families.md) |
| 8 | Evaluation | H1–H3, RQ3 analysis, false-negative slices; one Test evaluation after freeze | Design defined; runs after freeze | DR-002, DR-003, DR-005–DR-007, DR-009, DR-010 |
| 9 | Deployment (demonstration) | Research report; local demonstration application programming interface (API) on synthetic example requests (not IEEE-CIS rows) | Planned | — |
| 10 | Feedback (review) | Supervisor review; limitations; future work | Ongoing | — |

Two gates separate development from confirmation: **(1)** training, policy and analysis procedures are fixed before the development runs; **(2)** the fitted models, calibrators, thresholds, review budgets and H3 configuration are frozen, and the protocol is tagged, before the Test period is scored.

## Repository map

| Path | Contents |
|---|---|
| [`docs/protocol.md`](docs/protocol.md) | Research protocol: design, policies, statistics, open items, freeze conditions |
| [`docs/decisions/`](docs/decisions/README.md) | Decision Records DR-001–DR-011 (methodology) |
| [`docs/research-foundation.md`](docs/research-foundation.md) | Problem, aims, objectives, contribution |
| [`docs/literature-review.md`](docs/literature-review.md) | Literature review and research gap |
| [`docs/literature-evidence-matrix.md`](docs/literature-evidence-matrix.md) | Field-by-field comparison of the selected studies |
| [`docs/literature-search-method.md`](docs/literature-search-method.md) | How the literature was searched, screened and selected |
| [`docs/references.md`](docs/references.md) | Reference register with verification status |
| [`docs/dataset.md`](docs/dataset.md) | Dataset source, data-use terms, version-control policy |
| [`docs/feature-catalog.md`](docs/feature-catalog.md) | Column-by-column data dictionary |
| [`config/features.yaml`](config/features.yaml) | Machine-readable model-input roles (431 inputs) |
| [`notebooks/`](notebooks/README.md) | Data-understanding notebooks 01–03 |
| [`scripts/`](scripts/) | Engineering scripts (GPU memory feasibility probe, Issue #50) |
| [`AGENTS.md`](AGENTS.md) | Working rules for the project and AI-assisted work |
| `data/raw/` | Local raw data (never committed) |

## Dataset and temporal split

The labelled `train_transaction.csv` and `train_identity.csv` files of the IEEE-CIS Fraud Detection dataset ([Dataset source](#dataset-source)) are left-joined on `TransactionID` (590,540 transactions, 434 columns).

The data are split chronologically by relative day (`TransactionDT` in whole days from an undisclosed reference point):

| Partition | Relative days | Rows | Role |
|---|---|---|---|
| Train | 1–105 | 368,632 | Model fitting |
| Validation 1 (V1) | 106–126 | 65,544 | Tuning and early stopping |
| Validation 2 (V2) | 127–154 | 77,822 | Calibration, thresholds, daily budget K, H3 configuration |
| Test | 155–182 | 78,542 | **Sealed**; one confirmatory evaluation after protocol freeze |

Development prevalence (days 1–154) is 3.49 % fraud, about 28 legitimate transactions per fraud ([notebook 03](notebooks/03_merged_dataset_inspection.ipynb)). Test prevalence is not computed. See [Dataset documentation](docs/dataset.md) and [DR-001](docs/decisions/DR-001-temporal-split-strategy.md).

## Research Focus

The research focuses on:

- temporal generalisation;
- tree-based (LightGBM, CatBoost) and neural (ResNet-style MLP, FT-Transformer) model families;
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

## Status

**Research design established; data understanding complete; protocol not yet frozen.**

The ethics and governance status is documented in the [protocol](docs/protocol.md#ethics): topic approval was received, and no separate explicit ethics-route determination was provided despite the request; no formal ethics approval or exemption is claimed.

Next steps:

1. Design review with the supervisor (Issue #54) and resolution of the protocol's design items.
2. Research path stages 2–4: explicit analytic approach, data-requirements traceability matrix, dataset manifest.
3. Data preparation: validation contract, split reproduction and data pipeline.
4. Training rules: search procedure, early stopping, tie handling, seeds, neural-network representation, calibration details.
5. Development runs on Train, V1 and V2; protocol freeze; single Test evaluation; statistical analysis and reporting.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| Δ | H3 equivalence tolerance on r (0.004 = ±20 % of the 2 % target) |
| API | Application programming interface (local demonstration only) |
| δ_F | Model-family effect in Recall@2% (neural mean minus tree-based mean, averaged over models and weighting) |
| δ_W,m | Weighting effect of model m (balanced minus unweighted Recall@2%) |
| DR-NNN | Decision Record NNN in `docs/decisions/` |
| LGB, Cat, RES, FTT; -u, -b | LightGBM, CatBoost, ResNet-style MLP, FT-Transformer; unweighted, balanced class weighting (e.g. LGB-u) |
| m | A model: LightGBM, CatBoost, ResNet-style MLP or FT-Transformer |
| MLP | Multilayer perceptron |
| GPU | Graphics processing unit |
| H1, H2, H3 | Confirmatory hypotheses; subscript ₀ = null hypothesis, ₁ = alternative |
| IBM | International Business Machines Corporation (author of the methodology framing, REF-085) |
| IEEE-CIS | IEEE Computational Intelligence Society |
| K | Fixed daily review budget (reviews per day) |
| ML | Machine learning |
| Precision@Capacity | Share of reviewed transactions that are fraud at a fixed review capacity |
| r | Realised Test alert rate of the validation-selected 2 % threshold (confirmatory H3: the rule-selected H3 configuration only) |
| R0–R4 | Decision policies: default 0.5 threshold; F1-optimal threshold; capacity-rate threshold; daily top-K; calibrated cost-aware capped policy |
| Recall@2%, Recall@Capacity | Share of frauds reviewed at 2 % (a fixed) review capacity |
| REF-NNN | Reference identifier in `docs/references.md` |
| Relative day | `TransactionDT` divided by 86,400 seconds, rounded down; no calendar date inferred |
| RQ1–RQ3 | Research questions 1–3 |
| SESOI | Smallest Effect Size Of Interest |
| TOST | Two One-Sided Tests |
| `TransactionDT`, `TransactionID` | Time offset in seconds from an undisclosed reference point; row identifier and join key |
| V1, V2 | Validation 1 (relative days 106–126); Validation 2 (relative days 127–154) |
