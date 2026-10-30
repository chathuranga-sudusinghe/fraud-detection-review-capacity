# Research Methodology Decision Records

This folder records methodology decisions: what was decided, on what evidence, which alternatives were rejected, and what the decision costs. Records are short, and their numbers are permanent. How a record changes depends on whether the protocol has been frozen (see [Record maintenance policy](#record-maintenance-policy)).

## Record maintenance policy

| | Before the protocol freeze (current stage) | After the protocol freeze (protocol tagged in Git) |
|---|---|---|
| Methodology records (*Proposed*, *Accepted (pre-freeze)*) | Revised **in place** through a GitHub Issue and pull request, so that each record describes the current design only | Not edited (*Frozen*). A change requires a **new record that supersedes** the earlier one, whose status becomes *Superseded by DR-NNN* |
| Process and governance records (*Accepted*) | Revised in place through a GitHub Issue and pull request | Revised in place through a GitHub Issue and pull request |
| Revision history | Kept in Git commits, pull requests and issues, not as narrative inside the records | Visible in the records through supersession, and in Git |

At every stage:

- record numbers are permanent: a record is never renumbered, reused or deleted;
- each record keeps its "Alternatives considered" content: why rejected options were not chosen is research content, not history;
- status and approval information stays truthful, including anything still pending review;
- saved evidence, such as executed notebooks, is never altered to match a revised record.

## Conventions

- **File name:** `DR-NNN-short-kebab-title.md`, numbered sequentially from `DR-001`. Numbers are never reused.
- **Status values:**
  - *Proposed* — drafted, not yet applied to the protocol.
  - *Accepted (pre-freeze)* — applied to the working protocol; revised in place if the design changes before the freeze.
  - *Frozen* — part of the tagged confirmatory protocol; changed only by supersession.
  - *Accepted* — in force; used for process and governance decisions that are not part of the confirmatory protocol (e.g. DR-004).
  - *Superseded by DR-NNN* — replaced after the freeze by the named record.
- **Section template** (methodology records): a header table (Status, Date, Evidence source, Related), then **Context → Decision → Evidence → Alternatives considered → Rationale → Consequences → Threats and limitations**, an optional appendix for long derivations, **References** and **Symbols and notation**. Detailed specifications (e.g. statistical formulations, mathematical definitions) are subsections of the main section they belong to. A section is omitted only when the record has no content for it; reasons that concern one part of the decision may stay beside that part, and a separate Rationale section collects the record-level reasons. *Status* states the status and approval; *Date* states when the record was created and revised, and why. Process and governance records (DR-004) follow their own structure.
- **Evidence tags** (as in the feasibility notebook):
  - **[D]** observed in the dataset;
  - **[L]** literature, cited by `REF-NNN` from [`../references.md`](../references.md);
  - **[G]** University of Greenwich guidance;
  - **[I]** inference or recommendation.
- **Test-set rule:** no record may rely on target-derived information from the sealed Test period (relative days 155–182).

## Index

| ID | Title | Status | Date | Affects |
|---|---|---|---|---|
| [DR-001](DR-001-temporal-split-strategy.md) | Temporal split strategy | Accepted (pre-freeze) | 2026-09-23; revised 2026-10-09, 2026-10-10, 2026-10-30 (Issue #185) | Split structure; roles of V1 (Validation 1) and V2 (Validation 2), including H3-configuration selection on V2; no primary gap |
| [DR-002](DR-002-review-capacity-definition.md) | Review-capacity definition | Accepted (pre-freeze) | 2026-09-23; revised 2026-10-09 | Capacity grid; fixed daily review budget K; primary metric interpretation; pooled normalised recall |
| [DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md) | RQ2 / H3: policy-transfer fidelity | Accepted (pre-freeze) | 2026-09-23; revised 2026-09-24, 2026-10-09, 2026-10-30 (Issues #183, #185) | RQ2 (research question 2) wording; H3 (alert-rate transfer hypothesis) equivalence formulation, tolerance Δ = 0.004 (±20 % relative) and Holm-level decision; budget-fidelity outcomes and rate/volume decomposition |
| [DR-004](DR-004-github-issue-driven-workflow.md) | GitHub issue-driven workflow | Accepted | 2026-09-23; revised 2026-10-09, 2026-10-11 | Project workflow; artificial intelligence (AI)-assisted work rules (see [`AGENTS.md`](../../AGENTS.md)) |
| [DR-005](DR-005-sesoi-and-h1-h2-inference.md) | SESOI and H1/H2 inference plan | Accepted (pre-freeze) | 2026-09-24; revised 2026-10-09, 2026-10-30 (Issue #183) | Smallest Effect Size Of Interest (SESOI) = 2 percentage points (pp) of Recall@2%; H1 (model-family hypothesis: neural minus tree-based family) difference test with practical interpretation; H2 (class-weighting hypothesis) intersection-union equivalence over four models; bounded bootstrap p-values; Holm multiple-testing family {H1, H2, H3} |
| [DR-006](DR-006-bootstrap-inference-strategy.md) | Bootstrap inference strategy for H1 and H2 | Accepted (pre-freeze) | 2026-09-24; revised 2026-10-09, 2026-10-30 (Issues #183, #185) | Primary: two-way pigeonhole bootstrap (relative day × fine pseudo-entity) over eight configurations; sensitivity: Owen-corrected variance and day-block bootstrap (block length ℓ = 3 days); percentile confidence intervals (CIs), B = 10,000 bootstrap replicates |
| [DR-007](DR-007-rq3-cost-ratio-and-break-even.md) | RQ3 cost-ratio grid and break-even rule | Accepted (pre-freeze) | 2026-09-24; revised 2026-10-09, 2026-10-17 (Issue #91) and 2026-10-30 (Issues #183, #185) | Primary cost model (review + missed-fraud cost; the extra false-alert cost C_FP as a sensitivity relabelling); miss-to-review cost-ratio grid ρ ∈ {2, …, 1000}, exact curves over (1, 1000]; saturation and break-even ratios per capacity and configuration; amount outcomes as exposure proxies; R4 calibration: Platt scaling on the log-odds of the raw score with smoothed targets, isotonic check on the raw score, intervals conditional on the frozen V2 calibrators |
| [DR-008](DR-008-feature-declaration-and-model-input-policy.md) | Feature declaration and model-input policy | Accepted (pre-freeze) | 2026-09-24; revised 2026-10-09, 2026-10-10, 2026-10-12, 2026-10-13 | 49 categorical columns (organiser list), 382 numeric, 2 excluded (`TransactionID`, `TransactionDT`); one canonical input for all four models; missing categorical label; high-cardinality policy; tree-based model representation, with CatBoost's ordered target statistics in time order; population preservation (no corrective row removal or value change); `config/features.yaml` |
| [DR-009](DR-009-overflow-handling-for-threshold-policies.md) | Overflow handling for threshold-based policies R0, R1 and R2 | Accepted (pre-freeze) | 2026-09-25; revised 2026-10-09 and 2026-10-17 (Issue #115) | Report-only overflow rule: R0–R2 evaluated uncapped; alert demand A, overflow O, unused capacity U and load ratio L defined against the daily budget K; truncation to top-K rejected (it reproduces R3); overflow frauds not counted as missed or false negatives; downstream overflow handling out of scope; R1 threshold selection and exact tie handling (Issue #115) |
| [DR-010](DR-010-h3-confirmatory-configuration.md) | Confirmatory configuration for H3 | Accepted (pre-freeze) | 2026-09-26; revised 2026-10-09; selection result 2026-10-30 | H3 confirmatory for one of the eight model × weighting configurations, selected by a pre-specified rule (highest V2 Recall@2% under R3) before the Test period is unsealed; selected: LGB-b (Issue #181); Holm family {H1, H2, H3}; other configurations' H3 results descriptive, including a tree-vs-neural comparison of threshold transfer; defines the rolling-origin supplement's primary configuration |
| [DR-011](DR-011-model-set-tree-based-and-neural-families.md) | Model set: tree-based and neural model families | Accepted (pre-freeze) | 2026-10-09; revised 2026-10-10, 2026-10-12, 2026-10-15, 2026-10-17 (Issue #115) | Eight configurations: LightGBM, CatBoost, ResNet-style multilayer perceptron (MLP), Feature Tokenizer + Transformer (FT-Transformer) × unweighted/balanced; common-procedure fairness rule; training environment (local graphics processing unit (GPU), LightGBM on the central processing unit (CPU), bfloat16 mixed precision for the neural models); one unweighted, tuned logistic-regression reference reported descriptively outside H1–H3 (Issue #115) |
| [DR-012](DR-012-neural-network-input-representation.md) | Neural-network input representation | Accepted (pre-freeze) | 2026-10-09; separate record 2026-10-10; accepted 2026-10-17 (Issue #107); revised 2026-10-17 (Issue #112) | Identical representation for the ResNet-style MLP and FT-Transformer: category vocabularies with reserved `"__MISSING__"` and category dropout; quantile transformation with reference-implementation noise; missingness indicators; Train-only fitting; one shared Train-only preprocessing fitted with noise seed 0 |
| [DR-013](DR-013-training-device-policy.md) | Training-device policy | Accepted (pre-freeze) | 2026-10-12; revised 2026-10-15 | GPU training for CatBoost and both neural models through CUDA (NVIDIA's GPU computing platform); deterministic CPU training for LightGBM, whose CUDA learner fails on categorical features above 256 categories; model-specific device controls set explicitly (LightGBM without feature bundling, with column-wise histograms; CatBoost CPU-reference quantisation; neural precision and determinism); main seed 42; reproducibility and prediction rules; supportive CPU–GPU comparison for CatBoost; synthetic acceptance probe passed before development runs |
| [DR-014](DR-014-research-methodology-framing.md) | Research-methodology framing | Accepted (pre-freeze) | 2026-10-13 | Study organised along the ten stages of the Foundational Methodology for Data Science published by International Business Machines Corporation (IBM), adapted to the confirmatory design: iteration only on the development partitions before the freeze; one Test evaluation with no refinement; demonstration deployment; feedback as supervisor review and reporting; reproducibility and data-governance additions; CRISP-DM (Cross-Industry Standard Process for Data Mining) and Microsoft TDSP (Team Data Science Process) as alternatives |
| [DR-015](DR-015-common-training-rules.md) | Common training rules | Accepted (pre-freeze) | 2026-10-16; revised 2026-10-17 (Issues #112, #115), 2026-10-23 (Issue #151) | Random search with 10 trials per configuration (trial 1 = reference configuration; one candidate list per model for both weightings); early stopping on V1 (trees: patience 50 rounds, maximum 2,000; neural: patience 5 epochs, maximum 15); V1 objective: V1 Recall@2% under daily top-62 review, unweighted; main seed 42, with fixed-length Train-only refits (additional seeds 43 and 44; same-seed repeats) evaluated descriptively on V2; exact tie rules and the R3 cutoff tie rule (SHA-256, Secure Hash Algorithm, 256-bit, of `TransactionID`); tree-based search spaces (CatBoost: the large-dataset space of Gorishniy et al. (2021); LightGBM: Gorishniy et al. (2024); learning rate log-uniform on [0.01, 0.3] for both) with library-default reference configurations (Issue #106); neural search spaces (the large-dataset spaces of Gorishniy et al. (2021)) with the published FT-Transformer default and an adapted TabZilla ResNet default as reference configurations, an effective batch of 1,024 rows with a pre-search micro-batch plan and row-count loss normalisation (Issue #107); neural preprocessing seed 0 in a dedicated generator (Issue #112); search and fitting of the logistic-regression reference (Issue #115) |

## Source of evidence

DR-001 to DR-003 draw on [`notebooks/01_dataset_feasibility_audit.ipynb`](../../notebooks/01_dataset_feasibility_audit.ipynb). Its conclusion was *"Suitable with methodological changes."* DR-004 is a project-governance decision (GitHub Issue #6). DR-005 (Issue #8) draws on the notebook's power projections and the equivalence-testing literature. DR-006 (Issue #14) draws on development-period (V2) dependence diagnostics and the clustered-bootstrap literature. DR-007 (Issue #13) draws on a decision-theoretic derivation and the cost-sensitive, cost-curve and decision-curve literature; it uses no new dataset computations. DR-008 (Issue #23) draws on the raw-file schema, development-period values (target never loaded), the organiser's data description and the LightGBM and CatBoost documentation. DR-009 (Issue #27) draws on the protocol's policy definitions, earlier Decision Records and an algebraic argument; it uses no labels or model scores. DR-010 (Issue #29) draws on the protocol's hypothesis and multiplicity definitions and on DR-005's power projections; its rule uses no labels, model scores or new dataset computations, and its recorded selection result (Issue #181) applies the rule to the development-run V2 scores. DR-011 (Issue #54) draws on the structured literature search and evidence matrix (Issue #52) and on the GPU memory feasibility probe on synthetic inputs (Issue #50); DR-012 (Issue #54, as part of DR-008; separate record by Issue #57) draws on the primary source of the neural architectures and its reference implementation, a synthetic check of the quantile transformation and the same probe; the Issue #54 revisions of DR-001–DR-003 and DR-005–DR-010 extend them to the eight-configuration model set and refine their statistical specifications, without new dataset computations. DR-013 (Issue #66) draws on the official documentation and versioned source code of LightGBM, CatBoost, PyTorch and cuBLAS (CUDA Basic Linear Algebra Subprograms), on the primary source of the neural architectures and on the GPU probe of Issue #50. DR-014 (Issue #85) draws on the IBM methodology white paper, two peer-reviewed reviews of data-science process methodologies and University of Greenwich guidance; it uses no dataset values, labels or model results.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [D], [L], [G], [I] | Evidence tags: dataset observation; literature; University of Greenwich guidance; inference |
| A, O, U, L | Alert demand, overflow, unused capacity and load ratio of the threshold policies (DR-009) |
| F1 | Harmonic mean of precision and recall |
| B | Number of bootstrap replicates |
| C_FP | Extra false-alert (customer-friction) cost of reviewing a legitimate transaction |
| CI | Confidence interval |
| Δ | H3 equivalence tolerance on the realised alert rate (0.004) |
| AI | Artificial intelligence |
| DR | Decision Record |
| bf16 | bfloat16: 16-bit floating-point format used for mixed-precision neural-network training |
| CPU, GPU, CUDA | Central processing unit; graphics processing unit; NVIDIA's GPU computing platform |
| FT-Transformer | Feature Tokenizer + Transformer |
| CRISP-DM | Cross-Industry Standard Process for Data Mining |
| IBM | International Business Machines Corporation, publisher of the methodology framework of DR-014 |
| TDSP | Microsoft Team Data Science Process |
| cuBLAS | CUDA Basic Linear Algebra Subprograms library |
| H1, H2, H3 | Confirmatory hypotheses: model family (tree-based vs neural); class weighting; alert-rate transfer fidelity |
| MLP | Multilayer perceptron |
| LGB-b | LightGBM with balanced class weighting: the H3 configuration selected under DR-010 |
| K | Fixed daily review budget (reviews per relative day) |
| ℓ | Block length of a day-level moving-block bootstrap, in days |
| pp | Percentage points |
| Recall@2% | Share of frauds reviewed at the 2 % review capacity |
| REF-NNN | Reference identifier in `../references.md` |
| ρ | Miss-to-review cost ratio: assumed cost of a missed fraud ÷ cost of one review |
| R0–R3 | Decision policies: default 0.5 threshold; F1-optimal threshold; capacity-rate threshold; daily top-K |
| R4 | Decision policy: calibrated cost-aware capped review |
| Recall@2% under R3, V2 | Criterion for selecting the H3 configuration (DR-010) |
| RQ2, RQ3 | Research questions 2 (policy-transfer fidelity) and 3 (cost-aware capacity use) |
| SESOI | Smallest Effect Size Of Interest |
| SHA-256 | Secure Hash Algorithm, 256-bit (R3 cutoff tie rule, DR-015) |
| V1, V2 | Validation 1 (relative days 106–126); Validation 2 (relative days 127–154) |
