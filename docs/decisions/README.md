# Research Methodology Decision Records

This folder records methodology decisions: what was decided, on what evidence, which alternatives were rejected, and what the decision costs. Records are short and permanent. When a decision changes, a new record **supersedes** the old one; the old one is kept and its status updated.

## Conventions

- **File name:** `DR-NNN-short-kebab-title.md`, numbered sequentially from `DR-001`. Numbers are never reused.
- **Status values:**
  - *Proposed* — drafted, not yet applied to the protocol.
  - *Accepted (pre-freeze)* — applied to the working protocol; may still change before the freeze.
  - *Frozen* — part of the tagged confirmatory protocol.
  - *Accepted* — in force; used for process and governance decisions that are not part of the confirmatory protocol (e.g. DR-004).
  - *Superseded by DR-NNN*.
- **Evidence tags** (as in the feasibility notebook):
  - **[D]** observed in the dataset;
  - **[L]** literature, cited by `REF-NNN` from [`../references.md`](../references.md);
  - **[G]** University of Greenwich guidance;
  - **[I]** inference or recommendation.
- **Test-set rule:** no record may rely on target-derived information from the sealed Test period (relative days 155–182).

## Index

| ID | Title | Status | Date | Affects |
|---|---|---|---|---|
| [DR-001](DR-001-temporal-split-strategy.md) | Temporal split strategy | Accepted (pre-freeze) | 2026-09-23 | Split structure; roles of V1 (Validation 1) and V2 (Validation 2); no primary gap |
| [DR-002](DR-002-review-capacity-definition.md) | Review-capacity definition | Accepted (pre-freeze) | 2026-09-23 | Capacity grid; fixed daily review budget K; primary metric interpretation |
| [DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md) | RQ2 / H3: policy-transfer fidelity | Accepted (pre-freeze); H3 equivalence tolerance Δ = 0.004 (±20 % relative) added by Issue #7 | 2026-09-23; updated 2026-09-24 | RQ2 (research question 2) wording; H3 (alert-rate transfer hypothesis) equivalence formulation and tolerance; budget-fidelity outcomes |
| [DR-004](DR-004-github-issue-driven-workflow.md) | GitHub issue-driven workflow | Accepted | 2026-09-23 | Project workflow; AI-assisted work rules (see [`AGENTS.md`](../../AGENTS.md)) |
| [DR-005](DR-005-sesoi-and-h1-h2-inference.md) | SESOI and H1/H2 inference plan | Accepted (pre-freeze); primary bootstrap unit left open, later resolved by [DR-006](DR-006-bootstrap-inference-strategy.md) | 2026-09-24 | Smallest Effect Size Of Interest (SESOI) = 2 percentage points (pp) of Recall@2%; H1 (model-family hypothesis) difference test with practical interpretation; H2 (class-weighting hypothesis) intersection-union equivalence; Holm multiple-testing family {H1, H2, H3} |
| [DR-006](DR-006-bootstrap-inference-strategy.md) | Bootstrap inference strategy for H1 and H2 | Accepted (pre-freeze) | 2026-09-24 | Primary: two-way pigeonhole bootstrap (relative day × fine pseudo-entity); sensitivity: Owen-corrected variance and day-block bootstrap (block length ℓ = 3 days); percentile confidence intervals (CIs), B = 10,000 bootstrap replicates |
| [DR-007](DR-007-rq3-cost-ratio-and-break-even.md) | RQ3 cost-ratio grid and break-even rule | Accepted (pre-freeze) | 2026-09-24 | Primary cost model (review + missed-fraud cost; the extra false-alert cost C_FP as a sensitivity relabelling); miss-to-review cost-ratio grid ρ ∈ {2, …, 1000}, exact curves over (1, 1000]; saturation and break-even ratios per capacity and configuration; amount outcomes as exposure proxies |
| [DR-008](DR-008-feature-declaration-and-model-input-policy.md) | Feature declaration and model-input policy | Accepted (pre-freeze) | 2026-09-24 | 49 categorical columns (organiser list), 382 numeric, 2 excluded (`TransactionID`, `TransactionDT`); one canonical input for LightGBM and CatBoost; missing categorical label; high-cardinality policy; `config/features.yaml` |
| [DR-009](DR-009-overflow-handling-for-threshold-policies.md) | Overflow handling for threshold-based policies R0, R1 and R2 | Accepted (pre-freeze) | 2026-09-25 | Report-only overflow rule: R0–R2 evaluated uncapped; alert demand A, overflow O, unused capacity U and load ratio L defined against the daily budget K; truncation to top-K rejected (it reproduces R3); overflow frauds not counted as missed or false negatives; downstream overflow handling out of scope |
| [DR-010](DR-010-h3-confirmatory-configuration.md) | Confirmatory configuration for H3 | Accepted (pre-freeze) | 2026-09-26 | H3 confirmatory for one model × weighting configuration, selected by a frozen rule (highest V2 Recall@2% under R3) before the Test period is unsealed; Holm family stays {H1, H2, H3}; other configurations' H3 results descriptive; defines the rolling-origin supplement's primary configuration |

## Source of evidence

DR-001 to DR-003 draw on [`notebooks/01_dataset_feasibility_audit.ipynb`](../../notebooks/01_dataset_feasibility_audit.ipynb). Its conclusion was *"Suitable with methodological changes."* DR-004 is a project-governance decision (GitHub Issue #6). DR-005 (Issue #8) draws on the notebook's power projections and the equivalence-testing literature. DR-006 (Issue #14) draws on development-period (V2) dependence diagnostics and the clustered-bootstrap literature. DR-007 (Issue #13) draws on a decision-theoretic derivation and the cost-sensitive, cost-curve and decision-curve literature; it uses no new dataset computations. DR-008 (Issue #23) draws on the raw-file schema, development-period values (target never loaded), the organiser's data description and the LightGBM and CatBoost documentation. DR-009 (Issue #27) draws on the protocol's policy definitions, earlier Decision Records and an algebraic argument; it uses no labels or model scores. DR-010 (Issue #29) draws on the protocol's hypothesis and multiplicity definitions and on DR-005's power projections; it uses no labels, model scores or new dataset computations.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [D], [L], [G], [I] | Evidence tags: dataset observation; literature; University of Greenwich guidance; inference |
| A, O, U, L | Alert demand, overflow, unused capacity and load ratio of the threshold policies (DR-009) |
| B | Number of bootstrap replicates |
| C_FP | Extra false-alert (customer-friction) cost of reviewing a legitimate transaction |
| CI | Confidence interval |
| Δ | H3 equivalence tolerance on the realised alert rate (0.004) |
| DR | Decision Record |
| H1, H2, H3 | Confirmatory hypotheses: model family; class weighting; alert-rate transfer fidelity |
| K | Fixed daily review budget (reviews per relative day) |
| ℓ | Block length of a day-level moving-block bootstrap, in days |
| pp | Percentage points |
| Recall@2% | Share of frauds reviewed at the 2 % review capacity |
| REF-NNN | Reference identifier in `../references.md` |
| ρ | Miss-to-review cost ratio: assumed cost of a missed fraud ÷ cost of one review |
| R0–R3 | Decision policies: default 0.5 threshold; F1-optimal threshold; capacity-rate threshold; daily top-K |
| Recall@2% under R3, V2 | Criterion for selecting the H3 configuration (DR-010) |
| RQ2, RQ3 | Research questions 2 (policy-transfer fidelity) and 3 (cost-aware capacity use) |
| SESOI | Smallest Effect Size Of Interest |
| V1, V2 | Validation 1 (relative days 106–126); Validation 2 (relative days 127–154) |
