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
| [DR-005](DR-005-sesoi-and-h1-h2-inference.md) | SESOI and H1/H2 inference plan | Accepted (pre-freeze); primary bootstrap unit open | 2026-09-24 | Smallest Effect Size Of Interest (SESOI) = 2 percentage points (pp) of Recall@2%; H1 (model-family hypothesis) difference test with practical interpretation; H2 (class-weighting hypothesis) intersection-union equivalence; Holm multiple-testing family {H1, H2, H3} |
| [DR-006](DR-006-bootstrap-inference-strategy.md) | Bootstrap inference strategy for H1 and H2 | Accepted (pre-freeze) | 2026-09-24 | Primary: two-way pigeonhole bootstrap (relative day × fine pseudo-entity); sensitivity: Owen-corrected variance and day-block bootstrap (block length ℓ = 3 days); percentile confidence intervals (CIs), B = 10,000 bootstrap replicates |
| [DR-007](DR-007-rq3-cost-ratio-and-break-even.md) | RQ3 cost-ratio grid and break-even rule | Accepted (pre-freeze) | 2026-09-24 | Primary cost model (review + missed-fraud cost; the extra false-alert cost C_FP as a sensitivity relabelling); miss-to-review cost-ratio grid ρ ∈ {2, …, 1000}, exact curves over (1, 1000]; saturation and break-even ratios per capacity and configuration; amount outcomes as exposure proxies |

## Source of evidence

DR-001 to DR-003 draw on [`notebooks/01_dataset_feasibility_audit.ipynb`](../../notebooks/01_dataset_feasibility_audit.ipynb). Its conclusion was *"Suitable with methodological changes."* DR-004 is a project-governance decision (GitHub Issue #6). DR-005 (Issue #8) draws on the notebook's power projections and the equivalence-testing literature. DR-006 (Issue #14) draws on development-period (V2) dependence diagnostics and the clustered-bootstrap literature. DR-007 (Issue #13) draws on a decision-theoretic derivation and the cost-sensitive, cost-curve and decision-curve literature; it uses no new dataset computations.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [D], [L], [G], [I] | Evidence tags: dataset observation; literature; University of Greenwich guidance; inference |
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
| RQ2, RQ3 | Research questions 2 (policy-transfer fidelity) and 3 (cost-aware capacity use) |
| SESOI | Smallest Effect Size Of Interest |
| V1, V2 | Validation 1 (relative days 106–126); Validation 2 (relative days 127–154) |
