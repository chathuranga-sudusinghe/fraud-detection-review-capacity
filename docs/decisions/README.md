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
| [DR-001](DR-001-temporal-split-strategy.md) | Temporal split strategy | Accepted (pre-freeze) | 2026-09-23 | Split structure; roles of V1 and V2; no primary gap |
| [DR-002](DR-002-review-capacity-definition.md) | Review-capacity definition | Accepted (pre-freeze) | 2026-09-23 | Capacity grid; fixed daily budget K; primary metric interpretation |
| [DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md) | RQ2 / H3: policy-transfer fidelity | Accepted (pre-freeze); Δ = 0.004 (±20 % relative) added by Issue #7 | 2026-09-23; updated 2026-09-24 | RQ2 wording; H3 equivalence formulation and tolerance; budget-fidelity outcomes |
| [DR-004](DR-004-github-issue-driven-workflow.md) | GitHub issue-driven workflow | Accepted | 2026-09-23 | Project workflow; AI-assisted work rules (see [`AGENTS.md`](../../AGENTS.md)) |

## Source of evidence

DR-001 to DR-003 draw on [`notebooks/01_dataset_feasibility_audit.ipynb`](../../notebooks/01_dataset_feasibility_audit.ipynb). Its conclusion was *"Suitable with methodological changes."* DR-004 is a project-governance decision (GitHub Issue #6).
