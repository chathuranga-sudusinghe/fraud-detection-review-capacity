# AGENTS.md — Working Rules for AI-Assisted Work

Stable working rules for **ChatGPT, Claude and any future AI assistant** working in this repository, and for the student directing them. Read this file before making any change.

Rationale for these rules: [DR-004 — GitHub issue-driven workflow](docs/decisions/DR-004-github-issue-driven-workflow.md).

If this file conflicts with [`docs/protocol.md`](docs/protocol.md) on a methodological point, **the protocol wins**. Raise a GitHub Issue to resolve the conflict.

---

## Project scope

- **Repository:** `fraud-detection-review-capacity`
- **Nature:** standalone MSc Data Science research project, University of Greenwich, COMP1885 Individual Project.
- **Working title:** *Fraud Detection Under Limited Review Capacity: A Temporally Validated Study of Gradient-Boosting Models and Decision Policies*
- **Dataset:** IEEE-CIS Fraud Detection labelled training data.
- **Research basis:** this repository, local project data, University of Greenwich guidance, and verified literature.

---

## Issue-driven workflow

Every meaningful change follows this sequence:

```text
GitHub Issue
  → issue-specific branch
  → implementation / research / documentation
  → review
  → pull request
  → merge
  → automatic issue closure
```

- Put **`Closes #<issue-number>`** or **`Fixes #<issue-number>`** in the pull-request description. GitHub then closes the issue automatically on merge. This only happens when the PR targets the default branch (`main`).
- **Work only on the active issue.** Do not make unrelated changes outside it. If you notice something else that needs fixing, record it in a new issue or in your report rather than fixing it in passing.
- **Exception:** tiny typo-only fixes (spelling, punctuation, broken formatting) that do not change meaning may be made without an issue.
- **A GitHub Issue is always required first** for any change affecting:
  - research questions;
  - hypotheses;
  - methodology;
  - datasets;
  - experiments;
  - models;
  - evaluation;
  - references;
  - code;
  - documentation structure.

---

## Research integrity rules

- **The final Test period (relative days 155–182) remains sealed** until the frozen protocol permits the confirmatory evaluation. Do not load, compute on, or inspect its labels, and do not analyse Test-period feature–target relationships, thresholds or false negatives before then.
- **Test results must not influence** model selection, feature selection, thresholds, decision policies, hypotheses, cost assumptions or statistical decisions.
- **Do not create future-looking aggregates.** Any count, frequency or entity aggregate must use Train-only fitting or past-only windows.
- **Do not use manual target encoding.** Categorical handling is native to each library.
- **`TransactionID` and raw `TransactionDT` are not model features.**
- **Record methodological decisions** in [`docs/decisions/`](docs/decisions/README.md) as Decision Records (`DR-NNN-…`).
- **Update [`docs/references.md`](docs/references.md)** whenever external evidence is introduced. Give each source a `REF-NNN` ID and an honest verification status: *Verified*, *Partially verified* or *Unverified*. Never upgrade a status without checking.
- **Never invent** citations, dataset facts, University requirements or methodological claims. If something cannot be verified, say so and mark it as unverified.
- **Always distinguish the source of a claim:**

  | Tag | Meaning |
  |---|---|
  | **[D]** | Observed dataset evidence |
  | **[L]** | Literature evidence (cite `REF-NNN`) |
  | **[G]** | University of Greenwich guidance |
  | **[I]** | Inference or recommendation |

---

## Dataset

**IEEE-CIS Fraud Detection** (source and caveats: [`docs/dataset.md`](docs/dataset.md)).

Expected raw files:

- `data/raw/train_transaction.csv`
- `data/raw/train_identity.csv`

**Raw data must remain local and must never be committed.** `.gitignore` already excludes `data/raw/*`. Do not weaken that rule. Do not commit derived data files (e.g. Parquet copies or processed splits) either.

Current temporal structure ([DR-001](docs/decisions/DR-001-temporal-split-strategy.md)). Relative day = `TransactionDT // 86,400`; no calendar dates are inferred.

| Partition | Relative days | Role |
|---|---|---|
| Train | 1–105 | Model fitting (no refit) |
| V1 | 106–126 | Hyper-parameter tuning and early stopping |
| V2 | 127–154 | Calibration (R4), threshold selection, daily review budget K |
| Test | 155–182 | **Sealed.** One confirmatory evaluation after protocol freeze |

---

## Current research scope

Defined in [`docs/protocol.md`](docs/protocol.md) (working, pre-freeze).

- **Models:** LightGBM, CatBoost.
- **Class weighting:** unweighted; balanced class weighting, implemented as explicit sample weights w_pos = n_neg / n_pos computed from Train only.
- **Review-capacity scenarios:** 1 %, **2 % (primary)**, 5 %, 10 %. These are simulated research scenarios, not real institutional capacities. Each is turned into a fixed daily budget K = round(c × mean daily volume in V2) ([DR-002](docs/decisions/DR-002-review-capacity-definition.md)).
- **Decision policies:** R0–R4 as defined in the protocol.

**Do not add** new model families, resampling methods (e.g. SMOTE, over- or under-sampling) or feature-engineering studies unless an explicit GitHub Issue changes the protocol.

---

## Documentation

Important files:

- [`README.md`](README.md)
- [`docs/research-foundation.md`](docs/research-foundation.md)
- [`docs/protocol.md`](docs/protocol.md)
- [`docs/dataset.md`](docs/dataset.md)
- [`docs/literature-review.md`](docs/literature-review.md)
- [`docs/references.md`](docs/references.md)
- [`docs/decisions/`](docs/decisions/README.md)

When a methodological decision changes:

1. update the relevant research document(s);
2. create a new Decision Record, or update an existing one's status (e.g. *Superseded by DR-NNN*). Never renumber or delete records;
3. update `docs/references.md` where required.

---

## Coding and notebook rules

- Use the project virtual environment (`.venv`).
- Follow [`pyproject.toml`](pyproject.toml) for dependencies. Add a dependency only when the active issue requires it.
- Put reusable logic in reusable code rather than copying it between notebooks.
- Notebooks are research evidence and analysis tools, not the whole project architecture. Save executed notebooks with their outputs when they serve as evidence.
- Keep experiments deterministic where practical: fixed seeds, recorded package versions, hashed data splits.
- Add tests for reusable implementation code when appropriate.

---

## Git rules

- **Do not commit directly to `main`.**
- Work on issue-specific branches, e.g. `docs/issue-6-agent-workflow`.
- Keep changes scoped to the active issue.
- Use pull requests for review. The PR description must link the issue with `Closes #N` or `Fixes #N`.
- **AI assistants do not commit or push unless explicitly instructed** by the student for that task.

---

## AI-assisted research workflow

This project uses **ChatGPT** and **Claude** as AI assistants. They support:

- research planning;
- methodological review;
- literature discovery and verification;
- coding;
- notebook development;
- documentation drafting;
- debugging;
- reproducibility review;
- decision-record preparation.

**AI output must not be accepted automatically.** All AI-assisted work must be:

1. reviewed by the student;
2. checked against the dataset, the literature or University guidance where relevant;
3. corrected where necessary;
4. disclosed according to University of Greenwich requirements (the AI Reflection, AIR; see `REF-053` in [`docs/references.md`](docs/references.md)).

**AI assistants must not:**

- fabricate references;
- fabricate results;
- inspect sealed Test labels before this is permitted;
- change frozen methodology without a GitHub Issue and a Decision Record;
- commit or push unless explicitly instructed.

**The student remains responsible** for the final research design, implementation, analysis, interpretation and submission.
