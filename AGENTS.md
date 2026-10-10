# AGENTS.md — Working Rules for AI-Assisted Work

Stable working rules for **Claude, Codex, ChatGPT, and any future AI assistant** working in this repository, and for the student directing them. Read this file before making any change.

Rationale for these rules: [DR-004 — GitHub issue-driven workflow](docs/decisions/DR-004-github-issue-driven-workflow.md).

If this file conflicts with [`docs/protocol.md`](docs/protocol.md) on a methodological point, **the protocol wins**. Raise a GitHub Issue to resolve the conflict.

---

## Project scope

- **Repository:** `fraud-detection-review-capacity`
- **Nature:** standalone MSc Data Science research project, University of Greenwich, COMP1885 Individual Project.
- **Title:** *Fraud Detection Under Limited Review Capacity: A Temporally Validated Study of Tree-Based and Neural Models and Decision Policies*
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

- Put **`Closes #<issue-number>`** or **`Fixes #<issue-number>`** in the pull-request (PR) description. GitHub then closes the issue automatically on merge. This only happens when the PR targets the default branch (`main`).
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
- **Record methodological decisions** in [`docs/decisions/`](docs/decisions/README.md) as Decision Records (DR; files named `DR-NNN-…`).
- **Update [`docs/references.md`](docs/references.md)** whenever external evidence is introduced. Give each source a `REF-NNN` ID (reference identifier) and an honest verification status: *Verified*, *Partially verified* or *Unverified*. Never upgrade a status without checking.
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

The temporal partitions are defined in [DR-001](docs/decisions/DR-001-temporal-split-strategy.md). Each transaction's relative day is `TransactionDT` (the dataset's time offset in seconds from an undisclosed reference point) divided by 86,400 seconds and rounded down; no calendar dates are inferred. The partitions are Train (relative days 1–105), Validation 1 (V1, 106–126), Validation 2 (V2, 127–154) and Test (155–182, **sealed**). Use each partition only in the role DR-001 assigns to it.

---

## Current research scope

The research design is defined in [`docs/protocol.md`](docs/protocol.md) (working, pre-freeze) and the [Decision Records](docs/decisions/README.md); the definitions are not repeated here. In brief:

- **Models:** tree-based (LightGBM, CatBoost) and neural (ResNet-style multilayer perceptron, MLP; FT-Transformer), each trained unweighted and with balanced class weighting: eight configurations under one common training procedure ([DR-011](docs/decisions/DR-011-model-set-tree-based-and-neural-families.md)).
- **Review capacity:** simulated scenarios of 1 %, 2 % (primary), 5 % and 10 % of transactions, each turned into a fixed daily review budget K ([DR-002](docs/decisions/DR-002-review-capacity-definition.md)).
- **Decision policies:** R0–R4, defined in the protocol.

**Do not add** model families beyond DR-011, resampling methods (e.g. SMOTE, the Synthetic Minority Over-sampling Technique, or other over- or under-sampling) or feature-engineering studies unless an explicit GitHub Issue changes the protocol.

---

## Documentation

Important files:

- [`README.md`](README.md)
- [`docs/research-foundation.md`](docs/research-foundation.md)
- [`docs/protocol.md`](docs/protocol.md)
- [`docs/dataset.md`](docs/dataset.md)
- [`docs/feature-catalog.md`](docs/feature-catalog.md) — column-by-column data dictionary; model-input roles are machine-readable in [`config/features.yaml`](config/features.yaml)
- [`docs/literature-review.md`](docs/literature-review.md)
- [`docs/references.md`](docs/references.md)
- [`docs/decisions/`](docs/decisions/README.md)
- [`notebooks/`](notebooks/README.md) — index of the data-understanding notebooks and the notebook 01 section map

### Single source of truth

Each piece of content is written in full in **one owner file**. Elsewhere it appears only as:

- a short summary with a link to the owner file (at most a sentence or a one-line list item), which never serves as an independent specification; or
- the local definition of a symbol that the document uses, as required by [notation rule 1](#1-define-every-symbol-at-first-use-in-each-document).

Saved research evidence (executed notebooks) is exempt: it is never edited to match a later document.

| Content | Owner file |
|---|---|
| Title, research questions, hypotheses, analytic approach, data requirements, partition boundaries and roles, models and training procedure, feature policy, class weighting, review capacity, decision policies, calibration, cost model, outcomes, statistical plan, false-negative analysis, leakage controls, ethics status, design items, freeze conditions | [`docs/protocol.md`](docs/protocol.md) |
| Why each design choice was made: context, evidence, alternatives, consequences | The relevant Decision Record in [`docs/decisions/`](docs/decisions/README.md) |
| Research problem, aim, objectives and contribution | [`docs/research-foundation.md`](docs/research-foundation.md) |
| Literature synthesis and research gap | [`docs/literature-review.md`](docs/literature-review.md) |
| Literature search, screening and selection | [`docs/literature-search-method.md`](docs/literature-search-method.md) |
| Field-by-field comparison of selected studies | [`docs/literature-evidence-matrix.md`](docs/literature-evidence-matrix.md) |
| Sources and their verification status | [`docs/references.md`](docs/references.md) |
| Dataset source, provenance, use terms, version control and manifest | [`docs/dataset.md`](docs/dataset.md) |
| Column roles (canonical declaration) | [`config/features.yaml`](config/features.yaml); [`docs/feature-catalog.md`](docs/feature-catalog.md) is its human-readable companion and reads every role from it. If the two disagree, `features.yaml` is correct and the catalog is fixed |
| Raw-file checksums, partition boundaries and integrity anchors, for the code | [`config/dataset.yaml`](config/dataset.yaml): the machine-readable form of the DR-001 split that the code reads. It must match [DR-001](docs/decisions/DR-001-temporal-split-strategy.md) and the protocol; a difference is a defect, resolved through an issue |
| Orientation, status, research path and repository map | [`README.md`](README.md) |
| Working rules | This file |

A Decision Record states its decision in full, because the record is the dated evidence of that decision; the protocol carries the same design as the current specification. A difference between them is a defect, not a choice:

- precedence never approves a change: no edit to the protocol or to a record takes effect until it has been reviewed and approved by the project owner through an issue and pull request;
- before the protocol freeze, a difference is resolved through an issue, and in any case before a run that depends on the affected rule;
- after the freeze, the tagged protocol and the frozen records are authoritative, and any change follows the supersession process below.


When a methodological decision changes, open a GitHub Issue and follow the record-maintenance policy in [`docs/decisions/README.md`](docs/decisions/README.md#record-maintenance-policy):

1. **Before the protocol freeze** (the current stage), revise the relevant research documents and Decision Records **in place**, so that they describe the current design only. Do not add narrative about earlier designs; the revision history is kept in Git commits, pull requests and issues.
2. **After the protocol freeze** (the protocol is tagged in Git), do not edit frozen records. Record any change in a **new Decision Record that supersedes** the earlier one, and set the earlier record's status to *Superseded by DR-NNN*.
3. At every stage:
   - Decision Record numbers are permanent: never renumber, reuse or delete a record;
   - keep each record's "Alternatives considered" content and its truthful status and approval information, including anything still pending review;
   - leave saved evidence, such as executed notebooks, unchanged;
   - update `docs/references.md` where external evidence is introduced.

---

## Documentation and notation standard

These rules apply to **all Markdown work** in this repository: `README.md`, `AGENTS.md`, `docs/`, `docs/decisions/`, and Markdown cells in new notebooks. They were introduced by GitHub Issue #17; the LaTeX display-math format for equations was added by Issue #21, and the single-source scope of rule 1 by Issue #57. They govern presentation only and never change methodology.

### 1. Define every symbol at first use in each document

- When a mathematical symbol, variable, abbreviation or compact notation first appears in a Markdown file, define it **at that location**.
- Prefer a short human-readable meaning in parentheses where natural, for example:
  - `ρ` (miss-to-review cost ratio);
  - `K` (fixed daily review budget);
  - `p*` (cost-derived probability threshold);
  - `C_R` (cost of reviewing one transaction).
- Expand technical acronyms at first use in each document, e.g. Smallest Effect Size Of Interest (SESOI), Two One-Sided Tests (TOST), Confidence Interval (CI). After first use the acronym may be used alone.
- Do not assume the reader remembers a symbol's definition from another file. **Every document defines every symbol it uses.**
- This self-containment applies to **symbols and notation**. Design content (questions, hypotheses, rules, values) is written in full only in its owner file and is linked from elsewhere ([Single source of truth](#single-source-of-truth)).

### 2. Show important equations as LaTeX display math

- **Use LaTeX display math for important equations**, so they render like standard academic mathematics (for example the Mean Absolute Error, MAE, or Root Mean Squared Error, RMSE). Put each standalone equation between `$$` delimiters on their own lines:

  ```md
  $$
  p^* = \frac{C_R + C_{\mathrm{FP}}}{C_{\mathrm{miss}} + C_{\mathrm{FP}}}
  $$
  ```

  which renders as:

  $$
  p^* = \frac{C_R + C_{\mathrm{FP}}}{C_{\mathrm{miss}} + C_{\mathrm{FP}}}
  $$

- **For multi-step derivations, use an aligned block** with **one meaningful algebraic step per line**:

  ```md
  $$
  \begin{aligned}
  p^* &= \frac{C_R + C_{\mathrm{FP}}}{(C_R + C_{\mathrm{FP}}) + C_{\mathrm{miss}} - C_R} \\
      &= \frac{C_R + C_{\mathrm{FP}}}{C_{\mathrm{miss}} + C_{\mathrm{FP}}} \\
      &= \frac{1}{\rho'}
  \end{aligned}
  $$
  ```

- **Do not write long equations inline** in normal prose. Short references to an already-displayed quantity (e.g. "p* = 1/ρ, as defined above") may stay inline.
- **Do not use fenced plain-text (`text`) blocks for important equations** when LaTeX display math can represent them clearly. Plain-text blocks remain appropriate for workflows, diagrams and file listings.
- **Preserve mathematical meaning exactly** when converting an equation between formats: same symbols, same operations, same values.
- Rendering conventions that avoid known GitHub rendering problems:
  - leave a blank line before and after each `$$ … $$` block, and indent it with its list item when it belongs to one;
  - write multi-letter subscripts and labels in upright text: `C_{\mathrm{miss}}`, `\rho_{\mathrm{BE}}`, `\text{relative day}`;
  - write set braces as `\lbrace … \rbrace` and percent signs as `\%`;
  - keep symbols in `where:` blocks and symbol tables in the same plain-text form used in prose (e.g. `C_miss`), so they can be searched.

### 3. Add a local `where:` block after important equations

Immediately below an important equation, define the symbols it uses:

$$
J_\pi(\rho) = N_\pi + \rho \cdot \mathrm{FN}_\pi
$$

where:

- `J_π(ρ)` = scenario cost of policy π;
- `N_π` = number of reviewed transactions;
- `ρ` = miss-to-review cost ratio;
- `FN_π` = number of missed fraud transactions.

Do not repeat the same definitions after every algebraic step in the same local section.

### 4. End each document with a complete symbol table

- Every Markdown file that uses mathematical notation ends with a final content section headed **`## Symbols and notation`**.
- Every symbol, variable and abbreviation used in that document appears in that table.
- Do **not** place the full table near the beginning of a document. It is a reference index, not the introduction.

### 5. Keep notation consistent

- A symbol keeps the same meaning throughout a document and, where practical, across the repository.
- If a symbol is used inconsistently and fixing it could affect methodology, **report the conflict** (in the active issue or a new one) instead of silently changing it.

### 6. Human readability takes priority

A reader should never need to scroll to another section or open another document to understand an equation. Preferred pattern:

```text
first use with short meaning
  → displayed equation
  → local where: explanation
  → prose interpretation
  → final complete symbol table
```

---

## Coding and notebook rules

- Use the project virtual environment (`.venv`).
- Follow [`pyproject.toml`](pyproject.toml) for dependencies. Add a dependency only when the active issue requires it.
- Put reusable logic in reusable code rather than copying it between notebooks.
- Notebooks are research evidence and analysis tools, not the whole project architecture. Save executed notebooks with their outputs when they serve as evidence.
- Keep experiments deterministic where practical: fixed seeds, recorded package versions, hashed data splits.
- Add tests for reusable implementation code when appropriate.

---

## Research experiment ordering

Research work on this project follows the order below. A later step does not start until the earlier steps it depends on are complete and reviewed. The methodological content of each step is defined in [`docs/protocol.md`](docs/protocol.md) and the [Decision Records](docs/decisions/README.md); this section fixes only the order.

1. Inspect and understand the dataset only within the access allowed by the current protocol (see [Research integrity rules](#research-integrity-rules)).
2. Define or confirm the model-input and feature policy.
3. Review leakage, temporal ordering and prediction-time availability.
4. Confirm the roles of Train, V1, V2 and Test (see [Dataset](#dataset)).
5. Define and freeze the relevant:
   - model rules;
   - class-weighting rules;
   - calibration rules;
   - decision policies;
   - review-capacity rules;
   - statistical and inference rules.
6. Implement and test reusable pipelines.
7. Review the implementation against `docs/protocol.md` and the applicable Decision Records.
8. Obtain explicit human approval for the intended experimental run.
9. The project owner manually runs the consequential Train / V1 / V2 experiments (see [Experiment execution responsibility](#experiment-execution-responsibility)).
10. Review the resulting evidence before accepting or changing any research decision.
11. Freeze model-selection and policy-selection decisions before confirmatory Test access.
12. Perform the final Test evaluation only when the protocol explicitly permits it and all required pre-freeze conditions are satisfied.

In addition:

- **The final Test period remains sealed** until the frozen protocol permits the confirmatory evaluation. The sealed-Test rules in [Research integrity rules](#research-integrity-rules) apply at every step.
- **Provisional or development outputs are not final confirmatory evidence** unless the methodology explicitly defines them that way.
- **Coding or running something does not freeze it.** An implementation, model configuration, threshold, capacity rule or analytical choice is frozen only through the protocol and Decision Record process, not because it has been coded or executed.

---

## Experiment execution responsibility

These rules apply to every AI assistant (Claude, Codex, ChatGPT and any future assistant). They govern who runs what; they do **not** override `docs/protocol.md` or any Decision Record.

**AI assistants may support:**

- creating and editing source code;
- creating and updating documentation;
- creating and updating tests;
- implementing experiment runners and reproducible pipelines;
- repository inspection;
- static validation;
- lightweight engineering tests (defined below);
- preparing exact, reproducible terminal commands.

**AI assistants must not independently execute or authorise:**

- model-training runs;
- hyperparameter-search runs;
- full V1 or V2 experimental runs;
- confirmatory Test-period evaluation;
- long-running data-analysis jobs;
- computationally expensive research experiments;
- any run that may produce evidence used for model selection, hypothesis testing or final research conclusions.

These consequential experiment commands are normally **run manually by the project owner** (the student), after the methodology, the implementation and the experiment protocol have been reviewed and approved.

An AI assistant may prepare the exact terminal command and explain what it will do, but **must stop before running it** unless the project owner explicitly authorises that **specific run**.

Approval to implement code, modify documentation, fix bugs or review methodology does **not** imply approval to execute research experiments.

**Lightweight engineering validation** is allowed when it does not constitute a research experiment, for example:

- linting;
- type checking;
- unit tests;
- import checks;
- small deterministic tests on synthetic or non-research inputs.

Such checks must not train research models, inspect sealed Test labels or generate confirmatory research evidence.

---

## Git rules

- **Do not commit directly to `main`.**
- Work on issue-specific branches, e.g. `docs/issue-6-agent-workflow`.
- Keep changes scoped to the active issue.
- Use pull requests for review. The PR description must link the issue with `Closes #N` or `Fixes #N`.
- **AI assistants do not commit or push unless explicitly instructed** by the student for that task.

---

## AI-assisted research workflow

This project uses AI assistants such as **Claude**, **Codex** and **ChatGPT**; these rules apply equally to any future assistant. They support:

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

**The student remains responsible** for the final research design, implementation, analysis, interpretation and submission. What AI assistants must not do, and who holds approval authority, is set out in [Human review and approval authority](#human-review-and-approval-authority).

---

## Human review and approval authority

AI assistants are **supporting tools only**. The project owner (the student) holds final authority; human review remains the final control point.

**AI assistants do not have final authority over:**

- research questions;
- hypotheses;
- methodology;
- analytical design;
- feature and model-input decisions;
- model choices;
- class-weighting decisions;
- calibration decisions;
- review-capacity definitions;
- decision-policy definitions;
- evaluation protocols;
- metrics;
- statistical procedures;
- SESOI values;
- equivalence margins;
- bootstrap procedures;
- interpretation of experimental results;
- Decision Record changes;
- acceptance of source-code changes;
- experiment execution;
- final Test unsealing and evaluation;
- commits, pushes, pull requests, merges and releases.

**Typical roles.** These describe usual support, not authority:

- **Claude** and **Codex** work on the local repository and on GitHub, and may support implementation, documentation, tests, repository inspection and the preparation of reproducible commands.
- **ChatGPT** reads the GitHub repository only (committed and pushed content), and may support planning, research synthesis, literature review, methodology review, explanation, drafting, and code and documentation review.
- Any future assistant follows the same rules.

**No AI assistant may treat its own output, proposal, implementation or analysis as approved because it created it.** A proposal becomes a project decision only after explicit human review and approval.

**AI assistants must not:**

- fabricate references;
- fabricate results;
- create new research decisions without approval;
- reinterpret unresolved decisions as settled;
- silently expand the project scope;
- change frozen methodology without a GitHub Issue and a Decision Record;
- inspect sealed Test labels before this is permitted;
- execute consequential experiments without explicit, run-specific approval (see [Experiment execution responsibility](#experiment-execution-responsibility));
- commit, push, merge or otherwise finalise repository changes without explicit authorisation (see [Git rules](#git-rules));
- treat AI-generated text, analysis or results as authoritative research evidence without human review.

**When approval is unclear, stop and ask the project owner** rather than assuming approval.

---

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| AI | Artificial intelligence |
| AIR | AI Reflection: the University of Greenwich declaration of AI use |
| CI | Confidence Interval |
| DR / DR-NNN | Decision Record, numbered NNN |
| [D], [L], [G], [I] | Evidence tags: dataset, literature, University guidance, inference |
| K | Fixed daily review budget (reviews per relative day) |
| MLP | Multilayer perceptron (ResNet-style MLP: a neural model of DR-011) |
| MAE, RMSE | Mean Absolute Error, Root Mean Squared Error (named as examples of standard academic equations) |
| PR | Pull request |
| R0–R4 | Decision policies: default 0.5 threshold; F1-optimal threshold; capacity-rate threshold; daily top-K; calibrated cost-aware capped policy |
| REF-NNN | Reference identifier in `docs/references.md` |
| SESOI | Smallest Effect Size Of Interest |
| SMOTE | Synthetic Minority Over-sampling Technique |
| TOST | Two One-Sided Tests |
| `TransactionDT`, `TransactionID` | Dataset time offset in seconds from an undisclosed reference point; transaction identifier |
| Train, Test | Model-fitting partition (relative days 1–105); sealed confirmatory partition (relative days 155–182) |
| V1, V2 | Validation 1 (relative days 106–126) and Validation 2 (relative days 127–154) |
| ρ, ρ′, ρ_BE, p*, C_R, C_FP, C_miss, J_π, π, N_π, FN_π | Used only as layout examples in the notation standard: miss-to-review cost ratio; effective cost ratio including the false-alert cost; break-even cost ratio; cost-derived probability threshold; cost of reviewing one transaction; extra false-alert cost of reviewing a legitimate transaction; cost of missing a fraud; scenario cost of policy π; a decision policy; number of reviews; number of missed frauds |
