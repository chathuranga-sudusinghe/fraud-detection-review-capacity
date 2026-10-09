# AGENTS.md — Working Rules for AI-Assisted Work

Stable working rules for **ChatGPT, Claude and any future AI assistant** working in this repository, and for the student directing them. Read this file before making any change.

Rationale for these rules: [DR-004 — GitHub issue-driven workflow](docs/decisions/DR-004-github-issue-driven-workflow.md).

If this file conflicts with [`docs/protocol.md`](docs/protocol.md) on a methodological point, **the protocol wins**. Raise a GitHub Issue to resolve the conflict.

---

## Project scope

- **Repository:** `fraud-detection-review-capacity`
- **Nature:** standalone MSc Data Science research project, University of Greenwich, COMP1885 Individual Project.
- **Working title:** *Fraud Detection Under Limited Review Capacity: A Temporally Validated Study of Tree-Based and Neural Models and Decision Policies*
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

Current temporal structure ([DR-001](docs/decisions/DR-001-temporal-split-strategy.md)). Each transaction is assigned a relative day:

$$
\text{relative day} = \left\lfloor \frac{\texttt{TransactionDT}}{86{,}400} \right\rfloor
$$

where:

- `TransactionDT` = the dataset's time offset in seconds from an undisclosed reference point;
- `86,400` = number of seconds in one day;
- `⌊ · ⌋` = rounding down to a whole number (integer division, `//` in Python).

No calendar dates are inferred.

| Partition | Relative days | Role |
|---|---|---|
| Train | 1–105 | Model fitting (no refit) |
| V1 (Validation 1) | 106–126 | Hyper-parameter tuning and early stopping |
| V2 (Validation 2) | 127–154 | Calibration (for policy R4), threshold selection, daily review budget K, selection of the H3 configuration ([DR-010](docs/decisions/DR-010-h3-confirmatory-configuration.md)) |
| Test | 155–182 | **Sealed.** One confirmatory evaluation after protocol freeze |

---

## Current research scope

Defined in [`docs/protocol.md`](docs/protocol.md) (working, pre-freeze).

- **Models:** two families with two models each ([DR-011](docs/decisions/DR-011-model-set-tree-based-and-neural-families.md)): tree-based (LightGBM, CatBoost) and neural (ResNet-style multilayer perceptron, MLP; FT-Transformer). Each is trained unweighted and with balanced class weighting, giving **eight configurations** under one common training procedure.
- **Class weighting:** unweighted; balanced class weighting, implemented as explicit sample weights computed from Train only and applied in every model's loss. Each fraud row receives the weight

  $$
  w_{\mathrm{pos}} = \frac{n_{\mathrm{neg}}}{n_{\mathrm{pos}}}
  $$

  where:

  - `w_pos` = sample weight given to each fraud (positive-class) row; legitimate rows keep weight 1;
  - `n_neg` = number of legitimate transactions in Train;
  - `n_pos` = number of fraud transactions in Train.

- **Review-capacity scenarios:** 1 %, **2 % (primary)**, 5 %, 10 %. These are simulated research scenarios, not real institutional capacities. Each capacity share c is turned into a fixed daily review budget ([DR-002](docs/decisions/DR-002-review-capacity-definition.md)):

  $$
  K = \operatorname{round}\left( c \times \text{mean daily transaction volume in V2} \right)
  $$

  where:

  - `K` = fixed daily review budget (reviews per relative day);
  - `c` = review-capacity share (1 %, 2 %, 5 % or 10 %);
  - `V2` = Validation 2, relative days 127–154.

- **Decision policies:** R0–R4 as defined in the protocol: R0 (default 0.5 threshold), R1 (F1-optimal threshold), R2 (capacity-rate threshold), R3 (daily top-K review) and R4 (calibrated cost-aware capped policy).

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

These rules apply to **all Markdown work** in this repository: `README.md`, `AGENTS.md`, `docs/`, `docs/decisions/`, and Markdown cells in new notebooks. They were introduced by GitHub Issue #17; the LaTeX display-math format for equations was added by Issue #21. They govern presentation only and never change methodology.

### 1. Define every symbol at first use in each document

- When a mathematical symbol, variable, abbreviation or compact notation first appears in a Markdown file, define it **at that location**.
- Prefer a short human-readable meaning in parentheses where natural, for example:
  - `ρ` (miss-to-review cost ratio);
  - `K` (fixed daily review budget);
  - `p*` (cost-derived probability threshold);
  - `C_R` (cost of reviewing one transaction).
- Expand technical acronyms at first use in each document, e.g. Smallest Effect Size Of Interest (SESOI), Two One-Sided Tests (TOST), Confidence Interval (CI). After first use the acronym may be used alone.
- Do not assume the reader remembers a definition from another file. **Each Markdown document must be understandable on its own.**

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

These rules apply to every AI assistant (ChatGPT, Claude, Codex and any future assistant). They govern who runs what; they do **not** override `docs/protocol.md` or any Decision Record.

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

This project uses AI assistants such as **ChatGPT**, **Claude** and **Codex**; these rules apply equally to any future assistant. They support:

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

- **ChatGPT** may support planning, research synthesis, literature review, methodology review, explanation, drafting, and code and documentation review.
- **Claude** and **Codex** may support implementation, documentation, tests, repository inspection and the preparation of reproducible commands.
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
| AIR | AI Reflection: the University of Greenwich declaration of AI use |
| c | Review-capacity share (1 %, 2 %, 5 % or 10 % of transactions) |
| CI | Confidence Interval |
| DR / DR-NNN | Decision Record, numbered NNN |
| [D], [L], [G], [I] | Evidence tags: dataset, literature, University guidance, inference |
| K | Fixed daily review budget (reviews per relative day) |
| MLP | Multilayer perceptron (ResNet-style MLP: a neural model of DR-011) |
| MAE, RMSE | Mean Absolute Error, Root Mean Squared Error (named as examples of standard academic equations) |
| n_neg, n_pos | Numbers of legitimate and fraud transactions in Train |
| PR | Pull request |
| R0–R4 | Decision policies: default 0.5 threshold; F1-optimal threshold; capacity-rate threshold; daily top-K; calibrated cost-aware capped policy |
| REF-NNN | Reference identifier in `docs/references.md` |
| SESOI | Smallest Effect Size Of Interest |
| SMOTE | Synthetic Minority Over-sampling Technique |
| TOST | Two One-Sided Tests |
| `TransactionDT` | Dataset time offset in seconds from an undisclosed reference point |
| V1, V2 | Validation 1 (relative days 106–126) and Validation 2 (relative days 127–154) |
| w_pos | Sample weight for each fraud row under balanced class weighting, n_neg / n_pos |
| ⌊ · ⌋ | Round down to a whole number (integer division) |
| ρ, p*, C_R, C_FP, C_miss, J_π, π, N_π, FN_π | Used only as layout examples in the notation standard: miss-to-review cost ratio; cost-derived probability threshold; cost of reviewing one transaction; extra false-alert cost of reviewing a legitimate transaction; cost of missing a fraud; scenario cost of policy π; a decision policy; number of reviews; number of missed frauds |
