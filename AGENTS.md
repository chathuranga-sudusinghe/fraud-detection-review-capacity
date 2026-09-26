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

- **Models:** LightGBM, CatBoost.
- **Class weighting:** unweighted; balanced class weighting, implemented as explicit sample weights computed from Train only. Each fraud row receives the weight

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

**Do not add** new model families, resampling methods (e.g. SMOTE, the Synthetic Minority Over-sampling Technique, or other over- or under-sampling) or feature-engineering studies unless an explicit GitHub Issue changes the protocol.

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

When a methodological decision changes:

1. update the relevant research document(s);
2. create a new Decision Record, or update an existing one's status (e.g. *Superseded by DR-NNN*). Never renumber or delete records;
3. update `docs/references.md` where required.

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
