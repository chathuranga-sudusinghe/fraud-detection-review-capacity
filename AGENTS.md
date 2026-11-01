# AGENTS.md — Working Rules for AI-Assisted Work

Stable working rules for **Claude, Codex, ChatGPT, and any future artificial intelligence (AI) assistant** working in this repository, and for the student directing them. Read this file before making any change.

Rationale for these rules: [DR-004 — GitHub issue-driven workflow](docs/decisions/DR-004-github-issue-driven-workflow.md).

If this file conflicts with [`docs/protocol.md`](docs/protocol.md) on a methodological point, **the protocol wins**. Raise a GitHub Issue to resolve the conflict.

---

## Project scope

- **Repository:** `fraud-detection-review-capacity`
- **Nature:** standalone Master of Science (MSc) Data Science research project, University of Greenwich, COMP1885 Individual Project.
- **Title:** *Fraud Detection Under Limited Review Capacity: A Temporally Validated Study of Tree-Based and Neural Models and Decision Policies*
- **Dataset:** IEEE-CIS (IEEE Computational Intelligence Society) Fraud Detection labelled training data.
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

### Decision capture

Every decision reached with an AI assistant is recorded in the repository or its issues, so that a new chat session or another assistant finds it without re-discussing it. A chat session or an assistant's own memory is never the record.

- **Record final decisions the same day.** A decision the project owner finalizes is written into its owner file (see [Single source of truth](#single-source-of-truth)) through the active issue. If no issue covers it yet, it is recorded as a comment on the relevant GitHub issue, in the form `Decision (YYYY-MM-DD): …`.
- **List open items where they will be decided.** An item discussed but not yet decided is listed in the issue that will decide it, so that nothing discussed is lost.
- **Read before asking.** Before asking the project owner about a design item, check the owner files (`docs/protocol.md`, the Decision Records, `config/*.yaml`) and the open issues. A decided item is applied with its source cited, not asked again.
- **Close each working session with its state.** At the end of a working session, the active issue receives a short "current state / next step" comment.

Claude posts these issue comments itself ([Git rules](#git-rules)); Codex and ChatGPT include the proposed record in their output for the project owner to post.

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

The research design is defined in [`docs/protocol.md`](docs/protocol.md) (frozen at the tag `protocol-freeze-v1`) and the [Decision Records](docs/decisions/README.md); the definitions are not repeated here. In brief:

- **Models:** tree-based (LightGBM, CatBoost) and neural (ResNet-style multilayer perceptron, MLP; FT-Transformer, Feature Tokenizer + Transformer), each trained unweighted and with balanced class weighting: eight configurations under one common training procedure ([DR-011](docs/decisions/DR-011-model-set-tree-based-and-neural-families.md)).
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
- [`docs/research-lifecycle.md`](docs/research-lifecycle.md) — the ten IBM stages mapped to the study: work, status, evidence and gates
- [`docs/environment-setup.md`](docs/environment-setup.md) and [`docs/running-the-pipeline.md`](docs/running-the-pipeline.md) — environment creation and the development-pipeline command sequence
- [`docs/data-preparation.md`](docs/data-preparation.md) — data-preparation implementation and evidence record (methodology stage 6)
- [`docs/gpu-acceptance-probe.md`](docs/gpu-acceptance-probe.md) — graphics processing unit (GPU) acceptance probe of DR-013 §8: synthetic inputs, checks, criteria and tolerances, and its results
- [`docs/feature-catalog.md`](docs/feature-catalog.md) — column-by-column data dictionary; model-input roles are machine-readable in [`config/features.yaml`](config/features.yaml)
- [`docs/literature-review.md`](docs/literature-review.md)
- [`docs/references.md`](docs/references.md)
- [`docs/greenwich-criteria-checklist.md`](docs/greenwich-criteria-checklist.md)
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
| University of Greenwich requirements and how the project meets them | [`docs/greenwich-criteria-checklist.md`](docs/greenwich-criteria-checklist.md) |
| Sources and their verification status | [`docs/references.md`](docs/references.md) |
| Dataset source, provenance, use terms, version control and manifest | [`docs/dataset.md`](docs/dataset.md) |
| Data-preparation implementation record: what the implemented preparation does and does not do, development-only preparation diagnostics and their evidence, with links to the governing rules | [`docs/data-preparation.md`](docs/data-preparation.md) |
| GPU acceptance probe of DR-013 §8: synthetic inputs, checks, acceptance criteria, tolerances and results (DR-013 owns what must be confirmed) | [`docs/gpu-acceptance-probe.md`](docs/gpu-acceptance-probe.md) |
| Column roles (canonical declaration) | [`config/features.yaml`](config/features.yaml); [`docs/feature-catalog.md`](docs/feature-catalog.md) is its human-readable companion and reads every role from it. If the two disagree, `features.yaml` is correct and the catalog is fixed |
| Raw-file checksums, partition boundaries and integrity anchors, for the code | [`config/dataset.yaml`](config/dataset.yaml): the machine-readable form of the DR-001 split that the code reads. It must match [DR-001](docs/decisions/DR-001-temporal-split-strategy.md) and the protocol; a difference is a defect, resolved through an issue |
| Model device and numerical settings and the recorded LightGBM build, for the code | [`config/models.yaml`](config/models.yaml): the machine-readable form of [DR-013](docs/decisions/DR-013-training-device-policy.md) that the code reads. It must match DR-013 and the protocol; a difference is a defect, resolved through an issue |
| Search budget, sampler and training seeds, search spaces, reference configurations, fixed settings and early-stopping values of the training rules, for the code | [`config/training.yaml`](config/training.yaml): the machine-readable form of [DR-015](docs/decisions/DR-015-common-training-rules.md) that the code reads (the main seed and the device settings stay in `config/models.yaml`). It must match DR-015; a unit test compares the two, and a difference is a defect, resolved through an issue |
| Resampling (replicates, seed, streams, block lengths), interval levels, SESOI and H3 tolerances, capacity shares and budgets, configuration order, cell contrasts, calibration-reporting settings, RQ3 grid, false-negative rule parameters and the frozen data-derived false-negative slice boundaries (computed by `declare-fn-boundaries`, `confirmatory/boundaries.py`) of the confirmatory evaluation, for the code | [`config/evaluation.yaml`](config/evaluation.yaml): the machine-readable form of the protocol and [DR-003](docs/decisions/DR-003-rq2-h3-policy-transfer-fidelity.md), [DR-005](docs/decisions/DR-005-sesoi-and-h1-h2-inference.md), [DR-006](docs/decisions/DR-006-bootstrap-inference-strategy.md) and [DR-007](docs/decisions/DR-007-rq3-cost-ratio-and-break-even.md) that the code reads. It must match them and the development constants it repeats; a unit test compares them, and a difference is a defect, resolved through an issue |
| Contracts of the development pipeline's outputs: JavaScript Object Notation (JSON) record kinds (machine-readable) and the generated Markdown summary (human-readable), including its comparison tables | [`src/review_capacity/development/`](src/review_capacity/development/) (`records.py`, `summary.py`): versioned contracts that validate every record and specify its Markdown presentation; their reference documentation is generated from them (`reference_markdown`), never written by hand. Where and how a run writes them (dated directory, atomic records, closing `run_end` record), how the summary is generated and which files may be promoted to `evidence/`: `run_output.py`, `rendering.py`, `promotion.py`; how an analysis run reads a closed development run and what its summary adds: `analysis.py`, `analysis_summary.py` |
| Contracts of the confirmatory Test run's JSON records, the binding of the frozen artifacts and the identity of the frozen state; the Test command's phases | [`src/review_capacity/confirmatory/`](src/review_capacity/confirmatory/) (`records.py`, `frozen.py`, `identity.py`): versioned contracts that validate every record of the single authorised Test evaluation and recompute its redundant values; the run directory machinery is that of the development pipeline (`run_output.py`, with a record set); their reference documentation is generated from them (`reference_markdown`), never written by hand. The scoring phase of the command (authorisation, binding, Test features without labels, the single scoring pass, the lock and the consumption marker): `command.py`, `inputs.py`, `scoring.py`, `state.py`. The evaluation phase up to the outcome records (verification of the saved scores, the consumption marker, the Test labels, the frozen thresholds and calibrators, the slice inputs and the outcome records): `evaluation.py`, `frozen_policies.py`, `slices.py`, `outcome_records.py`. The inference record, the evaluation attempts and the closing of the run: `inference_record.py`, `attempts.py`, `closing.py` (with `command.py`). The Markdown summary of a closed confirmatory run, its cost figures and the promotion of its aggregate evidence: `summary.py`, `figures.py`, `outputs.py` (generated from the records, never written by hand) |
| Orientation, status, repository layout and reading order | [`README.md`](README.md) |
| Research lifecycle: the mapping of the ten IBM stages to the study's work, status and evidence, and the gates between development and confirmation | [`docs/research-lifecycle.md`](docs/research-lifecycle.md) |
| Environment creation (platform, uv, CUDA and NCCL prerequisites, LightGBM build, data preparation) | [`docs/environment-setup.md`](docs/environment-setup.md) |
| The development pipeline's command sequence and what each step reads and writes | [`docs/running-the-pipeline.md`](docs/running-the-pipeline.md) |
| Working rules | This file |

A Decision Record states its decision in full, because the record is the dated evidence of that decision; the protocol carries the same design as the current specification. A difference between them is a defect, not a choice:

- precedence never approves a change: no edit to the protocol or to a record takes effect until it has been reviewed and approved by the project owner through an issue and pull request;
- before the protocol freeze, a difference is resolved through an issue, and in any case before a run that depends on the affected rule;
- after the freeze, the tagged protocol and the frozen records are authoritative, and any change follows the supersession process below.


When a methodological decision changes, open a GitHub Issue and follow the record-maintenance policy in [`docs/decisions/README.md`](docs/decisions/README.md#record-maintenance-policy):

1. **Before the protocol freeze** (the stage until the tag `protocol-freeze-v1`), revise the relevant research documents and Decision Records **in place**, so that they describe the current design only. Do not add narrative about earlier designs; the revision history is kept in Git commits, pull requests and issues.
2. **After the protocol freeze** (the protocol is tagged in Git as `protocol-freeze-v1`; the current stage), do not edit frozen records. Record any change in a **new Decision Record that supersedes** the earlier one, and set the earlier record's status to *Superseded by DR-NNN*.
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
- **Exception:** only the repository-root [`README.md`](README.md), the project's landing page, has no final "Symbols and notation" table; it links to the owner documents, which define the notation in full.

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

- Use the project environment (`.venv`), created by the project owner with `uv sync --locked` as described in [`docs/environment-setup.md`](docs/environment-setup.md).
- Follow [`pyproject.toml`](pyproject.toml) for dependencies. Add a dependency only when the active issue requires it, pinned to an exact version; the project owner then updates [`uv.lock`](uv.lock).
- Put reusable logic in reusable code rather than copying it between notebooks.
- Notebooks are research evidence and analysis tools, not the whole project architecture. Save executed notebooks with their outputs when they serve as evidence.
- Keep experiments deterministic where practical: fixed seeds, recorded package versions, hashed data splits.
- Add tests for reusable implementation code when appropriate.
- **Logging, not printing.** Code reports progress, timing and diagnostics through logging ([`src/review_capacity/logging_setup.py`](src/review_capacity/logging_setup.py)): library modules create loggers and never configure logging. Each long step (data preparation, training, evaluation) is wrapped in `log_step`, which logs its start, duration, failure or interruption; each long loop (columns, folds, trials, epochs) reports bounded periodic progress through `ProgressLogger`, so that a running process always shows what it is doing. `print` is not used in the package or its tests (enforced by the ruff rule T20). Evidence-producing runs keep a log file (`--log-file`), which also records the traceback of any failure.
- **Errors are raised, never hidden.** Expected operational and validation failures are raised as project exceptions ([`src/review_capacity/errors.py`](src/review_capacity/errors.py)) with a clear message and their cause, and the command line summarises each in one line with its own exit code, while the log file keeps its traceback and cause. An `except` clause only adds context and re-raises, or handles an expected, documented case and logs it; an error is never silently swallowed (enforced by the ruff rule BLE). Any other exception is a defect and keeps its traceback.
- **Continuous integration.** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) defines the automated dependency setup and code checks authorised for CI; it never trains or scores a model on research data, reads research data or writes research output (unit tests may fit small models on synthetic data), and a pull request that triggers CI is merged only after its checks pass.

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

These rules apply to every AI assistant (Claude, Codex, ChatGPT and any future assistant). They govern who runs commands and who edits files; they do **not** override `docs/protocol.md` or any Decision Record.

**Terminal commands are run by the project owner.**

- No AI assistant runs a terminal command for the project or the research, except the code-quality checks, read-only inspection, GitHub-issue handling by Claude and the review exception below. The assistant gives the project owner the exact command, one command per block, with what it does and its expected result. The project owner runs it and shares the output. Claude uses that output to create, edit or update files; Codex and ChatGPT use it to produce review findings, suggested commands and proposed replacement text in their responses, without changing files.
- **First use.** A command used for the first time in the project is always run first by the project owner.
- **Prior owner execution.** An earlier run by the project owner may be established by the owner's statement or recorded output from any branch or session of this project. It must cover the same command with the same options and environment settings. A different branch or session alone does not require another first run. Before repeating a command, the reviewing assistant checks that its current implementation still meets every condition of the review exception.
- **Review exception.** When the project owner asks for a review, the reviewing assistant may run verification commands that the project owner has already run at least once (for example linting, type checking, unit and integration tests, validation commands in check-only mode and link checks), to detect errors and to compare its results with the owner's output. Such a command must not:
  - train, tune or score a model on research data (unit tests that fit small models on synthetic data are verification, not training);
  - write research output (processed data, evidence files, the dataset manifest) or change the environment (installing, locking or synchronising packages);
  - read sealed Test labels.
- **Code-quality checks.** Every assistant that has a terminal may run the following four checks on its own initiative, at any time, without a review request and without a prior run by the project owner, so that errors are found and fixed before a change is handed over:

  ```bash
  uv run --no-sync ruff format .
  uv run --no-sync ruff check .
  uv run --no-sync mypy
  uv run --no-sync pytest tests/unit
  ```

  `ruff format` rewrites files, so Claude, the only assistant that edits files, runs it in that form; Codex and ChatGPT run `ruff format --check .`, which reports without rewriting. The permission covers these four checks only (unit tests that fit small models on synthetic data are verification, not training). It does not cover the integration tests, which remain under the review exception, nor any command that trains, tunes or scores a model on research data, writes research output, changes the environment or reads sealed Test labels. An assistant without a terminal, such as one that reads only the pushed repository, gives the command and uses the output the project owner shares.
- **Read-only inspection.** Every assistant may read and search repository content and inspect Git status, diffs, history and GitHub issues, including through read-only commands such as `git status`, `git diff`, `git show` and `gh issue view`. This permission applies independently of the review exception and its first-use requirement; it does not authorise verification runs, changes to any file or to the environment, or access to sealed Test labels.
- **GitHub issues** are the one exception to the terminal-command rule: Claude may create, update and comment on issues itself, with `gh` or GitHub tools ([Git rules](#git-rules)).

In particular, only the project owner runs:

- model-training, hyperparameter-search, V1 and V2 runs;
- the confirmatory Test-period evaluation;
- data-preparation and data-analysis runs that produce research outputs or research evidence; integration tests and check-only validation are governed by the review exception above;
- commands that install packages, create or modify lock files, or synchronise or otherwise change the environment; checks that only verify an existing lock file are governed by the review exception above;
- any run that may produce evidence used for model selection, hypothesis testing or final research conclusions.

Consequential runs follow only after the methodology, the implementation and the experiment protocol have been reviewed and approved. Approval to implement code, modify documentation, fix bugs or review methodology does **not** imply approval to run anything.

---

## Git rules

- **Do not commit directly to `main`.**
- Work on issue-specific branches, e.g. `docs/issue-6-agent-workflow`.
- Keep changes scoped to the active issue.
- Use pull requests for review. The PR description must link the issue with `Closes #N` or `Fixes #N`.
- **AI assistants do not commit, push, open or merge pull requests.** The project owner does, using the commands the assistant provides.
- **GitHub issues.** Claude may create, update and comment on GitHub issues itself, with `gh` or GitHub tools. It asks the project owner before creating a new issue; updating an existing issue needs no separate approval. Codex and ChatGPT propose issue content in their review output.

---

## AI-assisted research workflow

This project uses AI assistants, **Claude**, **Codex** and **ChatGPT**, within the permissions defined in [Human review and approval authority](#human-review-and-approval-authority); these rules apply equally to any future assistant. Claude supports implementation and file updates; Codex and ChatGPT review the following activities and report recommendations without changing files:

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
4. disclosed according to University of Greenwich requirements: an AI Reflection (AIR), which from 2026/27 replaces the Declaration of AI use (see `REF-109` and `REF-111` in [`docs/references.md`](docs/references.md) and the [criteria checklist](docs/greenwich-criteria-checklist.md#5-use-of-ai)).

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

**Assistant roles.** The following permissions and restrictions apply; final approval remains with the project owner:

- **Claude** works with the project owner on the local repository and on GitHub. It is the **only assistant that creates, edits and updates files** (code, tests, configuration, documentation), and it runs the code-quality checks itself and prepares the other terminal commands for the project owner to run.
- **Codex** reads the local repository and GitHub and **reviews only**: it reports findings and does not edit files.
- **ChatGPT** reads the GitHub repository only (committed and pushed content) and **reviews only**: it may support planning, research synthesis, literature review, methodology review, explanation, and code and documentation review, and does not edit files.
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
- run terminal commands except for the code-quality checks, read-only inspection, GitHub-issue handling by Claude and the review exception (see [Experiment execution responsibility](#experiment-execution-responsibility));
- edit files, unless it is Claude working with the project owner (see the assistant roles above);
- commit, push, merge or otherwise finalise repository changes (see [Git rules](#git-rules));
- treat AI-generated text, analysis or results as authoritative research evidence without human review.

**When approval is unclear, stop and ask the project owner** rather than assuming approval.

---

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| AI | Artificial intelligence |
| AIR | AI Reflection: the University of Greenwich statement of how AI was used, replacing the Declaration of AI use from 2026/27 |
| CI | Confidence Interval |
| DR / DR-NNN | Decision Record, numbered NNN |
| FT-Transformer | Feature Tokenizer + Transformer (a neural model of DR-011) |
| GPU | Graphics processing unit |
| IEEE-CIS | IEEE Computational Intelligence Society |
| JSON | JavaScript Object Notation |
| [D], [L], [G], [I] | Evidence tags: dataset, literature, University guidance, inference |
| K | Fixed daily review budget (reviews per relative day) |
| MSc | Master of Science |
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
