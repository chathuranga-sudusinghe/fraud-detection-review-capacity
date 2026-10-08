# DR-004 — GitHub issue-driven workflow

| Field | Value |
|---|---|
| Status | Accepted |
| Date | 2026-09-23 |
| Evidence source | Project governance (GitHub Issue #6); GitHub documentation on linking pull requests to issues |
| Related | [`AGENTS.md`](../../AGENTS.md), [DR-001](DR-001-temporal-split-strategy.md), [DR-002](DR-002-review-capacity-definition.md), [DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md), [protocol](../protocol.md) |

## Context

This is a single-student MSc research project. Its credibility depends on being able to show *when* and *why* each methodological choice was made, and that no choice was influenced by the sealed Test period (relative days 155–182; [DR-001](DR-001-temporal-split-strategy.md)).

Work is carried out by the student with the help of two artificial intelligence (AI) assistants, **ChatGPT** and **Claude**. They are used for:
- research planning;
- methodological review;
- literature discovery and verification;
- coding and notebook development;
- documentation drafting;
- debugging;
- reproducibility review;
- decision-record preparation.

AI assistants can produce large, plausible-looking changes quickly, across many files at once. Without a scoping mechanism, several things become hard:
- reviewing each change;
- attributing it to a stated purpose;
- ensuring it was checked before being accepted.

These risks matter most before the protocol freeze, when research questions, hypotheses and design parameters are still open (see the open items in [`docs/protocol.md`](../protocol.md#open-items-before-protocol-freeze)).

## Decision

**Every meaningful research, documentation or implementation change must start with a GitHub Issue.**

The working rules that put this into practice, for the student and for every AI assistant, are recorded in the root [`AGENTS.md`](../../AGENTS.md).

**ChatGPT and Claude are recorded as the AI-assisted research and implementation tools used in this project.** Their output is never accepted automatically. The student reviews, checks and corrects it, and discloses AI use according to University of Greenwich requirements (AI Reflection, AIR; `REF-053` in [`../references.md`](../references.md)).

## Workflow

```text
GitHub Issue
  → issue-specific branch
  → implementation / research / documentation
  → review
  → pull request
  → merge
  → automatic issue closure
```

1. **Issue.** State the objective, scope and acceptance criteria before work starts.
2. **Branch.** One issue-specific branch per issue, e.g. `docs/issue-6-agent-workflow`. No direct commits to `main`.
3. **Work.** Only the changes the issue requires.
4. **Review.** The student reviews all changes, including AI-generated ones, against the issue's acceptance criteria and, where relevant, against the dataset, literature or University guidance.
5. **Pull request (PR).** The description links the issue with a closing keyword.
6. **Merge** into `main` after review.
7. **Automatic closure.** GitHub closes the linked issue when the PR is merged.

## Scope rules

- **One issue, one scope.** Changes outside the active issue are not made, even when they seem helpful. They are reported, or recorded in a new issue.
- **An issue is always required first** for any change affecting:
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
- **Methodological changes** additionally need an update to the relevant research document(s) and Decision Record(s) in `docs/decisions/`, following the [record maintenance policy](README.md#record-maintenance-policy): revised in place before the protocol freeze, superseded by a new record after it. Where external evidence is introduced, `docs/references.md` must also be updated.
- **AI assistants do not commit or push** unless the student explicitly instructs them to for that task.

## Automatic issue closure via PR keywords

The pull-request description includes **`Closes #<issue-number>`** or **`Fixes #<issue-number>`**. According to GitHub's documentation:
- the linked issue is closed automatically when the PR is merged;
- this happens **only if the PR targets the repository's default branch**. The default branch here is `main`.

GitHub also accepts `close`, `closed`, `fix`, `fixed`, `resolve`, `resolves` and `resolved`. This project uses `Closes` or `Fixes` for consistency.

Source: GitHub Docs, *Linking a pull request to an issue*, https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue (accessed 23 Sep 2026). This is operational tooling documentation, not research evidence, so it is not entered in the research reference register.

## Exceptions

Tiny **typo-only** fixes may be made without an issue: spelling, punctuation or broken Markdown formatting that does not change meaning. Any edit that changes the meaning of a research question, hypothesis, number, reference or decision is **not** a typo fix and needs an issue.

## Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Informal direct commits to `main` | No review step and no link between a change and its purpose. Methodology drift before the freeze would be hard to reconstruct. |
| Branches and PRs without issues | Records *what* changed but not the objective or acceptance criteria stated beforehand. Weaker as evidence that decisions came before results. |
| Decision Records only, without issues | Covers methodology but not code, data handling, references or documentation structure. It also gives no scoping mechanism for AI-assisted work. |

## Rationale

- **Traceability.** Every change links to a stated objective (issue), a reviewed diff (PR) and, for methodology, a Decision Record. Together these form an auditable timeline of what was decided before the Test period is unsealed. That timeline supports the protocol freeze and its Git tag.
- **Scope control for AI assistance.** An issue gives an AI assistant a bounded task and gives the student a checklist to review against. This reduces unrelated or unreviewed edits.
- **Research integrity.** A dated issue and PR history is concrete evidence that model, policy and threshold choices were not informed by Test results. This is consistent with the University's commitment to the Concordat to Support Research Integrity (`REF-052`) and with the requirement to disclose AI use (`REF-053`).

## Consequences

- Small changes carry some overhead: an issue, a branch and a PR. This is accepted as the cost of traceability.
- The GitHub issue and PR history becomes part of the project's research record and can be cited in the dissertation's methodology and reflection sections.
- `AGENTS.md` must be kept consistent with `docs/protocol.md`. If they conflict on a methodological point, the protocol takes precedence until the conflict is resolved through an issue.
- Commits made before this decision (up to and including the methodology-baseline merge) did not all follow this workflow. The rule applies from this decision onwards.

## Limitations

- **Process rules are not technical controls.** Nothing in this decision *enforces* the workflow. Branch protection on `main` (e.g. required PRs) could enforce part of it, but that is a repository setting outside this decision.
- **Automatic closure depends on the target branch.** A PR merged into any branch other than `main` will not close its issue.
- **Tool-specific instruction files.** Some AI coding tools read their own instruction file names by default and may not load `AGENTS.md` automatically. For those tools, `AGENTS.md` must be supplied explicitly or referenced from the tool's own instruction file.
- **Review quality depends on the student.** The workflow creates review points; it does not guarantee the review is thorough.

## Relationship to `AGENTS.md`

- **This record (DR-004)** documents *why* the workflow was adopted, and the alternatives.
- **[`AGENTS.md`](../../AGENTS.md)** is the operational rulebook: project scope, workflow steps, research-integrity rules, dataset and Test-sealing rules, current research scope, documentation, coding and Git rules, and the AI-assisted research workflow.
- Changes to the workflow itself require a new issue. They are recorded by updating `AGENTS.md` and, where the rationale changes, this record in place, following the [record maintenance policy](README.md#record-maintenance-policy) for process and governance records.

## References

- `REF-052` (University of Greenwich, *Research Integrity*) and `REF-053` (University of Greenwich, *AI for students*); see [`../references.md`](../references.md).
- GitHub Docs, *Linking a pull request to an issue* (operational documentation; cited inline above).
