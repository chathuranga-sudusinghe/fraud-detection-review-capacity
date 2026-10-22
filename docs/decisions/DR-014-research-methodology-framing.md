# DR-014 — Research-methodology framing

| Field | Value |
|---|---|
| Status | Accepted (pre-freeze), Issue #85, in line with the University of Greenwich criteria ([criteria checklist](../greenwich-criteria-checklist.md)) |
| Date | 2026-10-13 (Issue #85; resolves protocol design item 7) |
| Evidence source | The Foundational Methodology for Data Science published by International Business Machines Corporation (IBM) [REF-085], read in full (reference identifiers REF-NNN are listed in [`docs/references.md`](../references.md)); two peer-reviewed reviews of data-science process methodologies [REF-125, REF-126], read in the parts cited; University of Greenwich guidance [REF-054, REF-112, REF-113]. No dataset values, labels or model results were used. |
| Related | [DR-001](DR-001-temporal-split-strategy.md), [DR-004](DR-004-github-issue-driven-workflow.md), [protocol](../protocol.md), [README research path](../../README.md#research-path) |

## Context

A data-science study is easier to follow and to assess when its work is organised along a recognised process model. Such models describe the stages of a project, from understanding the problem to using the results, and the loops between them. The research path in the [README](../../README.md#research-path) already followed the ten stages of the Foundational Methodology for Data Science published by IBM [REF-085], but the framing was marked as proposed and awaiting a decision (protocol design item 7).

Two facts shape the decision:

- **[G]** For research-oriented projects, the University of Greenwich requires a rigorous methodology [REF-112]. Neither the project guidance [REF-112], the student handbook [REF-113] nor the programme page [REF-054] prescribes a particular data-science process model; all three were checked for this on 13 October 2026. The COMP1885 module handbook (Individual Project) [REF-055] has not been obtained.
- **[I]** The study is designed for confirmatory evaluation, and its protocol remains pre-freeze: its procedures are fixed before the development model runs, and the final Test period is sealed and evaluated once ([protocol: Protocol Freeze](../protocol.md#protocol-freeze)). General process models are iterative by design and refine a model after its evaluation and after deployment [REF-085]. Adopting one therefore requires stating where that iteration is allowed and where it is not.

## Decision

### 1. Process framework

The study is organised along the ten stages of the IBM Foundational Methodology for Data Science [REF-085]:

1. business understanding;
2. analytic approach;
3. data requirements;
4. data collection;
5. data understanding;
6. data preparation;
7. modeling;
8. evaluation;
9. deployment;
10. feedback.

The [README research path](../../README.md#research-path) maps each stage to the study's work, status and evidence. The framework organises the work and its documentation; the research design itself is specified in the [protocol](../protocol.md) and the Decision Records.

### 2. Adaptation to a confirmatory design

1. **Iteration.** Returning to an earlier stage is allowed only before the protocol freeze, only on the development partitions (Train (the model-fitting partition), Validation 1 (V1) and Validation 2 (V2)), and only within each partition's fixed role ([DR-001](DR-001-temporal-split-strategy.md); [protocol: Experimental Split Structure](../protocol.md#experimental-split-structure)).
2. **Evaluation (stage 8).** REF-085 describes a testing set that follows the same probability distribution as the training set and is used so that the model "can be refined as needed". This study instead evaluates transfer to later temporal partitions, where the distribution may change, under the partition roles of [DR-001](DR-001-temporal-split-strategy.md). Development evaluation on V1 and V2 may inform decisions before the freeze, within those roles and the protocol's gates. The confirmatory evaluation on the sealed Test period runs once, after the protocol is frozen and tagged, and nothing is refined from its results.
3. **Deployment (stage 9).** No model is deployed into a production environment. The stage is realised as a local demonstration application on synthetic example requests, not on IEEE-CIS (IEEE Computational Intelligence Society) dataset rows.
4. **Feedback (stage 10).** REF-085 uses results from an implemented model to refine it. No deployed model collects outcomes here, so the stage is realised as supervisor review, the reporting of limitations and the definition of future work. Feedback received before the freeze may inform revisions under the iteration rule above and the protocol's gates. Refinements suggested after the confirmatory evaluation are recorded as future work; they do not alter this study's confirmatory results.

### 3. Reproducibility and data-governance additions

A peer-reviewed review of the methodology reports that it lacks team roles and gives "zero concerns on reproducibility, knowledge accumulation and data security" [REF-125]. The study adds these through its existing controls:

- **reproducibility:** checksummed raw files, a dataset manifest and partition hashes ([dataset.md: Dataset Manifest](../dataset.md#dataset-manifest)), a pinned and locked environment, automated checks in continuous integration (CI), and a protocol tagged in Git before the Test evaluation;
- **knowledge accumulation:** Decision Records with their context, evidence and rejected alternatives, and an issue-driven workflow ([DR-004](DR-004-github-issue-driven-workflow.md));
- **data security:** raw data kept local and never committed, the dataset's use terms ([dataset.md: Data-Use Terms](../dataset.md#data-use-terms)) and the sealed Test period;
- **roles:** the final authority and the limited roles of artificial-intelligence (AI) assistants are defined in [`AGENTS.md`](../../AGENTS.md).

## Evidence

- **[L] The framework [REF-085].** Ten stages that "form an iterative process". The analytic approach expresses the problem "in the context of statistical and machine-learning techniques", and "[t]he chosen analytic approach determines the data requirements". Data preparation includes data cleaning (missing or invalid values, duplicates, formatting), combining sources and feature engineering. In evaluation, the testing set "is used to evaluate the model so it can be refined as needed". In feedback, results from the implemented model are used "to refine the model". The paper states that it "bears some similarities to recognized methodologies for data mining", citing the Cross-Industry Standard Process for Data Mining (CRISP-DM) among them.
- **[L] A peer-reviewed review of 19 methodologies (studies from 1996 to 2019) [REF-125].** The IBM methodology "has some similarities with CRISP-DM, but it provides a number of new practices", structures a project "in more phases (10) than CRISP-DM (6)", and is summarised as one that "Provides new practices and extends CRISP-DM process model". Its reported drawbacks are the lack of defined team roles and the reproducibility and data-security gaps quoted above; placing the analytic approach before any data exploration is noted as difficult in practice. The same review describes CRISP-DM as the most commonly used methodology, conceived in 1996 and not revised since, and Microsoft's Team Data Science Process (TDSP) as "very dependent on Microsoft services and policies".
- **[L] A peer-reviewed systematic literature review [REF-126].** Data-science process methodologies evolved from data-mining process models to "more recent methodologies such as … IBM's Foundational Methodology for Data Science … and Microsoft's Team Data Science Process", while CRISP-DM "has been consistently the most commonly used framework and de facto standard".
- **[G] University guidance [REF-054, REF-112, REF-113]** as stated in the [Context](#context).

## Alternatives considered

1. **CRISP-DM.** The most widely used process model [REF-125, REF-126], with six phases: business understanding, data understanding, data preparation, modeling, evaluation and deployment [REF-125]. **[I]** Not chosen because it has no separate phases for the analytic approach, the data requirements and the data collection. This preference concerns separate, visible stages; it does not imply that CRISP-DM omits the underlying activities. In this study these three carry central content: an analytic approach with predictive, prescriptive, inferential and descriptive components mapped to the research questions; data requirements that state what the dataset lacks (real costs, real review capacity, label-report dates, account identifiers) and how the design handles it; and the provenance of secondary data. Because the IBM stages extend CRISP-DM's phases [REF-125], a reader familiar with CRISP-DM can still map the study onto it.
2. **Microsoft TDSP.** Five iterative stages with a strong emphasis on team roles and collaboration, but dependent on Microsoft services and policies [REF-125]. **[I]** Not chosen: its team-management strengths have little use in a single-researcher study, and its tool dependence conflicts with a tool-independent methodology.
3. **No named framework.** The protocol and Decision Records alone. **[I]** Not chosen: a recognised process model makes the methodology easier to assess against the University's requirement for rigour [REF-112], and the research path already followed the IBM stages.

## Rationale

- **[I] Stage fit.** The three stages that the IBM methodology adds before data understanding match three central parts of this study (see [Alternatives considered](#alternatives-considered), item 1).
- **[I] Pre-specification after data understanding.** REF-125's concern is choosing an analytic approach before exploring the data. This study does not claim that: it used permitted development-data inspection (the dataset feasibility audit and the data-understanding notebooks, with the Test period sealed) to assess feasibility and inform the working design. What confirmatory research requires is that the analysis is fixed before its outcomes are seen: procedures are fixed before the development model runs, and the protocol is frozen and tagged before the Test evaluation ([protocol: Protocol Freeze](../protocol.md#protocol-freeze)).
- **[I] Reported gaps addressed.** The controls in [Reproducibility and data-governance additions](#3-reproducibility-and-data-governance-additions) address the limitations reported in REF-125 §4.14. Their inclusion does not establish that every reproducibility or data-security risk is eliminated.
- **[I] Compatibility.** Because the framework extends CRISP-DM [REF-125], the study remains readable to reviewers who use the more common model.

## Consequences

- Protocol design item 7 is resolved, and the README research path states the framing as decided.
- Stage-level documents may be added to give readers one entry point per stage; they link to the protocol and the Decision Records and do not restate them.
- The dissertation's methodology chapter follows the ten stages, citing REF-085 for the framework and REF-125 and REF-126 for its peer-reviewed description and assessment.

## Threats and limitations

- **The primary source is an industry white paper.** REF-085 is not peer-reviewed. Its content, its relation to CRISP-DM and its weaknesses are supported by the peer-reviewed reviews REF-125 and REF-126. The framework structures the work; it is not evidence for any result.
- **CRISP-DM is more widely used.** Some readers may expect it; the direct mapping between the two models reduces this cost.
- **A framework does not ensure rigour by itself.** Rigour comes from the protocol, the Decision Records, the sealed Test period and the protocol freeze.
- **Module handbook not obtained.** If the COMP1885 module handbook [REF-055] prescribes a process model, this record is revised before the protocol freeze.

## References

REF-054, REF-055, REF-085, REF-112, REF-113, REF-125, REF-126. Full entries and verification status are in [`docs/references.md`](../references.md).

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [G], [I], [L] | Evidence tags: University of Greenwich guidance; inference; literature |
| AI | Artificial intelligence |
| CI | Continuous integration: automated code checks defined in `.github/workflows/ci.yml` |
| COMP1885 | The University of Greenwich Individual Project module |
| CRISP-DM | Cross-Industry Standard Process for Data Mining |
| DR | Decision Record |
| IBM | International Business Machines Corporation, publisher of REF-085 |
| IEEE-CIS | IEEE Computational Intelligence Society (the dataset's source) |
| REF-NNN | Reference identifier in `docs/references.md` |
| TDSP | Microsoft Team Data Science Process |
| Test | Sealed confirmatory partition (relative days 155–182) |
| Train | Model-fitting partition (relative days 1–105) |
| V1, V2 | Validation 1 (relative days 106–126) and Validation 2 (relative days 127–154) |
