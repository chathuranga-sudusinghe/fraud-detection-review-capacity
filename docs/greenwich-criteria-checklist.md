# University of Greenwich Criteria Checklist

This checklist maps each University of Greenwich requirement that applies to this project to the place where the repository meets it, and lists the gaps with their actions. It was built in Issue #63.

- **Authority.** University requirements come only from official University of Greenwich sources, read directly ([G]). Literature ([L]) supports the research design but cannot establish a University requirement.
- **Project.** COMP1885 Individual Project, Master of Science (MSc) in Data Science and its Applications, University of Greenwich: a compulsory Level 7 module of 30 credits, delivered as a postgraduate dissertation [G, REF-114].
- **Decisions.** The project owner decides each point on this evidence. Decisions taken on 11 October 2026 are marked **Owner decision**; points that only the COMP1885 assessment brief or the module leader can settle are marked **to confirm**.

Evidence tags: **[G]** University of Greenwich source; **[L]** literature; **[I]** inference or recommendation.

## Sources

Each source, with its access details and verification status, is recorded in the [reference register](references.md).

| Topic | References |
|---|---|
| Assessment, marking, supervision and dissertations | [REF-107](references.md#ref-107) Assessment and Feedback Policy (2025) |
| Pass, compensation, late submission and award rules | [REF-108](references.md#ref-108) Academic Regulations for Taught Awards 2026-27 |
| Use of artificial intelligence (AI) | [REF-109](references.md#ref-109), [REF-110](references.md#ref-110), [REF-111](references.md#ref-111) |
| Project guidance and student handbook | [REF-112](references.md#ref-112), [REF-113](references.md#ref-113) |
| Ethics and research integrity | [REF-050](references.md#ref-050), [REF-051](references.md#ref-051), [REF-052](references.md#ref-052) |
| Research data | [REF-115](references.md#ref-115) |
| Programme and module | [REF-054](references.md#ref-054), [REF-114](references.md#ref-114); module handbook and assessment brief [REF-055](references.md#ref-055) not yet obtained |

## 1. Assessment and award

| Requirement [G] | Source | Where the project meets it | Status |
|---|---|---|---|
| Module pass mark 50 % at Level 7 | REF-108 D5.5 b; REF-107 Appendix 4 | Planning target, not a repository artefact | Noted |
| No compensation for the dissertation or major project: it must be passed outright | REF-108 D5.11 | — | Noted |
| Merit at a 60 % and Distinction at a 70 % programme average, rounded to the nearest whole number | REF-108 D6.8–D6.10 | — | Noted |
| Level 7 dissertations are double-marked (blind when the supervisor is the first marker); COMP1885 is a postgraduate dissertation | REF-107 Appendix 3; REF-114 | Every claim in the dissertation is traceable to a repository document, Decision Record or evidence file, so a second marker can check it without the author | Met by design |
| Dissertation length: a 30-credit Level 7 dissertation is 8,000–10,000 words; COMP1885 has 30 credits | REF-107 Appendix 7; REF-114 | **Owner decision:** plan to 8,000–10,000 words | **To confirm** whether the 2025 or 2019 policy applies to the programme (REF-107 policy page) |
| Submission through Moodle/Turnitin by the deadline | REF-113; REF-108 D5.2 | Submit by the deadline stated in the COMP1885 brief | Planned |
| Word limit: a tolerance of +10 %; beyond it, markers do not consider the content | REF-107 §18 | Dissertation planned to stay within 8,000–10,000 words (+10 %) | Planned |
| Referencing normally in the Harvard style, consistent across the programme | REF-107 §30 | Dissertation references in the Harvard style unless the module states otherwise | Planned |

## 2. Project requirements

| Requirement [G] | Source | Where the project meets it | Status |
|---|---|---|---|
| A product involving code, not a literature review alone | REF-112 | The `review_capacity` package, its tests and the reproducible pipeline (`src/`, `tests/`, [README](../README.md)) | Met (development continuing) |
| A new contribution informed by recent literature, beyond rote application | REF-112 | [Research foundation](research-foundation.md) (contribution); [literature review](literature-review.md) and [evidence matrix](literature-evidence-matrix.md) (research gap) | Met in design |
| For research-oriented projects: an in-depth, current literature review; how the contribution extends the field; a rigorous methodology | REF-112 | [Literature search method](literature-search-method.md); confirmatory [protocol](protocol.md) with Decision Records; sealed Test period | Met in design |
| The student leads the topic, plan and research and authors the dissertation; the supervisor advises only | REF-107 Appendix 7; REF-112 | Owner decisions recorded in issues and Decision Records; AI assistants support under the rules in [AGENTS.md](../AGENTS.md) | Met |
| Records of agreed actions from supervision | REF-107 Appendix 7 (student responsibility 7) | Supervision meetings and email; the design-review request is recorded in Issue #56 | Ongoing |
| Regular supervision: usually 4–6 meetings across the dissertation; fortnightly engagement at the project stage | REF-107 Appendix 7; REF-113 | Agree the supervision schedule with the supervisor | **To confirm** |
| Give the project presentation(s), which are mandatory | REF-112 | Confirm the presentation dates and requirements | Planned |

## 3. Greenwich marking rubric, Level 7

The rubric has six domains [G, REF-107 Appendix 4]. The module's customised rubric is not available (REF-055), so the generic Level 7 descriptors are used as the working standard.

| Domain | Level 7 descriptor (key terms) | Where the project addresses it |
|---|---|---|
| 1. Knowledge and understanding | Systematic, comprehensive knowledge; critical awareness of current problems at the forefront of the discipline | Literature review; research foundation |
| 2. Use of research-informed evidence | Originality in applying knowledge; critical evaluation of current research and methodologies | Evidence matrix; Decision Records (alternatives considered) |
| 3. Evaluation and analysis | Complex issues handled systematically; sound judgement with incomplete data; new hypotheses | Hypotheses H1–H3 and statistical plan ([protocol](protocol.md)); stated threats to validity |
| 4. Communication, organisation and presentation | Self-direction; well-judged experimentation; clear, accurate English | Dissertation (to be written); documentation and notation standard |
| 5. Referencing and coverage | All sources acknowledged, **including generative AI tools**; consistent citation; breadth and depth | [Reference register](references.md) with verification status; AI acknowledgement in the AI Reflection (AIR; section 5) |
| 6. Graduate employability and skills | Advanced practical and technology-based skills; sustained self-management | Engineering practice: locked environment, tests, provenance, issue workflow |

Referencing counts for at most 5 % of the marks where it is assessed (REF-107 §32).

## 4. Ethics

| Requirement [G] | Source | Where the project meets it | Status |
|---|---|---|---|
| The Research Ethics Policy applies to master's students' research | REF-050 §1.2 | [Protocol, Ethics](protocol.md#ethics) | Met |
| Fully anonymised secondary data should not need ethical approval; a taught student is advised by the programme or module leader | REF-051 | No ethics application required: the supervisor's written guidance of 3 October 2026 (no personal data or personally identifiable information, PII; no restrictions on data use, reporting, code or storage), recorded in Issue #31 and the [protocol](protocol.md#ethics); no formal approval or exemption claimed | Met; the supervisor's comment on the dataset is pending |
| Research involving criminal subject matter requires ethical consideration and is Higher Risk; Higher-Risk student projects are reviewed by the Faculty Research Ethics Committee and then the University Research Ethics Board, and research should not commence until the relevant approval has been granted | REF-050 §1.3.4, §2.2.1 iv, §2.4, §2.13 | Documented ethical considerations: false-positive and false-negative harms, re-identification risk, fairness limits ([protocol, Ethics](protocol.md#ethics)) | Met as a documented consideration under the supervisor's guidance |
| Data stored securely: University OneDrive where possible; laptops used for data must be encrypted | REF-051 | Raw and row-level data stay on the student's machine and are never committed ([dataset](dataset.md#data-use-terms)) | **Action:** confirm that the laptop's disk is encrypted |

## 5. Use of AI

| Requirement [G] | Source | Where the project meets it | Status |
|---|---|---|---|
| Each assessment brief states its AI-use tier; Tier 2 ("Supported by AI") is the expected default | REF-109; REF-110 | **Owner decision:** Tier 2 as the working basis | **To confirm** against the COMP1885 brief |
| Where AI contributed, an AI Reflection (AIR): how AI was used, how its outputs were reviewed or revised, and the student's own contribution. The AIR replaces the Declaration of AI use | REF-109; REF-111 | Written at submission; the AI-assisted workflow, review steps and approvals are recorded in [AGENTS.md](../AGENTS.md), [DR-004](decisions/DR-004-github-issue-driven-workflow.md) and the issue history | Planned |
| AI must not fabricate sources, data, quotations or evidence; the student remains responsible for accuracy and originality | REF-109 | Research integrity rules in AGENTS.md; reference verification statuses; owner-run experiments only | Met |
| Generative AI tools acknowledged in the text and the reference list | REF-107 Appendix 4, domain 5 | AIR; citation of the tools used | Planned |

## 6. Data handling and academic integrity

| Requirement [G] | Source | Where the project meets it | Status |
|---|---|---|---|
| Academic integrity; assessment misconduct procedure; Turnitin checks | REF-113; REF-109 | Student responsibility for analysis and writing; source verification statuses recorded in the [reference register](references.md) | Ongoing |
| Commitment to research integrity (Concordat) | REF-052 | Documented protocol awaiting freeze; sealed Test period; Decision Records and reproducible evidence | Ongoing |
| Research data: a data management plan; data retained for at least 10 years; data made open only where legally, commercially and ethically appropriate | REF-115 (applies to staff and postgraduate students actively engaged in research) | Data handling documented in [`dataset.md`](dataset.md); the competition data cannot be redistributed (REF-049), so code, metadata and aggregate results are kept in the repository and the data stay local | Met as far as applicable |
| Competition data used within its terms (non-commercial; no redistribution) | REF-049 (data owner's terms) | [Dataset data-use terms](dataset.md#data-use-terms); [data-access rule](dataset.md#data-access-after-the-parquet-build) | Met |

## Open points and actions

| # | Point | Action | Owner |
|---|---|---|---|
| 1 | COMP1885 AI-use tier | Confirm Tier 2 against the assessment brief | Student, with the module leader |
| 2 | Applicable Assessment and Feedback Policy (2025 or 2019) and the word count | Confirm with the module leader | Student |
| 3 | COMP1885 module handbook, brief and customised rubric (REF-055) | Obtain from Moodle when available; until then the generic Level 7 rubric is the working standard | Student |
| 4 | Supervisor's comment on the dataset | Requested after the supervisor's ethics guidance of 3 October 2026; record it when received | Student, with the supervisor |
| 5 | Encrypted storage of the local data | Confirm disk encryption on the laptop | Student |
| 6 | AI Reflection | Write the AIR at submission from the recorded workflow | Student |
| 7 | Supervision schedule and presentation | Agree the meeting schedule; confirm presentation dates and requirements | Student, with the supervisor |

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [G], [L], [I] | Evidence tags: University of Greenwich source; literature; inference or recommendation |
| AI | Artificial intelligence |
| AIR | AI Reflection: the student's statement of how AI was used in an assessment, replacing the Declaration of AI use (REF-111) |
| COMP1884, COMP1885 | Group Project and Individual Project modules |
| DR-NNN | Decision Record NNN in `decisions/` |
| H1–H3 | Confirmatory hypotheses of the protocol |
| Level 7 | Master's level of the UK qualifications framework |
| MSc | Master of Science |
| PII | Personally identifiable information |
| REF-NNN | Reference identifier in `references.md` |
| Test | Sealed final evaluation partition, defined in the protocol |
| Tier 1–3 | University AI-use tiers: independent of AI; supported by AI; augmented with AI |
