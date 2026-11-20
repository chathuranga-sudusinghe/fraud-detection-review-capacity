# University of Greenwich Criteria Checklist

This checklist maps each University of Greenwich requirement that applies to this project to the place where the repository meets it, and lists the gaps with their actions. It was built in Issue #63.

- **Authority.** University requirements come only from official University of Greenwich sources, read directly ([G]). Literature ([L]) supports the research design but cannot establish a University requirement.
- **Project.** COMP1885 Individual Project, Master of Science (MSc) in Data Science and its Applications, University of Greenwich: a compulsory Level 7 module of 30 credits, delivered as a postgraduate dissertation [G, REF-114].
- **Decisions.** Each point is decided on this evidence. Decisions taken on 11 October 2026 are marked **Decision**; points that only the COMP1885 assessment brief or the module leader can settle are marked **to confirm**.
- **Module briefs.** The COMP1885 assessment briefs (report and presentation), the report templates and the 2026-27 project guidance were obtained and recorded on 20 November 2026 (Issue #227; REF-137–REF-141). Where they set a module-specific requirement, they govern it; the points they settle are marked **Confirmed**.

Evidence tags: **[G]** University of Greenwich source; **[L]** literature; **[I]** inference or recommendation.

## Sources

Each source, with its access details and verification status, is recorded in the [reference register](references.md).

| Topic | References |
|---|---|
| Assessment, marking, supervision and dissertations | [REF-107](references.md#ref-107) Assessment and Feedback Policy (2025) |
| Pass, compensation, late submission and award rules | [REF-108](references.md#ref-108) Academic Regulations for Taught Awards 2026-27 |
| COMP1885 assessment briefs and report templates, 2026-27 | [REF-137](references.md#ref-137) report, [REF-138](references.md#ref-138) presentation, [REF-139](references.md#ref-139) final-report template, [REF-140](references.md#ref-140) initial-plan template |
| Use of artificial intelligence (AI) | [REF-109](references.md#ref-109), [REF-110](references.md#ref-110), [REF-111](references.md#ref-111), [REF-142](references.md#ref-142), [REF-143](references.md#ref-143) |
| Project guidance and student handbook | [REF-141](references.md#ref-141) (2026-27), [REF-112](references.md#ref-112) (2025-26), [REF-113](references.md#ref-113) |
| Ethics and research integrity | [REF-050](references.md#ref-050), [REF-051](references.md#ref-051), [REF-052](references.md#ref-052) |
| Research data | [REF-115](references.md#ref-115) |
| Programme and module | [REF-054](references.md#ref-054), [REF-114](references.md#ref-114); module handbook [REF-055](references.md#ref-055) not obtained (the assessment briefs are listed above) |

## 1. Assessment and award

| Requirement [G] | Source | Where the project meets it | Status |
|---|---|---|---|
| Module pass mark 50 % at Level 7 | REF-108 D5.5 b; REF-107 Appendix 4 | Planning target, not a repository artefact | Noted |
| No compensation for the dissertation or major project: it must be passed outright | REF-108 D5.11 | — | Noted |
| Merit at a 60 % and Distinction at a 70 % programme average, rounded to the nearest whole number | REF-108 D6.8–D6.10 | — | Noted |
| Level 7 dissertations are double-marked (blind when the supervisor is the first marker); COMP1885 is a postgraduate dissertation | REF-107 Appendix 3; REF-114 | Every claim in the dissertation is traceable to a repository document, Decision Record or evidence file, so a second marker can check it without the author | Met by design |
| COMP1885 weighting: report 80 %, presentation 20 % | REF-137; REF-138; REF-141 | — | Confirmed |
| Report length: 6,500–8,000 words, counted from the Introduction to the Conclusion; everything before and after is excluded | REF-137; REF-139 | Report planned to 6,500–8,000 words. The module brief sets the length, so the general 8,000–10,000-word range for a 30-credit dissertation (REF-107 Appendix 7) is not applied | Confirmed |
| Submission through the COMP1885 Moodle coursework link only (Turnitin checks apply); no paper copy | REF-137; REF-138; REF-113; REF-108 D5.2 | Submit by the deadline stated on Moodle; the briefs state the due date as "TBA" | Planned; **date to confirm** |
| Word limit: a tolerance of +10 %; beyond it, markers do not consider the content | REF-107 §18 | Report kept within 6,500–8,000 words; the tolerance is not relied on | Planned |
| Referencing in a consistent style, Harvard recommended; references hyperlinked; the bibliography typically 2–3 pages | REF-137; REF-139; REF-107 §30 | Report references in the Harvard style, hyperlinked, built from the [reference register](references.md) | Planned |

## 2. Project requirements

| Requirement [G] | Source | Where the project meets it | Status |
|---|---|---|---|
| A product involving code, not a literature review alone | REF-112 | The `review_capacity` package, its tests and the reproducible pipeline (`src/`, `tests/`, [README](../README.md)) | Met (development continuing) |
| A new contribution informed by recent literature, beyond rote application | REF-112 | [Research foundation](research-foundation.md) (contribution); [literature review](literature-review.md) and [evidence matrix](literature-evidence-matrix.md) (research gap) | Met in design |
| For research-oriented projects: an in-depth, current literature review; how the contribution extends the field; a rigorous methodology | REF-112 | [Literature search method](literature-search-method.md); confirmatory [protocol](protocol.md) with Decision Records; sealed Test period | Met in design |
| The student leads the topic, plan and research and authors the dissertation; the supervisor advises only | REF-107 Appendix 7; REF-112 | Decisions recorded in issues and Decision Records; AI assistants support under the rules in [AGENTS.md](../AGENTS.md) | Met |
| Records of agreed actions from supervision | REF-107 Appendix 7 (student responsibility 7) | Supervision meetings and email; the design-review request is recorded in Issue #56 | Ongoing |
| Regular supervision: usually 4–6 meetings across the dissertation; fortnightly engagement at the project stage | REF-107 Appendix 7; REF-113 | Agree the supervision schedule with the supervisor | **To confirm** |
| Give the project presentation, which is mandatory: delivered live to the supervisor and the second marker, in the two weeks before the deadline, face-to-face unless online delivery is agreed | REF-138; REF-141 | Agree the date and time with the supervisor and the second marker | Planned; **date to agree** |
| Evidence of incremental development: the project plan, self-reflection, project management steps and supervisory meetings; a private GitHub repository with commits over time, with the supervisor added, can provide part of it; insufficient evidence may incur a mark penalty | REF-137 | Issue-driven workflow ([DR-004](decisions/DR-004-github-issue-driven-workflow.md)): issues, pull requests and commits over the whole project; the repository is public (Issue #100, decision of 20 November 2026) | Met; add the supervisor to the repository |

## 3. Greenwich marking rubric, Level 7

The COMP1885 briefs set the module's marking criteria, and they govern the marking [G]. Each criterion is marked from 0 % to 100 % and the marks are averaged by weight; step marking (marks ending in 0, 2, 5 or 8) may be used.

| Component | Criterion (weight) | Source |
|---|---|---|
| Report | Understanding of the problem domain (25 %) | REF-137 |
| Report | Development of product and ideas (25 %) | REF-137 |
| Report | Product build and evaluation (25 %) | REF-137 |
| Report | Conclusions and critical review (25 %) | REF-137 |
| Report | Presentation standards and incremental development, applied to all criteria; the marking sheet also records Turnitin issues | REF-137 |
| Presentation | Content and organisation (30 %) | REF-138 |
| Presentation | Appearance and delivery (30 %) | REF-138 |
| Presentation | Question answering (40 %) | REF-138 |

The University's generic rubric has six domains [G, REF-107 Appendix 4]; its Level 7 descriptors remain the University-wide standard below the module criteria.

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
| Fully anonymised secondary data should not need ethical approval; a taught student is advised by the programme or module leader | REF-051 | No ethics application required: the supervisor's written guidance of 3 October 2026 (no personal data or personally identifiable information, PII; no restrictions on data use, reporting, code or storage), recorded in Issue #31 and the [protocol](protocol.md#ethics); no formal approval or exemption claimed | Met; the supervisor's email of 11 October 2026 approved the dataset |
| Research involving criminal subject matter requires ethical consideration and is Higher Risk; Higher-Risk student projects are reviewed by the Faculty Research Ethics Committee and then the University Research Ethics Board, and research should not commence until the relevant approval has been granted | REF-050 §1.3.4, §2.2.1 iv, §2.4, §2.13 | Documented ethical considerations: false-positive and false-negative harms, re-identification risk, fairness limits ([protocol, Ethics](protocol.md#ethics)) | Met as a documented consideration under the supervisor's guidance |
| Data stored securely: University OneDrive where possible; laptops used for data must be encrypted | REF-051 | Raw and row-level data stay on the local machine and are never committed ([dataset](dataset.md#data-use-terms)) | Confirmed on 2 November 2026: the local drives show the BitLocker drive-encryption lock in Windows file explorer (an observation, not a separate technical verification) |

## 5. Use of AI

| Requirement [G] | Source | Where the project meets it | Status |
|---|---|---|---|
| Each assessment brief states its AI-use tier; Tier 2 ("Supported by AI") is the expected default | REF-109; REF-110 | **Decision:** Tier 2 as the working basis. Both COMP1885 briefs allow AI tools for support and state that submitting writing generated by AI as one's own work is an Academic Offence (REF-137; REF-138); this matches Tier 2, although the briefs name no tier number | Confirmed in substance |
| Students must not use AI to write coursework or present AI-generated output as original work | REF-142; REF-137 | The report and slide text is the student's own writing; AI supports planning, evidence preparation, feedback, grammar and refinement | Planned |
| Where AI contributed, an AI Reflection (AIR): how, when and where AI was used, how its outputs were checked, revised or rejected, and the student's own contribution, structured with the CRITIC framework (context, role, interrogate, transform, impact, contribution). The AIR replaces the Declaration of AI use | REF-109; REF-111; REF-142 | Written at submission; the AI-assisted workflow, review steps and approvals are recorded in [AGENTS.md](../AGENTS.md), [DR-004](decisions/DR-004-github-issue-driven-workflow.md) and the issue history | Planned |
| Placement of the AI statement: the University guidance places the AIR at the end of the assignment, before the references; the report template places the Declaration of AI use after the Preface | REF-142; REF-139 | — | **To confirm** with the supervisor |
| AI must not fabricate sources, data, quotations or evidence; the student remains responsible for accuracy and originality | REF-109 | Research integrity rules in AGENTS.md; reference verification statuses; AI assistants run no experiments (execution rules in AGENTS.md) | Met |
| Generative AI tools acknowledged in the text and the reference list; AI-generated chat content referenced in the Harvard structure of the University guide | REF-107 Appendix 4, domain 5; REF-139; REF-143 | AIR; citation of the tools used | Planned |

## 6. Data handling and academic integrity

| Requirement [G] | Source | Where the project meets it | Status |
|---|---|---|---|
| Academic integrity; assessment misconduct procedure; Turnitin checks | REF-113; REF-109 | Analysis and writing remain independent work; source verification statuses recorded in the [reference register](references.md) | Ongoing |
| Commitment to research integrity (Concordat) | REF-052 | Protocol frozen at the tag `protocol-freeze-v1`; sealed Test period; Decision Records and reproducible evidence | Ongoing |
| Research data: a data management plan; data retained for at least 10 years; data made open only where legally, commercially and ethically appropriate | REF-115 (applies to staff and postgraduate students actively engaged in research) | Data handling documented in [`dataset.md`](dataset.md); the competition data cannot be redistributed (REF-049), so code, metadata and aggregate results are kept in the repository and the data stay local | Met as far as applicable |
| Competition data used within its terms (non-commercial; no redistribution) | REF-049 (data owner's terms) | [Dataset data-use terms](dataset.md#data-use-terms); [data-access rule](dataset.md#data-access-after-the-parquet-build) | Met |

## 7. COMP1885 deliverables and format

| Requirement [G] | Source | Where the project meets it | Status |
|---|---|---|---|
| Report as a PDF whose text is not an image; files virus-free, not password-protected and not corrupted; a missing or wrongly formatted file may incur a penalty | REF-137 | Report exported to PDF from the Moodle template | Planned |
| Report written in the Moodle template: 1.5 line spacing, 2.54 cm margins, Aptos, Calibri or Arial at 12 pt; table of contents, list of figures and list of tables; front matter in small Roman page numbers and the body in Arabic numbers; each chapter on a new page; numbered chapters and sections; LaTeX allowed if it follows the same format | REF-137; REF-139 | — | Planned |
| Title page with the project title, name, student identifier, supervisor, submission date and word count | REF-139 | — | Planned |
| Mandatory chapters: Abstract (100–200 words), Preface (how the project meets the MSc requirements and which modules informed it), Introduction, Literature review, Methods, Results, Conclusion and Self-reflection, and one section on legal, social, ethical and professional issues; Implementation and Discussion may stand alone or be merged | REF-139 | — | Planned |
| Appendix A: the initial plan written at the start of the project in the Moodle template; Appendix B: up to 3 further pages | REF-137; REF-139; REF-140 | — | Planned; **locate the initial plan** |
| Code as a zip of all the code used to create the final product, with a `README_1885.txt` on how to access or re-create the data sources, or a representative data sample | REF-137; REF-141 | The `review_capacity` package, tests, configuration and pipeline documentation; the competition data cannot be redistributed (REF-049), so `README_1885.txt` describes access to the data and its re-creation | Planned |
| Slides as a single PDF, with title and references slides; a plain design with a pastel or off-white background is recommended; small changes (at most 20 %) are allowed after feedback at the live presentation | REF-138; REF-141 | — | Planned |
| Recording of the full presentation (12–15 minutes) and the questions (10–15 minutes), as a link or a video upload, with permissions that let anyone in the University view and download it; code snippets or a demonstration of the product included; no grace period | REF-138; REF-141 | — | Planned |

## Open points and actions

| # | Point | Action |
|---|---|---|
| 1 | COMP1885 AI-use tier | Done on 20 November 2026: the briefs allow AI for support only, matching Tier 2 (REF-137; REF-138) |
| 2 | Applicable Assessment and Feedback Policy (2025 or 2019) and the word count | Word count done on 20 November 2026: 6,500–8,000 words from the module brief (REF-137). The applicable policy version remains open where the brief does not settle a point |
| 3 | COMP1885 module handbook, brief and customised rubric (REF-055) | Briefs and marking criteria obtained on 20 November 2026 (REF-137; REF-138). The module handbook (REF-055) was not obtained |
| 4 | Supervisor's comment on the dataset | Done: received by email on 11 October 2026, which approved the dataset |
| 5 | Encrypted storage of the local data | Done on 2 November 2026: BitLocker lock shown on both local drives (observation) |
| 6 | AI Reflection | Write the AIR at submission from the recorded workflow |
| 7 | Supervision schedule and presentation | Presentation requirements confirmed on 20 November 2026 (REF-138). Agree the meeting schedule and the presentation date with the supervisor and the second marker |
| 8 | Official assessment and submission dates, the assessment briefs and the Moodle materials | Briefs and templates obtained on 20 November 2026 (REF-137–REF-141). The briefs state the due dates as "TBA"; confirm them on Moodle or with the programme team (request emailed on 2 November 2026) |
| 9 | Placement of the AI statement (section 5) | Confirm with the supervisor whether the AIR follows the Preface (REF-139) or comes before the references (REF-142) |
| 10 | Initial plan for Appendix A (section 7) | Locate the initial plan presented to the supervisor, or record its status with the supervisor |
| 11 | Supervisor access to the repository (section 2) | Add the supervisor to the GitHub repository |

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [G], [L], [I] | Evidence tags: University of Greenwich source; literature; inference or recommendation |
| AI | Artificial intelligence |
| AIR | AI Reflection: the statement of how AI was used in an assessment, replacing the Declaration of AI use (REF-111) |
| COMP1884, COMP1885 | Group Project and Individual Project modules |
| CRITIC | Framework for writing an AIR: context, role, interrogate, transform, impact, contribution (REF-142) |
| PDF | Portable Document Format |
| `README_1885.txt` | Text file required in the COMP1885 code zip, describing how to access or re-create the data sources |
| TBA | To be announced |
| DR-NNN | Decision Record NNN in `decisions/` |
| H1–H3 | Confirmatory hypotheses of the protocol |
| Level 7 | Master's level of the UK qualifications framework |
| MSc | Master of Science |
| PII | Personally identifiable information |
| REF-NNN | Reference identifier in `references.md` |
| Test | Sealed final evaluation partition, defined in the protocol |
| Tier 1–3 | University AI-use tiers: independent of AI; supported by AI; augmented with AI |
