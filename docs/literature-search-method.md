# Literature Search Method

This document records how the literature for this study was searched, screened and selected, so that the evidence base can be reproduced and audited. The selected evidence is summarised in [`literature-evidence-matrix.md`](literature-evidence-matrix.md), discussed in [`literature-review.md`](literature-review.md) and listed with verification status in [`references.md`](references.md). It was produced under GitHub Issue #52.

The search is a **structured literature search** with documented screening, not a full systematic review: it uses one bibliographic database plus targeted searches, and screening was carried out by one reviewer with AI assistance (Section 6).

---

## 1. Purpose

The search supports the study's research questions on fraud detection under a fixed daily review budget (the number of transactions that can be reviewed per day). It looks for evidence on:

1. how class imbalance is handled in fraud detection, and whether resampling or class weighting is justified;
2. how tree-based models and neural models compare on tabular data;
3. how review capacity, alert budgets and human-in-the-loop (HITL) review are modelled and evaluated;
4. how decision thresholds behave when transferred to later data under drift;
5. how calibration and cost-based thresholds are used;
6. which studies use the IEEE-CIS (IEEE Computational Intelligence Society) Fraud Detection dataset, and how.

The themes are labelled T1–T6 in that order.

---

## 2. Sources searched

| Source | Role | Date |
|---|---|---|
| IEEE Xplore (Institute of Electrical and Electronics Engineers digital library), accessed through the University of Greenwich | Structured search for themes T1, T3, T4, T5 and T6; results exported as CSV (comma-separated values) | 8 October 2026 |
| Targeted searches (publisher sites, NeurIPS proceedings, arXiv, Crossref, OpenAlex, Semantic Scholar) | Theme T2 and known key sources not indexed by IEEE Xplore; citation follow-up from the closest studies | 8–9 October 2026 |
| Existing reference register (`references.md`, REF-001–REF-085) | Sources already verified in earlier issues | — |

Theme T2 (tree-based vs neural models on tabular data) was not searched in IEEE Xplore, because its main venues (NeurIPS, the Conference on Neural Information Processing Systems, and machine-learning journals) are largely outside that database.

---

## 3. Search strings and results

All IEEE Xplore searches used the basic search over all metadata, limited to publication years 2015–2026. The full export of each search was saved locally.

| Theme | Query | Records exported |
|---|---|---|
| T1 Class imbalance | `"fraud detection" AND (imbalance* OR "cost-sensitive" OR SMOTE)` | 1,000 (export limit; first 1,000 by relevance) |
| T3 Review capacity / HITL | `fraud AND ("alert budget" OR "review capacity" OR "top-k" OR "human-in-the-loop" OR investigator*)` | 126 |
| T4 Drift / threshold | `fraud AND ("concept drift" OR "dataset shift") AND (threshold* OR alert*)` | 50 |
| T5 Calibration / cost | `fraud AND (calibrat* OR "cost-sensitive" OR "decision threshold")` | 277 |
| T6 IEEE-CIS studies | `"IEEE-CIS" AND fraud` | 87 |
| **Total** | | **1,540** |

SMOTE = Synthetic Minority Over-sampling Technique. The T1 search returned more records than IEEE Xplore exports in one file; only the first 1,000 by relevance were exported, and the total hit count was not recorded.

Raw exports and downloaded full texts are kept locally in `research-materials/literature/` (not committed, because of publisher copyright).

---

## 4. Eligibility criteria

**Include** a source when all of the following hold:

- it is a peer-reviewed journal article or conference paper, or an authoritative methodological source (e.g. a NeurIPS paper or a recognised handbook);
- it concerns transaction, payment or card fraud, or a general tabular-learning or evaluation method directly used by this study;
- it informs at least one theme (T1–T6) and a research question, hypothesis or decision policy of this study;
- for empirical fraud studies: the evaluation design (data split, metrics) is reported clearly enough to be appraised.

**Exclude** a source when any of the following holds:

- **E-domain:** the fraud domain differs materially (e.g. insurance, healthcare, telecom, click, blockchain or accounting fraud), unless the method is directly transferable;
- **E-scope:** the method lies outside the study's scope (e.g. graph neural networks, federated learning, generative oversampling, large language models, reinforcement learning, quantum methods);
- **E-quality:** evaluation by accuracy alone on imbalanced data, resampling before the train–test split, or results too poorly reported to appraise;
- **E-type:** not peer-reviewed (preprints, vendor or blog material), retracted, or a duplicate.

Preprints are cited only when no peer-reviewed source covers the point, and are marked as such; none is used in the current selection.

Included sources are coded **I-core** (directly informs a hypothesis, a decision policy or the boundary of the study's contribution) or **I-support** (background or corroborating evidence).

---

## 5. Screening and selection

### 5.1 Flow of records

| Stage | Records |
|---|---|
| Identified in IEEE Xplore (five themes) | 1,540 |
| Duplicates across themes removed (by DOI, the Digital Object Identifier, or normalised title) | 212 |
| Unique records screened on title | 1,328 |
| Abstracts screened | 64 |
| Full texts assessed | 9 |
| Selected from the IEEE Xplore search | 13 (10 new; 3 already in the register) |
| Selected from targeted searches and citation follow-up | 7 |
| **New sources added to the register** | **17 (REF-086–REF-102)** |

### 5.2 How screening was done

1. **Priority ordering.** Each unique record was given a transparent keyword score (temporal split, capacity / HITL, IEEE-CIS, calibration, cost, class weighting, tabular deep learning, drift, review or benchmark; journal bonus; penalty when no card, payment or transaction term appears). The score only orders the list; it does not include or exclude anything.
2. **Title screening of all records.** All 1,328 titles were screened against Section 4. A random check of 100 low-scored records showed that the score misses some relevant papers (two known relevant papers were scored low), so low-scored records were title-screened in full rather than skipped.
3. **Abstract screening.** Abstracts were read for all 40 high-scored records and for 24 records flagged by title among the others.
4. **Full-text assessment.** Full texts were read for the records closest to the study's design and for sources that support a specific claim. When a full text was not accessible, only the abstract was used, and the evidence matrix says so.

The reason for each exclusion at abstract level is recorded in the local screening sheet with the codes of Section 4.

### 5.3 Targeted searches and citation follow-up

Targeted searches covered theme T2 and specific claims: the primary sources for the ResNet-style and FT-Transformer tabular models; large tabular benchmarks comparing gradient-boosted trees with neural networks; systematic reviews of fraud detection under the original class imbalance; evaluation leakage in fraud research; and the cost of manual alert review. Citation follow-up started from the closest comparators already in the register (REF-078, REF-080) and from the selected IEEE studies.

---

## 6. Reviewer and AI assistance

Searching in IEEE Xplore and downloading paywalled full texts were done by the project owner. Title and abstract screening, full-text reading, evidence extraction and bibliographic verification were carried out with an AI assistant (Claude), under the rules in [`AGENTS.md`](../AGENTS.md). All selection decisions are subject to the project owner's review before they are used in the dissertation, and the use of AI is disclosed in the AI Reflection (AIR) required by the University of Greenwich [REF-053].

Bibliographic details of every new source were checked against Crossref or the official proceedings record; verification status is recorded in [`references.md`](references.md).

---

## 7. Limitations

- **One database.** Only IEEE Xplore was searched in a structured way. Journals indexed elsewhere (e.g. ScienceDirect, SpringerLink, ACM Digital Library) were reached only through targeted searches and citation follow-up, so relevant studies may have been missed. A cross-publisher database search (Scopus or Web of Science) would strengthen the gap statement.
- **T1 export limit.** Only the first 1,000 of the T1 results were exported.
- **Single reviewer.** Screening was not duplicated by an independent second reviewer.
- **Keyword ordering.** The priority score is a heuristic; its known misses were addressed by title-screening every record.
- **Access.** Some full texts (e.g. MDPI and Elsevier articles) could not be read from the verification environment; those sources are used only for what their abstracts state.

---

## Symbols and notation

| Abbreviation | Meaning |
|---|---|
| ACM | Association for Computing Machinery |
| AIR | AI Reflection: the University of Greenwich declaration of AI use |
| CSV | Comma-separated values (export file format) |
| DOI | Digital Object Identifier |
| E-domain, E-scope, E-quality, E-type | Exclusion codes (Section 4) |
| HITL | Human-in-the-loop |
| I-core, I-support | Inclusion codes (Section 4) |
| IEEE | Institute of Electrical and Electronics Engineers |
| IEEE-CIS | IEEE Computational Intelligence Society (organiser of the Fraud Detection dataset) |
| NeurIPS | Conference on Neural Information Processing Systems |
| REF-NNN | Reference identifier in `references.md` |
| SMOTE | Synthetic Minority Over-sampling Technique |
| T1–T6 | Search themes: class imbalance; tree-based vs neural models; review capacity / HITL; drift / threshold; calibration / cost; IEEE-CIS studies |
