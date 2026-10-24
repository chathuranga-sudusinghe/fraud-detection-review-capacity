# Research Lifecycle

How the study is organised along the ten stages of the Foundational Methodology for Data Science published by International Business Machines Corporation (IBM) [REF-085](references.md#ref-085), adapted to a confirmatory research design ([DR-014](decisions/DR-014-research-methodology-framing.md)). This document owns the mapping of the stages to the study's work, status and evidence. The research design itself is specified in the [protocol](protocol.md) and the [Decision Records](decisions/README.md).

## Adaptation to a confirmatory design

- Stages 1–8 carry the confirmatory research; stage 9 (Deployment) is realised as a local demonstration application and stage 10 (Feedback) as supervisor review and reporting.
- Iteration between stages is allowed only before the protocol freeze, only on the development partitions (Train, Validation 1 and Validation 2), and only within each partition's fixed role ([protocol: Experimental Split Structure](protocol.md#experimental-split-structure)).
- The evaluation on the sealed final Test period runs once, and nothing is refined from it.

Train is the model-fitting partition (relative days 1–105), Validation 1 (V1) covers relative days 106–126, Validation 2 (V2) relative days 127–154, and Test relative days 155–182 ([DR-001](decisions/DR-001-temporal-split-strategy.md)).

## Stages

Research questions are abbreviated RQ1–RQ3 and hypotheses H1–H3; both are stated in full in the [protocol](protocol.md).

| # | Stage (IBM) | Work in this study | Status | Evidence |
|---|---|---|---|---|
| 1 | Business understanding | Research problem, gap, RQ1–RQ3, H1–H3; structured literature review | Done; accepted | [Protocol](protocol.md), [research foundation](research-foundation.md), [literature review](literature-review.md) |
| 2 | Analytic approach | Predictive, prescriptive, inferential and descriptive components, each mapped to RQ1–RQ3 and H1–H3 | Done; accepted | [Protocol: Analytic Approach](protocol.md#analytic-approach) |
| 3 | Data requirements | Data each question and hypothesis needs; handling of the dataset's limits (no real costs, real capacity, label-report dates or account identifiers) | Done; accepted | [Protocol: Data Requirements](protocol.md#data-requirements) |
| 4 | Data collection | Secondary data: source, use terms, checksums, manifest | Use terms verified; checksums recorded; manifest generated and committed (Issue #62) | [Dataset](dataset.md#dataset-manifest), [`data/datapackage.json`](../data/datapackage.json), notebooks 01–02 |
| 5 | Data understanding | Schema, join, time structure, class imbalance, drift, entity dependence | Done; computation findings corrected in notebook 04 ([Issue #59](https://github.com/chathuranga-sudusinghe/fraud-detection-review-capacity/issues/59)) | [Notebooks 01–04](../notebooks/README.md), [feature catalog](feature-catalog.md) |
| 6 | Data preparation | Chronological split; feature policy (433 inputs); tree-based and neural input representations; validation contract; data pipeline | Fixed; pipeline implemented and verified on the raw data (Issue #62) | [Data-preparation record](data-preparation.md), [DR-001](decisions/DR-001-temporal-split-strategy.md), [DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md), [DR-012](decisions/DR-012-neural-network-input-representation.md) |
| 7 | Modeling | Tree-based (LightGBM, CatBoost) and neural (ResNet-style multilayer perceptron, FT-Transformer) models, each unweighted and balanced, under one common training procedure | Model set, training-device policy and training rules fixed; development pipeline implemented and reviewed; development runs pending | [Protocol](protocol.md), [DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md), [DR-013](decisions/DR-013-training-device-policy.md), [DR-015](decisions/DR-015-common-training-rules.md), [graphics processing unit (GPU) acceptance probe](gpu-acceptance-probe.md) |
| 8 | Evaluation | H1–H3, RQ3 analysis, false-negative slices; one Test evaluation after the freeze | Design defined; development analyses implemented; runs pending; Test after the freeze | [Protocol: Statistical Plan](protocol.md#statistical-plan) |
| 9 | Deployment (demonstration) | Research report; local demonstration application programming interface (API) on synthetic example requests, not dataset rows | Planned | — |
| 10 | Feedback (review) | Supervisor review; limitations; future work | Ongoing | — |

FT-Transformer is the Feature Tokenizer + Transformer model ([DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md)).

## Gates between development and confirmation

1. **Before the development runs:** training, policy and analysis procedures are fixed.
2. **Before the Test period is scored:** the fitted models, calibrators, thresholds, review budgets and the H3 configuration (the model × weighting configuration on which H3 is tested) are frozen, and the protocol is tagged ([protocol: Protocol Freeze](protocol.md#protocol-freeze)).

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| API | Application programming interface |
| GPU | Graphics processing unit |
| H1–H3 | Hypotheses of the protocol |
| IBM | International Business Machines Corporation |
| RQ1–RQ3 | Research questions of the protocol |
| REF-NNN | Reference identifier in [`references.md`](references.md) |
| Train, V1, V2, Test | Model-fitting partition (relative days 1–105); Validation 1 (106–126); Validation 2 (127–154); sealed confirmatory partition (155–182) |
