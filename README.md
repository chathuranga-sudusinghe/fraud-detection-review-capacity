# Fraud Detection Under Limited Review Capacity:<br>A Temporally Validated Study of Tree-Based and Neural Models and Decision Policies

**University:** University of Greenwich\
**Programme:** MSc Data Science and Its Applications\
**Module:** COMP1885 Individual Project\
**Research type:** Quantitative, non-participant, secondary-data experimental study

## Research overview

Fraud teams can review only a limited number of transactions per day, so a fraud model is useful only through the rule that turns its scores into a daily review queue. Using the IEEE-CIS Fraud Detection data (IEEE-CIS = IEEE Computational Intelligence Society), this study trains two tree-based models (LightGBM, CatBoost) and two neural models (ResNet-style multilayer perceptron, MLP; Feature Tokenizer + Transformer, FT-Transformer), each with and without class weighting, on the earliest period, selects thresholds and review rules on later validation periods, and evaluates them **once** on a sealed final period. The contribution is an integrated, pre-specified evaluation of this full path ([research contribution](docs/research-foundation.md#research-contribution); full design in the [research protocol](docs/protocol.md)). Its most distinctive component is a test of whether a score threshold chosen on validation data still produces its intended alert rate on later data (research question RQ2, hypothesis H3). The study also compares the tree-based and neural families and class weighting at fixed capacity (RQ1, hypotheses H1 and H2) and asks when a calibrated cost-aware rule is cheaper than always filling the review budget (RQ3).

## Research questions and hypotheses

Stated in full, with their estimands and tests, in the [research protocol](docs/protocol.md). In brief:

| Question | Topic | Hypotheses / analysis |
|---|---|---|
| [Main research question](docs/protocol.md#main-research-question) | How model family, class weighting and score-to-review decision policies affect fraud capture, review workload and cost-sensitive outcomes under a fixed daily review budget on later data | Answered through RQ1–RQ3 |
| [RQ1](docs/protocol.md#rq1--model-family-and-weighting) | Tree-based vs neural model families, with and without class weighting, at fixed review capacity | [H1](docs/protocol.md#h1--model-family-effect) (model-family effect), [H2](docs/protocol.md#h2--class-weighting-effect) (class-weighting effect) |
| [RQ2](docs/protocol.md#rq2--policy-transfer-fidelity) | Whether a validation-selected score threshold keeps its intended alert rate on later data, and how its daily workload compares with the review budget | [H3](docs/protocol.md#h3--alert-rate-transfer-fidelity) (alert-rate transfer fidelity); daily workload against the budget reported descriptively |
| [RQ3](docs/protocol.md#rq3--cost-aware-capacity-use) | When a calibrated cost-aware review policy is cheaper than always filling the review budget, under assumed cost ratios | No confirmatory hypothesis: estimation and sensitivity analysis (break-even cost ratios) |

## Dataset source

**Dataset:** [IEEE-CIS Fraud Detection — Kaggle competition data](https://www.kaggle.com/competitions/ieee-fraud-detection/data) [REF-047](docs/references.md#ref-047), labelled training data only.

**Data provider:** Vesta Corporation; real-world e-commerce transactions.

**Use terms:** non-commercial use under the Competition Rules, Section 7 [REF-049](docs/references.md#ref-049); by project policy, raw data stay local and are never committed.

The data are used exactly as published: no rows are generated, simulated or altered. The [dataset documentation](docs/dataset.md) records provenance, use terms, version control and the dataset's limits; the temporal partitions are defined in [DR-001](docs/decisions/DR-001-temporal-split-strategy.md).

## Research lifecycle

The study follows the ten stages of the Foundational Methodology for Data Science published by IBM (International Business Machines Corporation) [REF-085](docs/references.md#ref-085), adapted to a confirmatory design ([DR-014](docs/decisions/DR-014-research-methodology-framing.md)). The [research lifecycle](docs/research-lifecycle.md) maps each stage to the study's work, status and evidence and states the two gates between development and confirmation. The sealed final Test period is evaluated once, after the protocol freeze.

## Repository layout

```text
fraud-detection-review-capacity/
├── README.md                            # this page: overview, layout, reading order, status
├── AGENTS.md                            # working rules for the project and AI-assisted work
├── docs/                                # research documentation
│   ├── protocol.md                      # questions, hypotheses, methods, statistical plan, freeze conditions
│   ├── research-foundation.md           # research problem, aim, objectives, contribution
│   ├── research-lifecycle.md            # the ten IBM stages: work, status, evidence, gates
│   ├── decisions/                       # Decision Records DR-001 to DR-015: why each design choice was made
│   ├── literature-review.md             # literature review and research gap
│   ├── literature-search-method.md      # how the literature was searched, screened and selected
│   ├── literature-evidence-matrix.md    # field-by-field comparison of the selected studies
│   ├── references.md                    # reference register with verification status
│   ├── dataset.md                       # dataset source, provenance, use terms, manifest
│   ├── data-preparation.md              # data-preparation implementation and evidence record
│   ├── feature-catalog.md               # column-by-column data dictionary
│   ├── gpu-acceptance-probe.md          # GPU acceptance probe of DR-013 and its results
│   ├── environment-setup.md             # create the environment and prepare the data
│   ├── running-the-pipeline.md          # command sequence of the development pipeline
│   └── greenwich-criteria-checklist.md  # University of Greenwich requirements and open points
├── config/                              # machine-readable settings that the code reads
│   ├── features.yaml                    # model-input roles (433 inputs)
│   ├── dataset.yaml                     # raw files, partitions and integrity anchors (DR-001)
│   ├── models.yaml                      # device and numerical settings, LightGBM build (DR-013)
│   └── training.yaml                    # search budget, seeds, spaces, early stopping (DR-015)
├── src/review_capacity/                 # research code and the review-capacity command
│   ├── data/                            # loading, validation and preparation
│   ├── features/                        # model inputs and the shared neural preprocessing
│   ├── models/                          # model interface, configurations, device controls
│   ├── training/                        # training declaration, candidate sampler, search-cost prediction
│   ├── evaluation/                      # review queue, threshold policies, calibration, cost analysis
│   ├── development/                     # search, scoring, analysis, record contracts, summaries
│   └── probe/                           # GPU and batching acceptance probes
├── tests/                               # unit tests on synthetic data; opt-in integration tests
├── notebooks/                           # data-understanding notebooks 01-04
├── evidence/                            # engineering evidence: probes, acceptance records, duplicate audit
├── scripts/                             # engineering scripts (GPU memory feasibility probe)
├── data/                                # dataset manifest and local data
│   ├── datapackage.json                 # dataset manifest (committed)
│   ├── raw/                             # raw Kaggle files (local, never committed)
│   └── processed/                       # processed development files (local, never committed)
├── outputs/                             # run outputs (ignored by Git)
├── .github/workflows/ci.yml             # continuous-integration checks
├── pyproject.toml                       # pinned dependencies
├── uv.lock                              # locked environment
└── .python-version                      # Python version
```

## Reading order

1. **Understand the study:** this page, then the [research foundation](docs/research-foundation.md), the [protocol](docs/protocol.md), the [Decision Records](docs/decisions/README.md) and the [literature review](docs/literature-review.md).
2. **Understand the data:** the [dataset](docs/dataset.md), the [notebooks](notebooks/README.md), the [data-preparation record](docs/data-preparation.md) and the [feature catalog](docs/feature-catalog.md).
3. **Reproduce it:** [environment setup](docs/environment-setup.md), then [running the pipeline](docs/running-the-pipeline.md).
4. **Work on the repository:** [`AGENTS.md`](AGENTS.md), the [`config/`](config/) files, the [reference register](docs/references.md) and the [criteria checklist](docs/greenwich-criteria-checklist.md).

## Quick start

Create the environment and prepare the data once ([environment setup](docs/environment-setup.md)), then run the steps in order ([running the pipeline](docs/running-the-pipeline.md)). Consequential runs follow the [execution rules](AGENTS.md#experiment-execution-responsibility).

```bash
uv sync --locked --extra ml --extra dl --extra dev
```

**Continuous integration:** code-affecting pull requests to `main` receive the checks defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml); CI never trains or scores a model on research data, reads research data or writes research output; unit tests fit only small models on synthetic data.

## Status

**Research design documented and accepted in line with the University of Greenwich criteria ([criteria checklist](docs/greenwich-criteria-checklist.md)); every design item resolved; protocol not yet frozen.**

The ethics and governance status is documented in the [protocol](docs/protocol.md#ethics); no formal ethics approval or exemption is claimed.

Next steps:

1. Development runs on Train, Validation 1 (V1) and Validation 2 (V2) ([running the pipeline](docs/running-the-pipeline.md)), then the analysis run on the completed development run, including the rolling-origin supplement.
2. Review of the development evidence; protocol freeze; single Test evaluation; statistical analysis and reporting.
