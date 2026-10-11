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

## Research path

The study is organised along the ten stages of the Foundational Methodology for Data Science published by IBM [REF-085](docs/references.md#ref-085), adapted to a confirmatory research design. This framing is proposed and awaits the project owner's decision (protocol design item 7); the accepted adaptation will be recorded in a Decision Record.

- Stages 1–8 carry the confirmatory research; stage 9 (Deployment) is realised as a local demonstration application and stage 10 (Feedback) as supervisor review and reporting.
- Iteration between stages is allowed only before the protocol freeze, only on the development partitions (Train, Validation 1 and Validation 2), and only within each partition's fixed role ([protocol: Experimental Split Structure](docs/protocol.md#experimental-split-structure)).
- The evaluation on the sealed final Test period runs once, and nothing is refined from it.

| # | Stage (IBM) | In this study | Status | Evidence |
|---|---|---|---|---|
| 1 | Business understanding | Research problem, gap, RQ1–RQ3, H1–H3; structured literature review | Done; accepted by the project owner | [Protocol](docs/protocol.md), [research foundation](docs/research-foundation.md), [literature review](docs/literature-review.md) |
| 2 | Analytic approach | Predictive, prescriptive, inferential and descriptive components, each mapped to RQ1–RQ3 and H1–H3 | Done; accepted by the project owner | [Protocol: Analytic Approach](docs/protocol.md#analytic-approach) |
| 3 | Data requirements | Data each research question and hypothesis needs, and how the dataset's limits are handled (no real costs, real capacity, label-report dates or account identifiers) | Done; accepted by the project owner | [Protocol: Data Requirements](docs/protocol.md#data-requirements) |
| 4 | Data collection | Secondary data: source, use terms, checksums, manifest | Use terms verified; checksums recorded; dataset manifest generated and committed (Issue #62) | [Dataset](docs/dataset.md#dataset-manifest), [`data/datapackage.json`](data/datapackage.json), notebooks 01–02 |
| 5 | Data understanding | Schema, join, time structure, class imbalance, drift, entity dependence | Done; computation findings corrected in notebook 04 ([Issue #59](https://github.com/chathuranga-sudusinghe/fraud-detection-review-capacity/issues/59)) | [Notebooks 01–04](notebooks/README.md), [feature catalog](docs/feature-catalog.md) |
| 6 | Data preparation | Chronological split; feature policy; validation contract; data pipeline | Split, feature policy (433 inputs) and tree-based input representation fixed; neural-network representation proposed; validation contract and data-preparation pipeline implemented and verified on the raw data (Issue #62) | [DR-001](docs/decisions/DR-001-temporal-split-strategy.md), [DR-008](docs/decisions/DR-008-feature-declaration-and-model-input-policy.md), [DR-012](docs/decisions/DR-012-neural-network-input-representation.md), [`src/review_capacity/`](src/review_capacity/) |
| 7 | Modeling | Tree-based (LightGBM, CatBoost) and neural (ResNet-style MLP, FT-Transformer) × unweighted/balanced; common training procedure; development runs | Model set defined; graphics processing unit (GPU) feasibility verified (Issue #50); training rules pending | [Protocol](docs/protocol.md), [DR-011](docs/decisions/DR-011-model-set-tree-based-and-neural-families.md) |
| 8 | Evaluation | H1–H3, RQ3 analysis, false-negative slices; one Test evaluation after freeze | Design defined; runs after freeze | [Protocol: Statistical Plan](docs/protocol.md#statistical-plan) |
| 9 | Deployment (demonstration) | Research report; local demonstration application programming interface (API) on synthetic example requests (not IEEE-CIS rows) | Planned | — |
| 10 | Feedback (review) | Supervisor review; limitations; future work | Ongoing | — |

Two gates separate development from confirmation: **(1)** training, policy and analysis procedures are fixed before the development runs; **(2)** the fitted models, calibrators, thresholds, review budgets and the H3 configuration (the model × weighting configuration on which H3 is tested) are frozen, and the protocol is tagged, before the Test period is scored ([protocol: Protocol Freeze](docs/protocol.md#protocol-freeze)).

## Reading order

1. This page: overview, research path, environment setup, repository map and status.
2. [Research protocol](docs/protocol.md): research questions, hypotheses, analytic approach, data requirements, methods, statistical plan, design items and freeze conditions.
3. [Research foundation](docs/research-foundation.md): research problem, aim, objectives and contribution.
4. [Decision Records](docs/decisions/README.md): why each design choice was made.
5. [Literature review](docs/literature-review.md): evidence base and research gap.
6. [Dataset](docs/dataset.md) and [notebooks](notebooks/README.md): the data, its use terms and the data-understanding evidence.

## Environment setup

The environment is pinned for reproducibility: every direct dependency has an exact version in [`pyproject.toml`](pyproject.toml), and the complete environment, including indirect dependencies, is locked with SHA-256 hashes in [`uv.lock`](uv.lock). It is managed with [uv](https://docs.astral.sh/uv/), a Python package and project manager.

**Tested platform:** Linux x86-64 (Ubuntu 24.04 on Windows Subsystem for Linux 2, WSL2), Python 3.12.3 (pinned in [`.python-version`](.python-version)), uv 0.13.0, and an NVIDIA GPU with a driver that supports CUDA 12.8 (NVIDIA's GPU computing platform; check with `nvidia-smi`). The lock covers this platform only.

1. Install uv:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create the environment exactly as locked, with the tree-based (`ml`), neural (`dl`, PyTorch for CUDA 12.8) and development (`dev`) groups:

   ```bash
   uv sync --locked --extra ml --extra dl --extra dev
   ```

3. Obtain the raw data from Kaggle by accepting the Competition Rules, and place both files in `data/raw/` ([dataset documentation](docs/dataset.md)). The data are never committed.

4. Validate the raw data and prepare the development files, **once** (run by the project owner; [execution rules](AGENTS.md#experiment-execution-responsibility)):

   ```bash
   .venv/bin/review-capacity --log-file logs/prepare-data.log prepare-data
   ```

   This one-time build ran on 10 October 2026. After it, every step reads only the verified processed files in `data/processed/`, and the raw files are read again only by the confirmatory Test evaluation ([data access rule](docs/dataset.md#data-access-after-the-parquet-build)).

   Every command logs its progress with timestamps to the terminal; `--log-file PATH` (before the command name) also keeps the log in a file, as an evidence-producing run should. `review-capacity --help` lists the options and exit codes.

## Repository map

| Path | Contents |
|---|---|
| [`docs/protocol.md`](docs/protocol.md) | Research protocol: questions, hypotheses, analytic approach, data requirements, methods, statistical plan, design items, freeze conditions |
| [`docs/decisions/`](docs/decisions/README.md) | Decision Records DR-001–DR-012: the reason for each design choice |
| [`docs/research-foundation.md`](docs/research-foundation.md) | Research problem, aim, objectives, contribution |
| [`docs/literature-review.md`](docs/literature-review.md) | Literature review and research gap |
| [`docs/literature-evidence-matrix.md`](docs/literature-evidence-matrix.md) | Field-by-field comparison of the selected studies |
| [`docs/greenwich-criteria-checklist.md`](docs/greenwich-criteria-checklist.md) | University of Greenwich requirements mapped to the repository, with open points |
| [`docs/literature-search-method.md`](docs/literature-search-method.md) | How the literature was searched, screened and selected |
| [`docs/references.md`](docs/references.md) | Reference register with verification status |
| [`docs/dataset.md`](docs/dataset.md) | Dataset source, provenance, data-use terms, version-control policy, dataset manifest |
| [`docs/feature-catalog.md`](docs/feature-catalog.md) | Column-by-column data dictionary |
| [`config/features.yaml`](config/features.yaml) | Machine-readable model-input roles (433 inputs: 431 declared columns and 2 time-of-day inputs) |
| [`config/dataset.yaml`](config/dataset.yaml) | Machine-readable raw files, DR-001 partitions and their integrity anchors |
| [`src/review_capacity/`](src/review_capacity/) | Research code: data loading, validation and preparation (`data/`), model inputs (`features/`); command `review-capacity` |
| [`pyproject.toml`](pyproject.toml), [`uv.lock`](uv.lock), [`.python-version`](.python-version) | Pinned dependencies, locked environment (uv) and Python version |
| [`evidence/`](evidence/gpu_probe/README.md) | Engineering evidence: GPU memory probe results (Issue #50) |
| [`tests/`](tests/) | Unit tests on synthetic data; opt-in integration tests on the local processed development files (no raw file is read), subject to the execution rules in [`AGENTS.md`](AGENTS.md#experiment-execution-responsibility) |
| [`notebooks/`](notebooks/README.md) | Data-understanding notebooks 01–04 |
| [`scripts/`](scripts/) | Engineering scripts (GPU memory feasibility probe, Issue #50) |
| [`AGENTS.md`](AGENTS.md) | Working rules for the project and AI-assisted work |
| [`data/datapackage.json`](data/datapackage.json) | Dataset manifest (Frictionless Data Package): file checksums, sizes, rows and label-free partition summary |
| `data/raw/`, `data/processed/` | Local raw data and processed development files (never committed) |

## Status

**Research design documented and accepted by the project owner in line with the University of Greenwich criteria ([criteria checklist](docs/greenwich-criteria-checklist.md)); remaining design items pending; protocol not yet frozen.**

The ethics and governance status is documented in the [protocol](docs/protocol.md#ethics); no formal ethics approval or exemption is claimed.

Next steps:

1. Resolution of the protocol's remaining design items, in line with the University of Greenwich criteria ([criteria checklist](docs/greenwich-criteria-checklist.md)); supervisor feedback welcome throughout.
2. Training rules: search procedure, early stopping, tie handling, seeds, neural-network representation, calibration details.
3. Development runs on Train, Validation 1 (V1) and Validation 2 (V2); protocol freeze; single Test evaluation; statistical analysis and reporting.
