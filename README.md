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

The study is organised along the ten stages of the Foundational Methodology for Data Science published by International Business Machines Corporation (IBM) [REF-085](docs/references.md#ref-085), adapted to a confirmatory research design ([DR-014](docs/decisions/DR-014-research-methodology-framing.md)).

- Stages 1–8 carry the confirmatory research; stage 9 (Deployment) is realised as a local demonstration application and stage 10 (Feedback) as supervisor review and reporting.
- Iteration between stages is allowed only before the protocol freeze, only on the development partitions (Train, Validation 1 and Validation 2), and only within each partition's fixed role ([protocol: Experimental Split Structure](docs/protocol.md#experimental-split-structure)).
- The evaluation on the sealed final Test period runs once, and nothing is refined from it.

| # | Stage (IBM) | In this study | Status | Evidence |
|---|---|---|---|---|
| 1 | Business understanding | Research problem, gap, RQ1–RQ3, H1–H3; structured literature review | Done; accepted | [Protocol](docs/protocol.md), [research foundation](docs/research-foundation.md), [literature review](docs/literature-review.md) |
| 2 | Analytic approach | Predictive, prescriptive, inferential and descriptive components, each mapped to RQ1–RQ3 and H1–H3 | Done; accepted | [Protocol: Analytic Approach](docs/protocol.md#analytic-approach) |
| 3 | Data requirements | Data each research question and hypothesis needs, and how the dataset's limits are handled (no real costs, real capacity, label-report dates or account identifiers) | Done; accepted | [Protocol: Data Requirements](docs/protocol.md#data-requirements) |
| 4 | Data collection | Secondary data: source, use terms, checksums, manifest | Use terms verified; checksums recorded; dataset manifest generated and committed (Issue #62) | [Dataset](docs/dataset.md#dataset-manifest), [`data/datapackage.json`](data/datapackage.json), notebooks 01–02 |
| 5 | Data understanding | Schema, join, time structure, class imbalance, drift, entity dependence | Done; computation findings corrected in notebook 04 ([Issue #59](https://github.com/chathuranga-sudusinghe/fraud-detection-review-capacity/issues/59)) | [Notebooks 01–04](notebooks/README.md), [feature catalog](docs/feature-catalog.md) |
| 6 | Data preparation | Chronological split; feature policy; validation contract; data pipeline | Split, feature policy (433 inputs) and tree-based and neural-network input representations fixed; validation contract and data-preparation pipeline implemented and verified on the raw data (Issue #62) | [Data-preparation record](docs/data-preparation.md), [DR-001](docs/decisions/DR-001-temporal-split-strategy.md), [DR-008](docs/decisions/DR-008-feature-declaration-and-model-input-policy.md), [DR-012](docs/decisions/DR-012-neural-network-input-representation.md), [`src/review_capacity/`](src/review_capacity/) |
| 7 | Modeling | Tree-based (LightGBM, CatBoost) and neural (ResNet-style MLP, FT-Transformer) × unweighted/balanced; common training procedure; development runs | Model set defined; neural graphics processing unit (GPU) memory probe completed (Issue #50); training-device policy fixed in Issue #66, with the LightGBM central processing unit (CPU) revision pending review ([DR-013](docs/decisions/DR-013-training-device-policy.md), Issue #101), and its synthetic acceptance probe pending; training rules pending | [Protocol](docs/protocol.md), [DR-011](docs/decisions/DR-011-model-set-tree-based-and-neural-families.md) |
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
7. [Data-preparation record](docs/data-preparation.md): how the data are integrated, validated, cleaned and represented, with the evidence.

## Environment setup

The environment is pinned for reproducibility: every direct dependency has an exact version in [`pyproject.toml`](pyproject.toml), and the Python dependencies, including indirect ones, are locked with artifact SHA-256 hashes in [`uv.lock`](uv.lock). Isolated build dependencies are version-constrained in `pyproject.toml`; the system prerequisites and the native build toolchain are recorded separately (step 2). It is managed with [uv](https://docs.astral.sh/uv/), a Python package and project manager.

**Tested platform:** Linux x86-64 (Ubuntu 24.04 on Windows Subsystem for Linux 2, WSL2), Python 3.12.3 (pinned in [`.python-version`](.python-version)), uv 0.13.0, and an NVIDIA GPU with a driver that supports CUDA 12.8 (NVIDIA's GPU computing platform; check with `nvidia-smi`). The lock covers this platform only.

1. Install uv:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install the system prerequisites of LightGBM's CUDA build ([DR-013](docs/decisions/DR-013-training-device-policy.md) §2). LightGBM is built from source for the project GPU (compute capability sm_120), because the standard wheel has no CUDA support. The recorded build is kept, and LightGBM trains with its CPU learner (DR-013 §1–§2):

   - the CUDA Toolkit 12.8 from NVIDIA's WSL-Ubuntu repository, with the compiler at `/usr/local/cuda-12.8/bin/nvcc` (tested: 12.8.93), and GCC, the GNU Compiler Collection (tested: 13.3.0);
   - NCCL (the NVIDIA Collective Communications Library) 2.26.2 for CUDA 12.8. The WSL-Ubuntu repository has no NCCL packages, so the two packages are taken from NVIDIA's Ubuntu 24.04 repository, installed at this exact version and held, so that an upgrade cannot pull in another CUDA version:

     ```bash
     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/libnccl2_2.26.2-1+cuda12.8_amd64.deb
     ```

     ```bash
     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/libnccl-dev_2.26.2-1+cuda12.8_amd64.deb
     ```

     ```bash
     sudo apt-get install -y ./libnccl2_2.26.2-1+cuda12.8_amd64.deb ./libnccl-dev_2.26.2-1+cuda12.8_amd64.deb
     ```

     ```bash
     sudo apt-mark hold libnccl2 libnccl-dev
     ```

     SHA-256 of the tested packages: `libnccl2` `91a95bb422b54fa6603f481ab72e6b2aa2bb805b9210053efc2dac91132a5274`; `libnccl-dev` `45824393a1e70d3407537e6f168186f312fd3d588edabd836d01fef4198e1c0c`.

   The build settings (CUDA on, target architecture 120, compiler path) and the versions of the build tools (scikit-build-core, CMake, Ninja) are fixed in [`pyproject.toml`](pyproject.toml); CMake and Ninja are required at those versions even when other versions are installed on the system, and a build is accepted only under the acceptance rule of [DR-013](docs/decisions/DR-013-training-device-policy.md) §2 (Ninja generator, matching tool versions, no conflicting overrides). The CUDA runtime and NCCL are linked statically into LightGBM.

3. Create the environment exactly as locked, with the tree-based (`ml`), neural (`dl`, PyTorch for CUDA 12.8) and development (`dev`) groups. This step builds LightGBM from source:

   ```bash
   uv sync --locked --extra ml --extra dl --extra dev
   ```

4. Obtain the raw data from Kaggle by accepting the Competition Rules, and place both files in `data/raw/` ([dataset documentation](docs/dataset.md)). The data are never committed.

5. Validate the raw data and prepare the development files, **once** ([execution rules](AGENTS.md#experiment-execution-responsibility)):

   ```bash
   .venv/bin/review-capacity --log-file logs/prepare-data.log prepare-data
   ```

   This one-time build ran on 10 October 2026. After it, every step reads only the verified processed files in `data/processed/`, and the raw files are read again only by the confirmatory Test evaluation ([data access rule](docs/dataset.md#data-access-after-the-parquet-build)).

   Every command logs its progress with timestamps to the terminal; `--log-file PATH` (before the command name) also keeps the log in a file, as an evidence-producing run should. `review-capacity --help` lists the options and exit codes.

6. Plan the search, then run the tree-based development search. Both are consequential steps governed by the [execution rules](AGENTS.md#experiment-execution-responsibility); the run starts only after its scope is approved. The plan reads no data and fits no model. Its `cost_prediction.json` is identified by its SHA-256, which the development run takes as the approval of the predicted cost ([DR-015](docs/decisions/DR-015-common-training-rules.md) §7):

   ```bash
   .venv/bin/review-capacity plan-search
   ```

   ```bash
   sha256sum outputs/search_plan/<UTC start time>/cost_prediction.json
   ```

   ```bash
   .venv/bin/review-capacity run-development --plan outputs/search_plan/<UTC start time> --approved-cost <SHA-256 of cost_prediction.json>
   ```

   `run-development` runs the declared search of each tree-based configuration (`--configurations` chooses a subset), selects on V1, scores V1 and V2, evaluates R3 and the ranking, and times scoring, writing dated, ignored records, `summary.md` and `run.log` to `outputs/development/<UTC start time>/` (`--output` must name a new directory inside `outputs/`). It refuses to start if the plan differs from the approved cost or from the training declaration. The conditions of the timing session are recorded with each timing as context; no condition is required ([DR-015](docs/decisions/DR-015-common-training-rules.md) §9). `review-capacity --help` lists every option.

**Continuous integration:** code-affecting pull requests to `main` receive the checks defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml); CI never trains or scores a model on research data, reads research data or writes research output; unit tests fit only small models on synthetic data.

## Repository map

| Path | Contents |
|---|---|
| [`docs/protocol.md`](docs/protocol.md) | Research protocol: questions, hypotheses, analytic approach, data requirements, methods, statistical plan, design items, freeze conditions |
| [`docs/decisions/`](docs/decisions/README.md) | Decision Records DR-001–DR-014: the reason for each design choice |
| [`docs/research-foundation.md`](docs/research-foundation.md) | Research problem, aim, objectives, contribution |
| [`docs/literature-review.md`](docs/literature-review.md) | Literature review and research gap |
| [`docs/literature-evidence-matrix.md`](docs/literature-evidence-matrix.md) | Field-by-field comparison of the selected studies |
| [`docs/greenwich-criteria-checklist.md`](docs/greenwich-criteria-checklist.md) | University of Greenwich requirements mapped to the repository, with open points |
| [`docs/literature-search-method.md`](docs/literature-search-method.md) | How the literature was searched, screened and selected |
| [`docs/references.md`](docs/references.md) | Reference register with verification status |
| [`docs/dataset.md`](docs/dataset.md) | Dataset source, provenance, data-use terms, version-control policy, dataset manifest |
| [`docs/data-preparation.md`](docs/data-preparation.md) | Data-preparation implementation and evidence record (methodology stage 6) |
| [`docs/gpu-acceptance-probe.md`](docs/gpu-acceptance-probe.md) | GPU acceptance probe of DR-013 §8: checks, tolerances and results |
| [`docs/feature-catalog.md`](docs/feature-catalog.md) | Column-by-column data dictionary |
| [`config/features.yaml`](config/features.yaml) | Machine-readable model-input roles (433 inputs: 431 declared columns and 2 time-of-day inputs) |
| [`config/dataset.yaml`](config/dataset.yaml) | Machine-readable raw files, DR-001 partitions and their integrity anchors |
| [`config/models.yaml`](config/models.yaml) | Machine-readable DR-013 device and numerical settings and the recorded LightGBM build |
| [`src/review_capacity/`](src/review_capacity/) | Research code: data loading, validation and preparation (`data/`), model inputs (`features/`), model interface, configurations and device controls (`models/`), development record contracts and Markdown summary specification (`development/`), the shared daily review queue and V1 objective (`evaluation/`), the training declaration, candidate sampler and search-cost prediction (`training/`); command `review-capacity` |
| [`pyproject.toml`](pyproject.toml), [`uv.lock`](uv.lock), [`.python-version`](.python-version) | Pinned dependencies, locked environment (uv) and Python version |
| `evidence/` | Engineering evidence: [GPU memory probe results](evidence/gpu_probe/README.md) (Issue #50); [GPU acceptance probe records](evidence/gpu_acceptance_probe/README.md) (Issue #77) |
| [`tests/`](tests/) | Unit tests on synthetic data; opt-in integration tests on the local processed development files (no raw file is read), subject to the execution rules in [`AGENTS.md`](AGENTS.md#experiment-execution-responsibility) |
| [`notebooks/`](notebooks/README.md) | Data-understanding notebooks 01–04 |
| [`scripts/`](scripts/) | Engineering scripts (GPU memory feasibility probe, Issue #50) |
| [`AGENTS.md`](AGENTS.md) | Working rules for the project and AI-assisted work |
| [`data/datapackage.json`](data/datapackage.json) | Dataset manifest (Frictionless Data Package): file checksums, sizes, rows and label-free partition summary |
| `data/raw/`, `data/processed/` | Local raw data and processed development files (never committed) |

## Status

**Research design documented and accepted in line with the University of Greenwich criteria ([criteria checklist](docs/greenwich-criteria-checklist.md)); remaining design items pending; protocol not yet frozen.**

The ethics and governance status is documented in the [protocol](docs/protocol.md#ethics); no formal ethics approval or exemption is claimed.

Next steps:

1. Resolution of the protocol's remaining design items, in line with the University of Greenwich criteria ([criteria checklist](docs/greenwich-criteria-checklist.md)); supervisor feedback welcome throughout.
2. Training rules: search procedure, early stopping, tie handling, seeds, neural-network representation, calibration details.
3. Development runs on Train, Validation 1 (V1) and Validation 2 (V2); protocol freeze; single Test evaluation; statistical analysis and reporting.
