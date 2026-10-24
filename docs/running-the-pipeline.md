# Running the Pipeline

The command sequence of the development pipeline, with what each step reads and writes. The environment and the processed data are prepared first ([environment setup](environment-setup.md)). The design the steps implement is in the [protocol](protocol.md) and [DR-015](decisions/DR-015-common-training-rules.md).

Every step below is consequential and is governed by the [execution rules](../AGENTS.md#experiment-execution-responsibility): the project owner runs it, and a run starts only after its scope is approved. Every run writes to a new, dated directory `outputs/<type>/<UTC start time>/` (ignored by Git) and keeps its log there. `review-capacity --help` lists every command, option and exit code.

| Step | Command | Reads | Writes |
|---|---|---|---|
| 1 | `plan-search` | no data | `outputs/search_plan/<UTC start time>/` |
| 2 | `fit-neural-preprocessing` | processed development files | `outputs/neural_preprocessing/<UTC start time>/` |
| 3 | `preflight-neural` | the preprocessing artifact; synthetic inputs | `outputs/neural_preflight/<UTC start time>/` |
| 4 | `run-development` | steps 1–3; processed development files | `outputs/development/<UTC start time>/` |
| 5 | `analyse-development` | the closed development run of step 4 | `outputs/analysis/<UTC start time>/` |
| 6 | `summarise-run`, `promote-evidence` | a completed run | `summary.md`; `evidence/<run type>/<run id>/` |

The Test period is never read by any of these steps.

## 1. Plan the search

The plan reads no data and fits no model. Its `cost_prediction.json` is identified by its SHA-256, which the development run takes as the approval of the predicted cost ([DR-015](decisions/DR-015-common-training-rules.md) §7):

```bash
.venv/bin/review-capacity plan-search
```

```bash
sha256sum outputs/search_plan/<UTC start time>/cost_prediction.json
```

## 2. Fit the shared neural preprocessing

The shared Train-only preprocessing of the two neural models ([DR-012](decisions/DR-012-neural-network-input-representation.md)) is fitted once, with the declared seed. It is needed by the neural models and by the logistic-regression reference:

```bash
.venv/bin/review-capacity fit-neural-preprocessing
```

It writes the artifact (`manifest.json`, `arrays.npz`), its SHA-256 in `result.json`, the Validation 1 and Validation 2 diagnostics and `run.log`.

## 3. Fix the neural micro-batches

The micro-batch preflight runs on the project GPU, with synthetic inputs of the vocabularies of the preprocessing artifact and no research data. It follows the neural batching acceptance (`review-capacity probe-neural-batching`, [GPU acceptance probe](gpu-acceptance-probe.md) Section 9):

```bash
.venv/bin/review-capacity preflight-neural --preprocessing outputs/neural_preprocessing/<UTC start time>
```

It writes `preflight.json`, which must have been made for the same artifact, training declaration, model settings, package source and GPU as the run that uses it.

## 4. Run the development search

```bash
.venv/bin/review-capacity run-development --plan outputs/search_plan/<UTC start time> --approved-cost <SHA-256 of cost_prediction.json> --preprocessing outputs/neural_preprocessing/<UTC start time> --preflight outputs/neural_preflight/<UTC start time>/preflight.json
```

In one run it handles the logistic-regression reference (`LR`, first, so that a convergence failure shows before the model searches start) and the eight configurations in [DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md) order (`--configurations` chooses a subset). For each it runs the declared search, selects on V1, scores V1 and V2, evaluates R3 and the ranking, and times scoring. It writes dated records, `summary.md` and `run.log`; `--output` must name a new directory inside `outputs/`.

- A neural model is reloaded only after its saved hyperparameters and batching are checked against the declaration and the recorded batch plan.
- The run record names the approved cost prediction, and the run refuses to start if the plan differs from the approved cost or from the training declaration.
- The conditions of the timing session are recorded with each timing as context; no condition is required ([DR-015](decisions/DR-015-common-training-rules.md) §9).

## 5. Analyse the completed run

The results that follow from a completed run are produced in a new **analysis run** that reads the closed development run and never changes it. The analysis directory holds a `source/` link to the development run, so that every reference resolves. Its stages are:

- `thresholds`: the threshold policies R0–R2 and their workload;
- `calibration`: the calibrators, R4 and the RQ3 cost analysis;
- `h3`: the selection of the H3 configuration and its label-latency check;
- `refits`: the fixed-length refits and the CatBoost CPU–GPU comparison. They train models, using the preprocessing artifact and the batch plans that the development run recorded, and are named explicitly.

Each stage can be repeated in a new analysis run. The `thresholds`, `calibration` and `h3` stages read the saved scores only and fit no model; the `h3` record is the binding selection of the H3 configuration, committed before the Test period is unsealed:

```bash
.venv/bin/review-capacity analyse-development --source outputs/development/<UTC start time>
```

```bash
.venv/bin/review-capacity analyse-development --source outputs/development/<UTC start time> --stages refits
```

## 6. Summarise and promote

`summarise-run` regenerates `summary.md` of a run directory from its records alone, also after an interrupted run; it reads no data and fits no model. `promote-evidence` copies a completed run's aggregate evidence (records, `summary.md`, `run.log` and a manifest with SHA-256 values) to `evidence/<run type>/<run id>/`, a new directory, and never row-level scores, fitted preprocessing artifacts or models. It is a separate, explicit step that no run calls.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| CPU, GPU | Central processing unit, graphics processing unit |
| DR-NNN | Decision Record, numbered NNN |
| H3 | Hypothesis 3 of the [protocol](protocol.md): alert-rate transfer fidelity |
| LR | Logistic-regression reference |
| R0–R4 | Decision policies of the protocol: default 0.5 threshold; F1-optimal threshold; capacity-rate threshold; daily top-K; calibrated cost-aware capped policy |
| RQ3 | Research question 3 of the protocol: cost-aware capacity use |
| SHA-256 | Secure Hash Algorithm, 256-bit digest |
| Train, V1, V2, Test | Model-fitting partition (relative days 1–105); Validation 1 (106–126); Validation 2 (127–154); sealed confirmatory partition (155–182) |
