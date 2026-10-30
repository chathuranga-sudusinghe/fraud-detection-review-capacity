# Development run

The aggregate records of the development run of the study: the search of the eight configurations of the model set and of the logistic-regression reference ([DR-011](../../docs/decisions/DR-011-model-set-tree-based-and-neural-families.md)) under the common training rules ([DR-015](../../docs/decisions/DR-015-common-training-rules.md)), the selection of each configuration on Validation 1 (V1, relative days 106–126), and its scoring of V1 and Validation 2 (V2, relative days 127–154). The run reads Train, V1 and V2 only; the Test period is sealed. The command sequence is in [running the pipeline](../../docs/running-the-pipeline.md), step 4.

Each run directory is written by `review-capacity promote-evidence` (Issue #179): the run's JSON (JavaScript Object Notation) records, the generated `summary.md`, `run.log` and a `manifest.json` with the SHA-256 (Secure Hash Algorithm, 256-bit) value of every file. Row-level scores, fitted models, the candidate lists and the fitted preprocessing artifacts stay in the local run directory under `outputs/development/`, which Git ignores. The saved records are unchanged copies and are never edited.

A configuration is a model with a class weighting, written `<model>-<weighting>`. The models are `LGB` (LightGBM), `Cat` (CatBoost), `RES` (the ResNet-style multilayer perceptron, MLP) and `FTT` (the Feature Tokenizer + Transformer, FT-Transformer); the weightings are `u` (unweighted) and `b` (balanced class weighting). `LR` is the logistic-regression reference, a descriptive baseline and not one of the eight configurations of the model set.

| Directory | Configurations | Files | Code commit |
|---|---|---|---|
| [`2026-10-25T134341Z/`](2026-10-25T134341Z/) | `LR` and the eight configurations `LGB-u`, `Cat-u`, `RES-u`, `FTT-u`, `LGB-b`, `Cat-b`, `RES-b`, `FTT-b` | 474 listed in `manifest.json` (475 with the manifest itself) | `6e2ff564e7b5a0dc505b7ee9c9465b4449da0092` (clean tree) |

SHA-256 of `2026-10-25T134341Z/manifest.json`: `28b7e6d9a59966b171eeca54a9b11bd6101983a8316f40a8ea0049663fe25d6d`.

**What the run summary fills.** Its comparison tables hold the selected settings, the search trials, the development metrics, the descriptive contrasts of the V2 Recall@2 % and the computational cost. The development metrics are the V1 objective and the V2 recall of every selected main-seed model under R3, the daily top-K policy: each relative day, the K transactions with the highest scores are reviewed, where K is the fixed daily review budget (reviews per relative day) of a review capacity of 1 %, 2 %, 5 % or 10 % of the transactions ([DR-002](../../docs/decisions/DR-002-review-capacity-definition.md)). Recall@2 % is the share of the V2 frauds among the transactions reviewed at the 2 % capacity (K = 56). The seed variation, the CatBoost central processing unit (CPU) against graphics processing unit (GPU) comparison and the RQ3 (research question 3, cost-aware capacity use) ratios appear in this summary as empty columns, because the [analysis runs](../analysis/README.md) produce them from the closed run.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| Train, V1, V2, Test | Model-fitting partition (relative days 1–105); Validation 1 (106–126); Validation 2 (127–154); sealed confirmatory partition (155–182) |
| `LGB`, `Cat`, `RES`, `FTT` | LightGBM; CatBoost; ResNet-style multilayer perceptron (MLP); Feature Tokenizer + Transformer (FT-Transformer) |
| `u`, `b` | Unweighted training; balanced class weighting |
| `LR` | Logistic-regression reference (descriptive baseline) |
| K | Fixed daily review budget (reviews per relative day) |
| Recall@2 % | Share of the V2 frauds among the transactions reviewed at the 2 % capacity (K = 56) |
| R3 | Decision policy: daily top-K |
| RQ3 | Research question 3: cost-aware capacity use |
| DR-NNN | Decision Record, numbered NNN |
| SHA-256 | Secure Hash Algorithm, 256-bit digest |
| JSON | JavaScript Object Notation |
| CPU, GPU | Central processing unit, graphics processing unit |
