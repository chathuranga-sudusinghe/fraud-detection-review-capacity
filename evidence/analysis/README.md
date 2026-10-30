# Analysis runs of the development run

The aggregate records of the three analysis runs that read the closed [development run](../development/README.md) and never change it ([running the pipeline](../../docs/running-the-pipeline.md), step 5). All use Train, Validation 1 (V1) and Validation 2 (V2) only; the Test period is sealed. The development partitions and their roles are in [DR-001](../../docs/decisions/DR-001-temporal-split-strategy.md).

Each run directory is written by `review-capacity promote-evidence` (Issue #179): the run's JSON (JavaScript Object Notation) records, the generated `summary.md`, `run.log` and a `manifest.json` with the SHA-256 (Secure Hash Algorithm, 256-bit) value of every file. Row-level scores, calibrators, fitted models and the refit job files stay in the local run directory under `outputs/analysis/`, which Git ignores. The saved records are unchanged copies and are never edited.

Terms used in the table:

- **R0–R2** are the threshold policies (a default 0.5 threshold; the threshold that maximises F1, the harmonic mean of precision and recall; and a threshold matched to the capacity share) and **R4** is the calibrated, cost-aware, capped policy. The workload of a policy is the number of transactions it flags per relative day against K, the fixed daily review budget (reviews per relative day).
- **RQ3** is research question 3 (cost-aware capacity use). Its analysis gives, per capacity, the saturation ratio and the break-even ratio of the miss-to-review cost.
- **Platt scaling** and **isotonic regression** are the two calibrators that map a model's scores to probabilities.
- **H3** is Hypothesis 3 (alert-rate transfer fidelity), tested on one configuration (a model with a class weighting) that the `h3` stage selects.
- **CPU–GPU** compares a CatBoost fit on the central processing unit (CPU) with the fit on the graphics processing unit (GPU).

| Directory | Stages | Files | What it fills |
|---|---|---|---|
| [`2026-10-29T013124Z/`](2026-10-29T013124Z/) | `thresholds`, `calibration`, `h3` | 1,510 listed in `manifest.json` (1,511 with the manifest itself) | threshold policies R0–R2 and their workload; the Platt and isotonic calibrators and the R4 evaluation; the RQ3 cost analysis and its saturation and break-even ratios; the binding selection of the H3 configuration and its label-latency check |
| [`2026-10-29T013414Z/`](2026-10-29T013414Z/) | `h3`, `rolling` | 25 listed (26 with the manifest) | the H3 selection again and the rolling-origin supplement of the H3 configuration (three origins) |
| [`2026-10-29T034844Z/`](2026-10-29T034844Z/) | `refits` | 188 listed (189 with the manifest) | the seed variation (seeds 42, 43 and 44), the same-seed repeats and the CatBoost CPU–GPU comparison |

All three runs were made from code commit `6e2ff564e7b5a0dc505b7ee9c9465b4449da0092` (clean tree), the same commit and package source fingerprint (`06540fd34ff853bfdc88ca3bc4ac291f656f392683bb6261f513353667abb5be`) as the development run.

| Manifest | SHA-256 |
|---|---|
| `2026-10-29T013124Z/manifest.json` | `a1efab8e2acab0c5915397df98bb385a746b1c85587072ca00f44c425b874865` |
| `2026-10-29T013414Z/manifest.json` | `c1c49e132f91debb9227bbebb2ad608e0be3a5912d117265f02725eb44441258` |
| `2026-10-29T034844Z/manifest.json` | `57523a05b7cde7e70380257f4c10308b037d2013baf6ded413fe79ddd1520a0b` |

**Which summary shows which table.** Each summary has the same comparison layout, and a table whose stage the run did not execute shows empty columns. The V1 and V2 metrics are in the development run; the RQ3 ratios in `2026-10-29T013124Z`; the H3 selection and the rolling-origin table in `2026-10-29T013414Z`; the seed variation and the CPU–GPU comparison in `2026-10-29T034844Z`.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| Train, V1, V2, Test | Model-fitting partition (relative days 1–105); Validation 1 (106–126); Validation 2 (127–154); sealed confirmatory partition (155–182) |
| R0–R4 | Decision policies: default 0.5 threshold; F1-optimal threshold; capacity-rate threshold; daily top-K; calibrated cost-aware capped policy |
| F1 | Harmonic mean of precision and recall |
| K | Fixed daily review budget (reviews per relative day) |
| RQ3 | Research question 3: cost-aware capacity use |
| Platt scaling, isotonic regression | The two calibrators of R4 (logistic and monotone mappings from score to probability) |
| DR-NNN | Decision Record, numbered NNN |
| H3 | Hypothesis 3: alert-rate transfer fidelity |
| CPU, GPU | Central processing unit, graphics processing unit |
| SHA-256 | Secure Hash Algorithm, 256-bit digest |
| JSON | JavaScript Object Notation |
