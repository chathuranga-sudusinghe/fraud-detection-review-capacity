# Confirmatory Test evaluation

The aggregate evidence of the single confirmatory Test evaluation of the study ([protocol](../../docs/protocol.md), frozen at the Git tag `protocol-freeze-v1`; command sequence in [running the pipeline](../../docs/running-the-pipeline.md), step 7). The evaluation scored the sealed Test period (relative days 155–182) once and evaluated H1, H2, H3, RQ3 and the false-negative slices on the frozen models, thresholds and calibrators.

The run directory is written by `review-capacity promote-evidence` (Issue #221): the run's JSON (JavaScript Object Notation) records, the generated `summary.md`, the four figures, `run.log` and a `manifest.json` with the SHA-256 (Secure Hash Algorithm, 256-bit) value of every listed file. The row-level score files and the evaluation working files stay local.

| Directory | Run | Tag commit | Records | Files in `manifest.json` |
|---|---|---|---|---|
| [`2026-11-01T104408Z/`](2026-11-01T104408Z/) | The single confirmatory Test evaluation: scoring and evaluation phases, one attempt each, closed as completed | `9997da7a292c266908e97c5458d10912fbd81d74` | 70 listed in the closing record (71 JSON files with the closing record) | 77 |

**What the run summary holds.** The primary confirmatory results (Holm decisions over H1, H2 and H3), the sensitivities, the secondary outcomes of every configuration, the daily-budget outcomes of R2, the calibration diagnostics with their reliability tables, the RQ3 analysis (with the calibration sensitivity of isotonic against Platt and the non-monotone curves), the false-negative slices, the scoring times and the fields of every record. Values are recorded fields, or deterministic comparisons and arithmetic derived from recorded fields.

**Figures.** `figures/cost_difference.svg` and `figures/utilisation.svg` (RQ3, Platt), `figures/reliability_platt.svg` and `figures/reliability_isotonic.svg` (calibration on the Test rows, descriptive).

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| H1, H2, H3 | Confirmatory hypotheses of the protocol: family effect, weighting equivalence, alert-rate transfer |
| RQ3 | Research question 3: cost-aware capacity use |
| R2 | Capacity-rate threshold policy |
| Test | Sealed confirmatory partition (relative days 155–182) |
| JSON | JavaScript Object Notation |
| SHA-256 | Secure Hash Algorithm, 256-bit digest |
