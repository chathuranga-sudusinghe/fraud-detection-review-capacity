# Notebooks

Executed notebooks are research evidence for the data-understanding stage. Each is run and saved with its outputs. Committed outputs contain aggregates and schema information only: raw competition data never enter the repository ([`docs/dataset.md`](../docs/dataset.md#data-use-terms)).

The final Test period (relative days 155–182) is sealed ([DR-001](../docs/decisions/DR-001-temporal-split-strategy.md)). A relative day is `TransactionDT` (time offset in seconds from an undisclosed reference point) divided by 86,400 seconds and rounded down.

## Index

| Notebook | Purpose | Data read | Test-sealing approach |
|---|---|---|---|
| [`01_dataset_feasibility_audit.ipynb`](01_dataset_feasibility_audit.ipynb) | Pre-freeze feasibility audit: integrity, temporal structure, split choice, development-data target structure, capacity, power, features, model pair, weighting, policies, costs, drift, entity dependence, leakage, suitability | Full raw tables | Split chosen from transaction volume only; target domain checked once before the split was fixed (logged in §3.6); all target-based analysis restricted to development rows |
| [`02_raw_table_schema_inventory.ipynb`](02_raw_table_schema_inventory.ipynb) | Schema inventory of each raw table separately: checksums, column inventory, target location, shared key, match with [`config/features.yaml`](../config/features.yaml) | Headers; key columns; raw-row preview (local only) | Test-period rows read without `isFraud`; preview outputs removed before commit |
| [`03_merged_dataset_inspection.ipynb`](03_merged_dataset_inspection.ipynb) | Merged (joined) development dataset: join validity, shape and roles, class imbalance (bar charts), identity coverage, merged missingness | Keys (all rows); full columns for the first 511,998 rows only (relative days 1–154) | Test represented by its row count only; no Test-period label read |
| [`04_development_data_corrections.ipynb`](04_development_data_corrections.ipynb) | Corrections to notebooks 01 and 02: raw header names, oracle ceilings on Validation 2, Population Stability Index with a separate missing bin, `V` missingness blocks; each recomputed with the original and the corrected method ([Issue #59](https://github.com/chathuranga-sudusinghe/fraud-detection-review-capacity/issues/59)) | Header lines; keys (all rows); full columns for the first 511,998 rows only (relative days 1–154) | As notebook 03: day-154 boundary asserted; no Test-period label read |

| [`05_report_figures.ipynb`](05_report_figures.ipynb) | Report figures ([Issues #230](https://github.com/chathuranga-sudusinghe/fraud-detection-review-capacity/issues/230) and [#232](https://github.com/chathuranga-sudusinghe/fraud-detection-review-capacity/issues/232)): exports the saved class-balance and prevalence charts of notebooks 03 and 01 byte for byte, draws Recall@2% by configuration and the daily R2 alert demand against K (fixed daily review budget) from the confirmatory evidence, and redraws the run's two R4 cost figures as 2 × 2 panels with the run's own curve functions; writes the figures and a SHA-256 manifest to `outputs/report_figures/<UTC time>/` | Saved outputs of notebooks 01 and 03; promoted aggregate records of the confirmatory run `evidence/confirmatory/2026-11-01T104408Z/records/` | Runs after the single confirmatory evaluation and reads only its aggregate records; no raw, processed or row-level data |

Reading order: 02 → 03 for the current data view; 01 for the full feasibility evidence and split rationale; 04 for the corrections to 01 and 02; 05 for the report figures.

From the data-preparation stage onwards, the reusable code is the `review_capacity` package ([`src/review_capacity/`](../src/review_capacity/), Issue #62). New notebooks load, validate and partition the data through it instead of repeating that logic. Notebooks 01–04 stay unchanged as saved evidence.

## Notebook 01 section map

Notebook 01 is retained unchanged as the timestamped pre-freeze record. Documents cite its sections by number:

| Section | Topic | Cited by |
|---|---|---|
| §2.1 | Files, sizes and Secure Hash Algorithm, 256-bit (SHA-256) checksums | [`references.md`](../docs/references.md) (REF-047); notebook 02 |
| §2.3 | Column types | [DR-008](../docs/decisions/DR-008-feature-declaration-and-model-input-policy.md), [feature catalog](../docs/feature-catalog.md) |
| §2.7 | Obvious data-quality checks | [Feature catalog](../docs/feature-catalog.md) |
| §3 | Temporal structure (volume only) | [DR-001](../docs/decisions/DR-001-temporal-split-strategy.md), [DR-003](../docs/decisions/DR-003-rq2-h3-policy-transfer-fidelity.md), [DR-009](../docs/decisions/DR-009-overflow-handling-for-threshold-policies.md), [protocol](../docs/protocol.md) |
| §4, §4.3 | Split candidates; fixed split and partition hashes | DR-001 |
| §5, §5.1 | Target structure; prevalence over time (development data) | DR-001, feature catalog; notebook 03 |
| §6 | Review-capacity feasibility | [DR-002](../docs/decisions/DR-002-review-capacity-definition.md), DR-003, [DR-005](../docs/decisions/DR-005-sesoi-and-h1-h2-inference.md), [DR-007](../docs/decisions/DR-007-rq3-cost-ratio-and-break-even.md) |
| §7 | Statistical power projection | DR-002, DR-005 |
| §8, §8.1 | Feature suitability; feature-group assessment | DR-008, feature catalog |
| §9 | LightGBM vs CatBoost as a model pair | DR-008 |
| §11 | Decision-policy feasibility | DR-002, DR-007 |
| §12 | Cost-sensitive and calibration feasibility | DR-007 |
| §14 | Temporal drift between partitions | DR-001, DR-003 |
| §15 | Entity recurrence and label dependence | DR-001, DR-005, [DR-006](../docs/decisions/DR-006-bootstrap-inference-strategy.md) |
| §17 | Master of Science (MSc) research suitability | DR-003 |

Sections not listed (e.g. §10 class weighting, §13 false-negative feasibility, §16 leakage audit, §18–19 practice and University alignment) inform the protocol without section-number citations.

## Reading notebook 01 against the current design

Notebook 01 is the pre-freeze feasibility record and is kept unchanged. Where its recommendations differ from the current design, the record in the right-hand column applies.

| Notebook 01 says | Current design | Applies now |
|---|---|---|
| Title "…Study of Gradient-Boosting Models and Decision Policies"; programme "MSc Data Science" (also in notebooks 02 and 03) | Title naming tree-based and neural models; programme MSc Data Science and Its Applications | [Protocol](../docs/protocol.md), [README](../README.md) |
| §9, §17 and the recommended model scope: LightGBM and CatBoost as the model pair; no neural networks | Eight configurations: LightGBM, CatBoost, ResNet-style multilayer perceptron (MLP) and Feature Tokenizer + Transformer (FT-Transformer), each unweighted and with balanced class weighting | [DR-011](../docs/decisions/DR-011-model-set-tree-based-and-neural-families.md) |
| Decision section: H3₀ (the null hypothesis of H3) stated as the realised alert rate lying **within** the tolerance band | Equivalence formulation: H3₀ is the rate lying **outside** the band, tested by Two One-Sided Tests (TOST) | [DR-003](../docs/decisions/DR-003-rq2-h3-policy-transfer-fidelity.md) |
| §11: calibration matters for R0 and R4 | R0 applies 0.5 to the raw model score; calibration applies to R4 only | [Protocol](../docs/protocol.md#calibration) |
| §12: cost-ratio grid starting at 1 | ρ (miss-to-review cost ratio) ∈ {2, …, 1000}; ρ ≤ 1 excluded | [DR-007](../docs/decisions/DR-007-rq3-cost-ratio-and-break-even.md) |
| §8–§9: 432 raw features | 431 declared input columns (`TransactionID` and raw `TransactionDT` excluded), plus two time-of-day inputs derived from `TransactionDT`: 433 model inputs | [DR-008](../docs/decisions/DR-008-feature-declaration-and-model-input-policy.md), [`config/features.yaml`](../config/features.yaml) |
| §17: H1 and H2 as model-engineering comparisons | H1 is a model-family (tree-based vs neural) hypothesis | [Research foundation](../docs/research-foundation.md#research-contribution), DR-011 |
| §19 and references L46–L47: Kaggle rules and two sources still to be verified | Rules verified (REF-049); REF-041 verified; REF-042 partially verified | [Dataset](../docs/dataset.md#data-use-terms), [references](../docs/references.md) |

**Computation findings**, corrected in [notebook 04](04_development_data_corrections.ipynb):

- **Notebook 01 §6, oracle recall and precision ceilings:** computed from pooled totals instead of the daily oracle captures. For the scenarios used by the design (1 %, 2 %, 5 %, 10 %) the recall ceilings are unchanged and the precision ceilings agree to three decimals; the exploratory 3 % and 3.5 % rows were overstated (notebook 04 §4).
- **Notebook 01 §14.1, Population Stability Index (PSI) of categorical columns:** missing values fell into the "other" bin instead of a separate missing bin. With the separate bin, the PSI of 12 categorical features changes by more than 0.001 in at least one period pair (largest: `id_13`), but no feature changes band (notebook 04 §5).
- **Notebook 01 §8, missingness blocks of `V1`–`V339`:** columns were grouped by their number of missing Train rows. Notebook 04 §6 confirms that each of the 14 groups shares one missing-row pattern.
- **Notebook 02 §2.2, duplicate column names:** the check could not fail, because pandas renames duplicate headers. The raw header lines contain no repeated name (notebook 04 §2).

## Running and committing

- Use the project kernel **Python (fraud-detection-review-capacity)** from the repository root's `.venv`.
- **The raw files are read only by the one-time build of the processed development files** ([Data Access After the Parquet Build](../docs/dataset.md#data-access-after-the-parquet-build)). Notebooks 01–03 are historical evidence records that read the raw files as they ran: **none of them is rerun** while the Test period is sealed. Their saved outputs are the evidence; corrections and any new analysis are made in new notebooks that read the verified processed development files, through the issue workflow.
- Notebook 01's loader reads the complete target column before restricting analysis to development rows, which is a further reason not to rerun it. **Notebook 02** read headers, key columns and (locally) a raw-row preview with `isFraud` excluded for Test-period rows. **Notebook 03** read keys for all rows and full columns for development rows only. Neither trains a model.
- Before committing notebook 02, remove the outputs of cells tagged `local-only`:

  ```bash
  .venv/bin/jupyter nbconvert --to notebook --inplace --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_all_outputs_tags local-only notebooks/02_raw_table_schema_inventory.ipynb
  ```

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| DR-NNN | Decision Record NNN in `docs/decisions/` |
| MSc | Master of Science |
| FT-Transformer | Feature Tokenizer + Transformer (a neural model for tabular data) |
| H1, H2, H3; H3₀ | Confirmatory hypotheses: model-family effect; class-weighting effect; alert-rate transfer fidelity. H3₀ is the null hypothesis of H3 |
| K | Fixed daily review budget (reviews per relative day); 56 at the primary 2 % capacity |
| LGB-b | LightGBM with balanced class weighting, the configuration of H3 |
| MLP | Multilayer perceptron |
| PSI | Population Stability Index (a measure of distribution shift) |
| R0, R2, R3, R4 | Decision policies: default 0.5 threshold; validation-selected 2 % rate threshold; daily top-K review; calibrated cost-aware capped review |
| Recall@2% | Share of all frauds reviewed under the daily top-K policy at 2 % capacity |
| UTC | Coordinated Universal Time (run-directory timestamps) |
| REF-NNN | Reference identifier in `docs/references.md` |
| ρ | Miss-to-review cost ratio |
| `isFraud` | Target: 1 = fraud, 0 = legitimate |
| Relative day | `TransactionDT` divided by 86,400 seconds, rounded down |
| SHA-256 | Secure Hash Algorithm, 256-bit (file checksum) |
| TOST | Two One-Sided Tests (equivalence test) |
| Test | Relative days 155–182; labels sealed |
| `TransactionDT` | Time offset in seconds from an undisclosed reference point |
| `TransactionID` | Transaction identifier (excluded from model input) |
