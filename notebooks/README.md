# Notebooks

Executed notebooks are research evidence for the data-understanding stage. Each is run by the project owner and saved with its outputs. Committed outputs contain aggregates and schema information only: raw competition data never enter the repository ([`docs/dataset.md`](../docs/dataset.md#data-use-terms)).

The final Test period (relative days 155–182) is sealed ([DR-001](../docs/decisions/DR-001-temporal-split-strategy.md)). A relative day is `TransactionDT` (time offset in seconds from an undisclosed reference point) divided by 86,400 seconds and rounded down.

## Index

| Notebook | Purpose | Data read | Test-sealing approach |
|---|---|---|---|
| [`01_dataset_feasibility_audit.ipynb`](01_dataset_feasibility_audit.ipynb) | Pre-freeze feasibility audit: integrity, temporal structure, split choice, development-data target structure, capacity, power, features, model pair, weighting, policies, costs, drift, entity dependence, leakage, suitability | Full raw tables | Split chosen from transaction volume only; target domain checked once before the split was frozen (logged in §3.6); all target-based analysis restricted to development rows |
| [`02_raw_table_schema_inventory.ipynb`](02_raw_table_schema_inventory.ipynb) | Schema inventory of each raw table separately: checksums, column inventory, target location, shared key, match with [`config/features.yaml`](../config/features.yaml) | Headers; key columns; raw-row preview (local only) | Test-period rows read without `isFraud`; preview outputs removed before commit |
| [`03_merged_dataset_inspection.ipynb`](03_merged_dataset_inspection.ipynb) | Merged (joined) development dataset: join validity, shape and roles, class imbalance (bar charts), identity coverage, merged missingness | Keys (all rows); full columns for the first 511,998 rows only (relative days 1–154) | Test represented by its row count only; no Test-period label read |

Reading order: 02 → 03 for the current data view; 01 for the full feasibility evidence and split rationale.

## Notebook 01 section map

Notebook 01 is retained unchanged as the timestamped pre-freeze record. Documents cite its sections by number:

| Section | Topic | Cited by |
|---|---|---|
| §2.1 | Files, sizes and SHA-256 checksums | [`references.md`](../docs/references.md) (REF-047); notebook 02 |
| §2.3 | Column types | [DR-008](../docs/decisions/DR-008-feature-declaration-and-model-input-policy.md), [feature catalog](../docs/feature-catalog.md) |
| §2.7 | Obvious data-quality checks | [Feature catalog](../docs/feature-catalog.md) |
| §3 | Temporal structure (volume only) | [DR-001](../docs/decisions/DR-001-temporal-split-strategy.md), [DR-003](../docs/decisions/DR-003-rq2-h3-policy-transfer-fidelity.md), [DR-009](../docs/decisions/DR-009-overflow-handling-for-threshold-policies.md), [protocol](../docs/protocol.md) |
| §4, §4.3 | Split candidates; frozen split and partition hashes | DR-001 |
| §5, §5.1 | Target structure; prevalence over time (development data) | DR-001, feature catalog; notebook 03 |
| §6 | Review-capacity feasibility | [DR-002](../docs/decisions/DR-002-review-capacity-definition.md), DR-003, [DR-005](../docs/decisions/DR-005-sesoi-and-h1-h2-inference.md), [DR-007](../docs/decisions/DR-007-rq3-cost-ratio-and-break-even.md) |
| §7 | Statistical power projection | DR-002, DR-005 |
| §8, §8.1 | Feature suitability; feature-group assessment | DR-008, feature catalog |
| §9 | LightGBM vs CatBoost as a model pair | DR-008 |
| §11 | Decision-policy feasibility | DR-002, DR-007 |
| §12 | Cost-sensitive and calibration feasibility | DR-007 |
| §14 | Temporal drift between partitions | DR-001, DR-003 |
| §15 | Entity recurrence and label dependence | DR-001, DR-005, [DR-006](../docs/decisions/DR-006-bootstrap-inference-strategy.md) |
| §17 | MSc research suitability | DR-003 |

Sections not listed (e.g. §10 class weighting, §13 false-negative feasibility, §16 leakage audit, §18–19 practice and University alignment) inform the protocol without section-number citations.

## Running and committing

- Use the project kernel **Python (fraud-detection-review-capacity)** from the repository root's `.venv`; raw files are expected in `data/raw/`.
- **Notebook 01 is a historical evidence record and must not be rerun** while the Test period is sealed: its loader reads the complete target column before restricting analysis to development rows. Its saved outputs are the evidence; corrections are made in new notebooks through the issue workflow, not by re-executing it.
- **Notebook 02** reads headers, key columns and (locally) a raw-row preview with `isFraud` excluded for Test-period rows. **Notebook 03** reads keys for all rows and full columns for development rows only. Both are run by the project owner; neither trains a model.
- Before committing notebook 02, remove the outputs of cells tagged `local-only`:

  ```bash
  .venv/bin/jupyter nbconvert --to notebook --inplace --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_all_outputs_tags local-only notebooks/02_raw_table_schema_inventory.ipynb
  ```

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| DR-NNN | Decision Record NNN in `docs/decisions/` |
| `isFraud` | Target: 1 = fraud, 0 = legitimate |
| Relative day | `TransactionDT` divided by 86,400 seconds, rounded down |
| SHA-256 | Secure Hash Algorithm, 256-bit (file checksum) |
| Test | Relative days 155–182; labels sealed |
| `TransactionDT` | Time offset in seconds from an undisclosed reference point |
