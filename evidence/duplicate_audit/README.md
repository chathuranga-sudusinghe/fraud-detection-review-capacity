# Content-duplicate audit of the development partitions (Issue #86)

Descriptive evidence on repeated input vectors in the development rows: rows with different `TransactionID` values whose model inputs are identical. It was produced by the `audit-duplicates` command of the `review_capacity` package ([`src/review_capacity/data/duplicates.py`](../../src/review_capacity/data/duplicates.py)), which reads the verified processed development file only: no raw file, no label and no row of the sealed Test period (relative days 155–182) was read. Nothing was removed, changed or flagged.

## Files

| File | Content |
|---|---|
| [`duplicate_audit.json`](duplicate_audit.json) | The counts below with their provenance: the SHA-256 (Secure Hash Algorithm, 256-bit) of the processed input file, of `config/features.yaml` and of `config/dataset.yaml`, the code commit, a fingerprint of the package source, and the Python and package versions |

The file is an unchanged copy of the run's output `data/processed/duplicate_audit.json`, which is not tracked by Git. SHA-256 of `duplicate_audit.json`: `32f59cfa55ad6cc4a9a3fc8a67cd7d2e9f8a2957e6591ee540c5b861d0ca9c69`. The run's log file was kept locally.

## Run

- **Date:** 13 October 2026, run by the project owner on the project workstation.
- **Command:** `review-capacity --log-file evidence/duplicate_audit/duplicate_audit.log audit-duplicates --save`
- **Code:** commit `01f0d8f`, with no uncommitted changes in the working tree.
- **Input:** `development_joined.parquet` (SHA-256 `8c657e5c4094915cf4d1271d52d36fe61e3366489f44174cac845589c0ee8d85`), verified against its provenance record by the reader; 511,998 development rows, read without the target.
- **Duration:** 12.9 seconds for the audit.

## Method

A *repeated input vector* is a set of two or more rows whose values are identical on every compared column. Two definitions are audited on the canonical model inputs of [DR-008](../../docs/decisions/DR-008-feature-declaration-and-model-input-policy.md) (Decision Record 8):

- **primary:** all 433 model inputs, including the two time-of-day inputs `tod_sin` and `tod_cos`;
- **secondary:** the 431 supplied inputs, without the time-of-day pair, which also finds rows that repeat the same values at a different time of the relative day.

`TransactionID`, the raw `TransactionDT`, the target, the relative day and the partition are never compared. Values are compared exactly in their canonical form: categorical inputs as text labels, with missing values as the declared missing label; numeric inputs as 64-bit floating-point numbers, with no rounding or tolerance. Two missing numeric values are equal, and so are `0.0` and `-0.0`.

## Results

**Primary definition (433 inputs)**

| Scope | Rows | Repeated vectors | Rows in them | Excess rows |
|---|---:|---:|---:|---:|
| Train | 368,632 | 1 | 2 | 1 |
| V1 | 65,544 | 0 | 0 | 0 |
| V2 | 77,822 | 1 | 2 | 1 |
| All development rows | 511,998 | 2 | 4 | 2 |

No vector occurs in more than one partition.

**Secondary definition (431 inputs, without the time of day)**

| Scope | Rows | Repeated vectors | Rows in them | Excess rows |
|---|---:|---:|---:|---:|
| Train | 368,632 | 206 | 430 | 224 |
| V1 | 65,544 | 18 | 36 | 18 |
| V2 | 77,822 | 13 | 26 | 13 |
| All development rows | 511,998 | 332 | 699 | 367 |

110 vectors occur in more than one partition, on 238 rows. Vectors shared by each pair of partitions: Train and V1, 52; Train and V2, 39; V1 and V2, 23.

## Reading the counts

- **Rows in them** counts every row of a repeated vector; **excess rows** counts the rows beyond one per vector (rows in them minus repeated vectors).
- The *All development rows* line includes repeats that cross partitions, so it is not the sum of the partition lines.
- The pairwise counts are inclusive: a vector found in all three partitions is counted in each of the three pairs, so the pairs do not add up to the number of vectors in more than one partition.
- **[D]** Exactly repeated records are almost absent (4 of 511,998 development rows), and no complete input vector is shared between Train and a validation partition. Without the time of day, 699 rows (0.14 % of development rows) share their supplied values with another row.
- **[I]** Why a vector repeats (an ingestion duplicate, a retried transaction, card testing or another cause) cannot be established from the data and is not inferred. Near-identical rows, whose values differ slightly, are outside the audit: it finds exact matches only.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [D], [I] | Evidence tags: dataset observation; inference |
| DR-NNN | Decision Record NNN in `docs/decisions/` |
| SHA-256 | Secure Hash Algorithm, 256-bit |
| Test | Sealed confirmatory partition (relative days 155–182) |
| Train | Model-fitting partition (relative days 1–105) |
| V1, V2 | Validation 1 (relative days 106–126) and Validation 2 (relative days 127–154) |
