# Data preparation — implementation and evidence record

This document records how stage 6 (Data preparation) of the study's methodology framework ([DR-014](decisions/DR-014-research-methodology-framing.md), Decision Record 14) is carried out: what the implementation does to the IEEE-CIS (IEEE Computational Intelligence Society) data, what it deliberately does not do, how the result is checked, and where the evidence is. It is a record and a reading path. The governing rules are written in full in their owner files and are linked, not restated:

| Content | Owner |
|---|---|
| Partition boundaries and roles; feature policy; leakage controls | [Protocol](protocol.md): [Experimental Split Structure](protocol.md#experimental-split-structure), [Feature Policy](protocol.md#feature-policy), [Leakage Controls](protocol.md#leakage-controls) |
| Why each choice was made | [DR-001](decisions/DR-001-temporal-split-strategy.md) (split), [DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md) (feature declaration and model inputs), [DR-012](decisions/DR-012-neural-network-input-representation.md) (neural-network inputs, proposed) |
| Dataset provenance, manifest and data-access rule | [Dataset](dataset.md) |
| Column roles and meanings | [`config/features.yaml`](../config/features.yaml) and the [feature catalog](feature-catalog.md) |

Evidence tags: **[D]** dataset observation; **[L]** literature, cited by reference identifier (REF-NNN) from [`references.md`](references.md); **[I]** inference or recommendation.

## 1. Scope and status

- **Development data: prepared.** The data-preparation stage (`review-capacity prepare-data`, Issue #62) ran once, on 10 October 2026. It wrote the processed development files (Train (the model-fitting partition), Validation 1 (V1) and Validation 2 (V2); relative days 1–154) and the dataset manifest ([Dataset Manifest](dataset.md#dataset-manifest)). Every later step reads only these files ([Data Access After the Parquet Build](dataset.md#data-access-after-the-parquet-build)).
- **Tree-based input representation: fixed** ([DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md) §5). **Neural-network input representation: proposed** ([DR-012](decisions/DR-012-neural-network-input-representation.md)).
- **Test period (relative days 155–182): sealed.** Development preparation wrote no Test-period file and did not parse Test labels ([Dataset Manifest](dataset.md#dataset-manifest)); the earlier one-off full-data structural check of the labels is disclosed in [Data Provenance](dataset.md#data-provenance). Confirmatory preparation is still to come under the data-access rule: after the protocol freeze, the Test period is prepared once with the same code, reusing the fixed row-local transformations and the Train-fitted representations unchanged. Nothing is fitted on, or selected from, the Test period.

## 2. Integration and temporal organisation

1. **Two source files, one row per transaction.** `train_transaction.csv` (590,540 rows) and `train_identity.csv` (144,233 rows) are read with the declared column types; the target is never read with the features ([`data/loading.py`](../src/review_capacity/data/loading.py)).
2. **Transaction-led left join on `TransactionID` (transaction identifier).** Every transaction is kept, including those without an identity row, whose identity columns are missing. The join is validated as one-to-one; it stops the run if any identity row matches no transaction, or if the join changes the transaction rows or their order. An inner join would have removed every transaction without identity information and changed the study population.
3. **Time and partitions.** `TransactionDT` (transaction time offset in seconds from an undisclosed reference point) is validated, the file is checked to be in non-decreasing time order, and each row's relative day (`TransactionDT` divided by 86,400 seconds, rounded down) assigns it to its partition by the boundaries of [DR-001](decisions/DR-001-temporal-split-strategy.md) ([`data/partitions.py`](../src/review_capacity/data/partitions.py)).
4. **Processed files.** The stage writes the joined development rows (`development_joined.parquet`) and one model-input file per development partition, each with a provenance record ([§7](#7-artifacts-and-reproducibility)).

## 3. Validation and label handling

The validation contract runs before anything is written ([`data/loading.py`](../src/review_capacity/data/loading.py), [`data/validation.py`](../src/review_capacity/data/validation.py)). Each check stops the run with an error; nothing is repaired silently.

| Check | What stops the run |
|---|---|
| Raw-file integrity | A SHA-256 (Secure Hash Algorithm, 256-bit) checksum differs from the value recorded in [`config/dataset.yaml`](../config/dataset.yaml) |
| Schema | A header differs from the declared columns |
| Parsing | A value cannot be parsed as its declared type |
| Keys | A missing or repeated `TransactionID` in either file |
| Join | An unmatched identity row; a change in transaction rows or order |
| Time | A missing, non-numeric, non-finite or non-integer `TransactionDT`; a relative day outside 1–182; a file not in time order |
| Partitions | A partition's rows, `TransactionDT` bounds or identifier-hash prefix differ from the DR-001 anchors; the total differs |
| Categorical values | A non-finite or non-integer numeric-coded categorical value, or one whose absolute magnitude exceeds 2^53 (the implementation's bound for guaranteed exact whole-number representation); a text value equal to the reserved missing label |
| Numeric types | A numeric column not read as 64-bit floating point |

**Labels.** The development labels are read separately from the features, only up to the last development row: the file is in time order, so the reader stops before the first Test row and no Test label is parsed. The stage refuses to start unless the Test partition is declared sealed. The labels must align row by row with the features and be complete and binary (0 or 1). The preparation stage writes label values only to processed development files; the target is never a model input.

**Results [D].** On 10 October 2026 the real files passed every check and reproduced every DR-001 anchor, and on 11 October 2026 the processed files were verified with the raw-file paths disabled ([Dataset Manifest](dataset.md#dataset-manifest); [Data Access After the Parquet Build](dataset.md#data-access-after-the-parquet-build)).

**Limit.** The contract establishes that the data conform to the declared structure and rules. It cannot establish the meaning of the masked columns or whether every organiser-engineered value was available at prediction time ([§8](#8-limitations)).

## 4. Cleaning and population preservation

Preparation performs declared parsing and input-representation conversions; it does not discard transactions or alter numeric model-input values as corrective cleaning.

**Performed**, identically for every partition:

- typed parsing of every column by its declared role;
- missing-value recognition by role: in text categoricals only an empty field is missing, so text such as `"NA"` or `"None"` is kept exactly; numeric columns use the parser's standard missing-value tokens;
- canonical categorical labels: text kept exactly (no case folding, trimming or merging of rare levels), numeric codes written as whole-number text (`100.0` becomes `"100"`), and missing values given the explicit label `"__MISSING__"` ([DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md) §5).

**Not performed:**

- no transaction is discarded as a cleaning operation: no content-based filtering, de-duplication or outlier-row removal; the temporal partition selection is separate ([§2](#2-integration-and-temporal-organisation));
- no corrective alteration of numeric model inputs: no clipping, winsorising, numeric imputation or scaling in the canonical frame or the tree-based inputs;
- no resampling and no feature selection beyond the declared exclusions ([Feature Policy](protocol.md#feature-policy)).

**Neural-network inputs (proposed, [DR-012](decisions/DR-012-neural-network-input-representation.md)).** The neural representation does transform values: numeric inputs are quantile-transformed with Train-fitted quantiles, and missing numeric values are set to 0 after the transformation, with a missingness indicator for each numeric input that has missing Train values. The quantile transformations and the indicator list are fitted on Train only; the zero fill after the transformation is a fixed constant, reused in later partitions.

**Repeated input vectors [D]** (owner-run audit, 13 October 2026; Issue #86; [definitions and evidence](../evidence/duplicate_audit/README.md)). The label-free audit compares exact canonical input values among the 511,998 development rows:

- **all 433 model inputs identical:** 2 repeated vectors on 4 rows (one pair in Train, one in V2), and no vector shared between partitions;
- **the 431 supplied inputs identical, excluding the two derived time-of-day inputs:** 332 repeated vectors on 699 rows (0.14 %), 367 of them beyond the first row of their vector; 110 of these vectors occur in more than one partition.

The audit removed or changed no rows. **[I]** It does not establish why a vector repeats, and no cause is inferred ([reading the counts](../evidence/duplicate_audit/README.md#reading-the-counts)).

**Rule and rationale.** The preservation rule is part of the [Feature Policy](protocol.md#feature-policy); why transactions and supplied values are preserved is recorded in [DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md) §9.

## 5. Feature engineering

- **[L]** The supplied columns are themselves engineered by the data provider: counting features (`C1`–`C14`), time deltas (`D1`–`D15`), match flags (`M1`–`M9`) and 339 engineered features (`V1`–`V339`), whose individual meanings are masked [REF-048, partially verified] ([feature catalog](feature-catalog.md)).
- **The only added inputs** are the two time-of-day inputs `tod_sin` (sine) and `tod_cos` (cosine), computed from each transaction's own `TransactionDT` (its position within the relative day; [Feature Policy](protocol.md#feature-policy); [DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md) §4). They are row-local and label-free and use no fitted statistic. Together with the 431 supplied inputs they give 433 model inputs.
- **Not added:** frequency, count or entity aggregates, target encodings and engineered identifiers ([Feature Policy](protocol.md#feature-policy)). There is no feature-engineering study.

## 6. Model input representation

All models start from one canonical frame of the 433 inputs in declared order ([`features/model_inputs.py`](../src/review_capacity/features/model_inputs.py)):

- **LightGBM:** pandas categoricals whose category lists are fixed from Train rows only, so a category unseen in Train becomes missing; numeric inputs unchanged ([DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md) §5).
- **CatBoost:** the implemented conversion preserves the canonical text labels; the training-order requirement is specified separately in [DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md) §5.
- **ResNet-style multilayer perceptron (MLP) and FT-Transformer (Feature Tokenizer + Transformer):** the representation of [DR-012](decisions/DR-012-neural-network-input-representation.md) (proposed), fitted on Train only.

## 7. Artifacts and reproducibility

| Artifact | Where | Tracked in Git |
|---|---|---|
| Raw files | `data/raw/` | No (local only) |
| Processed development files (`development_joined.parquet`; one model-input file each for Train, V1 and V2) | `data/processed/` | No (row-level derived data; [Data-Use Terms](dataset.md#data-use-terms)) |
| Processed-file provenance records ([Dataset Manifest](dataset.md#dataset-manifest)) | `data/processed/` | No |
| Dataset manifest: raw-file checksums and the label-free partition summary | [`data/datapackage.json`](../data/datapackage.json) | Yes |
| Duplicate-audit counts with their provenance | [`evidence/duplicate_audit/`](../evidence/duplicate_audit/README.md) | Yes |

- **Verified reading.** Before returning any data, the reader checks each processed file against its provenance record, the declared schema and the DR-001 anchors ([`data/processed.py`](../src/review_capacity/data/processed.py)); a file built under another feature declaration is rejected.
- **Built once.** The stage refuses to write over existing processed files.
- **Run by the project owner.** Data preparation and the audit are run by the project owner; the commands, dates and results are recorded in Issues #62 and #86 and in the files above.

## 8. Limitations

- **Organiser-engineered features.** The look-back windows of the counting, time-delta and engineered features are undocumented, so preparation cannot certify that every supplied value was available when its transaction was made ([Leakage Controls](protocol.md#leakage-controls)).
- **No label-report dates.** The dataset does not record when each fraud label became known ([Data Requirements](protocol.md#data-requirements)).
- **Exact matches only.** The duplicate audit finds identical vectors; near-identical rows are outside its scope.

## References

REF-048. Full entries and verification status are in [`references.md`](references.md).

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [D], [I], [L] | Evidence tags: dataset observation; inference or recommendation; literature |
| `"__MISSING__"` | Explicit canonical label for a missing categorical value |
| `C1`–`C14`, `D1`–`D15`, `M1`–`M9`, `V1`–`V339` | Supplied counting, time-delta, match and organiser-engineered feature groups; the `V` column names are distinct from the validation partitions V1 and V2 |
| 2^53 | Magnitude bound that guarantees an exact whole-number representation of numeric-coded categorical values |
| DR-NNN | Decision Record NNN in `docs/decisions/` |
| FT-Transformer | Feature Tokenizer + Transformer (a neural model) |
| IEEE-CIS | IEEE Computational Intelligence Society (the dataset's source) |
| MLP | Multilayer perceptron |
| REF-NNN | Reference identifier in `docs/references.md` |
| SHA-256 | Secure Hash Algorithm, 256-bit |
| `TransactionDT` | Time offset of a transaction, in seconds from an undisclosed reference point |
| `TransactionID` | Transaction identifier, the join key |
| `tod_sin`, `tod_cos` | The two time-of-day inputs: the sine and cosine of the transaction's position within its relative day |
| Test | Sealed confirmatory partition (relative days 155–182) |
| Train | Model-fitting partition (relative days 1–105) |
| V1, V2 | Validation 1 (relative days 106–126) and Validation 2 (relative days 127–154) |
