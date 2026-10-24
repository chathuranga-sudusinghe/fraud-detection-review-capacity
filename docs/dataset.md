# Dataset

## Dataset Name

**IEEE-CIS Fraud Detection**

## Official Source

Kaggle competition data page:

https://www.kaggle.com/competitions/ieee-fraud-detection/data

Competition organiser context:

- IEEE Computational Intelligence Society (IEEE-CIS)
- Vesta Corporation transaction-fraud data
- Task: predict the probability that an online transaction is fraudulent
- Target: `isFraud`

## Data Provenance

The IEEE-CIS data are **real-world e-commerce transactions** provided by Vesta Corporation through the Kaggle competition [REF-047]. This project neither generates nor simulates them:

- the data are used as published by the organiser; no rows are added, generated or altered;
- the research questions and hypotheses were designed around the fields the dataset actually provides, and its gaps (no real review capacity, costs or label-report dates) are stated as limitations rather than filled with invented data;
- the temporal split was chosen from transaction volume alone, without using fraud prevalence or feature–target relationships. Before the split was fixed, a one-off full-data check confirmed that `isFraud` existed, contained only binary values and had no missing values, as disclosed in [DR-001](decisions/DR-001-temporal-split-strategy.md).

In this repository, **"simulated"** refers only to the review-capacity and cost **scenarios** (DR-002, DR-007), and **"synthetic"** only to SMOTE (the Synthetic Minority Over-sampling Technique, excluded from the design), small test fixtures, and example inputs for the demonstration interface.

## Data Structure

The official competition data are split into transaction and identity tables joined by `TransactionID`.

Primary training files used by this research:

- `train_transaction.csv`
- `train_identity.csv`

The Kaggle competition also provides:

- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

For this MSc study, the labelled Kaggle training data are used to construct the project's own reproducible chronological train, validation and final test partitions. The original Kaggle competition test set does not contain the `isFraud` labels and therefore cannot serve as the primary labelled holdout for the dissertation's statistical evaluation.

## Key Dataset Characteristics

A column-by-column data dictionary (meanings, observed types, missingness and model-input roles) is in the [feature catalog](feature-catalog.md).

Data-understanding evidence is in the [notebooks](../notebooks/README.md): notebook 01 (feasibility audit), notebook 02 (schema inventory of each raw table) and notebook 03 (merged development dataset and class imbalance).

- Binary target: `isFraud` (1 = fraud, 0 = legitimate), present only in `train_transaction.csv`
- Transaction (394 columns) and identity (41 columns) tables joined by `TransactionID`, their only shared column; the left join keeps all 590,540 transactions and yields 434 columns
- Not every transaction has corresponding identity information: coverage falls from 27.1 % (Train) to 20.7 % (Validation 1) and 17.5 % (Validation 2), and fraud prevalence is much higher where identity information exists (notebook 03)
- `TransactionDT` is a time delta from an undisclosed reference datetime rather than an absolute timestamp
- Fraud is the minority class: the data are imbalanced / low-prevalence (see below)
- Many feature meanings are masked or partially anonymised

These properties are directly relevant to the research questions on class imbalance (low prevalence: ≈ 28 legitimate transactions per fraud, a 28 : 1 ratio, in development data; see [references](references.md) REF-002), temporal evaluation, decision thresholds, false-negative risk, and constrained review capacity.

## Local Research Data

The raw research data are stored locally under:

```text
data/raw/
```

Current local source files include:

```text
train_transaction.csv
train_identity.csv
```

Local Parquet copies may be produced for efficient experimental processing:

```text
train_transaction.parquet
train_identity.parquet
```

Parquet files are derivative storage formats only; they are not a separate dataset source.

## Data-Use Terms

The competition data are used under the official IEEE-CIS Fraud Detection Competition Rules, Section 7 [REF-049]. The Rules were accepted on a Kaggle account (Issue #25). This acceptance is recorded as a report; it was not independently checked.

### What the Competition Rules require

- **Permitted use (Section 7A).** The Rules state: "You may access and use the Competition Data for non-commercial purposes only, including for participating in the Competition and on Kaggle.com forums, and for academic research and education." This non-commercial MSc study is academic research and education.
- **Data security and sharing (Section 7B).** The Rules state: "You agree to use reasonable and suitable measures to prevent persons who have not formally agreed to these Rules from gaining access to the Competition Data. You agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition." Any unauthorised access must be reported to Kaggle.
- **Winner licence is not a data licence.** The Rules' open-source licence requirement applies only to *winning Submissions and their source code*. It gives no right to share or redistribute the Competition Data.

Section 7B therefore requires keeping the Competition Data from persons who have not formally agreed to the Rules, and from any party not participating in the Competition. It does not say which other people, if any, the data may be shared with. **The Rules do not explicitly address** row-level data derived from the Competition Data, or aggregate statistics computed from it.

### Project policy (stricter than the Rules) [I]

This project adopts the following rules. Where they go beyond the Competition Rules, that is a deliberate, conservative project choice, not a Kaggle requirement:

- **Raw competition files** (`data/raw/`) stay on the local machine and are never committed, published or uploaded. **They are not sent to anyone, including supervisors or examiners, even if that person has accepted the Rules.** The project does not rely on any sharing that the Rules might permit. A reader who needs the data obtains it from Kaggle by accepting the Rules themselves.
- **Row-level derived data** (e.g. Parquet copies, processed splits, per-transaction scores or predictions, feature extracts) is handled exactly like raw data. It can still reproduce Competition Data row by row, so the project treats it as covered by Section 7B. This is the project's interpretation.
- **Aggregate results** (counts, rates, metrics, figures), together with code, documentation, column names, schema information and file checksums, may be kept in the repository and published. The Rules neither permit nor forbid aggregate statistics explicitly; publishing them is standard research practice and is the project's interpretation, not an explicit permission.

## Version-Control Policy

Raw and generated datasets must not be committed to GitHub. This is a project policy that supports compliance with Section 7B of the Competition Rules ([Data-Use Terms](#data-use-terms)).

The repository `.gitignore` keeps only the directory structure while excluding raw, interim, processed, and external datasets.

Research reproducibility should instead be supported through lightweight metadata such as:

- dataset source URL;
- file names;
- file sizes;
- cryptographic checksums;
- row and column counts;
- target prevalence;
- split-generation rules;
- generated split hashes where practical.

The label-free file metadata and split information are recorded in the [dataset manifest](#dataset-manifest). Development-period target prevalence is documented in the [data-understanding notebooks](../notebooks/README.md).

## Dataset Manifest

The dataset manifest resolves protocol design item 8 (Issue #62). It is generated by the data-preparation code and committed as [`data/datapackage.json`](../data/datapackage.json), a descriptor in the Frictionless Data Package format, version 2 [REF-106]. The standard requires this file name at the top level of the data directory; project-specific properties carry an underscore prefix, as the standard's private-properties convention recommends.

**What the manifest records.** It contains metadata only: no data row and no label statistic.

| Raw file | Size (bytes) | Rows | Columns |
|---|---|---|---|
| `train_transaction.csv` | 683,351,067 | 590,540 | 394 |
| `train_identity.csv` | 26,529,680 | 144,233 | 41 |

- Each file's SHA-256 (Secure Hash Algorithm, 256-bit) checksum, written in the standard's `sha256:` form.
- Under `_split`, a label-free summary of each partition of [DR-001](decisions/DR-001-temporal-split-strategy.md): row count, `TransactionDT` bounds and the full SHA-256 of the partition's sorted `TransactionID`s. The first 16 characters of each hash are the anchors recorded in DR-001.

**How it is produced and checked.** The data-preparation stage runs on the local raw files:

```text
review-capacity prepare-data               # validate, then write the processed files and the manifest
review-capacity prepare-data --check-only  # validate and print the summary; write nothing
```

- The machine-readable declaration of the raw files, the partitions and their anchors, and of the label-latency fitting range of V2, is [`config/dataset.yaml`](../config/dataset.yaml). Column roles come from [`config/features.yaml`](../config/features.yaml).
- The stage stops with an error unless both file checksums match, every partition reproduces its DR-001 row count, `TransactionDT` bounds and hash prefix, and the data validation contract passes (DR-008 §§4–5). On 10 October 2026 the real files reproduced every anchor, and the integration tests passed.
- Test-period labels are never read: labels are parsed only up to the last row of the development partitions, and the stage refuses to start unless the Test partition is declared sealed.

**Processed files.** The stage also writes the development partitions to `data/processed/`: the joined development rows and one model-input file each for Train, Validation 1 and Validation 2, in Parquet format. Each file has a provenance record (`*.provenance.json`) with its SHA-256, row and column counts, the raw-file and configuration checksums, the Git commit, a fingerprint of the code that produced it and the software versions. These files are row-level derived data: they stay local and are never committed ([Data-Use Terms](#data-use-terms)). No Test-period file is written before the protocol freeze.

## Data Access After the Parquet Build

**Rule (decision of 11 October 2026; Issue #70).** The raw files are read only to build the processed development files. That build ran once, on 10 October 2026 ([Dataset Manifest](#dataset-manifest)). From then until the confirmatory Test evaluation:

- every step, including data profiling, the integration tests, training, tuning, calibration, threshold selection and every Validation 1 and Validation 2 analysis, reads the development data only from `data/processed/`, through the verified reader of the `review_capacity` package;
- no step reads `data/raw/`; this includes `prepare-data --check-only`, which validates the raw files;
- if the stored values change before the protocol freeze (the column roles, the derived inputs or the canonical categorical form in `config/features.yaml`), the partition files are rebuilt from the joined development file `development_joined.parquet`, never from the raw files; a change that cannot alter a stored value (a library-handling block, a prose reason or a comment) needs no rebuild;
- the raw files are read once more, by the confirmatory Test evaluation after the protocol freeze, which prepares the whole Test period (relative days 155–182) with the same preparation code as the development partitions.

No separate copy of the Test rows is made: they stay only in the raw files, whose Test-period labels the preparation code does not parse.

**How the code enforces the rule.**

- **Verified reader.** Before it returns any data, the reader checks each processed file against its provenance record: SHA-256, rows, columns, the raw-file checksums declared in `config/dataset.yaml`, the fingerprint of the feature declaration (a SHA-256 over the column roles the build uses, taken from the loaded `config/features.yaml`: the key, time and target names, the excluded, categorical, numeric-coded, numeric and derived inputs and the missing label; a file built under another declaration is rejected as stale, a change that cannot alter a stored value, such as a comment, a library-handling block or a prose reason, does not invalidate it, and a record without a fingerprint is rejected until it is migrated), the declared columns and their storage types, and the [DR-001](decisions/DR-001-temporal-split-strategy.md) anchors recomputed from the file's identifier and time columns before any label is read.
- **One raw-data module.** Only the module that reads the raw files can open them, and only the data-preparation stage and its validation contract use it; an automated test enforces this boundary.
- **Built once.** The data-preparation stage refuses to write over processed files that already exist.
- **Development labels only.** Labels are parsed only up to the last development row ([Dataset Manifest](#dataset-manifest)).

**Provenance migration (Issue #125).** The records written by the build of 10 October 2026 carry only the whole-file SHA-256 of `config/features.yaml`, so any edit of that file, even one that cannot change a stored value, would have made every processed file stale. `review-capacity migrate-provenance` verifies each processed file with all the checks of the verified reader's file verification (SHA-256, rows, columns, storage types, raw-source hashes and the DR-001 anchors; for a record that has no fingerprint yet, the recorded whole-file hash must also still equal the current file's, so the migration runs with the declaration the files were built with, before any later edit of it) and writes its record again with the fingerprint, the SHA-256 of the previous record and the code identity of the migration. The data files are not rewritten, so their SHA-256 values and the DR-001 anchors are unchanged by construction; no raw file is read. All files are verified before any record is written, and `--check-only` verifies without writing.

**Verification.** **[D]** On 11 October 2026 the integration tests ran on the processed files only, with the raw-file paths disabled. The DR-001 anchors were reproduced from the files' content, and the stored Train, Validation 1 and Validation 2 files were reproduced value for value from `development_joined.parquet` by the current code.

## Research Split Policy

The chronological partitions and their roles are defined in [DR-001](decisions/DR-001-temporal-split-strategy.md) and the [protocol](protocol.md#experimental-split-structure). The final Test period is never used to select any model, input, policy, capacity or cost setting.

## Ethics and Data Governance Note

The study uses secondary competition data and recruits no human participants. The ethics and governance status, and the considerations the ethics assessment covers, are documented in the [protocol](protocol.md#ethics); the data-handling controls are the [data-use terms](#data-use-terms) above.

## Citation / Attribution

The dissertation and repository should cite the official IEEE-CIS Fraud Detection competition source and any associated academic or organiser documentation used to describe the dataset.

Local Parquet copies are derivative storage formats and should not be described as the original dataset source.

## Status

**Dataset source documented; data understanding complete** (notebooks 01–04); **dataset manifest and data preparation complete** (Issue #62; [Dataset Manifest](#dataset-manifest)); **raw data read again only by the confirmatory Test evaluation** (Issue #70; [Data Access After the Parquet Build](#data-access-after-the-parquet-build)).

Computation findings in the data-understanding notebooks are corrected in development-only notebook 04 ([Issue #59](https://github.com/chathuranga-sudusinghe/fraud-detection-review-capacity/issues/59)).

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| 28 : 1 | Imbalance ratio: legitimate transactions per fraud in development data |
| Competition Data | The data made available on the competition website, as defined in the Competition Rules, Section 7 |
| DR-NNN | Decision Record NNN in `decisions/` |
| Frictionless Data Package | Open standard for describing a dataset in a `datapackage.json` descriptor [REF-106] |
| Parquet | Column-oriented file format used for the local processed files |
| Provenance record | `*.provenance.json` file describing how a processed file was produced |
| IEEE-CIS | IEEE Computational Intelligence Society |
| [D] | Evidence tag: observed dataset evidence |
| [I] | Evidence tag: inference or recommendation |
| `isFraud` | Binary target: 1 = fraud, 0 = legitimate |
| `TransactionDT` | Time offset from an undisclosed reference point (seconds) |
| `TransactionID` | Key joining the transaction and identity tables |
| REF-NNN | Reference identifier in `references.md` |
| SHA-256 | Secure Hash Algorithm, 256-bit (file checksum) |
| SMOTE | Synthetic Minority Over-sampling Technique (excluded from the design) |
| Test | Final partition: relative days 155–182, sealed until one confirmatory evaluation |
| Relative day | `TransactionDT` divided by 86,400 seconds and rounded down |
| Train | Development partition: relative days 1–105 |
| Validation 1, Validation 2 | Development partitions: relative days 106–126 and 127–154 |
