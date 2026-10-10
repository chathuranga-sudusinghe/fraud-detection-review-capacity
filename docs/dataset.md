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

The competition data are used under the official IEEE-CIS Fraud Detection Competition Rules, Section 7 [REF-049]. The student reports having accepted these Rules on their Kaggle account (Issue #25). This acceptance is recorded as the student's report; it was not independently checked.

### What the Competition Rules require

- **Permitted use (Section 7A).** The Rules state: "You may access and use the Competition Data for non-commercial purposes only, including for participating in the Competition and on Kaggle.com forums, and for academic research and education." This non-commercial MSc study is academic research and education.
- **Data security and sharing (Section 7B).** The Rules state: "You agree to use reasonable and suitable measures to prevent persons who have not formally agreed to these Rules from gaining access to the Competition Data. You agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition." Any unauthorised access must be reported to Kaggle.
- **Winner licence is not a data licence.** The Rules' open-source licence requirement applies only to *winning Submissions and their source code*. It gives no right to share or redistribute the Competition Data.

Section 7B therefore requires keeping the Competition Data from persons who have not formally agreed to the Rules, and from any party not participating in the Competition. It does not say which other people, if any, the data may be shared with. **The Rules do not explicitly address** row-level data derived from the Competition Data, or aggregate statistics computed from it.

### Project policy (stricter than the Rules) [I]

This project adopts the following rules. Where they go beyond the Competition Rules, that is a deliberate, conservative project choice, not a Kaggle requirement:

- **Raw competition files** (`data/raw/`) stay on the student's local machine and are never committed, published or uploaded. **They are not sent to anyone, including supervisors or examiners, even if that person has accepted the Rules.** The project does not rely on any sharing that the Rules might permit. A reader who needs the data obtains it from Kaggle by accepting the Rules themselves.
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

The dataset manifest (file checksums, sizes, row counts and partition hashes) is created together with the split-generation code (protocol design item 8) and recorded in this file.

## Research Split Policy

The chronological partitions and their roles are defined in [DR-001](decisions/DR-001-temporal-split-strategy.md) and the [protocol](protocol.md#experimental-split-structure). The final Test period is never used to select any model, input, policy, capacity or cost setting.

## Ethics and Data Governance Note

The study uses secondary competition data and recruits no human participants. The ethics and governance status, and the considerations the ethics assessment covers, are documented in the [protocol](protocol.md#ethics); the data-handling controls are the [data-use terms](#data-use-terms) above.

## Citation / Attribution

The dissertation and repository should cite the official IEEE-CIS Fraud Detection competition source and any associated academic or organiser documentation used to describe the dataset.

Local Parquet copies are derivative storage formats and should not be described as the original dataset source.

## Status

**Dataset source documented; data understanding complete** (notebooks 01–04).

Next dataset-governance task: create a reproducible dataset manifest with checksums and summary statistics, and split-generation code that reproduces the DR-001 partition hashes (protocol design item 8). The SHA-256 (Secure Hash Algorithm, 256-bit) checksums of both raw files are already recorded in notebooks 01 and 02. Computation findings in the data-understanding notebooks are corrected in development-only notebook 04 ([Issue #59](https://github.com/chathuranga-sudusinghe/fraud-detection-review-capacity/issues/59)).

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| 28 : 1 | Imbalance ratio: legitimate transactions per fraud in development data |
| Competition Data | The data made available on the competition website, as defined in the Competition Rules, Section 7 |
| DR-NNN | Decision Record NNN in `decisions/` |
| IEEE-CIS | IEEE Computational Intelligence Society |
| [I] | Evidence tag: inference or recommendation |
| `isFraud` | Binary target: 1 = fraud, 0 = legitimate |
| `TransactionDT` | Time offset from an undisclosed reference point (seconds) |
| `TransactionID` | Key joining the transaction and identity tables |
| REF-NNN | Reference identifier in `references.md` |
| SHA-256 | Secure Hash Algorithm, 256-bit (file checksum) |
| SMOTE | Synthetic Minority Over-sampling Technique (excluded from the design) |
| Test | Final partition: relative days 155–182, sealed until one confirmatory evaluation |
| Train | Development partition: relative days 1–105 |
| Relative day | `TransactionDT` expressed in whole days from the undisclosed reference point |
| Validation 1, Validation 2 | Development partitions: relative days 106–126 and 127–154 |
