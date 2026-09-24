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

## Data Structure

The official competition data are split into transaction and identity tables joined by `TransactionID`.

Primary training files used by this research:

- `train_transaction.csv`
- `train_identity.csv`

The Kaggle competition also provides:

- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

For this MSc study, the labelled Kaggle training data will be used to construct the project's own reproducible chronological train, validation, and final test partitions. The original Kaggle competition test set does not contain the `isFraud` labels and therefore cannot serve as the primary labelled holdout for the dissertation's statistical evaluation.

## Key Dataset Characteristics

- Binary target: `isFraud` (1 = fraud, 0 = legitimate)
- Transaction and identity tables joined by `TransactionID`
- Not every transaction has corresponding identity information
- `TransactionDT` is a time delta from an undisclosed reference datetime rather than an absolute timestamp
- The dataset contains substantial class imbalance
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

## Version-Control Policy

Raw and generated datasets must not be committed to GitHub.

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

A dataset manifest will be added before the final experiments are run.

## Research Split Policy

The final MSc experiments must use a reproducible time-aware protocol based on `TransactionDT`.

Conceptually:

```text
earlier observations
    -> training

later observations
    -> validation

latest labelled observations
    -> final test
```

The exact cut points were subsequently frozen in [DR-001](decisions/DR-001-temporal-split-strategy.md): Train relative days 1–105, Validation 1 106–126, Validation 2 127–154 and a sealed final Test period 155–182 (relative day = `TransactionDT` in whole days from the undisclosed reference point). Partition hashes are recorded there.

The final test partition must not be used to select:

- model family;
- hyperparameters;
- class weights;
- probability calibration;
- decision threshold;
- review-capacity policy;
- cost-sensitive policy.

All such choices must be made using training and validation data only.

## Ethics and Data Governance Note

This project uses secondary competition data and does not require recruitment of human participants for the core experimental design.

However, the final ethics position must follow current University of Greenwich guidance and supervisor / Faculty Research Ethics Committee requirements. The project concerns fraud, financial transactions, and automated decision support, so the ethics assessment should explicitly consider:

- the status and anonymisation of the secondary data;
- permitted use under Kaggle competition rules;
- privacy and re-identification risk;
- false-positive and false-negative harms;
- bias and fairness;
- human oversight;
- responsible interpretation of model outputs.

## Citation / Attribution

The dissertation and repository should cite the official IEEE-CIS Fraud Detection competition source and any associated academic or organiser documentation used to describe the dataset.

Local Parquet copies are derivative storage formats and should not be described as the original dataset source.

## Status

**Dataset source documented.**

Next dataset-governance task: create a reproducible dataset manifest with checksums and summary statistics before the frozen experimental split is generated.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| 28 : 1 | Imbalance ratio: legitimate transactions per fraud in development data |
| IEEE-CIS | IEEE Computational Intelligence Society |
| `isFraud` | Binary target: 1 = fraud, 0 = legitimate |
| `TransactionDT` | Time offset from an undisclosed reference point (seconds) |
| `TransactionID` | Key joining the transaction and identity tables |
| Relative day | `TransactionDT` expressed in whole days from the undisclosed reference point |
| Validation 1, Validation 2 | Development partitions: relative days 106–126 and 127–154 |
