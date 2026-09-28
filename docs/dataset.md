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

A column-by-column data dictionary (meanings, observed types, missingness and model-input roles) is in the [feature catalog](feature-catalog.md).

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
- permitted use under Kaggle competition rules (verified: non-commercial academic research and education, Section 7A; no sharing with anyone who has not agreed to the Rules, Section 7B; the stricter project policy is in [Data-Use Terms](#data-use-terms));
- privacy and re-identification risk;
- false-positive and false-negative harms;
- bias and fairness;
- human oversight;
- responsible interpretation of model outputs.

**Current status (Issue #31).** Topic approval was received; a separate explicit ethics-route determination was requested but not provided. The project proceeds under the available University of Greenwich guidance and the data-governance controls above, without claiming formal ethics approval, exemption or an ethics identifier (see [`protocol.md`](protocol.md#ethics)).

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
| Competition Data | The data made available on the competition website, as defined in the Competition Rules, Section 7 |
| IEEE-CIS | IEEE Computational Intelligence Society |
| [I] | Evidence tag: inference or recommendation |
| `isFraud` | Binary target: 1 = fraud, 0 = legitimate |
| `TransactionDT` | Time offset from an undisclosed reference point (seconds) |
| `TransactionID` | Key joining the transaction and identity tables |
| REF-NNN | Reference identifier in `references.md` |
| Relative day | `TransactionDT` expressed in whole days from the undisclosed reference point |
| Validation 1, Validation 2 | Development partitions: relative days 106–126 and 127–154 |
