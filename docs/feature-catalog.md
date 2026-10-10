# Feature Catalog — IEEE-CIS Fraud Detection

Human-readable data dictionary for every column of the joined IEEE-CIS (IEEE Computational Intelligence Society) Fraud Detection training data used by this research. It is the reference for dataset inspection and later model implementation.

- **Model-input roles** are fixed by [DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md) (Decision Record 8) and declared machine-readably in [`config/features.yaml`](../config/features.yaml). This catalog explains those roles. It does not change them, and every role below is read from that file.
- **Dataset source and governance:** [`dataset.md`](dataset.md). **Temporal split:** [DR-001](decisions/DR-001-temporal-split-strategy.md).

## How to read this catalog

### Evidence and Test sealing

- **Column names** come from the raw-file headers of `train_transaction.csv` (394 columns) and `train_identity.csv` (41 columns). After the left join on `TransactionID` there are 434 columns.
- **How the files were read.** The transaction file was read with every column except `isFraud` (so Test-period feature rows were loaded into memory), and the identity file in full. Both were filtered to development rows **before any statistic was computed**.
- **All statistics are computed on development rows only**, i.e. relative days 1–154:
  - Train = days 1–105 (368,632 rows);
  - Validation 1 = days 106–126;
  - Validation 2 = days 127–154 (77,822 rows);
  - 511,998 development rows in total.
- A relative day is `TransactionDT` counted in whole days (86,400 seconds) from the dataset's undisclosed reference point.
- **The Test period (days 155–182) is sealed.** No Test statistic appears here.
- **The target `isFraud` was not loaded** for this catalog. Its description comes from the organiser and the feasibility audit.
- Statistics were produced by a development-only profiling script (not committed) and agree with `notebooks/01_dataset_feasibility_audit.ipynb` §§2.3, 2.7, 8.
- The column inventory, table membership and roles are reproduced from the raw headers in [notebook 02](../notebooks/02_raw_table_schema_inventory.ipynb). Group-level missingness of the merged development data is reproduced in [notebook 03](../notebooks/03_merged_dataset_inspection.ipynb) §6.

### Sources of meaning

- **Official meaning** is taken only from the organiser's documentation:
  - the Kaggle data description [REF-047] states the target, the join key, the `TransactionDT` definition and the lists of categorical columns;
  - group-level descriptions of the other fields (card, address, distance, email, count, time-delta, match, engineered and identity groups) come from the organiser's forum post [REF-048]. **REF-048 is only partially verified:** its text was seen through search-result extracts and the feasibility notebook §8.1, not read in full.
- The organiser states that the identity field names are masked and that a pairwise dictionary is not provided. **Where the organiser gives no meaning, the catalog says "Masked" and does not guess.**
- "Observed range / most frequent values" is what the data contain [D]. It is not a claim about meaning.

### Column legend

| Field | Values |
|---|---|
| Interpretation | `identifier`, `temporal`, `target`, `categorical` (with "numeric-coded" when stored as numbers), `numeric`, `ambiguous` |
| Model input | `include` (one of the 431 inputs given identically to all four models of DR-011) or `exclude` |
| Missing (dev) | Share of development rows with no value |
| Levels in Train | Number of distinct non-missing values in Train rows |
| Validation 2 values unseen in Train | Share of non-missing Validation 2 values never observed in Train (categorical columns only) |

### Summary of roles

| Role | Columns | Count |
|---|---|---|
| Target | `isFraud` | 1 |
| Excluded (identifier / raw time) | `TransactionID`, `TransactionDT` | 2 |
| Categorical input | `ProductCD`, `card1`–`card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `M1`–`M9`, `DeviceType`, `DeviceInfo`, `id_12`–`id_38` | 49 |
| Numeric input | `TransactionAmt`, `dist1`, `dist2`, `C1`–`C14`, `D1`–`D15`, `V1`–`V339`, `id_01`–`id_11` | 382 |
| **Total** | | **434** |

**Encoding reminder (DR-008).**
- Categorical inputs are converted to text labels, and missing categorical values become the label `"__MISSING__"`.
- Numeric inputs stay `float64`, with missing values left as NaN (not-a-number): handled natively by the tree-based libraries, and represented by a Train-fitted quantile transformation with missingness indicators for the neural networks (proposed, [DR-012](decisions/DR-012-neural-network-input-representation.md)).
- The canonical frame and the tree-based models receive no imputation, scaling or grouping. No engineered feature is added for any model.

## 1. Key, target and time

| Column | Table | Official meaning | Observed dtype | Observed range / values | Interpretation | Model input | Reason for treatment | Missing (dev) | Caution |
|---|---|---|---|---|---|---|---|---|---|
| `TransactionID` | both (join key) | Transaction identifier joining the transaction and identity tables [REF-047] | int64 | 2,987,000 to 3,498,997; unique per row | identifier | **exclude** | Row identifier; increases with time (Spearman rank correlation 1.000 with `TransactionDT`), so it is a disguised timestamp | 0.0 % | Join key only |
| `isFraud` | transaction | Binary target: whether the transaction is fraudulent [REF-047] | binary 0/1 (not loaded for this catalog) | 1 = fraud, 0 = legitimate; complete in development data (notebook §5) | target | **exclude** (target) | Never a model input | 0.0 % (notebook §5) | Labels are chargeback-derived; linked later transactions are also labelled fraud [REF-048, partially verified] |
| `TransactionDT` | transaction | "A timedelta from a given reference datetime (not an actual timestamp)" [REF-047]; in seconds | int64 | 86,400 to 13,391,998 (development rows) | temporal | **exclude** | Raw time. Trees cannot extrapolate a monotone time value to a later period. Used only to derive relative days and the split | 0.0 % | No calendar dates can be inferred |

## 2. Transaction amount and product

| Column | Table | Official meaning | Observed dtype | Observed range / most frequent values | Interpretation | Model input | Reason for treatment | Missing (dev) | Levels in Train | Caution |
|---|---|---|---|---|---|---|---|---|---|---|
| `TransactionAmt` | transaction | Transaction payment amount in United States dollars (USD) [REF-048, partially verified] | float64 | 0.251 to 31,937.391 | numeric | include | Continuous amount; organiser does not list it as categorical | 0.0 % | 14,689 | 10.5 % of amounts have more than two decimal places (notebook §2.7) |
| `ProductCD` | transaction | Product code; codes undisclosed [REF-048, partially verified]. Listed as categorical by the organiser [REF-047] | text | `W`, `C`, `R`, `H`, `S` | categorical | include | Organiser-declared categorical | 0.0 % | 5 | — |

## 3. Card fields (`card1`–`card6`)

Group description: payment-card information, such as card type, card category, issuing bank and country [REF-048, partially verified]. **Which field holds which attribute is masked.**

| Column | Table | Official meaning | Observed dtype | Observed range / most frequent values | Interpretation | Model input | Reason for treatment | Missing (dev) | Levels in Train | Caution |
|---|---|---|---|---|---|---|---|---|---|---|
| `card1` | transaction | Masked: individual meaning not disclosed (card group). Listed as categorical by the organiser [REF-047] | int64 | 1,000 to 18,396 | categorical (numeric-coded) | include | Organiser-declared categorical; integer codes treated as unordered labels | 0.0 % | 11,870 | Very high cardinality; 27 % of Train rows are in levels with fewer than 100 rows; can support entity memorisation (DR-008); Validation 2 values unseen in Train: 1.3 % |
| `card2` | transaction | Masked: individual meaning not disclosed (card group). Listed as categorical by the organiser [REF-047] | float64 | 100 to 600 | categorical (numeric-coded) | include | Organiser-declared categorical; integer codes treated as unordered labels | 1.4 % | 499 | — |
| `card3` | transaction | Masked: individual meaning not disclosed (card group). Listed as categorical by the organiser [REF-047] | float64 | 100 to 231 | categorical (numeric-coded) | include | Organiser-declared categorical; integer codes treated as unordered labels | 0.2 % | 103 | — |
| `card4` | transaction | Masked: individual meaning not disclosed (card group). Listed as categorical by the organiser [REF-047] | text | `visa`, `mastercard`, `american express`, `discover` | categorical | include | Organiser-declared categorical | 0.2 % | 4 | — |
| `card5` | transaction | Masked: individual meaning not disclosed (card group). Listed as categorical by the organiser [REF-047] | float64 | 100 to 237 | categorical (numeric-coded) | include | Organiser-declared categorical; integer codes treated as unordered labels | 0.6 % | 109 | — |
| `card6` | transaction | Masked: individual meaning not disclosed (card group). Listed as categorical by the organiser [REF-047] | text | `debit`, `credit`, `debit or credit`, `charge card` | categorical | include | Organiser-declared categorical | 0.2 % | 4 | — |

## 4. Address and distance (`addr1`, `addr2`, `dist1`, `dist2`)

Group description: `addr` fields relate to the purchaser's billing region and country, and `dist` fields are distances [REF-048, partially verified]. **The distance reference points are masked.**

| Column | Table | Official meaning | Observed dtype | Observed range / most frequent values | Interpretation | Model input | Reason for treatment | Missing (dev) | Levels in Train | Caution |
|---|---|---|---|---|---|---|---|---|---|---|
| `addr1` | transaction | Address field, billing region per organiser group description; codes masked. Listed as categorical by the organiser [REF-047] | float64 | 100 to 540 | categorical (numeric-coded) | include | Organiser-declared categorical; integer codes treated as unordered labels | 11.2 % | 310 | — |
| `addr2` | transaction | Address field, billing country per organiser group description; codes masked. Listed as categorical by the organiser [REF-047] | float64 | 10 to 102 | categorical (numeric-coded) | include | Organiser-declared categorical; integer codes treated as unordered labels | 11.2 % | 65 | — |
| `dist1` | transaction | Distance; reference points masked | float64 | 0 to 10,286 | numeric | include | Numeric distance; not organiser-categorical | 60.3 % | 2,549 | Sparse |
| `dist2` | transaction | Distance; reference points masked | float64 | 0 to 11,623 | numeric | include | Numeric distance; not organiser-categorical | 93.4 % | 1,552 | Very sparse |

## 5. Email domains

| Column | Table | Official meaning | Observed dtype | Observed range / most frequent values | Interpretation | Model input | Reason for treatment | Missing (dev) | Levels in Train | Caution |
|---|---|---|---|---|---|---|---|---|---|---|
| `P_emaildomain` | transaction | Purchaser email domain [REF-048, partially verified]. Listed as categorical by the organiser [REF-047] | text | `gmail.com`, `yahoo.com`, `hotmail.com`, `anonymous.com`, `aol.com` | categorical | include | Organiser-declared categorical; text domain names kept exactly | 15.7 % | 59 | — |
| `R_emaildomain` | transaction | Recipient email domain [REF-048, partially verified]. Listed as categorical by the organiser [REF-047] | text | `gmail.com`, `hotmail.com`, `anonymous.com`, `yahoo.com`, `aol.com` | categorical | include | Organiser-declared categorical; text domain names kept exactly | 76.5 % | 60 | Mostly missing |

## 6. Counting features (`C1`–`C14`)

Group description: counting features, e.g. how many addresses are associated with the payment card; **the actual meaning of each count is masked** [REF-048, partially verified]. The look-back window of the counts is undocumented, which is a known threat to validity (protocol, Leakage Controls).

| Column | Table | Official meaning | Observed dtype | Observed range / most frequent values | Interpretation | Model input | Reason for treatment | Missing (dev) | Levels in Train | Caution |
|---|---|---|---|---|---|---|---|---|---|---|
| `C1` | transaction | Masked: individual meaning not disclosed (count group) | float64 | 0 to 4,685 | numeric | include | Integer-valued counts; numeric | 0.0 % | 1,539 | — |
| `C2` | transaction | Masked: individual meaning not disclosed (count group) | float64 | 0 to 5,691 | numeric | include | Integer-valued counts; numeric | 0.0 % | 1,000 | — |
| `C3` | transaction | Masked: individual meaning not disclosed (count group) | float64 | 0 to 25 | numeric | include | Integer-valued counts; numeric | 0.0 % | 21 | — |
| `C4` | transaction | Masked: individual meaning not disclosed (count group) | float64 | 0 to 2,253 | numeric | include | Integer-valued counts; numeric | 0.0 % | 1,196 | — |
| `C5` | transaction | Masked: individual meaning not disclosed (count group) | float64 | 0 to 349 | numeric | include | Integer-valued counts; numeric | 0.0 % | 303 | — |
| `C6` | transaction | Masked: individual meaning not disclosed (count group) | float64 | 0 to 2,253 | numeric | include | Integer-valued counts; numeric | 0.0 % | 1,278 | — |
| `C7` | transaction | Masked: individual meaning not disclosed (count group) | float64 | 0 to 2,255 | numeric | include | Integer-valued counts; numeric | 0.0 % | 1,087 | — |
| `C8` | transaction | Masked: individual meaning not disclosed (count group) | float64 | 0 to 3,331 | numeric | include | Integer-valued counts; numeric | 0.0 % | 1,198 | — |
| `C9` | transaction | Masked: individual meaning not disclosed (count group) | float64 | 0 to 210 | numeric | include | Integer-valued counts; numeric | 0.0 % | 203 | — |
| `C10` | transaction | Masked: individual meaning not disclosed (count group) | float64 | 0 to 3,257 | numeric | include | Integer-valued counts; numeric | 0.0 % | 1,114 | — |
| `C11` | transaction | Masked: individual meaning not disclosed (count group) | float64 | 0 to 3,188 | numeric | include | Integer-valued counts; numeric | 0.0 % | 1,389 | — |
| `C12` | transaction | Masked: individual meaning not disclosed (count group) | float64 | 0 to 3,188 | numeric | include | Integer-valued counts; numeric | 0.0 % | 1,172 | — |
| `C13` | transaction | Masked: individual meaning not disclosed (count group) | float64 | 0 to 2,918 | numeric | include | Integer-valued counts; numeric | 0.0 % | 1,581 | — |
| `C14` | transaction | Masked: individual meaning not disclosed (count group) | float64 | 0 to 1,429 | numeric | include | Integer-valued counts; numeric | 0.0 % | 1,079 | — |

## 7. Time-delta features (`D1`–`D15`)

Group description: time deltas, e.g. days between a transaction and a previous event; **the reference events are masked** [REF-048, partially verified].

| Column | Table | Official meaning | Observed dtype | Observed range / most frequent values | Interpretation | Model input | Reason for treatment | Missing (dev) | Levels in Train | Caution |
|---|---|---|---|---|---|---|---|---|---|---|
| `D1` | transaction | Masked: individual meaning not disclosed (time-delta group) | float64 | 0 to 640 | numeric | include | Numeric time delta; negative values kept as observed | 0.2 % | 637 | — |
| `D2` | transaction | Masked: individual meaning not disclosed (time-delta group) | float64 | 0 to 640 | numeric | include | Numeric time delta; negative values kept as observed | 48.4 % | 637 | — |
| `D3` | transaction | Masked: individual meaning not disclosed (time-delta group) | float64 | 0 to 749 | numeric | include | Numeric time delta; negative values kept as observed | 45.5 % | 586 | — |
| `D4` | transaction | Masked: individual meaning not disclosed (time-delta group) | float64 | -122 to 835 | numeric | include | Numeric time delta; negative values kept as observed | 29.2 % | 722 | 15 negative values |
| `D5` | transaction | Masked: individual meaning not disclosed (time-delta group) | float64 | 0 to 819 | numeric | include | Numeric time delta; negative values kept as observed | 53.2 % | 617 | — |
| `D6` | transaction | Masked: individual meaning not disclosed (time-delta group) | float64 | -83 to 847 | numeric | include | Numeric time delta; negative values kept as observed | 87.8 % | 743 | 3 negative values |
| `D7` | transaction | Masked: individual meaning not disclosed (time-delta group) | float64 | 0 to 801 | numeric | include | Numeric time delta; negative values kept as observed | 93.7 % | 527 | — |
| `D8` | transaction | Masked: individual meaning not disclosed (time-delta group) | float64 | 0 to 1,707.792 | numeric | include | Numeric time delta; negative values kept as observed | 87.3 % | 10,478 | Non-integer values |
| `D9` | transaction | Masked: individual meaning not disclosed (time-delta group) | float64 | 0 to 0.958 | numeric | include | Numeric time delta; negative values kept as observed | 87.3 % | 24 | Equals (hour of the relative day) ÷ 24 on 94.9 % of non-missing development rows [D]: a cyclic time-of-day value, kept because it is not a monotone timestamp (DR-008). This interpretation is inferred, not documented |
| `D10` | transaction | Masked: individual meaning not disclosed (time-delta group) | float64 | 0 to 847 | numeric | include | Numeric time delta; negative values kept as observed | 13.5 % | 738 | — |
| `D11` | transaction | Masked: individual meaning not disclosed (time-delta group) | float64 | -53 to 644 | numeric | include | Numeric time delta; negative values kept as observed | 50.1 % | 599 | 7 negative values |
| `D12` | transaction | Masked: individual meaning not disclosed (time-delta group) | float64 | -83 to 634 | numeric | include | Numeric time delta; negative values kept as observed | 89.0 % | 565 | 2 negative values |
| `D13` | transaction | Masked: individual meaning not disclosed (time-delta group) | float64 | 0 to 818 | numeric | include | Numeric time delta; negative values kept as observed | 89.8 % | 495 | — |
| `D14` | transaction | Masked: individual meaning not disclosed (time-delta group) | float64 | -193 to 844 | numeric | include | Numeric time delta; negative values kept as observed | 89.6 % | 729 | 3 negative values |
| `D15` | transaction | Masked: individual meaning not disclosed (time-delta group) | float64 | -83 to 851 | numeric | include | Numeric time delta; negative values kept as observed | 15.8 % | 778 | 15 negative values |

## 8. Match flags (`M1`–`M9`)

Group description: match flags, e.g. whether names on the card and address match; **which comparison each flag records is masked** [REF-048, partially verified].

| Column | Table | Official meaning | Observed dtype | Observed range / most frequent values | Interpretation | Model input | Reason for treatment | Missing (dev) | Levels in Train | Caution |
|---|---|---|---|---|---|---|---|---|---|---|
| `M1` | transaction | Masked: individual meaning not disclosed (match group). Listed as categorical by the organiser [REF-047] | text | `T`, `F` | categorical | include | Organiser-declared categorical; T/F labels | 48.5 % | 2 | — |
| `M2` | transaction | Masked: individual meaning not disclosed (match group). Listed as categorical by the organiser [REF-047] | text | `T`, `F` | categorical | include | Organiser-declared categorical; T/F labels | 48.5 % | 2 | — |
| `M3` | transaction | Masked: individual meaning not disclosed (match group). Listed as categorical by the organiser [REF-047] | text | `T`, `F` | categorical | include | Organiser-declared categorical; T/F labels | 48.5 % | 2 | — |
| `M4` | transaction | Masked: individual meaning not disclosed (match group). Listed as categorical by the organiser [REF-047] | text | `M0`, `M2`, `M1` | categorical | include | Organiser-declared categorical; T/F / M0–M2 labels | 48.0 % | 3 | — |
| `M5` | transaction | Masked: individual meaning not disclosed (match group). Listed as categorical by the organiser [REF-047] | text | `F`, `T` | categorical | include | Organiser-declared categorical; T/F labels | 59.7 % | 2 | — |
| `M6` | transaction | Masked: individual meaning not disclosed (match group). Listed as categorical by the organiser [REF-047] | text | `F`, `T` | categorical | include | Organiser-declared categorical; T/F labels | 29.2 % | 2 | — |
| `M7` | transaction | Masked: individual meaning not disclosed (match group). Listed as categorical by the organiser [REF-047] | text | `F`, `T` | categorical | include | Organiser-declared categorical; T/F labels | 61.6 % | 2 | — |
| `M8` | transaction | Masked: individual meaning not disclosed (match group). Listed as categorical by the organiser [REF-047] | text | `F`, `T` | categorical | include | Organiser-declared categorical; T/F labels | 61.6 % | 2 | — |
| `M9` | transaction | Masked: individual meaning not disclosed (match group). Listed as categorical by the organiser [REF-047] | text | `T`, `F` | categorical | include | Organiser-declared categorical; T/F labels | 61.6 % | 2 | — |

## 9. Organiser-engineered features (`V1`–`V339`)

Group description: "Vesta engineered rich features", including ranking, counting and other entity relations [REF-048, partially verified]. **Every individual `V` column is masked**, so the catalog documents the family as a whole, plus the per-column statistics in [Appendix A](#appendix-a--v1v339-per-column-statistics).

*Notation note:* `V1`–`V339` are feature **columns**. They are unrelated to the Validation 1 and Validation 2 **periods**, which this catalog always writes in full.

| Property | Value |
|---|---|
| Table | transaction |
| Official meaning | Masked; organiser-engineered (Vesta) |
| Observed dtype | float64 (all 339) |
| Interpretation | numeric |
| Model input | include (all 339) |
| Reason for treatment | Organiser does not list them as categorical; stored as numbers |
| Missing (dev) | 0.0 % to 85.7 % per column |
| Levels in Train | 1 to 25,530 per column; 7 columns are two-valued; 263 are integer-valued in development data |
| Structure | 14 blocks of columns with the same number of missing Train rows (table below); whether each block also shares an identical missing-row pattern is checked in [Issue #59](https://github.com/chathuranga-sudusinghe/fraud-detection-review-capacity/issues/59) |
| Caution | Construction undocumented, so potential look-back issues cannot be checked; heavy redundancy; uninterpretable, so not used for false-negative slicing |

**Missingness blocks (Train).** Columns in a block are missing on exactly the same number of Train rows.

| Block | Columns | Missing in Train |
|---|---|---|
| 1 | `V1`–`V11` (11 columns) | 57.5 % |
| 2 | `V12`–`V34` (23 columns) | 15.8 % |
| 3 | `V35`–`V52` (18 columns) | 31.1 % |
| 4 | `V53`–`V74` (22 columns) | 16.1 % |
| 5 | `V75`–`V94` (20 columns) | 17.8 % |
| 6 | `V95`–`V137` (43 columns) | 0.0 % |
| 7 | `V138`–`V166` (29 columns) | 83.2 % |
| 8 | `V167`–`V168`, `V172`–`V173`, `V176`–`V179`, `V181`–`V183`, `V186`–`V187`, `V190`–`V193`, `V196`, `V199`, `V202`–`V207`, `V211`–`V216` (31 columns) | 73.8 % |
| 9 | `V169`–`V171`, `V174`–`V175`, `V180`, `V184`–`V185`, `V188`–`V189`, `V194`–`V195`, `V197`–`V198`, `V200`–`V201`, `V208`–`V210` (19 columns) | 73.8 % |
| 10 | `V217`–`V219`, `V223`–`V226`, `V228`–`V233`, `V235`–`V237`, `V240`–`V244`, `V246`–`V249`, `V252`–`V254`, `V257`–`V258`, `V260`–`V269`, `V273`–`V278` (46 columns) | 75.1 % |
| 11 | `V220`–`V222`, `V227`, `V234`, `V238`–`V239`, `V245`, `V250`–`V251`, `V255`–`V256`, `V259`, `V270`–`V272` (16 columns) | 73.3 % |
| 12 | `V279`–`V280`, `V284`–`V287`, `V290`–`V295`, `V297`–`V299`, `V302`–`V312`, `V316`–`V321` (32 columns) | 0.0 % |
| 13 | `V281`–`V283`, `V288`–`V289`, `V296`, `V300`–`V301`, `V313`–`V315` (11 columns) | 0.1 % |
| 14 | `V322`–`V339` (18 columns) | 83.1 % |

## 10. Identity fields (`id_01`–`id_38`)

Group description: identity information, i.e. network connection information (e.g. IP address, internet service provider, proxy) and digital signature (e.g. user agent, browser, operating system, version), collected by Vesta's fraud-protection system and digital-security partners. **"The field names are masked and pairwise dictionary will not be provided"** [REF-048, partially verified]. `id_12`–`id_38` are organiser-declared categorical [REF-047]; `id_01`–`id_11` are not, so they are numeric. Identity data exist for about a quarter of transactions, so every identity field is mostly missing.

| Column | Table | Official meaning | Observed dtype | Observed range / most frequent values | Interpretation | Model input | Reason for treatment | Missing (dev) | Levels in Train | Caution |
|---|---|---|---|---|---|---|---|---|---|---|
| `id_01` | identity | Masked: individual meaning not disclosed | float64 | -100 to 0 | numeric | include | Not organiser-categorical; numeric | 75.2 % | 65 | 108671 negative values |
| `id_02` | identity | Masked: individual meaning not disclosed | float64 | 1 to 999,595 | numeric | include | Not organiser-categorical; numeric | 75.8 % | 83,905 | Near-continuous (many distinct values) |
| `id_03` | identity | Masked: individual meaning not disclosed | float64 | -13 to 10 | numeric | include | Not organiser-categorical; numeric | 88.8 % | 22 | 169 negative values |
| `id_04` | identity | Masked: individual meaning not disclosed | float64 | -28 to 0 | numeric | include | Not organiser-categorical; numeric | 88.8 % | 15 | 530 negative values |
| `id_05` | identity | Masked: individual meaning not disclosed | float64 | -72 to 52 | numeric | include | Not organiser-categorical; numeric | 76.4 % | 89 | 4883 negative values |
| `id_06` | identity | Masked: individual meaning not disclosed | float64 | -100 to 0 | numeric | include | Not organiser-categorical; numeric | 76.4 % | 99 | 39830 negative values |
| `id_07` | identity | Masked: individual meaning not disclosed | float64 | -46 to 61 | numeric | include | Not organiser-categorical; numeric | 99.1 % | 80 | 289 negative values |
| `id_08` | identity | Masked: individual meaning not disclosed | float64 | -100 to 0 | numeric | include | Not organiser-categorical; numeric | 99.1 % | 93 | 4343 negative values |
| `id_09` | identity | Masked: individual meaning not disclosed | float64 | -36 to 25 | numeric | include | Not organiser-categorical; numeric | 87.3 % | 39 | 412 negative values |
| `id_10` | identity | Masked: individual meaning not disclosed | float64 | -100 to 0 | numeric | include | Not organiser-categorical; numeric | 87.3 % | 56 | 1842 negative values |
| `id_11` | identity | Masked: individual meaning not disclosed | float64 | 90 to 100 | numeric | include | Not organiser-categorical; numeric | 75.7 % | 337 | — |
| `id_12` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | text | `NotFound`, `Found` | categorical | include | Organiser-declared categorical; text values kept exactly | 75.2 % | 2 | — |
| `id_13` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | float64 | 10 to 64 | categorical (numeric-coded) | include | Organiser-declared categorical; integer codes treated as unordered labels, although they may look ordered | 78.3 % | 49 | Strong category drift over time; Validation 2 values unseen in Train: 51.2 % |
| `id_14` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | float64 | -660 to 720 | categorical (numeric-coded) | include | Organiser-declared categorical; integer codes treated as unordered labels, although they may look ordered | 86.0 % | 25 | 70682 negative values |
| `id_15` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | text | `Found`, `New`, `Unknown` | categorical | include | Organiser-declared categorical; text values kept exactly | 75.7 % | 3 | — |
| `id_16` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | text | `Found`, `NotFound` | categorical | include | Organiser-declared categorical; text values kept exactly | 77.7 % | 2 | — |
| `id_17` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | float64 | 100 to 229 | categorical (numeric-coded) | include | Organiser-declared categorical; integer codes treated as unordered labels, although they may look ordered | 76.0 % | 101 | — |
| `id_18` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | float64 | 10 to 29 | categorical (numeric-coded) | include | Organiser-declared categorical; integer codes treated as unordered labels, although they may look ordered | 92.3 % | 18 | — |
| `id_19` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | float64 | 100 to 671 | categorical (numeric-coded) | include | Organiser-declared categorical; integer codes treated as unordered labels, although they may look ordered | 76.0 % | 494 | Validation 2 values unseen in Train: 0.2 % |
| `id_20` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | float64 | 100 to 661 | categorical (numeric-coded) | include | Organiser-declared categorical; integer codes treated as unordered labels, although they may look ordered | 76.0 % | 352 | Validation 2 values unseen in Train: 0.4 % |
| `id_21` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | float64 | 100 to 854 | categorical (numeric-coded) | include | Organiser-declared categorical; integer codes treated as unordered labels, although they may look ordered | 99.1 % | 401 | High cardinality relative to its very sparse coverage; Validation 2 values unseen in Train: 7.8 % |
| `id_22` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | float64 | 10 to 44 | categorical (numeric-coded) | include | Organiser-declared categorical; integer codes treated as unordered labels, although they may look ordered | 99.1 % | 22 | Validation 2 values unseen in Train: 0.2 % |
| `id_23` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | text | `IP_PROXY:TRANSPARENT`, `IP_PROXY:ANONYMOUS`, `IP_PROXY:HIDDEN` | categorical | include | Organiser-declared categorical; text values kept exactly | 99.1 % | 3 | — |
| `id_24` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | float64 | 11 to 26 | categorical (numeric-coded) | include | Organiser-declared categorical; integer codes treated as unordered labels, although they may look ordered | 99.2 % | 10 | Validation 2 values unseen in Train: 2.0 % |
| `id_25` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | float64 | 100 to 548 | categorical (numeric-coded) | include | Organiser-declared categorical; integer codes treated as unordered labels, although they may look ordered | 99.1 % | 295 | High cardinality relative to its very sparse coverage; Validation 2 values unseen in Train: 6.3 % |
| `id_26` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | float64 | 100 to 216 | categorical (numeric-coded) | include | Organiser-declared categorical; integer codes treated as unordered labels, although they may look ordered | 99.1 % | 81 | Validation 2 values unseen in Train: 2.0 % |
| `id_27` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | text | `Found`, `NotFound` | categorical | include | Organiser-declared categorical; text values kept exactly | 99.1 % | 2 | — |
| `id_28` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | text | `Found`, `New` | categorical | include | Organiser-declared categorical; text values kept exactly | 75.7 % | 2 | — |
| `id_29` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | text | `Found`, `NotFound` | categorical | include | Organiser-declared categorical; text values kept exactly | 75.7 % | 2 | — |
| `id_30` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | text | `Windows 10`, `Windows 7`, `iOS 11.2.1`, `iOS 11.1.2`, `Android 7.0` | categorical | include | Organiser-declared categorical; text values kept exactly | 86.5 % | 71 | Validation 2 values unseen in Train: 0.1 % |
| `id_31` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | text | `chrome 63.0`, `mobile safari generic`, `mobile safari 11.0`, `ie 11.0 for desktop`, `safari generic` | categorical | include | Organiser-declared categorical; text values kept exactly | 75.8 % | 101 | Strong category drift over time; Validation 2 values unseen in Train: 40.8 % |
| `id_32` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | float64 | 0 to 32 | categorical (numeric-coded) | include | Organiser-declared categorical; integer codes treated as unordered labels, although they may look ordered | 86.5 % | 4 | — |
| `id_33` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | text | `1920x1080`, `1366x768`, `1334x750`, `2208x1242`, `1440x900` | categorical | include | Organiser-declared categorical; text values kept exactly | 87.3 % | 166 | High cardinality; Validation 2 values unseen in Train: 5.1 % |
| `id_34` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | text | `match_status:2`, `match_status:1`, `match_status:0`, `match_status:-1` | categorical | include | Organiser-declared categorical; text values kept exactly | 86.5 % | 4 | — |
| `id_35` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | text | `T`, `F` | categorical | include | Organiser-declared categorical; text values kept exactly | 75.7 % | 2 | — |
| `id_36` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | text | `F`, `T` | categorical | include | Organiser-declared categorical; text values kept exactly | 75.7 % | 2 | — |
| `id_37` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | text | `T`, `F` | categorical | include | Organiser-declared categorical; text values kept exactly | 75.7 % | 2 | — |
| `id_38` | identity | Masked: individual meaning not disclosed. Listed as categorical by the organiser [REF-047] | text | `T`, `F` | categorical | include | Organiser-declared categorical; text values kept exactly | 75.7 % | 2 | — |

## 11. Device fields

| Column | Table | Official meaning | Observed dtype | Observed range / most frequent values | Interpretation | Model input | Reason for treatment | Missing (dev) | Levels in Train | Caution |
|---|---|---|---|---|---|---|---|---|---|---|
| `DeviceType` | identity | Device type [REF-047 lists it by name]. Listed as categorical by the organiser [REF-047] | text | `desktop`, `mobile` | categorical | include | Organiser-declared categorical | 75.8 % | 2 | — |
| `DeviceInfo` | identity | Device information; free-text-like values. Listed as categorical by the organiser [REF-047] | text | `Windows`, `iOS Device`, `MacOS`, `Trident/7.0`, `rv:11.0` | categorical | include | Organiser-declared categorical; values kept exactly (no grouping) | 79.5 % | 1,470 | Very high cardinality; 98 % of its Train levels have fewer than 100 rows; Validation 2 values unseen in Train: 3.7 % |

## 12. Cross-cutting cautions

- **Masked semantics.** Apart from the target, the key, `TransactionDT`, `TransactionAmt` and the email and device fields, meanings are at most group-level. No field-level meaning is claimed for masked columns.
- **Structured missingness.** The identity block is missing for about three quarters of rows. The V, D and M groups have shared missingness patterns. Missing values are informative: the tree-based models handle them natively, and the neural networks receive a value of 0 together with a missingness indicator for each numeric input with at least one missing Train value (proposed, [DR-012](decisions/DR-012-neural-network-input-representation.md)).
- **High cardinality.** `card1` (11,870 Train levels), `DeviceInfo` (1,470), `card2`, `id_19`, `id_20`, `id_21`, `id_25`, `addr1` and `id_33` stay categorical for every model (DR-008 §7).
- **Category drift.** `id_13` and `id_31` show large shares of Validation 2 values that were never seen in Train. Under DR-008 they become missing values for LightGBM (the Train-fixed category list maps unseen values to NaN, matching LightGBM's documented default), new values for CatBoost, and the `"__MISSING__"` index for the neural networks (proposed, [DR-012](decisions/DR-012-neural-network-input-representation.md)).
- **Entity memorisation.** Card, address and time-delta fields together can partly identify recurring entities. This optimism risk is recorded in DR-001 and in the protocol.
- **Undocumented construction.** The look-back windows of the C, D and V groups are not documented, a stated threat to validity.

## Appendix A — `V1`–`V339` per-column statistics

<details>
<summary>Show all 339 V columns (development rows only)</summary>

| Column | Missing (dev) | Levels in Train | Observed range | Integer-valued |
|---|---|---|---|---|
| `V1` | 50.1 % | 2 | 0 to 1 | yes |
| `V2` | 50.1 % | 7 | 0 to 8 | yes |
| `V3` | 50.1 % | 7 | 0 to 8 | yes |
| `V4` | 50.1 % | 6 | 0 to 6 | yes |
| `V5` | 50.1 % | 7 | 0 to 6 | yes |
| `V6` | 50.1 % | 7 | 0 to 9 | yes |
| `V7` | 50.1 % | 8 | 0 to 9 | yes |
| `V8` | 50.1 % | 7 | 0 to 8 | yes |
| `V9` | 50.1 % | 7 | 0 to 8 | yes |
| `V10` | 50.1 % | 5 | 0 to 4 | yes |
| `V11` | 50.1 % | 5 | 0 to 4 | yes |
| `V12` | 13.6 % | 4 | 0 to 3 | yes |
| `V13` | 13.6 % | 6 | 0 to 6 | yes |
| `V14` | 13.6 % | 2 | 0 to 1 | yes |
| `V15` | 13.6 % | 8 | 0 to 7 | yes |
| `V16` | 13.6 % | 15 | 0 to 15 | yes |
| `V17` | 13.6 % | 16 | 0 to 15 | yes |
| `V18` | 13.6 % | 16 | 0 to 15 | yes |
| `V19` | 13.6 % | 8 | 0 to 7 | yes |
| `V20` | 13.6 % | 15 | 0 to 15 | yes |
| `V21` | 13.6 % | 6 | 0 to 5 | yes |
| `V22` | 13.6 % | 9 | 0 to 8 | yes |
| `V23` | 13.6 % | 14 | 0 to 13 | yes |
| `V24` | 13.6 % | 14 | 0 to 13 | yes |
| `V25` | 13.6 % | 7 | 0 to 7 | yes |
| `V26` | 13.6 % | 13 | 0 to 13 | yes |
| `V27` | 13.6 % | 4 | 0 to 4 | yes |
| `V28` | 13.6 % | 4 | 0 to 4 | yes |
| `V29` | 13.6 % | 6 | 0 to 5 | yes |
| `V30` | 13.6 % | 8 | 0 to 9 | yes |
| `V31` | 13.6 % | 8 | 0 to 7 | yes |
| `V32` | 13.6 % | 15 | 0 to 15 | yes |
| `V33` | 13.6 % | 7 | 0 to 7 | yes |
| `V34` | 13.6 % | 13 | 0 to 13 | yes |
| `V35` | 29.2 % | 4 | 0 to 3 | yes |
| `V36` | 29.2 % | 5 | 0 to 5 | yes |
| `V37` | 29.2 % | 55 | 0 to 54 | yes |
| `V38` | 29.2 % | 55 | 0 to 54 | yes |
| `V39` | 29.2 % | 15 | 0 to 14 | yes |
| `V40` | 29.2 % | 16 | 0 to 24 | yes |
| `V41` | 29.2 % | 2 | 0 to 1 | yes |
| `V42` | 29.2 % | 9 | 0 to 8 | yes |
| `V43` | 29.2 % | 9 | 0 to 8 | yes |
| `V44` | 29.2 % | 49 | 0 to 48 | yes |
| `V45` | 29.2 % | 49 | 0 to 48 | yes |
| `V46` | 29.2 % | 7 | 0 to 6 | yes |
| `V47` | 29.2 % | 8 | 0 to 12 | yes |
| `V48` | 29.2 % | 6 | 0 to 5 | yes |
| `V49` | 29.2 % | 6 | 0 to 5 | yes |
| `V50` | 29.2 % | 6 | 0 to 5 | yes |
| `V51` | 29.2 % | 7 | 0 to 6 | yes |
| `V52` | 29.2 % | 8 | 0 to 12 | yes |
| `V53` | 13.7 % | 6 | 0 to 5 | yes |
| `V54` | 13.7 % | 7 | 0 to 6 | yes |
| `V55` | 13.7 % | 18 | 0 to 17 | yes |
| `V56` | 13.7 % | 20 | 0 to 29 | yes |
| `V57` | 13.7 % | 7 | 0 to 6 | yes |
| `V58` | 13.7 % | 8 | 0 to 7 | yes |
| `V59` | 13.7 % | 12 | 0 to 12 | yes |
| `V60` | 13.7 % | 12 | 0 to 12 | yes |
| `V61` | 13.7 % | 7 | 0 to 6 | yes |
| `V62` | 13.7 % | 8 | 0 to 7 | yes |
| `V63` | 13.7 % | 8 | 0 to 7 | yes |
| `V64` | 13.7 % | 8 | 0 to 7 | yes |
| `V65` | 13.7 % | 2 | 0 to 1 | yes |
| `V66` | 13.7 % | 6 | 0 to 5 | yes |
| `V67` | 13.7 % | 9 | 0 to 8 | yes |
| `V68` | 13.7 % | 3 | 0 to 2 | yes |
| `V69` | 13.7 % | 6 | 0 to 5 | yes |
| `V70` | 13.7 % | 7 | 0 to 6 | yes |
| `V71` | 13.7 % | 7 | 0 to 6 | yes |
| `V72` | 13.7 % | 8 | 0 to 7 | yes |
| `V73` | 13.7 % | 6 | 0 to 5 | yes |
| `V74` | 13.7 % | 9 | 0 to 8 | yes |
| `V75` | 15.8 % | 5 | 0 to 4 | yes |
| `V76` | 15.8 % | 7 | 0 to 6 | yes |
| `V77` | 15.8 % | 31 | 0 to 30 | yes |
| `V78` | 15.8 % | 31 | 0 to 30 | yes |
| `V79` | 15.8 % | 8 | 0 to 7 | yes |
| `V80` | 15.8 % | 20 | 0 to 19 | yes |
| `V81` | 15.8 % | 20 | 0 to 19 | yes |
| `V82` | 15.8 % | 8 | 0 to 7 | yes |
| `V83` | 15.8 % | 8 | 0 to 7 | yes |
| `V84` | 15.8 % | 8 | 0 to 7 | yes |
| `V85` | 15.8 % | 8 | 0 to 7 | yes |
| `V86` | 15.8 % | 31 | 0 to 30 | yes |
| `V87` | 15.8 % | 31 | 0 to 30 | yes |
| `V88` | 15.8 % | 2 | 0 to 1 | yes |
| `V89` | 15.8 % | 3 | 0 to 2 | yes |
| `V90` | 15.8 % | 6 | 0 to 5 | yes |
| `V91` | 15.8 % | 7 | 0 to 6 | yes |
| `V92` | 15.8 % | 8 | 0 to 7 | yes |
| `V93` | 15.8 % | 8 | 0 to 7 | yes |
| `V94` | 15.8 % | 2 | 0 to 2 | yes |
| `V95` | 0.0 % | 28 | 0 to 27 | yes |
| `V96` | 0.0 % | 127 | 0 to 126 | yes |
| `V97` | 0.0 % | 49 | 0 to 48 | yes |
| `V98` | 0.0 % | 13 | 0 to 12 | yes |
| `V99` | 0.0 % | 89 | 0 to 88 | yes |
| `V100` | 0.0 % | 26 | 0 to 28 | yes |
| `V101` | 0.0 % | 23 | 0 to 22 | yes |
| `V102` | 0.0 % | 116 | 0 to 115 | yes |
| `V103` | 0.0 % | 46 | 0 to 45 | yes |
| `V104` | 0.0 % | 16 | 0 to 15 | yes |
| `V105` | 0.0 % | 39 | 0 to 38 | yes |
| `V106` | 0.0 % | 24 | 0 to 23 | yes |
| `V107` | 0.0 % | 1 | 0 to 1 | yes |
| `V108` | 0.0 % | 8 | 0 to 7 | yes |
| `V109` | 0.0 % | 8 | 0 to 7 | yes |
| `V110` | 0.0 % | 8 | 0 to 7 | yes |
| `V111` | 0.0 % | 10 | 0 to 9 | yes |
| `V112` | 0.0 % | 10 | 0 to 9 | yes |
| `V113` | 0.0 % | 10 | 0 to 9 | yes |
| `V114` | 0.0 % | 7 | 0 to 6 | yes |
| `V115` | 0.0 % | 7 | 0 to 6 | yes |
| `V116` | 0.0 % | 7 | 0 to 6 | yes |
| `V117` | 0.0 % | 4 | 0 to 3 | yes |
| `V118` | 0.0 % | 4 | 0 to 3 | yes |
| `V119` | 0.0 % | 4 | 0 to 3 | yes |
| `V120` | 0.0 % | 4 | 0 to 3 | yes |
| `V121` | 0.0 % | 4 | 0 to 3 | yes |
| `V122` | 0.0 % | 4 | 0 to 3 | yes |
| `V123` | 0.0 % | 14 | 0 to 13 | yes |
| `V124` | 0.0 % | 14 | 0 to 13 | yes |
| `V125` | 0.0 % | 14 | 0 to 13 | yes |
| `V126` | 0.0 % | 6,288 | 0 to 160,000 | no |
| `V127` | 0.0 % | 16,802 | 0 to 160,000 | no |
| `V128` | 0.0 % | 9,470 | 0 to 160,000 | no |
| `V129` | 0.0 % | 1,508 | 0 to 55,125 | no |
| `V130` | 0.0 % | 9,174 | 0 to 55,125 | no |
| `V131` | 0.0 % | 3,364 | 0 to 55,125 | no |
| `V132` | 0.0 % | 3,704 | 0 to 63,874.781 | no |
| `V133` | 0.0 % | 6,385 | 0 to 63,874.781 | no |
| `V134` | 0.0 % | 5,018 | 0 to 63,874.781 | no |
| `V135` | 0.0 % | 2,505 | 0 to 90,750 | no |
| `V136` | 0.0 % | 3,281 | 0 to 90,750 | no |
| `V137` | 0.0 % | 2,847 | 0 to 90,750 | no |
| `V138` | 85.7 % | 23 | 0 to 22 | yes |
| `V139` | 85.7 % | 34 | 0 to 33 | yes |
| `V140` | 85.7 % | 34 | 0 to 33 | yes |
| `V141` | 85.7 % | 6 | 0 to 5 | yes |
| `V142` | 85.7 % | 10 | 0 to 9 | yes |
| `V143` | 85.7 % | 55 | 0 to 54 | yes |
| `V144` | 85.7 % | 63 | 0 to 62 | yes |
| `V145` | 85.7 % | 255 | 0 to 297 | yes |
| `V146` | 85.7 % | 18 | 0 to 17 | yes |
| `V147` | 85.7 % | 18 | 0 to 17 | yes |
| `V148` | 85.7 % | 16 | 0 to 15 | yes |
| `V149` | 85.7 % | 16 | 0 to 15 | yes |
| `V150` | 85.7 % | 1,985 | 1 to 3,389 | yes |
| `V151` | 85.7 % | 56 | 1 to 57 | yes |
| `V152` | 85.7 % | 27 | 1 to 69 | yes |
| `V153` | 85.7 % | 8 | 0 to 7 | yes |
| `V154` | 85.7 % | 9 | 0 to 8 | yes |
| `V155` | 85.7 % | 15 | 0 to 14 | yes |
| `V156` | 85.7 % | 17 | 0 to 16 | yes |
| `V157` | 85.7 % | 16 | 0 to 15 | yes |
| `V158` | 85.7 % | 17 | 0 to 16 | yes |
| `V159` | 85.7 % | 6,312 | 0 to 55,125 | no |
| `V160` | 85.7 % | 8,667 | 0 to 641,511.438 | no |
| `V161` | 85.7 % | 67 | 0 to 3,300 | no |
| `V162` | 85.7 % | 132 | 0 to 3,300 | no |
| `V163` | 85.7 % | 90 | 0 to 3,300 | no |
| `V164` | 85.7 % | 690 | 0 to 20,400 | no |
| `V165` | 85.7 % | 1,115 | 0 to 20,400 | no |
| `V166` | 85.7 % | 878 | 0 to 101,640 | no |
| `V167` | 76.0 % | 59 | 0 to 58 | yes |
| `V168` | 76.0 % | 213 | 0 to 215 | yes |
| `V169` | 76.0 % | 20 | 0 to 19 | yes |
| `V170` | 76.0 % | 49 | 0 to 48 | yes |
| `V171` | 76.0 % | 49 | 0 to 61 | yes |
| `V172` | 76.0 % | 32 | 0 to 31 | yes |
| `V173` | 76.0 % | 8 | 0 to 7 | yes |
| `V174` | 76.0 % | 9 | 0 to 8 | yes |
| `V175` | 76.0 % | 15 | 0 to 14 | yes |
| `V176` | 76.0 % | 49 | 0 to 48 | yes |
| `V177` | 76.0 % | 48 | 0 to 47 | yes |
| `V178` | 76.0 % | 127 | 0 to 130 | yes |
| `V179` | 76.0 % | 69 | 0 to 68 | yes |
| `V180` | 76.0 % | 67 | 0 to 68 | yes |
| `V181` | 76.0 % | 17 | 0 to 16 | yes |
| `V182` | 76.0 % | 50 | 0 to 50 | yes |
| `V183` | 76.0 % | 30 | 0 to 30 | yes |
| `V184` | 76.0 % | 17 | 0 to 16 | yes |
| `V185` | 76.0 % | 17 | 0 to 16 | yes |
| `V186` | 76.0 % | 39 | 0 to 38 | yes |
| `V187` | 76.0 % | 210 | 0 to 218 | yes |
| `V188` | 76.0 % | 18 | 0 to 30 | yes |
| `V189` | 76.0 % | 18 | 0 to 30 | yes |
| `V190` | 76.0 % | 43 | 0 to 42 | yes |
| `V191` | 76.0 % | 22 | 0 to 21 | yes |
| `V192` | 76.0 % | 44 | 0 to 44 | yes |
| `V193` | 76.0 % | 38 | 0 to 37 | yes |
| `V194` | 76.0 % | 8 | 0 to 7 | yes |
| `V195` | 76.0 % | 9 | 0 to 8 | yes |
| `V196` | 76.0 % | 39 | 0 to 38 | yes |
| `V197` | 76.0 % | 15 | 0 to 14 | yes |
| `V198` | 76.0 % | 15 | 0 to 14 | yes |
| `V199` | 76.0 % | 46 | 0 to 45 | yes |
| `V200` | 76.0 % | 46 | 0 to 45 | yes |
| `V201` | 76.0 % | 46 | 0 to 55 | yes |
| `V202` | 76.0 % | 6,065 | 0 to 101,640 | no |
| `V203` | 76.0 % | 8,556 | 0 to 101,640 | no |
| `V204` | 76.0 % | 7,312 | 0 to 101,640 | no |
| `V205` | 76.0 % | 1,486 | 0 to 55,125 | no |
| `V206` | 76.0 % | 1,140 | 0 to 55,125 | no |
| `V207` | 76.0 % | 2,169 | 0 to 55,125 | no |
| `V208` | 76.0 % | 1,647 | 0 to 3,300 | no |
| `V209` | 76.0 % | 2,174 | 0 to 8,050 | no |
| `V210` | 76.0 % | 1,816 | 0 to 3,300 | no |
| `V211` | 76.0 % | 3,945 | 0 to 13,200 | no |
| `V212` | 76.0 % | 4,717 | 0 to 66,000 | no |
| `V213` | 76.0 % | 4,395 | 0 to 16,950 | no |
| `V214` | 76.0 % | 1,420 | 0 to 101,640 | no |
| `V215` | 76.0 % | 1,695 | 0 to 101,640 | no |
| `V216` | 76.0 % | 1,566 | 0 to 101,640 | no |
| `V217` | 77.4 % | 54 | 0 to 53 | yes |
| `V218` | 77.4 % | 99 | 0 to 219 | yes |
| `V219` | 77.4 % | 78 | 0 to 79 | yes |
| `V220` | 75.7 % | 24 | 0 to 25 | yes |
| `V221` | 75.7 % | 72 | 0 to 384 | yes |
| `V222` | 75.7 % | 72 | 0 to 384 | yes |
| `V223` | 77.4 % | 17 | 0 to 16 | yes |
| `V224` | 77.4 % | 77 | 0 to 144 | yes |
| `V225` | 77.4 % | 34 | 0 to 51 | yes |
| `V226` | 77.4 % | 34 | 0 to 242 | yes |
| `V227` | 75.7 % | 42 | 0 to 360 | yes |
| `V228` | 77.4 % | 54 | 0 to 54 | yes |
| `V229` | 77.4 % | 90 | 0 to 176 | yes |
| `V230` | 77.4 % | 65 | 0 to 65 | yes |
| `V231` | 77.4 % | 52 | 0 to 51 | yes |
| `V232` | 77.4 % | 52 | 0 to 56 | yes |
| `V233` | 77.4 % | 52 | 0 to 51 | yes |
| `V234` | 75.7 % | 121 | 0 to 121 | yes |
| `V235` | 77.4 % | 16 | 0 to 15 | yes |
| `V236` | 77.4 % | 29 | 0 to 34 | yes |
| `V237` | 77.4 % | 17 | 0 to 16 | yes |
| `V238` | 75.7 % | 19 | 0 to 23 | yes |
| `V239` | 75.7 % | 19 | 0 to 23 | yes |
| `V240` | 77.4 % | 5 | 0 to 7 | yes |
| `V241` | 77.4 % | 4 | 0 to 5 | yes |
| `V242` | 77.4 % | 20 | 0 to 20 | yes |
| `V243` | 77.4 % | 42 | 0 to 57 | yes |
| `V244` | 77.4 % | 22 | 0 to 22 | yes |
| `V245` | 75.7 % | 50 | 0 to 262 | yes |
| `V246` | 77.4 % | 45 | 0 to 45 | yes |
| `V247` | 77.4 % | 12 | 0 to 12 | yes |
| `V248` | 77.4 % | 15 | 0 to 36 | yes |
| `V249` | 77.4 % | 15 | 0 to 22 | yes |
| `V250` | 75.7 % | 15 | 0 to 17 | yes |
| `V251` | 75.7 % | 15 | 0 to 17 | yes |
| `V252` | 77.4 % | 18 | 0 to 18 | yes |
| `V253` | 77.4 % | 59 | 0 to 163 | yes |
| `V254` | 77.4 % | 38 | 0 to 60 | yes |
| `V255` | 75.7 % | 36 | 0 to 87 | yes |
| `V256` | 75.7 % | 42 | 0 to 87 | yes |
| `V257` | 77.4 % | 48 | 0 to 48 | yes |
| `V258` | 77.4 % | 66 | 0 to 66 | yes |
| `V259` | 75.7 % | 64 | 0 to 285 | yes |
| `V260` | 77.4 % | 8 | 0 to 8 | yes |
| `V261` | 77.4 % | 40 | 0 to 49 | yes |
| `V262` | 77.4 % | 21 | 0 to 20 | yes |
| `V263` | 77.4 % | 6,173 | 0 to 153,600 | no |
| `V264` | 77.4 % | 8,116 | 0 to 153,600 | no |
| `V265` | 77.4 % | 7,035 | 0 to 153,600 | no |
| `V266` | 77.4 % | 1,388 | 0 to 55,125 | no |
| `V267` | 77.4 % | 2,379 | 0 to 55,125 | no |
| `V268` | 77.4 % | 1,804 | 0 to 55,125 | no |
| `V269` | 77.4 % | 128 | 0 to 55,125 | no |
| `V270` | 75.7 % | 1,508 | 0 to 4,000 | no |
| `V271` | 75.7 % | 1,817 | 0 to 4,000 | no |
| `V272` | 75.7 % | 1,631 | 0 to 4,000 | no |
| `V273` | 77.4 % | 4,119 | 0 to 51,200 | no |
| `V274` | 77.4 % | 4,826 | 0 to 66,000 | no |
| `V275` | 77.4 % | 4,468 | 0 to 51,200 | no |
| `V276` | 77.4 % | 1,421 | 0 to 101,640 | no |
| `V277` | 77.4 % | 1,618 | 0 to 101,640 | no |
| `V278` | 77.4 % | 1,529 | 0 to 101,640 | no |
| `V279` | 0.0 % | 56 | 0 to 66 | yes |
| `V280` | 0.0 % | 153 | 0 to 323 | yes |
| `V281` | 0.2 % | 23 | 0 to 22 | yes |
| `V282` | 0.2 % | 33 | 0 to 32 | yes |
| `V283` | 0.2 % | 52 | 0 to 68 | yes |
| `V284` | 0.0 % | 13 | 0 to 12 | yes |
| `V285` | 0.0 % | 96 | 0 to 95 | yes |
| `V286` | 0.0 % | 9 | 0 to 8 | yes |
| `V287` | 0.0 % | 31 | 0 to 31 | yes |
| `V288` | 0.2 % | 11 | 0 to 10 | yes |
| `V289` | 0.2 % | 12 | 0 to 12 | yes |
| `V290` | 0.0 % | 56 | 1 to 67 | yes |
| `V291` | 0.0 % | 150 | 1 to 1,055 | yes |
| `V292` | 0.0 % | 105 | 1 to 323 | yes |
| `V293` | 0.0 % | 55 | 0 to 66 | yes |
| `V294` | 0.0 % | 236 | 0 to 1,093 | yes |
| `V295` | 0.0 % | 145 | 0 to 322 | yes |
| `V296` | 0.2 % | 60 | 0 to 61 | yes |
| `V297` | 0.0 % | 13 | 0 to 12 | yes |
| `V298` | 0.0 % | 44 | 0 to 47 | yes |
| `V299` | 0.0 % | 26 | 0 to 26 | yes |
| `V300` | 0.2 % | 12 | 0 to 11 | yes |
| `V301` | 0.2 % | 12 | 0 to 12 | yes |
| `V302` | 0.0 % | 11 | 0 to 10 | yes |
| `V303` | 0.0 % | 14 | 0 to 17 | yes |
| `V304` | 0.0 % | 11 | 0 to 11 | yes |
| `V305` | 0.0 % | 2 | 1 to 2 | yes |
| `V306` | 0.0 % | 10,193 | 0 to 108,800 | no |
| `V307` | 0.0 % | 25,530 | 0 to 121,000 | no |
| `V308` | 0.0 % | 15,057 | 0 to 108,800 | no |
| `V309` | 0.0 % | 2,962 | 0 to 55,125 | no |
| `V310` | 0.0 % | 13,730 | 0 to 55,125 | no |
| `V311` | 0.0 % | 2,123 | 0 to 55,125 | no |
| `V312` | 0.0 % | 5,707 | 0 to 55,125 | no |
| `V313` | 0.2 % | 3,839 | 0 to 4,817.470 | no |
| `V314` | 0.2 % | 8,045 | 0 to 7,519.870 | no |
| `V315` | 0.2 % | 4,890 | 0 to 4,817.470 | no |
| `V316` | 0.0 % | 5,874 | 0 to 63,874.781 | no |
| `V317` | 0.0 % | 9,921 | 0 to 86,715.961 | no |
| `V318` | 0.0 % | 7,749 | 0 to 63,874.781 | no |
| `V319` | 0.0 % | 3,206 | 0 to 101,640 | no |
| `V320` | 0.0 % | 4,331 | 0 to 101,640 | no |
| `V321` | 0.0 % | 3,695 | 0 to 101,640 | no |
| `V322` | 85.7 % | 33 | 0 to 32 | yes |
| `V323` | 85.7 % | 50 | 0 to 49 | yes |
| `V324` | 85.7 % | 34 | 0 to 33 | yes |
| `V325` | 85.7 % | 13 | 0 to 12 | yes |
| `V326` | 85.7 % | 45 | 0 to 44 | yes |
| `V327` | 85.7 % | 19 | 0 to 18 | yes |
| `V328` | 85.7 % | 16 | 0 to 15 | yes |
| `V329` | 85.7 % | 25 | 0 to 24 | yes |
| `V330` | 85.7 % | 16 | 0 to 15 | yes |
| `V331` | 85.7 % | 340 | 0 to 160,000 | no |
| `V332` | 85.7 % | 823 | 0 to 160,000 | no |
| `V333` | 85.7 % | 543 | 0 to 160,000 | no |
| `V334` | 85.7 % | 118 | 0 to 55,125 | no |
| `V335` | 85.7 % | 508 | 0 to 55,125 | no |
| `V336` | 85.7 % | 279 | 0 to 55,125 | no |
| `V337` | 85.7 % | 149 | 0 to 64,800 | no |
| `V338` | 85.7 % | 213 | 0 to 64,800 | no |
| `V339` | 85.7 % | 175 | 0 to 64,800 | no |

</details>

## References

REF-047, REF-048\* — see [`references.md`](references.md). Model-input policy: [DR-008](decisions/DR-008-feature-declaration-and-model-input-policy.md).

\* Partially verified.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [D] | Evidence tag: observed in the development data |
| `__MISSING__` | Label given to a missing categorical value before modelling (DR-008) |
| dev | Development rows: relative days 1–154 (Train, Validation 1, Validation 2) |
| DR | Decision Record |
| dtype | Stored data type: `int64`, `float64` or text |
| `float64`, `int64` | 64-bit floating-point and integer types |
| IEEE-CIS | IEEE Computational Intelligence Society |
| IP | Internet Protocol (address) |
| NaN | Not-a-number: missing-value marker for numeric data |
| REF-NNN | Reference identifier in `references.md` |
| T / F | True / False values of the match flags |
| Train | Relative days 1–105 |
| USD | United States dollars |
| `V1`–`V339` | Organiser-engineered feature columns (not the validation periods) |
| Validation 1, Validation 2 | Relative days 106–126 and 127–154 |
| Column names (`TransactionID`, `card1`, `id_01`, …) | Defined in their own rows above |
