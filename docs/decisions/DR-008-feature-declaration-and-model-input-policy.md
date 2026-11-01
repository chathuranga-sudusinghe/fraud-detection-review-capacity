# DR-008 — Feature declaration and model-input policy

| Field | Value |
|---|---|
| Status | Frozen (accepted before the freeze), Issue #23, in line with the University of Greenwich criteria ([criteria checklist](../greenwich-criteria-checklist.md)); supervisor feedback welcome, not a precondition |
| Date | 2026-09-24; revised 2026-10-09 (Issue #54), 2026-10-10 (Issue #57: neural-network representation moved to DR-012) and 2026-10-10 (Issue #64: time-of-day inputs) and 2026-10-12 (Issue #66: CatBoost row order) and 2026-10-13 (Issue #87: population preservation, §9) |
| Evidence source | Raw-file headers (schema only); development-period values (relative days 1–154) for type, missingness and cardinality checks, **with the target column never loaded**; organiser data description; LightGBM and CatBoost documentation; input-type checks on synthetic data; `notebooks/01_dataset_feasibility_audit.ipynb` §§2.3, 2.6, 8, 9 (§2.6: label-free completeness of `TransactionDT` over all 182 relative days); fraud-detection and software literature on periodic time features [REF-007, REF-104]; the label-free audit of repeated input vectors in the development rows (Issue #86) |
| Related | [DR-001](DR-001-temporal-split-strategy.md), [DR-002](DR-002-review-capacity-definition.md), [DR-011](DR-011-model-set-tree-based-and-neural-families.md), [DR-012](DR-012-neural-network-input-representation.md), [protocol](../protocol.md), [`config/features.yaml`](../../config/features.yaml), [feature catalog](../feature-catalog.md) |

## Context

The study compares four models in two families, as research question 1 (RQ1) and hypothesis H1 (model-family effect) require ([DR-011](DR-011-model-set-tree-based-and-neural-families.md)): the gradient-boosting libraries LightGBM and CatBoost, and the neural networks ResNet-style multilayer perceptron (MLP) and Feature Tokenizer + Transformer (FT-Transformer). The protocol requires all four to receive **the same declared model inputs** under one common training procedure, so that a difference between the families can be attributed to the model-and-representation pipelines compared, not to unequal data or effort. Each family keeps the representation its models require: §5 for the tree-based models, [DR-012](DR-012-neural-network-input-representation.md) for the neural networks.

The declaration is not trivial:
- many nominal (unordered, label-like) fields are stored as numbers (`card1`, `addr1`, several `id_*` fields);
- the models accept categorical data and missing values in **different technical forms**;
- no model may be given information that another model does not receive.

This record fixes, before any model is implemented: the target, the excluded columns, the categorical columns, the numeric columns, the two time-of-day inputs derived from `TransactionDT`, the canonical frame and the exact technical conversions for the tree-based libraries. The neural-network representation is in [DR-012](DR-012-neural-network-input-representation.md).

**Scope limits.** Apart from the two time-of-day inputs (§4), no feature is engineered, target-encoded, aggregated or created from pseudo-entity keys. No model is trained or tuned. The temporal split ([DR-001](DR-001-temporal-split-strategy.md)), capacity settings ([DR-002](DR-002-review-capacity-definition.md)), class weighting and hypotheses are defined elsewhere.

## Decision

### 1. Target

`isFraud` is the target. **It is never a model input**, in any form.

### 2. Excluded columns (2)

| Column | Reason |
|---|---|
| `TransactionID` | Row identifier; monotone in time, i.e. a disguised timestamp [D] |
| `TransactionDT` | Raw time offset. Used to derive relative days and the temporal split ([DR-001](DR-001-temporal-split-strategy.md)) and as the source of the two time-of-day inputs (§4); never a model input itself |

**No further column is excluded.**
- High missingness (up to 99.2 %, `id_24`) and high cardinality are **not** reasons for exclusion: every model handles missing values (natively for the trees, through indicators for the neural networks, [DR-012](DR-012-neural-network-input-representation.md)), and the categorical columns are the organiser's own list.
- The known risks of the undocumented look-back windows of the C, D and V feature groups, and of entity memorisation through `card1`-type fields, are already stated as threats to validity in the protocol. They are not new exclusion grounds.
- Heuristic pseudo-entity keys remain audit tools only and are **not** model inputs.

### 3. Categorical columns (49) — the organiser's list, unchanged

| Group | Columns | Count | Reason |
|---|---|---|---|
| Product | `ProductCD` | 1 | Organiser-declared categorical; 5 text codes |
| Card | `card1`, `card2`, `card3`, `card4`, `card5`, `card6` | 6 | Organiser-declared. `card4` and `card6` are text; `card1`–`card3` and `card5` are **integer-coded** labels whose numeric order has no documented meaning |
| Address | `addr1`, `addr2` | 2 | Organiser-declared; integer-coded billing region and country labels |
| Email domain | `P_emaildomain`, `R_emaildomain` | 2 | Organiser-declared; text domain names |
| Match flags | `M1`–`M9` | 9 | Organiser-declared; T/F flags, and M0/M1/M2 for `M4` |
| Device | `DeviceType`, `DeviceInfo` | 2 | Organiser-declared; text |
| Identity | `id_12`–`id_38` | 27 | Organiser-declared. 15 are text. The other 12 (`id_13`, `id_14`, `id_17`–`id_22`, `id_24`–`id_26`, `id_32`) are **integer-coded** labels |

**Why the organiser's list is adopted as it stands [I].**
- It is the only documented source of variable roles for these masked fields.
- Overriding it would require inventing semantics that the organiser withholds.
- Several numeric-coded identity fields look ordered (e.g. `id_14` ranges from −660 to 720 and `id_32` takes the values 0/16/24/32). Their meaning is **not documented**, so no ordinal interpretation is imposed; they stay categorical as declared.
- The feasibility audit already used this list (notebook §8).

### 4. Numeric inputs (384) — every remaining joined column and two time-of-day inputs

**Numeric joined columns (382).**

| Group | Columns | Count |
|---|---|---|
| Amount | `TransactionAmt` | 1 |
| Distance | `dist1`, `dist2` | 2 |
| Counts | `C1`–`C14` | 14 |
| Time deltas | `D1`–`D15` (including `D9`; negative values kept) | 15 |
| Organiser-engineered | `V1`–`V339` (the feature columns, not the validation partitions) | 339 |
| Identity numeric | `id_01`–`id_11` | 11 |

**Rule:** a joined column is numeric **if and only if** it is neither the target, nor excluded, nor in the categorical list. Binary-valued V columns remain numeric: they have no organiser categorical label, and a two-valued numeric split is equivalent for trees.

**Time-of-day inputs (2).** Two numeric inputs place each transaction on the cycle of its relative day (the `TransactionDT` time offset in seconds divided by 86,400 seconds and rounded down; [DR-001](DR-001-temporal-split-strategy.md)). With `t` the transaction's `TransactionDT` value in seconds:

$$
\begin{aligned}
\theta &= 360^\circ \times \frac{t \bmod 86400}{86400} \\
\mathrm{tod\_sin} &= \sin\theta \\
\mathrm{tod\_cos} &= \cos\theta
\end{aligned}
$$

where:

- `t` = `TransactionDT` of the transaction, in seconds from the undisclosed reference point;
- `t mod 86400` = seconds elapsed since the start of the transaction's relative day;
- `θ` = time-of-day angle: the fraction of the relative day elapsed, expressed as an angle on a full circle of 360°, with 0° ≤ θ < 360°;
- `tod_sin`, `tod_cos` = the two time-of-day inputs, each between −1 and 1.

The pair is a point on the unit circle: the end of one relative day lies next to the start of the next, and the pair identifies every time of day uniquely.

- **Continuous angle.** θ keeps the full one-second resolution of `TransactionDT`; it is not rounded to whole hours.
- **Relative-day phase, not clock time.** The reference point of `TransactionDT` is undisclosed [REF-047], so θ is a position within the relative day, not the local time of day. No time zone, clock hour or calendar label is inferred.
- **Row-local and label-free.** Each pair is computed from the transaction's own `TransactionDT` only: no other row, label or fitted statistic is used, so the same rule applies unchanged to Train, Validation 1, Validation 2 and Test.
- **Equal exposure.** All four models receive both inputs, as ordinary numeric inputs: the tree-based models receive their values unchanged; the neural networks apply the same numeric rules as to every other numeric input ([DR-012](DR-012-neural-network-input-representation.md)). No missingness indicator arises, because the inputs are never missing.
- **Validation of `TransactionDT`.** Data preparation stops with an error if any `TransactionDT` value is missing, non-finite or not a whole number of seconds, or if its relative day lies outside 1–182. Such a row cannot be assigned to a partition ([DR-001](DR-001-temporal-split-strategy.md)) or to a daily review budget ([DR-002](DR-002-review-capacity-definition.md)), and imputing it would invent both its partition and its time of day; dropping it would change the evaluated population and the daily volumes. The completeness of the two inputs therefore follows from this validation rule, not only from the observed completeness of `TransactionDT`.
- **`D9` is kept unchanged.** `D9` overlaps with the time of day only partly (Evidence) and is neither filled, overwritten nor corrected from the new inputs.

**Why the time-of-day inputs are added [I].** Transaction time is an established fraud-detection signal: Correa Bahnsen et al. model the time of a transaction as a periodic quantity on a 24-hour clock and report an average 13 % increase in savings from their periodic features [REF-007]; those are customer-level features combined with aggregate features, so no benefit is claimed for the row-local pair used here. In this dataset, `D9` is the declared input that the inspection identified as partly encoding the hour of the relative day; it is missing on 87.3 % of development rows [D], and other masked columns are not known to carry the time of day; `TransactionDT` is complete on all 182 relative days [D]. The sine and cosine pair is the standard encoding of a periodic quantity without a jump at the end of the period [REF-104]. The decision was informed by development-data inspection (the `D9` evidence) and was taken before any model was run; it is not selected or tuned on Validation 1 or Validation 2 performance.

**Why the continuous angle, not the whole hour [I].** The reference point of `TransactionDT` is undisclosed, so whole-hour boundaries counted from it have no documented meaning. If the recorded offsets are shifted by a constant c seconds (t′ = t + c for every transaction), the continuous pair is only rotated:

$$
\theta' = \left(\theta + 360^\circ \times \frac{c}{86400}\right) \bmod 360^\circ
$$

where:

- `c` = a constant number of seconds added to every recorded `TransactionDT` value (a shift of the reference point by c seconds in the opposite direction);
- `t′` = the shifted value t + c;
- `θ′` = the angle computed from t′.

The shift adds the same angle to every transaction, so the pair (`tod_sin`, `tod_cos`) is rotated as a whole and the relative positions of all times of day are unchanged.

With whole hours, the same statement holds only for shifts of whole hours; any other shift moves transactions between hourly bins. The continuous angle also keeps the resolution within the hour. Whole hours would allow only 24 possible pairs, and each coordinate would take only 13 distinct values (for example, the sine is the same at 1 and 11 hours), giving large ties in the neural networks' quantile transformation; the continuous angle avoids these coarse ties, although transactions recorded at the same second still share a pair. Rotation of the inputs does not imply identical predictions from axis-aligned tree splits or from per-input transformations; the point is that the continuous pair carries no arbitrary boundary.

### 5. One canonical input, then the model-specific form

All four models start from **one canonical data frame**, built identically:

1. Validate `TransactionDT` and derive `tod_sin` and `tod_cos` from it (§4); then drop `isFraud`, `TransactionID` and `TransactionDT`.
2. **Categorical columns → canonical text labels:**
   - text values are kept exactly (no case folding, trimming, merging or grouping of rare levels);
   - integer-valued numeric codes are written as whole-number text (`100.0` → `"100"`, `1000` → `"1000"`);
   - the implementation must **fail loudly** if a numeric-coded categorical value is not a whole number;
   - missing values become the explicit label `"__MISSING__"`.
3. **Numeric columns and the two time-of-day inputs → `float64`**, with missing values left as NaN. No imputation, scaling, clipping, winsorising or transformation.

Then, for each tree-based library (the neural-network form is in [DR-012](DR-012-neural-network-input-representation.md)):

| Step | LightGBM | CatBoost |
|---|---|---|
| Categorical type | pandas `Categorical`, with the **category list fixed from Train rows only** and reused unchanged for Validation 1, Validation 2 and Test | Canonical text columns passed directly |
| Declared categorical columns | `categorical_feature` = the 49 names | `cat_features` = the 49 names |
| Missing categorical value | The level `"__MISSING__"` (a normal category) | The level `"__MISSING__"` (a normal category) |
| Category unseen in Train | Becomes NaN **during preprocessing**, because it is absent from the Train-fixed pandas category list; LightGBM then treats it as missing. This reproduces LightGBM's own documented predict-time behaviour [REF-076] | Passed as its text label; handled natively by CatBoost as a value not seen in training |
| Numeric missing value | Native (`use_missing = true`, `zero_as_missing = false`; library defaults) | Native (`nan_mode = Min`; library default) |
| Extra encodings | None | None. CatBoost's internal ordered target statistics are part of the model family, not preprocessing [REF-012] |
| Row order | No order-dependent statistics | `has_time = true`: the training rows are passed in time order (`TransactionDT`, with `TransactionID` as a deterministic tie-break), and the ordered target-dependent statistics follow this Train order instead of a random permutation [REF-012; REF-118] |

**Why CatBoost uses the time order [I].** CatBoost's ordered target statistics follow an online-learning principle: each example's statistic uses only the "history" before it, and for data without a time order the method introduces an artificial time, a random permutation of the training examples [REF-012, Section 3.2]. These data have a recorded time order, and the study's design is temporal throughout. With `has_time = true`, CatBoost uses the supplied Train order for its ordered target-dependent categorical statistics. Ordering by `TransactionDT` with a `TransactionID` tie-break is an ordering convention: it does not establish strictly earlier timestamps for tied rows, nor when fraud labels became available. CatBoost's frequency statistics (Counter on the central processing unit (CPU), FeatureFreq on the graphics processing unit (GPU)) remain computed over the whole Train set under `counter_calc_method = SkipTest` and use no labels. Both the time order and random permutations stay within Train, so neither leaks information from the validation or Test periods; the time order is chosen for consistency with the study's temporal design.

**Why the missing-value label is explicit for categoricals [I].** CatBoost cannot accept NaN in a categorical column (checked under [Library requirements](#library-requirements-l-checked-on-synthetic-data)), so some label is technically required. Using the **same** label in both libraries gives them identical categorical information. For LightGBM a missing value simply becomes one more category it can split on. Numeric missing values stay native in both libraries, as the protocol prefers.

**Why LightGBM categories are fixed from Train [I].** It keeps the category-to-code mapping free of later-period information. It also matches LightGBM's own behaviour of aligning categories to those seen in training [REF-076].

### 6. Parity statement

- **Same semantic input:** all four models receive the same 433 inputs (the 431 declared columns and the two time-of-day inputs), the same 49 categorical declarations, the same canonical category labels and the same numeric values.
- **No other engineered information**: apart from the two time-of-day inputs, which every model receives identically, no counts, frequencies, target statistics or entity keys are computed outside the models.
- **Family-specific representation.** The trees use native categorical and missing-value handling (§5); the neural networks receive the same information as integer category indices, transformed numeric values and missingness indicators ([DR-012](DR-012-neural-network-input-representation.md)). These forms are part of each family's design and therefore of the family contrast.
- **Two unavoidable differences between the tree libraries remain.** Both are part of each library's native design, not of this study's preprocessing:
  1. how categories are split (LightGBM's gradient-sorted category partitions versus CatBoost's ordered target statistics);
  2. how an unseen category is treated. For LightGBM the Train-fixed category list turns it into a missing value, matching LightGBM's documented default; CatBoost handles it natively as a new value.

  These are "categorical treatment" differences that belong to the models themselves (feasibility notebook §9).
- **Tuning parameters are not fixed here.** LightGBM's categorical-regularisation parameters (`max_cat_threshold`, `cat_l2`, `cat_smooth`, `min_data_per_group`) and CatBoost's corresponding settings belong to the equal-budget tuning search on Validation 1, which is defined separately. This record only requires that neither library's categorical handling be switched off or bypassed.

### 7. High-cardinality policy

- **All high-cardinality columns stay categorical for every model** (`card1`, `DeviceInfo`, `card2`, `id_19`, `id_20`, `id_21`, `id_25`, `addr1`, `id_33`, the email domains).
- They are not merged, frequency-encoded or dropped: any of these would be feature engineering, and treating them as numeric in one library only would break parity.
- **Documented cautions [I]:**
  - **Entity memorisation.** `card1` has 11,870 Train levels, and 27 % of Train rows fall in levels with fewer than 100 rows. Together with other card and address fields it can let a model partly recognise recurring entities. This is the optimism risk already recorded in DR-001 and in the protocol's threats to validity. The "recurring vs new pseudo-entity" false-negative slice will show its effect.
  - **Non-stationary categories.** `id_13` (51 % of Validation 2 values unseen in Train), `id_31` (41 %), `id_21` (8 %) and `id_25` (6 %) will often become missing values for LightGBM and the neural networks (through the Train-fixed category list and vocabulary) or new values for CatBoost in later periods. This is part of the temporal-transfer problem the study measures, not a defect to remove.
  - **LightGBM guidance.** LightGBM's documentation notes that high-cardinality features often work best treated as numeric [REF-076]. That advice is **not** adopted, because it would contradict the organiser's declaration and break parity. Its effect is instead controlled through LightGBM's categorical-regularisation parameters in the equal-budget tuning.
  - **Computation.** Very large category counts increase LightGBM histogram cost and CatBoost statistic computation. This affects run time, not validity.

### 8. Machine-readable declaration

[`config/features.yaml`](../../config/features.yaml) lists the target, the excluded columns with reasons, all 49 categorical and all 382 numeric column names explicitly, the two derived time-of-day inputs with their source column and rule, and the encoding rules above. It contains **no data values and no fitted statistics**. It was checked against the raw-file headers: all 434 joined columns appear exactly once. The derived inputs are declared separately from the joined columns, so the 434-column source schema is unchanged.

The human-readable counterpart is the [feature catalog](../feature-catalog.md). It documents every column's official meaning (or states that it is masked), observed type, missingness, cardinality, role and cautions, using development data only.

### 9. Population preservation

**Rule.** Data preparation applies only the declared parsing and input-representation conversions of §§4–5. It discards no transaction by content-based filtering, de-duplication or outlier-row removal, and alters no numeric model-input value as corrective cleaning: no clipping, winsorising, numeric imputation or scaling in the canonical frame. Assigning rows to the temporal partitions ([DR-001](DR-001-temporal-split-strategy.md)) is not cleaning. Repeated input vectors are counted descriptively and kept (Issue #86). The neural-network representation transforms values under its own record ([DR-012](DR-012-neural-network-input-representation.md)).

**Why [I].**

- **Workload denominators.** Removing transactions changes daily volumes and the population used for fitting or evaluation, and can change workload and outcome denominators. The fixed daily review budget K uses only the mean daily transaction volume of Validation 2 ([DR-002](DR-002-review-capacity-definition.md)); removing Validation 2 rows changes that input and may change K after rounding, while removal from other partitions does not directly change K.
- **One rule for every period.** Any row filter would need a fixed rule available at prediction time and applied consistently across partitions. Filtering only the retrospective research data, without applying the same rule when scoring transactions, would make the evaluated and intended scoring populations differ.
- **No reliable repair rule.** For masked model inputs, the available semantics do not justify treating an extreme value or a repeated input vector as an error [REF-048, partially verified]. Invalid time values and malformed categorical codes instead stop validation (§§4–5).
- **Fewer data-dependent choices.** Each cleaning decision taken after seeing the data adds a researcher degree of freedom; preservation keeps preparation fixed by the declaration.

## Evidence

### Test-sealing and data use

- Column names were taken from the **raw-file headers**.
- **How the files were read.**
  - The transaction file was read **with every column except `isFraud`**, so Test-period *feature* rows were loaded into memory.
  - The identity file was read in full; it has no label column.
  - Both were then **filtered to development rows** (`TransactionDT` below 155 × 86,400 seconds, i.e. relative days 1–154: Train, Validation 1 and Validation 2) **before any statistic was computed**.
- **No statistic, type observation or check in this record uses Test rows.** Observed types, missingness, cardinality, unseen-category shares, the `D9` check and the rank-correlation screen are all computed on development rows.
- The target column `isFraud` was **excluded from every file read**, so no label (Test or development) was loaded for this record.
- Library behaviour was checked on small **synthetic** data frames, not on project data.

### Joined schema [D]

The transaction table has 394 columns and the identity table 41; they share the join key `TransactionID`. The joined column count is:

$$
\begin{aligned}
N_{\mathrm{joined}} &= N_{\mathrm{tx}} + N_{\mathrm{id}} - 1 \\
&= 394 + 41 - 1 \\
&= 434
\end{aligned}
$$

where:

- `N_joined` = number of columns after the left join of transaction ← identity on `TransactionID`;
- `N_tx`, `N_id` = numbers of columns in `train_transaction.csv` and `train_identity.csv`;
- the `− 1` removes the duplicated join key.

Every joined column is assigned to exactly one role:

$$
\begin{aligned}
N_{\mathrm{joined}} &= N_{\mathrm{target}} + N_{\mathrm{excl}} + N_{\mathrm{cat}} + N_{\mathrm{num}} \\
434 &= 1 + 2 + 49 + 382
\end{aligned}
$$

where:

- `N_target` = 1 (the target `isFraud`);
- `N_excl` = 2 excluded columns;
- `N_cat` = 49 categorical model inputs;
- `N_num` = 382 numeric model inputs.

So the declared input columns number N_cat + N_num = **431**. This matches the feasibility audit's count of 432 raw features, which still included `TransactionDT` (notebook §9). With the two time-of-day inputs (§4), each model receives:

$$
\begin{aligned}
N_{\mathrm{in}} &= N_{\mathrm{cat}} + N_{\mathrm{num}} + N_{\mathrm{tod}} \\
&= 49 + 382 + 2 \\
&= 433
\end{aligned}
$$

where:

- `N_in` = number of model inputs given to every model;
- `N_tod` = 2 time-of-day inputs (`tod_sin`, `tod_cos`).

Observed value types in development data [D]:
- 31 columns are stored as text: `ProductCD`, `card4`, `card6`, `P_emaildomain`, `R_emaildomain`, `M1`–`M9`, `id_12`, `id_15`, `id_16`, `id_23`, `id_27`–`id_31`, `id_33`–`id_38`, `DeviceType`, `DeviceInfo`.
- Every numeric-stored field that the organiser lists as categorical (`card1`–`card3`, `card5`, `addr1`, `addr2`, and `id_13`, `id_14`, `id_17`–`id_22`, `id_24`–`id_26`, `id_32`) is **integer-valued** in development data. Each can therefore be written as a whole-number label without loss.

### Organiser documentation [L]

- The organiser's data description lists the categorical features explicitly [REF-047]:
  - **Transaction:** `ProductCD`, `card1`–`card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `M1`–`M9`;
  - **Identity:** `DeviceType`, `DeviceInfo`, `id_12`–`id_38`.
- The same page states that `TransactionDT` "is a timedelta from a given reference datetime (not an actual timestamp)" [REF-047].
- This text was read in the Internet Archive copy of the Kaggle competition page dated 16 September 2019, because the live page needs a login.
- Field meanings beyond these lists (e.g. card attributes, billing region and country, match flags, "Vesta engineered rich features") come from the organiser's forum post [REF-048]. Its metadata (author "Lynn@Vesta", posted 24 July 2019, "Data Description (Details and Discussion)") was confirmed, but its body could not be read without a login. **Semantics of masked fields are therefore not claimed beyond the organiser's category lists.**

### Library requirements [L; checked on synthetic data]

- **LightGBM** [REF-076]:
  - categorical features must be non-negative integers below Int32.MaxValue; negative values are treated as missing;
  - floating-point values in categorical features are rounded towards 0;
  - for pandas input, integer codes are taken from pandas categoricals, and categories are aligned to those seen in training;
  - at prediction time, **categories not seen during training are treated as missing**;
  - missing values are handled natively by default;
  - for high-cardinality features, the documentation notes that treating them as numeric "often works best".
- **CatBoost** [REF-077]:
  - numeric missing values are handled natively, by default as the minimum value (`nan_mode = Min`);
  - "CatBoost does not process categorical features in any specific way" with respect to missing values;
  - one-hot encoding before training is discouraged;
  - `has_time` uses the order of objects in the input data, without random permutations, when categorical features are transformed and tree structures are chosen; it is supported on CPU and GPU [REF-118].
- **Type check on synthetic data, CatBoost 1.2.10** (installed version):
  - a categorical column containing float values or NaN is rejected with the message "cat_features must be integer or string, real number values and NaN values should be converted to string";
  - integer and string values are accepted, including the pandas 3 default string type.
- **Type check on synthetic data, LightGBM 4.7.0**: a pandas categorical with Train-fixed categories accepts unseen values and missing values, and gives them the same prediction.

### Checks for hidden time or identifier information [D]

- Spearman rank correlation with `TransactionDT` over a 100,000-row development sample:
  - `TransactionID` has |ρ| = 1.000, a disguised timestamp;
  - the next-highest column, `V151`, has |ρ| = 0.324;
  - no other column behaves as a time index.
- `D9` takes 24 fractional values. Stored to six decimal places, it equals (hour of the relative day) ÷ 24 on every development row where it is present (100 %; largest difference 6.95 × 10⁻⁷). It is a bounded, cyclic time-of-day value, not a monotone timestamp.
- **`TransactionDT` is excluded because a monotone time value cannot be extrapolated by trees into a later period.** That reason does not apply to a cyclic hour-of-day value, so `D9` is kept.
- No column has a single constant value in development data.

### Time of day [D; L]

- **[D]** `TransactionDT` has no missing value on any of the 182 relative days (notebook 01 §2.6, all rows, label-free). Its development-row range is 86,400 to 13,391,998 seconds ([feature catalog](../feature-catalog.md)).
- **[D]** `D9` is missing on 87.3 % of development rows and, where present, equals (hour of the relative day) ÷ 24 on 100 % of rows, to its six stored decimal places (above): it carries whole-hour information for a minority of rows only.
- **[L]** Correa Bahnsen et al. model the time of a card transaction as a periodic quantity on a 24-hour clock, at minute resolution, because an arithmetic mean of transaction times ignores the wrap-around at midnight; their features are customer-level von Mises statistics, not the row-local pair used here [REF-007, §3.2].
- **[L]** The scikit-learn documentation encodes a periodic time feature as a sine and cosine pair with the matching period, so that the two features carry the same information without a jump between the end and the start of the period [REF-104].

### High-cardinality and non-stationary categories (Train counts; development rows only) [D]

| Column | Levels in Train | Levels with < 100 Train rows | Share of Train rows in those levels | Validation 1 values unseen in Train | Validation 2 values unseen in Train |
|---|---|---|---|---|---|
| `card1` | 11,870 | 11,370 | 26.9 % | 1.2 % | 1.3 % |
| `DeviceInfo` | 1,470 | 1,443 | 15.8 % | 2.3 % | 3.7 % |
| `card2` | 499 | 234 | 3.7 % | < 0.1 % | < 0.1 % |
| `id_19` | 494 | 434 | 7.0 % | 0.2 % | 0.2 % |
| `id_21` | 401 | 399 | 46.2 % | 8.7 % | 7.8 % |
| `id_20` | 352 | 282 | 2.9 % | 0.3 % | 0.4 % |
| `addr1` | 310 | 250 | 0.5 % | < 0.1 % | < 0.1 % |
| `id_25` | 295 | 292 | 40.7 % | 7.5 % | 6.3 % |
| `id_33` | 166 | 135 | 2.8 % | 2.6 % | 5.1 % |
| `id_31` | 101 | 58 | 1.7 % | 35.1 % | 40.8 % |
| `id_13` | 49 | 31 | 0.5 % | 36.7 % | 51.2 % |
| `P_emaildomain` / `R_emaildomain` | 59 / 60 | 15 / 32 | 0.3 % / 1.1 % | 0 % | 0 % |

"Unseen" shares are computed over non-missing values. The threshold of 100 rows is LightGBM's default `min_data_per_group` (minimum rows per categorical group) [REF-076].

## Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Treat numeric-coded fields (`card1`, `addr1`, `id_*` codes) as numeric | Contradicts the organiser's declaration and imposes an undocumented order on masked labels |
| Treat only high-cardinality fields as numeric (LightGBM's guidance) | A chosen cardinality cut-off would be arbitrary. Applying it to one library only breaks parity; applying it to both removes CatBoost's native categorical handling for exactly the fields where the families differ |
| Merge rare levels / frequency-encode | Feature engineering, outside the accepted scope. Frequency counts would also have to be fitted on Train only, adding a leakage surface |
| Drop sparse identity fields (> 99 % missing) | Missingness is informative and handled natively; dropping is not methodologically required |
| Let each library use its own defaults (e.g. LightGBM auto-detecting pandas categories; CatBoost only on text columns) | Would give the two libraries **different** categorical sets, which the protocol forbids |
| NaN for categorical missing values in LightGBM, a text label in CatBoost | The same information would enter in two forms. A shared explicit label is simpler and identical |
| Exclude `D9` (time-of-day) | It is cyclic and bounded, not a monotone timestamp, so the reason for excluding `TransactionDT` does not apply |
| Declared columns only, without time-of-day inputs | A valid simpler design, not shown to be inferior. Not chosen because `D9`, the declared column identified as partly encoding the hour of the relative day, is missing on 87.3 % of development rows, while `TransactionDT` gives a complete, row-local time of day |
| Whole hour of the relative day (24 values) instead of the continuous angle | Whole-hour boundaries counted from an undisclosed reference point have no documented meaning, and a shift of the reference point that is not a whole number of hours moves transactions between hourly bins. It also discards the time within the hour and allows only 24 possible pairs, with 13 distinct values per coordinate, to the neural networks' quantile transformation |
| Time of day as one linear value (θ or the hour) | Places the end of a relative day far from the start of the next, although they are adjacent in time |
| Periodic spline features [REF-104] | More inputs per period and chosen knots, with no evidence of benefit here |
| An additional seven-day phase | Weekdays are unknown, because the reference point is undisclosed. A relative seven-day phase could be computed, but it adds inputs without a documented weekly meaning |
| Fill or correct `D9` from `TransactionDT` | Would alter an organiser field. `D9` is kept as observed, and its missingness remains information |
| CatBoost's default random permutations (`has_time = false`) | The permutation is the method's artificial time for data without a time order [REF-012, Section 3.2]. Both settings stay within Train, so neither is leakage into the validation or Test periods; the time order is chosen for consistency with the temporal design. It is fixed before any run, not tuned |
| Content-based cleaning: removing repeated or extreme rows; clipping, winsorising or imputing numeric values in the canonical frame | Row removal changes the transaction population and daily volumes; the available semantics do not justify corrective alteration of masked inputs (§9). Tree-based models handle numeric missing values natively (§5); the neural transformation and zero fill are governed by [DR-012](DR-012-neural-network-input-representation.md) |
| Entity-level aggregates (counts, frequencies, past-window statistics) | Depend on heuristic pseudo-entity keys and on assumptions about label availability, and add a leakage surface; outside the scope of this study |

## Consequences

- The implementation must read [`config/features.yaml`](../../config/features.yaml). It must assert that the loaded columns match the declaration exactly and that numeric-coded categorical values are whole numbers.
- LightGBM category lists are built from **Train rows only** and stored with each model.
- The CatBoost training rows are asserted to be in time order (`TransactionDT`, ties by `TransactionID`) before the two columns are dropped.
- Data preparation removes no row and changes no numeric model-input value as cleaning (§9); repeated input vectors are reported, not removed.
- The implementation validates `TransactionDT` and computes `tod_sin` and `tod_cos` as in §4, before the partitions are assigned and the raw time column is dropped; the validation stops data preparation on any invalid value.
- Any later change to feature roles requires a new GitHub Issue. Before the protocol freeze, this record is revised in place; after the freeze, the change requires a record that supersedes this one ([record maintenance policy](README.md#record-maintenance-policy)).

## Threats and limitations

- **Masked semantics.** Beyond the organiser's category lists, the meaning of most fields is undisclosed. The roles follow the organiser; the organiser's forum post [REF-048] was not readable in full.
- **Organiser list taken as authoritative.** Some integer-coded identity fields may be ordinal in reality. Treating them as nominal can cost some efficiency but introduces no leakage.
- **Different unseen-category handling** between the libraries remains. The raw information is identical. For LightGBM the unseen-to-missing mapping is applied in preprocessing by the Train-fixed category list, deliberately reproducing LightGBM's documented default rather than adding any information. CatBoost uses its native handling.
- **The `D9` interpretation** (hour-of-day) is inferred from the data [D], not documented.
- **CatBoost time order.** With one ordering instead of several random permutations, the earliest Train rows have higher-variance target statistics, a property of ordered statistics noted by its authors [REF-012, Section 3.2]. Its effect on predictive performance is not assumed in advance.
- **Relative time of day.** `tod_sin` and `tod_cos` describe the phase within the relative day defined by an undisclosed reference point. They support no claim about local clock times, time zones or calendar effects.
- **No ablation of the time-of-day inputs.** The study does not estimate their incremental contribution. Results are conditional on the fixed 433-input design.
- **Look-back windows** of the C, D and V groups remain undocumented, a known threat to validity (protocol, Leakage Controls).

## References

REF-007, REF-012, REF-047, REF-048\*, REF-076, REF-077, REF-104, REF-118 — see [`../references.md`](../references.md).

\* Partially verified.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [D], [L], [I] | Evidence tags: dataset observation, literature/documentation, inference |
| c | Constant number of seconds added to every recorded `TransactionDT` value |
| `__MISSING__` | Explicit label given to a missing categorical value, identical for all models; reserved in every neural-network vocabulary, where values unseen in Train also map to its index (LightGBM maps them to NaN instead) |
| FT-Transformer, ResNet-style MLP | The neural models of DR-011 (Feature Tokenizer + Transformer; residual multilayer perceptron) |
| MLP | Multilayer perceptron |
| `C1`–`C14`, `D1`–`D15`, `M1`–`M9` | Organiser feature groups: counts, time deltas, match flags |
| `card1`–`card6`, `addr1`, `addr2` | Organiser card and address fields (categorical) |
| `cat_features`, `categorical_feature` | CatBoost and LightGBM arguments naming the categorical columns |
| `cat_l2`, `cat_smooth`, `max_cat_threshold`, `min_data_per_group` | LightGBM categorical-regularisation parameters (tuned elsewhere) |
| Counter, FeatureFreq | CatBoost's label-free category-frequency statistics on CPU and GPU |
| CPU, GPU | Central processing unit; graphics processing unit |
| `counter_calc_method = SkipTest` | CatBoost setting that computes frequency statistics on the training data only |
| `D9` | Organiser time-delta field; equals hour of the relative day ÷ 24 on 100 % of non-missing development rows, to its six stored decimal places |
| `DeviceType`, `DeviceInfo` | Organiser device fields (categorical) |
| DR | Decision Record |
| `float64` | 64-bit floating-point numeric type |
| H1 | Confirmatory hypothesis 1 (model-family effect) |
| `has_time` | CatBoost setting: use the given row order instead of random permutations for ordered target statistics |
| `id_01`–`id_38` | Organiser identity fields; `id_12`–`id_38` categorical, `id_01`–`id_11` numeric |
| `isFraud` | Target: 1 = fraud, 0 = legitimate |
| K | Fixed daily review budget (reviews per relative day), derived from the mean daily transaction volume of Validation 2 (DR-002) |
| Int32.MaxValue | Largest 32-bit signed integer (2,147,483,647) |
| N_joined | Number of columns after the join (434) |
| N_tx, N_id | Numbers of columns in the transaction (394) and identity (41) files |
| N_target, N_excl, N_cat, N_num | Numbers of target (1), excluded (2), categorical (49) and numeric (382) joined columns |
| N_in | Number of model inputs given to every model (433) |
| N_tod | Number of time-of-day inputs (2) |
| NaN | Not-a-number: the missing-value marker for numeric columns |
| `nan_mode` | CatBoost setting for numeric missing values (default Min) |
| `P_emaildomain`, `R_emaildomain` | Purchaser and recipient email domains (categorical) |
| pandas `Categorical` | pandas data type holding a fixed category list and integer codes |
| `ProductCD` | Organiser product code (categorical) |
| ρ | Spearman rank correlation |
| Relative day | `TransactionDT` divided by 86,400 seconds and rounded down (DR-001) |
| RQ1 | Research question 1 (model and weighting) |
| sin, cos | Sine and cosine functions |
| t, t′ | `TransactionDT` value of a transaction, in seconds; t′ = t + c after a shift |
| `t mod 86400` | Remainder of t after division by 86,400: seconds since the start of the relative day |
| `x mod 360°` | Remainder of an angle x after division by 360° (one full turn) |
| `tod_sin`, `tod_cos` | Time-of-day inputs: sine and cosine of θ |
| θ, θ′ | Time-of-day angle: fraction of the relative day elapsed × 360° (0° ≤ θ < 360°); θ′ computed from t′ = t + c |
| `TransactionAmt` | Transaction amount (numeric) |
| `TransactionDT` | Time offset in seconds from an undisclosed reference point (excluded) |
| `TransactionID` | Row identifier and join key (excluded) |
| `use_missing`, `zero_as_missing` | LightGBM missing-value settings (defaults true, false) |
| `V1`–`V339` | Organiser-engineered numeric **feature columns** (distinct from the Validation 1/2 partitions, which this record writes in full) |
| `dist1`, `dist2` | Organiser distance fields (numeric) |
