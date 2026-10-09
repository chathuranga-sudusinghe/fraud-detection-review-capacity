# DR-008 — Feature declaration and model-input policy

| Field | Value |
|---|---|
| Status | §§1–8 Accepted (pre-freeze) by Issue #23; supervisor review pending. §9 (neural-network representation) **Proposed** for the design review (Issue #54). |
| Date | 2026-09-24; revised 2026-10-09 (Issue #54) |
| Evidence source | Raw-file headers (schema only); development-period values (relative days 1–154) for type, missingness and cardinality checks, **with the target column never loaded**; organiser data description; LightGBM and CatBoost documentation; input-type checks on synthetic data; `notebooks/01_dataset_feasibility_audit.ipynb` §§2.3, 8, 9 |
| Related | [DR-001](DR-001-temporal-split-strategy.md), [DR-002](DR-002-review-capacity-definition.md), [DR-011](DR-011-model-set-tree-based-and-neural-families.md), [protocol](../protocol.md), [`config/features.yaml`](../../config/features.yaml), [feature catalog](../feature-catalog.md) |

## Context

The study compares four models in two families, as research question 1 (RQ1) and hypothesis H1 (model-family effect) require ([DR-011](DR-011-model-set-tree-based-and-neural-families.md)): the gradient-boosting libraries LightGBM and CatBoost, and the neural networks ResNet-style MLP and FT-Transformer. The protocol requires all four to receive **the same declared model inputs** under one common training procedure, so that a difference between the families can be attributed to the model-and-representation pipelines compared, not to unequal data or effort. Each family keeps the representation its models require (§§5, 9).

The declaration is not trivial:
- many nominal (unordered, label-like) fields are stored as numbers (`card1`, `addr1`, several `id_*` fields);
- the models accept categorical data and missing values in **different technical forms**;
- no model may be given extra engineered information.

This record fixes, before any model is implemented: the target, the excluded columns, the categorical columns, the numeric columns, and the exact technical conversions for each model family.

**Scope limits.** No feature is engineered, target-encoded, aggregated or created from pseudo-entity keys. No model is trained or tuned. The temporal split ([DR-001](DR-001-temporal-split-strategy.md)), capacity settings ([DR-002](DR-002-review-capacity-definition.md)), class weighting and hypotheses are defined elsewhere.

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

So each model receives N_cat + N_num = **431 declared input columns**. This matches the feasibility audit's count of 432 raw features, which still included `TransactionDT` (notebook §9).

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
  - one-hot encoding before training is discouraged.
- **Type check on synthetic data, CatBoost 1.2.10** (installed version):
  - a categorical column containing float values or NaN is rejected with the message "cat_features must be integer or string, real number values and NaN values should be converted to string";
  - integer and string values are accepted, including the pandas 3 default string type.
- **Type check on synthetic data, LightGBM 4.7.0**: a pandas categorical with Train-fixed categories accepts unseen values and missing values, and gives them the same prediction.

### Checks for hidden time or identifier information [D]

- Spearman rank correlation with `TransactionDT` over a 100,000-row development sample:
  - `TransactionID` has |ρ| = 1.000, a disguised timestamp;
  - the next-highest column, `V151`, has |ρ| = 0.324;
  - no other column behaves as a time index.
- `D9` takes 24 fractional values. It equals (hour of the relative day) ÷ 24 on 94.9 % of development rows where it is present. It is a bounded, cyclic time-of-day value, not a monotone timestamp.
- **`TransactionDT` is excluded because a monotone time value cannot be extrapolated by trees into a later period.** That reason does not apply to a cyclic hour-of-day value, so `D9` is kept.
- No column has a single constant value in development data.

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

## Decision

### 1. Target

`isFraud` is the target. **It is never a model input**, in any form.

### 2. Excluded columns (2)

| Column | Reason |
|---|---|
| `TransactionID` | Row identifier; monotone in time, i.e. a disguised timestamp [D] |
| `TransactionDT` | Raw time offset. Used only to derive relative days and the temporal split ([DR-001](DR-001-temporal-split-strategy.md)), never as a model input |

**No further column is excluded.**
- High missingness (up to 99.2 %, `id_24`) and high cardinality are **not** reasons for exclusion: every model handles missing values (natively for the trees, through indicators for the neural networks, §9), and the categorical columns are the organiser's own list.
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

### 4. Numeric columns (382) — every remaining joined column

| Group | Columns | Count |
|---|---|---|
| Amount | `TransactionAmt` | 1 |
| Distance | `dist1`, `dist2` | 2 |
| Counts | `C1`–`C14` | 14 |
| Time deltas | `D1`–`D15` (including `D9`; negative values kept) | 15 |
| Organiser-engineered | `V1`–`V339` (the feature columns, not the validation partitions) | 339 |
| Identity numeric | `id_01`–`id_11` | 11 |

**Rule:** a joined column is numeric **if and only if** it is neither the target, nor excluded, nor in the categorical list. Binary-valued V columns remain numeric: they have no organiser categorical label, and a two-valued numeric split is equivalent for trees.

### 5. One canonical input, then the model-specific form

All four models start from **one canonical data frame**, built identically:

1. Drop `isFraud`, `TransactionID` and `TransactionDT`.
2. **Categorical columns → canonical text labels:**
   - text values are kept exactly (no case folding, trimming, merging or grouping of rare levels);
   - integer-valued numeric codes are written as whole-number text (`100.0` → `"100"`, `1000` → `"1000"`);
   - the implementation must **fail loudly** if a numeric-coded categorical value is not a whole number;
   - missing values become the explicit label `"__MISSING__"`.
3. **Numeric columns → `float64`**, with missing values left as NaN. No imputation, scaling, clipping, winsorising or transformation.

Then, for each tree-based library (the neural-network form is in §9):

| Step | LightGBM | CatBoost |
|---|---|---|
| Categorical type | pandas `Categorical`, with the **category list fixed from Train rows only** and reused unchanged for Validation 1, Validation 2 and Test | Canonical text columns passed directly |
| Declared categorical columns | `categorical_feature` = the 49 names | `cat_features` = the 49 names |
| Missing categorical value | The level `"__MISSING__"` (a normal category) | The level `"__MISSING__"` (a normal category) |
| Category unseen in Train | Becomes NaN **during preprocessing**, because it is absent from the Train-fixed pandas category list; LightGBM then treats it as missing. This reproduces LightGBM's own documented predict-time behaviour [REF-076] | Passed as its text label; handled natively by CatBoost as a value not seen in training |
| Numeric missing value | Native (`use_missing = true`, `zero_as_missing = false`; library defaults) | Native (`nan_mode = Min`; library default) |
| Extra encodings | None | None. CatBoost's internal ordered target statistics are part of the model family, not preprocessing [REF-012] |

**Why the missing-value label is explicit for categoricals [I].** CatBoost cannot accept NaN in a categorical column (checked above), so some label is technically required. Using the **same** label in both libraries gives them identical categorical information. For LightGBM a missing value simply becomes one more category it can split on. Numeric missing values stay native in both libraries, as the protocol prefers.

**Why LightGBM categories are fixed from Train [I].** It keeps the category-to-code mapping free of later-period information. It also matches LightGBM's own behaviour of aligning categories to those seen in training [REF-076].

### 6. Parity statement

- **Same semantic input:** all four models receive the same 431 columns, the same 49 categorical declarations, the same canonical category labels and the same numeric values.
- **No model is given engineered information**: no counts, frequencies, target statistics or entity keys are computed outside the models.
- **Family-specific representation.** The trees use native categorical and missing-value handling (§5); the neural networks receive the same information as integer category indices, transformed numeric values and missingness indicators (§9). These forms are part of each family's design and therefore of the family contrast.
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

[`config/features.yaml`](../../config/features.yaml) lists the target, the excluded columns with reasons, all 49 categorical and all 382 numeric column names explicitly, and the encoding rules above. It contains **no data values and no fitted statistics**. It was checked against the raw-file headers: all 434 joined columns appear exactly once.

The human-readable counterpart is the [feature catalog](../feature-catalog.md). It documents every column's official meaning (or states that it is masked), observed type, missingness, cardinality, role and cautions, using development data only.

### 9. Neural-network representation (proposed for the design review)

Neural networks need finite numeric inputs and integer category indices. The proposed representation keeps the information of the canonical frame (§5), is fitted on Train rows only and is **identical for ResNet-style MLP and FT-Transformer**. Its rules are attached to **input roles**, not to column names: every categorical input, including any input added later through an approved GitHub Issue, follows the categorical rules, and every numeric input follows the numeric and indicator rules.

| Step | ResNet-style MLP and FT-Transformer (identical for both) |
|---|---|
| Categorical vocabularies | Per categorical input, a vocabulary of the canonical labels observed in Train. `"__MISSING__"` is **reserved in every vocabulary**, also for inputs without a missing Train value, so its index always exists. Each label maps to an integer index; a value unseen in Train maps to the `"__MISSING__"` index, the same unseen-to-missing rule that LightGBM applies (§5). Each input has a learned embedding inside the model. |
| Category dropout (training only) | During training, each categorical value of a Train row is replaced by `"__MISSING__"` with probability 0.01, independently per value and per epoch, drawn from the declared training seed. The probability is fixed at 0.01 and is not tuned, so it adds no hyperparameter that could differ between the two neural models. This trains the `"__MISSING__"` embedding of every input, so that values unseen in Train reach a learned embedding in Validation 1, Validation 2 and Test. No dropout is applied when scoring Validation 1, Validation 2 or Test. |
| Numeric transformation | Quantile transformation to a normal output distribution, fitted per numeric input on its non-missing Train values, as in the primary source of both architectures [REF-087]. When the quantiles are fitted, Gaussian noise is added to **every** numeric Train value, with a fixed seed; the fitted transformation is then applied to the original, noise-free values of Train, Validation 1, Validation 2 and Test. The noise scale and the number of quantiles follow the reference implementation of REF-087 exactly [REF-103], as defined below the table. This is the device that REF-087 (arXiv v5, Appendix B.2) uses for inputs with few distinct values. |
| Numeric missing values | 0 after the transformation (the centre of the normal output distribution). |
| Missingness indicators | One binary indicator per numeric input with at least one missing Train value, so that the networks receive the missingness information the trees use natively. For the declared inputs this adds about 290 indicators: 264 `V` columns (Train missingness blocks 1–5, 7–11, 13 and 14 of the [feature catalog](../feature-catalog.md)) and 28 other numeric columns [D]. |
| Input count | 431 declared columns + about 290 indicators ≈ 720 inputs. FT-Transformer receives one token per input plus the [CLS] token. The engineering probe of Issue #50 (synthetic tensors) showed that FT-Transformer training fits on the project GPU up to 800 input tokens, over the width and depth grid of the large-dataset search space of REF-087 (Table 13, space (B)) with the feed-forward size factor fixed at 4/3 ([DR-011](DR-011-model-set-tree-based-and-neural-families.md) §4). At 800 input tokens with bf16 mixed precision and memory-efficient attention, the largest setting fits at batch size 64; gradient checkpointing raises this to 128. The probe is repeated before the development runs if the input count exceeds 800. |
| Fitting and reuse | Vocabularies, quantile transformations and the indicator list are fitted on Train only, stored with each model and applied unchanged to Validation 1, Validation 2 and Test. |

The quantile noise and the number of quantiles follow the reference implementation [REF-103]:

$$
\begin{aligned}
\sigma_{\mathrm{noise},i} &= \frac{10^{-3}}{\max\left(s_i,\ 10^{-3}\right)} \\[6pt]
n_{\mathrm{q}} &= \max\left(\min\left(\left\lfloor n_{\mathrm{Train}} / 30 \right\rfloor,\ 1000\right),\ 10\right) = 1000
\end{aligned}
$$

where:

- `σ_noise,i` = standard deviation of the zero-mean Gaussian noise added to numeric input i when its quantiles are fitted;
- `i` = index of a numeric input;
- `s_i` = population standard deviation (ddof = 0, the `np.std` default used by the reference implementation) of the non-missing Train values of numeric input i;
- `n_q` = number of quantiles of each transformation;
- `n_Train` = number of Train rows (368,632), so ⌊n_Train / 30⌋ = 12,287 and the cap of 1,000 applies;
- `⌊ · ⌋` = rounding down to a whole number.

All non-missing Train values are used to fit the quantiles (no subsampling). The training rules fix the remaining implementation values before the development runs: the noise seed and the handling of numeric inputs that are constant or entirely missing in Train.

**Why [I].**
- The quantile transformation is the numeric preprocessing under which both neural architectures were evaluated [REF-087]. It limits the influence of the skewed, heavy-tailed features on which neural networks are known to be weaker [REF-089].
- **Noise for every numeric input.** Many inputs here have few distinct values or a large mass at one value. A synthetic check (100,000 values: 60 % zeros, 40 % log-normal; scikit-learn quantile transformation with n_q = 1,000) mapped 0 to −5.20 without noise and to −0.52 with the reference noise rule: without noise, the most common value is pushed to the extreme of the output distribution. Applying the same noise rule to every numeric input keeps the representation uniform and needs no per-column judgement.
- **Reserved `"__MISSING__"` with category dropout.** A reserved index alone would leave the embedding of an input without missing Train values untrained, so unseen later values would reach an arbitrary initial embedding. Category dropout gives every such embedding training signal at a low rate.
- Missingness is informative in this dataset (whole blocks of `V` columns are missing together); indicators carry it to the networks without engineering new features.
- One representation for both neural models keeps the within-family comparison clean.

**Alternatives for §9.**

| Alternative | Why not proposed |
|---|---|
| Impute missing values without indicators | Discards informative missingness that the trees use natively. |
| A learned "missing" embedding per feature in FT-Transformer only | Would give the two neural models different representations of the same information. |
| Standardisation instead of the quantile transformation | Leaves heavy tails and extreme values that destabilise neural training [REF-089]; departs from the evaluated setting of REF-087. |
| Quantile transformation without noise | Maps a dominant repeated value to an extreme of the output distribution (−5.20 instead of −0.52 in the synthetic check). |
| Noise of a fixed standard deviation 10⁻³ for every input, as a reading of the paper's notation N(0, 1e−3) | The paper's notation does not state whether 10⁻³ is a standard deviation or a variance, and its text refers to the source code for the exact implementation. Following the reference implementation [REF-103] removes this ambiguity and makes the preprocessing reproducible against a pinned commit. |
| Noise only for inputs with few distinct values | Requires a researcher-chosen cut-off per input; the uniform rule is simpler and matches REF-087. |
| `"__MISSING__"` added only where Train has missing values | The index would not exist for other inputs, so the unseen-to-missing rule could not be applied to them. |
| A fixed zero vector, instead of a trained embedding, for unseen values | Gives unseen values no learned representation and differs from how observed missing values are represented. |
| One indicator per missingness block (14 `V` blocks) | Fewer inputs, but the block grouping is a derived feature. Retained as a contingency only if training time requires it, to be decided before the development runs. |

## Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Treat numeric-coded fields (`card1`, `addr1`, `id_*` codes) as numeric | Contradicts the organiser's declaration and imposes an undocumented order on masked labels |
| Treat only high-cardinality fields as numeric (LightGBM's guidance) | A researcher-chosen cardinality cut-off would be arbitrary. Applying it to one library only breaks parity; applying it to both removes CatBoost's native categorical handling for exactly the fields where the families differ |
| Merge rare levels / frequency-encode | Feature engineering, outside the accepted scope. Frequency counts would also have to be fitted on Train only, adding a leakage surface |
| Drop sparse identity fields (> 99 % missing) | Missingness is informative and handled natively; dropping is not methodologically required |
| Let each library use its own defaults (e.g. LightGBM auto-detecting pandas categories; CatBoost only on text columns) | Would give the two libraries **different** categorical sets, which the protocol forbids |
| NaN for categorical missing values in LightGBM, a text label in CatBoost | The same information would enter in two forms. A shared explicit label is simpler and identical |
| Exclude `D9` (time-of-day) | It is cyclic and bounded, not a monotone timestamp, so the reason for excluding `TransactionDT` does not apply |

## Consequences

- The implementation must read [`config/features.yaml`](../../config/features.yaml). It must assert that the loaded columns match the declaration exactly and that numeric-coded categorical values are whole numbers.
- LightGBM category lists, neural-network vocabularies and quantile transformations are built from **Train rows only** and stored with each model.
- The quantile preprocessing is implemented against the pinned commit of the reference implementation [REF-103]. A unit test on a synthetic input with many tied values checks the noise scale, the number of quantiles and that the transformation is applied to the original values.
- Any later change to feature roles requires a new GitHub Issue. Before the protocol freeze, this record is revised in place; after the freeze, the change requires a record that supersedes this one ([record maintenance policy](README.md#record-maintenance-policy)).

## Threats and limitations

- **Masked semantics.** Beyond the organiser's category lists, the meaning of most fields is undisclosed. The roles follow the organiser; the organiser's forum post [REF-048] was not readable in full.
- **Organiser list taken as authoritative.** Some integer-coded identity fields may be ordinal in reality. Treating them as nominal can cost some efficiency but introduces no leakage.
- **Different unseen-category handling** between the libraries remains. The raw information is identical. For LightGBM the unseen-to-missing mapping is applied in preprocessing by the Train-fixed category list, deliberately reproducing LightGBM's documented default rather than adding any information. CatBoost uses its native handling.
- **The `D9` interpretation** (hour-of-day) is inferred from the data [D], not documented.
- **Look-back windows** of the C, D and V groups remain undocumented, a known threat to validity (protocol, Leakage Controls).

## References

REF-012, REF-047, REF-048\*, REF-076, REF-077, REF-087, REF-089, REF-103 — see [`../references.md`](../references.md).

\* Partially verified.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [D], [L], [I] | Evidence tags: dataset observation, literature/documentation, inference |
| `__MISSING__` | Explicit label given to a missing categorical value, identical for all models; reserved in every neural-network vocabulary; also the target of categories unseen in Train for LightGBM and the neural networks |
| bf16 | bfloat16: 16-bit floating-point format used for mixed-precision neural-network training |
| σ_noise,i | Standard deviation of the quantile-fitting noise for numeric input i |
| i | Index of a numeric input |
| s_i | Population standard deviation (ddof = 0) of the non-missing Train values of numeric input i |
| ddof | Delta degrees of freedom: the divisor of a standard deviation is (number of values − ddof) |
| n_q | Number of quantiles per transformation (1,000) |
| n_Train | Number of Train rows (368,632) |
| ⌊ · ⌋ | Round down to a whole number |
| [CLS] | Classification token of FT-Transformer |
| FT-Transformer, ResNet-style MLP | The neural models of DR-011 (Feature Tokenizer + Transformer; residual multilayer perceptron) |
| GPU | Graphics processing unit |
| MLP | Multilayer perceptron |
| `C1`–`C14`, `D1`–`D15`, `M1`–`M9` | Organiser feature groups: counts, time deltas, match flags |
| `card1`–`card6`, `addr1`, `addr2` | Organiser card and address fields (categorical) |
| `cat_features`, `categorical_feature` | CatBoost and LightGBM arguments naming the categorical columns |
| `cat_l2`, `cat_smooth`, `max_cat_threshold`, `min_data_per_group` | LightGBM categorical-regularisation parameters (tuned elsewhere) |
| `D9` | Organiser time-delta field; equals hour of the relative day ÷ 24 on 94.9 % of non-missing development rows |
| `DeviceType`, `DeviceInfo` | Organiser device fields (categorical) |
| DR | Decision Record |
| `float64` | 64-bit floating-point numeric type |
| H1 | Confirmatory hypothesis 1 (model-family effect) |
| `id_01`–`id_38` | Organiser identity fields; `id_12`–`id_38` categorical, `id_01`–`id_11` numeric |
| `isFraud` | Target: 1 = fraud, 0 = legitimate |
| Int32.MaxValue | Largest 32-bit signed integer (2,147,483,647) |
| N_joined | Number of columns after the join (434) |
| N_tx, N_id | Numbers of columns in the transaction (394) and identity (41) files |
| N_target, N_excl, N_cat, N_num | Numbers of target (1), excluded (2), categorical (49) and numeric (382) columns |
| NaN | Not-a-number: the missing-value marker for numeric columns |
| `nan_mode` | CatBoost setting for numeric missing values (default Min) |
| `P_emaildomain`, `R_emaildomain` | Purchaser and recipient email domains (categorical) |
| pandas `Categorical` | pandas data type holding a fixed category list and integer codes |
| `ProductCD` | Organiser product code (categorical) |
| ρ | Spearman rank correlation |
| RQ1 | Research question 1 (model and weighting) |
| `TransactionAmt` | Transaction amount (numeric) |
| `TransactionDT` | Time offset in seconds from an undisclosed reference point (excluded) |
| `TransactionID` | Row identifier and join key (excluded) |
| `use_missing`, `zero_as_missing` | LightGBM missing-value settings (defaults true, false) |
| `V1`–`V339` | Organiser-engineered numeric **feature columns** (distinct from the Validation 1/2 partitions, which this record writes in full) |
| `dist1`, `dist2` | Organiser distance fields (numeric) |
