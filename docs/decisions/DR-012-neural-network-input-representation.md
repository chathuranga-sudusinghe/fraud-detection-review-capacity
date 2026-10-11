# DR-012 — Neural-network input representation

| Field | Value |
|---|---|
| Status | **Proposed**; not yet accepted (protocol design item 2, owner decision pending) |
| Date | 2026-10-09 (Issue #54, as DR-008 §9); separate record 2026-10-10 (Issue #57); revised 2026-10-10 (Issue #59: indicator count from notebook 04) and 2026-10-10 (Issue #64: time-of-day inputs; numeric-transformation specification) |
| Evidence source | Primary source of both neural architectures [REF-087] (REF-NNN = reference identifier in `references.md`; arXiv v5, Appendix B.2) and its reference implementation [REF-103]; scikit-learn documentation of the quantile transformation [REF-105]; tabular benchmark evidence on skewed and heavy-tailed features [REF-089]; Train missingness counts and patterns ([notebook 04](../../notebooks/04_development_data_corrections.ipynb)) [D]; Train value counts of the [feature catalog](../feature-catalog.md) [D]; synthetic checks of the quantile transformation; the graphics processing unit (GPU) memory feasibility probe on synthetic tensors (Issue #50). No labels, model scores or Test-period data were used. |
| Related | [DR-008](DR-008-feature-declaration-and-model-input-policy.md), [DR-011](DR-011-model-set-tree-based-and-neural-families.md), [protocol](../protocol.md), [`config/features.yaml`](../../config/features.yaml) |

## Context

The two neural models of [DR-011](DR-011-model-set-tree-based-and-neural-families.md), a ResNet-style multilayer perceptron (MLP) and FT-Transformer (Feature Tokenizer + Transformer), receive the same 433 model inputs as the tree-based models (the 431 declared columns and the two time-of-day inputs `tod_sin` and `tod_cos`, [DR-008](DR-008-feature-declaration-and-model-input-policy.md) §4), built from one canonical frame ([DR-008](DR-008-feature-declaration-and-model-input-policy.md) §5): categorical values as text labels with missing values labelled `"__MISSING__"`, and numeric values as `float64` with missing values left as NaN (not-a-number). The tree-based libraries apply the conversions in [DR-008](DR-008-feature-declaration-and-model-input-policy.md) §5 and use native categorical and missing-value handling. Neural networks need finite numeric inputs and integer category indices. The proposed representation keeps the information of the canonical frame ([DR-008](DR-008-feature-declaration-and-model-input-policy.md) §5), is fitted on Train rows only and is **identical for ResNet-style MLP and FT-Transformer**. Its rules are attached to **input roles**, not to column names: every categorical input, including any input added later through an approved GitHub Issue, follows the categorical rules, and every numeric input follows the numeric and indicator rules.

## Decision

| Step | ResNet-style MLP and FT-Transformer (identical for both) |
|---|---|
| Categorical vocabularies | Per categorical input, a vocabulary of the canonical labels observed in Train. `"__MISSING__"` is **reserved in every vocabulary**, also for inputs without a missing Train value, so its index always exists. Each label maps to an integer index; a value unseen in Train maps to the `"__MISSING__"` index, so a missing value and an unseen value share one index. LightGBM also treats an unseen value as missing, but as NaN, separate from its `"__MISSING__"` category; CatBoost treats it as a new value ([DR-008](DR-008-feature-declaration-and-model-input-policy.md) §5). Each input has a learned embedding inside the model. |
| Category dropout (training only) | During training, each categorical value of a Train row is replaced by `"__MISSING__"` with probability 0.01, independently per value and per epoch, drawn from the declared training seed. The probability is fixed at 0.01 and is not tuned, so it adds no hyperparameter that could differ between the two neural models. This trains the `"__MISSING__"` embedding of every input, so that values unseen in Train reach a learned embedding in Validation 1, Validation 2 and Test. No dropout is applied when scoring Validation 1, Validation 2 or Test. |
| Numeric transformation | Quantile transformation to a normal output distribution, as in the primary source of both architectures [REF-087], fitted per numeric input on its **non-missing** Train values. When the quantiles are fitted, Gaussian noise is added to every non-missing Train value; the fitted transformation is then applied to the original, noise-free values of Train, Validation 1, Validation 2 and Test. The noise-scaling formula and the number-of-quantiles rule follow the reference implementation of REF-087 [REF-103], as defined below the table; as part of the adaptation described next, s_i is computed from the non-missing Train values, not from the reference implementation's mean-filled Train array; noise at fitting is the device that REF-087 (arXiv v5, Appendix B.2) uses for inputs with few distinct values. **Adaptation of the reference implementation:** REF-103 fills missing numeric values with the Train mean **before** fitting, so the mean enters the fit and the noise scale, and it adds no missingness indicator. Here missing values are left out of the fit and represented by indicators (below), so that missingness is neither placed at an artificial mass at the mean nor hidden from the network. The time-of-day inputs follow this rule unchanged, like every other numeric input. |
| Numeric missing values | 0 after the transformation (the centre of the normal output distribution). |
| Missingness indicators | One binary indicator per numeric input with at least one missing Train value, so that the networks receive the missingness information the trees use natively. For the declared inputs this adds 324 indicators: 296 `V` columns (Train missingness blocks 1–5 and 7–14 of the [feature catalog](../feature-catalog.md)) and 28 other numeric columns [D; [notebook 04](../../notebooks/04_development_data_corrections.ipynb) §6]. |
| Input count | 433 model inputs + 324 indicators = 757 inputs. FT-Transformer receives one feature token per input (757) plus the [CLS] (classification) token, a sequence of 758 tokens. The engineering probe of Issue #50 (synthetic tensors) counted feature tokens without [CLS] and tested up to 800 (a sequence of 801), so 757 feature tokens lie within the tested range. The probe showed that FT-Transformer training fits on the project GPU up to 800 feature tokens, over the width and depth grid of the large-dataset search space of REF-087 (Table 13, space (B)) with the feed-forward size factor fixed at 4/3 ([DR-011](DR-011-model-set-tree-based-and-neural-families.md) §4). At 800 feature tokens with bf16 (bfloat16, a 16-bit floating-point format) mixed precision and memory-efficient attention, the largest setting fits at batch size 64; gradient checkpointing raises this to 128. The probe measured peak memory over three training steps, not the memory use or training time of the final implementation. It is repeated before the development runs if the feature-token count exceeds 800. |
| Fitting and reuse | Vocabularies, quantile transformations and the indicator list are fitted on Train only, stored with each model and applied unchanged to Validation 1, Validation 2 and Test. |

The noise-scaling formula and the number-of-quantiles rule follow the reference implementation [REF-103], with s_i adapted to the non-missing Train values:

$$
\begin{aligned}
\sigma_{\mathrm{noise},i} &= \frac{10^{-3}}{\max\left(s_i,\ 10^{-3}\right)} \\[6pt]
n_{\mathrm{q}} &= \max\left(\min\left(\left\lfloor n_{\mathrm{Train}} / 30 \right\rfloor,\ 1000\right),\ 10\right) = 1000
\end{aligned}
$$

where:

- `σ_noise,i` = standard deviation of the zero-mean Gaussian noise added to numeric input i when its quantiles are fitted;
- `i` = index of a numeric input;
- `s_i` = population standard deviation (ddof = 0, i.e. delta degrees of freedom 0, so the divisor is the number of values; the `np.std` default used by the reference implementation) of the non-missing Train values of numeric input i; the reference implementation computes it on the Train values after filling missing values with the Train mean, so for an input with missing values the resulting noise level can differ;
- `n_q` = number of quantiles of each transformation;
- `n_Train` = number of Train rows (368,632), so ⌊n_Train / 30⌋ = 12,287 and the cap of 1,000 applies;
- `⌊ · ⌋` = rounding down to a whole number.

All non-missing Train values are used to fit the quantiles (no subsampling). The remaining cases are fixed as follows:

- **Noise seed.** The noise is drawn from one fixed seed, independent of the model-training seeds, so the fitted transformation is identical for all four neural configurations. The seed value is fixed in the training rules.
- **Inputs constant in Train.** No special rule is needed. For an input with a single observed Train value (`V107` [D; [feature catalog](../feature-catalog.md)]), s_i = 0, so the noise standard deviation is 1; the transformation then maps the observed value to about 0 and a later value to about its distance from it, within the output bounds (synthetic check under Evidence).
- **Inputs entirely missing in Train.** Such an input cannot be fitted, and data preparation stops with an error. No declared numeric input is entirely missing in Train [D; [feature catalog](../feature-catalog.md): every numeric input has at least one observed Train value].
- **Values outside the fitted range.** The fitted range of an input is the range of its noise-perturbed, non-missing Train values, which extends slightly beyond the range of the original Train values. Values below or above the fitted range are mapped to the bounds of the output distribution [REF-105] (±5.199 in the synthetic check). The share of such values per input in Validation 1 and Validation 2, measured against the endpoints of the fitted transformation, is reported as a diagnostic.

## Evidence

- **[L]** Both neural architectures were evaluated with a quantile transformation of numeric inputs, with noise added when the quantiles are fitted [REF-087]; the noise-scaling formula and the number-of-quantiles rule are taken from the reference implementation at a pinned commit [REF-103].
- **[L]** Neural networks are weaker than gradient-boosted trees on skewed and heavy-tailed features [REF-089].
- **[D]** Numeric inputs with at least one missing Train value: 324 (296 `V` columns and 28 others), counted on Train rows in [notebook 04](../../notebooks/04_development_data_corrections.ipynb) §6.
- **Synthetic engineering check:** on 100,000 values (60 % zeros, 40 % log-normal), a quantile transformation with n_q = 1,000 mapped 0 to −5.20 without noise and to −0.52 with the reference noise rule. **[I]** Without noise, a dominant repeated value is pushed to an extreme of the output distribution.
- **[L]** The reference implementation fills missing numeric values with the Train mean (`num_nan_policy = 'mean'`) before the normalisation function adds the noise and fits the quantile transformation; it adds no missingness indicator [REF-103, `lib/data.py`].
- **[L]** The scikit-learn quantile transformation is applied to each input independently and is non-linear; values outside the fitted range are mapped to the bounds of the output distribution; missing values are disregarded in fitting [REF-105].
- **Synthetic checks (scikit-learn 1.9.1, 368,632 rows, n_q = 1,000, reference noise rule):**
  - a constant input (all values 0) was fitted with noise standard deviation 1 and mapped 0 → 0.00, 1 → 1.00, −1 → −1.00 and 10 → 5.199 (the output bound);
  - the time-of-day pair (`tod_sin`, `tod_cos`: sine and cosine of θ, the fraction of the relative day elapsed × 360°; [DR-008](DR-008-feature-declaration-and-model-input-policy.md) §4), fitted on uniformly distributed angles θ, kept a distinct transformed pair for each of eight test angles 45° apart, while the transformed pairs no longer lay on a circle (e.g. θ = 0° → (0.00, 2.45), θ = 45° → (0.68, 0.67)).
- **[D]** 60 numeric inputs (58 declared columns and the two time-of-day inputs) have no missing Train value and therefore no indicator (384 numeric inputs − 324 with indicators).
- **[I]** The GPU memory probe of Issue #50 (synthetic tensors) bounds the feasible input count for FT-Transformer ([DR-011](DR-011-model-set-tree-based-and-neural-families.md) §4).

## Alternatives considered

| Alternative | Why not proposed |
|---|---|
| Impute missing values without indicators (the reference implementation fills with the Train mean before fitting [REF-103]) | Discards informative missingness that the trees use natively, and places every missing value at the mean inside the fit. |
| One indicator for every numeric input (384) | Would also cover inputs complete in Train but missing later. It gives 817 inputs, above the 800 feature tokens tested in Issue #50, and adds 60 indicators that are constant in Train. |
| A separate index for values unseen in Train | No Train value is unseen, so the index would be trained only through category dropout, adding a second dropout target; the `"__MISSING__"` index already receives that training. |
| Exempting the time-of-day inputs from the quantile transformation | Would make them the only numeric inputs handled differently within the family. They follow the same rule as every other numeric input. |
| A learned "missing" embedding per feature in FT-Transformer only | Would give the two neural models different representations of the same information. |
| Standardisation instead of the quantile transformation | Leaves heavy tails and extreme values that destabilise neural training [REF-089]; departs from the evaluated setting of REF-087. |
| Quantile transformation without noise | Maps a dominant repeated value to an extreme of the output distribution (−5.20 instead of −0.52 in the synthetic check). |
| Noise of a fixed standard deviation 10⁻³ for every input, as a reading of the paper's notation N(0, 1e−3) (a normal distribution with mean 0 and second parameter 10⁻³) | The paper's notation does not state whether 10⁻³ is a standard deviation or a variance, and its text refers to the source code for the exact implementation. Following the reference implementation [REF-103] removes this ambiguity and makes the preprocessing reproducible against a pinned commit. |
| Noise only for inputs with few distinct values | Requires a researcher-chosen cut-off per input; the uniform rule is simpler and matches REF-087. |
| `"__MISSING__"` added only where Train has missing values | The index would not exist for other inputs, so the unseen-to-missing rule could not be applied to them. |
| A fixed zero vector, instead of a trained embedding, for unseen values | Gives unseen values no learned representation and differs from how observed missing values are represented. |
| One indicator per missingness block (14 `V` blocks) | Fewer inputs, but the block grouping is a derived feature. Retained as a contingency only if training time requires it, to be decided before the development runs. |

## Rationale

The reasons below are inferences **[I]** from the evidence above.

- The quantile transformation is the numeric preprocessing under which both neural architectures were evaluated [REF-087]. It limits the influence of the skewed, heavy-tailed features on which neural networks are known to be weaker [REF-089].
- **Noise for every numeric input.** Many inputs here have few distinct values or a large mass at one value. A synthetic check (100,000 values: 60 % zeros, 40 % log-normal; scikit-learn quantile transformation with n_q = 1,000) mapped 0 to −5.20 without noise and to −0.52 with the reference noise rule: without noise, the most common value is pushed to the extreme of the output distribution. Applying the same noise rule to every numeric input keeps the representation uniform and needs no per-column judgement.
- **Reserved `"__MISSING__"` with category dropout.** A reserved index alone would leave the embedding of an input without missing Train values untrained, so unseen later values would reach an arbitrary initial embedding. Category dropout gives every such embedding training signal at a low rate.
- Numeric-input missingness is informative; per-input indicators preserve that information without engineering new features. Each of the 14 `V` missingness blocks shares one missing-row pattern on Train rows ([notebook 04](../../notebooks/04_development_data_corrections.ipynb) §6).
- One representation for both neural models keeps the within-family comparison clean.

## Consequences

- Vocabularies, quantile transformations and the indicator list are built from **Train rows only** and stored with each model.
- The quantile preprocessing is implemented against the pinned commit of the reference implementation [REF-103]. A unit test on a synthetic input with many tied values checks the noise scale, the number of quantiles and that the transformation is applied to the original values.
- The training rules fix the value of the noise seed before the development runs.
- Unit tests on synthetic data also cover: an input constant in Train; the error for an input entirely missing in Train; values outside the fitted range; and the time-of-day pair over the whole cycle, including the wrap-around at the end of the relative day, with finite outputs and identical results after the fitted transformation is saved and reloaded.
- Diagnostics reported for Validation 1 and Validation 2: the share of values outside the fitted range (the endpoints of the fitted transformation) per numeric input, and the number of missing values in inputs that have no indicator.
- If the record is accepted, it becomes the neural-network part of the common training procedure in the [protocol](../protocol.md#training-procedure); before then, the protocol lists it as a design item.

## Threats and limitations

- **Proposed, not accepted.** The representation awaits the project owner's decision (protocol design item 2).
- **Representation differences between families.** Trees and neural networks receive the same information in family-specific forms; these differences are part of the family contrast ([DR-011](DR-011-model-set-tree-based-and-neural-families.md)). The comparison is therefore between model-and-representation pipelines, not between architectures alone.
- **Missing and unseen categories share an index** in the neural networks, whereas LightGBM and CatBoost keep them apart (Decision). The raw information is identical; its form is part of the family contrast.
- **Missingness that first appears after Train.** An input without missing Train values has no indicator, so a later missing value is set to 0 with no indicator, the centre of the output distribution. The diagnostic above reports how often this occurs.
- **Saturation.** Values beyond the fitted range (of the noise-perturbed Train values) are mapped to the output bounds, so differences among them are lost. The diagnostic above reports how often this occurs.
- **Time-of-day geometry.** The two time-of-day inputs are transformed separately. Each transformation preserves the order of its input and is the same for every relative day, so the daily periodicity is kept; but the pair no longer lies on a circle and the distances between times of day change. Order preservation does not by itself guarantee that every time of day keeps a distinct pair: flat segments of a fitted transformation or saturation could merge nearby values. The synthetic check shows distinct pairs for the angles tested only.

## References

REF-087, REF-089, REF-103, REF-105 — see [`../references.md`](../references.md).

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [D], [L], [I] | Evidence tags: dataset observation, literature, inference |
| `__MISSING__` | Explicit label for a missing categorical value; reserved in every neural-network vocabulary and the target of values unseen in Train |
| bf16 | bfloat16: 16-bit floating-point format used for mixed-precision neural-network training |
| [CLS] | Classification token of FT-Transformer, added to the feature tokens |
| Feature token | FT-Transformer token for one model input (757); the sequence adds [CLS] (758) |
| ddof | Delta degrees of freedom: the divisor of a standard deviation is (number of values − ddof) |
| DR-NNN | Decision Record NNN |
| `float64` | 64-bit floating-point numeric type |
| FT-Transformer | Feature Tokenizer + Transformer (a neural model for tabular data) |
| GPU | Graphics processing unit |
| i | Index of a numeric input |
| MLP | Multilayer perceptron |
| n_q | Number of quantiles per transformation (1,000) |
| n_Train | Number of Train rows (368,632) |
| NaN | Not-a-number: the missing-value marker for numeric values |
| σ_noise,i | Standard deviation of the quantile-fitting noise for numeric input i |
| s_i | Population standard deviation (ddof = 0) of the non-missing Train values of numeric input i |
| `num_nan_policy` | Reference-implementation setting for missing numeric values (`'mean'`: fill with the Train mean) |
| Relative day | `TransactionDT` (time offset in seconds from an undisclosed reference point) divided by 86,400 seconds and rounded down |
| Test | Final partition (relative days 155–182), sealed |
| `tod_sin`, `tod_cos` | Time-of-day inputs: sine and cosine of θ (DR-008 §4) |
| θ | Time-of-day angle: fraction of the relative day elapsed × 360° (DR-008 §4) |
| Train | Model-fitting partition (relative days 1–105) |
| `V107` | Organiser numeric column with a single observed Train value |
| Validation 1, Validation 2 | Tuning partition (relative days 106–126); calibration and threshold partition (relative days 127–154) |
| N(0, 1e−3) | The primary source's notation for the quantile-fitting noise: a normal distribution with mean 0 and second parameter 10⁻³ (standard deviation or variance not stated; see Alternatives considered) |
| REF-NNN | Reference identifier in `../references.md` |
| `V` columns | Organiser-engineered numeric feature columns `V1`–`V339` (distinct from the Validation 1/2 partitions, written in full here) |
| ⌊ · ⌋ | Round down to a whole number |
