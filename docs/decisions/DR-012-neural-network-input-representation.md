# DR-012 — Neural-network input representation

| Field | Value |
|---|---|
| Status | **Proposed** for the design review with the supervisor (Issue #56); not yet accepted |
| Date | 2026-10-09 (Issue #54, as DR-008 §9); separate record 2026-10-10 (Issue #57) |
| Evidence source | Primary source of both neural architectures [REF-087] (REF-NNN = reference identifier in `references.md`; arXiv v5, Appendix B.2) and its reference implementation [REF-103]; tabular benchmark evidence on skewed and heavy-tailed features [REF-089]; Train missingness counts of the [feature catalog](../feature-catalog.md) [D]; a synthetic check of the quantile transformation; the graphics processing unit (GPU) memory feasibility probe on synthetic tensors (Issue #50). No labels, model scores or Test-period data were used. |
| Related | [DR-008](DR-008-feature-declaration-and-model-input-policy.md), [DR-011](DR-011-model-set-tree-based-and-neural-families.md), [protocol](../protocol.md), [`config/features.yaml`](../../config/features.yaml) |

## Context

The two neural models of [DR-011](DR-011-model-set-tree-based-and-neural-families.md), a ResNet-style multilayer perceptron (MLP) and FT-Transformer (Feature Tokenizer + Transformer), receive the same 431 declared inputs as the tree-based models, built from one canonical frame ([DR-008](DR-008-feature-declaration-and-model-input-policy.md) §5): categorical values as text labels with missing values labelled `"__MISSING__"`, and numeric values as `float64` with missing values left as NaN (not-a-number). The tree-based libraries apply the conversions in [DR-008](DR-008-feature-declaration-and-model-input-policy.md) §5 and use native categorical and missing-value handling. Neural networks need finite numeric inputs and integer category indices. The proposed representation keeps the information of the canonical frame ([DR-008](DR-008-feature-declaration-and-model-input-policy.md) §5), is fitted on Train rows only and is **identical for ResNet-style MLP and FT-Transformer**. Its rules are attached to **input roles**, not to column names: every categorical input, including any input added later through an approved GitHub Issue, follows the categorical rules, and every numeric input follows the numeric and indicator rules.

## Decision

| Step | ResNet-style MLP and FT-Transformer (identical for both) |
|---|---|
| Categorical vocabularies | Per categorical input, a vocabulary of the canonical labels observed in Train. `"__MISSING__"` is **reserved in every vocabulary**, also for inputs without a missing Train value, so its index always exists. Each label maps to an integer index; a value unseen in Train maps to the `"__MISSING__"` index, the same unseen-to-missing rule that LightGBM applies ([DR-008](DR-008-feature-declaration-and-model-input-policy.md) §5). Each input has a learned embedding inside the model. |
| Category dropout (training only) | During training, each categorical value of a Train row is replaced by `"__MISSING__"` with probability 0.01, independently per value and per epoch, drawn from the declared training seed. The probability is fixed at 0.01 and is not tuned, so it adds no hyperparameter that could differ between the two neural models. This trains the `"__MISSING__"` embedding of every input, so that values unseen in Train reach a learned embedding in Validation 1, Validation 2 and Test. No dropout is applied when scoring Validation 1, Validation 2 or Test. |
| Numeric transformation | Quantile transformation to a normal output distribution, fitted per numeric input on its non-missing Train values, as in the primary source of both architectures [REF-087]. When the quantiles are fitted, Gaussian noise is added to **every** numeric Train value, with a fixed seed; the fitted transformation is then applied to the original, noise-free values of Train, Validation 1, Validation 2 and Test. The noise scale and the number of quantiles follow the reference implementation of REF-087 exactly [REF-103], as defined below the table. This is the device that REF-087 (arXiv v5, Appendix B.2) uses for inputs with few distinct values. |
| Numeric missing values | 0 after the transformation (the centre of the normal output distribution). |
| Missingness indicators | One binary indicator per numeric input with at least one missing Train value, so that the networks receive the missingness information the trees use natively. For the declared inputs this adds about 290 indicators: 264 `V` columns (Train missingness blocks 1–5, 7–11, 13 and 14 of the [feature catalog](../feature-catalog.md)) and 28 other numeric columns [D]. |
| Input count | 431 declared columns + about 290 indicators ≈ 720 inputs. FT-Transformer receives one token per input plus the [CLS] (classification) token. The engineering probe of Issue #50 (synthetic tensors) showed that FT-Transformer training fits on the project GPU up to 800 input tokens, over the width and depth grid of the large-dataset search space of REF-087 (Table 13, space (B)) with the feed-forward size factor fixed at 4/3 ([DR-011](DR-011-model-set-tree-based-and-neural-families.md) §4). At 800 input tokens with bf16 (bfloat16, a 16-bit floating-point format) mixed precision and memory-efficient attention, the largest setting fits at batch size 64; gradient checkpointing raises this to 128. The probe is repeated before the development runs if the input count exceeds 800. |
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
- `s_i` = population standard deviation (ddof = 0, i.e. delta degrees of freedom 0, so the divisor is the number of values; the `np.std` default used by the reference implementation) of the non-missing Train values of numeric input i;
- `n_q` = number of quantiles of each transformation;
- `n_Train` = number of Train rows (368,632), so ⌊n_Train / 30⌋ = 12,287 and the cap of 1,000 applies;
- `⌊ · ⌋` = rounding down to a whole number.

All non-missing Train values are used to fit the quantiles (no subsampling). The training rules fix the remaining implementation values before the development runs: the noise seed and the handling of numeric inputs that are constant or entirely missing in Train.

## Evidence

- **[L]** Both neural architectures were evaluated with a quantile transformation of numeric inputs, with noise added when the quantiles are fitted [REF-087]; the exact noise scale and number of quantiles are taken from the reference implementation at a pinned commit [REF-103].
- **[L]** Neural networks are weaker than gradient-boosted trees on skewed and heavy-tailed features [REF-089].
- **[D]** Numeric inputs with missing Train values: about 290 (264 `V` columns and 28 others), from the Train missingness counts of the [feature catalog](../feature-catalog.md).
- **Synthetic engineering check:** on 100,000 values (60 % zeros, 40 % log-normal), a quantile transformation with n_q = 1,000 mapped 0 to −5.20 without noise and to −0.52 with the reference noise rule. **[I]** Without noise, a dominant repeated value is pushed to an extreme of the output distribution.
- **[I]** The GPU memory probe of Issue #50 (synthetic tensors) bounds the feasible input count for FT-Transformer ([DR-011](DR-011-model-set-tree-based-and-neural-families.md) §4).

## Alternatives considered

| Alternative | Why not proposed |
|---|---|
| Impute missing values without indicators | Discards informative missingness that the trees use natively. |
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
- Numeric-input missingness is informative; per-input indicators preserve that information without engineering new features. Whether the count-based `V` groups share identical missing-row patterns remains to be checked in [Issue #59](https://github.com/chathuranga-sudusinghe/fraud-detection-review-capacity/issues/59).
- One representation for both neural models keeps the within-family comparison clean.

## Consequences

- Vocabularies, quantile transformations and the indicator list are built from **Train rows only** and stored with each model.
- The quantile preprocessing is implemented against the pinned commit of the reference implementation [REF-103]. A unit test on a synthetic input with many tied values checks the noise scale, the number of quantiles and that the transformation is applied to the original values.
- The training rules fix the noise seed and the handling of numeric inputs that are constant or entirely missing in Train before the development runs.
- If the record is accepted, it becomes the neural-network part of the common training procedure in the [protocol](../protocol.md#training-procedure); before then, the protocol lists it as a design item.

## Threats and limitations

- **Proposed, not accepted.** The representation is subject to the design review (Issue #56).
- **Missingness blocks.** The `V` missingness blocks are grouped by their number of missing Train rows; whether they share identical missing-row patterns is checked in [Issue #59](https://github.com/chathuranga-sudusinghe/fraud-detection-review-capacity/issues/59). The indicator count is unaffected, because one indicator is created per numeric input; the block-level contingency in the alternatives depends on the result.
- **Representation differences between families.** Trees and neural networks receive the same information in family-specific forms; these differences are part of the family contrast ([DR-011](DR-011-model-set-tree-based-and-neural-families.md)).

## References

REF-087, REF-089, REF-103 — see [`../references.md`](../references.md).

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [D], [L], [I] | Evidence tags: dataset observation, literature, inference |
| `__MISSING__` | Explicit label for a missing categorical value; reserved in every neural-network vocabulary and the target of values unseen in Train |
| bf16 | bfloat16: 16-bit floating-point format used for mixed-precision neural-network training |
| [CLS] | Classification token of FT-Transformer |
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
| Relative day | `TransactionDT` (time offset in seconds from an undisclosed reference point) divided by 86,400 seconds and rounded down |
| Test | Final partition (relative days 155–182), sealed |
| Train | Model-fitting partition (relative days 1–105) |
| N(0, 1e−3) | The primary source's notation for the quantile-fitting noise: a normal distribution with mean 0 and second parameter 10⁻³ (standard deviation or variance not stated; see Alternatives considered) |
| REF-NNN | Reference identifier in `../references.md` |
| `V1`–`V339` | Organiser-engineered numeric feature columns (distinct from the Validation 1/2 partitions, written in full here) |
| ⌊ · ⌋ | Round down to a whole number |
