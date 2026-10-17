# DR-011 — Model set: tree-based and neural model families

| Field | Value |
|---|---|
| Status | Accepted (pre-freeze) by the project owner (Issue #54) in line with the University of Greenwich criteria ([criteria checklist](../greenwich-criteria-checklist.md)); supervisor feedback welcome, not a precondition; revision for the LightGBM training device pending review (Issue #101) |
| Date | 2026-10-09 (Issue #54); revised 2026-10-10 (Issue #64: 433 model inputs), 2026-10-10 (Issue #60: link to the probe results), 2026-10-12 (Issue #66: training device and main seed, [DR-013](DR-013-training-device-policy.md)), 2026-10-15 (Issue #101: LightGBM training device, [DR-013](DR-013-training-device-policy.md)); 2026-10-17 (Issue #115: logistic-regression reference baseline, §5) |
| Evidence source | Structured literature search and evidence matrix (Issue #52: [`literature-search-method.md`](../literature-search-method.md), [`literature-evidence-matrix.md`](../literature-evidence-matrix.md)); GPU memory feasibility probe on synthetic inputs (Issue #50, `scripts/gpu_feasibility_probe.py`; results table in [`evidence/gpu_probe/`](../../evidence/gpu_probe/README.md)). For §5: the logistic-regression use of REF-078 and REF-089 (TabZilla code, REF-130), scikit-learn 1.9.1 sources and documentation [REF-135] and an independent review in Issue #115. No labels, model scores or Test-period data were used. |
| Related | [DR-005](DR-005-sesoi-and-h1-h2-inference.md), [DR-006](DR-006-bootstrap-inference-strategy.md), [DR-008](DR-008-feature-declaration-and-model-input-policy.md), [DR-010](DR-010-h3-confirmatory-configuration.md), [DR-012](DR-012-neural-network-input-representation.md), [DR-013](DR-013-training-device-policy.md), [protocol](../protocol.md) |

## Context

Research question 1 (RQ1) asks how model family and class weighting affect fraud capture at a fixed daily review budget on a later temporal period. The model set therefore defines what "model family" means in hypothesis H1 (model-family effect), which configurations H2 (class-weighting effect) covers, and the selection pool of H3 (alert-rate transfer fidelity, [DR-010](DR-010-h3-confirmatory-configuration.md)).

The family question that remains open in the literature for tabular data is **gradient-boosted decision trees (GBDTs) versus neural networks** (see Evidence).

## Decision

### 1. Model set: 2 × 2 × 2 = 8 configurations

| Family | Models | Class weighting |
|---|---|---|
| **Tree-based (T)** | LightGBM (LGB) [REF-011]; CatBoost (Cat) [REF-012] | unweighted (u); balanced (b) |
| **Neural (N)** | ResNet-style multilayer perceptron, MLP (RES); Feature Tokenizer + Transformer, FT-Transformer (FTT) [REF-087] | unweighted (u); balanced (b) |

Each configuration is written model-weighting, e.g. LGB-u or FTT-b. The eight configurations, in the fixed order used for deterministic tie-breaks, are:

LGB-u, Cat-u, RES-u, FTT-u, LGB-b, Cat-b, RES-b, FTT-b.

### 2. Architectures

The neural architectures follow their primary source [REF-087] (main text Sections 3.2–3.3; arXiv v5 Appendix E.1 and F.1):

- **ResNet-style MLP.** A linear input layer, a stack of residual blocks and a prediction head:

  $$
  \begin{aligned}
  \operatorname{ResNetBlock}(x) &= x + \operatorname{Dropout}\big(\operatorname{Linear}(\operatorname{Dropout}(\operatorname{ReLU}(\operatorname{Linear}(\operatorname{BatchNorm}(x)))))\big) \\
  \operatorname{Prediction}(x) &= \operatorname{Linear}\big(\operatorname{ReLU}(\operatorname{BatchNorm}(x))\big)
  \end{aligned}
  $$

  where:

  - `x` = the hidden representation entering a block;
  - `Linear`, `ReLU`, `BatchNorm`, `Dropout` = affine layer, rectified linear unit, batch normalisation and dropout.

- **FT-Transformer.** A feature tokenizer maps every model input to an embedding, a [CLS] (classification) token is appended, a stack of pre-normalisation (PreNorm) Transformer layers with multi-head self-attention and a ReGLU (rectified gated linear unit) feed-forward module is applied, and the prediction is made from the final [CLS] representation.

The tree-based models use the native categorical and missing-value handling of each library ([DR-008](DR-008-feature-declaration-and-model-input-policy.md)).

### 3. Common procedure (comparison fairness)

Fairness is defined as **one common procedure for all eight configurations**, not identical hyperparameter values. The study does not alter any model's internal computation. Every choice the study controls is made identical at the widest level the models permit, under a three-level rule:

| Level | Scope | Elements |
|---|---|---|
| **1. Identical for all eight configurations** | Every choice that does not depend on the model | Data partitions: Train for fitting; Validation 1 (V1) for tuning and early stopping; Validation 2 (V2) for calibration, thresholds, the daily budget K and the H3 configuration ([DR-001](DR-001-temporal-split-strategy.md)). Model inputs: the same 433 inputs (the 431 declared columns and the two time-of-day inputs) from one canonical frame, and any input added later through an approved GitHub Issue enters all eight configurations ([DR-008](DR-008-feature-declaration-and-model-input-policy.md)). Class weights: the same explicit Train-only sample weights. Hyperparameter search: the same search method and the same budget per configuration. V1 objective, early-stopping rule and seeds. Scores: raw model outputs (fraud scores) are used for ranking by every decision policy; calibration is fitted on V2 for R4 only. Decision policies, capacity settings and evaluation. |
| **2. Identical within each family** | Choices that must follow the input form of the family | Tree-based (both models): native categorical and missing-value handling ([DR-008](DR-008-feature-declaration-and-model-input-policy.md) §5); weighted log-loss; the device settings of [DR-013](DR-013-training-device-policy.md). Neural (both models): the representation of [DR-012](DR-012-neural-network-input-representation.md) (vocabularies with reserved `"__MISSING__"` and category dropout, quantile transformation with fitting noise, missingness indicators); weighted binary cross-entropy; bf16 mixed-precision training and float32 prediction on the GPU ([DR-013](DR-013-training-device-policy.md)). |
| **3. Model-specific** | Components that define each model | Architecture and internal computation: self-attention (FT-Transformer), residual blocks with BatchNorm (ResNet-style MLP), ordered target statistics (CatBoost), leaf-wise growth and gradient-sorted categorical splits (LightGBM). Training device ([DR-013](DR-013-training-device-policy.md) §1): LightGBM on the central processing unit (CPU), fixed by a defect of LightGBM's CUDA learner; CatBoost and both neural models on the GPU. Each model's own search space (for the neural models, the large-dataset spaces (B) of REF-087; §4). Memory-efficient attention applies to FT-Transformer only. These components are tuned under the Level 1 search method and budget. |

All eight configurations use the main seed 42, with a common list of additional seeds that measures stochastic variation only ([DR-013](DR-013-training-device-policy.md) §6). The values of the remaining Level 1 elements (search method, budget, objective, patience, the list of additional seeds) are fixed in the training rules before the development runs.

### 4. Training environment

- **Device.** CatBoost and both neural models (six configurations) train on the local GPU (NVIDIA RTX 5060 Laptop, 8 GB) through CUDA (NVIDIA's GPU computing platform). The two LightGBM configurations train on the local CPU with deterministic training, following the documented failure of the pinned CUDA learner on categorical features with more than 256 categories ([DR-013](DR-013-training-device-policy.md) §2). The device-sensitive settings of each model (LightGBM's build, bins, dataset construction and determinism; CatBoost's quantisation, bootstrap type and categorical statistics; the neural models' precision, determinism, attention backend and float32 prediction), the reproducibility rules and the supportive CPU–GPU comparison of CatBoost are fixed in [DR-013](DR-013-training-device-policy.md).
- **Neural models:** **bf16 mixed-precision training** (bfloat16 computation; float32 weights and optimiser state) for both neural models, and **memory-efficient attention** for FT-Transformer.
- **Neural search spaces.** REF-087 (arXiv v5: Table 13 in Appendix E.2 for FT-Transformer; Table 14 in Appendix F.1 for the ResNet-style MLP) defines a smaller-dataset space (A) and a large-dataset space (B) for each neural model. The neural models use the **large-dataset spaces (B)**, the spaces the authors applied to datasets of this size: Train has 368,632 rows, and the authors' Covertype dataset, tuned in space (B) for both models, has 371,847 training rows. Space (B) gives FT-Transformer 1–6 layers, embedding size 64–512 and a fixed feed-forward size factor of 4/3, and the ResNet-style MLP 1–16 layers and layer size 64–1,024. The remaining search details are fixed in the training rules.
- **Engineering basis.** The probe of Issue #50 measured peak GPU memory on synthetic tensors (three training steps per successful batch-size attempt; no data read; full results in [`evidence/gpu_probe/gpu_probe_full.csv`](../../evidence/gpu_probe/gpu_probe_full.csv)) over the width and depth grid of space (B) of FT-Transformer, with the feed-forward size factor fixed at 4/3 as in that space:
  - at 431 input tokens, bf16 with memory-efficient attention fits the default setting (embedding size 192, 3 layers) at batch size 512 and the largest setting (embedding size 512, 6 layers) at batch size 128;
  - float32 with standard attention runs out of memory at 4–6 layers of the largest embedding size; float32 with memory-efficient attention fits the largest setting at batch size 64 only;
  - at 800 input tokens, bf16 with memory-efficient attention fits the largest setting at batch size 64; adding gradient checkpointing raises the feasible batch size to 512 for the default setting and 128 for the largest;
  - the ResNet-style MLP was measured at 431 inputs, layer size 1,024, 16 layers, hidden-size factor 2 and float32, and fits at batch size 4,096 (the probe's batch cap) with a peak of 2.9 GB. Training in bf16 uses less memory than this float32 measurement; hidden-size factors above 2 (space (B) allows up to 4) were not measured.

  bf16 with memory-efficient attention is therefore the chosen setting. Batch size and gradient checkpointing are fixed in the training rules within these measured limits. Batch limits were found by doubling: where the probe stopped at an out-of-memory failure, the true maximum lies between the listed value and twice it; where it reached its batch cap (ResNet-style MLP), the listed value is a feasible batch size, not a maximum.

### 5. Logistic-regression reference baseline

One **unweighted L2-regularised (ridge-penalised) logistic regression** is added as a descriptive reference, allowing the eight configurations to be compared with a simple model that is linear in its transformed inputs. It is not a ninth configuration: it is outside H1–H3, the Holm family, the H3-configuration selection and RQ2/RQ3, and the fairness rule of §3 governs the eight configurations only.

- **Representation.** The shared Train-fitted preprocessing artifact of [DR-012](DR-012-neural-network-input-representation.md) (quantile-transformed numeric inputs, numeric missing values set to 0, missingness indicators, Train vocabularies), with every categorical input one-hot encoded over all its vocabulary categories, including `"__MISSING__"`; no category is dropped, no category dropout is applied and no further scaling is added. Neural embeddings are not used.
- **Fitting and search.** Fitted to convergence on Train with fixed solver settings; C (inverse regularisation strength) is selected from ten candidates by the V1 objective of [DR-015](DR-015-common-training-rules.md) §3 ([DR-015](DR-015-common-training-rules.md) §8 fixes the search). It shares the search budget, the V1 objective and the trial tie rule of the eight configurations, but not their checkpoint-based early stopping.
- **Reporting.** R3 recall at the four capacities with the oracle-normalised recall, the random-review context and the computational cost; V2 descriptively and Test only in the authorised confirmatory evaluation.

The full specification is in the [protocol](../protocol.md#models).

## Evidence

- **[L]** Tree ensembles remain strong on tabular data [REF-013, REF-014, REF-090], but no family is universally superior: a ResNet-style multilayer perceptron (MLP) is a strong deep baseline and FT-Transformer (Feature Tokenizer + Transformer) the strongest deep model on most of eleven benchmarks [REF-087]. Across 176 datasets the neural-vs-tree gap is often negligible relative to tuning effects, and GBDTs handle skewed, heavy-tailed features better [REF-089]. Well-regularised MLPs can outperform XGBoost [REF-088].
- **[L]** On IEEE-CIS (the IEEE Computational Intelligence Society Fraud Detection dataset) the fraud-specific evidence is limited: comparisons use non-temporal splits [REF-099], different user groups [REF-100], abstract-level results [REF-101] or unverifiable reporting [REF-098]. The closest capacity study compares logistic regression, LightGBM and TabNet without a family hypothesis [REF-078] ([`literature-evidence-matrix.md`](../literature-evidence-matrix.md), Section 3).

No study found compares tree-based and neural families under one tuning procedure, a chronological holdout and a fixed daily review budget.

## Alternatives considered

| Alternative | Why not chosen |
|---|---|
| No simple reference model | The tree-based and neural models could not be compared with a simple linear model; the closest comparator study reports logistic regression [REF-078] |
| An untuned default logistic regression, as in TabZilla [REF-130] | An untuned reference may underrepresent the model's performance; tuning only its regularisation strength gives a bounded search under the shared V1 objective [I]. Improved later-period performance is not assumed |
| Logistic regression with both weightings | Weighting is examined by H2 on the four principal models; a second reference configuration adds descriptive work without a design need |
| Logistic regression as a ninth configuration in H1 | H1 is a tree-versus-neural family contrast; a linear model belongs to neither family |
| One-hot encoding with one category dropped per input | Under an L2 penalty, dropping a category breaks the symmetry of the representation and can bias the model [REF-135] |
| Two tree-based libraries only (LightGBM vs CatBoost as H1) | Compares two members of one family; the open family question for tabular data is trees versus neural networks [REF-087, REF-089]. The within-tree contrast is retained as a descriptive result. |
| One neural model (e.g. FT-Transformer only) against two trees | An unbalanced design confounds the family effect with a single architecture. Two models per family let H1 average over architectures. |
| TabNet as the neural model (as in REF-078) | Ranked below ResNet-style MLP and FT-Transformer under a common tuning protocol [REF-087]. |
| XGBoost as a third tree model | Adds a model, not a family contrast; LightGBM and CatBoost already cover the two leading GBDT implementations with native categorical handling. |
| Self-supervised pretraining or transaction-sequence models [REF-098, REF-099] | Require entity sequences or engineered histories that IEEE-CIS does not identify, and fall outside the raw-input policy of [DR-008](DR-008-feature-declaration-and-model-input-policy.md). |
| Identical hyperparameter values across families | Not meaningful across model families; fairness is defined by the common procedure instead. |
| Model-specific tuning budgets (e.g. more trials for the slower neural models) | Would make the family comparison depend on unequal effort. The budget is common; run time is reported. |

## Rationale

Two models per family give H1 a family-level estimand that does not rest on one architecture. The chosen neural models are the strongest documented baselines under a common tuning protocol [REF-087], and the tree models are the two GBDT libraries already declared in the input policy. The three-level procedure makes a family difference attributable to the model-and-representation pipelines compared, not to unequal data access or effort.

## Consequences

- **H1** compares the families: the mean Recall@2% of the four neural configurations minus the mean of the four tree configurations ([DR-005](DR-005-sesoi-and-h1-h2-inference.md)). Within-family contrasts (CatBoost vs LightGBM; FT-Transformer vs ResNet) are descriptive.
- **H2** covers all four models through one intersection-union equivalence test ([DR-005](DR-005-sesoi-and-h1-h2-inference.md)).
- **H3** selects its configuration from all eight ([DR-010](DR-010-h3-confirmatory-configuration.md)); threshold transfer is also compared descriptively between families.
- The Holm family stays {H1, H2, H3} with three members.
- The logistic-regression reference (§5) is reported descriptively next to the eight configurations; it changes none of H1–H3, the Holm family or the H3 selection.

## Threats and limitations

- **Reference baseline.** The logistic regression is linear in the transformed representation of [DR-012](DR-012-neural-network-input-representation.md), not in the original values. A `"__MISSING__"` category never observed in Train receives no training evidence in it, because category dropout is not applied. It supports no conclusion about class weighting for logistic regression.
- **Pipeline-specific conclusions.** H1 concerns these four models under this procedure and input policy. It is not a claim about every tree-based or neural method.
- **Representation differences.** Trees and neural networks receive the same information in family-specific forms ([DR-008](DR-008-feature-declaration-and-model-input-policy.md), [DR-012](DR-012-neural-network-input-representation.md)). These differences are part of the family contrast.
- **Tuning efficiency.** A common budget measured in search trials may favour the family that converges faster; run time and the number of completed trials are reported per configuration.
- **Seed variability.** Training variability can differ between configurations; main results use the main seed 42, and the common additional-seed analysis measures this variability for every configuration without assuming a family ordering ([DR-013](DR-013-training-device-policy.md) §6).
- **Mixed precision.** bf16 training is the project's engineering adaptation for the 8 GB GPU; numerical differences from float32 are possible, so prediction is in float32 and training stability is checked ([DR-013](DR-013-training-device-policy.md)).

## References

REF-011, REF-012, REF-013, REF-014, REF-078, REF-087, REF-088, REF-089, REF-090, REF-098, REF-099, REF-100, REF-101, REF-130, REF-135 — see [`../references.md`](../references.md).

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [L] | Evidence tag: literature |
| [I] | Evidence tag: inference or recommendation |
| bf16 | bfloat16: 16-bit floating-point format with the exponent range of float32 |
| BatchNorm, Dropout, Linear, ReLU | Batch normalisation; dropout; affine layer; rectified linear unit |
| `__MISSING__` | Explicit label for a missing categorical value (DR-008); reserved in every neural-network vocabulary (DR-012) |
| Cat, LGB | CatBoost, LightGBM |
| [CLS] | Classification token whose final representation gives the FT-Transformer prediction |
| CPU, GPU | Central and graphics processing units |
| CUDA | NVIDIA's GPU computing platform |
| FTT, RES | FT-Transformer; ResNet-style MLP |
| FT-Transformer | Feature Tokenizer + Transformer model for tabular data |
| GBDT | Gradient-boosted decision trees |
| H1, H2, H3 | Confirmatory hypotheses: model family; class weighting; alert-rate transfer fidelity |
| IEEE-CIS | IEEE Computational Intelligence Society Fraud Detection dataset |
| K | Fixed daily review budget (56 at 2 % capacity) |
| MLP | Multilayer perceptron |
| N, T | Neural family {RES, FTT}; tree-based family {LGB, Cat} |
| PreNorm | Transformer layer with normalisation at the start of each residual branch |
| R4 | Decision policy: calibrated cost-aware capped review |
| Recall@2% | Share of frauds reviewed under daily top-K review with K = 56 |
| ReGLU | Rectified gated linear unit (feed-forward activation of FT-Transformer) |
| RQ1 | Research question 1 (model family and class weighting) |
| u, b | Unweighted; balanced class weighting |
| V1, V2 | Validation 1 (relative days 106–126); Validation 2 (relative days 127–154) |
| x | Hidden representation entering a ResNet block |
| C | Inverse regularisation strength of the logistic-regression reference |
| L2 | Squared-value (ridge) penalty on the logistic-regression coefficients |
| R3 | Decision policy: daily top-K review |
| RQ2, RQ3 | Research questions 2 (policy-transfer fidelity) and 3 (cost-aware capacity use) |
