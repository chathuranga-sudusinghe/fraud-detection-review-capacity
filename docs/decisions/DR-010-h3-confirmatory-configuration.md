# DR-010 — Confirmatory configuration for H3

| Field | Value |
|---|---|
| Status | Accepted (pre-freeze), Issue #29, in line with the University of Greenwich criteria ([criteria checklist](../greenwich-criteria-checklist.md)); supervisor feedback welcome, not a precondition |
| Date | 2026-09-26; revised 2026-10-09 (Issue #54: selection pool of eight configurations; Holm-level H3 decision) |
| Evidence source | The protocol's hypothesis, policy and multiplicity definitions; [DR-001](DR-001-temporal-split-strategy.md), [DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md) and [DR-005](DR-005-sesoi-and-h1-h2-inference.md); the existing power projections of DR-005. No labels, model scores or new dataset computations were used, no models were trained, and no Test-period data were inspected for this record. |
| Related | [DR-001](DR-001-temporal-split-strategy.md), [DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md), [DR-005](DR-005-sesoi-and-h1-h2-inference.md), [DR-006](DR-006-bootstrap-inference-strategy.md), [DR-009](DR-009-overflow-handling-for-threshold-policies.md), [DR-011](DR-011-model-set-tree-based-and-neural-families.md), [protocol](../protocol.md) |

## Context

The study trains eight model × class-weighting configurations ([DR-011](DR-011-model-set-tree-based-and-neural-families.md)): two tree-based models (LightGBM, CatBoost) and two neural models (ResNet-style multilayer perceptron (MLP), Feature Tokenizer + Transformer (FT-Transformer)), each unweighted and with balanced class weighting (each fraud row weighted by the Train ratio of legitimate to fraud transactions).

Hypothesis H3 (alert-rate transfer fidelity, [DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md)) concerns the score threshold τ₂ that policy R2 (the capacity-rate threshold) selects on Validation 2 (V2, relative days 127–154) so that 2 % of V2 transactions are flagged. H3 asks whether τ₂, applied unchanged to the Test period (relative days 155–182), still flags about 2 % of transactions. The realised Test alert rate is:

$$
r = \frac{\sum_d n_d}{\sum_d N_d}
$$

where:

- `r` = realised Test alert rate at τ₂, pooled over Test days;
- `d` = a Test relative day;
- `n_d` = number of transactions on day d with score ≥ τ₂;
- `N_d` = total number of transactions on day d;
- `Σ_d` = sum over the 28 Test days.

H3 is tested with TOST (Two One-Sided Tests) against the band (0.016, 0.024), i.e. within Δ = 0.004 (the equivalence tolerance, ±20 % of the 2 % target), using a day-level moving-block bootstrap (block length ℓ = 3 days, B = 10,000 replicates; [DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md)). H3₁ (the alternative, "the rate is preserved") is concluded only if the Holm-adjusted TOST p-value is at most 0.05; the 90 % confidence interval (CI) for r is reported descriptively.

Each configuration produces its own scores, so each has its own τ₂ and its own r. DR-003 defines H3 inference as **conditional on the trained model and the frozen τ₂**, so the model on which H3 is confirmatory must be specified. [DR-005](DR-005-sesoi-and-h1-h2-inference.md) fixes the multiple-testing plan as Holm's procedure at family-wise α = 0.05 (α = significance level) over {H1, H2, H3}; every additional confirmatory H3 configuration would add a family member. The rolling-origin supplement ([DR-001](DR-001-temporal-split-strategy.md), protocol) also needs a defined "primary configuration".

### Methodological problem

The choice must be fixed before any Test access, and must not depend on which configuration is expected to transfer best. It has three consequences:

1. **Estimand.** H3 either describes the transferred threshold of one pre-specified configuration or makes a claim across several models.
2. **Multiplicity.** Every extra confirmatory H3 test enlarges the Holm family. This makes the multiple-testing procedure more stringent and can reduce the power of H1 (model-family effect) and H2 (class-weighting effect), especially through the smaller early Holm thresholds. DR-005 projects their power as limited (H1 power ≈ 0.47–0.64 at the Smallest Effect Size Of Interest, SESOI, of 2 percentage points, pp).
3. **Timing.** A configuration picked by a data-dependent rule is not known until models are developed, so the *rule*, not the configuration, must be frozen.

## Decision

**H3 is confirmatory for a single configuration, the *H3 configuration*, selected by a pre-specified development-only rule.** The H3 configuration is a **pre-specified, validation-selected deployment proxy**: a configuration chosen on held-out development data, whose V2-selected threshold is then transferred to the Test period. The study does not model any particular organisation's operational model-selection process.

### Selection rule

After all eight configurations have been trained on Train, tuned and early-stopped on V1 (Validation 1, relative days 106–126), and frozen, each is scored on V2. The H3 configuration is:

$$
(f^*, w^*) = \underset{(f,\,w)}{\operatorname{arg\,max}} \; R^{f,w}_{\mathrm{V2}}
$$

where:

- `(f*, w*)` = the H3 configuration;
- `f` = model: LightGBM (LGB), CatBoost (Cat), ResNet-style MLP (RES) or FT-Transformer (FTT);
- `w` = class weighting, u (unweighted) or b (balanced);
- `R^f,w_V2` = pooled V2 Recall@2% of configuration (f, w) under policy R3 (daily top-K review, fixed daily review budget K = 56), i.e. the share of V2 frauds that fall inside the daily top-56 of their relative day;
- `arg max` = the configuration at which the quantity is largest.

The H3 configuration is the one that captures the most V2 frauds at the primary 2 % capacity.

### Confirmatory estimand

For confirmatory H3, r refers **only** to the H3 configuration. Its threshold τ₂^f*,w* is chosen on V2 from the H3 configuration's scores, as in DR-003, and:

$$
r^{f^*,w^*} = \frac{\sum_d n_d^{f^*,w^*}}{\sum_d N_d}
$$

where:

- `r^f*,w*` = realised Test alert rate of the H3 configuration, pooled over Test days;
- `n_d^f*,w*` = number of Test transactions on day d whose H3-configuration score is ≥ τ₂^f*,w*;
- `τ₂^f*,w*` = the H3 configuration's V2-selected 2 % threshold;
- `N_d` and `Σ_d` = as defined above.

This is DR-003's r evaluated for one named configuration. It is not a rate pooled or averaged across configurations, and a confirmatory H3 conclusion does not apply to the other seven. The hypotheses H3₀ and H3₁, the tolerance Δ = 0.004, the TOST and the moving-block bootstrap are applied to r^f*,w* exactly as specified in DR-003.

**Tie-break**, applied in order, only when V2 fraud counts captured are exactly equal:

1. higher V2 Average Precision (PR-AUC, Precision–Recall Area Under the Curve) [REF-010];
2. fixed order: LGB-u, Cat-u, RES-u, FTT-u, LGB-b, Cat-b, RES-b, FTT-b ([DR-011](DR-011-model-set-tree-based-and-neural-families.md)). This order is arbitrary and is declared only to make the rule deterministic.

**Procedural safeguards.**

- R3 on V2 uses the protocol's daily top-K definition; its tie rule for equal scores is fixed in the training rules (protocol design item 3).
- The eight V2 recall values, the tie-break steps used, if any, and the selected configuration are recorded in the repository (committed) **before the Test period is unsealed**. The selection is final once recorded. It is not revisited after Test results are seen.
- The rule uses V2 labels. That is within V2's role: V2 labels are already used for R1 (the F1-optimal threshold) and for the R4 calibrator. The H3 outcome r itself needs no labels.

### Why V2 Recall@2% under R3

- **Validation-selected deployment proxy [I].** The configuration is chosen on held-out development data and only then has its operating threshold set, so H3 asks whether the threshold of a validation-selected configuration transfers in time. This matches DR-003's estimand, which is conditional on the trained model and the frozen τ₂. It is a research proxy, not a model of any organisation's selection process.
- **Held-out and closest to Test [I].** V1 scores are optimistic because V1 drives early stopping ([DR-001](DR-001-temporal-split-strategy.md)). V2 is out-of-sample for all eight configurations and is the period on which τ₂ is set.
- **Aligned with the primary metric [I].** Recall@2% under R3 is the study's primary metric, so the H3 configuration is the one best suited to the primary capacity by the study's own criterion.
- **Not tuned to the H3 outcome [I].** The criterion is ranking quality, not alert-rate stability. It does not favour configurations whose rate transfers well (see [Alternatives considered](#alternatives-considered)).

### Multiplicity

The confirmatory family is **Holm at family-wise α = 0.05 over {H1, H2, H3}**, three members [REF-029], where H3 is the single TOST on r^f*,w* (the H3 configuration) with p-value p_H3 (the larger of the two one-sided p-values, as in DR-003). The Holm steps compare the ordered p-values with:

$$
\alpha_{(1)} = \frac{0.05}{3} \approx 0.0167, \qquad \alpha_{(2)} = \frac{0.05}{2} = 0.025, \qquad \alpha_{(3)} = 0.05
$$

where `α_(k)` = significance level applied to the k-th smallest p-value in Holm's procedure. Decisions use the Holm-adjusted p-values of [DR-005](DR-005-sesoi-and-h1-h2-inference.md), which enforce the step-down stopping rule: once a comparison fails, no later hypothesis, H3 included, is rejected. The 90 % CI for r is reported descriptively in every case.

### Relationship to RQ2, H1 and H2

- **RQ2 (research question 2, policy-transfer fidelity).** H3 answers the confirmatory part of RQ2 for the H3 configuration only. The descriptive daily-budget outcomes of RQ2 ([DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md), [DR-009](DR-009-overflow-handling-for-threshold-policies.md)) are reported for all eight configurations; the H3 configuration's are presented first.
- **H1 and H2.** H1 and H2 compare all eight configurations on the Test period and do not depend on the selection. The V2 selection defines a deployment proxy; it is not an inference about model family or weighting. It must **not** be reported as evidence for H1 or H2, and the selected configuration is not called the "best" model.
- **Independence from Test.** The selection uses Train, V1 and V2 only, so no Test information reaches H3's configuration, τ₂ or tolerance.

### Non-confirmatory H3 results for the other seven configurations

For each of the other seven configurations, report r, the 90 % CI from the same moving-block bootstrap, and the position of the CI relative to (0.016, 0.024). These are labelled **descriptive (non-confirmatory)**:

- they are not in the Holm family and carry no multiplicity adjustment;
- no equivalence conclusion is drawn from them;
- they show how sensitive H3's conclusion is to the choice of configuration;
- they support a **descriptive tree-vs-neural comparison of threshold transfer**: the realised rates r of the four tree configurations and of the four neural configurations, with their relative deviations from the 2 % target, are presented side by side.

The non-confirmatory H3 sensitivity analyses (ℓ = 1 and 7 days; tolerance bounds ±10 %, ±25 %, ±50 %; smallest tolerance at which equivalence would be concluded) apply to the H3 configuration and may be reported descriptively for the others.

### Related definitions

- **Primary configuration.** The "primary configuration" of the rolling-origin supplement ([DR-001](DR-001-temporal-split-strategy.md), protocol) is the H3 configuration's model and weighting.
- **Label-latency sensitivity.** Because the rule uses V2 labels, the label-latency sensitivity also reports which configuration the rule would select using only V2 days 127–147. This is descriptive and does not change the H3 configuration.

## Evidence

- **[I]** DR-003 already defines H3 conditionally on the trained model and the frozen τ₂, so a single configuration matches the existing estimand.
- **[I]** DR-005's power projections show H1 and H2 powered only for moderate effects. Enlarging the Holm family would make the procedure more stringent and could reduce their power further, especially through smaller early thresholds (below).
- **[L]** Holm's procedure controls the family-wise error rate under arbitrary dependence, and its first-step level falls with family size [REF-029].
- **[L]** Operating thresholds should be chosen on held-out data, and the choice is part of what is evaluated [REF-038].

## Alternatives considered

| Alternative | Why not chosen |
|---|---|
| **All eight configurations, each a separate Holm member** (family of 10) | Makes Holm more stringent: the first step falls from 0.05 / 3 ≈ 0.0167 to 0.05 / 10 = 0.005 and the early steps are smaller, which can reduce the power of H1 and H2, depending on the ordering of the p-values. The eight H3 tests are strongly dependent (same data, same days), so they add little independent information. Their results are still reported descriptively. |
| **All eight configurations, one intersection-union test** ("rate preserved for every configuration") | Keeps the family at three, but changes the H3 statement into a stricter claim about all models, with lower power. It answers a different question from DR-003's estimand, which is conditional on one trained model and its τ₂. |
| **Two configurations** (e.g. one per family or per weighting) | Enlarges the family without a clear estimand. Any particular pair would need its own justification. The family difference in transfer is reported descriptively instead. |
| **Selection pool restricted to the tree-based configurations** | Would exclude half of the model set from the deployment-proxy choice without a methodological reason. |
| **One configuration fixed now** (e.g. LightGBM unweighted) | Needs no data, but the choice of model is essentially arbitrary. It may fix a configuration that validation would not have chosen, which weakens its interpretation as a validation-selected deployment proxy. |
| **Select by V1 performance** | V1 scores are optimistic because of early stopping ([DR-001](DR-001-temporal-split-strategy.md)). |
| **Select by development-period rate transfer** (e.g. the configuration whose V1-selected threshold best preserves 2 % on V2) | Optimises the H3 outcome on development data. It biases the confirmatory test towards equivalence and turns H3 into a best-case claim. |
| **Select by V2 PR-AUC** | A whole-curve metric not tied to the 2 % capacity. It is kept only as a tie-break. |
| **Choose the configuration after the Test evaluation** | Forbidden: the configuration could then be chosen to fit the result. |
| **Decide H3 by the 90 % CI regardless of Holm's procedure** | The 90 % CI corresponds to an unadjusted α = 0.05. A 90 % rule would bypass the family-wise error control and Holm's stopping rule that the confirmatory family commits to. The 90 % CI is reported descriptively. |

## Rationale

A single, rule-selected configuration gives H3 a clear meaning (does the V2-selected threshold of a pre-specified, validation-selected deployment proxy survive temporal shift?), keeps the pre-specified family of three confirmatory tests and avoids a more stringent Holm procedure, and is fixed by a rule that uses no Test information and does not target the H3 outcome. Reporting the other seven configurations descriptively keeps the configuration dependence, and the tree-vs-neural difference in transfer, visible without spending α on it.

## Consequences

- The Holm family is {H1, H2, H3 on the H3 configuration}: three members. The H3 decision uses its Holm-adjusted p-value.
- V2's role gains one step: selection of the H3 configuration by the rule above.
- Model development produces V2 Recall@2% under R3 for all eight configurations, and the selection record is committed before the Test period is unsealed.
- The rolling-origin supplement's "primary configuration" is the H3 configuration.

## Threats and limitations

- **Configuration-conditional conclusion.** The confirmatory H3 result applies to one configuration. A different configuration could transfer differently; the descriptive results for the other seven show how much.
- **Selection noise.** V2 recall differences between configurations may be smaller than their sampling error, so the selected configuration may not be better in any meaningful sense. This does not threaten validity, because the rule's purpose is a reproducible choice of deployment proxy, not a claim of superiority.
- **Indirect link between selection and transfer.** Ranking quality and rate stability are different properties, but they are not guaranteed to be independent. Selection by recall could be mildly associated with how well τ₂ transfers. The rule does not target the rate, and the other seven configurations are reported.
- **Label latency.** The rule uses V2 labels, which might not yet be complete at the time a threshold is set in practice. The label-latency sensitivity reports whether the choice would change.

## References

No new external references. REF-010, REF-029 and REF-038 are existing, verified entries in [`../references.md`](../references.md).

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [I], [L] | Evidence tags: inference; literature |
| α | Significance level (family-wise 0.05) |
| α_(k) | Holm significance level applied to the k-th smallest p-value |
| arg max | The configuration at which a quantity is largest |
| B | Number of bootstrap replicates (10,000) |
| Cat, LGB, RES, FTT | CatBoost, LightGBM, ResNet-style MLP, FT-Transformer |
| CI | Confidence interval |
| d | A Test relative day |
| Δ | H3 equivalence tolerance on r (0.004 = 20 % of the 2 % target) |
| DR | Decision Record |
| f, w | Model (LGB, Cat, RES or FTT); class weighting (u unweighted, b balanced) |
| (f*, w*) | The H3 configuration selected by the rule |
| LGB-u, Cat-u, RES-u, FTT-u, LGB-b, Cat-b, RES-b, FTT-b | The eight model × weighting configurations (model-weighting), in tie-break order |
| MLP | Multilayer perceptron |
| FT-Transformer | Feature Tokenizer + Transformer |
| F1 | Harmonic mean of precision and recall |
| H1, H2, H3 | Confirmatory hypotheses: model family; class weighting; alert-rate transfer fidelity. H3₁ = alternative |
| K | Fixed daily review budget (56 at 2 % capacity) |
| ℓ | Block length of the moving-block bootstrap, in days |
| n_d | Number of transactions on day d with score ≥ τ₂ |
| N_d | Total number of transactions on day d |
| p_H3 | TOST p-value of H3: the larger of its two one-sided p-values |
| pp | Percentage points |
| PR-AUC | Precision–Recall Area Under the Curve (Average Precision) |
| r | Realised Test alert rate at τ₂, pooled over Test days (DR-003) |
| r^f*,w* | Confirmatory H3 estimand: r for the H3 configuration only |
| n_d^f*,w* | Test transactions on day d with H3-configuration score ≥ τ₂^f*,w* |
| R1–R4 | Decision policies: F1-optimal threshold; capacity-rate threshold; daily top-K; calibrated cost-aware capped policy |
| R^f,w_V2 | Pooled V2 Recall@2% of configuration (f, w) under R3 |
| Recall@2% | Share of frauds reviewed at 2 % review capacity |
| REF-NNN | Reference identifier in `../references.md` |
| RQ1–RQ3 | Research questions 1–3; RQ2 = policy-transfer fidelity |
| SESOI | Smallest Effect Size Of Interest (2 pp of Recall@2%) |
| Σ_d | Sum over the 28 Test days |
| τ₂ | V2-selected score threshold flagging 2 % of V2 transactions |
| τ₂^f*,w* | The H3 configuration's V2-selected 2 % threshold |
| TOST | Two One-Sided Tests |
| u, b | Unweighted; balanced class weighting |
| V1, V2 | Validation 1 (relative days 106–126); Validation 2 (relative days 127–154) |
