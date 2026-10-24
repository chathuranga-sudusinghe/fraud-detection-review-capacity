# DR-005 — SESOI and H1/H2 inference plan

| Field | Value |
|---|---|
| Status | Accepted (pre-freeze), Issue #8, in line with the University of Greenwich criteria ([criteria checklist](../greenwich-criteria-checklist.md)); supervisor feedback welcome, not a precondition |
| Date | 2026-09-24 (Issue #8: SESOI fixed at 2 percentage points of Recall@2%); revised 2026-10-09 (Issue #54: estimands for the eight-configuration model set; bootstrap p-value definition) |
| Evidence source | `notebooks/01_dataset_feasibility_audit.ipynb`, §§6–7, 15; development-period (V2) projections; equivalence-testing and practical-significance literature |
| Related | [DR-001](DR-001-temporal-split-strategy.md), [DR-002](DR-002-review-capacity-definition.md), [DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md), [DR-006](DR-006-bootstrap-inference-strategy.md), [DR-010](DR-010-h3-confirmatory-configuration.md), [DR-011](DR-011-model-set-tree-based-and-neural-families.md), [protocol](../protocol.md) |

## Context

Confirmatory hypothesis H1 compares the two model families, tree-based and neural, and hypothesis H2 compares class weighting (balanced vs unweighted), across the eight configurations of [DR-011](DR-011-model-set-tree-based-and-neural-families.md): LightGBM (LGB) and CatBoost (Cat) in the tree-based family, ResNet-style multilayer perceptron (MLP) (RES) and Feature Tokenizer + Transformer (FT-Transformer) (FTT) in the neural family, each unweighted (u) and balanced (b). Both hypotheses use the primary outcome: **Recall@2% (the share of frauds reviewed at 2 % review capacity) under the daily top-K policy R3 (review the K highest-scoring transactions each relative day), with K (fixed daily review budget) = 56 reviews per relative day** ([DR-002](DR-002-review-capacity-definition.md)).

The feasibility audit found four things:
- **[D]** At 2 % capacity the oracle recall ceiling is ≈ 0.595 (V2).
- **[D]** About 2,660 Test frauds are projected from V2 prevalence. This is a projection; Test labels are sealed.
- **[D]** Frauds cluster by heuristic pseudo-entity, with an upper-bound design effect (DE: the factor by which clustering inflates the variance relative to independent observations) of ≈ 4.3 in Validation 2 (V2, relative days 127–154).
- **[I]** Differences between configurations may be small. The minimum detectable paired difference was ≈ 1.2–5.6 percentage points (pp), depending on discordance and clustering.

Class weighting is expected mainly to distort calibration, with little effect on ranking [REF-019, REF-020], so a small or null H2 effect is plausible, and large tabular benchmarks find that the neural-vs-tree gap is often small relative to tuning effects [REF-089]. A non-significant result must not be read as "no effect" [REF-030]. A **smallest effect size of interest (SESOI)** is therefore fixed before any Test evaluation, together with the H1/H2 estimands and test logic.

## Decision

### 1. SESOI

> **SESOI = 2 percentage points (0.02) of Recall@2%**, on the absolute recall scale, for both H1 and H2.

### 2. Estimands

R^m,w is the pooled Test Recall@2% of model m ∈ {LGB, Cat, RES, FTT} with weighting w ∈ {u, b}. It is the share of Test frauds that fall within their day's top-56 scores, over relative days 155–182. All eight configurations are evaluated on the **same** Test frauds, so every contrast is **paired**.

| Estimand | Definition | Used by |
|---|---|---|
| δ_F (model-family effect) | Mean of the four neural configurations minus mean of the four tree configurations | H1 (confirmatory) |
| δ_W,m (weighting simple effects) | R^m,b − R^m,u for each of the four models | H2 (confirmatory) |
| Within-family model contrasts | Cat − LGB and FTT − RES, each averaged over weighting | Descriptive |
| Family × weighting interaction | (N_b − T_b) − (N_u − T_u), where N_w and T_w are the family means at weighting w | Descriptive |
| Cell contrasts | Every paired configuration difference used above, per weighting | Descriptive |

The confirmatory estimands, displayed:

$$
\begin{aligned}
\delta_F &= \frac{1}{4} \sum_{m \in N} \sum_{w \in \lbrace u, b \rbrace} R^{m,w} \;-\; \frac{1}{4} \sum_{m \in T} \sum_{w \in \lbrace u, b \rbrace} R^{m,w} \\
\delta_{W,m} &= R^{m,b} - R^{m,u} \qquad \text{for } m \in \lbrace \mathrm{LGB}, \mathrm{Cat}, \mathrm{RES}, \mathrm{FTT} \rbrace
\end{aligned}
$$

where:

- `δ_F` = model-family effect: neural-minus-tree recall difference, averaged over the two models of each family and over both weighting conditions;
- `δ_W,m` = weighting simple effect of model m: balanced minus unweighted recall;
- `T` = tree-based family {LGB, Cat}; `N` = neural family {RES, FTT};
- `R^m,w` = pooled Test Recall@2% of model m with weighting w.

δ_F is positive when the neural family reviews more frauds than the tree-based family; δ_W,m is positive when balanced weighting helps model m.

### 3. H1 — model-family effect: a difference test plus pre-specified practical interpretation

> **H1₀:** Averaged over the two models of each family and over the unweighted and balanced-weighting conditions, tree-based and neural models do not differ in Recall@2% on the Test period (δ_F = 0).
>
> **H1₁:** Averaged over the two models of each family and over the unweighted and balanced-weighting conditions, tree-based and neural models differ in Recall@2% on the Test period (δ_F ≠ 0).

H1 is a two-sided **difference** test, because research question RQ1 asks *whether and how* the families differ. It is not a superiority or winner-selection test.

In addition, the magnitude of δ_F is interpreted against ±SESOI using its **90 % confidence interval (CI)**. This interpretation is **pre-specified but not a separate confirmatory test**:

| H1 difference test (significant = Holm-adjusted p-value ≤ 0.05) | 90 % CI for δ_F relative to ±0.02 | Reported interpretation |
|---|---|---|
| Significant | Entirely beyond +0.02 or entirely beyond −0.02 | Evidence of a difference of practically meaningful size |
| Significant | Crosses ±0.02 | Evidence of a difference; whether it reaches practical size is uncertain |
| Significant | Entirely within (−0.02, +0.02) | Evidence of a difference that is too small to matter in practice |
| Not significant | Entirely within (−0.02, +0.02) | Data compatible only with practically negligible differences |
| Not significant | Extends beyond ±0.02 | Inconclusive: uncertainty too large to distinguish practically meaningful effects |

**Qualification rules (descriptive).**
- If the 95 % CI of the family × weighting interaction excludes 0 **and** its point estimate exceeds the SESOI in magnitude, the family effect is reported separately for each weighting condition.
- The two within-family model contrasts are always reported, so that a family effect driven by one model is visible.

### 4. H2 — class-weighting effect: equivalence (intersection-union TOST, Two One-Sided Tests)

> **H2₀:** For at least one of the four models (LightGBM, CatBoost, ResNet-style MLP or FT-Transformer), balanced class weighting changes Recall@2% by 2 percentage points or more relative to the corresponding unweighted model (|δ_W,m| ≥ 0.02 for at least one m).
>
> **H2₁:** For all four models, balanced class weighting changes Recall@2% by less than 2 percentage points relative to the corresponding unweighted model (|δ_W,m| < 0.02 for every m).

**Why equivalence.**
- The substantive question for H2 is whether weighting matters *materially* for capacity-constrained recall. The literature expectation is that it mostly does not [REF-019, REF-020].
- An equivalence test puts the burden of proof on the "no meaningful effect" claim and prevents non-significant results being read as "no effect" [REF-030, REF-063, REF-061].

**Why every model (intersection-union), not an averaged effect.**
- An average can cancel opposite effects. For example, +3 pp for one model and −3 pp for another average to 0 and would wrongly suggest equivalence.
- Requiring equivalence **for each model** is an intersection-union test. Each component TOST is performed at the full level, with no further split [REF-062].

**TOST logic.**
- For each model m there are two one-sided tests: H0a,m: δ_W,m ≤ −0.02 and H0b,m: δ_W,m ≥ +0.02.
- p_H2 (the H2 p-value) = the maximum of the eight one-sided p-values, so H2 can be rejected only if **all eight** one-sided nulls are rejected.
- H2₁ is concluded only if the Holm-adjusted p-value of H2 is at most 0.05 (Multiplicity, below).
- The 90 % CIs of the four δ_W,m and their position relative to (−0.02, +0.02) are reported. They describe uncertainty only and do not determine rejection: the decision uses the Holm-adjusted p-value, so all four intervals can lie inside the bounds while H2 is not rejected after multiplicity adjustment.
- The conventional two-sided difference CIs for δ_W,m are also reported, **descriptively**, so a statistically detectable but practically negligible weighting effect remains visible.

### Statistical inference plan (H1, H2)

| Element | Specification |
|---|---|
| **Estimands** | δ_F (H1); δ_W,LGB, δ_W,Cat, δ_W,RES and δ_W,FTT (H2); within-family model contrasts, family × weighting interaction and cell contrasts (descriptive). All are paired differences in pooled Test Recall@2% under R3, K = 56. |
| **Primary uncertainty method** | **Joint paired bootstrap.** Each replicate resamples the same units for all eight configurations, preserving the pairing, and recomputes every estimand. B (number of bootstrap replicates) = 10,000; percentile CIs [REF-044]. |
| **Clustering / resampling unit** | Two-way (pigeonhole) bootstrap over relative days × fine pseudo-entities, with an Owen-corrected variance and a day-level moving-block bootstrap as sensitivity analyses ([DR-006](DR-006-bootstrap-inference-strategy.md)). Transaction-level or fraud-level resampling that assumes independence is **not** acceptable. The SESOI decision does not depend on this choice: the power scenarios in [Resolution of the design](#resolution-of-the-design-approximate-planning-assumptions-i) bracket DE = 1–4. |
| **CI types** | 95 % CI for difference estimates (H1, descriptive contrasts). 90 % CI for practical interpretation against ±SESOI (H1) and for the four δ_W,m against the H2 equivalence bounds (descriptive). Confirmatory decisions use the p-values. |
| **p-values** | H1: two-sided bootstrap p-value p_H1 for δ_F = 0 (displayed below the table). H2: p_H2 = max of the eight one-sided bootstrap p-values (intersection-union). H3: as defined in [DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md). |
| **Role of McNemar** | **Secondary and descriptive only.** An exact McNemar test [REF-035, REF-028] is reported for the cell-level paired contrasts. It assumes independent frauds, which the clustering evidence contradicts, so it is **not** in the confirmatory family. |
| **Role of TOST** | Confirmatory for **H2** (intersection-union over all four models) and **H3** ([DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md)). Descriptive for H1's practical interpretation. |
| **Multiplicity** | **Holm** [REF-029] at family-wise α = 0.05 over the confirmatory family {H1, H2, H3}, using p_H1, p_H2 and p_H3 (H3 is the alert-rate transfer hypothesis, tested once on the H3 configuration; [DR-010](DR-010-h3-confirmatory-configuration.md)). Decisions use the Holm-adjusted p-values defined below the table, so Holm's step-down stopping rule applies. Holm controls the family-wise error rate under arbitrary dependence and can combine difference and equivalence p-values, since each is a valid p-value for its own null. H2 is one family member because its intersection-union structure needs no internal split [REF-062]. |
| **Is Holm appropriate?** | Yes. It is conservative, but the confirmatory claims are few and pre-specified, and conservatism suits a single-holdout study. The cost is lower power at the first Holm step (α = 0.0167); this is shown in the power table. |
| **Effect-size reporting** | For every estimand: the point estimate in pp; frauds per 28 days and per day; the Precision@K equivalent; normalised recall; CIs. Sensitivity: H1/H2 interpretations are also shown against SESOIs of 1, 3 and 5 pp (not confirmatory). |

The H1 bootstrap p-value is:

$$
p_{\mathrm{H1}} = \min\left\lbrace 1,\ 2 \cdot \min\left( \frac{1 + \#\lbrace \delta^*_F \le 0 \rbrace}{B + 1},\ \frac{1 + \#\lbrace \delta^*_F \ge 0 \rbrace}{B + 1} \right) \right\rbrace
$$

where:

- `δ*_F` = the value of δ_F in one bootstrap replicate;
- `#{ · }` = the number of bootstrap replicates satisfying the condition;
- `B` = number of bootstrap replicates (10,000).

It doubles the smaller tail of the bootstrap distribution beyond zero, giving a two-sided test of δ_F = 0. Adding one to each count and to B keeps the p-value strictly positive and the cap at 1 keeps it a valid probability, including the case where many replicates equal 0 exactly.

The Holm-adjusted p-values of the confirmatory family are:

$$
\tilde{p}_{(k)} = \max_{i \le k} \ \min\left\lbrace 1,\ (4 - i) \cdot p_{(i)} \right\rbrace, \qquad k = 1, 2, 3
$$

where:

- `p_(1) ≤ p_(2) ≤ p_(3)` = p_H1, p_H2 and p_H3 sorted in increasing order;
- `i`, `k` = rank positions in that order (1, 2 or 3);
- `p̃_(k)` = Holm-adjusted p-value of the hypothesis with the k-th smallest p-value.

A hypothesis is rejected if and only if its adjusted p-value is at most 0.05. The running maximum enforces Holm's step-down stopping rule: once the comparison of an ordered p-value with its level (0.05 / 3, then 0.05 / 2, then 0.05) fails, no later hypothesis is rejected. For example, p-values {0.020, 0.021, 0.040} give adjusted values {0.060, 0.060, 0.060}, and none is rejected. The unadjusted and the Holm-adjusted p-value are both reported for H1, H2 and H3; only the adjusted value decides.

## Evidence

### Literature findings

- **[L] There is no fraud-domain standard for a practically meaningful Recall@capacity difference.** The structured literature search found none ([`literature-search-method.md`](../literature-search-method.md)). Fraud studies report metric differences but do not define a minimum meaningful difference. The capacity-evaluation literature [REF-004, REF-005, REF-006] defines the metric, not a margin.
- **[L] The concept is established in other fields.** A *minimal clinically important difference* is defined by practical, not statistical, meaning: "the smallest difference in score in the domain of interest which patients perceive as beneficial and which would mandate … a change in the patient's management" [REF-066]. This project adapts the principle, not any clinical value.
- **[L] Equivalence bounds express the SESOI.** Equivalence is tested by TOST, which corresponds to a (1 − 2α) CI inside the bounds [REF-061, REF-030, REF-063]. An intersection-union test needs no α split among its components [REF-062].
- **[L] The only machine-learning (ML) precedent is analyst-chosen.** Benavoli et al. treat a mean accuracy difference of less than 1 % as "practically equivalent" (a region of practical equivalence, ROPE, of ±0.01) for generic classifier benchmarks [REF-067]. That is an analyst-chosen value for **accuracy averaged over datasets**, not for Recall@capacity in fraud, so it is **not** adopted as the SESOI here.

**Consequence.** The SESOI is a **pre-specified judgement**, justified below. It is not a literature convention.

### Operational meaning of candidate SESOIs (projection; no Test data)

Inputs [D]:
- about 2,660 projected Test frauds;
- K = 56 reviews/day over 28 days, i.e. 1,568 reviews in total;
- oracle ceiling ≈ 0.595;
- random-review precision ≈ 0.034 (V2 prevalence).

Under fixed K, every extra fraud captured replaces a non-fraud in the queue, so 1 pp of Recall@2% ≈ **26.6 frauds per 28 days (≈ 0.95 per day)** ≈ **+1.7 pp Precision@K** (share of reviewed transactions that are fraud) ≈ +1.7 pp normalised recall (frauds reviewed ÷ the most that could be reviewed each day; [DR-002](DR-002-review-capacity-definition.md)).

| Candidate SESOI | Frauds per 28 days | Frauds per day (of ≈ 95 per day) | Precision@K equivalent | Share of oracle ceiling |
|---|---|---|---|---|
| 1 pp | 27 | 1.0 | ±1.7 pp | 1.7 % |
| **2 pp** | **53** | **1.9** | **±3.4 pp** | **3.4 %** |
| 3 pp | 80 | 2.9 | ±5.1 pp | 5.0 % |
| 5 pp | 133 | 4.8 | ±8.5 pp | 8.4 % |

### Resolution of the design (approximate; planning assumptions [I])

The standard error (SE) of a paired recall difference is approximately:

$$
\mathrm{SE} \approx \sqrt{\frac{p_{\mathrm{disc}} \times \mathrm{DE}}{F}}
$$

where:

- `p_disc` = share of frauds captured by exactly one of the two configurations (discordant frauds);
- `DE` = design effect from clustering;
- `F` = number of Test frauds (projected ≈ 2,660).

More discordance or more clustering widens the uncertainty; more frauds narrows it.

For the family effect, the planning assumption is that its SE equals the SE of a single paired contrast. The true SE depends on the correlation structure of the eight configurations' capture indicators, which is unknown before model scores exist; the bootstrap of [DR-006](DR-006-bootstrap-inference-strategy.md) estimates it directly. The table gives power as difference test at true δ = SESOI / TOST at true δ = 0, where δ is the true paired difference. The first value in each cell uses significance level α = 0.05, the second the first Holm step, α = 0.0167.

| SESOI | p_disc | DE = 1 | DE = 2 | DE = 4 |
|---|---|---|---|---|
| 1 pp | 0.10 | 0.37/0.00 · 0.22/0.00 | 0.21/0.00 · 0.11/0.00 | 0.13/0.00 · 0.06/0.00 |
| 2 pp | 0.05 | 1.00/1.00 · 0.99/0.99 | 0.90/0.89 · 0.81/0.74 | 0.64/0.49 · 0.47/0.14 |
| **2 pp** | **0.10** | **0.90/0.89 · 0.81/0.74** | **0.64/0.49 · 0.47/0.14** | **0.37/0.00 · 0.22/0.00** |
| 2 pp | 0.20 | 0.64/0.49 · 0.47/0.14 | 0.37/0.00 · 0.22/0.00 | 0.21/0.00 · 0.11/0.00 |
| 3 pp | 0.10 | 1.00/1.00 · 0.99/0.99 | 0.93/0.93 · 0.86/0.82 | 0.69/0.58 · 0.52/0.25 |
| 5 pp | 0.10 | 1.00/1.00 · 1.00/1.00 | 1.00/1.00 · 1.00/1.00 | 0.98/0.99 · 0.95/0.95 |

The TOST power figures are for a single contrast. H2's intersection-union test over four models has lower power than a single contrast, because all four must show equivalence.

## Alternatives considered

| Alternative | Why not chosen |
|---|---|
| No SESOI; plain difference tests for H1 and H2 | Non-significant results could not be interpreted, and an H2 "null" would invite "no effect" reasoning [REF-030]. |
| SESOI = 1 pp (close to Benavoli et al.'s ±1 % accuracy ROPE) | Operationally modest (≈ 1 fraud per day), and below the design's resolution: equivalence is essentially unprovable. That ROPE was set for accuracy across benchmark datasets, not Recall@capacity [REF-067]. |
| SESOI = 3 pp or 5 pp | Better powered, but labels 3–5 frauds per day at fixed workload as "negligible". Choosing a margin for power is not a justification of practical importance. |
| Relative SESOI (e.g. % of baseline recall) | The baseline recall is unknown before Test evaluation. A relative margin would depend on a Test outcome. |
| H1 as a within-tree contrast (CatBoost vs LightGBM) | Compares two members of one family; the model-family question of RQ1 is tree-based vs neural ([DR-011](DR-011-model-set-tree-based-and-neural-families.md)). The within-family contrasts are reported descriptively. |
| H1 as four pairwise model contrasts with Holm | A larger family lowers power, and pairwise contrasts do not answer the family question. |
| H2 as equivalence on the *averaged* weighting effect | Opposite effects in different models can cancel and produce false equivalence. |
| H2 tested within one family only | Leaves the weighting claim unverified for half of the model set. |
| H1 reframed as a superiority test of one family | Turns RQ1 into winner selection; RQ1 asks how the families differ. |
| No multiplicity correction (one family per RQ) | Defensible in some frameworks, but the protocol commits to family-wise control over its confirmatory claims. Holm keeps that commitment at modest cost. |
| Uncorrected bootstrap p-value 2 · min(P*(δ* ≤ 0), P*(δ* ≥ 0)) | Can exceed 1 when replicates equal 0 exactly, and can equal 0 with finite B. The (1 + count)/(B + 1) form with a cap at 1 avoids both. |

## Rationale

- **1 pp is below the design's resolution in every plausible scenario [I].** Equivalence at ±1 pp is essentially unprovable (TOST power ≈ 0 except under the most optimistic assumptions). A SESOI of 1 pp would make H2 untestable by design. It is kept as a sensitivity bound only.
- **2 pp is the smallest candidate that is both operationally meaningful and resolvable under favourable assumptions [I].**
  - 2 pp ≈ **53 additional frauds per 4 weeks (≈ 2 per day)** at *identical* review workload, ≈ 3.4 pp Precision@K, ≈ 3.4 % of the oracle ceiling.
  - For scale, ≈ 2 frauds per day is about what **K = 56 extra reviews per day chosen at random** would find at V2 prevalence (≈ 0.034 × 56 ≈ 1.9). A difference of this size between two modelling choices that cost the same is worth a practitioner's attention.
  - Under low-to-moderate discordance and clustering, the design can detect or rule out a 2 pp difference.
- **3 pp and 5 pp were rejected as the SESOI.**
  - They would declare differences of up to ≈ 3–5 frauds per day "negligible" at fixed workload.
  - Choosing a larger SESOI *in order to* gain power would misrepresent the smallest effect of interest [REF-063, REF-030].
  - They are reported as sensitivity bounds.
- **The SESOI is absolute (pp of recall), not relative.** The outcome is a paired difference of proportions on one fixed capacity and period, so the absolute scale is directly interpretable in frauds per day. Unlike H3's equivalence tolerance Δ on the alert rate ([DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md)), there is no target value to scale against.

**Accepted consequence.** Under central assumptions (p_disc ≈ 0.10, DE ≈ 2), power is limited: ≈ 0.47–0.64 for the H1 difference and ≈ 0.14–0.49 for single-contrast equivalence. **Inconclusive H1 or H2 results are a pre-declared, legitimate outcome.** They will be reported as such, never as "no difference".

## Consequences

- All eight configurations must be scored on identical Test data, and the paired bootstrap must resample jointly across them.
- The multiplicity structure for {H1, H2, H3} is fixed; H3 is tested once, on the configuration selected by [DR-010](DR-010-h3-confirmatory-configuration.md).
- Results may be inconclusive for H1 and/or H2. The write-up pre-commits to reporting them neutrally.

## Threats and limitations

- **The SESOI is a judgement.** No fraud-domain standard exists. The operational anchors (frauds per day at fixed workload, and the random-review comparison) are illustrative framings, not market or institutional values.
- **The power figures rest on assumed discordance and design effects.** No model scores exist yet, and the approximation treats the family effect's SE as equal to a single contrast's.
- **The operational meaning scales with the realised Test fraud count**, which is unknown and sealed. The SESOI itself is fixed on the recall scale, so it does not depend on that count.
- **The pseudo-entity key is heuristic** ([DR-001](DR-001-temporal-split-strategy.md), [DR-006](DR-006-bootstrap-inference-strategy.md)); residual dependence may remain.
- **The family effect is defined for these four models** ([DR-011](DR-011-model-set-tree-based-and-neural-families.md)); it is not a claim about every tree-based or neural method.
- **The qualification rules are descriptive.** They are not formal tests of the interaction or of within-family heterogeneity.

## References

REF-004, REF-005, REF-006, REF-019, REF-020, REF-028, REF-029, REF-030, REF-035, REF-044, REF-061, REF-062, REF-063, REF-066, REF-067, REF-089 — see [`../references.md`](../references.md).

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [D], [L], [I] | Evidence tags: dataset observation, literature, inference |
| #{ · } | Number of bootstrap replicates satisfying a condition |
| α | Significance level (family-wise 0.05; 0.0167 at the first Holm step) |
| p_(k), p̃_(k) | k-th smallest of p_H1, p_H2, p_H3; its Holm-adjusted p-value |
| i, k | Rank positions (1–3) of the ordered p-values in Holm's procedure |
| B | Number of bootstrap replicates (10,000) |
| b, u | Balanced and unweighted class weighting |
| Cat, LGB, RES, FTT | CatBoost, LightGBM, ResNet-style MLP, FT-Transformer |
| CI | Confidence interval |
| δ | A true paired recall difference (in power calculations) |
| δ_F | Model-family effect: neural mean minus tree mean of Recall@2% (H1 estimand) |
| δ*_F | δ_F in one bootstrap replicate |
| δ_W,m | Weighting simple effect of model m (H2 estimands) |
| DE | Design effect: variance inflation due to clustering |
| Δ | H3 equivalence tolerance on the alert rate (defined in DR-003) |
| F | Number of Test frauds (projected ≈ 2,660) |
| H1, H2, H3 | Confirmatory hypotheses: model family; class weighting; alert-rate transfer (DR-003) |
| H1₀, H1₁, H2₀, H2₁ | Null and alternative forms of H1 and H2 |
| H0a,m, H0b,m | One-sided null hypotheses of the H2 TOST for model m |
| K | Fixed daily review budget (56 at 2 % capacity) |
| m, w | Model (LGB, Cat, RES or FTT); weighting (u or b) |
| ML | Machine learning |
| MLP | Multilayer perceptron |
| FT-Transformer | Feature Tokenizer + Transformer |
| N, T | Neural family {RES, FTT}; tree-based family {LGB, Cat} |
| N_w, T_w | Mean Recall@2% of the neural and tree-based family at weighting w |
| p_disc | Share of frauds captured by exactly one of two configurations |
| p_H1, p_H2, p_H3 | p-values of H1, H2 and H3 |
| pp | Percentage points |
| Precision@K | Share of reviewed transactions that are fraud |
| R3 | Decision policy: daily top-K review |
| R^m,w | Pooled Test Recall@2% of model m with weighting w |
| Recall@2% | Share of frauds reviewed at the 2 % review capacity |
| ROPE | Region of practical equivalence |
| RQ1 | Research question 1 (model family and class weighting) |
| SE | Standard error |
| SESOI | Smallest Effect Size Of Interest (2 percentage points of Recall@2%) |
| TOST | Two One-Sided Tests |
| V2 | Validation 2 (relative days 127–154) |
