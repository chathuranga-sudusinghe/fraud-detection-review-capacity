# DR-005 — SESOI and H1/H2 inference plan

| Field | Value |
|---|---|
| Status | Accepted (pre-freeze). SESOI fixed at 2 percentage points of Recall@2 % by Issue #8, pending supervisor review. At the time of this record the primary bootstrap unit remained open (protocol open item 3). **Current status:** resolved by [DR-006](DR-006-bootstrap-inference-strategy.md) (Issue #14). |
| Date | 2026-09-24 |
| Evidence source | `notebooks/01_dataset_feasibility_audit.ipynb`, §§6–7, 15; development-period (V2) projections; equivalence-testing and practical-significance literature |
| Related | [DR-001](DR-001-temporal-split-strategy.md), [DR-002](DR-002-review-capacity-definition.md), [DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md), [DR-006](DR-006-bootstrap-inference-strategy.md) (later: resolves the bootstrap unit left open here), [protocol](../protocol.md) |

## Context

Confirmatory hypothesis H1 compares model families (CatBoost vs LightGBM), and hypothesis H2 compares class weighting (balanced vs unweighted). Both use the primary outcome: **Recall@2 % (the share of frauds reviewed at 2 % review capacity) under the daily top-K policy R3 (review the K highest-scoring transactions each relative day), with K (fixed daily review budget) = 56 reviews per relative day** ([DR-002](DR-002-review-capacity-definition.md)).

The feasibility audit found four things:
- **[D]** At 2 % capacity the oracle recall ceiling is ≈ 0.595 (V2).
- **[D]** About 2,660 Test frauds are projected from V2 prevalence. This is a projection; Test labels are sealed.
- **[D]** Frauds cluster by heuristic pseudo-entity, with an upper-bound design effect (DE: the factor by which clustering inflates the variance relative to independent observations) of ≈ 4.3 in Validation 2 (V2, relative days 127–154).
- **[I]** Model-family and weighting differences may be small. The minimum detectable paired difference was ≈ 1.2–5.6 percentage points (pp), depending on discordance and clustering.

Class weighting is expected mainly to distort calibration, with little effect on ranking [REF-019, REF-020], so a small or null H2 effect is plausible. A non-significant result must not be read as "no effect" [REF-030]. A **smallest effect size of interest (SESOI)** therefore had to be fixed before any Test evaluation, and the H1/H2 test logic finalised.

Before this decision, H1 and H2 were ambiguous about *which* configurations they compare. The 2 × 2 design (model family × weighting) contains two model-family contrasts and two weighting contrasts. This record resolves that ambiguity.

## Decision

### 1. SESOI

> **SESOI = 2 percentage points (0.02) of Recall@2 %**, on the absolute recall scale, for both H1 and H2.

### 2. Estimands

Rᶠʷ (also written R^f,w) is the pooled Test Recall@2 % of model family f ∈ {LGB (LightGBM), Cat (CatBoost)} with weighting w ∈ {u (unweighted), b (balanced)}. It is the share of Test frauds that fall within their day's top-56 scores, over relative days 155–182. All four configurations are evaluated on the **same** Test frauds, so every contrast is **paired**.

| Estimand | Definition | Used by |
|---|---|---|
| δ_M (model-family main effect) | ½[(R^Cat,u − R^LGB,u) + (R^Cat,b − R^LGB,b)] | H1 (confirmatory) |
| δ_W,LGB and δ_W,Cat (weighting simple effects) | R^LGB,b − R^LGB,u and R^Cat,b − R^Cat,u | H2 (confirmatory) |
| Interaction | (R^Cat,b − R^LGB,b) − (R^Cat,u − R^LGB,u) | Descriptive |
| Cell contrasts | R^Cat,u − R^LGB,u and R^Cat,b − R^LGB,b | Descriptive |

The two confirmatory estimands, displayed:

$$
\begin{aligned}
\delta_M &= \frac{1}{2}\left[ \left(R^{\mathrm{Cat},u} - R^{\mathrm{LGB},u}\right) + \left(R^{\mathrm{Cat},b} - R^{\mathrm{LGB},b}\right) \right] \\
\delta_{W,f} &= R^{f,b} - R^{f,u} \qquad \text{for } f \in \lbrace \mathrm{LGB}, \mathrm{Cat} \rbrace
\end{aligned}
$$

where:

- `δ_M` = model-family main effect: the CatBoost-minus-LightGBM recall difference, averaged over the two weighting conditions;
- `δ_W,f` = weighting simple effect in family f: balanced minus unweighted recall;
- `R^f,w` = pooled Test Recall@2 % of family f with weighting w.

δ_M is positive when CatBoost reviews more frauds than LightGBM; δ_W,f is positive when balanced weighting helps family f.

### 3. H1 — model-family effect: a difference test plus pre-specified practical interpretation

> **H1₀:** Averaged over the unweighted and balanced-weighting conditions, CatBoost and LightGBM do not differ in Recall@2% on the Test period (δ_M = 0).
>
> **H1₁:** Averaged over the unweighted and balanced-weighting conditions, CatBoost and LightGBM differ in Recall@2% on the Test period (δ_M ≠ 0).

H1 stays a two-sided **difference** test, because research question RQ1 asks *whether and how* the families differ. It is not reframed as a superiority or winner-selection test.

In addition, the magnitude of δ_M is interpreted against ±SESOI using its **90 % confidence interval (CI)**. This interpretation is **pre-specified but not a separate confirmatory test**:

| H1 difference test (Holm-adjusted: corrected for multiple testing by Holm's method) | 90 % CI for δ_M relative to ±0.02 | Reported interpretation |
|---|---|---|
| Significant | Entirely beyond +0.02 or entirely beyond −0.02 | Evidence of a difference of practically meaningful size |
| Significant | Crosses ±0.02 | Evidence of a difference; whether it reaches practical size is uncertain |
| Significant | Entirely within (−0.02, +0.02) | Evidence of a difference that is too small to matter in practice |
| Not significant | Entirely within (−0.02, +0.02) | Data compatible only with practically negligible differences |
| Not significant | Extends beyond ±0.02 | Inconclusive: uncertainty too large to distinguish practically meaningful effects |

If the 95 % CI of the interaction excludes 0 **and** its point estimate exceeds the SESOI in magnitude, the main-effect interpretation is qualified, and the two cell contrasts are reported as the primary description of the family difference.

### 4. H2 — class-weighting effect: equivalence (intersection-union TOST, Two One-Sided Tests)

> **H2₀:** For at least one model family (LightGBM or CatBoost), balanced class weighting changes Recall@2% by 2 percentage points or more relative to the corresponding unweighted model (|δ_W,f| ≥ 0.02 for at least one f).
>
> **H2₁:** For both LightGBM and CatBoost, balanced class weighting changes Recall@2% by less than 2 percentage points relative to the corresponding unweighted model (|δ_W,f| < 0.02 for both f).

**Why equivalence.**
- The substantive question for H2 is whether weighting matters *materially* for capacity-constrained recall. The literature expectation is that it mostly does not [REF-019, REF-020].
- An equivalence test puts the burden of proof on the "no meaningful effect" claim, and prevents non-significant results being read as "no effect" [REF-030, REF-063, REF-061].

**Why both families (intersection-union), not the averaged effect.**
- An average can cancel opposite effects. For example, +3 pp for LightGBM and −3 pp for CatBoost averages to 0 and would wrongly suggest equivalence.
- Requiring equivalence **in each family** is an intersection-union test. Each component TOST is performed at the full level α, with no further split [REF-062].

**TOST logic.**
- For each family f there are two one-sided tests: H0a,f: δ_W,f ≤ −0.02 and H0b,f: δ_W,f ≥ +0.02.
- H2₁ is concluded only if **all four** one-sided nulls are rejected at the Holm-assigned level α_H2 (the significance level Holm's procedure assigns to H2). Equivalently, the (1 − 2α_H2) CIs for both δ_W,LGB and δ_W,Cat lie entirely inside the open interval (−0.02, +0.02).
- p_H2 (the H2 p-value) = the maximum of the four one-sided p-values.
- The conventional two-sided difference CIs for δ_W,f are also reported, **descriptively**, so a statistically detectable but practically negligible weighting effect remains visible.

## Evidence

### Literature findings

- **[L] There is no fraud-domain standard for a practically meaningful Recall@capacity difference.** A targeted search found none. Fraud studies report metric differences but do not define a minimum meaningful difference. The capacity-evaluation literature [REF-004, REF-005, REF-006] defines the metric, not a margin.
- **[L] The concept is established in other fields.** A *minimal clinically important difference* is defined by practical, not statistical, meaning: "the smallest difference in score in the domain of interest which patients perceive as beneficial and which would mandate … a change in the patient's management" [REF-066]. This project adapts the principle, not any clinical value.
- **[L] Equivalence bounds express the SESOI.** Equivalence is tested by TOST, which corresponds to a (1 − 2α) CI inside the bounds [REF-061, REF-030, REF-063]. An intersection-union test needs no α split among its components [REF-062].
- **[L] The only machine-learning (ML) precedent is analyst-chosen.** Benavoli et al. treat a mean accuracy difference of less than 1 % as "practically equivalent" (a region of practical equivalence, ROPE, of ±0.01) for generic classifier benchmarks [REF-067]. That is an analyst-chosen value for **accuracy averaged over datasets**, not for Recall@capacity in fraud, so it is **not** adopted as the SESOI here.

**Consequence.** The SESOI is a **pre-specified researcher judgement**, justified below. It is not a literature convention.

### Operational meaning of candidate SESOIs (projection; no Test data)

Inputs [D]:
- about 2,660 projected Test frauds;
- K = 56 reviews/day over 28 days, i.e. 1,568 reviews in total;
- oracle ceiling ≈ 0.595;
- random-review precision ≈ 0.034 (V2 prevalence).

Under fixed K, every extra fraud captured replaces a non-fraud in the queue, so 1 pp of Recall@2 % ≈ **26.6 frauds per 28 days (≈ 0.95 per day)** ≈ **+1.7 pp Precision@K** (share of reviewed transactions that are fraud) ≈ +1.7 pp normalised recall (frauds reviewed ÷ the most that could be reviewed each day).

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

For main effects, this is conservatively assumed to equal the SE of a single paired contrast. The table gives power as difference test at true δ = SESOI / TOST at true δ = 0, where δ is the true paired difference. The first value in each cell uses significance level α = 0.05, the second the first Holm step, α = 0.0167.

| SESOI | p_disc | DE = 1 | DE = 2 | DE = 4 |
|---|---|---|---|---|
| 1 pp | 0.10 | 0.37/0.00 · 0.22/0.00 | 0.21/0.00 · 0.11/0.00 | 0.13/0.00 · 0.06/0.00 |
| 2 pp | 0.05 | 1.00/1.00 · 0.99/0.99 | 0.90/0.89 · 0.81/0.74 | 0.64/0.49 · 0.47/0.14 |
| **2 pp** | **0.10** | **0.90/0.89 · 0.81/0.74** | **0.64/0.49 · 0.47/0.14** | **0.37/0.00 · 0.22/0.00** |
| 2 pp | 0.20 | 0.64/0.49 · 0.47/0.14 | 0.37/0.00 · 0.22/0.00 | 0.21/0.00 · 0.11/0.00 |
| 3 pp | 0.10 | 1.00/1.00 · 0.99/0.99 | 0.93/0.93 · 0.86/0.82 | 0.69/0.58 · 0.52/0.25 |
| 5 pp | 0.10 | 1.00/1.00 · 1.00/1.00 | 1.00/1.00 · 1.00/1.00 | 0.98/0.99 · 0.95/0.95 |

The TOST power figures are for a single contrast. H2's intersection-union test over two families has somewhat lower power.

## Rationale for SESOI = 2 pp

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

## Statistical inference plan (H1, H2)

| Element | Specification |
|---|---|
| **Estimands** | δ_M (H1); δ_W,LGB and δ_W,Cat (H2); interaction and cell contrasts (descriptive). All are paired differences in pooled Test Recall@2 % under R3, K = 56. |
| **Primary uncertainty method** | **Joint paired bootstrap.** Each replicate resamples the same units for all four configurations, preserving the pairing, and recomputes every estimand. B (number of bootstrap replicates) = 10,000; percentile CIs [REF-044]. |
| **Clustering / resampling unit** | **Not fixed by this decision; protocol open item 3 remained open at the time of this record.** *Current status: resolved by [DR-006](DR-006-bootstrap-inference-strategy.md), which adopts a two-way (pigeonhole) bootstrap over relative days × fine pseudo-entities as primary, with an Owen-corrected variance and a day-level moving-block bootstrap as sensitivity analyses.* Candidates considered here: (a) heuristic pseudo-entity clusters, resampling frauds' capture indicators by cluster with daily top-K membership fixed from the realised rankings [REF-036]; (b) moving blocks of relative days (whole days resampled; top-K recomputed within day) [REF-037, REF-064]. Transaction-level or fraud-level resampling that assumes independence is **not** acceptable as primary. The SESOI decision does not depend on this choice: the power scenarios above bracket DE = 1–4. |
| **CI types** | 95 % CI for difference estimates (H1, descriptive contrasts). 90 % CI for practical interpretation against ±SESOI (H1). (1 − 2α_H2) CIs for the H2 TOST decision. |
| **p-values** | H1: two-sided bootstrap p-value p_H1 for δ_M = 0 (displayed below the table). H2: p_H2 = max of the four one-sided bootstrap p-values (intersection-union). H3: as defined in DR-003. |
| **Role of McNemar** | **Secondary and descriptive only.** An exact McNemar test [REF-035, REF-028] is reported for each of the four cell-level paired contrasts. It assumes independent frauds, which the clustering evidence contradicts, so it is **not** in the confirmatory family. |
| **Role of TOST** | Confirmatory for **H2** (intersection-union over both families) and **H3** (DR-003). Descriptive for H1's practical interpretation. |
| **Multiplicity** | **Holm** [REF-029] at family-wise α = 0.05 over the confirmatory family {H1, H2, H3}, using p_H1, p_H2 and p_H3 (H3 is the alert-rate transfer hypothesis of DR-003). Holm controls the family-wise error rate under arbitrary dependence and can combine difference and equivalence p-values, since each is a valid p-value for its own null. H2 is one family member because its intersection-union structure needs no internal split [REF-062]. If H3 is tested on more than one model configuration (protocol open item 7), each H3 test becomes an additional family member. |
| **Is Holm still appropriate?** | Yes. It is conservative, but the confirmatory claims are few and pre-specified, and conservatism suits a single-holdout study. The cost is lower power at the first Holm step (α = 0.0167); this is shown in the power table. |
| **Effect-size reporting** | For every estimand: the point estimate in pp; frauds per 28 days and per day; the Precision@K equivalent; normalised recall; CIs. Sensitivity: H1/H2 interpretations are also shown against SESOIs of 1, 3 and 5 pp (not confirmatory). |

The H1 bootstrap p-value is:

$$
p_{\mathrm{H1}} = 2 \cdot \min\left( P^*(\delta^*_M \le 0),\ P^*(\delta^*_M \ge 0) \right)
$$

where:

- `δ*_M` = the value of δ_M in one bootstrap replicate;
- `P*( · )` = the proportion of bootstrap replicates satisfying the condition.

It doubles the smaller tail of the bootstrap distribution beyond zero, giving a two-sided test of δ_M = 0.

## Alternatives considered

| Alternative | Why not chosen |
|---|---|
| No SESOI; plain difference tests for H1 and H2 | Non-significant results could not be interpreted, and an H2 "null" would invite "no effect" reasoning [REF-030]. |
| SESOI = 1 pp (close to Benavoli et al.'s ±1 % accuracy ROPE) | Operationally modest (≈ 1 fraud per day), and below the design's resolution: equivalence is essentially unprovable. That ROPE was set for accuracy across benchmark datasets, not Recall@capacity [REF-067]. |
| SESOI = 3 pp or 5 pp | Better powered, but labels 3–5 frauds per day at fixed workload as "negligible". Choosing a margin for power is not a justification of practical importance. |
| Relative SESOI (e.g. % of baseline recall) | The baseline recall is unknown before Test evaluation. A relative margin would depend on a Test outcome. |
| H2 as equivalence on the *averaged* weighting effect | Opposite effects in the two families can cancel and produce false equivalence. |
| H1 and H2 as four cell-level tests with Holm | A larger family lowers power. The main effect (H1) plus intersection-union (H2) answers the RQ1 questions with three confirmatory tests, while cell contrasts remain reported. |
| H1 reframed as a superiority test of one family | Turns RQ1 into winner selection; RQ1 asks how the families differ. |
| No multiplicity correction (one family per RQ) | Defensible in some frameworks, but the protocol already commits to family-wise control over its confirmatory claims. Holm keeps that commitment at modest cost. |

## Consequences

- Protocol open item 2 (SESOI) is resolved, pending supervisor review.
- Open item 3 (primary bootstrap unit for H1/H2) **remained open** at the time of this record. *Current status: resolved by [DR-006](DR-006-bootstrap-inference-strategy.md) (Issue #14), pending supervisor review.*
- The multiplicity structure for {H1, H2, H3} is now defined. Open item 7 is narrowed to *which* configuration(s) H3 is tested on.
- All four configurations must be scored on identical Test data, and the paired bootstrap must resample jointly across them.
- Results may be inconclusive for H1 and/or H2. The write-up must pre-commit to reporting them neutrally.

## Threats and limitations

- **The SESOI is a researcher judgement.** No fraud-domain standard exists. The operational anchors (frauds per day at fixed workload, and the random-review comparison) are illustrative framings, not market or institutional values.
- **The power figures rest on assumed discordance and design effects.** No model scores exist yet, and the approximation treats the main effect's SE as equal to a single contrast's.
- **The operational meaning scales with the realised Test fraud count**, which is unknown and sealed. The SESOI itself is fixed on the recall scale, so it does not depend on that count.
- **The pseudo-entity key is heuristic** ([DR-001](DR-001-temporal-split-strategy.md)). If it is chosen as the bootstrap unit, residual dependence may remain.
- **The interaction qualification rule is descriptive.** It is not a formal test of the interaction.

## References

REF-004, REF-005, REF-006, REF-019, REF-020, REF-028, REF-029, REF-030, REF-035, REF-036, REF-037\*, REF-044, REF-061, REF-062, REF-063, REF-064, REF-066, REF-067 — see [`../references.md`](../references.md).

\* Partially verified (page range pending).

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [D], [L], [I] | Evidence tags: dataset observation, literature, inference |
| α | Significance level (family-wise 0.05; 0.0167 at the first Holm step) |
| α_H2 | Significance level assigned to H2 by Holm's procedure |
| B | Number of bootstrap replicates (10,000) |
| b, u | Balanced and unweighted class weighting |
| Cat, LGB | CatBoost, LightGBM |
| CI | Confidence interval |
| δ | A true paired recall difference (in power calculations) |
| δ_M | Model-family main effect (H1 estimand) |
| δ*_M | δ_M in one bootstrap replicate |
| δ_W,f | Weighting simple effect in model family f (H2 estimands) |
| DE | Design effect: variance inflation due to clustering |
| Δ | H3 equivalence tolerance on the alert rate (defined in DR-003) |
| F | Number of Test frauds (projected ≈ 2,660) |
| f, w | Model family (LGB or Cat); weighting (u or b) |
| H1, H2, H3 | Confirmatory hypotheses: model family; class weighting; alert-rate transfer (DR-003) |
| H1₀, H1₁, H2₀, H2₁ | Null and alternative forms of H1 and H2 |
| K | Fixed daily review budget (56 at 2 % capacity) |
| ML | Machine learning |
| P*( · ) | Proportion of bootstrap replicates satisfying a condition |
| p_disc | Share of frauds captured by exactly one of two configurations |
| p_H1, p_H2, p_H3 | p-values of H1, H2 and H3 |
| pp | Percentage points |
| Precision@K | Share of reviewed transactions that are fraud |
| R3 | Decision policy: daily top-K review |
| R^f,w (Rᶠʷ) | Pooled Test Recall@2 % of family f with weighting w |
| Recall@2 % | Share of frauds reviewed at the 2 % review capacity |
| ROPE | Region of practical equivalence |
| RQ1 | Research question 1 (model and weighting) |
| SE | Standard error |
| SESOI | Smallest Effect Size Of Interest (2 percentage points of Recall@2 %) |
| TOST | Two One-Sided Tests |
| V2 | Validation 2 (relative days 127–154) |
