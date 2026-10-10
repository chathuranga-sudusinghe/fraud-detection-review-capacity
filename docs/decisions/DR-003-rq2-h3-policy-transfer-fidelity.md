# DR-003 — RQ2 / H3: policy-transfer fidelity

| Field | Value |
|---|---|
| Status | Accepted (pre-freeze); supervisor review pending (Issue #56) |
| Date | 2026-09-23; revised 2026-09-24 (Issue #7: tolerance Δ = 0.004, ±20 % relative) and 2026-10-09 (Issue #54: Holm-level H3 decision; zero-alert days in the rate/volume decomposition) |
| Evidence source | `notebooks/01_dataset_feasibility_audit.ipynb`, §§3, 6, 14, 17. For Δ: V2 daily transaction counts (label-free) and the equivalence-testing literature |
| Related | [DR-001](DR-001-temporal-split-strategy.md), [DR-002](DR-002-review-capacity-definition.md), [DR-005](DR-005-sesoi-and-h1-h2-inference.md) (multiplicity plan), [DR-006](DR-006-bootstrap-inference-strategy.md) (H1/H2 bootstrap), [DR-009](DR-009-overflow-handling-for-threshold-policies.md) (overflow rule and workload definitions), [DR-010](DR-010-h3-confirmatory-configuration.md) (H3 configuration), [protocol](../protocol.md) |

## Context

Research question 2 (RQ2) concerns what happens when a validation-selected score threshold is applied to later data. A question framed as "does the threshold preserve the intended review workload?", with a point-null hypothesis, **mixes two different mechanisms**, as the feasibility audit shows:

1. **Alert-rate fidelity.** Does the score threshold τ (a model-score cut-off), chosen so that 2 % of Validation 2 (V2, relative days 127–154) transactions exceed it, still flag about 2 % of transactions in a later period? This depends on **score-distribution and composition drift**, i.e. on the model and the data.
2. **Fixed daily-budget fidelity.** Does the *number* of flagged transactions per day match the fixed daily review budget K (K = 56 reviews per relative day at 2 % capacity; [DR-002](DR-002-review-capacity-definition.md))? This depends on alert-rate drift **and** on **daily volume variation**, which exists whatever the model does.

**Evidence [D].**
- Daily transaction volume is non-stationary: coefficient of variation 0.26 over 182 days.
- Even a *perfectly* transferring 2 % rate threshold would produce V2 daily review counts between **−27 % and +39 %** of the fixed K = 56 (SD 16 %), purely from volume variation.
- Composition also shifts: identity coverage falls from 27.1 % to 17.5 % between Train and V2, and fraud prevalence among identity-present rows changes.

The rate component is therefore a genuine object of study.

A point-null hypothesis on the rate is **uninformative**. With about 78,500 Test transactions, almost any non-zero deviation from exactly 2 % would be "significant", and a non-rejection would not show that the rate is preserved [REF-030].

## Decision

### RQ2

> **RQ2: When a score threshold selected on validation data is transferred to a later temporal period, how well does it preserve the intended alert rate, and how does the resulting daily review workload compare with the fixed review budget?**

### H3 — Alert-rate transfer fidelity (equivalence formulation)

**Definitions.**
- τ₂ (the 2 % rate threshold) is the score threshold chosen on V2 so that 2 % of V2 transactions have score ≥ τ₂. This is decision policy R2 (the capacity-rate threshold).
- The **realised alert rate** r is the proportion of Test transactions with score ≥ τ₂, pooled over the 28 Test days:

  $$
  r = \frac{\sum_d n_d}{\sum_d N_d}
  $$

  where:

  - `d` = a Test relative day (155–182);
  - `n_d` = number of transactions on day d with score ≥ τ₂;
  - `N_d` = total number of transactions on day d;
  - `Σ_d` = sum over the 28 Test days.

  r is the share of all Test transactions that the transferred threshold flags.

- The **equivalence band** is built from the tolerance Δ (the equivalence tolerance: r preserves the target when it deviates from 2 % by less than Δ):

  $$
  \begin{aligned}
  \Delta &= 0.20 \times 0.02 \\
  &= 0.004 \\[6pt]
  \text{equivalence band} &= (0.02 - \Delta,\ 0.02 + \Delta) \\
  &= (0.016,\ 0.024) \\
  &= (1.6\%,\ 2.4\%)
  \end{aligned}
  $$

  Δ is a **relative tolerance of 20 % of the target rate**. See [Equivalence tolerance Δ](#equivalence-tolerance-δ).

> **H3₀:** The realised alert rate under the validation-selected 2% threshold lies outside the pre-specified equivalence band around the 2% target (|r − 0.02| ≥ Δ, where Δ = 0.004, i.e. ±20% of the target).
>
> **H3₁:** The realised alert rate under the validation-selected 2% threshold lies within the pre-specified equivalence band around the 2% target (|r − 0.02| < Δ, where Δ = 0.004, i.e. ±20% of the target).

**Test.** Two One-Sided Tests (TOST) using a day-level moving-block bootstrap, decided by the Holm-adjusted p-value of H3 ([DR-005](DR-005-sesoi-and-h1-h2-inference.md)). Details are in [Statistical formulation of H3](#statistical-formulation-of-h3).

**Configuration and multiplicity.**
- H3 is confirmatory for a single *H3 configuration*: the one of the eight model × weighting configurations with the highest pooled V2 Recall@2% under R3 (policy R3, daily top-K; K = 56), selected by a pre-specified rule before the Test period is unsealed ([DR-010](DR-010-h3-confirmatory-configuration.md)). H3 results for the other seven configurations are descriptive.
- H3 is one member of the confirmatory family {H1 (model family), H2 (class weighting), H3}, controlled by Holm's procedure at family-wise α = 0.05 [REF-029] ([DR-005](DR-005-sesoi-and-h1-h2-inference.md)).

### Daily-budget fidelity — a separate operational outcome (not a confirmatory hypothesis)

For each Test relative day d, with N_d the day's transaction volume, n_d the number of transactions with score ≥ τ₂, and K = 56:

| Statistic | Definition |
|---|---|
| Threshold-generated alerts (review demand) | n_d |
| Deviation from fixed K | n_d − K |
| Relative deviation | (n_d − K) / K |
| Absolute deviation | \|n_d − K\|; reported as a mean and maximum over days |
| Days above K (overflow) | % of days with n_d > K; total overflow Σ max(0, n_d − K) |
| Days below K (underuse) | % of days with n_d < K; total unused capacity Σ max(0, K − n_d) |
| Rate / volume decomposition | See the display equation below the table. It separates the *rate* component (model and data drift) from the *volume* component |

n_d is the number of **threshold-generated alerts**, i.e. review demand; it is not a count of reviews actually performed. Actual hard-capacity review is R3's min(K, N_d) reviews per day. [DR-009](DR-009-overflow-handling-for-threshold-policies.md) formalises the overflow and unused-capacity sums as O_d = max(n_d − K, 0) and U_d = max(K − n_d, 0) (in DR-009's notation, A_d,R2 = n_d at 2 %), adds the load ratio n_d / K, and applies the same definitions to R0 and R1.

The rate / volume decomposition is:

$$
\begin{aligned}
\log\left(\frac{n_d}{K}\right) &\approx \log\left(\frac{r_d}{0.02}\right) + \log\left(\frac{N_d}{\bar{N}_{\mathrm{V2}}}\right) \\
r_d &= \frac{n_d}{N_d}
\end{aligned}
$$

where:

- `r_d` = realised alert rate on day d;
- `N̄_V2` = mean daily transaction volume in V2 (so that K ≈ 0.02 × N̄_V2);
- `log` = natural logarithm.

The first term on the right is the *rate* component (model and data drift); the second is the *volume* component, which exists whatever the model does.

**Days with no alerts.** On a day with n_d = 0 the logarithm is undefined. Such days are excluded from the logarithmic summary, counted and reported separately, and their deviation from K is attributed entirely to the rate component (r_d = 0). Every relative day has at least 2,048 transactions (feasibility notebook §3), so the volume term is always defined.

These statistics are reported descriptively with bootstrap intervals where appropriate. Secondary descriptive consequences, as bounded by [DR-009](DR-009-overflow-handling-for-threshold-policies.md):
- frauds below the threshold, which are false negatives of the threshold classifier;
- descriptively and conditional on score-priority ordering, "flagged frauds in excess demand". Frauds among alerts in excess of K are **not** counted as missed, because the labels do not show what would happen to excess alerts;
- review capacity left unused on underuse days.

### Equivalence tolerance Δ

#### Literature findings

- **[L] No fraud-specific or alerting-specific standard exists.** A targeted search found no peer-reviewed standard for an acceptable deviation of a transferred alert rate from its target, in fraud detection or in alert-budget / top-k evaluation. The fraud evaluation literature establishes capacity-based evaluation [REF-004, REF-005, REF-006] but does not define a tolerance for alert-rate drift. The only related items found were non-peer-reviewed preprints and patents, which are not used as evidence.
- **[L] The general framework is well established.**
  - Equivalence is tested with the TOST procedure. At α = 0.05 this is equivalent to requiring the 90 % confidence interval (CI) to lie inside the equivalence bounds [REF-061, REF-062].
  - TOST is an intersection-union test, so the two one-sided tests are each carried out at α without splitting it [REF-062].
  - The equivalence bounds express the **Smallest Effect Size Of Interest (SESOI)**, which the researcher must justify [REF-030, REF-063].
- **[L] Precedent for relative (ratio) margins.** In regulatory bioequivalence, the acceptance interval is defined on the *ratio* of test to reference (80.00–125.00 %), and the 90 % CI of the ratio must lie within it [REF-065]. This is cited only as a precedent that equivalence margins for positive quantities are commonly defined **relative to a reference value**. It gives **no** support for any particular magnitude in fraud-alerting, because that margin was set for a different field and purpose.

**Consequence.** The magnitude of Δ cannot be taken from the literature. It is a **pre-specified researcher judgement**, justified below by the research design and by development-period evidence that uses no labels and no Test data.

#### Candidate definitions compared

| Definition | Form | Interpretability | Statistical validity | Transferability across the capacity grid | Dependence on volume | Usefulness for RQ2 |
|---|---|---|---|---|---|---|
| **A. Absolute** | 2 % ± x percentage points | Clear, but the same x means very different things at 1 % and 10 % capacity | Valid for TOST | Poor: x must be re-chosen per capacity | None | Adequate for the 2 % test only |
| **B. Relative** | 2 % × (1 ± x), x = relative tolerance | Direct: x is the % over- or under-run of the intended alert volume | Valid for TOST (a linear transformation of the absolute bound at a fixed target) | **Good:** the same x applies at every capacity | None | **Best**: it is the quantity a review team plans against |
| **C. Budget-derived** | Δ such that the expected daily reviews at a reference volume differ from K by ≤ y reviews | Very operational (reviews/day) | Valid only if the reference volume is fixed in advance; using Test volume would mix volume drift back into a rate hypothesis, which is exactly what DR-003 separates | Poor: y must be re-chosen per K | **Yes**: requires a volume assumption | Useful as an **interpretation** of B, not as a definition |

**Decision [I].** Use **B (relative)**. Here x is the relative tolerance and y (in C) a tolerance in reviews per day. At the V2 reference volume, B and C coincide: a relative rate deviation of x changes expected daily reviews by x × K. The relative form keeps the rate hypothesis free of volume assumptions, and it can be reported unchanged, descriptively, at 1 %, 5 % and 10 %.

#### Choice of magnitude: x = 20 % (Δ = 0.004)

Evidence used (V2 daily transaction counts only; no labels; no Test data) [D]:
- mean V2 volume 2,779.4 per day, so K = 56 at 2 %;
- volume-only deviation of daily workload from K, assuming a perfectly transferring 2 % rate: median 8.2 %, upper quartile 16.7 %, 90th percentile 26.5 %, standard deviation (SD) 15.7 %, range −26.9 % to +39.1 %;
- V2 days on which volume alone moves workload by at least the given margin: **13 of 28** for 10 %, **5 of 28** for 20 %, **4 of 28** for 25 %.

| Candidate x | Band for r | Reviews/day at K = 56 | Assessment [I] |
|---|---|---|---|
| 10 % | 1.8–2.2 % | ± 5.6 | **Too tight.** Smaller than volume-driven swings on nearly half of V2 days (13/28), so it is operationally invisible. The 90 % CI half-width may itself reach 6–19 % relative (below), so equivalence could be unprovable even for a threshold that transfers perfectly. |
| **20 %** | **1.6–2.4 %** | **± 11** | **Chosen.** A *persistent* shift of this size is larger than volume-driven day-to-day variation on 23 of 28 V2 days (82 %), so the team would notice it as a systematic under- or over-staffing problem rather than ordinary noise. It remains achievable under plausible day-level variation (below). |
| 25 % | 1.5–2.5 % | ± 14 | Defensible, and close to the bioequivalence ratio convention (0.80–1.25). Less strict: it tolerates a persistent quarter-budget over-run, exceeded by volume alone on only 4 of 28 days. Kept as a sensitivity bound. |
| 50 % | 1–3 % | ± 28 | **Too loose.** Admits persistent workload changes of half the budget; "equivalence" would be operationally meaningless. |

**Rationale for 20 % [I].**
- A systematic shift in the alert rate affects **every** day, whereas volume variation fluctuates around zero.
- A tolerance at the upper end of routine daily variation (between the V2 upper quartile, 16.7 %, and 90th percentile, 26.5 %) is therefore the largest shift that could plausibly go unnoticed by a team staffed for K, and the smallest that would clearly matter operationally.
- 20 % is a round value in that interval, about 1.3 SD of the volume-driven variation.

**Feasibility check [I].**
- Under binomial sampling alone, the 90 % CI half-width for r at about 78,000 Test transactions is about 4 % of the target.
- Day-level variation in the rate widens this. Under a first-order autoregressive, AR(1), approximation for the mean of 28 daily rates, the half-width is approximately:

  | day-level relative SD \ lag-1 autocorrelation | 0 | 0.3 | 0.6 |
  |---|---|---|---|
  | 10 % | 3.1 % | 4.2 % | 6.2 % |
  | 20 % | 6.2 % | 8.5 % | 12.4 % |
  | 30 % | 9.3 % | 12.7 % | 18.7 % |

- A day-level SD of at least about 13 % is unavoidable: per-day binomial noise at about 2,800 transactions per day is already about that size.
- With x = 20 %, H3₁ can be concluded under moderate day-level variation, and only under near-perfect transfer in the most pessimistic scenario.
- These are planning assumptions, not measured Test quantities. No model scores exist yet.

**Sensitivity reporting (non-confirmatory).**
- Report the 90 % CI for r against the alternative bounds ±10 %, ±25 % and ±50 %.
- Report the smallest relative tolerance at which equivalence would be concluded, so readers can apply their own margin.
- These do not change the confirmatory H3 decision, which uses ±20 % only.


### Statistical formulation of H3

| Element | Specification |
|---|---|
| **Estimand** | r = Σ_d n_d / Σ_d N_d over Test relative days 155–182. r is the pooled share of Test transactions with score ≥ τ₂. Inference is **conditional on the trained model and the frozen τ₂**: H3 concerns the threshold actually deployed. |
| **Equivalence bounds** | (0.016, 0.024), i.e. relative deviation r / 0.02 − 1 ∈ (−0.20, +0.20). Open bounds, so a CI endpoint exactly on a bound does **not** support H3₁. |
| **Null hypothesis H3₀** | r ≤ 0.016 **or** r ≥ 0.024 (\|r − 0.02\| ≥ 0.004) |
| **Alternative H3₁** | 0.016 < r < 0.024 (\|r − 0.02\| < 0.004) |
| **Test** | TOST: H0a: r ≤ 0.016 and H0b: r ≥ 0.024 (the two one-sided null hypotheses), with one-sided bootstrap p-values p_a = (1 + #{r* ≤ 0.016}) / (B + 1) and p_b = (1 + #{r* ≥ 0.024}) / (B + 1), where r* is r in one bootstrap replicate. p_H3 = max(p_a, p_b) enters Holm's procedure [REF-061, REF-062, REF-029]. H3₁ is concluded only if the **Holm-adjusted p-value of H3 is at most 0.05**. |
| **CI interpretation** | The 90 % CI is reported in every case. The 90 % CI describes uncertainty only and does not determine rejection: the decision uses the Holm-adjusted p-value, so a 90 % CI inside the band can accompany a non-rejection after multiplicity adjustment. The unadjusted and the Holm-adjusted p-value are both reported. Inside the band: rate preserved within ±20 %. Entirely outside: rate not preserved. Overlapping a bound: **inconclusive**, which must not be reported as "preserved". |
| **Analysis unit** | The **relative day**. Flags are deterministic given the score, but a day's scores share that day's transaction composition (e.g. identity coverage, product mix), so transactions within a day are not independent draws [D; DR-001]. A transaction-level binomial interval would be anti-conservative. |
| **Dependence between days** | Days are autocorrelated (daily-volume lag-1 autocorrelation 0.75 over days 1–182; 0.27 within V2 [D]). The CI therefore uses a **moving-block bootstrap over days** [REF-037]: blocks of ℓ consecutive Test days are resampled with replacement to 28 days, the pooled ratio is recomputed, B = 10,000 bootstrap replicates, percentile intervals [REF-044]. |
| **Block length** | **ℓ (block length) = 3 days, primary.** This follows the n^(1/3) and n^(1/4) orders for variance and one-sided distribution estimation (≈ 3.0 and 2.3 for n = 28, where n = number of Test days) [REF-064]. Those are asymptotic orders with unknown constants, so ℓ is pre-specified rather than estimated from Test data. **Sensitivity:** ℓ = 1 (independent days) and ℓ = 7 (weekly). |
| **Entity clustering** | Within-day entity clustering is absorbed by the day unit. Cross-day recurrence is partly absorbed by the blocks. Because H3 involves **no labels**, the label-propagation dependence that affects H1/H2 ([DR-001](DR-001-temporal-split-strategy.md); addressed by the two-way bootstrap of [DR-006](DR-006-bootstrap-inference-strategy.md)) does not apply to H3. |
| **Threshold ties** | τ₂ is the score of the transaction ranked ⌈0.02 × N_V2⌉ in descending V2 order (N_V2 = number of V2 transactions; ⌈ · ⌉ = round up), and transactions are flagged if score ≥ τ₂. The realised V2 rate r_V2 (share of V2 transactions with score ≥ τ₂) is reported. If tied scores make r_V2 differ from 0.02 by more than 1 % relative, this is reported and discussed, but the H3 target remains 0.02. |
| **Not propagated** | Sampling uncertainty in τ₂ from V2 (binomial standard error, SE, of the V2 rate ≈ 0.05 percentage points, pp) is not added, because H3 is conditional on the deployed threshold. Variation across calibration origins is addressed by the development-only rolling-origin supplement, not by H3. |

## Evidence

- **[D]** Volume-only workload deviation: −27 % to +39 % of K across V2 days.
- **[D]** Composition drift between Train, V1 (Validation 1, days 106–126) and V2 in identity coverage, M-flag missingness (58 % → 36 %) and several features with Population Stability Index (PSI) ≥ 0.25. These are plausible drivers of score-distribution shift, and therefore of rate drift.
- **[L]** Thresholds should be selected on held-out data, and threshold choice is part of what is evaluated [REF-038, REF-009]. Under concept drift, validation-time operating points can degrade [REF-005, REF-023].
- **[L]** Absence of a significant difference is not evidence of equivalence. Equivalence claims need a pre-specified bound and TOST [REF-030].

## Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Point-null H3 ("the threshold preserves the intended workload") | Near-certain rejection at this sample size, and a non-rejection proves nothing [REF-030]. It also mixes rate drift with volume drift. |
| Test daily-budget fidelity as H3 | Budget fidelity is largely driven by volume variation that exists regardless of the model [D]. A hypothesis on it would mostly test the data-generating process, not policy transfer. |
| Make the rate threshold adapt daily (re-estimate the quantile each day) | That is a different policy, close to R3 (daily top-K). R2 is specifically the *static* validation-selected threshold whose transfer is the object of study. |
| Leave Δ unfixed until after the Test run | Makes the equivalence test meaningless: bounds chosen after seeing the data can be fitted to the result. Δ is fixed before any Test access. |
| Decide H3 by the 90 % CI regardless of Holm's procedure | Bypasses the family-wise error control of the confirmatory family, including Holm's stopping rule ([DR-010](DR-010-h3-confirmatory-configuration.md)). |

## Rationale

The rate component answers the research question, *does a validation-selected operating point survive temporal shift?*, and is attributable to the model–data pair. The budget component answers the operational question, *what workload does the team actually face?* Separating them avoids blaming the model for volume swings, and the equivalence framing makes a "preserved" conclusion testable.

## Consequences

- Missed-fraud consequences of workload drift are **secondary descriptive outcomes** of the budget-fidelity analysis, bounded by [DR-009](DR-009-overflow-handling-for-threshold-policies.md): only frauds below the threshold are false negatives; frauds in overflow are reported only as "flagged frauds in excess demand", conditional on score-priority ordering.
- Only one V2 → Test transfer event exists ([DR-001](DR-001-temporal-split-strategy.md)). The **development-only rolling-origin supplement** provides exploratory evidence on how variable transfer is across origins.
- Label-free: R2's threshold needs V2 scores but no V2 labels, so R2 is not affected by the label-latency sensitivity.

## Threats and limitations

- **Δ is a researcher judgement.** No fraud- or alerting-specific literature standard exists. The 20 % magnitude rests on development-period volume variation and operational reasoning, and is reported with sensitivity bounds. It should be reviewed with the supervisor before the protocol freeze.
- **The feasibility table relies on planning assumptions** (day-level SD, autocorrelation), not measured model behaviour.
- **With 28 days and ℓ = 3, the bootstrap has about 26 distinct blocks.** Coverage of the 90 % interval is approximate.
- **The volume anchor uses V2 only (28 days).** A different reference period would give somewhat different percentiles.
- n = 28 Test days limits daily-level inference, and days are autocorrelated (daily-volume lag-1 autocorrelation 0.75 over the full period [D]).
- The rate/volume decomposition follows from n_d = r_d × N_d and K ≈ 0.02 × N̄_V2, and is exact up to the rounding of K. Attributing changes in r_d to "model" versus "data" drift is interpretive.

## References

REF-004, REF-005, REF-006, REF-009, REF-023, REF-029, REF-030, REF-037, REF-038, REF-044, REF-061, REF-062, REF-063, REF-064, REF-065 — see [`../references.md`](../references.md).

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [D], [L], [I] | Evidence tags: dataset observation, literature, inference |
| α | Family-wise significance level (0.05) |
| #{ · } | Number of bootstrap replicates satisfying a condition |
| AR(1) | First-order autoregressive model |
| B | Number of bootstrap replicates (10,000) |
| CI | Confidence interval |
| d | A relative day (Test days 155–182 unless stated) |
| Δ | H3 equivalence tolerance: 0.004 (20 % of the 2 % target) |
| H1, H2, H3 | Confirmatory hypotheses: model family; class weighting; alert-rate transfer fidelity |
| R0, R1 | Decision policies: default 0.5 threshold; F1-optimal threshold |
| H3₀, H3₁ | Null and alternative forms of H3 |
| H0a, H0b | The two one-sided null hypotheses of the H3 TOST |
| K | Fixed daily review budget (56 at 2 % capacity) |
| ℓ | Block length of the moving-block bootstrap, in days |
| log | Natural logarithm |
| n | Number of Test days (28), in the block-length orders n^(1/3), n^(1/4) |
| n_d | Number of transactions on day d with score ≥ τ₂ |
| O_d, U_d, A_d,R2 | DR-009 notation: overflow max(n_d − K, 0); unused capacity max(K − n_d, 0); alerts generated by R2 at 2 % (= n_d) |
| N_d | Total number of transactions on day d |
| N_V2 | Number of V2 transactions |
| N̄_V2 | Mean daily transaction volume in V2 |
| p_a, p_b, p_H3 | p-values of H0a and H0b; H3 TOST p-value max(p_a, p_b) |
| r* | r in one bootstrap replicate |
| pp | Percentage points |
| PSI | Population Stability Index |
| r | Realised Test alert rate at τ₂, pooled over Test days |
| r_d | Realised alert rate on day d |
| r_V2 | Realised V2 alert rate at τ₂ |
| R2 | Decision policy: capacity-rate threshold chosen on V2 |
| R3 | Decision policy: daily top-K review |
| RQ2 | Research question 2 (policy-transfer fidelity) |
| SD | Standard deviation |
| SE | Standard error |
| SESOI | Smallest Effect Size Of Interest |
| τ, τ₂ | A score threshold; the V2-selected 2 % rate threshold |
| TOST | Two One-Sided Tests |
| V1, V2 | Validation 1 (days 106–126); Validation 2 (days 127–154) |
| x, y | Candidate relative tolerance; candidate tolerance in reviews per day |
| Σ_d | Sum over days |
| ⌈ · ⌉ | Round up to a whole number |
