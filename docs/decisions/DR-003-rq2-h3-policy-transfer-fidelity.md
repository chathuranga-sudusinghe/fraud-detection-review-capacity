# DR-003 — RQ2 / H3: policy-transfer fidelity

| Field | Value |
|---|---|
| Status | Accepted (pre-freeze). **The tolerance Δ is not yet fixed.** |
| Date | 2026-09-23 |
| Evidence source | `notebooks/01_dataset_feasibility_audit.ipynb`, §§3, 6, 14, 17 |
| Related | [DR-001](DR-001-temporal-split-strategy.md), [DR-002](DR-002-review-capacity-definition.md), [protocol](../protocol.md) |

## Context

The previous RQ2 asked how accurately validation-selected thresholds "preserve the intended review workload". Its H3 was a point null ("a validation-selected 2 % review threshold preserves the intended review workload").

The feasibility audit found that this wording **mixes two different mechanisms**:

1. **Alert-rate fidelity.** Does the score threshold τ, chosen so that 2 % of V2 transactions exceed it, still flag about 2 % of transactions in a later period? This depends on **score-distribution and composition drift**, i.e. on the model and the data.
2. **Fixed daily-budget fidelity.** Does the *number* of flagged transactions per day match the fixed budget K ([DR-002](DR-002-review-capacity-definition.md))? This depends on alert-rate drift **and** on **daily volume variation**, which exists whatever the model does.

**Evidence [D].**
- Daily transaction volume is non-stationary: coefficient of variation 0.26 over 182 days.
- Even a *perfectly* transferring 2 % rate threshold would produce V2 daily review counts between **−27 % and +39 %** of the fixed K = 56 (SD 16 %), purely from volume variation.
- Composition also shifts: identity coverage falls from 27.1 % to 17.5 % between Train and V2, and fraud prevalence among identity-present rows changes.

The rate component is therefore a genuine object of study.

The old point-null H3 is **uninformative**. With about 78,500 Test transactions, almost any non-zero deviation from exactly 2 % would be "significant". A non-rejection would not show that the rate is preserved [REF-030].

## Decision

### Revised RQ2

> **RQ2: When a score threshold selected on validation data is transferred to a later temporal period, how well does it preserve the intended alert rate, and how does the resulting daily review workload compare with the fixed review budget?**

### Revised H3 — Alert-rate transfer fidelity (equivalence formulation)

**Definitions.**
- τ₂ is the score threshold chosen on V2 so that 2 % of V2 transactions have score ≥ τ₂ (policy R2).
- The **realised alert rate** r̂ is the proportion of Test transactions with score ≥ τ₂, pooled over the 28 Test days.
- The **equivalence band** is [0.02 − Δ, 0.02 + Δ].

> **H3₀:** The realised alert rate under the validation-selected 2 % threshold lies **outside** the pre-specified equivalence band around the 2 % target (|r − 0.02| ≥ Δ).
>
> **H3₁:** The realised alert rate under the validation-selected 2 % threshold lies **within** the pre-specified equivalence band around the 2 % target (|r − 0.02| < Δ).

**Test.**
- Two one-sided tests (TOST) at α = 0.05 [REF-030]. This is equivalent to concluding H3₁ when the 90 % confidence interval for r lies entirely inside the band.
- The confidence interval comes from the **cluster- or day-block-aware bootstrap** declared in the protocol [REF-036, REF-037, REF-044].

**Δ is deliberately not fixed here.** Before the protocol freeze it must be:
- (a) expressed as an absolute percentage-point or a relative tolerance;
- (b) justified on operational grounds, e.g. the budget over-run a review team could absorb, **not** from any Test information;
- (c) recorded in a superseding or amending decision record.

**Still to specify at freeze:**
- which model configuration(s) H3 is tested on (e.g. each of the four model × weighting configurations, or a single pre-declared primary one);
- how H3 enters the multiplicity plan with H1 and H2 [REF-029].

### Daily-budget fidelity — a separate operational outcome (not a confirmatory hypothesis)

For each Test relative day d, with N_d the day's transaction volume, n_d the number of transactions with score ≥ τ₂, and K = 56:

| Statistic | Definition |
|---|---|
| Realised daily review count | n_d |
| Deviation from fixed K | n_d − K |
| Relative deviation | (n_d − K) / K |
| Absolute deviation | \|n_d − K\|; reported as a mean and maximum over days |
| Days above K (overflow) | % of days with n_d > K; total overflow Σ max(0, n_d − K) |
| Days below K (underuse) | % of days with n_d < K; total unused capacity Σ max(0, K − n_d) |
| Rate / volume decomposition | log(n_d / K) ≈ log(r_d / 0.02) + log(N_d / N̄_V2), where r_d = n_d / N_d and N̄_V2 is mean V2 daily volume. This separates the *rate* component (model and data drift) from the *volume* component |

These are reported descriptively with bootstrap intervals where appropriate. Secondary descriptive consequences:
- frauds missed because of overflow, under the declared overflow rule ([DR-002](DR-002-review-capacity-definition.md));
- review capacity left unused on underuse days.

## Evidence

- **[D]** Volume-only workload deviation: −27 % to +39 % of K across V2 days.
- **[D]** Composition drift between Train, V1 and V2 in identity coverage, M-flag missingness (58 % → 36 %) and several features with PSI ≥ 0.25. These are plausible drivers of score-distribution shift, and therefore of rate drift.
- **[L]** Thresholds should be selected on held-out data, and threshold choice is part of what is evaluated [REF-038, REF-009]. Under concept drift, validation-time operating points can degrade [REF-005, REF-023].
- **[L]** Absence of a significant difference is not evidence of equivalence. Equivalence claims need a pre-specified bound and TOST [REF-030].

## Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Keep the point-null H3 | Near-certain rejection at this sample size, and a non-rejection proves nothing [REF-030]. It also mixes rate drift with volume drift. |
| Test daily-budget fidelity as H3 | Budget fidelity is largely driven by volume variation that exists regardless of the model [D]. A hypothesis on it would mostly test the data-generating process, not policy transfer. |
| Make the rate threshold adapt daily (re-estimate the quantile each day) | That is a different policy, close to R3 (daily top-K). R2 is specifically the *static* validation-selected threshold whose transfer is the object of study. |
| Fix Δ now | Would be arbitrary. Δ needs an operational justification and should be agreed with the supervisor before the freeze. |

## Rationale

The rate component answers the research question, *does a validation-selected operating point survive temporal shift?*, and is attributable to the model–data pair. The budget component answers the operational question, *what workload does the team actually face?* Separating them avoids blaming the model for volume swings, and the equivalence framing makes a "preserved" conclusion testable.

## Consequences

- RQ2 no longer includes "what effect does workload drift have on missed fraud" as its main question. Missed-fraud consequences become **secondary descriptive outcomes** of the budget-fidelity analysis.
- Only one V2 → Test transfer event exists ([DR-001](DR-001-temporal-split-strategy.md)). The **development-only rolling-origin supplement** provides exploratory evidence on how variable transfer is across origins.
- Label-free: R2's threshold needs V2 scores but no V2 labels, so R2 is not affected by the label-latency sensitivity.

## Threats and limitations

- Δ is a researcher-chosen tolerance. Its justification is a judgement and must be reported transparently.
- n = 28 Test days limits daily-level inference, and days are autocorrelated (daily-volume lag-1 autocorrelation 0.75 over the full period [D]).
- The rate/volume decomposition follows from n_d = r_d × N_d and K ≈ 0.02 × N̄_V2, and is exact up to the rounding of K. Attributing changes in r_d to "model" versus "data" drift is interpretive.

## References

REF-005, REF-009, REF-023, REF-029, REF-030, REF-036, REF-037, REF-038, REF-044 — see [`../references.md`](../references.md).
