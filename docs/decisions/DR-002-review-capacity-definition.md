# DR-002 — Review-capacity definition

| Field | Value |
|---|---|
| Status | Accepted (pre-freeze) |
| Date | 2026-09-23 |
| Evidence source | `notebooks/01_dataset_feasibility_audit.ipynb`, §§6, 7, 11 |
| Related | [DR-001](DR-001-temporal-split-strategy.md), [DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md), [protocol](../protocol.md) |

## Context

The research studies fraud detection when only a limited number of transactions can be manually reviewed. The dataset contains **no information about real review capacity**. Capacity must therefore be introduced as an explicit, simulated experimental factor, and defined so that:
- it uses no Test-period information;
- it produces a workload that is fixed in advance, as a staffed review team would face.

## Decision

1. **Capacity scenarios:** c ∈ {**1 %, 2 %, 5 %, 10 %**} of transactions. **Primary: 2 %.**
2. **These are simulated research scenarios.** They are not claimed to be the capacity of any real bank or payment provider, and must be described that way in every document.
3. **Fixed daily review budget:**

   > **K_c = round(c × mean daily transaction volume in V2)**

   V2 = relative days 127–154 ([DR-001](DR-001-temporal-split-strategy.md)). K_c is fixed before any Test data are touched and applied unchanged to every Test relative day.

   | c | K_c (reviews per relative day) | Reviews over the 28-day Test window | capacity ÷ V2 prevalence | Regime |
   |---|---|---|---|---|
   | 1 % | 28 | 784 | 0.30 | binding |
   | **2 %** | **56** | **1,568** | **0.59** | **binding** |
   | 5 % | 139 | 3,892 | 1.48 | slack |
   | 10 % | 278 | 7,784 | 2.95 | slack |

   Inputs: mean V2 daily volume 2,779.4; V2 prevalence 3.39 % [D].
4. **Report capacity ÷ prevalence alongside every capacity result**, so results can be compared across base rates.
5. **Interpret the primary metric against its theoretical ceiling.** Recall@K is reported together with:
   - the **oracle recall ceiling**, Σ_d min(K, F_d) / Σ_d F_d, where F_d is the number of frauds on day d;
   - **normalised recall**, TP / min(K, F_d), computed per day and pooled.
6. **Recall@K and Precision@K are not independent evidence.** On a fixed evaluation period both K and the fraud total F are fixed, so

   > Precision@K = Recall@K × F / (number of reviewed transactions)

   With a fully used budget, the denominator is fixed. The two metrics then differ only by a constant, and any model ranking on one is the same ranking on the other. Only **one** (Recall@K) is used for confirmatory testing; Precision@K is reported descriptively.

## Evidence

- **[D]** Mean V2 day: 2,779 transactions and 94 frauds (range 61–136).
- **[D]** At 2 %, K = 56 is **below the daily fraud count on every V2 day** (100 % of days). The pooled oracle recall ceiling at 2 % is **≈ 0.595**, while oracle precision can reach 1.0.
- **[D]** At 5 % and 10 %, K exceeds the daily fraud count on every V2 day. Recall can then reach 1.0, but the oracle precision ceiling falls to ≈ 0.68 and ≈ 0.34.
- **[D]** Random review has precision equal to prevalence (≈ 0.034), so maximum lift over random is ≈ 29.5 in the binding regime.
- **[L]** Evaluating fraud detection given a maximum number of investigations, or a fixed number of alerts, is established practice [REF-006, REF-004, REF-005, REF-046]. Top-quantile accuracy has a formal basis [REF-033], and top-k evaluation under resource limits is used outside fraud as well [REF-032]. **This study therefore claims no novelty for capacity-constrained evaluation itself.**

## Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Capacity as a fraction of *each day's own* volume (a floating budget) | Uses same-day volume, which is not known in advance. It also removes the workload-overflow phenomenon that RQ2 studies. |
| K from mean **development** (days 1–154) volume (2 % → 66/day) | Inflated by the early high-volume period, which is unrepresentative of later days [D]. V2 is closest in time and duration to Test. |
| K from **Test** volume | Uses future information (leakage). |
| Choosing capacities as multiples of prevalence (e.g. 0.5×, 1×, 2×, 3×) | Plausible, but it would tie the design to a prevalence estimate. The {1, 2, 5, 10} % grid already spans both the binding and slack regimes, and the ratio is reported anyway. |
| Primary capacity 5 % | Slack regime: the budget exceeds the number of frauds. Binding capacity is the operationally motivating case [REF-004, REF-005]. |
| Adding 3.5 % (≈ prevalence) | Optional. Not required to answer the RQs; may be reported descriptively. |

## Rationale

- 2 % sits in the **binding regime**: capacity is below the daily fraud count on every day. This is the situation the research question is about, where not all fraud can be reviewed.
- A fixed daily K mirrors a pre-staffed review team. It makes both **workload overflow and underuse** measurable ([DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md)).
- Deriving K from V2 keeps all policy parameters inside the development data.

## Consequences

- **Primary metric:** Recall@2 %, evaluated under the daily top-K policy R3 with K = 56. It is interpreted against the ≈ 0.6 ceiling and reported with normalised recall.
- **RQ3 must be evaluated across the capacity grid.** In the binding regime, the marginal (K-th) reviewed transaction is likely to have a high fraud probability. A cost-aware policy (R4) will then rarely leave budget unused, and R4 will often coincide with R3. The informative contrasts are expected at 5 % and 10 % and at low miss-to-review cost ratios [I]. The **main RQ3 output is the break-even miss-to-review cost ratio for each capacity** [REF-008, REF-039, REF-040].
- Policies that are not capped (R0, R1, R2) can exceed K on a given day. The protocol must declare an **overflow rule**: either report the overflow, or truncate to the top-K of the flagged transactions.
- Review effort is counted **per transaction**.

## Threats and limitations

- **Construct validity.** Real review queues work at account or case level. Reviewing one transaction of a compromised account usually triggers action on the whole account, so transaction-level K overstates the effort needed to stop entity-level fraud. IEEE-CIS has no true account or card identifier, so card-level precision [REF-046] cannot be computed.
- **External validity.** The capacity percentages are not calibrated to any institution.
- **Ceiling compression.** With a ceiling of ≈ 0.6, absolute differences in Recall@2 % between models are compressed. Minimum detectable effects are discussed in the protocol.

## References

REF-004, REF-005, REF-006, REF-008, REF-032, REF-033, REF-039, REF-040, REF-046 — see [`../references.md`](../references.md).
