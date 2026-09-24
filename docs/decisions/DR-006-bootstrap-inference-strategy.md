# DR-006 — Bootstrap inference strategy for H1 and H2

| Field | Value |
|---|---|
| Status | Accepted (pre-freeze) by Issue #14, pending supervisor review. Resolves protocol open item 3 (primary bootstrap unit for H1/H2). |
| Date | 2026-09-24 |
| Evidence source | Development-period (V2, relative days 127–154) dependence diagnostics, computed for this decision with no Test labels and no model scores; `notebooks/01_dataset_feasibility_audit.ipynb` §15; bootstrap and clustered-inference literature |
| Related | [DR-001](DR-001-temporal-split-strategy.md), [DR-002](DR-002-review-capacity-definition.md), [DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md), [DR-005](DR-005-sesoi-and-h1-h2-inference.md), [protocol](../protocol.md) |

## Context

H1 and H2 compare paired differences in pooled Test **Recall@2 %** under the daily top-K policy R3 (K = 56 per relative day) across four model × weighting configurations ([DR-005](DR-005-sesoi-and-h1-h2-inference.md)). DR-005 required a **joint paired bootstrap**. It ruled out independence-based resampling, but left the **resampling unit** open. The two candidates were heuristic pseudo-entity clusters and day blocks.

The trade-off:
- entity resampling targets within-entity dependence, but relies on a **heuristic** key;
- day resampling targets temporal dependence and keeps each day's top-K intact, but the Test period has only **28 days**.

This record resolves that choice using development-period evidence only.

## Evidence

### Observed dependence in V2 [D]

Two heuristic pseudo-entity keys from the feasibility audit were used:
- **fine**: card1 + addr1 + purchaser email domain + (relative day − D1);
- **coarse**: card attributes + addr1.

Neither is a verified customer, card or account identifier.

| Quantity (V2: 28 days, 77,822 transactions, 2,637 frauds) | Value |
|---|---|
| Fraud clusters, fine key | 1,353. 61.7 % are singletons; **68.3 % of frauds** are in clusters of ≥ 2; largest 29 |
| Upper-bound design effect for fraud-level recall, fine key (Σm² / Σm) | **4.25** |
| V2 frauds whose fine-key cluster spans more than one day | **44.7 %** |
| Same-cluster fraud pairs, by day gap (fine key) | same day 29.1 %; 1–2 days 21.4 %; 3–6 days 18.9 %; **≥ 7 days 30.7 %** |
| Fine-key cluster span (last − first fraud day), multi-fraud clusters | median 1 day, 90th percentile 14 days |
| Daily fraud-count dispersion (variance ÷ binomial variance) | **5.3** in V2 (8.6 over days 1–154) |
| Lag-1 autocorrelation of daily prevalence | **0.01** in V2 (0.39 over days 1–154, which includes between-period level shifts) |
| Every V2 day has more frauds than K = 56 | Yes, all 28 days (capacity binds daily; [DR-002](DR-002-review-capacity-definition.md)) |
| Coarse key | 735 clusters; upper-bound DE 15.2; 76.6 % of frauds in clusters spanning > 1 day |

### Proxy variance diagnostic [D, with interpretation I]

**Why a proxy is needed.** Model capture indicators do not exist yet: no models have been trained and no Test scores exist. As a proxy, four **entity-linked binary attributes of V2 frauds** were used: identity present; ProductCD = C; card6 = credit; amount above the V2 fraud median.

**Method.** For each attribute, the variance of the fraud-level proportion was estimated under several resampling schemes (4,000 replicates each, fixed seed). Each variance is expressed relative to a naive fraud-level bootstrap.

| Attribute (fine key) | Day (iid days) | Day blocks, ℓ = 3 | Entity (fine key) | Two-way pigeonhole |
|---|---|---|---|---|
| Identity present | 4.05 | 4.66 | 4.02 | 9.67 |
| ProductCD = C | 3.46 | 2.23 | 3.52 | 8.27 |
| card6 = credit | 2.47 | 1.96 | 4.18 | 8.25 |
| Amount > median | 2.45 | 3.32 | 3.15 | 7.09 |

**Interpretation [I].**
1. **Transaction- or fraud-level independence is untenable.** Every clustered scheme gives 2–10 times the naive variance.
2. **Day-level and entity-level dependence are of similar size** (≈ 2.5–4 times naive each), and they are **crossed, not nested**: 45 % of frauds belong to entities that span several days.
3. The pigeonhole variance, minus Owen's documented over-count of about 2 times naive [REF-068], is ≈ 5–7.7 times naive. So the day and entity components appear roughly additive, and **either one-way scheme would understate the variance by roughly 30–50 %** for entity-linked quantities.
4. Short day-blocks do not absorb entity dependence: about half of same-entity fraud pairs are 3 or more days apart, and 31 % are 7 or more days apart. Blocks long enough to absorb that (≈ 14 days) would leave only two blocks in 28 days.
5. The coarse key's entity ratios (7.6–20.9) are inflated mechanically, because the key is built from the same card attributes that the proxies depend on. It is **not** a usable entity definition.

**Caveat.** These attributes are proxies. The real statistics are *paired differences* in capture between configurations. Shared entity effects may partly cancel in those differences, so actual design effects could be smaller. The diagnostic shows *which dimensions* matter, not the exact size of the Test variance.

### Literature findings [L]

- **Clustered and multiway dependence.**
  - Resampling must respect the dependence structure. Clusters are the resampling unit for clustered data [REF-036].
  - When dependence runs along two **non-nested** dimensions, inference must account for both [REF-070].
  - For crossed, unbalanced random-effects data, **resampling rows and columns independently** (the pigeonhole bootstrap) is mean-consistent. It is **mildly conservative**: the variance estimate is close to ((ν_A + 2)σ²_A + (ν_B + 2)σ²_B + 3σ²_E)/N, against a true (ν_Aσ²_A + ν_Bσ²_B + σ²_E)/N. It therefore overestimates by about twice the naive variance, and Owen suggests subtracting twice the naive variance as a correction. A naive case bootstrap "can be seriously misleading" [REF-068].
- **Few clusters.** With few (five to thirty) clusters, standard cluster-robust tests can over-reject [REF-069]. With 28 Test days, this applies to any day-only method.
- **Temporal dependence.** Block bootstraps handle short-range serial dependence [REF-037]; block-length orders are given by [REF-064]. They cannot absorb dependence at lags comparable to the series length.
- **Paired binary comparisons under clustering.** "In the presence of intracluster correlation, the size of McNemar's test can greatly exceed the nominal level" [REF-071]. Paired classifier comparisons that assume independent observations [REF-028, REF-035] are therefore not valid primary tests here.
- **No fraud-specific standard.** No verified source prescribes a bootstrap design for capacity-constrained recall in fraud detection. The choice below is an application of the general methods above.

## Candidate methods compared

| | A. Transaction-level | B. Pseudo-entity cluster (one-way) | C. Day-level, iid days | D. Moving-block days (ℓ = 3) | **E. Two-way pigeonhole (day × fine entity)** |
|---|---|---|---|---|---|
| Dependence handled | None | Within-entity, including across days | Within-day, including same-day entity clustering and daily capacity competition | Within-day plus short-range (≤ ≈ 3 days) serial dependence | **Both** day-level and entity-level, including cross-day entity recurrence |
| Dependence ignored | All | Day-level shocks and capacity competition shared across entities | Cross-day entity recurrence (45 % of frauds) | Entity recurrence beyond ≈ 3 days (≈ half of same-entity pairs) | Residual dependence not captured by the heuristic key; serial dependence between days (small in V2) |
| Key assumptions | Independence (contradicted) | The key captures entities; days exchangeable | Days exchangeable | Short-range stationarity | Crossed random-effects structure; iid days; key approximately captures entities |
| Recall@2 % compatibility | Yes | Yes, via fixed capture indicators | Yes, exactly (days intact) | Yes, exactly | Yes, via fixed capture indicators |
| Paired comparison | Yes | Yes | Yes | Yes | Yes |
| Preserves daily top-K | No | Conditionally (rankings fixed) | **Exactly** | **Exactly** | Conditionally (rankings fixed; day dimension carries day-level variation) |
| Feasible with 28 days | — | Not day-dependent (≈ 1,350 entity clusters) | Few-cluster risk [REF-069] | Worse: ≈ 26 distinct blocks, ≈ 10 per replicate | Day dimension carries the few-cluster risk; entity dimension large |
| Dependence on heuristic key | None | **Total** | None | None | Partial: if the key fails, E degrades towards a day + fraud-level bootstrap, which is still conservative relative to C |
| Computation | Trivial | Trivial | Trivial | Trivial | Trivial (product weights on ≈ 2,700 frauds) |
| Interpretability | High but wrong | Moderate | High | Moderate | Moderate |
| Verdict | **Rejected** | Sensitivity input only (via E) | Rejected as primary: misses cross-day entity dependence | **Sensitivity (key-free)** | **Primary** |

## Decision

### Primary method: two-way (pigeonhole) paired bootstrap over relative days × fine pseudo-entities

**Why [I, grounded in D and L].**
- Both day-level and entity-level dependence are observed, of similar size, and crossed.
- Each one-way method demonstrably misses one of them.
- The pigeonhole bootstrap addresses both [REF-068, REF-070].
- Its known bias is **conservative**. That is the safe direction for confirmatory claims, and especially for **H2 equivalence**, where an underestimated variance would produce false "equivalence" conclusions.
- Its dependence on the heuristic key is limited: the day dimension does not use the key, and a poor key still adds variance rather than removing it.

**Why not day-only as primary.** It does not use the heuristic key and it preserves top-K exactly, but the evidence indicates it would understate variance by roughly 30–50 % for entity-linked quantities. With 28 days it also carries the few-cluster risk alone [REF-069].

### Sensitivity analyses (pre-declared; not confirmatory)

- **S1 — Owen-corrected two-way variance.** SE² = V_pigeonhole − 2·V_naive, where V_naive is from a fraud-level iid bootstrap. Wald intervals δ̂ ± z·SE [REF-068]. This removes the known over-count and shows how far conclusions depend on the primary method's conservatism. If the corrected variance is ≤ 0, S1 is reported as not estimable.
- **S2 — Day-level moving-block bootstrap, ℓ = 3 days.** Circular wrap is not used, to match H3 ([DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md)). This is a key-free check that preserves each day's top-K exactly. The same B and interval type as the primary are used.

**Robustness rule.**
- Confirmatory decisions (H1, H2) use the **primary method only**; there is no switching after results are seen.
- If S1 or S2 would lead to a different H1 or H2 conclusion, this is reported explicitly as sensitivity of the conclusion to dependence assumptions.

## Exact paired resampling procedure (primary)

Inputs from the one confirmatory Test evaluation, computed **once**:
- For each configuration m ∈ {LGB-u, LGB-b, Cat-u, Cat-b} and each Test relative day d, rank that day's transactions by m's score. Set c_{m,i} = 1 if fraud i is among day(i)'s top K = 56 under m, else 0. Ties are handled as pre-declared in the implementation.
- For each Test fraud i: its day d(i), and its fine pseudo-entity e(i), built from Test-period features only (card1, addr1, P_emaildomain, relative day − D1).

For b = 1, …, B:
1. Draw day multiplicities a_d ~ Multinomial(28; 1/28, …, 1/28) over the 28 Test days.
2. Independently, draw entity multiplicities b_e ~ Multinomial(G; 1/G, …, 1/G) over the G fraud entities in Test.
3. Weight each fraud w_i = a_{d(i)} · b_{e(i)}.
4. For **every** configuration m, using the **same** weights: R*_m = Σ_i w_i c_{m,i} / Σ_i w_i.
5. Compute δ*_M = ½[(R*_Cat-u − R*_LGB-u) + (R*_Cat-b − R*_LGB-b)] and δ*_W,f = R*_f-b − R*_f-u for f ∈ {LGB, Cat}, plus the descriptive contrasts and the interaction.

**Why this is valid for paired comparison.**
- The weights depend only on each fraud's (day, entity) membership, never on model outputs.
- Applying the same weights to all four configurations keeps the within-fraud correlation between configurations. The bootstrap distribution of each difference therefore reflects paired, not independent-sample, variability [REF-068, REF-036].

**Daily top-K: why the ranking is not recomputed in each replicate.**
- In the **primary**, recomputing top-K on a "resampled dataset" is not well defined. Duplicating an entity inside a day would create tied duplicate transactions and change that day's volume while K stays 56, so the capacity rate would no longer be 2 %.
- The primary therefore conditions on the realised within-day ranking. Uncertainty comes from *which days and which fraud entities occur*. Day-level variation in capacity competition is carried by the day weights, which move all of a day's frauds together.
- As a consequence, a replicate's weighted captured count for a day can exceed K. The replicate is a weighted estimator used for variance approximation, not a literal simulated Test period.
- In **S2**, whole days are resampled. Each day's top-K is then *exactly* preserved, and recomputation would give identical results, so none is needed.

## Confidence intervals and replicates

| Item | Specification | Justification |
|---|---|---|
| Interval type (primary) | **Percentile** intervals from the pigeonhole distribution | Simple, respects the non-normal shape of a bounded ratio, and matches H3's percentile intervals. Mildly conservative by construction [REF-068]. |
| BCa | **Not used** | Its acceleration constant is normally estimated by a jackknife, and no standard jackknife exists for crossed two-way data. Using BCa here would add an unvalidated component. |
| Basic / studentized | Not used as primary | A studentized interval needs a per-replicate variance estimate, which would need a second-level two-way bootstrap. The added complexity is not justified for an MSc confirmatory analysis. S1 provides a Wald-type check. |
| Levels | 95 % for difference estimates; 90 % for H1's practical interpretation; (1 − 2α_H2) for the H2 TOST (e.g. 96.7 % at a Holm step with α = 0.0167) | As DR-005 |
| Replicates | **B = 10,000**, fixed seed recorded | The most extreme interval used, (1 − 2 × 0.0167), puts 1.67 % in each tail: ≈ 167 replicates per tail at B = 10,000, against ≈ 17 at B = 1,000, which is too few for stable tail quantiles. Computation is negligible. Monte Carlo standard errors of reported p-values are also reported. |
| Small-cluster corrections | **None applied**; limitation stated | Wild cluster bootstrap-t corrections [REF-069] are designed for regression coefficients with few clusters in one dimension. Here the entity dimension is large (≈ 1,350 clusters projected), the day dimension (28) carries the few-cluster risk, and the primary's conservatism partly offsets it. |
| Day blocks | Primary: iid days inside the two-way scheme. Sensitivity S2: fixed ℓ = 3 | Within V2, lag-1 autocorrelation of daily prevalence is 0.01 [D]. Once entity recurrence is modelled explicitly, the remaining serial dependence is small. S2 keeps the H3 block length (DR-003; [REF-064]) for comparability. |

## Relationship to H1, H2, TOST and McNemar

- **H1 (difference).** p_H1 = 2·min(P*(δ*_M ≤ 0), P*(δ*_M ≥ 0)), the percentile-interval inversion. The 90 % percentile interval gives DR-005's five-way practical interpretation against ±0.02.
- **H2 (intersection-union TOST).** For each family f: p_a,f = P*(δ*_W,f ≤ −0.02) and p_b,f = P*(δ*_W,f ≥ +0.02). Then p_H2 = max over f and both sides. This is equivalent to requiring the (1 − 2α_H2) percentile intervals for both δ_W,LGB and δ_W,Cat to lie inside (−0.02, +0.02) [REF-061, REF-062]. **TOST uses the bootstrap-derived intervals**, because no closed-form interval accounts for the crossed dependence.
- **Multiplicity.** Holm over {H1, H2, H3} is unchanged ([DR-005](DR-005-sesoi-and-h1-h2-inference.md)).
- **H3** is unchanged. It is label-free and uses the day-level moving-block bootstrap of DR-003.
- **McNemar remains secondary and descriptive.** It assumes independent paired units, and its size "can greatly exceed the nominal level" under intracluster correlation [REF-071]. It is reported for the four cell-level contrasts only, labelled as independence-based, and is excluded from the Holm family.

## Consequences

- Protocol open item 3 is resolved, pending supervisor review.
- The Test evaluation must save, for every configuration: per-fraud capture indicators; each fraud's relative day and fine pseudo-entity; per-day TP and fraud counts (for S2).
- **Power may be lower than DR-005's central scenario.** The proxy diagnostic suggests two-way variance ratios of ≈ 5–10 times naive for entity-linked quantities, whereas DR-005 assumed design effects of 1–4. Paired differences may be less affected, because shared entity effects cancel, but inconclusive H1/H2 outcomes are **more** likely than DR-005 indicated. The SESOI is not changed; this is recorded so that inconclusive results are interpreted correctly.
- The same fine pseudo-entity definition is used for the bootstrap and for the "recurring vs new entity" false-negative slice. Both are heuristic.

## Threats and limitations

- **The fine pseudo-entity key is heuristic.** If it splits true entities, residual cross-day dependence is not captured. If it merges unrelated ones, variance is overstated.
- **The primary method is conservative by design.** It reduces power, particularly for H2 equivalence. S1 quantifies how much.
- **Conditioning on the realised rankings** means variability from the legitimate-transaction background within a day is represented only through the day weights, not by re-ranking. S2 re-ranks nothing either, but keeps whole days intact.
- **Only 28 days in the day dimension.** Coverage may fall slightly below nominal [REF-069]. This is partly offset by the conservative primary.
- **The proxy diagnostic** uses fraud attributes, not model capture indicators. It informs the choice of dimensions, not the exact size of the Test variance.
- **Parts of the literature were read only in part.** REF-069 and REF-070 were read at abstract level. REF-068's conservativeness and correction results were read in full text (arXiv version).

## References

REF-028, REF-035, REF-036, REF-037\*, REF-044, REF-061, REF-062, REF-064, REF-068, REF-069, REF-070, REF-071 — see [`../references.md`](../references.md).

\* Partially verified (page range pending).
